// =============================================================================
// tuner/autotuner.cpp — AutoTuner implementation
//
// WHAT THIS FILE DOES:
//   Implements runtime benchmarking of tile sizes, JSON cache I/O (without
//   external dependencies — hand-written minimal JSON serializer), and
//   hardware detection via sysctl on macOS.
//
// WHY NO EXTERNAL JSON LIBRARY?
//   tinyinfer has zero runtime dependencies beyond libstdc++ and POSIX.
//   Adding nlohmann/json for a 10-field cache file would pull in 18,000 lines
//   of header. Our cache format is simple enough to parse with string operations.
//
// TIMING STRATEGY:
//   std::chrono::high_resolution_clock gives nanosecond resolution on macOS.
//   We use the POSIX clock_gettime(CLOCK_MONOTONIC) path internally (same thing).
//   To minimize measurement noise:
//     - Run each tile_size TUNE_ITERATIONS times
//     - Sort and take the median (index TUNE_ITERATIONS/2)
//     - The median filters out outliers from OS preemption and thermal throttling
// =============================================================================

#include "tuner/autotuner.h"
#include "kernels/matmul.h"

#include <chrono>
#include <algorithm>
#include <cstdlib>    // posix_memalign, free, getenv
#include <cstring>    // memset
#include <cstdio>     // fopen, fclose, fprintf
#include <fstream>    // ifstream, ofstream
#include <sstream>
#include <iostream>
#include <cassert>
#include <sys/stat.h> // mkdir, stat
#include <unistd.h>   // access, F_OK

#ifdef __APPLE__
#include <sys/sysctl.h>   // sysctlbyname for CPU brand string
#endif

namespace tinyinfer {

// ---------------------------------------------------------------------------
// Helper: read macOS sysctl string (e.g. "machdep.cpu.brand_string")
// ---------------------------------------------------------------------------
static std::string sysctl_string(const char* name) {
#ifdef __APPLE__
    char buf[256] = {};
    size_t len = sizeof(buf);
    if (::sysctlbyname(name, buf, &len, nullptr, 0) == 0) {
        return std::string(buf);
    }
#endif
    return "unknown";
}

// ---------------------------------------------------------------------------
// AutoTuner singleton
// ---------------------------------------------------------------------------
AutoTuner& AutoTuner::instance() {
    static AutoTuner tuner;
    return tuner;
}

AutoTuner::AutoTuner() {
    hardware_id_ = detect_hardware();
}

std::string AutoTuner::detect_hardware() {
    // On Apple Silicon, "machdep.cpu.brand_string" returns "Apple M4" etc.
    std::string brand = sysctl_string("machdep.cpu.brand_string");
    if (brand.empty() || brand == "unknown") {
        // Fallback: try the ARM-specific sysctl.
        brand = sysctl_string("hw.machine");
    }
    return brand.empty() ? "unknown" : brand;
}

// ---------------------------------------------------------------------------
// Cache file path
// ---------------------------------------------------------------------------
std::string AutoTuner::cache_file_path() {
    const char* home = ::getenv("HOME");
    if (!home) home = "/tmp";
    return std::string(home) + "/.tinyinfer/tuning_cache.json";
}

void AutoTuner::ensure_cache_dir() {
    const char* home = ::getenv("HOME");
    if (!home) home = "/tmp";
    const std::string dir = std::string(home) + "/.tinyinfer";
    // mkdir returns 0 on success, -1 if already exists (EEXIST) — both are fine.
    ::mkdir(dir.c_str(), 0755);
}

// ---------------------------------------------------------------------------
// Benchmark one tile size: returns median latency in nanoseconds.
// ---------------------------------------------------------------------------
double AutoTuner::benchmark_one(
    size_t tile_size,
    size_t M, size_t N, size_t K,
    int iterations)
{
    // Allocate aligned matrices for benchmarking.
    // We use posix_memalign here (not TinyAllocator) to keep the benchmark
    // self-contained and avoid confounding effects from our own allocator.
    constexpr size_t ALIGN = 64;  // one cache line

    void *raw_A = nullptr, *raw_B = nullptr, *raw_C = nullptr;
    posix_memalign(&raw_A, ALIGN, M * K * sizeof(float));
    posix_memalign(&raw_B, ALIGN, K * N * sizeof(float));
    posix_memalign(&raw_C, ALIGN, M * N * sizeof(float));

    float* A = reinterpret_cast<float*>(raw_A);
    float* B = reinterpret_cast<float*>(raw_B);
    float* C = reinterpret_cast<float*>(raw_C);

    // Fill with small non-zero values to avoid denormals (which are slow on
    // some hardware; M4 handles them in hardware but it's good practice).
    for (size_t i = 0; i < M * K; ++i) A[i] = static_cast<float>(i % 17) * 0.01f;
    for (size_t i = 0; i < K * N; ++i) B[i] = static_cast<float>(i % 13) * 0.01f;

    // Warm-up run: execute once to bring data into cache and amortize
    // any one-time JIT or branch-predictor warm-up costs.
    kernels::tiled_matmul_f32(A, B, C, M, N, K, tile_size);

    // Timed runs: collect individual iteration latencies.
    std::vector<double> times_ns(iterations);
    for (int iter = 0; iter < iterations; ++iter) {
        memset(C, 0, M * N * sizeof(float));  // reset output
        const auto t0 = std::chrono::high_resolution_clock::now();
        kernels::tiled_matmul_f32(A, B, C, M, N, K, tile_size);
        const auto t1 = std::chrono::high_resolution_clock::now();
        times_ns[iter] = static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    }

    std::free(raw_A);
    std::free(raw_B);
    std::free(raw_C);

    // Return median: sort and pick middle element.
    // Median is robust to outliers from OS preemption (rare but real on macOS).
    std::sort(times_ns.begin(), times_ns.end());
    return times_ns[iterations / 2];
}

// ---------------------------------------------------------------------------
// Main tuning function: benchmark all candidate tile sizes, pick the best.
// ---------------------------------------------------------------------------
TuningResult AutoTuner::benchmark_tile_sizes(size_t M, size_t N, size_t K) {
    const double flops_per_call = 2.0 * M * N * K;  // GEMM: 2*M*N*K FLOP

    std::cout << "\n[AutoTuner] Benchmarking tile sizes for " << M << "×" << N
              << "×" << K << " GEMM on " << hardware_id_ << "\n";
    std::cout << "[AutoTuner] Running " << TUNE_ITERATIONS
              << " iterations per tile size...\n\n";

    TuningResult result;
    result.best_gflops = 0.0;

    const auto tune_start = std::chrono::high_resolution_clock::now();

    // Print table header.
    std::cout << "  Tile | Median ns | GFLOP/s\n";
    std::cout << "  -----|-----------|--------\n";

    for (size_t tile : CANDIDATE_TILE_SIZES) {
        const double median_ns = benchmark_one(tile, M, N, K, TUNE_ITERATIONS);
        const double gflops = flops_per_call / median_ns;  // GFLOP/s (ns cancels 1e9)

        std::cout << "   " << tile << "  |  " << static_cast<long>(median_ns)
                  << "  |  " << gflops << "\n";

        if (gflops > result.best_gflops) {
            result.best_gflops       = gflops;
            result.optimal_tile_size = tile;
        }
    }

    const auto tune_end = std::chrono::high_resolution_clock::now();
    result.tune_time_ms = static_cast<double>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            tune_end - tune_start).count());

    std::cout << "\n[AutoTuner] Winner: tile_size=" << result.optimal_tile_size
              << " at " << result.best_gflops << " GFLOP/s\n";
    std::cout << "[AutoTuner] Tuning completed in " << result.tune_time_ms << " ms\n\n";

    return result;
}

// ---------------------------------------------------------------------------
// Minimal JSON serializer (no external dependency)
// ---------------------------------------------------------------------------

static void save_json(
    const std::string& path,
    const std::string& hardware,
    const std::unordered_map<std::string, TuningResult>& results)
{
    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "[AutoTuner] Warning: could not write cache to " << path << "\n";
        return;
    }

    f << "{\n";
    f << "  \"version\": 1,\n";
    f << "  \"hardware\": \"" << hardware << "\",\n";
    f << "  \"tile_sizes\": {\n";

    bool first = true;
    for (const auto& [key, res] : results) {
        if (!first) f << ",\n";
        first = false;
        f << "    \"" << key << "\": {\n";
        f << "      \"optimal_tile\": " << res.optimal_tile_size << ",\n";
        f << "      \"gflops\": "       << res.best_gflops       << ",\n";
        f << "      \"tune_ms\": "      << res.tune_time_ms      << "\n";
        f << "    }";
    }

    f << "\n  }\n}\n";
    std::cout << "[AutoTuner] Saved tuning cache to " << path << "\n";
}

// ---------------------------------------------------------------------------
// Minimal JSON parser (hand-rolled, handles our specific format only)
// ---------------------------------------------------------------------------
static bool parse_json_cache(
    const std::string& path,
    std::string& out_hardware,
    std::unordered_map<std::string, TuningResult>& out_results)
{
    std::ifstream f(path);
    if (!f.is_open()) return false;

    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());

    // Extract hardware string.
    auto find_str_value = [&](const std::string& key) -> std::string {
        const std::string search = "\"" + key + "\": \"";
        const size_t pos = content.find(search);
        if (pos == std::string::npos) return "";
        const size_t start = pos + search.size();
        const size_t end = content.find("\"", start);
        if (end == std::string::npos) return "";
        return content.substr(start, end - start);
    };
    auto find_num_value = [&](const std::string& full_text, const std::string& key) -> double {
        const std::string search = "\"" + key + "\": ";
        const size_t pos = full_text.find(search);
        if (pos == std::string::npos) return 0.0;
        return std::stod(full_text.substr(pos + search.size()));
    };

    out_hardware = find_str_value("hardware");

    // Parse tile_sizes block: find each quoted key followed by a JSON object.
    const std::string ts_marker = "\"tile_sizes\": {";
    const size_t ts_pos = content.find(ts_marker);
    if (ts_pos == std::string::npos) return false;

    size_t search_from = ts_pos + ts_marker.size();
    while (true) {
        // Find next key: "MxNxK"
        const size_t key_start = content.find("\"", search_from);
        if (key_start == std::string::npos) break;
        const size_t key_end = content.find("\"", key_start + 1);
        if (key_end == std::string::npos) break;
        const std::string key = content.substr(key_start + 1, key_end - key_start - 1);
        if (key == "tile_sizes" || key.empty()) break;  // end of tile_sizes block

        // Find the corresponding value object.
        const size_t obj_start = content.find("{", key_end);
        const size_t obj_end   = content.find("}", obj_start);
        if (obj_start == std::string::npos || obj_end == std::string::npos) break;

        const std::string obj = content.substr(obj_start, obj_end - obj_start + 1);

        TuningResult res;
        res.optimal_tile_size = static_cast<size_t>(find_num_value(obj, "optimal_tile"));
        res.best_gflops       = find_num_value(obj, "gflops");
        res.tune_time_ms      = find_num_value(obj, "tune_ms");

        if (res.optimal_tile_size > 0) {
            out_results[key] = res;
        }

        search_from = obj_end + 1;
    }

    return !out_results.empty();
}

// ---------------------------------------------------------------------------
// load_from_cache / save_to_cache
// ---------------------------------------------------------------------------

bool AutoTuner::load_from_cache() {
    const std::string path = cache_file_path();
    if (::access(path.c_str(), F_OK) != 0) return false;  // file doesn't exist

    std::string cached_hw;
    std::unordered_map<std::string, TuningResult> cached_results;
    if (!parse_json_cache(path, cached_hw, cached_results)) {
        std::cout << "[AutoTuner] Cache file malformed, re-tuning.\n";
        return false;
    }

    // Invalidate cache if the hardware has changed (user swapped machines).
    if (cached_hw != hardware_id_) {
        std::cout << "[AutoTuner] Hardware changed (" << cached_hw
                  << " → " << hardware_id_ << "), re-tuning.\n";
        return false;
    }

    results_ = std::move(cached_results);
    std::cout << "[AutoTuner] Loaded " << results_.size()
              << " tuning result(s) from cache.\n";
    return true;
}

void AutoTuner::save_to_cache() const {
    ensure_cache_dir();
    save_json(cache_file_path(), hardware_id_, results_);
}

// ---------------------------------------------------------------------------
// load_or_tune: the primary public entry point.
// ---------------------------------------------------------------------------
void AutoTuner::load_or_tune() {
    if (load_from_cache()) {
        tuned_ = true;
        return;
    }

    // Cache miss: run benchmarks for the representative problem size.
    const std::string key = make_key(
        TUNE_MATRIX_SIZE, TUNE_MATRIX_SIZE, TUNE_MATRIX_SIZE);
    results_[key] = benchmark_tile_sizes(
        TUNE_MATRIX_SIZE, TUNE_MATRIX_SIZE, TUNE_MATRIX_SIZE);

    save_to_cache();
    tuned_ = true;
}

// ---------------------------------------------------------------------------
// get_optimal_tile_size
// ---------------------------------------------------------------------------
std::string AutoTuner::make_key(size_t M, size_t N, size_t K) {
    return std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K);
}

size_t AutoTuner::get_optimal_tile_size(size_t M, size_t N, size_t K) const {
    // Try exact match first.
    const std::string key = make_key(M, N, K);
    auto it = results_.find(key);
    if (it != results_.end()) {
        return it->second.optimal_tile_size;
    }

    // Fall back to the representative tuning result.
    const std::string default_key = make_key(
        TUNE_MATRIX_SIZE, TUNE_MATRIX_SIZE, TUNE_MATRIX_SIZE);
    auto it2 = results_.find(default_key);
    if (it2 != results_.end()) {
        return it2->second.optimal_tile_size;
    }

    // No tuning data at all — return a sensible default.
    return 64;
}

void AutoTuner::print_results() const {
    std::cout << "\n[AutoTuner] Cached results for hardware: " << hardware_id_ << "\n";
    for (const auto& [key, res] : results_) {
        std::cout << "  " << key << " → tile=" << res.optimal_tile_size
                  << ", " << res.best_gflops << " GFLOP/s\n";
    }
}

} // namespace tinyinfer
