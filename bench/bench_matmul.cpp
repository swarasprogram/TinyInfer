// =============================================================================
// bench/bench_matmul.cpp — Matmul benchmark: naive vs tiled vs NEON INT8
//
// WHAT THIS FILE DOES:
//   Benchmarks all three matmul implementations across 4 matrix sizes.
//   Prints a clean ASCII table with GFLOP/s, GB/s memory bandwidth, and
//   speedup relative to naive. Also runs the AutoTuner first.
//
// HOW TO READ THE OUTPUT:
//   GFLOP/s (Giga Floating-Point Operations per second):
//     matmul(M,N,K) requires 2*M*N*K FLOP (one multiply + one add per element).
//     Higher is better. M4 P-core FP32 peak ≈ 27 GFLOP/s (2-wide FMA, 3.4 GHz).
//
//   GB/s (memory bandwidth):
//     Reads: A (M*K) + B (K*N) bytes. Writes: C (M*N) bytes.
//     Higher is better. M4 peak unified memory bandwidth ≈ 100 GB/s.
//     When GB/s approaches 100, the kernel is memory-bandwidth-limited.
//     When GFLOP/s approaches 27, it's compute-limited. Both limits are good.
//
//   Speedup: GFLOP/s_optimized / GFLOP/s_naive
//
// RUN:
//   ./build/bench_matmul
// =============================================================================

#include "kernels/matmul.h"
#include "tuner/autotuner.h"

#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <iostream>
#include <iomanip>

using namespace tinyinfer;

// ---------------------------------------------------------------------------
// Aligned allocation helper (benchmark-local, no TinyAllocator overhead)
// ---------------------------------------------------------------------------
static float* alloc_f32(size_t n) {
    void* p = nullptr;
    posix_memalign(&p, 64, n * sizeof(float));
    memset(p, 0, n * sizeof(float));
    return reinterpret_cast<float*>(p);
}

static int8_t* alloc_i8(size_t n) {
    void* p = nullptr;
    posix_memalign(&p, 64, n * sizeof(int8_t));
    memset(p, 0, n * sizeof(int8_t));
    return reinterpret_cast<int8_t*>(p);
}

// ---------------------------------------------------------------------------
// Timing helper: run a function N times, return median nanoseconds.
// ---------------------------------------------------------------------------
template<typename F>
double median_ns(F&& fn, int iterations = 10) {
    std::vector<double> times(iterations);
    for (int i = 0; i < iterations; ++i) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        fn();
        const auto t1 = std::chrono::high_resolution_clock::now();
        times[i] = static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    }
    std::sort(times.begin(), times.end());
    return times[iterations / 2];
}

// ---------------------------------------------------------------------------
// Compute GFLOP/s and GB/s from timing and matrix sizes.
// ---------------------------------------------------------------------------
static double to_gflops(size_t M, size_t N, size_t K, double ns) {
    // 2*M*N*K FLOP per matmul call (multiply + accumulate for each element).
    return (2.0 * M * N * K) / ns;  // GFLOP/s (ns and 1e9 cancel)
}

static double to_gbps(size_t M, size_t N, size_t K, double ns) {
    // Memory traffic: read A (M*K) + read B (K*N) + write C (M*N), all float32.
    const double bytes = static_cast<double>((M*K + K*N + M*N) * sizeof(float));
    return bytes / ns;  // GB/s (ns and 1e9 cancel)
}

// ---------------------------------------------------------------------------
// Print separator line
// ---------------------------------------------------------------------------
static void print_sep() {
    std::cout << "  " << std::string(84, '-') << "\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    std::cout << "\n";
    std::cout << "  ╔══════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "  ║              tinyinfer — MATMUL Benchmark (Apple M4)                        ║\n";
    std::cout << "  ╚══════════════════════════════════════════════════════════════════════════════╝\n\n";

    // Step 1: Run AutoTuner (loads from cache or benchmarks).
    std::cout << "  ── Phase 1: AutoTuner ─────────────────────────────────────────────────────\n";
    auto& tuner = AutoTuner::instance();
    tuner.load_or_tune();
    tuner.print_results();
    std::cout << "\n";

    // Matrix sizes to benchmark: from small (MLP hidden layer) to large (attention).
    // These correspond to real transformer layer dimensions:
    //   256  → small GPT-2 attention head dim
    //   512  → LLaMA-7B intermediate size / 4
    //   1024 → LLaMA-7B hidden dim
    //   2048 → LLaMA-7B full FFN intermediate size
    const std::array<size_t, 4> sizes = {256, 512, 1024, 2048};

    // Iterations: fewer for large sizes so the bench runs in <60s total.
    // Naive at 2048 is so slow it's skipped entirely.
    constexpr int BENCH_ITERS = 10;

    // Print table header. Five columns: naive, tiled, NEON-unpacked, NEON-packed, speedup.
    std::cout << "  ── Phase 2: Benchmark Results ─────────────────────────────────────────────\n\n";
    std::cout << "  " << std::left
              << std::setw(10) << "Size"
              << std::setw(16) << "Naive GF/s"
              << std::setw(16) << "Tiled GF/s"
              << std::setw(20) << "NEON-unpacked GF/s"
              << std::setw(20) << "NEON-packed GF/s"
              << "Speedup(packed/naive)\n";
    print_sep();

    for (const size_t sz : sizes) {
        const size_t M = sz, N = sz, K = sz;
        const size_t tile = tuner.get_optimal_tile_size(M, N, K);

        // ---------------------------------------------------------------------------
        // Allocate all matrices.
        // ---------------------------------------------------------------------------
        float*  A_f = alloc_f32(M * K);
        float*  B_f = alloc_f32(K * N);
        float*  C_f = alloc_f32(M * N);

        int8_t* A_q = alloc_i8(M * K);
        int8_t* B_q = alloc_i8(K * N);
        float*  C_q = alloc_f32(M * N);

        // Pre-packed B: allocated 64-byte aligned, packed ONCE here (outside timing loop).
        // This models the production use-case: pack at model-load time, not per-token.
        int8_t* B_packed = nullptr;
        {
            void* tmp = nullptr;
            const size_t pack_bytes = ((N * K + 63) / 64) * 64;
            posix_memalign(&tmp, 64, pack_bytes);
            B_packed = reinterpret_cast<int8_t*>(tmp);
        }

        // Fill with non-trivial values (avoids degenerate branch-prediction patterns).
        for (size_t i = 0; i < M * K; ++i) A_f[i] = (i % 23) * 0.01f - 0.1f;
        for (size_t i = 0; i < K * N; ++i) B_f[i] = (i % 19) * 0.01f - 0.09f;

        float scale_A, scale_B;
        kernels::quantize_f32_to_i8(A_f, A_q, M * K, &scale_A);
        kernels::quantize_f32_to_i8(B_f, B_q, K * N, &scale_B);

        // Pack B into column-major layout. Time it so we can report the one-time cost.
        const auto pack_t0 = std::chrono::high_resolution_clock::now();
        kernels::pack_b_int8(B_q, B_packed, N, K);
        const auto pack_t1 = std::chrono::high_resolution_clock::now();
        const double pack_ms = static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(pack_t1 - pack_t0).count()) / 1000.0;

        // ---------------------------------------------------------------------------
        // Benchmark 1: naive FP32 (skip at 2048 — would take ~30s per iteration).
        // ---------------------------------------------------------------------------
        const bool skip_naive = (sz >= 2048);
        double naive_gflops = 0.0;
        if (!skip_naive) {
            const double naive_ns = median_ns([&]{
                kernels::naive_matmul_f32(A_f, B_f, C_f, M, N, K);
            }, BENCH_ITERS);
            naive_gflops = to_gflops(M, N, K, naive_ns);
        }

        // ---------------------------------------------------------------------------
        // Benchmark 2: tiled FP32.
        // ---------------------------------------------------------------------------
        const double tiled_ns     = median_ns([&]{
            kernels::tiled_matmul_f32(A_f, B_f, C_f, M, N, K, tile);
        }, BENCH_ITERS);
        const double tiled_gflops = to_gflops(M, N, K, tiled_ns);
        const double tiled_gbps   = to_gbps(M, N, K, tiled_ns);
        (void)tiled_gbps;

        // ---------------------------------------------------------------------------
        // Benchmark 3: NEON INT8, unpacked (old behavior).
        // pack_b=true means it gathers B column-wise internally — the slow path.
        // The packing inside the call is part of the measured time deliberately,
        // to show the true per-call cost when you don't pre-pack.
        // ---------------------------------------------------------------------------
        const double neon_unp_ns = median_ns([&]{
            kernels::neon_matmul_int8(A_q, B_q, C_q, M, N, K, scale_A, scale_B,
                                      /*pack_b=*/true);
        }, BENCH_ITERS);
        const double neon_unp_gf = to_gflops(M, N, K, neon_unp_ns);

        // ---------------------------------------------------------------------------
        // Benchmark 4: NEON INT8, pre-packed (new fast path).
        // B_packed was prepared OUTSIDE the timing loop — models inference hot path.
        // pack_b=false: the kernel uses B_packed directly with no overhead.
        // ---------------------------------------------------------------------------
        const double neon_pk_ns = median_ns([&]{
            kernels::neon_matmul_int8(A_q, B_packed, C_q, M, N, K, scale_A, scale_B,
                                     /*pack_b=*/false);
        }, BENCH_ITERS);
        const double neon_pk_gf = to_gflops(M, N, K, neon_pk_ns);

        // Speedup: packed NEON vs naive (or vs tiled when naive is skipped).
        const double base    = (naive_gflops > 0.0) ? naive_gflops : tiled_gflops;
        const double speedup = neon_pk_gf / base;

        // ---------------------------------------------------------------------------
        // Format and print row.
        // ---------------------------------------------------------------------------
        auto fmt = [](double v, int w = 6) {
            std::ostringstream s;
            s << std::fixed << std::setprecision(2) << std::setw(w) << v;
            return s.str();
        };
        std::ostringstream size_str;
        size_str << sz << "×" << sz;

        std::cout << "  " << std::left
                  << std::setw(10) << size_str.str()
                  << std::setw(16) << (skip_naive ? "  skipped" : fmt(naive_gflops))
                  << std::setw(16) << fmt(tiled_gflops)
                  << std::setw(20) << fmt(neon_unp_gf)
                  << std::setw(20) << fmt(neon_pk_gf)
                  << fmt(speedup) << "x"
                  << "  (pack=" << std::fixed << std::setprecision(2) << pack_ms << "ms)\n";

        std::free(A_f); std::free(B_f); std::free(C_f);
        std::free(A_q); std::free(B_q); std::free(C_q); std::free(B_packed);
    }

    print_sep();

    // Print interpretation guide.
    std::cout << "\n  ── Notes ───────────────────────────────────────────────────────────────────\n";
    std::cout << "  M4 P-core theoretical peaks:\n";
    std::cout << "    FP32 GFLOP/s (per core): ~27  (2-wide FMA × 4 lanes × 3.4 GHz)\n";
    std::cout << "    INT8 GOPS/s  (per core): ~109 (vdotq × 4 groups × 16 elems × 3.4 GHz)\n";
    std::cout << "    Memory BW               : ~100 GB/s (unified LPDDR5X)\n\n";
    std::cout << "  What each column shows:\n";
    std::cout << "    Naive          : i-j-k loop, B column accesses cause 100% cache miss rate\n";
    std::cout << "    Tiled          : i-k-j loop with cache-blocking, L1 stays warm\n";
    std::cout << "    NEON-unpacked  : vdotq + gather; 16 scalar loads to fill each B register\n";
    std::cout << "    NEON-packed    : vdotq + pack_b_int8; B columns are stride-1 → true SIMD\n\n";
    std::cout << "  B packing is a one-time cost at model load (shown as 'pack=Xms').\n";
    std::cout << "  In inference, B = weight matrix (fixed). Pack once, run forever.\n";
    std::cout << "  AutoTuner selected tile_size="
              << tuner.get_optimal_tile_size(512, 512, 512)
              << " for 512×512 workloads on this hardware.\n\n";

    return 0;
}
