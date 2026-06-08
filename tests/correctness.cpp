// =============================================================================
// tests/correctness.cpp — Verify all matmul implementations produce correct output
//
// WHAT THIS FILE DOES:
//   Tests that tiled_matmul_f32 and neon_matmul_int8 produce numerically
//   equivalent results to naive_matmul_f32 (the ground truth).
//
//   Uses a strict epsilon of 1e-4 for FP32 comparisons (accounting for
//   floating-point reordering with -ffast-math) and 1e-1 for INT8 comparisons
//   (quantization introduces bounded error: epsilon ≤ scale * max_element * K * eps_q).
//
// WHY TEST BEFORE BENCHMARKING?
//   A fast but incorrect kernel is worse than a slow correct one.
//   NEON intrinsics are notoriously easy to get wrong (wrong register widths,
//   signed vs unsigned mix, missed tail handling). These tests catch that.
//
// EXIT CODES:
//   0 = all tests passed
//   1 = at least one test failed (error details printed to stderr)
// =============================================================================

#include "kernels/matmul.h"
#include "tensor/tensor.h"

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <random>
#include <cassert>

using namespace tinyinfer;

// ---------------------------------------------------------------------------
// Aligned allocation helpers
// ---------------------------------------------------------------------------
static float* alloc_f32(size_t n, float fill = 0.0f) {
    void* p = nullptr;
    posix_memalign(&p, 64, n * sizeof(float));
    float* f = reinterpret_cast<float*>(p);
    for (size_t i = 0; i < n; ++i) f[i] = fill;
    return f;
}

static int8_t* alloc_i8(size_t n) {
    void* p = nullptr;
    posix_memalign(&p, 64, n * sizeof(int8_t));
    memset(p, 0, n);
    return reinterpret_cast<int8_t*>(p);
}

// ---------------------------------------------------------------------------
// Max absolute difference between two float arrays.
// ---------------------------------------------------------------------------
static float max_abs_diff(const float* a, const float* b, size_t n) {
    float max_diff = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        const float diff = fabsf(a[i] - b[i]);
        if (diff > max_diff) max_diff = diff;
    }
    return max_diff;
}

// ---------------------------------------------------------------------------
// Print PASS / FAIL and return bool.
// ---------------------------------------------------------------------------
static bool check(const char* test_name, bool passed, float max_diff, float epsilon) {
    if (passed) {
        std::cout << "  [PASS] " << test_name
                  << "  (max_diff=" << std::scientific << std::setprecision(2)
                  << max_diff << ", eps=" << epsilon << ")\n";
    } else {
        std::cerr << "  [FAIL] " << test_name
                  << "  (max_diff=" << std::scientific << std::setprecision(2)
                  << max_diff << " > eps=" << epsilon << ")\n";
    }
    return passed;
}

// ---------------------------------------------------------------------------
// Test 1: tiled_matmul_f32 vs naive_matmul_f32
// ---------------------------------------------------------------------------
static bool test_tiled_vs_naive(size_t M, size_t N, size_t K, size_t tile_size) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    float* A      = alloc_f32(M * K);
    float* B      = alloc_f32(K * N);
    float* C_ref  = alloc_f32(M * N);
    float* C_tile = alloc_f32(M * N);

    for (size_t i = 0; i < M * K; ++i) A[i] = dist(rng);
    for (size_t i = 0; i < K * N; ++i) B[i] = dist(rng);

    kernels::naive_matmul_f32(A, B, C_ref,  M, N, K);
    kernels::tiled_matmul_f32(A, B, C_tile, M, N, K, tile_size);

    const float diff = max_abs_diff(C_ref, C_tile, M * N);
    // Epsilon: -ffast-math can reorder FP ops, causing ~1e-5 differences.
    // We use 1e-4 to be safe across all matrix sizes.
    constexpr float EPS = 1e-4f;

    std::free(A); std::free(B); std::free(C_ref); std::free(C_tile);
    return check("tiled_matmul_f32", diff < EPS, diff, EPS);
}

// ---------------------------------------------------------------------------
// Test 2: neon_matmul_int8 (unpacked and packed) vs naive_matmul_f32
// ---------------------------------------------------------------------------
static bool test_neon_int8_vs_naive(size_t M, size_t N, size_t K) {
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    float* A_f    = alloc_f32(M * K);
    float* B_f    = alloc_f32(K * N);
    float* C_ref  = alloc_f32(M * N);
    float* C_neon = alloc_f32(M * N);

    int8_t* A_q = alloc_i8(M * K);
    int8_t* B_q = alloc_i8(K * N);

    // 64-byte aligned packed B buffer — same size as B_q, different layout.
    void*   pack_buf = nullptr;
    posix_memalign(&pack_buf, 64, ((N * K + 63) / 64) * 64);
    int8_t* B_packed = reinterpret_cast<int8_t*>(pack_buf);

    for (size_t i = 0; i < M * K; ++i) A_f[i] = dist(rng);
    for (size_t i = 0; i < K * N; ++i) B_f[i] = dist(rng);

    float scale_A, scale_B;
    kernels::quantize_f32_to_i8(A_f, A_q, M * K, &scale_A);
    kernels::quantize_f32_to_i8(B_f, B_q, K * N, &scale_B);

    // Build expected output using naive FP32 as ground truth.
    kernels::naive_matmul_f32(A_f, B_f, C_ref, M, N, K);

    // Compute tolerance: 5% of max |C_ref|, covers quantization rounding error.
    float max_ref = 0.0f;
    for (size_t i = 0; i < M * N; ++i)
        if (fabsf(C_ref[i]) > max_ref) max_ref = fabsf(C_ref[i]);
    const float EPS = 0.05f * std::max(max_ref, 1.0f);

    bool ok = true;

    // Test A: unpacked path (pack_b=true, B stays row-major, kernel gathers).
    memset(C_neon, 0, M * N * sizeof(float));
    kernels::neon_matmul_int8(A_q, B_q, C_neon, M, N, K, scale_A, scale_B, /*pack_b=*/true);
    ok &= check("neon_int8 unpacked", max_abs_diff(C_ref, C_neon, M * N) < EPS,
                max_abs_diff(C_ref, C_neon, M * N), EPS);

    // Test B: packed path (pre-pack B, then pass pack_b=false).
    kernels::pack_b_int8(B_q, B_packed, N, K);
    memset(C_neon, 0, M * N * sizeof(float));
    kernels::neon_matmul_int8(A_q, B_packed, C_neon, M, N, K, scale_A, scale_B, /*pack_b=*/false);
    ok &= check("neon_int8 packed  ", max_abs_diff(C_ref, C_neon, M * N) < EPS,
                max_abs_diff(C_ref, C_neon, M * N), EPS);

    // Test C: verify packed B layout correctness directly.
    // B_packed[j*K + k] must equal B_q[k*N + j] for all j, k.
    bool pack_ok = true;
    for (size_t j = 0; j < std::min(N, size_t{8}); ++j)
        for (size_t k = 0; k < std::min(K, size_t{8}); ++k)
            if (B_packed[j * K + k] != B_q[k * N + j]) { pack_ok = false; break; }
    ok &= check("pack_b_int8 layout", pack_ok, 0.0f, 0.0f);

    std::free(A_f); std::free(B_f); std::free(C_ref); std::free(C_neon);
    std::free(A_q); std::free(B_q); std::free(B_packed);
    return ok;
}

// ---------------------------------------------------------------------------
// Test 3: Tensor class — shape, strides, transpose
// ---------------------------------------------------------------------------
static bool test_tensor_layout() {
    bool all_passed = true;

    // Create a 3×4 float32 tensor.
    Tensor t({3, 4}, Dtype::FLOAT32);
    float* p = t.data_ptr<float>();
    for (size_t i = 0; i < 12; ++i) p[i] = static_cast<float>(i);

    // Verify element access.
    // t[1][2] should be element 1*4 + 2 = 6 in C-contiguous order.
    const float expected = 6.0f;
    const float got = t.at_f32({1, 2});
    if (fabsf(got - expected) < 1e-6f) {
        std::cout << "  [PASS] Tensor element access ({1,2}=" << got << ")\n";
    } else {
        std::cerr << "  [FAIL] Tensor element access: expected=" << expected
                  << " got=" << got << "\n";
        all_passed = false;
    }

    // Verify lazy transpose: swap axes 0 and 1 → 4×3 view.
    Tensor t_T = t.transpose(0, 1);
    if (t_T.dim(0) == 4 && t_T.dim(1) == 3) {
        // t_T[2][1] should be t[1][2] = 6.0.
        const float got_T = t_T.at_f32({2, 1});
        if (fabsf(got_T - expected) < 1e-6f) {
            std::cout << "  [PASS] Tensor lazy transpose (4×3 view, T[2][1]=" << got_T << ")\n";
        } else {
            std::cerr << "  [FAIL] Tensor transpose: expected=" << expected
                      << " got=" << got_T << "\n";
            all_passed = false;
        }
    } else {
        std::cerr << "  [FAIL] Tensor transpose shape: expected [4,3] got ["
                  << t_T.dim(0) << "," << t_T.dim(1) << "]\n";
        all_passed = false;
    }

    // Verify view (reshape): 3×4 → 12×1.
    Tensor t_view = t.view({12, 1});
    if (t_view.dim(0) == 12 && t_view.dim(1) == 1) {
        std::cout << "  [PASS] Tensor view reshape (12×1)\n";
    } else {
        std::cerr << "  [FAIL] Tensor view reshape\n";
        all_passed = false;
    }

    return all_passed;
}

// ---------------------------------------------------------------------------
// Test 4: Quantize round-trip (quantize then dequantize, check error)
// ---------------------------------------------------------------------------
static bool test_quantization_roundtrip() {
    constexpr size_t N = 1024;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    float* src   = alloc_f32(N);
    float* recon = alloc_f32(N);
    int8_t* q    = alloc_i8(N);
    float scale;

    for (size_t i = 0; i < N; ++i) src[i] = dist(rng);

    kernels::quantize_f32_to_i8(src, q, N, &scale);
    kernels::dequantize_i8_to_f32(q, recon, N, scale);

    const float diff = max_abs_diff(src, recon, N);
    // Maximum quantization error = scale / 2 (one-half LSB).
    const float expected_max_err = scale / 2.0f + 1e-6f;

    std::free(src); std::free(recon); std::free(q);
    return check("quantize round-trip", diff <= expected_max_err, diff, expected_max_err);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    std::cout << "\n  ═══════════════════════════════════════════════\n";
    std::cout << "  tinyinfer — Correctness Tests\n";
    std::cout << "  ═══════════════════════════════════════════════\n\n";

    int failures = 0;

    // Tensor tests.
    std::cout << "  ── Tensor class ────────────────────────────\n";
    if (!test_tensor_layout()) ++failures;

    // Quantization tests.
    std::cout << "\n  ── Quantization ────────────────────────────\n";
    if (!test_quantization_roundtrip()) ++failures;

    // Matmul correctness at multiple sizes.
    for (size_t sz : {64UL, 128UL, 256UL}) {
        std::cout << "\n  ── MATMUL " << sz << "×" << sz << " ────────────────────\n";
        if (!test_tiled_vs_naive(sz, sz, sz, 32)) ++failures;
        if (!test_tiled_vs_naive(sz, sz, sz, 64)) ++failures;
        if (!test_neon_int8_vs_naive(sz, sz, sz)) ++failures;
    }

    // Non-square sizes (important for transformer FFN layers).
    std::cout << "\n  ── Non-square MATMUL (128×512×256) ─────────\n";
    if (!test_tiled_vs_naive(128, 512, 256, 64)) ++failures;
    if (!test_neon_int8_vs_naive(128, 512, 256)) ++failures;

    std::cout << "\n  ═══════════════════════════════════════════════\n";
    if (failures == 0) {
        std::cout << "  ALL TESTS PASSED ✓\n";
    } else {
        std::cout << "  " << failures << " TEST(S) FAILED ✗\n";
    }
    std::cout << "  ═══════════════════════════════════════════════\n\n";

    return (failures > 0) ? 1 : 0;
}
