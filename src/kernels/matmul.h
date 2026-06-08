// =============================================================================
// kernels/matmul.h — Three matrix multiplication implementations
//
// WHAT THIS FILE DOES:
//   Declares three progressively optimized MATMUL kernels:
//   1. naive_matmul_f32   — baseline triple loop (cache-unfriendly)
//   2. tiled_matmul_f32   — cache-blocked with configurable tile size
//   3. neon_matmul_int8   — ARM NEON INT8 with DOT PRODUCT instructions
//
// WHY THREE IMPLEMENTATIONS?
//   Each one isolates a specific optimization:
//     naive  → tiled  : shows the impact of cache blocking alone
//     tiled  → NEON   : shows the impact of SIMD + quantization together
//   The benchmark prints a table comparing all three. This lets a reader
//   (e.g., a Ziroh Labs engineer) understand *why* each technique matters,
//   not just *that* the final version is fast.
//
// HARDWARE CONTEXT — WHAT IS A CACHE LINE?
//   A cache line is the unit of data transfer between DRAM and L1/L2 cache.
//   On Apple M4 (ARM Cortex-X): cache line = 64 bytes = 16 float32 values.
//
//   When you access M[i][j], the CPU fetches the entire 64-byte cache line
//   containing that address — you get M[i][j] through M[i][j+15] "for free".
//
//   Implication: if your inner loop reads M[i][j], M[i][j+1], ..., you pay
//   for one cache miss and get 15 hits. That's good spatial locality.
//
//   But if your inner loop reads M[0][j], M[1][j], M[2][j] (column access),
//   each access is in a DIFFERENT row → different cache line → N cache misses
//   for N rows. That's the catastrophic case that naive_matmul_f32 hits for B.
//
// WHY i-k-j LOOP ORDER IS BETTER THAN i-j-k:
//   For C[i][j] += A[i][k] * B[k][j]:
//
//   i-j-k (naive): inner loop over k
//     - A[i][k]: sequential access (good)
//     - B[k][j]: fixed j, varying k → column access → CACHE MISS EVERY STEP
//
//   i-k-j (tiled): inner loop over j
//     - A[i][k]: fixed for entire j-loop (stays in register)
//     - B[k][j]: sequential row access → CACHE HIT on 15/16 accesses
//     - C[i][j]: sequential row access → CACHE HIT on 15/16 accesses
//
//   The i-k-j order turns matmul from memory-bandwidth-bound into
//   compute-bound — which is where we want it.
//
// REFERENCES:
//   [1] "Optimizing Matrix Multiply" — CMU 15-418/618 (Spring 2023)
//   [2] ARM NEON Programmer's Guide — §5 "Integer SIMD"
//   [3] BLIS framework paper: "Anatomy of High-Performance Matrix Multiplication"
//       van Zee & van de Geijn, TOMS 2015
// =============================================================================

#pragma once

#include <cstddef>   // size_t
#include <cstdint>   // int8_t, int32_t

namespace tinyinfer {
namespace kernels {

// ---------------------------------------------------------------------------
// IMPLEMENTATION 1: naive_matmul_f32
//
// Triple loop in i-j-k order. Intentionally cache-unfriendly as a baseline.
// This is what every CS101 student writes; this is what we beat.
//
// Expected performance on M4 (1024×1024):
//   ~0.5–1.0 GFLOP/s (limited by B matrix column cache misses)
// ---------------------------------------------------------------------------
void naive_matmul_f32(
    const float* A,    // [M × K], row-major
    const float* B,    // [K × N], row-major
    float*       C,    // [M × N], row-major (output, caller zeroes first)
    size_t M, size_t N, size_t K
);

// ---------------------------------------------------------------------------
// IMPLEMENTATION 2: tiled_matmul_f32
//
// Cache-blocked matmul. Divides A, B, C into (tile_size × tile_size) tiles.
// Each tile fits in L1 cache → dramatically reduces cache misses.
//
// HOW TILE SIZE AFFECTS PERFORMANCE:
//   L1 data cache on M4 P-core: 192 KB
//   Each tile of A: tile^2 * 4 bytes (float)
//   Each tile of B: tile^2 * 4 bytes
//   Each tile of C: tile^2 * 4 bytes
//   Total: 3 * tile^2 * 4 bytes must fit in L1.
//   Solving: tile^2 = 192*1024 / (3*4) = 16384 → tile = 128
//
//   But tile_size=128 assumes PERFECT cache utilization with no conflict misses.
//   In practice, 64 or 80 often wins due to set-associativity effects.
//   This is why the autotuner is necessary — theory gives a ceiling, not an answer.
//
// Expected performance on M4 (1024×1024, tile=64):
//   ~4–8 GFLOP/s (cache miss rate drops ~20×)
// ---------------------------------------------------------------------------
void tiled_matmul_f32(
    const float* A,
    const float* B,
    float*       C,
    size_t M, size_t N, size_t K,
    size_t tile_size    // set by AutoTuner::get_optimal_tile_size()
);

// ---------------------------------------------------------------------------
// B-PACKING: pack_b_int8
//
// WHAT THE GATHER PENALTY IS (and why packing eliminates it):
//
//   Without packing, the inner loop for dot(A[i,:], B[:,j]) accesses B column-wise:
//     B[0][j], B[1][j], ..., B[K-1][j]
//   In row-major storage these addresses are: B_q[j], B_q[N+j], B_q[2N+j], ...
//   Each address is N bytes apart. For N=1024, that's 1024 bytes between accesses.
//   A 16-byte vld1q_s8 load fetches 16 consecutive bytes — but we only USE byte [j]
//   from each cache line. The other 63 bytes fetched are thrown away.
//   COST: 16 separate cache-line fetches to fill one NEON register. This is
//   called a "gather" because we are gathering scattered elements into one register.
//
//   With packing, B is transposed into column-major layout:
//     B_packed[j * K + k] = B[k * N + j]   (j-th column stored as K contiguous bytes)
//   Now dot(A[i,:], B[:,j]) reads: B_packed[j*K+0], B_packed[j*K+1], ..., B_packed[j*K+K-1]
//   These are stride-1 sequential bytes → one vld1q_s8 fetches exactly 16 useful bytes.
//   COST: 1 cache-line fetch per 64 elements. 64× better cache utilization.
//
// ANALOGY TO BLIS "PACK PHASE":
//   BLIS (the BLAS-like library used internally by NumPy, MKL, etc.) has an explicit
//   "pack" phase before every GEMM: it copies panels of A and B into a contiguous
//   scratch buffer aligned to the micro-kernel's register width.
//   Our pack_b_int8() is exactly this: a one-time O(K*N) copy that amortizes across
//   many forward passes (B = weight matrix, same every call; A = activations, changes).
//
// ALIGNMENT:
//   B_packed must be 64-byte aligned so the first vld1q_s8 in every column hits
//   a cache-line boundary. Caller allocates with posix_memalign(64).
//   Required size: K * N * sizeof(int8_t), rounded up to 64 bytes.
// ---------------------------------------------------------------------------
void pack_b_int8(
    const int8_t* B,         // input: [K × N] row-major
    int8_t*       B_packed,  // output: [N × K] column-major, must be 64B-aligned, caller allocates
    size_t        N,
    size_t        K
);

// ---------------------------------------------------------------------------
// IMPLEMENTATION 3: neon_matmul_int8
//
// ARM NEON INT8 matmul using the DOT PRODUCT extension (available on M4).
//
// HOW INT8 QUANTIZATION ENABLES THIS:
//   FP32: 4 bytes/element → 4 elements per 16-byte NEON register
//   INT8: 1 byte/element → 16 elements per 16-byte NEON register
//   → 4× data density → 4× fewer cache misses for the same work
//
// THE DOT PRODUCT INSTRUCTION (vdotq_s32):
//   vdotq_s32(acc, a, b) → int32x4_t
//   Semantics: for each lane i in 0..3:
//     acc[i] += a[4i]*b[4i] + a[4i+1]*b[4i+1] + a[4i+2]*b[4i+2] + a[4i+3]*b[4i+3]
//   One instruction: 4 groups × (4 muls + 3 adds) = 28 ops on 32 input bytes.
//   M4 throughput: 2 per cycle → 56 effective INT8 ops/cycle/core.
//
// MICRO-KERNEL REGISTER BLOCKING (4×4 tile):
//   For a 4-row panel of A and 4-column panel of B_packed, we allocate:
//     4 int8x16_t for A rows  (a[0..3])
//     4 int8x16_t for B cols  (b[0..3])
//     16 int32x4_t accumulators (acc[i][j] for i,j in 0..3)
//   = 24 NEON registers total (fits within ARM64's 32 available).
//   Per K-step: 8 loads + 16 vdotq_s32 = 24 NEON instructions.
//   Arithmetic intensity: 4×4×16×2 = 512 ops / (8+4)×16 bytes = 2.67 ops/byte.
//
// PACK_B PARAMETER:
//   pack_b=true  (default): packs B into a temporary heap buffer internally.
//                           Safe for any call. Heap alloc is amortized cost.
//   pack_b=false: B_q is assumed to already be in packed column-major layout
//                 (as produced by pack_b_int8). Use this in inference loops
//                 where the same weight matrix is used for many forward passes.
//                 Eliminates the O(K*N) pack cost per call.
//
// Expected performance on M4 (1024×1024, INT8, packed):
//   ~30–50 GFLOP/s (4× gain over unpacked from eliminating gather penalty)
// ---------------------------------------------------------------------------
void neon_matmul_int8(
    const int8_t* A_q,          // quantized A [M × K], row-major
    const int8_t* B_q,          // quantized B: [K × N] row-major if pack_b=true,
                                //              [N × K] column-major if pack_b=false
    float*        C,            // output [M × N], dequantized float
    size_t M, size_t N, size_t K,
    float scale_A,              // per-tensor scale factor for A
    float scale_B,              // per-tensor scale factor for B
    bool  pack_b = true         // if false, B_q must be pre-packed via pack_b_int8()
);

// ---------------------------------------------------------------------------
// Helper: quantize a float array to INT8 with per-tensor scaling.
//
// scale = max(|x|) / 127.0
// x_q   = clamp(round(x / scale), -128, 127)
//
// Per-tensor (one scale for the whole matrix) is less accurate than per-channel
// (one scale per row/column) but simpler for demonstration. The quantize.h
// module will implement per-channel later.
// ---------------------------------------------------------------------------
void quantize_f32_to_i8(
    const float* src,
    int8_t*      dst,
    size_t       n_elements,
    float*       out_scale     // computed scale factor written here
);

void dequantize_i8_to_f32(
    const int8_t* src,
    float*        dst,
    size_t        n_elements,
    float         scale
);

} // namespace kernels
} // namespace tinyinfer
