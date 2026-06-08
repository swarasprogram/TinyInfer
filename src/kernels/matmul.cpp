// =============================================================================
// kernels/matmul.cpp — Three matmul implementations for Apple M4
//
// WHAT THIS FILE DOES:
//   Implements naive, tiled, and NEON INT8 matmul, plus quantization helpers.
//
// KEY HARDWARE NUMBERS (Apple M4 P-core):
//   L1 data cache  : 192 KB, 8-way set-associative, 64B line
//   L2 cache       : 16 MB shared across cluster, 64B line
//   NEON FP32 peak : 2 * 4 floats * 3.4 GHz ≈ 27 GFLOP/s (per core, 2-wide FMA)
//   NEON INT8 peak : 2 * 16 int8s * 3.4 GHz ≈ 109 GOPS/s (per core, sdot)
//   Memory BW      : ~100 GB/s (unified LPDDR5X)
//
// REFERENCES:
//   [1] ARM Architecture Reference Manual: sdot, vmlal_s8 encoding
//   [2] "Anatomy of High-Performance GEMM" — van Zee & van de Geijn, TOMS 2015
//   [3] Flash Attention paper — Dao et al., NeurIPS 2022 (online softmax idea)
// =============================================================================

#include "kernels/matmul.h"

#include <cstring>    // memset
#include <cmath>      // fabsf, roundf
#include <algorithm>  // std::min, std::max, std::clamp
#include <cstdint>

// ARM NEON intrinsics header — available on all Apple Silicon targets.
// #include <arm_neon.h> is needed for vld1q_s8, sdot, vst1q_f32, etc.
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace tinyinfer {
namespace kernels {

// ---------------------------------------------------------------------------
// Quantization helpers (used by neon_matmul_int8 caller)
// ---------------------------------------------------------------------------

void quantize_f32_to_i8(
    const float* src,
    int8_t*      dst,
    size_t       n,
    float*       out_scale)
{
    // Find max absolute value (used as the range for scaling).
    float max_abs = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        const float abs_val = fabsf(src[i]);
        if (abs_val > max_abs) max_abs = abs_val;
    }

    // scale: maps the float range [-max_abs, max_abs] → INT8 range [-127, 127].
    // We use 127 (not 128) so the mapping is symmetric: -max_abs → -127, +max_abs → +127.
    // This avoids the asymmetry of two's complement where -128 has no positive counterpart.
    constexpr float INT8_MAX_SYMMETRIC = 127.0f;
    const float scale = (max_abs < 1e-9f) ? 1.0f : (max_abs / INT8_MAX_SYMMETRIC);
    *out_scale = scale;

    const float inv_scale = 1.0f / scale;
    for (size_t i = 0; i < n; ++i) {
        // round to nearest, clamp to [-127, 127].
        const float scaled = src[i] * inv_scale;
        const int32_t rounded = static_cast<int32_t>(scaled + (scaled >= 0.0f ? 0.5f : -0.5f));
        dst[i] = static_cast<int8_t>(
            std::clamp(rounded, static_cast<int32_t>(-127), static_cast<int32_t>(127)));
    }
}

void dequantize_i8_to_f32(
    const int8_t* src,
    float*        dst,
    size_t        n,
    float         scale)
{
    for (size_t i = 0; i < n; ++i) {
        dst[i] = static_cast<float>(src[i]) * scale;
    }
}

// ---------------------------------------------------------------------------
// IMPLEMENTATION 1: naive_matmul_f32
//
// i-j-k loop order: for each output element C[i][j], iterate over k.
//
// Cache behavior (why this is slow):
//   - A[i][k]: as k increases, we move right along row i → sequential → good
//   - B[k][j]: as k increases, j is fixed → we jump down a column of B
//              Each step skips N floats = N*4 bytes. For N=1024, that's 4096 bytes.
//              Each 64-byte cache line holds 16 floats. We only use 1 of them.
//              Cache miss rate ≈ 100% for B → every B access is a cache miss.
// ---------------------------------------------------------------------------
void naive_matmul_f32(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__       C,
    size_t M, size_t N, size_t K)
{
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ---------------------------------------------------------------------------
// IMPLEMENTATION 2: tiled_matmul_f32
//
// Cache-blocked in i-k-j order. Processes (tile×tile) submatrices that fit
// in L1 cache so they can be reused without re-loading from L2/DRAM.
//
// Tile loop structure (3 levels of blocking):
//   outer i-tile loop:  selects rows   of A and C
//   outer k-tile loop:  selects cols   of A and rows of B
//   outer j-tile loop:  selects cols   of B and C
//   inner loops: accumulate within the tile
//
// Cache occupancy analysis for tile_size=64 on M4 P-core:
//   Tile of A: 64×64 × 4B = 16 KB   ← fits in L1 (192KB)
//   Tile of B: 64×64 × 4B = 16 KB   ← fits in L1
//   Tile of C: 64×64 × 4B = 16 KB   ← fits in L1
//   Total: 48 KB < 192 KB → no L1 eviction within a tile computation ✓
// ---------------------------------------------------------------------------
void tiled_matmul_f32(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__       C,
    size_t M, size_t N, size_t K,
    size_t tile_size)
{
    // Zero the output. Caller may pass uninitialized C.
    memset(C, 0, M * N * sizeof(float));

    // Outer tile loops: step through A, B, C in tile_size chunks.
    for (size_t i0 = 0; i0 < M; i0 += tile_size) {
        // i_end: handle the partial tile at the last row block.
        const size_t i_end = std::min(i0 + tile_size, M);

        for (size_t k0 = 0; k0 < K; k0 += tile_size) {
            const size_t k_end = std::min(k0 + tile_size, K);

            for (size_t j0 = 0; j0 < N; j0 += tile_size) {
                const size_t j_end = std::min(j0 + tile_size, N);

                // Inner loops: compute C[i0:i_end][j0:j_end] +=
                //              A[i0:i_end][k0:k_end] × B[k0:k_end][j0:j_end]
                //
                // Inner loop order is i-k-j:
                //   A[i][k]: k changes slowly relative to the j loop → stays in register
                //   B[k][j]: j increases → sequential row access → cache friendly ✓
                //   C[i][j]: j increases → sequential row access → cache friendly ✓
                for (size_t i = i0; i < i_end; ++i) {
                    for (size_t k = k0; k < k_end; ++k) {
                        // Hoist A[i][k] out of j-loop: it's the same value for all j.
                        // Compiler should do this automatically with -O3, but being
                        // explicit helps readability and guarantees register allocation.
                        const float a_ik = A[i * K + k];
                        for (size_t j = j0; j < j_end; ++j) {
                            C[i * N + j] += a_ik * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// B-PACKING IMPLEMENTATION
//
// Memory layout transformation:
//   Input  B      [K × N] row-major:    B[k][n]      stored at B[k*N + n]
//   Output B_packed [N × K] col-major: B_packed[n][k] stored at B_packed[n*K + k]
//
// After packing, column j of B (i.e., B[:,j]) is the contiguous span:
//   B_packed[j*K + 0], B_packed[j*K + 1], ..., B_packed[j*K + K-1]
// This is stride-1 memory — a single vld1q_s8 fetches 16 consecutive useful bytes.
//
// Without packing, column j requires reads at:
//   B_q[0*N+j], B_q[1*N+j], ..., B_q[(K-1)*N+j]
// Stride between reads = N bytes. For N=1024: 1024 bytes apart. Each vld1q_s8
// touches a fresh 64-byte cache line but only uses 1 byte from it → 64× waste.
// This is the gather penalty: we pay for 64 bytes of cache traffic per useful byte.
// ---------------------------------------------------------------------------
void pack_b_int8(
    const int8_t* __restrict__ B,
    int8_t*       __restrict__ B_packed,
    size_t N,
    size_t K)
{
    // Transpose B[K×N] → B_packed[N×K].
    // Loop order: j (column) outer, k (row) inner, so writes to B_packed are
    // sequential (stride-1) while reads from B are strided — reads from a large
    // matrix are unavoidably strided during packing; we pay this cost once.
    //
    // An optimized packer would tile this loop to improve read locality too
    // (BLIS does this), but for K,N ≤ 4096 on M4 the L2 cache absorbs it.
    for (size_t j = 0; j < N; ++j) {
        for (size_t k = 0; k < K; ++k) {
            B_packed[j * K + k] = B[k * N + j];
        }
    }
}

// ---------------------------------------------------------------------------
// 4×4 REGISTER-TILED MICRO-KERNEL (packed B only)
//
// Computes C[i0:i0+M_rows][j0:j0+j_count] += A[i0:i0+M_rows][:] · B_packed[j0:j0+j_count][:]
//
// Register allocation for the full 4×4 case:
//   a[0..3]      : 4 × int8x16_t — 16 consecutive elements from rows of A
//   b[0..3]      : 4 × int8x16_t — 16 consecutive elements from columns of B_packed
//   acc[i][j]    : 4×4 = 16 × int32x4_t — partial dot-product accumulators
//   TOTAL        : 4 + 4 + 16 = 24 NEON registers (ARM64 has 32 available)
//
// Per k-step (16 elements):
//   8 vld1q_s8 loads  (A rows + B cols, all stride-1 → maximum cache efficiency)
//   16 vdotq_s32 ops  (full 4×4 tile of partial dot products)
//   = 24 NEON instructions for 4×4×16×2 = 512 equivalent FLOP
//   Arithmetic intensity: 512 / (8×16 bytes) = 4.0 ops/byte
//
// Compare to old gather kernel:
//   Per output element: 16 scalar gathers + 1 vdotq = 17 ops for 32 FLOP
//   Arithmetic intensity: 32 / (16×1 + 1×16) bytes = 1.0 ops/byte
// ---------------------------------------------------------------------------
#ifdef __ARM_NEON
static void micro_kernel_4x4(
    const int8_t* __restrict__ A_rows,   // A[i0][0], stride K between rows
    const int8_t* __restrict__ B_cols,   // B_packed[j0][0], stride K between cols
    float*        __restrict__ C,        // output row-major, stride N
    size_t M_rows,   // number of A rows in this panel: 1–4
    size_t j_count,  // number of B cols in this tile: 1–4
    size_t N,        // full output width (stride of C rows)
    size_t K,        // inner dimension (stride of A rows and B_packed cols)
    float  combined_scale)
{
    // 16 accumulator registers: acc[i][j] accumulates the partial dot product
    // between A-row i and B-col j over chunks of 16 K-elements.
    // vdotq_s32 produces 4 int32 lanes; we reduce them all at the end.
    int32x4_t acc[4][4];
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            acc[i][j] = vdupq_n_s32(0);

    // k_vec: largest multiple of 16 ≤ K. Elements [k_vec, K) handled in scalar tail.
    const size_t k_vec = (K / 16) * 16;

    for (size_t k = 0; k < k_vec; k += 16) {
        // --- LOAD PHASE ---
        // A rows: A_rows + i*K + k is stride-1 within each row → sequential loads.
        int8x16_t a[4];
        for (size_t i = 0; i < M_rows; ++i)
            a[i] = vld1q_s8(A_rows + i * K + k);

        // B cols: B_cols + j*K + k is stride-1 within each packed column → sequential loads.
        // THIS is the critical fix over the old kernel: no gather, pure sequential access.
        int8x16_t b[4];
        for (size_t j = 0; j < j_count; ++j)
            b[j] = vld1q_s8(B_cols + j * K + k);

        // --- COMPUTE PHASE: 4×4 = 16 dot products ---
        // vdotq_s32(acc, a, b): each of 4 lanes computes
        //   acc[lane] += a[4*lane+0]*b[4*lane+0] + ... + a[4*lane+3]*b[4*lane+3]
        // After the full k-loop, summing all 4 lanes gives the full dot product.
        for (size_t i = 0; i < M_rows; ++i)
            for (size_t j = 0; j < j_count; ++j)
                acc[i][j] = vdotq_s32(acc[i][j], a[i], b[j]);
    }

    // --- STORE + SCALAR TAIL PHASE ---
    // For each output element C[i0+i][j0+j]:
    //   1. Horizontally sum the 4 int32 lanes of acc[i][j]
    //   2. Add any remaining K elements not covered by the NEON loop (tail)
    //   3. Multiply by combined_scale and write to C
    for (size_t i = 0; i < M_rows; ++i) {
        for (size_t j = 0; j < j_count; ++j) {
            int32_t sum = vaddvq_s32(acc[i][j]);

            // Scalar tail: K elements in [k_vec, K)
            for (size_t k = k_vec; k < K; ++k) {
                sum += static_cast<int32_t>(A_rows[i * K + k]) *
                       static_cast<int32_t>(B_cols[j * K + k]);
            }

            C[i * N + j] = static_cast<float>(sum) * combined_scale;
        }
    }
}
#endif // __ARM_NEON

// ---------------------------------------------------------------------------
// IMPLEMENTATION 3: neon_matmul_int8 (with optional B packing)
//
// When pack_b=true (default):
//   Allocates a temporary K*N buffer, calls pack_b_int8(), then runs the
//   packed micro-kernel. The pack step is O(K*N) but happens only once per call.
//   Use this when B changes each call (unusual for inference — weights are fixed).
//
// When pack_b=false:
//   B_q must already be in packed column-major layout (from pack_b_int8()).
//   Zero overhead from packing. Use this in all inference hot paths where
//   B represents a static weight matrix packed once at model load time.
// ---------------------------------------------------------------------------
void neon_matmul_int8(
    const int8_t* __restrict__ A_q,
    const int8_t* __restrict__ B_q,
    float*        __restrict__ C,
    size_t M, size_t N, size_t K,
    float scale_A,
    float scale_B,
    bool  pack_b)
{
    const float combined_scale = scale_A * scale_B;

#ifdef __ARM_NEON

    // Conditionally pack B into column-major layout.
    void*         pack_buf    = nullptr;
    const int8_t* B_packed_ptr = B_q;

    if (pack_b) {
        // Allocate 64-byte aligned buffer for packed B.
        // Size: N * K bytes (same element count as B, different layout).
        // 64-byte alignment ensures the first vld1q_s8 of every packed column
        // starts on a cache-line boundary (assuming K is a multiple of 64).
        const size_t pack_bytes = ((N * K + 63) / 64) * 64;
        posix_memalign(&pack_buf, 64, pack_bytes);
        pack_b_int8(B_q, reinterpret_cast<int8_t*>(pack_buf), N, K);
        B_packed_ptr = reinterpret_cast<const int8_t*>(pack_buf);
    }

    // Outer loop: process A in panels of 4 rows at a time.
    // Processing 4 rows together keeps 4 A registers live across the j-loop,
    // allowing the compiler to hoist the A loads out of the j-tile loop.
    constexpr size_t ROW_PANEL = 4;
    constexpr size_t COL_TILE  = 4;

    for (size_t i0 = 0; i0 < M; i0 += ROW_PANEL) {
        const size_t M_rows = std::min(ROW_PANEL, M - i0);  // handle M % 4 != 0

        for (size_t j0 = 0; j0 < N; j0 += COL_TILE) {
            const size_t j_count = std::min(COL_TILE, N - j0);  // handle N % 4 != 0

            micro_kernel_4x4(
                A_q + i0 * K,            // start of A row panel
                B_packed_ptr + j0 * K,   // start of B col tile (packed layout)
                C + i0 * N + j0,         // output tile top-left corner
                M_rows, j_count,
                N, K,
                combined_scale
            );
        }
    }

    if (pack_buf) std::free(pack_buf);

#else
    // Scalar fallback: correct on non-NEON platforms, slow by design.
    // Respects pack_b=false: if B is already packed, use packed layout.
    if (pack_b) {
        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < N; ++j) {
                int32_t acc = 0;
                for (size_t k = 0; k < K; ++k)
                    acc += static_cast<int32_t>(A_q[i*K+k]) * static_cast<int32_t>(B_q[k*N+j]);
                C[i*N+j] = static_cast<float>(acc) * combined_scale;
            }
    } else {
        // B is already packed [N×K]
        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < N; ++j) {
                int32_t acc = 0;
                for (size_t k = 0; k < K; ++k)
                    acc += static_cast<int32_t>(A_q[i*K+k]) * static_cast<int32_t>(B_q[j*K+k]);
                C[i*N+j] = static_cast<float>(acc) * combined_scale;
            }
    }
#endif
}

} // namespace kernels
} // namespace tinyinfer
