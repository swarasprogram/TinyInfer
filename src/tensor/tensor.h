// =============================================================================
// tensor/tensor.h — Core tensor abstraction for tinyinfer
//
// WHAT THIS FILE DOES:
//   Defines the Tensor class: an N-dimensional array with a typed dtype, shape,
//   strides, and a 64-byte-aligned heap buffer. Supports lazy transpose/reshape
//   (stride manipulation only — zero data copies) and a templated data accessor.
//
// WHY IT IS DESIGNED THIS WAY:
//   Transformer inference has two memory access patterns:
//     1. Sequential (matmul rows/columns)     → wants cache lines filled front-to-back
//     2. Strided   (attention head selection) → wants zero-copy reinterpretation
//   Storing shape + strides separately lets us do both without extra allocations.
//   This is the same design used by PyTorch (aten::Tensor) and NumPy (ndarray).
//
// HARDWARE ASSUMPTIONS (Apple M4):
//   - L2 cache line = 64 bytes → aligning tensor data to 64 bytes ensures the
//     first element of any tensor lands at a cache line boundary, avoiding
//     false sharing and wasted prefetch bandwidth.
//   - NEON SIMD loads (vld1q_f32, vld1q_s8) require 16-byte alignment for
//     maximum throughput; 64-byte alignment satisfies this trivially.
//   - 64 bytes also matches AVX-512 alignment requirements, making this code
//     portable to x86 data-center hardware without changes.
//
// REFERENCES:
//   [1] ARM Cortex-A Programmer's Guide, §6.3 "Cache line size"
//   [2] PyTorch internal docs: https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/TensorImpl.h
//   [3] "What Every Programmer Should Know About Memory" — Ulrich Drepper, §3.2
// =============================================================================

#pragma once

#include <cstddef>       // size_t
#include <cstdint>       // int8_t, int32_t
#include <vector>
#include <memory>
#include <stdexcept>
#include <numeric>       // std::accumulate
#include <functional>    // std::multiplies
#include <cassert>

namespace tinyinfer {

// ---------------------------------------------------------------------------
// ALIGNMENT_BYTES: 64 bytes = one L2 cache line on ARM Cortex-A / Apple M4.
//
// Why 64 and not 16 (NEON minimum) or 128?
//   - 16 bytes: satisfies NEON, but a tensor starting at offset +16 within a
//     cache line means the first load pulls in 48 "wasted" bytes from the
//     previous cache line.
//   - 128 bytes: wastes up to 127 bytes of padding for small tensors.
//   - 64 bytes: exactly one cache line. The hardware prefetcher fetches in
//     64-byte chunks, so aligning to 64 means every prefetch is fully used.
// ---------------------------------------------------------------------------
inline constexpr size_t ALIGNMENT_BYTES = 64;

// ---------------------------------------------------------------------------
// Dtype — supported numeric formats.
//
// FLOAT32: 32-bit IEEE 754 float. Standard training/inference precision.
//          4 bytes per element → 4 elements per 16-byte NEON register.
//
// INT8: 8-bit signed integer. Post-quantization format.
//       1 byte per element → 16 elements per 16-byte NEON register.
//       4× data density means 4× less memory bandwidth for the same operation,
//       which is the primary speedup from quantization (M4 is memory-bandwidth-
//       bound at large matrix sizes, not compute-bound).
// ---------------------------------------------------------------------------
enum class Dtype : uint8_t {
    FLOAT32 = 0,
    INT8    = 1,
};

// Returns the size in bytes of a single element of the given dtype.
inline constexpr size_t dtype_size(Dtype dt) {
    switch (dt) {
        case Dtype::FLOAT32: return 4;
        case Dtype::INT8:    return 1;
        default:             return 0;
    }
}

inline const char* dtype_name(Dtype dt) {
    switch (dt) {
        case Dtype::FLOAT32: return "float32";
        case Dtype::INT8:    return "int8";
        default:             return "unknown";
    }
}

// ---------------------------------------------------------------------------
// Tensor — the central data structure of tinyinfer.
// ---------------------------------------------------------------------------
class Tensor {
public:
    // -----------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------

    // Default: empty tensor (null data, zero shape). Useful as a placeholder.
    Tensor() = default;

    // Allocate a new tensor with the given shape and dtype.
    // Memory is 64-byte aligned via posix_memalign.
    explicit Tensor(std::vector<size_t> shape, Dtype dtype = Dtype::FLOAT32);

    // Construct a tensor that is a *view* over existing memory (no allocation).
    // The caller is responsible for keeping the source memory alive.
    // Used for sub-tensors (attention heads) and mmap'd weight files.
    Tensor(std::vector<size_t> shape,
           std::vector<size_t> strides,
           Dtype dtype,
           void* external_data);

    ~Tensor();

    // Non-copyable (tensors can be large; force explicit clone() calls).
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // Movable.
    Tensor(Tensor&&) noexcept;
    Tensor& operator=(Tensor&&) noexcept;

    // -----------------------------------------------------------------------
    // Metadata accessors
    // -----------------------------------------------------------------------

    const std::vector<size_t>& shape()   const noexcept { return shape_; }
    const std::vector<size_t>& strides() const noexcept { return strides_; }
    Dtype                       dtype()  const noexcept { return dtype_; }
    size_t                      ndim()   const noexcept { return shape_.size(); }
    size_t                      dim(size_t i) const { return shape_.at(i); }

    // Total number of elements (product of all dimensions).
    size_t numel() const noexcept;

    // Total allocated bytes (numel * dtype_size, rounded up to ALIGNMENT_BYTES).
    size_t nbytes() const noexcept;

    bool is_contiguous() const noexcept;
    bool owns_data()     const noexcept { return owns_data_; }

    // -----------------------------------------------------------------------
    // Raw data access
    // -----------------------------------------------------------------------

    // Typed pointer to the start of data.
    // Template specializations enforce type safety at compile time.
    //
    // Usage:
    //   float*  p = tensor.data_ptr<float>();
    //   int8_t* q = tensor.data_ptr<int8_t>();
    //
    // Why not just return void*?
    //   Because casting void* in calling code is error-prone and loses the
    //   type information that lets the compiler generate correct NEON loads.
    template <typename T>
    T* data_ptr() {
        static_assert(sizeof(T) == dtype_size(dtype_for<T>()),
            "data_ptr<T>: T size does not match tensor dtype size. "
            "Use data_ptr<float>() for FLOAT32 or data_ptr<int8_t>() for INT8.");
        return reinterpret_cast<T*>(data_);
    }

    template <typename T>
    const T* data_ptr() const {
        return reinterpret_cast<const T*>(data_);
    }

    // -----------------------------------------------------------------------
    // Lazy reshape / transpose — ZERO DATA COPIES
    // -----------------------------------------------------------------------

    // Returns a new Tensor that is a VIEW with a different shape.
    // The underlying data pointer is shared. Modifying one modifies the other.
    // Requires: numel() unchanged and tensor is contiguous.
    //
    // How it works: compute new strides from new shape (C-contiguous order),
    // point to same data_ buffer, set owns_data_ = false in the view.
    Tensor view(std::vector<size_t> new_shape) const;

    // Returns a new Tensor with axes i and j swapped (lazy transpose).
    // No data is moved — only strides[i] and strides[j] are swapped.
    //
    // Example: a (B, H, T, D) tensor transposed on axes 2,3 becomes (B, H, D, T)
    // with the same memory layout — extremely fast for attention head reshaping.
    Tensor transpose(size_t axis_i, size_t axis_j) const;

    // Deep copy — allocates new memory and copies all elements.
    Tensor clone() const;

    // -----------------------------------------------------------------------
    // Element access (slow, for testing only — use data_ptr in kernels)
    // -----------------------------------------------------------------------
    float  at_f32(std::vector<size_t> idx) const;
    int8_t at_i8 (std::vector<size_t> idx) const;

    // Fill all elements with a scalar value.
    void fill_zero();
    void fill_f32(float value);

    // -----------------------------------------------------------------------
    // Debug / pretty print
    // -----------------------------------------------------------------------
    void print_info(const char* name = "") const;

private:
    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    // Compute C-contiguous strides from shape.
    // C-contiguous (row-major): stride[i] = product(shape[i+1..N-1])
    // This is the natural layout for row-by-row matrix access.
    static std::vector<size_t> compute_strides(const std::vector<size_t>& shape);

    // Compute linear offset from multi-index using strides.
    size_t linear_offset(const std::vector<size_t>& idx) const;

    // Map C++ type T to the corresponding Dtype enum value.
    // Used in static_assert inside data_ptr<T>.
    template <typename T> static constexpr Dtype dtype_for();

    // -----------------------------------------------------------------------
    // Data members
    // -----------------------------------------------------------------------

    void*               data_     = nullptr;   // 64-byte aligned raw pointer
    std::vector<size_t> shape_;                // [batch, seq, heads, dim, ...]
    std::vector<size_t> strides_;              // element strides (not byte strides)
    Dtype               dtype_    = Dtype::FLOAT32;
    bool                owns_data_ = false;    // true if we called posix_memalign
};

// ---------------------------------------------------------------------------
// Template specializations for dtype_for<T>()
// ---------------------------------------------------------------------------
template<> inline constexpr Dtype Tensor::dtype_for<float>()   { return Dtype::FLOAT32; }
template<> inline constexpr Dtype Tensor::dtype_for<int8_t>()  { return Dtype::INT8; }

} // namespace tinyinfer
