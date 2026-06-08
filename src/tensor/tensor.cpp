// =============================================================================
// tensor/tensor.cpp — Tensor implementation
//
// WHAT THIS FILE DOES:
//   Implements the Tensor class: allocation, stride computation, lazy reshape,
//   lazy transpose, element access, and debug printing.
//
// MEMORY LAYOUT PRIMER (for readers new to stride-based tensors):
//
//   A 2D matrix M[rows][cols] stored row-major in memory looks like:
//
//     M[0][0], M[0][1], ..., M[0][cols-1],   ← row 0
//     M[1][0], M[1][1], ..., M[1][cols-1],   ← row 1
//     ...
//
//   The address of M[i][j] = base + (i * cols + j) * sizeof(float)
//
//   In stride notation: stride[0] = cols, stride[1] = 1
//   So M[i][j] = base + i*stride[0] + j*stride[1]
//
//   Transposing (swapping rows and cols) in stride notation:
//     Just swap stride[0] and stride[1] — the DATA stays in the same place.
//   This is O(1) regardless of matrix size. This is the power of strides.
//
// HARDWARE ASSUMPTIONS (Apple M4):
//   - posix_memalign(64) aligns to one ARM cache line.
//   - free() is safe on posix_memalign pointers (POSIX guarantees this).
//
// REFERENCES:
//   [1] "What Every Programmer Should Know About Memory" §3.2 — Drepper (2007)
//   [2] NumPy internal array protocol: https://numpy.org/doc/stable/reference/arrays.interface.html
// =============================================================================

#include "tensor/tensor.h"

#include <cstdlib>     // posix_memalign, free
#include <cstring>     // memset, memcpy
#include <stdexcept>
#include <sstream>
#include <iostream>
#include <cmath>       // std::abs

namespace tinyinfer {

// ---------------------------------------------------------------------------
// Static helpers
// ---------------------------------------------------------------------------

std::vector<size_t> Tensor::compute_strides(const std::vector<size_t>& shape) {
    // C-contiguous (row-major) strides.
    // stride[i] = product of all dimensions after i.
    // Example: shape = [2, 3, 4] → strides = [12, 4, 1]
    //   Element [i][j][k] is at offset i*12 + j*4 + k*1.
    const size_t ndim = shape.size();
    std::vector<size_t> strides(ndim, 1);
    for (int i = static_cast<int>(ndim) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

size_t Tensor::linear_offset(const std::vector<size_t>& idx) const {
    assert(idx.size() == shape_.size() && "index dimensionality mismatch");
    size_t offset = 0;
    for (size_t i = 0; i < idx.size(); ++i) {
        assert(idx[i] < shape_[i] && "index out of bounds");
        offset += idx[i] * strides_[i];
    }
    return offset;
}

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

Tensor::Tensor(std::vector<size_t> shape, Dtype dtype)
    : shape_(std::move(shape))
    , strides_(compute_strides(shape_))
    , dtype_(dtype)
    , owns_data_(true)
{
    const size_t n_bytes = nbytes();
    if (n_bytes == 0) {
        data_ = nullptr;
        return;
    }

    // posix_memalign: standard POSIX function for aligned allocation.
    // Arguments: (void** ptr, size_t alignment, size_t size)
    // Alignment must be a power of 2 and a multiple of sizeof(void*).
    // 64 satisfies both: 64 = 2^6, and sizeof(void*) = 8 on ARM64.
    void* raw = nullptr;
    const int ret = posix_memalign(&raw, ALIGNMENT_BYTES, n_bytes);
    if (ret != 0 || raw == nullptr) {
        throw std::bad_alloc();
    }

    // Zero-initialize so unwritten elements don't cause undefined behavior
    // in partial-tile matmul operations.
    std::memset(raw, 0, n_bytes);
    data_ = raw;
}

Tensor::Tensor(std::vector<size_t> shape,
               std::vector<size_t> strides,
               Dtype dtype,
               void* external_data)
    : data_(external_data)
    , shape_(std::move(shape))
    , strides_(std::move(strides))
    , dtype_(dtype)
    , owns_data_(false)  // caller owns memory — we must NOT free it
{}

Tensor::~Tensor() {
    if (owns_data_ && data_ != nullptr) {
        // posix_memalign memory is freed with the standard free().
        std::free(data_);
        data_ = nullptr;
    }
}

Tensor::Tensor(Tensor&& other) noexcept
    : data_(other.data_)
    , shape_(std::move(other.shape_))
    , strides_(std::move(other.strides_))
    , dtype_(other.dtype_)
    , owns_data_(other.owns_data_)
{
    other.data_      = nullptr;
    other.owns_data_ = false;   // prevent double-free
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this == &other) return *this;
    if (owns_data_ && data_) std::free(data_);

    data_      = other.data_;
    shape_     = std::move(other.shape_);
    strides_   = std::move(other.strides_);
    dtype_     = other.dtype_;
    owns_data_ = other.owns_data_;

    other.data_      = nullptr;
    other.owns_data_ = false;
    return *this;
}

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------

size_t Tensor::numel() const noexcept {
    if (shape_.empty()) return 0;
    return std::accumulate(shape_.begin(), shape_.end(),
                           size_t{1}, std::multiplies<size_t>{});
}

size_t Tensor::nbytes() const noexcept {
    // Round up to nearest multiple of ALIGNMENT_BYTES so the buffer can
    // always be loaded in whole cache lines without reading past the end.
    const size_t raw = numel() * dtype_size(dtype_);
    return (raw + ALIGNMENT_BYTES - 1) & ~(ALIGNMENT_BYTES - 1);
}

bool Tensor::is_contiguous() const noexcept {
    // A tensor is C-contiguous if its strides equal what compute_strides
    // would return for the current shape.
    const auto expected = compute_strides(shape_);
    return strides_ == expected;
}

// ---------------------------------------------------------------------------
// Lazy reshape (view)
// ---------------------------------------------------------------------------

Tensor Tensor::view(std::vector<size_t> new_shape) const {
    // Guard: element count must be preserved.
    const size_t new_numel = std::accumulate(
        new_shape.begin(), new_shape.end(), size_t{1}, std::multiplies<size_t>{});
    if (new_numel != numel()) {
        throw std::invalid_argument(
            "Tensor::view: new shape has different numel than original");
    }
    if (!is_contiguous()) {
        throw std::invalid_argument(
            "Tensor::view: tensor must be contiguous for reshape. "
            "Call clone() first to make a contiguous copy.");
    }
    // Share the same data pointer — no allocation, no copy.
    return Tensor(std::move(new_shape), compute_strides(new_shape), dtype_, data_);
}

// ---------------------------------------------------------------------------
// Lazy transpose
// ---------------------------------------------------------------------------

Tensor Tensor::transpose(size_t axis_i, size_t axis_j) const {
    if (axis_i >= ndim() || axis_j >= ndim()) {
        throw std::out_of_range("Tensor::transpose: axis out of range");
    }
    // Copy shape and strides, then swap the two axes.
    auto new_shape   = shape_;
    auto new_strides = strides_;
    std::swap(new_shape[axis_i],   new_shape[axis_j]);
    std::swap(new_strides[axis_i], new_strides[axis_j]);

    // Key insight: after swapping strides, element [i][j] in the transposed
    // tensor maps to the same physical memory location as [j][i] in the
    // original tensor. No data moved. Cost: O(1).
    return Tensor(std::move(new_shape), std::move(new_strides), dtype_, data_);
}

// ---------------------------------------------------------------------------
// Deep copy
// ---------------------------------------------------------------------------

Tensor Tensor::clone() const {
    Tensor out(shape_, dtype_);
    // If contiguous, a single memcpy suffices.
    if (is_contiguous()) {
        std::memcpy(out.data_, data_, numel() * dtype_size(dtype_));
    } else {
        // For strided (non-contiguous) tensors, copy element by element.
        // This is slow but correct; in practice, call clone() only when needed.
        if (dtype_ == Dtype::FLOAT32) {
            const float* src = reinterpret_cast<const float*>(data_);
            float*       dst = reinterpret_cast<float*>(out.data_);
            // Iterate in flat output order, computing source offset via strides.
            // Simple implementation for N<=4 dims (sufficient for transformers).
            const size_t n = numel();
            for (size_t flat = 0; flat < n; ++flat) {
                // Decompose flat index into per-dim indices.
                std::vector<size_t> idx(ndim());
                size_t rem = flat;
                for (int d = static_cast<int>(ndim()) - 1; d >= 0; --d) {
                    idx[d] = rem % shape_[d];
                    rem   /= shape_[d];
                }
                dst[flat] = src[linear_offset(idx)];
            }
        }
        // INT8 path omitted for brevity — add symmetrically if needed.
    }
    return out;
}

// ---------------------------------------------------------------------------
// Element access (slow path — testing only)
// ---------------------------------------------------------------------------

float Tensor::at_f32(std::vector<size_t> idx) const {
    assert(dtype_ == Dtype::FLOAT32 && "at_f32 called on non-FLOAT32 tensor");
    const float* p = reinterpret_cast<const float*>(data_);
    return p[linear_offset(idx)];
}

int8_t Tensor::at_i8(std::vector<size_t> idx) const {
    assert(dtype_ == Dtype::INT8 && "at_i8 called on non-INT8 tensor");
    const int8_t* p = reinterpret_cast<const int8_t*>(data_);
    return p[linear_offset(idx)];
}

// ---------------------------------------------------------------------------
// Fill helpers
// ---------------------------------------------------------------------------

void Tensor::fill_zero() {
    if (data_) std::memset(data_, 0, nbytes());
}

void Tensor::fill_f32(float value) {
    assert(dtype_ == Dtype::FLOAT32);
    float* p = reinterpret_cast<float*>(data_);
    const size_t n = numel();
    for (size_t i = 0; i < n; ++i) p[i] = value;
}

// ---------------------------------------------------------------------------
// Debug print
// ---------------------------------------------------------------------------

void Tensor::print_info(const char* name) const {
    std::cout << "[Tensor";
    if (name && name[0]) std::cout << " '" << name << "'";
    std::cout << "]\n";
    std::cout << "  dtype  : " << dtype_name(dtype_) << "\n";
    std::cout << "  shape  : [";
    for (size_t i = 0; i < shape_.size(); ++i) {
        std::cout << shape_[i];
        if (i + 1 < shape_.size()) std::cout << ", ";
    }
    std::cout << "]\n";
    std::cout << "  strides: [";
    for (size_t i = 0; i < strides_.size(); ++i) {
        std::cout << strides_[i];
        if (i + 1 < strides_.size()) std::cout << ", ";
    }
    std::cout << "]\n";
    std::cout << "  numel  : " << numel() << "\n";
    std::cout << "  nbytes : " << nbytes() << " (aligned to "
              << ALIGNMENT_BYTES << "B)\n";
    std::cout << "  data   : " << data_ << "\n";
    std::cout << "  owns   : " << (owns_data_ ? "yes" : "no (view)") << "\n";
    std::cout << "  contiguous: " << (is_contiguous() ? "yes" : "no") << "\n";
}

} // namespace tinyinfer
