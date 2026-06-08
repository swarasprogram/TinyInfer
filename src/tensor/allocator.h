// =============================================================================
// tensor/allocator.h — NUMA-aware memory allocator for Apple Silicon
//
// WHAT THIS FILE DOES:
//   Provides a custom allocator that uses mmap() instead of malloc() for
//   large tensor allocations, with madvise() hints to pre-warm cache pages
//   before inference begins. Tracks memory usage statistics (total, peak,
//   fragmentation). Thread-local pools avoid mutex contention across cores.
//
// UNIFIED MEMORY ON APPLE SILICON — WHY THIS MATTERS:
//   Traditional GPU inference (e.g., NVIDIA A100):
//     CPU RAM ←──PCIe (16 GB/s)──→ GPU VRAM
//     Weights live in VRAM. Every forward pass: load activations from CPU,
//     wait for PCIe transfer, compute on GPU. PCIe is a hard bottleneck.
//
//   Apple M4 unified memory architecture:
//     CPU cores and GPU cores share the SAME physical DRAM pool.
//     A weight matrix allocated in CPU RAM IS already in "GPU memory".
//     There is no transfer cost — both CPU NEON and GPU Metal shaders can
//     read the same physical pages directly.
//
//   This allocator exploits that by:
//     1. Using mmap() with MAP_SHARED so pages can be directly mapped into
//        Metal command buffers in future GPU phases.
//     2. Calling madvise(MADV_WILLNEED) to trigger async page-in before
//        inference starts — eliminates page-fault latency during the first
//        forward pass (critical for interactive latency targets).
//     3. Calling madvise(MADV_SEQUENTIAL) on KV-cache regions to hint the
//        hardware prefetcher that access will be linear (ring buffer reads).
//
// NUMA ON APPLE M4:
//   M4 has two clusters of cores:
//     - 4 Efficiency cores (E-cores): lower frequency, smaller L1 cache (128KB)
//     - 6 Performance cores (P-cores): higher frequency, larger L1 cache (192KB)
//   The L2 cache is shared across the cluster but NOT across clusters.
//   Allocating KV-cache data on pages that are accessed by E-cores and weight
//   matrices on pages accessed by P-cores reduces cross-cluster cache pressure.
//   We approximate this with MADV hints since macOS doesn't expose full NUMA APIs.
//
// HARDWARE ASSUMPTIONS:
//   - macOS mmap with MAP_ANONYMOUS is always available.
//   - madvise() returns 0 on success; non-zero is advisory (not fatal).
//   - M4 has 16MB L2 cache; allocations >16MB benefit most from MADV_WILLNEED.
//
// REFERENCES:
//   [1] Apple Silicon Memory Model: https://developer.apple.com/documentation/metal/gpu_and_cpu_synchronization
//   [2] madvise(2) man page — Linux and Darwin implementations
//   [3] "NUMA-Aware Data Structures" — Lepers et al., ATC'15
// =============================================================================

#pragma once

#include <cstddef>
#include <cstdint>
#include <atomic>
#include <vector>
#include <mutex>

namespace tinyinfer {

// ---------------------------------------------------------------------------
// AllocationHint: tells the allocator what the memory will be used for.
// The allocator uses this to select appropriate madvise() flags.
// ---------------------------------------------------------------------------
enum class AllocationHint : uint8_t {
    WEIGHT,     // Weight matrices: compute-heavy, random access across rows
                // → MADV_WILLNEED (pre-warm) + MADV_RANDOM
    KV_CACHE,   // Key/Value cache: read-heavy, mostly sequential ring-buffer access
                // → MADV_WILLNEED + MADV_SEQUENTIAL
    ACTIVATION, // Intermediate activations: short-lived, written then read once
                // → MADV_SEQUENTIAL (no pre-warm; written before read)
    GENERAL,    // Default: no special hints
};

// ---------------------------------------------------------------------------
// AllocStats: memory usage tracking for profiling and diagnostics.
// ---------------------------------------------------------------------------
struct AllocStats {
    std::atomic<size_t> total_allocated_bytes{0};  // current live bytes
    std::atomic<size_t> peak_allocated_bytes{0};   // high-water mark
    std::atomic<size_t> total_allocation_count{0}; // number of active allocs
    std::atomic<size_t> total_mmap_calls{0};       // mmap() calls made

    // Fragmentation ratio: 1.0 = perfectly packed, >1.0 = wasted space.
    // Computed as (sum of mmap region sizes) / (sum of requested sizes).
    // We track this separately since mmap rounds to page size (4KB or 16KB on M4).
    std::atomic<size_t> total_requested_bytes{0};
    std::atomic<size_t> total_mapped_bytes{0};

    double fragmentation_ratio() const noexcept {
        const size_t req = total_requested_bytes.load();
        if (req == 0) return 1.0;
        return static_cast<double>(total_mapped_bytes.load()) /
               static_cast<double>(req);
    }

    void print() const;
};

// ---------------------------------------------------------------------------
// MmapAllocation: RAII wrapper around a single mmap'd region.
// Returned by TinyAllocator::allocate(); freed on destruction.
// ---------------------------------------------------------------------------
struct MmapAllocation {
    void*  ptr          = nullptr;
    size_t mapped_bytes = 0;   // actual mmap size (page-rounded)
    size_t request_bytes = 0;  // original requested size

    MmapAllocation() = default;
    MmapAllocation(void* p, size_t mapped, size_t req)
        : ptr(p), mapped_bytes(mapped), request_bytes(req) {}

    // Non-copyable, movable (same pattern as Tensor).
    MmapAllocation(const MmapAllocation&) = delete;
    MmapAllocation& operator=(const MmapAllocation&) = delete;
    MmapAllocation(MmapAllocation&&) noexcept;
    MmapAllocation& operator=(MmapAllocation&&) noexcept;

    ~MmapAllocation();
};

// ---------------------------------------------------------------------------
// TinyAllocator: the main allocator singleton.
//
// Usage:
//   auto& alloc = TinyAllocator::instance();
//   auto region = alloc.allocate(1024 * 1024, AllocationHint::WEIGHT);
//   float* p = reinterpret_cast<float*>(region.ptr);
//   // ... use p ...
//   // region is freed when it goes out of scope (RAII)
// ---------------------------------------------------------------------------
class TinyAllocator {
public:
    static TinyAllocator& instance();

    // Allocate 'bytes' of aligned memory with the given usage hint.
    // Returns an RAII MmapAllocation. Throws std::bad_alloc on failure.
    MmapAllocation allocate(size_t bytes, AllocationHint hint = AllocationHint::GENERAL);

    // Pre-warm a memory region: trigger async page-in via MADV_WILLNEED.
    // Call this during model loading, before the first forward pass.
    // On M4, this can eliminate 10–100ms of page-fault latency on first inference.
    static void prewarm(void* ptr, size_t bytes);

    // Hint that a memory region will be accessed sequentially (ring buffer, KV cache).
    // The hardware prefetcher can then prefetch full cache lines ahead of time.
    static void hint_sequential(void* ptr, size_t bytes);

    // Hint that a region will be accessed randomly (weight matrix row selection).
    static void hint_random(void* ptr, size_t bytes);

    // Release kernel page cache for a region (equivalent to madvise MADV_FREE).
    // Use after freeing a large activation buffer to return memory to the OS.
    static void release_pages(void* ptr, size_t bytes);

    const AllocStats& stats() const { return stats_; }

    // Print a summary of allocation statistics to stdout.
    void print_stats() const;

private:
    TinyAllocator() = default;

    // Page size on M4 is 16KB (for large allocations; 4KB for small).
    // We round up to 16KB to be safe — mmap always rounds to OS page size anyway.
    static constexpr size_t PAGE_SIZE = 16 * 1024;  // 16KB = M4 large page size

    // Minimum size to use mmap vs posix_memalign.
    // For small allocations, mmap overhead (syscall + page table update) exceeds
    // the benefit. Use posix_memalign for anything < 1MB.
    static constexpr size_t MMAP_THRESHOLD = 1 * 1024 * 1024;  // 1MB

    // Round bytes up to the nearest page boundary.
    static size_t round_to_page(size_t bytes) {
        return (bytes + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1);
    }

    AllocStats stats_;
    std::mutex mu_;  // guards stats_ updates (atomic ops make this rare)
};

} // namespace tinyinfer
