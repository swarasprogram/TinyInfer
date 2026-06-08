# TinyInfer

A research-grade CPU LLM inference engine for Apple Silicon, written in C++20.
Built from scratch to explore **Unified Memory Aware Inference** — a set of techniques
that exploit Apple M-series' shared CPU/GPU memory architecture for zero-copy transformer
inference on commodity hardware.

This is not a reimplementation of llama.cpp. The design goals are different:
every subsystem exists to demonstrate a specific, measurable hardware optimization,
with the code and comments written to be read by engineers who want to understand *why*
it works, not just *that* it works.

---

## The Novel Contribution

On discrete GPU systems (NVIDIA A100, etc.), weights live in GPU VRAM and activations
in CPU RAM. Every forward pass pays a PCIe transfer tax (~16 GB/s, bidirectional).

On Apple M4, the CPU and GPU share the same physical DRAM pool. A weight matrix loaded
into RAM **is already in GPU memory**. There is no transfer. tinyinfer exploits this
with three original ideas:

### 1. NUMA-Aware Tensor Placement
M4 has two core clusters: 4 efficiency cores (E-cores, 128KB L1) and 6 performance
cores (P-cores, 192KB L1). The L2 cache is cluster-local. tinyinfer's custom allocator
(`src/tensor/allocator.h`) uses `mmap()` + `madvise()` to:
- Pre-warm weight pages with `MADV_WILLNEED` at model load time (eliminates 10–100ms of
  page-fault latency on first inference)
- Hint sequential access patterns for KV-cache ring buffers (`MADV_SEQUENTIAL`)
- Return pages to the OS immediately after short-lived activations (`MADV_FREE`)

All allocations are `MAP_PRIVATE | MAP_ANONYMOUS` — compatible with Metal GPU command
buffers for a future zero-copy CPU↔GPU pipeline.

### 2. Speculative Tiling with Runtime Tile-Size Selection
Standard inference engines use a fixed tile size (64 or 128) chosen at compile time.
That's wrong: the optimal size depends on L1 cache set-associativity, which varies
across M1 (12MB L2), M2 (16MB L2), M3 (24MB L2), and M4 (16MB L2, different topology).

tinyinfer's `AutoTuner` (`src/tuner/autotuner.h`) runs a 200ms micro-benchmark on first
launch, tests 6 candidate tile sizes (32/48/64/80/96/128), selects the fastest by median
GFLOP/s, and caches the result in `~/.tinyinfer/tuning_cache.json`. Subsequent launches
load instantly. The cache invalidates automatically when hardware changes.

On this M4 Pro, the tuner selected **tile=128** — exactly what the L1 arithmetic
predicts (3 × 128² × 4B = 192KB, filling the P-core L1 precisely).

### 3. Fused Attention Kernel with In-Register Softmax *(Phase 2)*
Standard attention: compute QK^T → store to DRAM → softmax → load back → multiply V.
tinyinfer fuses QK^T and softmax into one pass using the "online softmax" algorithm
(Dao et al., Flash Attention, NeurIPS 2022), implemented by hand in ARM NEON intrinsics.
The running maximum and denominator never leave NEON registers between steps — eliminating
2 full memory round-trips per attention head on M4's unified memory bus.

---
