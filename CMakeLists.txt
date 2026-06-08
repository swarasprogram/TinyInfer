# =============================================================================
# CMakeLists.txt — tinyinfer build system
#
# What this does:
#   Configures a multi-target C++20 project for Apple M4 (ARM64) CPU inference.
#   Three build targets: tinyinfer_lib (static), bench_*, tests/*.
#
# Hardware assumptions:
#   - Apple M4: ARMv8.2-a + FP16 half-precision + DOT PRODUCT extension
#   - DOT PRODUCT (vdot / vmlal_s8 family) gives 4× throughput on INT8 MATMUL
#   - -march=armv8.2-a+fp16+dotprod tells the compiler to emit these instructions
#   - LTO (link-time optimization) allows cross-TU inlining of hot kernel paths
#
# Why LLVM clang, not Apple clang?
#   Apple's clang (from Xcode) lags LLVM by ~6 months on OpenMP runtime support
#   and some ARM intrinsic headers. LLVM clang from Homebrew gives us full
#   <arm_neon.h> intrinsic coverage and proper -fopenmp linking.
#
# References:
#   ARM Architecture Reference Manual ARMv8, for ARMv8-A profile (ARM DDI 0487)
#   LLVM 17 release notes: https://releases.llvm.org/17.0.0/docs/ReleaseNotes.html
# =============================================================================

cmake_minimum_required(VERSION 3.20)

project(tinyinfer
    VERSION 0.1.0
    DESCRIPTION "Unified-Memory-Aware LLM Inference Engine for Apple Silicon"
    LANGUAGES CXX
)

# ---------------------------------------------------------------------------
# Toolchain: prefer Homebrew LLVM clang over Apple clang
# ---------------------------------------------------------------------------
# We set this BEFORE project() in real toolchain files, but setting the path
# explicitly here ensures reproducibility even without a toolchain file.
# If the user runs cmake with -DCMAKE_CXX_COMPILER=/path/to/clang++, that wins.
if(NOT CMAKE_CXX_COMPILER)
    find_program(LLVM_CLANGPP
        NAMES clang++
        PATHS
            /opt/homebrew/opt/llvm/bin   # Apple Silicon Homebrew default
            /usr/local/opt/llvm/bin      # Intel Mac Homebrew default
        NO_DEFAULT_PATH
    )
    if(LLVM_CLANGPP)
        set(CMAKE_CXX_COMPILER "${LLVM_CLANGPP}" CACHE STRING "C++ compiler" FORCE)
        message(STATUS "tinyinfer: using Homebrew LLVM clang++ at ${LLVM_CLANGPP}")
    else()
        message(WARNING "tinyinfer: Homebrew LLVM not found, falling back to system clang++. "
                        "Run: brew install llvm")
    endif()
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)       # -std=c++20, not -std=gnu++20

# ---------------------------------------------------------------------------
# Build type default
# ---------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "tinyinfer: build type = ${CMAKE_BUILD_TYPE}")

# ---------------------------------------------------------------------------
# Architecture flags — M4 specific
# ---------------------------------------------------------------------------
# armv8.2-a  : baseline for M-series (M1 onwards)
# +fp16      : native half-precision arithmetic in NEON (vfma_f16 family)
# +dotprod   : INT8 dot-product instructions (vdot, sdot) — 4× INT8 MATMUL throughput
#
# -O3        : full optimization including auto-vectorization
# -ffast-math: allows reassociation and fused multiply-add; safe for ML workloads
#              where last-ULP accuracy isn't required
# -ffp-contract=fast: emit fused multiply-add (FMADD) wherever possible;
#              on NEON each FMADD is free vs a separate FMUL+FADD
set(TINYINFER_ARCH_FLAGS
    -march=armv8.2-a+fp16+dotprod
    -O3
    -ffast-math
    -ffp-contract=fast
)

# ---------------------------------------------------------------------------
# OpenMP — for multi-core parallelism in matmul outer loops
# ---------------------------------------------------------------------------
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "tinyinfer: OpenMP ${OpenMP_CXX_VERSION} found")
else()
    message(WARNING "tinyinfer: OpenMP not found. Parallel matmul will be single-threaded. "
                    "Install with: brew install libomp")
endif()

# ---------------------------------------------------------------------------
# Link-Time Optimization (LTO / IPO)
# ---------------------------------------------------------------------------
# LTO merges all translation units at link time, allowing the linker to inline
# hot paths across .cpp boundaries — e.g., the tensor accessor into the matmul loop.
# This typically gives 5–15% additional throughput on kernel-heavy code.
include(CheckIPOSupported)
check_ipo_supported(RESULT LTO_SUPPORTED OUTPUT LTO_ERROR)
if(LTO_SUPPORTED)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    message(STATUS "tinyinfer: LTO enabled")
else()
    message(WARNING "tinyinfer: LTO not supported: ${LTO_ERROR}")
endif()

# ---------------------------------------------------------------------------
# Warning flags — strict but practical
# ---------------------------------------------------------------------------
set(TINYINFER_WARN_FLAGS
    -Wall
    -Wextra
    -Wpedantic
    -Wno-unused-parameter   # common in stub implementations during development
    -Wno-missing-field-initializers
)

# ---------------------------------------------------------------------------
# Core library: tinyinfer_lib
# ---------------------------------------------------------------------------
# All inference logic lives here as a static library.
# Benchmarks and tests link against it — no code duplication.
add_library(tinyinfer_lib STATIC
    src/tensor/tensor.cpp
    src/tensor/allocator.cpp
    src/kernels/matmul.cpp
    src/tuner/autotuner.cpp
)

target_include_directories(tinyinfer_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_options(tinyinfer_lib PRIVATE
    ${TINYINFER_ARCH_FLAGS}
    ${TINYINFER_WARN_FLAGS}
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(tinyinfer_lib PUBLIC OpenMP::OpenMP_CXX)
endif()

# ---------------------------------------------------------------------------
# Benchmark: bench_matmul
# ---------------------------------------------------------------------------
add_executable(bench_matmul bench/bench_matmul.cpp)
target_link_libraries(bench_matmul PRIVATE tinyinfer_lib)
target_compile_options(bench_matmul PRIVATE ${TINYINFER_ARCH_FLAGS} ${TINYINFER_WARN_FLAGS})
target_include_directories(bench_matmul PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# ---------------------------------------------------------------------------
# Tests: correctness
# ---------------------------------------------------------------------------
add_executable(correctness tests/correctness.cpp)
target_link_libraries(correctness PRIVATE tinyinfer_lib)
target_compile_options(correctness PRIVATE ${TINYINFER_ARCH_FLAGS} ${TINYINFER_WARN_FLAGS})
target_include_directories(correctness PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Enable CTest integration so `ctest` works
enable_testing()
add_test(NAME correctness COMMAND correctness)

# ---------------------------------------------------------------------------
# Installation paths (optional, for packaging)
# ---------------------------------------------------------------------------
include(GNUInstallDirs)
install(TARGETS tinyinfer_lib
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY src/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tinyinfer
    FILES_MATCHING PATTERN "*.h"
)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
message(STATUS "")
message(STATUS "══════════════════════════════════════════")
message(STATUS "  tinyinfer ${PROJECT_VERSION} configuration")
message(STATUS "  Compiler : ${CMAKE_CXX_COMPILER}")
message(STATUS "  Build    : ${CMAKE_BUILD_TYPE}")
message(STATUS "  Arch     : armv8.2-a+fp16+dotprod")
message(STATUS "  LTO      : ${LTO_SUPPORTED}")
message(STATUS "  OpenMP   : ${OpenMP_CXX_FOUND}")
message(STATUS "══════════════════════════════════════════")
message(STATUS "")
