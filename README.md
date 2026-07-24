# DDARX

**A CUDA virtual memory KV cache layer for paged attention systems**

Built by [Thunderfield]

DDARX is a systems level prototype that changes how the KV cache in paged attention is managed. Instead of routing every allocation and every cross request cache lookup through a software block table inside the attention kernel, DDARX moves memory allocation onto the GPU driver's own virtual memory system and moves cache sharing onto a lightweight host side index. The kernel is left to do nothing but math.

---

## Table of Contents

- [The Problem](#the-problem)
- [The DDARX Approach](#the-ddarx-approach)
- [Project Status](#project-status)
- [How It Works](#how-it-works)
- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [Running the Demo](#running-the-demo)
- [Running Tests](#running-tests)
- [Project Structure](#project-structure)
- [Roadmap](#roadmap)
- [License](#license)

---

## The Problem

Paged attention gave inference engines near zero VRAM waste by splitting the KV cache into fixed size blocks that can live anywhere in physical memory. To make that work, every attention call has to walk a **software block table** to translate a logical token position into a physical memory location. That block table does two jobs at once:

1. **Elastic allocation** — growing the KV cache for a sequence one block at a time without wasting VRAM.
2. **Content sharing** — letting multiple requests reuse the same physical blocks for prefix caching, beam search, and parallel sampling.

Bundling both jobs into one in kernel mechanism means every attention call pays for indirection, extra branching, and register pressure, even on requests that never share anything with another sequence.

## The DDARX Approach

DDARX splits those two jobs apart and gives each one the tool that is actually built for it:

- **Allocation → CUDA Virtual Memory Management.** Each sequence reserves one flat, contiguous virtual address range up front with `cuMemAddressReserve`. Physical VRAM is only committed when it's actually needed, using `cuMemCreate` and `cuMemMap`. The attention kernel never sees a block table, it just sees one contiguous tensor per sequence.
- **Sharing → a host side radix cache.** Prefix reuse, eviction, and copy on write are resolved entirely on the CPU by hashing token blocks into a radix tree. When a match is found, sharing a page across two sequences is just a pointer remap (`cuMemMap` against an existing physical handle) — it never touches the kernel.
- **Latency hiding → background mapping.** Because decode grows a sequence by exactly one token per step, the next physical page can be mapped on a background host thread while the GPU is still busy with the current step, so the mapping call is hidden instead of stalling the pipeline.

## Project Status

**Experimental prototype.** This repository currently contains a CUDA Driver API console application (`kvvmm-demo`) that exercises the full mechanism end to end: reserve two virtual sequences, commit and map a shared prefix, insert it into the radix cache, look it up from a second sequence, map a decode page in the background, perform a copy on write, and verify every write landed in the right physical page through flat virtual addresses.

What this project **is**:
- A working, verifiable implementation of the VMM allocator, the radix cache, and the background mapper described above.

What this project **is not yet**:
- Wired into an actual transformer attention kernel or an inference framework.
- Benchmarked head to head against a production paged attention allocator.

Treat the efficiency argument as a validated mechanism with an unproven end to end performance claim, not a finished serving engine.

## How It Works

| Component | File | Responsibility |
|---|---|---|
| `DeviceContext` / `ContextGuard` / `DriverStream` | `cuda_driver.hpp/.cpp` | Thin RAII wrappers around the CUDA Driver API: context, device, and stream lifetime, plus a `KVVMM_CU_CHECK` macro for error handling. |
| `VmmAllocator` | `vmm_allocator.hpp/.cpp` | Reserves virtual address ranges, creates and maps physical pages on demand, and performs copy on write. |
| `VirtualSequence` | `vmm_allocator.hpp/.cpp` | Represents one sequence's contiguous virtual address range and its current page mappings. |
| `PhysicalPage` | `vmm_allocator.hpp/.cpp` | A reference counted handle to a single committed physical allocation (`CUmemGenericAllocationHandle`). |
| `BackgroundMapper` | `vmm_allocator.hpp/.cpp` | Issues the next page mapping asynchronously on a background thread so it overlaps with GPU compute. |
| `RadixCache<T>` | `radix_cache.hpp/.cpp` | A token block radix tree that maps prefix signatures to shared physical pages, with weak pointer based eviction. |
| `DriverKernelModule` | `driver_kernels.hpp/.cpp` | Loads a small embedded PTX module (`cuModuleLoadDataEx`) and launches fill and patch kernels against flat virtual addresses. |

The project deliberately stays on the **CUDA Driver API**. It never includes `cuda_runtime.h` and never links `cudart` — no `cudaMalloc`, `cudaMemcpy`, or `cudaLaunchKernel` anywhere in the codebase. Everything goes through `cuMem*` and `cuLaunchKernel`.

## Requirements

| Requirement | Minimum |
|---|---|
| GPU | NVIDIA, compute capability **6.0 (Pascal)** or newer — VMM (`cuMemCreate` / `cuMemMap`) is not available on older architectures |
| NVIDIA driver | A driver build that reports `CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED` (R450 or newer is a safe baseline) |
| CUDA Toolkit | **11.0 or newer** (for `cuda.h` headers and the `CUDA::cuda_driver` import library; toolkit 12.x recommended) |
| Compiler | A C++20 compiler — MSVC (Visual Studio 2022) on Windows, GCC 11+ or Clang 14+ on Linux |
| CMake | **3.24 or newer** |
| OS | Windows or Linux. CUDA is not available on macOS, so it is not supported. |

## Getting Started

Clone the repository and make sure `nvcc`/the CUDA Toolkit is on your path so CMake's `find_package(CUDAToolkit REQUIRED)` can locate it.

### Windows (PowerShell)

```powershell
git clone https://github.com/Thunderfield/DDARX.git
cd DDARX

# Configure, build, and run the unit tests
.\scripts\build.ps1
```

`scripts/build.ps1` accepts a couple of optional parameters:

```powershell
# Build a Debug configuration
.\scripts\build.ps1 -Config Debug

# Skip building and running the unit tests
.\scripts\build.ps1 -NoTests
```

### Linux (bash)

```bash
git clone https://github.com/Thunderfield/DDARX.git
cd DDARX

cmake -S . -B build -DKVVMM_BUILD_TESTS=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

## Running the Demo

The demo executable, `kvvmm-demo`, reserves two virtual sequences, shares a prefix between them through the radix cache, performs a copy on write, and prints a verification summary.

**Windows:**

```powershell
.\scripts\run-demo.ps1 -Device 0 -PrefixBlocks 4 -ReserveMiB 256
```

**Linux, or calling the binary directly on Windows:**

```bash
./build/kvvmm-demo --device 0 --prefix-blocks 4 --reserve-mib 256
```

### CLI Options

| Flag | Default | Description |
|---|---|---|
| `--device N` | `0` | CUDA device ordinal to run on. |
| `--reserve-mib N` | `256` | Virtual address space reserved per sequence, in MiB. |
| `--page-kib N` | `0` | Requested physical page size in KiB. `0` uses the driver's minimum VMM granularity. |
| `--prefix-blocks N` | `4` | Number of shared prefix pages to create and reuse between the two demo sequences. |
| `--quiet` | off | Print only the final verification line. |
| `--help` | — | Show usage and exit. |

A successful run ends with:

```
Verification OK: shared prefix pages were reused, request-B COW stayed private, and all GPU writes used flat virtual addresses.
```

## Running Tests

The unit tests cover the radix cache in isolation (no GPU required):

```bash
ctest --test-dir build -C Release --output-on-failure
```

or run the compiled test binary directly:

```bash
./build/kvvmm-radix-tests
```

## Project Structure

```
DDARX/
├── CMakeLists.txt
├── include/kvvmm/
│   ├── cuda_driver.hpp        # Driver API RAII wrappers and error checking
│   ├── driver_kernels.hpp     # Embedded PTX module loader and launcher
│   ├── radix_cache.hpp        # Token block radix tree for prefix sharing
│   └── vmm_allocator.hpp      # Virtual memory reservation, mapping, COW
├── src/                       # Implementations of the above
├── tests/
│   └── radix_cache_tests.cpp  # CPU only unit tests for the radix cache
└── scripts/
    ├── build.ps1               # Configure, build, and test (Windows)
    └── run-demo.ps1             # Run the demo executable (Windows)
```

## Roadmap

- [ ] Bind `VmmAllocator` and `RadixCache` into an actual attention kernel (FlashAttention-3 or FlashInfer style) instead of the synthetic fill/patch PTX kernels used in the demo.
- [ ] Add a Python/PyTorch binding so the allocator can sit underneath a real model runner.
- [ ] Head to head throughput and memory fragmentation benchmarks against a standard block table paged attention allocator, across both prefill heavy and decode heavy workloads.
- [ ] Tunable page granularity presets for short/bursty workloads versus long context workloads.
- [ ] Linux focused build scripts to match the existing PowerShell scripts.

Contributions and issue reports are welcome — please open an issue before submitting large structural changes.

## License

No license has been added to this repository yet. Until one is added, all rights are reserved by Thunderfield.
