# APM Brahma — Modular First-Hit Search with Early Stop

Refactored, class-based CUDA implementation of the APM (Almost Principal Minors) exhaustive search.

## Quick Start

```bash
# Build
make

# Run (default: groups 25–50)
./apm_brahma

# Run specific groups
./apm_brahma 32 35

# Run with log file
make run_log ARGS="25 27"
```

## Architecture

| File | Purpose |
|------|---------|
| `apm_types.hpp` | Shared data structures and constants |
| `mod_arith.cuh` | Device-side modular arithmetic (safe for primes up to 2^50) |
| `file_utils.hpp/cpp` | Directory, file collection, timer, combinatorics |
| `matrix_parser.hpp/cpp` | Parse Sage/Python matrix format |
| `prime_loader.hpp/cpp` | Load primes from files (25–50) with hardcoded fallbacks |
| `gpu_buffers.cuh` | GPU memory manager — auto-detects VRAM at runtime |
| `apm_search.cuh/cu` | CUDA kernel + first-hit search + group-deviation processing |
| `result_writer.hpp/cpp` | Write per-matrix results, group results, summaries |
| `main.cu` | Argument parsing, GPU detection, orchestration |

## Key Improvements over Original

1. **Modular design**: Each responsibility in its own class/file
2. **50-bit prime support**: Safe modular multiply using binary method
3. **Auto-detect GPU**: Makefile queries `nvidia-smi` for compute capability — no hardcoded SM
4. **Auto-detect VRAM**: Uses `cudaMemGetInfo()` instead of hardcoded 1500 MB limit
5. **Extended range**: Supports prime groups 25–50 (original: 25–35)
6. **No external dependencies**: Only CUDA runtime and standard C++ libraries

## How It Works

For each prime group `g` in the specified range:
1. Load the prime for group `g` from files (fallback to hardcoded for 25–35)
2. Collect all matrix files from `kernel_output/<g>/`
3. For each deviation `d = 2, 3, ..., n-3`:
   - For each matrix: search for the **first** zero minor (first-hit, then stop)
   - Count how many matrices had at least one hit
   - If 100 matrices hit → **early stop**, skip remaining deviations
4. Write `result.txt` for the group with the best deviation

## Output Structure

```
Results_brahma/
  <group>/
    result.txt                     # Best deviation summary
    deviation_<d>/
      <matrix>_result.txt          # Per-matrix result
      SUMMARY_detailed.txt         # Detailed summary with zero minors
      SUMMARY_brief.txt            # One-page summary
```

## Requirements

- CUDA Toolkit (any version ≥ 10.0)
- C++14 compatible compiler (g++, gcc-10, gcc-12)
- GPU with CUDA support (any compute capability)
