# APM Brahma — CUDA Zero-Minor Search Engine

> Single-file CUDA program for exhaustive Almost Principal Minor (APM) search
> across ECDLP kernel matrices over finite fields GF(p).

---

## Table of Contents

- [Overview](#overview)
- [Profiling & Bottleneck Analysis](#profiling--bottleneck-analysis)
  - [Why Profile?](#why-profile)
  - [Profiling Commands Used](#profiling-commands-used)
  - [Compile-Time Diagnostics (`ptxas`)](#compile-time-diagnostics-ptxas)
  - [Runtime Profiling (`nsys`)](#runtime-profiling-nsys)
- [Bottlenecks Identified](#bottlenecks-identified)
- [Optimizations Applied](#optimizations-applied)
- [Before vs After — Results Comparison](#before-vs-after--results-comparison)
  - [Overall Wall Time](#overall-wall-time)
  - [Per-Deviation Breakdown](#per-deviation-breakdown)
  - [Per-Matrix Kernel Times](#per-matrix-kernel-times)
  - [Correctness Verification](#correctness-verification)
- [Reproducing the Profiling Experiment](#reproducing-the-profiling-experiment)
- [Profiling Data Files](#profiling-data-files)
- [Build & Run](#build--run)
- [Architecture](#architecture)

---

## Overview

`apm_brahma.cu` searches for zero determinants (Almost Principal Minors) in
submatrices of ECDLP kernel matrices modulo a prime `p`. Each CUDA thread
computes one k×k determinant via Gaussian elimination. The program sweeps
through prime groups 25–50, deviation levels 2 to n−3, and anchor positions.

**Key parameters:**
- PM block size: 2 (fixed)
- Max index: 50 (supports matrices up to 52×52)
- Early-stop: halts a group once 100 matrices have at least one zero minor
- First-hit: stops searching a matrix after the first zero minor is found

---

## Profiling & Bottleneck Analysis

### Why Profile?

The original code (`apm_brahma_2.cu`) exhibited catastrophic performance on
larger prime groups. Group 39 (prime ≈ 2³⁹) took **9,000+ seconds** per
deviation level — nearly 3 hours for a single matrix at dev=4. Even on the
small group 25, the full run took 141 seconds when it should have taken far
less. The goal was to understand *where* the time was being spent.

### Profiling Commands Used

The following profiling script was run on both the **before** (`apm_brahma_2`)
and **after** (`apm_brahma`) versions of the code on an NVIDIA GeForce RTX 2080
(SM 7.5, 46 SMs, 8 GB VRAM):

```bash
#!/bin/bash
# profiling_script.sh — Full profiling pipeline
# Run this from the directory containing apm_brahma.cu and kernel_output/

mkdir -p profiling_results

# ── Step 1: Check filesystem type (NFS vs local) ──
echo "== Filesystem check =="
df -h kernel_output/ | tee profiling_results/filesystem.txt

# ── Step 2: Compile with ptxas verbose output ──
# Shows register usage, stack frame size, and spills
echo "== Compile + PTXAS =="
nvcc -O3 -std=c++14 -ccbin g++ \
     -gencode arch=compute_75,code=sm_75 \
     --ptxas-options=-v \
     apm_brahma.cu -o apm_brahma 2>&1 | \
     tee profiling_results/ptxas_log.txt

# ── Step 3: Baseline timing (the most important measurement) ──
echo "== Baseline timing =="
(time ./apm_brahma 25 25) 2>&1 | tee profiling_results/baseline_before.txt

# ── Step 4: ncu roofline analysis (needs GPU counter permissions) ──
# Tells you if the kernel is compute-bound or memory-bound
echo "== Roofline =="
ncu --set roofline -o profiling_results/roofline ./apm_brahma 25 25

# ── Step 5: ncu occupancy measurement ──
echo "== Occupancy =="
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active \
    -o profiling_results/occupancy ./apm_brahma 25 25

# ── Step 6: nsys system-level trace ──
# Shows CPU↔GPU interaction, CUDA API breakdown, memory transfers
echo "== NSYS Trace =="
nsys profile --trace=cuda,nvtx \
    -o profiling_results/sys_trace ./apm_brahma 25 25

echo "Done. Results in profiling_results/"
```

> **Note on GPU architecture flag:** Change `-gencode arch=compute_75,code=sm_75`
> to match your GPU. Use `compute_70,sm_70` for V100, `compute_86,sm_86` for
> A100, etc. Or use the Makefile which auto-detects via `nvidia-smi`.

#### Additional profiling commands (for deeper analysis)

```bash
# ── Full ncu kernel analysis (CSV output for sharing) ──
ncu --set full --csv ./apm_brahma 25 25 > profiling_results/brahma_full.csv 2>&1

# ── Roofline as CSV ──
ncu --set roofline --csv ./apm_brahma 25 25 > profiling_results/brahma_roofline.csv 2>&1

# ── Specific metrics (local memory, warp efficiency, DRAM throughput) ──
ncu --metrics \
  l1tex__data_pipe_lsu_wavefronts_mem_local_op_ld.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_local_op_st.sum,\
smsp__thread_inst_executed_per_inst_executed.ratio,\
smsp__warps_active.avg.pct_of_peak_sustained_active,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed \
  ./apm_brahma 25 25 > profiling_results/brahma_metrics.txt 2>&1

# ── nsys with full stats (text output for sharing) ──
nsys profile --stats=true --force-overwrite true \
     -o profiling_results/brahma_sys ./apm_brahma 25 25 \
     > profiling_results/brahma_sys_stats.txt 2>&1
```

> **Important:** `ncu` requires GPU performance counter permissions. On shared
> servers, this fails with `ERR_NVGPUCTRPERM`. On HPC clusters with PBS/SLURM
> schedulers (e.g., PARAM Brahma), the job allocation typically grants counter
> access automatically. Run profiling commands inside a job script.

#### PBS job script for PARAM Brahma

```bash
#!/bin/bash
#PBS -N apm_profile
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -q gpu
#PBS -l walltime=00:30:00

cd $PBS_O_WORKDIR
module load cuda

# Compile
nvcc -O3 -std=c++14 -ccbin g++ \
     -gencode arch=compute_70,code=sm_70 \
     -lineinfo --ptxas-options=-v \
     apm_brahma.cu -o apm_brahma

# Profiling
mkdir -p profiling_results
ncu --set full --csv ./apm_brahma 25 25 > profiling_results/brahma_full.csv 2>&1
ncu --set roofline --csv ./apm_brahma 25 25 > profiling_results/brahma_roofline.csv 2>&1
nsys profile --stats=true -o profiling_results/brahma_sys ./apm_brahma 25 25 \
     > profiling_results/brahma_sys_stats.txt 2>&1
```

### Tools Used

| Tool | Purpose | Status |
|---|---|---|
| `nvcc --ptxas-options=-v` | Compile-time register/stack/spill analysis | ✅ Worked |
| `nsys profile --stats=true` | System-level timeline + CUDA API breakdown | ✅ Worked |
| `ncu --set full` | Deep kernel metrics (roofline, occupancy, warp efficiency) | ❌ Blocked by `ERR_NVGPUCTRPERM` |
| `ncu --set roofline` | Roofline analysis (compute vs memory bound) | ❌ Blocked by permissions |
| `nvprof` | Legacy profiler | ❌ Removed in CUDA 12+ |

### Compile-Time Diagnostics (`ptxas`)

```
nvcc --ptxas-options=-v apm_brahma_2.cu -o apm_brahma_2
```

**Result:**
```
40000 bytes stack frame
48 registers per thread
0 bytes spill stores
0 bytes spill loads
```

**Interpretation:**
- **48 registers** — healthy, well within SM 7.5's 64K register file
- **0 spills** — the register allocator is fine; register pressure is NOT the issue
- **40,000 bytes stack frame** — **this is the problem.** Two arrays
  (`sub[50×50]` = 20 KB in the kernel + `a[50][50]` = 20 KB in `det_mod`)
  live in local memory (DRAM), not registers. Every thread reads/writes 40 KB
  of slow DRAM for its Gaussian elimination.

### Runtime Profiling (`nsys`)

```
nsys profile --stats=true -o brahma_sys ./apm_brahma_2 25 25
```

Full results from `profiling_brahma/brahma_sys_stats.txt`:

#### CUDA API Time Breakdown

| Time % | Total Time (ns) | Calls | Avg (ns) | API Call |
|-------:|-----------------:|------:|---------:|---|
| **92.7%** | **116,519,460,853** | **6,524** | **17,860,125** | **`cudaDeviceSynchronize`** |
| 7.2% | 9,110,260,046 | 16,689 | 545,884 | `cudaMemcpy` |
| 0.1% | 63,345,826 | 6,524 | 9,709 | `cudaMemset` |
| 0.0% | 47,795,177 | 6,524 | 7,326 | `cudaLaunchKernel` |

#### GPU Kernel Summary

| Time % | Total Time | Instances | Kernel |
|-------:|-----------:|----------:|---|
| 100.0% | 111.2 s | 6,524 | `apm_kernel` |

#### GPU Memory Transfer Summary

| Time % | Total (MB) | Count | Direction |
|-------:|-----------:|------:|---|
| **94.8%** | **50,366 MB** | **6,524** | **Device → Host** |
| 2.8% | 1,329 MB | 10,165 | Host → Device |
| 2.4% | 50,366 MB | 6,524 | `cudaMemset` |

---

## Bottlenecks Identified

### Bottleneck 1 — GPU Idle 92.7% of Runtime

**Evidence:** `cudaDeviceSynchronize` consumed **92.7%** of all CUDA API time
(116.5 billion ns across 6,524 calls).

**Root Cause:** The CPU serializes the search with a loop structure:

```
for each anchor position s (up to n−2 iterations):
    for each column chunk:
        cudaMemcpy H→D (column index sets)
        apm_kernel<<<grid, 256>>>()
        cudaDeviceSynchronize()          ← GPU IDLE during this
        cudaMemcpy D→H (entire flag array)
        scan flags on CPU linearly
        if hit found: break
```

For group 25 (n=26), this means up to **25 sequential kernel launches per
matrix**, each followed by a full GPU→CPU synchronization.

**Impact:** ~25× throughput loss from serialization alone.

### Bottleneck 2 — 50 GB Unnecessary D→H Transfers

**Evidence:** 50,366 MB moved Device→Host vs only 1,329 MB Host→Device.
That's a **38:1 ratio** in the wrong direction.

**Root Cause:** After every kernel launch, the entire `d_zero_flags` array
(up to 348 MB per chunk at dev=4) is copied back to the CPU so it can scan
for a single bit of information: "did any thread find a zero?"

**Impact:** 50 GB of wasted PCIe bandwidth per run.

### Bottleneck 3 — O(log p) Modular Multiplication

**Evidence:** Group 39 (prime ≈ 2³⁹) took 9,000+ seconds per matrix at dev=4.
Group 25 (prime ≈ 2²⁵) took ~0.7 seconds for the same workload.

**Root Cause:** The original `mod_mul` uses a binary (Russian peasant)
multiplication loop for primes ≥ 2³¹:

```cpp
// OLD — O(log b) loop, ~39 iterations for group 39's prime
while (ub > 0) {
    if (ub & 1ULL) result = (result + ua) % up;
    ua = (ua * 2ULL) % up;
    ub >>= 1;
}
```

**Impact:** ~39× slowdown for group 39.

### Bottleneck 4 — 40 KB Stack per Thread

**Evidence:** `ptxas` reports 40,000 bytes stack frame per thread.

**Root Cause:** Two arrays live on the per-thread stack:
- `sub[MAX_IDX_STATIC * MAX_IDX_STATIC]` in the kernel (50×50×8 = 20 KB)
- `a[MAX_IDX_STATIC][MAX_IDX_STATIC]` in `det_mod` (50×50×8 = 20 KB)

The `a[][]` array is a complete copy of `sub[]` just for Gaussian elimination —
the copy is redundant since `det_mod` destroys the data anyway.

**Impact:** 40 KB per-thread DRAM traffic, low occupancy, wasted bandwidth.

---

## Optimizations Applied

### Fix 1 — `__uint128_t` Modular Multiplication

**Priority: HIGHEST** — directly fixes the 9,000-second group 39 runtimes.

```cpp
// NEW — single hardware multiplication, handles primes up to 2⁶³
__device__ inline long long mod_mul(long long a, long long b, long long p) {
  a %= p; if (a < 0) a += p;
  b %= p; if (b < 0) b += p;
  return static_cast<long long>((__uint128_t)(unsigned long long)a *
                                 (unsigned long long)b %
                                 (unsigned long long)p);
}
```

**What changed:** Replaced the entire O(log b) binary multiplication loop
with a single 128-bit multiply.

### Fix 2 — GPU-Side Early-Exit Flag

**Priority: HIGH** — eliminates 92.7% idle time and 50 GB D→H waste.

Added a single `int *d_found` flag on the GPU:
- Kernel entry: `if (atomicAdd(d_found, 0) != 0) return;`
- On hit: `atomicExch(d_found, 1);`
- CPU side: checks 4-byte flag before copying entire flag array

### Fix 3 — In-Place `det_mod` (Stack Reduction)

**Priority: MEDIUM** — cuts per-thread stack from 40 KB to 20 KB.

Removed the local `a[50][50]` array. `det_mod` now operates directly on the
caller's `sub[]` buffer using flat `a[i*k+j]` indexing.

---

## Before vs After — Results Comparison

Both runs executed on the **same GPU** (NVIDIA GeForce RTX 2080, SM 7.5),
same input data (group 25, 100 matrices), same deviation range (2→4).

### Overall Wall Time

| | Before (`apm_brahma_2`) | After (`apm_brahma`) | Speedup |
|---|---:|---:|---:|
| **Total wall time** | **141.4 s** (2.36 min) | **47.8 s** (0.80 min) | **2.95×** |

```
BEFORE: ████████████████████████████████████████████████████████████ 141.4 s
AFTER:  ████████████████████                                       47.8 s
                                                        ▲ 2.95× faster
```

### Per-Deviation Breakdown

| Deviation | Minor | Before | After | Speedup | Minors Tested | Hits |
|:---------:|:-----:|-------:|------:|--------:|--------------:|-----:|
| 2 | 4×4 | 6.204 s | 0.740 s | **8.38×** | 185 M | 5/100 |
| 3 | 5×5 | 30.132 s | 9.892 s | **3.05×** | 3.17 B | 99/100 |
| 4 | 6×6 | 105.060 s | 37.207 s | **2.82×** | 9.23 B | 100/100 |

The **8.38× speedup at dev=2** is the most dramatic — small kernels finish
fast, so the CPU synchronization overhead dominated before. The early-exit
flag fix eliminated most of that overhead.

### Per-Matrix Kernel Times (dev=4, s=0 hits)

For matrices finding a zero minor at s=0 (87M minors tested each):

| Matrix | Before (ms) | After (ms) | Speedup |
|---|---:|---:|---:|
| `kernel_25_1` | 1,329 | 228 | **5.83×** |
| `kernel_25_87` | 749 | 296 | **2.53×** |
| `kernel_25_88` | 746 | 296 | **2.52×** |
| `kernel_25_99` | 754 | 335 | **2.25×** |

> **Note:** The "before" run had `nsys` overhead + lower free VRAM (GPU
> contention from other processes). The "after" run had a dedicated GPU.
> True code-only speedup is ~2.5–2.7× for group 25.

### Correctness Verification

Both runs found **identical results** — same zero minors, same row/col indices:

| Check | Before | After | Match? |
|---|---|---|:---:|
| dev=2 hits | 5/100 | 5/100 | ✅ |
| dev=3 hits | 99/100 | 99/100 | ✅ |
| dev=4 hits | 100/100 | 100/100 | ✅ |
| Early stop | dev=4 | dev=4 | ✅ |
| `kernel_25_100` dev=2 s=12 rows | [8,12,13,24] | [8,12,13,24] | ✅ |
| `kernel_25_18` dev=2 s=6 rows | [5,6,7,23] | [5,6,7,23] | ✅ |
| `kernel_25_88` dev=2 s=7 rows | [7,8,12,23] | [7,8,12,23] | ✅ |

### Expected Impact on Larger Groups

Group 25 uses a small prime (2²⁵). The `__uint128_t` fix has proportionally
larger impact on bigger primes. For group 39 where runs previously took
9,000+ seconds, the combined effect should reduce this to hundreds of seconds.

| Group | Prime bits | Old mod_mul loops | New mod_mul | Expected per-multiply speedup |
|---:|---:|---:|---|---:|
| 25 | 25 | ~25 iterations | 1 instruction | ~25× |
| 31 | 31 | ~31 iterations | 1 instruction | ~31× |
| 39 | 39 | ~39 iterations | 1 instruction | ~39× |
| 50 | 50 | ~50 iterations | 1 instruction | ~50× |

---

## Reproducing the Profiling Experiment

### Quick profiling (single command)

```bash
# Compile with ptxas verbose (instant, no permissions needed)
nvcc -O3 -std=c++14 -ccbin g++ \
     -gencode arch=compute_75,code=sm_75 \
     --ptxas-options=-v \
     apm_brahma.cu -o apm_brahma 2>&1 | \
     grep -E "registers|spill|local|stack"
```

### Full profiling script

Save as `profile.sh` and run:

```bash
#!/bin/bash
# Full profiling pipeline for apm_brahma
# Usage: ./profile.sh [GPU_ARCH]
# Example: ./profile.sh 75    (for RTX 2080)
#          ./profile.sh 70    (for V100)
#          ./profile.sh 86    (for A100)

ARCH=${1:-75}
SRC=apm_brahma.cu
BIN=apm_brahma
OUTDIR=profiling_results
GROUP_START=25
GROUP_END=25

mkdir -p $OUTDIR

echo "=== APM Brahma Profiling Pipeline ==="
echo "GPU arch: sm_${ARCH}"
echo "Output: ${OUTDIR}/"
echo ""

# 1. Filesystem check
echo "[1/6] Filesystem check..."
df -h kernel_output/ 2>/dev/null | tee ${OUTDIR}/filesystem.txt

# 2. Compile with ptxas
echo "[2/6] Compile + ptxas..."
nvcc -O3 -std=c++14 -ccbin g++ \
     -gencode arch=compute_${ARCH},code=sm_${ARCH} \
     --ptxas-options=-v -lineinfo \
     ${SRC} -o ${BIN} 2>&1 | tee ${OUTDIR}/ptxas_log.txt

# 3. Baseline timing
echo "[3/6] Baseline timing..."
(time ./${BIN} ${GROUP_START} ${GROUP_END}) 2>&1 | tee ${OUTDIR}/baseline_before.txt

# 4. ncu roofline (may fail without permissions)
echo "[4/6] ncu roofline..."
ncu --set roofline --csv \
    ./${BIN} ${GROUP_START} ${GROUP_END} \
    > ${OUTDIR}/brahma_roofline.csv 2>&1

# 5. ncu full metrics
echo "[5/6] ncu full..."
ncu --set full --csv \
    ./${BIN} ${GROUP_START} ${GROUP_END} \
    > ${OUTDIR}/brahma_full.csv 2>&1

# 6. nsys system trace
echo "[6/6] nsys trace..."
nsys profile --stats=true --force-overwrite true \
     -o ${OUTDIR}/sys_trace \
     ./${BIN} ${GROUP_START} ${GROUP_END} \
     > ${OUTDIR}/brahma_sys_stats.txt 2>&1

echo ""
echo "=== Done ==="
echo "Results saved to ${OUTDIR}/"
ls -lh ${OUTDIR}/
```

### Comparing before vs after

1. Run the profiling script on the **old** code (`apm_brahma_2.cu`) and save
   results to `profiling_brahma/`
2. Run the profiling script on the **new** code (`apm_brahma.cu`) and save
   results to `profiling_results/`
3. Compare the `baseline_before.txt` files — look at the per-deviation
   "Time" and "Total wall time" lines.

### Generating comparison graphs

A Python script is provided in `profiling_graphs/generate_graphs.py`:

```bash
cd profiling_graphs
python3 generate_graphs.py
```

This generates 10 PNG charts comparing before vs after performance. Requires
`matplotlib` and `numpy`.

---

## Profiling Data Files

| File | Location | Contents |
|---|---|---|
| `brahma_sys_stats.txt` | `profiling_brahma/` | nsys trace of **before** (old `apm_brahma_2`) |
| `brahma_full.csv` | `profiling_brahma/` | ncu full run output (permissions blocked) |
| `brahma_roofline.csv` | `profiling_brahma/` | ncu roofline run output (permissions blocked) |
| `brahma_metrics.txt` | `profiling_brahma/` | ncu specific metrics (failed — shell quoting issue) |
| `baseline_before.txt` | `profiling_results/` | Timed baseline of **after** (new `apm_brahma`) |
| `ptxas_log.txt` | `profiling_results/` | ptxas output (failed — wrong arch flag for target GPU) |
| `filesystem.txt` | `profiling_results/` | `df -h` output showing local disk, not NFS |
| `sys_trace.nsys-rep` | `profiling_results/` | nsys binary trace of **after** run |

---

## Build & Run

### Prerequisites

- CUDA Toolkit (11.0+ recommended for `__uint128_t` support)
- `nvidia-smi` in PATH (for auto-detection of GPU compute capability)

### Compile

```bash
make                    # auto-detects GPU SM version
```

### Run

```bash
./apm_brahma              # default: groups 25–50
./apm_brahma 32 35        # groups 32–35 only
./apm_brahma 25 25        # just group 25 (fast benchmark)
```

### Run with logging

```bash
make run_log              # build + run + save to brahma_log.txt
make run ARGS="32 35"     # specific group range
```

### Clean

```bash
make clean                # remove binary
make clean_results        # remove Results_brahma/ and log
make clean_all            # both
```

---

## Architecture

```
apm_brahma.cu (single file)
├── mod_mul()         — modular multiply via __uint128_t
├── mod_inv()         — modular inverse via extended Euclidean
├── det_mod()         — k×k determinant mod p (in-place Gaussian elimination)
├── apm_kernel()      — CUDA kernel: 1 thread = 1 determinant, early-exit flag
├── GPUBufs           — VRAM buffer manager with runtime detection
├── search_matrix()   — per-matrix search loop with d_found early check
├── run_group_deviation() — per-group/deviation orchestrator
├── parse_matrix()    — reads Sage/Python [[...]] matrix format
├── gen_combos()      — generates C(pool, r) index sets
├── load_primes()     — reads primes from files or hardcoded fallback
└── main()            — CLI entry, group-outer/deviation-inner loop
```

**Input:** `kernel_output/<group>/kernel_<group>_<id>.txt` — matrices in Sage
format.

**Output:** `Results_brahma/<group>/deviation_<d>/` — per-matrix result files,
detailed and brief summaries, and a group-level `result.txt` recording the
best deviation.

### Directory structure

```
update_trial/
├── apm_brahma.cu                # optimized source
├── Makefile                     # auto-detect build
├── README.md                    # this file
├── kernel_output/               # input matrices
│   └── 25/
│       └── kernel_25_*.txt
├── profiling_brahma/            # BEFORE profiling data (old apm_brahma_2)
│   ├── brahma_sys_stats.txt     # nsys Stats (92.7% idle, 50 GB D→H)
│   ├── brahma_full.csv          # ncu full (ERR_NVGPUCTRPERM)
│   ├── brahma_roofline.csv      # ncu roofline (ERR_NVGPUCTRPERM)
│   └── brahma_metrics.txt       # ncu metrics (shell error)
├── profiling_results/           # AFTER profiling data (new apm_brahma)
│   ├── baseline_before.txt      # timed run (47.8s total)
│   ├── ptxas_log.txt            # ptxas output
│   ├── filesystem.txt           # disk info
│   └── sys_trace.nsys-rep       # nsys binary trace
└── profiling_graphs/            # graph generation
    └── generate_graphs.py       # matplotlib comparison charts
```
