# Zero Minor Problem in Cryptography

> **Research project:** GPU-accelerated exhaustive search for zero minors in Arithmetic Progression in Primes (APM) matrices — empirically establishing a structural boundary in the zero-minor deviation threshold across prime bit-sizes 25–41.

## Supervisor
- **Dr. Abdulla Ansari**

## Description
The goal of this project is to research and implement an algorithm that can effectively solve or contribute to solving the Zero Minor Problem, which has implications in cryptographic systems and matrix theory.

The core engine is named **APM Brahma** — a CUDA-based search system that exhaustively tests zero-determinant submatrices (zero minors) across hundreds of independently generated APM matrices for primes of increasing bit-size, building empirical evidence for a structural mathematical boundary.

---

## Table of Contents

1. [Mathematical Background](#1-mathematical-background)
2. [Key Finding](#2-key-finding)
3. [Repository Structure](#3-repository-structure)
4. [Matrix Generation — `lasvegas-ecdlp`](#4-matrix-generation--lasvegas-ecdlp)
5. [CUDA Search Engine — `optimized_further`](#5-cuda-search-engine--optimized_further)
6. [Build & Run](#6-build--run)
7. [Multi-GPU Parallel Execution](#7-multi-gpu-parallel-execution)
8. [Results Layout & How to Interpret Them](#8-results-layout--how-to-interpret-them)
9. [Analysis Pipeline — `Analysis_Scripts`](#9-analysis-pipeline--analysis_scripts)
10. [Summary Data — APM 2.0 & APM 3.5](#10-summary-data--apm-20--apm-35)
11. [Profiling Results](#11-profiling-results)
12. [Utility Scripts](#12-utility-scripts)
13. [Development History](#13-development-history)
14. [Hardware Tested](#14-hardware-tested)
15. [Reproducing the Results](#15-reproducing-the-results)

---

## 1. Mathematical Background

An **APM matrix** of size `n×n` is generated from a prime `p` of bit-size `g` (i.e. `2^(g-1) < p < 2^g`). The matrix encodes arithmetic-progression structure in the prime field `Z_p`.

A **zero minor** of size `k×k` at deviation `d = k − 2` is a submatrix whose determinant is **zero mod p**. The central question is:

> *For a given prime bit-size `g`, what is the smallest deviation `d` such that **every** independently generated APM matrix contains at least one zero minor of size `(d+2)×(d+2)`?*

The search works as follows:
- For each prime group `g`, 100 independently generated matrices are tested.
- For each matrix, all `C(n − 2, d) × C(n − 2, d)` candidate submatrices (anchored by a fixed 2×2 principal block at position `s`) are tested for zero determinant mod `p`.
- The CUDA kernel stops as soon as one zero minor is found in a matrix (early-stop).
- A **hit** means at least one zero minor was found; **hit ratio = hits / 100**.

---

## 2. Key Finding

| Min deviation for 100% hit rate | Prime bit groups |
|:---:|:---|
| **deviation = 4** (6×6 minor) | Groups 25–32 (`p` up to ~4.3 × 10⁹) |
| **deviation = 5** (7×7 minor) | Groups 33–40 (`p` up to ~1.1 × 10¹²) |

The threshold **shifts at exactly group 33**, suggesting a structural boundary around the `2^33` prime range. For any matrix from a prime of bit-size 25–32, a zero 6×6 minor is **always** findable. For groups 33–40, the guaranteed size is a 7×7 minor.

This has been **empirically tested up to group 41** and the architecture theoretically supports up to **group 60** (primes near `2^60`).

> **Note on stray hits:** Groups 25, 26, and 29 showed a small number of hits (8, 2, 1 respectively) at deviation 2. These are not mathematically meaningful at ~1–8% hit rate and represent corner-case matrices, not a general guarantee.

---

## 3. Repository Structure

```
zero_minor_problem/
│
├── lasvegas-ecdlp/              ← Matrix generator (C++ / MPI / NTL)
│   └── apm/                    ← APM-specific kernel generation code
│       ├── main.cpp
│       ├── include/             ← EC_lasVegas, makeKernelDatabase templates
│       ├── kernel_output/       ← Generated kernel files (25–40)
│       ├── kernel_output_25_40/ ← Archived kernel sets
│       └── README_kernel_generation_changes.md
│
├── optimized_further/           ← ★ LATEST CUDA search engine
│   ├── apm_brahma.cu            ← Main CUDA kernel (1617 lines)
│   ├── Makefile                 ← Auto-detects GPU SM, one binary
│   ├── submit_parallel.sh       ← Multi-GPU job launcher
│   ├── check_results.py         ← Validate result completeness
│   ├── merge_summaries.py       ← Merge fragmented parallel summaries
│   ├── sort_kernel_files.py     ← Numerically sort input kernel files
│   └── kernel_output/           ← Input matrices (groups 41–50 tested)
│
├── Analysis_Scripts/            ← Data analysis & visualization pipeline
│   ├── analyze_results.py       ← Main analysis script (generates all plots/CSVs)
│   ├── data/                    ← CSV datasets (all_hits, summary_per_group_dev)
│   └── plots/                   ← All visualizations (global + per group/deviation)
│
├── APM_2.0_graphs_and_data/     ← APM 2.0 summary Excel + graphs
│   ├── APM_2.0_Summary.xlsx     ← Per-deviation hit data (groups 25–35)
│   ├── APM_3.5_Summary.xlsx     ← ★ NEW — per-group summary from CSV
│   ├── graph_2.0/               ← Heatmap, multi-line, small-multiples bar
│   ├── graph_3.5/               ← Heatmap, multi-line, grouped bar (Blues palette)
│   ├── create_APM_3.5.py        ← Script that generated APM_3.5_Summary.xlsx
│   ├── generate_graphs_2.0.py   ← Script for APM 2.0 graphs
│   └── generate_graphs_3.5.py   ← Script for APM 3.5 graphs
│
├── profiling_results/           ← GPU profiling data (Nsight Systems / ncu)
│   ├── baseline_before.txt      ← Full run log from RTX 2080 (group 25)
│   ├── sys_trace.nsys-rep       ← Nsight Systems trace file
│   └── profiling_brahma_2/      ← ncu metric CSVs (brahma_full, roofline)
│
├── server_brahma/               ← Earlier server kernel (groups 36–40)
├── server_brahma_2/             ← Intermediate server version
├── results_combined/            ← Merged canonical result set (groups 25–40)
└── README.md                    ← This file
```

---

## 4. Matrix Generation — `lasvegas-ecdlp`

**Location:** `lasvegas-ecdlp/apm/`

The matrix generator is based on the **Las Vegas** elliptic-curve discrete log approach adapted for APM kernel generation. It uses C++, MPI (for parallel generation), and NTL.

### How it works

1. For a given prime bit-size `g` and input file (e.g. `input/25_29/25_1.txt`), the generator runs the `fun_ZZp()` path in `main.cpp`.
2. It calls `makeKernelDB()` with `offset = 0.2`, which means the output matrix size is truncated to 20% of the full internal size:
   - Internal: full `n × n` computations
   - Output: `r_target × r_target` matrix written to disk

| Bits | Internal n | r_target (output size) |
|:---:|:---:|:---:|
| 25 | 25 | 15×15 |
| 26 | 26 | 15×15 |
| 27 | 27 | 16×16 |
| 28 | 28 | 16×16 |
| 29 | 29 | 17×17 |
| 35 | 35 | 21×21 |

3. Each kernel file is written to `Kernel_output/<bits>/<bits>_<i>.txt` — one matrix per file, in `[[row0],[row1],...]` format.
4. A companion `_RN.txt` random-number file is also written alongside each kernel.

### Building the generator

```bash
cd lasvegas-ecdlp/apm
make
mpirun -np <N> ./lasVegas
```

Requires: NTL, OpenMPI, GMP.

### Key implementation notes

- `offset` type was changed from `int` → `double` to correctly pass `0.2` (previously truncated to `0`).
- Only `r` and `mat_row` are scaled by offset; `k` and `t` remain full-scale to avoid segfaults in `generateMatrix()`.
- Output paths redirect to `Kernel_output/<bits>/` instead of the old `kernel_DB/new/` path.
- MPI C++ API (`MPI::Get_processor_name`) replaced with C API (`MPI_Get_processor_name`) for compatibility.

See `lasvegas-ecdlp/apm/README_kernel_generation_changes.md` for the full bug-fix history.

---

## 5. CUDA Search Engine — `optimized_further`

**Location:** `optimized_further/`  
**Main file:** `apm_brahma.cu` (1617 lines)  
**Status:** ★ **Latest and most optimized version**

This is the production search engine. It reads kernel files (matrices), and for each matrix searches exhaustively for a zero minor at each deviation level.

### Architecture Overview

```
Host (CPU)                          Device (GPU)
─────────────────────────────────   ────────────────────────────────────
parse_matrix()                  →   d_matrix (long long n×n)
gen_combos_flat() → row/col idx →   d_row_idx, d_col_idx (flat int[])
launch_apm_kernel(k, ...)       →   apm_kernel_k<K><<<grid, block>>>
                                        for each (row_set, col_set):
                                          extract K×K submatrix
                                          det_mod() via Gauss elim mod p
                                          if det==0: atomicExch(found,1)
collect d_zero_flags            ←   d_zero_flags[], d_found
write result file               ←
```

### Key Optimizations

#### 1. Templated kernel `apm_kernel_k<K>`
Each minor size `K` (from 4 to 60) is a **separate template instantiation** compiled with its own optimal stack frame size `sub[K*K]`. This allows the compiler to allocate exact register counts per K, maximizing SM occupancy:

| Minor size K | Stack per thread | V100 occupancy |
|:---:|:---:|:---:|
| k = 4 | 128 bytes | 100% |
| k = 6 | 288 bytes | 100% |
| k = 10 | 800 bytes | 100% |
| k = 27 | 5832 bytes | ~68% |
| k = 50 | 20000 bytes | ~20% |

> Compilation takes 45–90 seconds due to 57 template instantiations. This is expected and correct.

#### 2. `__int128` modular multiply (SM ≥ 70)
On V100 and newer (SM ≥ 700), `mod_mul` uses `unsigned __int128` — a single multiply instruction with no overflow risk for primes up to `2^60`. Older GPUs fall back to `__umul64hi` + binary loop.

```c
#if __CUDA_ARCH__ >= 700
  return (long long)((unsigned __int128)ua * ub % up);
#else
  // __umul64hi fallback for SM < 70
#endif
```

> If `__int128` fails on your RTX 20xx (SM 75), change the threshold to `>= 800`.

#### 3. `__constant__` memory for `pow2_64`
`pow2_64 = 2^64 mod p` is stored in CUDA constant memory (L1-cached on Volta/Turing). Set once per matrix search via `cudaMemcpyToSymbol()` — zero parameter overhead, accessed by every `mod_mul` call.

#### 4. Flat `int[]` index arrays
Row/column index combinations are stored as flat `int[]` arrays rather than `std::vector<IndexSet>`. This gives **5–8× VRAM savings**, enabling larger minor sizes to fit in GPU memory.

#### 5. Kernel range arguments `[kmin, kmax]`
Each invocation can process a subset of the 100 kernel files:
```bash
./apm_brahma 38 38 1 50    # kernels 1–50 on GPU 0
./apm_brahma 38 38 51 100  # kernels 51–100 on GPU 1
```

#### 6. Numerical file sorting
Files are sorted by kernel number (1, 2, 3 … 100), not lexicographically (1, 10, 100, 11 …). Works with both padded (`kernel_38_0001.txt`) and unpadded (`kernel_38_1.txt`) filenames.

#### 7. Runtime VRAM detection
Uses `cudaMemGetInfo()` to use 90% of available VRAM. The column-chunk size for `d_col_idx` is chosen dynamically as the largest power-of-2 fitting within remaining memory, then falls back gracefully with a skip message if VRAM is insufficient.

#### 8. Early-stop via `atomicExch`
Once a zero minor is found in a matrix, `atomicExch(d_found, 1)` sets a flag visible to all warps. Every thread checks `atomicAdd(d_found, 0)` at the start of its work — if already found, it returns immediately.

### Supported prime groups

Hardcoded fallback primes are included for all groups **10–60**. The search has been:
- **Fully tested:** Groups 25–41
- **Theoretically supported:** Groups 10–60 (limited only by GPU memory for large `K`)

---

## 6. Build & Run

### Prerequisites
- CUDA toolkit ≥ 11.x
- g++ ≥ 9 (host compiler)
- GPU with SM ≥ 60 (SM ≥ 70 recommended for `__int128` path)

### Build

```bash
cd optimized_further
make           # auto-detects GPU SM via nvidia-smi
make help      # show all usage options
```

### Run — single group, all kernels

```bash
./apm_brahma 38 38              # group 38, all kernel files, all deviations
./apm_brahma 38 38 1 35         # group 38, kernels 1–35
./apm_brahma 38 38 1 35 2       # group 38, kernels 1–35, deviation 2 ONLY
```

### Run — group range

```bash
./apm_brahma 25 40              # all groups 25 through 40
```

### Output

Results are written to `Results_further/<group>/deviation_<d>/`:
```
Results_further/
└── 38/
    └── deviation_4/
        ├── kernel_38_1_result.txt
        ├── kernel_38_2_result.txt
        ├── ...
        ├── SUMMARY_brief.txt
        └── SUMMARY_detailed.txt
```

---

## 7. Multi-GPU Parallel Execution

The search can be split across multiple GPUs using the kernel range arguments:

```bash
# 4-GPU split for group 38, deviation 2
CUDA_VISIBLE_DEVICES=0 ./apm_brahma 38 38  1  25 2 &
CUDA_VISIBLE_DEVICES=1 ./apm_brahma 38 38 26  50 2 &
CUDA_VISIBLE_DEVICES=2 ./apm_brahma 38 38 51  75 2 &
CUDA_VISIBLE_DEVICES=3 ./apm_brahma 38 38 76 100 2 &
wait
```

Or use the provided shell script:

```bash
./submit_parallel.sh 38 100 4 2    # group 38, 100 kernels, 4 GPUs, deviation 2
```

After a parallel run, merge the fragmented summaries:

```bash
python3 merge_summaries.py 38 --deviation 2
```

Verify completeness:

```bash
python3 check_results.py 38 --deviation 2
```

---

## 8. Results Layout & How to Interpret Them

### Per-matrix result file (`kernel_<g>_<N>_result.txt`)

```
Group 38, Deviation 4
Prime (p): 274877906837
Matrix size n: 31
Minors tested: 284121077673
Zero minors found: 1
Time: 2874552.08 ms

[Zero #1]
  Minor size : 6×6
  Anchor s   : 3
  Rows       : [3, 4, 7, 12, 19, 22]
  Cols       : [3, 4, 8, 11, 20, 25]
  Det mod p  : 0
```

**Field explanations:**

| Field | Meaning |
|---|---|
| **Anchor s** | The starting position of the fixed 2×2 principal block. The minor always includes rows/cols `{s, s+1}` plus `d` additional indices. |
| **Rows / Cols** | The `k = d+2` row and column indices that form the zero minor submatrix. |
| **Minors tested** | How many `K×K` determinants were evaluated before the zero minor was found (early-stop). |
| **Principal** | A minor is *principal* when its row indices equal its column indices (`row_idx == col_idx`). From the data, nearly 100% of hits are **non-principal** — meaning the zero structure is off-diagonal. |

### Summary files

`SUMMARY_brief.txt` — one line per matrix, hit or miss:
```
kernel_38_1: HIT  dev=4  t=2874552ms  minors=284121077673
kernel_38_2: MISS dev=4
...
```

`SUMMARY_detailed.txt` — includes full index sets for each zero minor found.

### Hit ratio interpretation

A **hit ratio of 1.00 (100%)** at deviation `d` means every single one of 100 independently generated random APM matrices for prime group `g` contained at least one zero `(d+2)×(d+2)` minor. This is the **guaranteed existence** result.

A hit ratio below 100% means some matrices had no zero minor at that deviation — those matrices require a higher deviation.

---

## 9. Analysis Pipeline — `Analysis_Scripts`

**Location:** `Analysis_Scripts/`  
**Main script:** `analyze_results.py`

Run once to regenerate all plots and CSVs from raw result files:

```bash
python3 Analysis_Scripts/analyze_results.py
```

### Data sources (canonical whitelist)

The script reads from a fixed whitelist of result directories to exclude legacy/duplicate runs:

| Group | Prime | Source |
|:---:|---:|---|
| 25–32 | 33,554,393 – 4,294,967,291 | `Results_hits_one_check_till_100/<g>/` |
| 33–35 | 8,589,934,583 – 34,359,738,337 | `Parambrahma_data_20April/Results_brahma_2/<g>/` |
| 36–38 | 68,719,476,503 – 274,877,906,837 | `Parambrahma_data_20April/Results_brahma/<g>/` |
| 39–40 | 549,755,813,657 – 1,099,511,627,689 | `Results_further/<g>/` |

### Output layout

```
Analysis_Scripts/
├── plots/
│   ├── global/
│   │   ├── 0a_key_finding_min_dev_100pct.png   ← Core result: bar chart of min dev per group
│   │   ├── 0b_hit_rate_heatmap.png              ← Core result: hit rate heatmap
│   │   ├── 0c_row_index_heatmap.png             ← Index frequency across all groups
│   │   ├── 1_min_dev_per_group.png              ← Min dev at ≥10% hit rate
│   │   ├── 2_hit_ratio_by_deviation.png         ← Hit ratio curves per group
│   │   ├── 6_time_to_hit_boxplot.png
│   │   ├── 7_scatter_time_vs_group.png
│   │   └── 8_scatter_time_vs_minors.png
│   └── group_<G>/deviation_<D>/
│       ├── 3_anchor_s.png                       ← Anchor position histogram
│       ├── 4_principal_check.txt                ← Principal vs non-principal report
│       ├── 5_idx_recurrence.png                 ← Row index frequency bar chart
│       └── 9_scatter_indices.png                ← Row×col scatter of all hits
└── data/
    ├── all_hits.csv                             ← Every individual hit (~2141 rows)
    ├── summary_per_group_dev.csv                ← One row per (group, deviation)
    └── group_<G>_hits.csv                       ← Per-group slices
```

### Key plot: `0a_key_finding_min_dev_100pct.png`
Bar chart showing the minimum deviation required for 100% hit rate. **Blue bars** (groups 25–32) reach saturation at deviation 4. **Red bars** (groups 33–40) require deviation 5. The step at group 33 is the main mathematical claim.

### Key plot: `0b_hit_rate_heatmap.png`
Colour heatmap of hit rate across all `(group × deviation)` combinations. The diagonal transition from 0% to 100% is visible, with the phase boundary shifting right at group 33.

### Note on anchor `s` over-representation
Because the CUDA kernel uses early-stop (`atomicExch`), lower anchor positions (`s = 0, 1, 2`) are systematically over-represented in the data — it reflects search order, not mathematical frequency. Plot 3 documents this but it is **expected behaviour**, not a bias.

---

## 10. Summary Data — APM 2.0 & APM 3.5

### APM 2.0 — `APM_2.0_graphs_and_data/APM_2.0_Summary.xlsx`

Contains per-deviation, per-minor-size hit data for groups 25–35. Each group section lists every tested deviation with the minor size, number of hits out of 100, total minors tested, zero minors found, and hit ratio.

Graphs in `graph_2.0/`:
- `1_heatmap_hit_ratio.png` — hit ratio heatmap (group × deviation, Blues colormap)
- `2_multiline_hit_ratio_vs_deviation.png` — line curves per group
- `3_bar_hit_ratio_per_group.png` — small-multiples bar chart (one panel per group)

### APM 3.5 — `APM_2.0_graphs_and_data/APM_3.5_Summary.xlsx`

Generated from `Analysis_Scripts/data/summary_per_group_dev.csv`. One row per `(group, deviation)` with:

| Column | Description |
|---|---|
| APM Prime Bit | Prime bit-size group (25–40) |
| Matrix Size (n×n) | Minor size tested `(dev+2)×(dev+2)` |
| Hits | `total_hits / 100` |
| Total | Always 100 |
| Minors Tested | `minors_mean × matrices` |
| Total Zero Minors | Count of zero minors found |
| Hit Ratio | As percentage |

Graphs in `graph_3.5/`:
- `1_heatmap_hit_ratio.png` — clean white background, Blues colormap
- `2_multiline_hit_ratio_vs_deviation.png` — one line per group
- `3_grouped_bar_hit_ratio.png` — grouped bars by deviation

---

## 11. Profiling Results

**Location:** `profiling_results/`

### Baseline run log — `baseline_before.txt`

Full console output from an RTX 2080 (SM 7.5, 8 GB VRAM) run on group 25. Key metrics:

- **GPU:** NVIDIA GeForce RTX 2080, 46 SMs, 8156 MB VRAM
- **Group 25, deviation 2 (4×4 minors):**
  - 100 matrices × ~1,904,400 minors each
  - Time per matrix: ~7–10 ms
  - Total for dev 2: **0.740 s**
  - Zero minors found: 5/100
- **Group 25, deviation 3 (5×5 minors):**
  - ~30,085,236 minors mean per matrix
  - Time per matrix: 5 ms – 300 ms (wide range, early-stop)
  - Zero minors found: 97/100

### Nsight Systems trace — `sys_trace.nsys-rep`

Full system trace captured during a brahma run. Open with:
```bash
nsys-ui profiling_results/sys_trace.nsys-rep
```

### NCU roofline data — `profiling_brahma_2/brahma_roofline.csv`

Roofline model CSV from Nsight Compute. Shows arithmetic intensity vs. achieved throughput per kernel instantiation. The smaller-K kernels (k=4–10) are compute-bound at near-peak SM throughput.

---

## 12. Utility Scripts

All scripts are in `optimized_further/`:

### `check_results.py`
Verifies result completeness after a run.

```bash
python3 check_results.py 38                    # all kernels, all deviations
python3 check_results.py 38 --deviation 4      # deviation 4 only
python3 check_results.py 38 --kmin 1 --kmax 50 # kernel range check
```

Output shows complete/missing/empty counts. Prints compressed range notation for missing kernels (e.g. `12-15, 23, 47-50`).

### `merge_summaries.py`
After a multi-GPU parallel run, each GPU writes its own `SUMMARY_brief_k<min>_k<max>.txt`. This script merges them into a single canonical `SUMMARY_brief.txt` and `SUMMARY_detailed.txt`.

```bash
python3 merge_summaries.py 38 --deviation 4
python3 merge_summaries.py 38 40              # merge groups 38 and 40
```

### `sort_kernel_files.py`
Renames kernel files to zero-padded format for consistent numerical ordering (one-time setup per group).

```bash
python3 sort_kernel_files.py 38   # renames 38_1.txt → kernel_38_0001.txt etc.
```

---

## 13. Development History

The codebase evolved through several major versions:

| Version | Location | Key changes |
|---|---|---|
| **Semester 1 / initial** | `Semester_1/` | Proof-of-concept CPU search |
| **Changed to flat array** | `changed_to_flat_array/` | First GPU port, flat index arrays |
| **Optimized flat array** | `optimized_flatarray/` | VRAM-aware chunking, reserve() fix |
| **Optimized further** | `optimized_further/` ★ | Kernel range args, `__int128`, numerical sort |
| **Server brahma** | `server_brahma/` | Cluster-optimized version for groups 36–40 |
| **Server brahma 2** | `server_brahma_2/` | Intermediate with `check_and_move.py` |
| **APM experiment trial** | `APM_experiment_trial/` | Exploratory parameter sweeps |

---

## 14. Hardware Tested

| GPU | SM | VRAM | Groups tested |
|---|---|---|---|
| NVIDIA GeForce RTX 2080 | 7.5 | 8 GB | 25–32 (local) |
| NVIDIA V100 (Parambrahma HPC) | 7.0 | 32 GB | 25–40 |
| Multiple V100 (multi-GPU) | 7.0 | 32 GB × N | 36–41 |

> For SM < 70, the `__int128` path is disabled and the `__umul64hi` fallback is used. This is slower for groups ≥ 40 (primes > 2^40) but numerically correct.

---

## 15. Reproducing the Results

### Step 1 — Generate kernel files (matrices)

```bash
cd lasvegas-ecdlp/apm
make
mpirun -np 4 ./lasVegas        # generates ~100 files per group into Kernel_output/
```

Pre-generated kernel files for groups 25–50 are already present in `lasvegas-ecdlp/apm/kernel_output_25_40/` and `optimized_further/kernel_output/`.

### Step 2 — Sort kernel files (one-time)

```bash
cd optimized_further
python3 sort_kernel_files.py 38
```

### Step 3 — Build the CUDA engine

```bash
cd optimized_further
make
```

### Step 4 — Run the search

```bash
# Single GPU
./apm_brahma 38 38

# Multi-GPU (4 GPUs, deviation 4)
./submit_parallel.sh 38 100 4 4
```

### Step 5 — Verify completeness

```bash
python3 check_results.py 38 --deviation 4
```

### Step 6 — Merge parallel summaries (if multi-GPU)

```bash
python3 merge_summaries.py 38 --deviation 4
```

### Step 7 — Run analysis pipeline

```bash
cd ..
python3 Analysis_Scripts/analyze_results.py
```

Results appear in `Analysis_Scripts/plots/` and `Analysis_Scripts/data/`.

---

## Citation / Contact

This is an active research project investigating the zero-minor structure of APM matrices over prime fields.  
If you use or build upon this work, please cite accordingly.

---

*Last updated: May 2026*
