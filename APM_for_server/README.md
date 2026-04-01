# APM Exhaustive Search — Integrated Implementation

## Overview
This project performs an **exhaustive search** for Almost Principal Minors (APM) in ECDLP-related matrices. The goal is to find submatrices (minors) that have a zero determinant modulo a given prime.

---

## Versions

| Version | Source | Input | Primes | Notes |
|---------|--------|-------|--------|-------|
| V3 | `apm_research_3.cu` | `p_1048573_*` folders | 1048573 | Original, single prime |
| V5 | `apm_research_5.cu` | `p_1048573_*` + `25_29/` | 1048573 + 25–29 | Multi-prime, `int` arithmetic |
| V5new | `apm_research_5_new.cu` | `new/` folder | 25–35 | Long long kernel, flat folder |
| **V6** | **`apm_research_6.cu`** | **`kernel_output/25..35/`** | **25–35 (11 primes)** | **Production rewrite, all bugs fixed** |

---

## V6 — What's New

### Bug Fixes from V5
- **No MinorJob arrays** — threads compute `(r,c)` from `tid` via integer division
- **No ChunkResult/atomicCAS** — `d_zero_flags[tid]` captures ALL zero minors
- **Runtime COL_CHUNK** — computed per `(n, dev)` to stay within 1500 MB VRAM
- **All `long long` arithmetic** — correct for 35-bit primes (up to 34359738337)
- **Runtime max_dev** — `n - 3` computed from actual matrix size, not hardcoded to 5
- **`MAX_IDX_STATIC = 32`** — supports matrices up to 34×34

### New Features
- **11 prime groups** (25–35 bits) processed from `kernel_output/` folders
- **Two summary files** per (group, deviation):
  - `SUMMARY_detailed.txt` — full zero minor info (only matrices with hits)
  - `SUMMARY_brief.txt` — hit ratios, totals, timing
- **Prime file reader** — reads from `25_29/` and `exp/` with hardcoded fallback

---

## Directory Structure

### Input
```
kernel_output/
  25/  kernel_25_1.txt ... kernel_25_100.txt  (skip *_RN.txt)
  26/  27/  28/  29/  30/  31/  32/  33/  34/  35/

25_29/
  25_1.txt  26_1.txt  27_1.txt  28_1.txt  29_1.txt   (prime files)

exp/
  30_1.txt  31_1.txt  32_1.txt  33_1.txt  34_1.txt  35_1.txt  (prime files)
```

### Output (V6)
```
Results_6/
  25/
    deviation_2/
      SUMMARY_detailed.txt
      SUMMARY_brief.txt
      kernel_25_1_result.txt  ...  kernel_25_100_result.txt
    deviation_3/  ...  deviation_12/
  26/  27/  ...  35/
```

---

## Build & Run

### V6 (recommended)
```bash
make apm6              # compile
make run6              # compile + run
make run6_log          # compile + run + save to run6_log.txt
```

### V5 (legacy)
```bash
make apm5              # compile apm_research_5.cu
make run5_log          # run with log
```

### V5new (legacy)
```bash
make apm5new           # compile apm_research_5_new.cu
make run5new_log       # run with log
```

### Clean
```bash
make clean6            # remove apm6 binary
make clean6_results    # remove Results_6/ and log
make clean_all         # remove everything
```

---

## Execution Order (V6)

For each **group** G from 25 to 35:
1. Read prime from file (`25_29/G_1.txt` or `exp/G_1.txt`)
2. Collect ~100 matrices from `kernel_output/G/`
3. For each **deviation** D from 2 to n−3:
   - Process all matrices on GPU
   - Write per-matrix result files + summaries

---

## Prime Table

| Group | Prime | Bits |
|-------|-------|------|
| 25 | 33554393 | 25 |
| 26 | 44923183 | 26 |
| 27 | 134217689 | 27 |
| 28 | 268435399 | 28 |
| 29 | 536870909 | 29 |
| 30 | 1073741789 | 30 |
| 31 | 2147483647 | 31 |
| 32 | 4294967291 | 32 |
| 33 | 8589934583 | 33 |
| 34 | 17179869143 | 34 |
| 35 | 34359738337 | 35 |

> **Note:** Primes 32–35 exceed `INT_MAX`. All GPU arithmetic uses `long long`.

---

## GPU Kernel Design

- Each thread extracts a `(2+d)×(2+d)` submatrix and computes its determinant mod p
- Zero determinant → flag set in `d_zero_flags[tid]`
- Host scans ALL flags after each kernel launch — captures every zero minor
- Split multiplication (`mod_mul`) avoids 128-bit overflow for 35-bit primes
- Gaussian elimination with pivoting for determinant computation

---

## Hardware Requirements

- **GPU:** NVIDIA GTX 750 Ti (SM 5.0, 2 GB VRAM) or better
- **CUDA:** 11.4+
- **Compiler:** gcc-10 (gcc-11 causes `__malloc__` errors with CUDA 11.4)
- **OS:** Ubuntu 22.04
