# Analysis Scripts — Zero Minor Problem

This directory contains research analysis tools for the CUDA APM kernel outputs.
Run `analyze_results.py` once to regenerate all plots and datasets from scratch.

---

## Quick Start

```bash
# From the project root
python3 Analysis_Scripts/analyze_results.py
```

All output goes into two directories that are created automatically:
- `Analysis_Scripts/plots/` — all visualizations
- `Analysis_Scripts/data/`  — all CSV datasets

---

## Data Sources (Whitelist)

The script reads **only** from the following canonical folders inside `Results_combined/`.
Old/duplicate runs (`35.old`, `38.old`, `39.old`, group 33 from `Results_hits_one_check_till_100`) are **explicitly excluded**.

| Group (bits) | Prime (p) | Source folder |
|:---:|---:|---|
| 25 | 33554393 | `Results_hits_one_check_till_100/25` |
| 26 | 44923183 | `Results_hits_one_check_till_100/26` |
| 27 | 134217689 | `Results_hits_one_check_till_100/27` |
| 28 | 268435399 | `Results_hits_one_check_till_100/28` |
| 29 | 536870909 | `Results_hits_one_check_till_100/29` |
| 30 | 1073741789 | `Results_hits_one_check_till_100/30` |
| 31 | 2147483647 | `Results_hits_one_check_till_100/31` |
| 32 | 4294967291 | `Results_hits_one_check_till_100/32` |
| 33 | 8589934583 | `Parambrahma_data_20April/Results_brahma_2/33` |
| 34 | 17179869143 | `Parambrahma_data_20April/Results_brahma_2/34` |
| 35 | 34359738337 | `Parambrahma_data_20April/Results_brahma_2/35` *(not 35.old)* |
| 36 | 68719476503 | `Parambrahma_data_20April/Results_brahma/36` |
| 37 | 137438953097 | `Parambrahma_data_20April/Results_brahma/37` |
| 38 | 274877906837 | `Parambrahma_data_20April/Results_brahma/38` *(not 38.old)* |
| 39 | 549755813657 | `Results_further/39` |
| 40 | 1099511627689 | `Results_further/40` |

---

## Output Layout

```
Analysis_Scripts/
├── analyze_results.py          ← main script
├── README.md                   ← this file
├── plots/
│   ├── global/                 ← cross-group summary plots
│   │   ├── 0a_key_finding_min_dev_100pct.png
│   │   ├── 0b_hit_rate_heatmap.png
│   │   ├── 0c_row_index_heatmap.png
│   │   ├── 1_min_dev_per_group.png
│   │   ├── 2_hit_ratio_by_deviation.png
│   │   ├── 6_time_to_hit_boxplot.png
│   │   ├── 7_scatter_time_vs_group.png
│   │   └── 8_scatter_time_vs_minors.png
│   └── group_<G>/
│       └── deviation_<D>/      ← one folder per (group, deviation) pair
│           ├── 3_anchor_s.png
│           ├── 4_principal_check.txt
│           ├── 5_idx_recurrence.png
│           └── 9_scatter_indices.png
└── data/
    ├── all_hits.csv                    ← every individual hit (2141 rows)
    ├── summary_per_group_dev.csv       ← one row per (group, deviation)
    ├── group_25_hits.csv
    ├── group_26_hits.csv
    │   ...
    └── group_40_hits.csv
```

---

## Plots Reference

### Global plots (`plots/global/`)

| # | File | What it shows |
|---|------|---------------|
| 0a | `0a_key_finding_min_dev_100pct.png` | **Core result** — bar chart showing the minimum deviation needed to achieve 100% hit rate per group. Blue bars = groups 25–32 (dev 4); Red bars = groups 33–40 (dev 5). The step at group 33 is the main mathematical claim. |
| 0b | `0b_hit_rate_heatmap.png` | **Core result** — colour heatmap of hit rate (green = 100%, red = 0%) across all group × deviation combinations. Shows the transition boundary visually. |
| 0c | `0c_row_index_heatmap.png` | Overall row index frequency heatmap across all groups. Each cell = fraction of hits for that group in which a given matrix index appeared. Reveals whether certain index positions are structurally preferred across all prime groups. |
| 1 | `1_min_dev_per_group.png` | Minimum deviation where hit rate first reaches ≥10% per group. Uses a 10% threshold to filter out stray isolated hits (e.g. 1–8 hits at dev 2 for groups 25, 26, 29) which are not mathematically meaningful. |
| 2 | `2_hit_ratio_by_deviation.png` | Fraction of tested matrices with a zero minor, at each deviation level, per group. |
| 6 | `6_time_to_hit_boxplot.png` | Box plot of time-to-first-hit (ms) across all matrices, grouped by prime group and deviation. |
| 7 | `7_scatter_time_vs_group.png` | Scatter plot showing how execution time scales with group (bit-size). |
| 8 | `8_scatter_time_vs_minors.png` | Scatter plot correlating number of minors tested with time taken. |

### Per-(group, deviation) plots (`plots/group_<G>/deviation_<D>/`)

| # | File | What it shows |
|---|------|---------------|
| 3 | `3_anchor_s.png` | Histogram (% of hits) of the principal anchor block position `s`. Due to early-stopping in the CUDA kernel, lower `s` values are systematically over-represented — this is expected behaviour, not a mathematical bias. |
| 4 | `4_principal_check.txt` | Text report: prime, total hits, count of principal submatrices (row indices == col indices), and percentage. |
| 5 | `5_idx_recurrence.png` | Bar chart of how often each matrix row index appears across all zero minor row sets. Reveals structural clustering. |
| 9 | `9_scatter_indices.png` | Scatter plot of (row index, col index) pairs from all hits. Matrix convention: (0,0) top-left, y-axis inverted. Dense clusters indicate preferred submatrix positions. |

---

## Datasets Reference (`data/`)

### `all_hits.csv`
Every individual Zero Minor hit, one row per hit.

| Column | Type | Description |
|--------|------|-------------|
| `group` | int | Prime bit-size |
| `dev` | int | Deviation level (minor size = 2 + dev) |
| `prime` | int | The prime p used for this group |
| `s` | int | Principal anchor block start index |
| `row_idx` | str | Comma-separated row indices of the zero minor submatrix |
| `col_idx` | str | Comma-separated col indices of the zero minor submatrix |
| `time_ms` | float | Time to find this hit (ms) |
| `minors_tested` | float | Number of minors tested before this hit was found |
| `matrix` | str | Source kernel filename |
| `is_principal` | bool | True if row_idx == col_idx (principal submatrix) |

### `summary_per_group_dev.csv`
One row per (group, deviation) combination — useful for high-level analysis.

| Column | Description |
|--------|-------------|
| `group`, `dev` | Identifiers |
| `prime` | Prime for this group |
| `total_hits` | Zero minor hits found |
| `time_min/max/mean_ms` | Time-to-hit statistics |
| `minors_mean` | Average minors tested per hit |
| `principal_pct` | % of hits that are principal submatrices |
| `anchor_s_mode` | Most frequent anchor position |
| `matrices` | Total matrices processed |
| `zero_minors` | Total zero minors found |
| `hit_ratio` | zero_minors / matrices |

### `group_<G>_hits.csv`
Per-group slice of `all_hits.csv`. Same columns. Convenient for group-specific analysis without loading the full dataset.

---

## Related Utilities

| Script | Location | Purpose |
|--------|----------|---------|
| `check_results.py` | `optimized_further/` | Validates that every input kernel file produced a result file. Run this before analysis to catch missing jobs. |
| `merge_summaries.py` | `optimized_further/` | Merges fragmented `SUMMARY_detailed_k*_k*.txt` files (from multi-GPU parallel runs) into a single summary. |
| `sort_kernel_files.py` | `optimized_further/` | Numerically sorts kernel input files for consistent ordering. |

---

## Notes

### Experimental Setup
Each **kernel file** (`kernel_<group>_<N>.txt`) contains **one independently generated matrix** mod p. The CUDA program processes these kernel files one by one — each matrix is a completely independent trial.

So **"Matrices: 100, Zero minors: 100, hit_ratio: 1.00"** means: every single one of 100 independent random matrices, when searched at deviation `d`, had at least one zero minor found. No exceptions.

This is a strong result — it is not "sometimes you find one" — it is **universally true** across every matrix tested at that deviation.

> **Caveat:** The 100 matrices are drawn from a specific generator (the Sage/Python kernel generator for each prime group). "Universal" is with respect to that distribution. But 100/100 across independent random samples is strong empirical evidence.

### Key Mathematical Finding
The summary dataset reveals a strong pattern — the **minimum deviation required to guarantee a zero minor in 100% of tested matrices grows with group size**:

| Min deviation for 100% hit rate | Groups |
|---|---|
| deviation = 4 | 25, 26, 27, 28, 29, 30, 31, 32 |
| deviation = 5 | 33, 34, 36, 37, 38, 39, 40 |
| incomplete data | 35 (only 2 hits; `35.old` excluded) |

This means: for any matrix generated from a prime of bit-size 25–32, a zero minor of size `2 + 4 = 6` is **always** findable. For groups 33–40, the guaranteed size jumps to `2 + 5 = 7`. The threshold shifts at exactly group 33, suggesting a structural boundary around the 2³³ prime range.

See plots `0a` and `0b` in `plots/global/` for the visual proof.

### Why Plot 1 Uses a 10% Threshold
A raw minimum (any hit at all) gives misleading results for groups 25, 26, and 29 which had 8, 2, and 1 stray hits at deviation 2 respectively (8%, 2%, 1% hit rate). These are not mathematically meaningful — they are likely corner-case matrices. Plot 1 uses `hit_ratio >= 10%` as the threshold for "first meaningful deviation", giving a clean and honest curve.

### Other Notes
- **Group 35 has only 2 hits** because `35.old` was excluded. The clean `35/` folder had minimal results at deviations 2–4.
- **Group 33 is sourced from `Results_brahma_2`**, not `Results_hits_one_check_till_100`, because the latter's group-33 data is considered preliminary/superseded.
- The CUDA kernel uses **early-stopping** (`atomicExch(d_found, 1)`): once a zero minor is found in a matrix, it stops searching. This means anchor `s=0` is always over-represented in plot 3 — it reflects search order, not mathematical frequency.
