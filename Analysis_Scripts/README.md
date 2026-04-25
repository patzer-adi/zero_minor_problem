# Zero Minor Problem - Analysis Scripts

This directory contains Python scripts for processing, validating, and generating insights from the CUDA APM kernel outputs.

## Overview of Scripts

### 1. `analyze_results.py`
This is the main research analysis script. It aggregates the data from all parallel GPU runs, extracts hit data (Zero Minors) across different prime groups and deviation levels, and automatically generates visualizations.

**How it works:**
- It recursively scans the `Results_combined` folder for `SUMMARY_detailed*.txt` files.
- It parses these files to extract critical metrics per matrix hit: Group ID, Deviation, Anchor `s`, Row/Col Indices, Time to Find (ms), and Minors Tested.
- It converts this data into a Pandas DataFrame and generates multiple plots in the `plots/` directory to help answer key mathematical questions (e.g., whether hit deviation scales with group size, whether hits are principal submatrices, and recurrence patterns).

**Output Plots Generated:**
1. `1_min_dev_per_group.png` - Shows minimum deviation required to find a hit, by group.
2. `2_hit_ratio_curve.png` - Plots hit ratio against deviation.
3. `3_anchor_s_group*.png` - Histograms showing the distribution of anchor `s` values.
4. `4_principal_submatrix_report.txt` - Evaluates what percentage of hits are principal submatrices.
5. `5_idx_recurrence_group*.png` - Bar charts showing the frequency of matrix indices that yield zero minors.
6. `6_time_to_hit.png` - Box plots showing execution time until first hit by group and deviation.
7. `7_scatter_time_vs_group.png` - Scatterplot analyzing the relationship between prime group size and execution time.
8. `8_scatter_time_vs_minors.png` - Scatterplot examining correlation between number of minors tested and execution time.
9. `9_scatter_indices.png` - Scatterplot of the actual matrix elements (Row vs. Column indices) that form the Zero Minors, revealing positional clustering.

### 2. `../optimized_further/check_results.py`
A validation utility script used before running the analysis script. It verifies that all input matrix files passed to the GPU actually produced a corresponding result file, ensuring no dropped jobs or silent failures.

**How it works:**
- Compares the `kernel_output/` input folder against the `Results_further/` (or `Results_combined/`) output folder.
- Reports missing or empty result files.
- Allows filtering by specific kernel ranges (`--kmin`, `--kmax`) to check partial parallel runs.

**Usage:**
```bash
python3 ../optimized_further/check_results.py 38 --deviation 4
```

## Running the Analysis

To run the full data analysis and re-generate all plots:

```bash
# Ensure you are at the project root directory
python3 Analysis_Scripts/analyze_results.py
```

The plots will be saved to `Analysis_Scripts/plots/`.
