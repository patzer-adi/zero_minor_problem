#!/usr/bin/env python3
"""
merge_summaries.py
──────────────────
Merges all range-specific SUMMARY_brief_kN_kM.txt files from parallel GPU runs
into a single combined SUMMARY_brief_COMBINED.txt.

After running 4 parallel GPU jobs on group 38, deviation 2:
  Results_further/38/deviation_2/
    SUMMARY_brief_k1_k25.txt       ← GPU 0
    SUMMARY_brief_k26_k50.txt      ← GPU 1
    SUMMARY_brief_k51_k75.txt      ← GPU 2
    SUMMARY_brief_k76_k100.txt     ← GPU 3

This script reads all of them, sums up the stats, and writes:
    SUMMARY_brief_COMBINED.txt     ← merged totals

Usage:
    python3 merge_summaries.py 38 2                          # group 38, deviation 2
    python3 merge_summaries.py 38 2 --results-dir Results_further
    python3 merge_summaries.py 38 2 3 4 5                    # merge deviations 2-5
"""

import os
import re
import sys
import argparse


def parse_summary_brief(filepath):
    """Parse a SUMMARY_brief file and extract numeric stats."""
    stats = {
        "total_matrices": 0,
        "matrices_hit": 0,
        "total_minors_tested": 0.0,
        "total_zero_minors": 0,
        "total_time_s": 0.0,
        "kernel_range": "",
        "source_file": os.path.basename(filepath),
    }

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()

            m = re.match(r"Total matrices\s*:\s*(\d+)", line)
            if m:
                stats["total_matrices"] = int(m.group(1))

            m = re.match(r"Matrices hit\s*:\s*(\d+)", line)
            if m:
                stats["matrices_hit"] = int(m.group(1))

            m = re.match(r"Total minors tested\s*:\s*([\d.eE+]+)", line)
            if m:
                stats["total_minors_tested"] = float(m.group(1))

            m = re.match(r"Total zero minors\s*:\s*(\d+)", line)
            if m:
                stats["total_zero_minors"] = int(m.group(1))

            m = re.match(r"Total time\s*:\s*([\d.]+)\s*s", line)
            if m:
                stats["total_time_s"] = float(m.group(1))

            # Extract kernel range from filename: SUMMARY_brief_k1_k30.txt
            m2 = re.search(r"_k(\d+)_k(\d+)", os.path.basename(filepath))
            if m2:
                stats["kernel_range"] = f"{m2.group(1)}-{m2.group(2)}"

    return stats


def merge_for_deviation(group, dev, results_base="Results_further"):
    """Merge all SUMMARY_brief_kN_kM.txt files for a group+deviation."""

    result_dir = f"{results_base}/{group}/deviation_{dev}"

    if not os.path.exists(result_dir):
        print(f"[ERROR] Directory not found: {result_dir}")
        return

    # Find all range-specific summary files
    summary_files = sorted([
        f for f in os.listdir(result_dir)
        if f.startswith("SUMMARY_brief_k") and f.endswith(".txt")
    ])

    if not summary_files:
        # Check if there's a single non-range summary
        if os.path.exists(os.path.join(result_dir, "SUMMARY_brief.txt")):
            print(f"  Group {group}, deviation {dev}: single SUMMARY_brief.txt "
                  f"(no parallel splits to merge)")
        else:
            print(f"  Group {group}, deviation {dev}: no summary files found")
        return

    print(f"\nGroup {group}, Deviation {dev}")
    print(f"  Found {len(summary_files)} partial summaries:")

    # Parse each summary
    all_stats = []
    for sf in summary_files:
        path = os.path.join(result_dir, sf)
        stats = parse_summary_brief(path)
        all_stats.append(stats)
        print(f"    {sf:40s}  matrices={stats['total_matrices']:4d}  "
              f"hit={stats['matrices_hit']:4d}  "
              f"time={stats['total_time_s']:.1f}s")

    # Sum up
    total_matrices = sum(s["total_matrices"] for s in all_stats)
    total_hit = sum(s["matrices_hit"] for s in all_stats)
    total_minors = sum(s["total_minors_tested"] for s in all_stats)
    total_zeros = sum(s["total_zero_minors"] for s in all_stats)
    max_time = max(s["total_time_s"] for s in all_stats)  # wall time = longest job
    sum_time = sum(s["total_time_s"] for s in all_stats)  # total GPU-hours

    hit_pct = 100.0 * total_hit / total_matrices if total_matrices > 0 else 0.0
    avg_time = sum_time / total_matrices if total_matrices > 0 else 0.0

    # Read group/prime info from first summary
    prime_str = "N/A"
    minor_str = "N/A"
    matrix_size = "N/A"
    first_path = os.path.join(result_dir, summary_files[0])
    with open(first_path, "r") as f:
        for line in f:
            if "Prime (p)" in line:
                prime_str = line.split(":")[-1].strip()
            if "Minor size" in line:
                minor_str = line.split(":")[-1].strip()
            if "Matrix size" in line:
                matrix_size = line.split(":")[-1].strip()

    # Write combined summary
    combined_path = os.path.join(result_dir, "SUMMARY_brief_COMBINED.txt")
    with open(combined_path, "w") as f:
        f.write("============================================================\n")
        f.write("APM Brief Summary  (COMBINED from parallel GPU runs)\n")
        f.write("============================================================\n")
        f.write(f"Prime group      : {group}\n")
        f.write(f"Prime (p)        : {prime_str}\n")
        f.write(f"Deviation level  : {dev}\n")
        f.write(f"Minor size       : {minor_str}\n")
        f.write(f"Matrix size (n)  : {matrix_size}\n")
        f.write(f"Input folder     : kernel_output/{group}/\n")
        f.write(f"Output folder    : {result_dir}\n")
        f.write(f"GPU jobs merged  : {len(all_stats)}\n")
        f.write("------------------------------------------------------------\n")
        f.write(f"Total matrices   : {total_matrices}\n")
        f.write(f"Matrices hit     : {total_hit}      (at least one zero minor)\n")
        f.write(f"Hit ratio        : {total_hit}/{total_matrices} = {hit_pct:.2f}%\n")
        f.write(f"Total minors tested  : {total_minors:.0f}\n")
        f.write(f"Total zero minors    : {total_zeros}\n")
        f.write(f"Wall time (longest)  : {max_time:.3f} s\n")
        f.write(f"Total GPU time       : {sum_time:.3f} s\n")
        f.write(f"Avg time per matrix  : {avg_time:.3f} s\n")
        f.write("------------------------------------------------------------\n")
        f.write("Per-chunk breakdown:\n")
        f.write(f"  {'Range':<15s} {'Matrices':>10s} {'Hit':>6s} {'Hit%':>8s} {'Time(s)':>10s}\n")
        f.write(f"  {'-'*15:<15s} {'-'*10:>10s} {'-'*6:>6s} {'-'*8:>8s} {'-'*10:>10s}\n")
        for s in all_stats:
            rng = s["kernel_range"] or "all"
            pct = 100.0 * s["matrices_hit"] / s["total_matrices"] if s["total_matrices"] > 0 else 0
            f.write(f"  {rng:<15s} {s['total_matrices']:>10d} {s['matrices_hit']:>6d} "
                    f"{pct:>7.1f}% {s['total_time_s']:>10.1f}\n")
        f.write("============================================================\n")

    print(f"\n  COMBINED SUMMARY:")
    print(f"    Matrices    : {total_hit}/{total_matrices} hit ({hit_pct:.1f}%)")
    print(f"    Zero minors : {total_zeros}")
    print(f"    Wall time   : {max_time:.1f}s (longest job)")
    print(f"    GPU time    : {sum_time:.1f}s (sum of all jobs)")
    print(f"    Written to  : {combined_path}")


# Also merge result_kN_kM.txt files at group level
def merge_group_results(group, results_base="Results_further"):
    """Merge range-specific result_kN_kM.txt into result_COMBINED.txt."""

    group_dir = f"{results_base}/{group}"
    if not os.path.exists(group_dir):
        return

    result_files = sorted([
        f for f in os.listdir(group_dir)
        if f.startswith("result_k") and f.endswith(".txt")
    ])

    if not result_files:
        return

    print(f"\n  Merging {len(result_files)} group-level result files...")

    combined_path = os.path.join(group_dir, "result_COMBINED.txt")
    with open(combined_path, "w") as out:
        out.write("============================================================\n")
        out.write(f"APM Combined Result  (group {group})\n")
        out.write("============================================================\n")
        out.write(f"Merged from {len(result_files)} parallel runs:\n\n")

        for rf in result_files:
            path = os.path.join(group_dir, rf)
            out.write(f"--- {rf} ---\n")
            with open(path, "r") as inp:
                out.write(inp.read())
            out.write("\n")

        out.write("============================================================\n")

    print(f"    Written to: {combined_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge parallel GPU summary files into a combined summary"
    )
    parser.add_argument("group", type=int,
                        help="Group number (e.g. 38)")
    parser.add_argument("deviations", type=int, nargs="+",
                        help="Deviation level(s) to merge (e.g. 2 3 4)")
    parser.add_argument("--results-dir", default="Results_further",
                        help="Results base directory")

    args = parser.parse_args()

    for dev in args.deviations:
        merge_for_deviation(args.group, dev, args.results_dir)

    merge_group_results(args.group, args.results_dir)
    print()
