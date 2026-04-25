#!/usr/bin/env python3
"""
check_results.py
────────────────
Verify that all kernel files have corresponding results after parallel runs.
Supports kernel range filtering to match the GPU job splits.

Usage:
    python3 check_results.py 38                        # check group 38, ALL kernels
    python3 check_results.py 38 --kmin 1 --kmax 30     # check kernels 1-30 only
    python3 check_results.py 38 --deviation 4          # check specific deviation
    python3 check_results.py 38 --results-dir Results_further
"""

import os
import re
import sys
import argparse


def check_group(group, deviation=None, input_base="kernel_output",
                results_base="Results_further", kmin=None, kmax=None):
    """Check which input files have/lack result files."""

    input_dir = f"{input_base}/{group}"

    if not os.path.exists(input_dir):
        print(f"[ERROR] Input dir not found: {input_dir}")
        return

    # Find all input kernel files (not _RN)
    pattern = re.compile(rf"kernel_{group}_(\d+)\.txt$")
    input_files = []
    for f in os.listdir(input_dir):
        m = pattern.match(f)
        if m:
            knum = int(m.group(1))
            # Filter by kernel range if specified
            if kmin is not None and knum < kmin:
                continue
            if kmax is not None and knum > kmax:
                continue
            input_files.append((knum, f))

    input_files.sort(key=lambda x: x[0])

    if not input_files:
        print(f"[ERROR] No kernel files found in {input_dir}"
              + (f" (range {kmin}-{kmax})" if kmin or kmax else ""))
        return

    range_str = ""
    if kmin is not None or kmax is not None:
        range_str = f" (range {kmin or 'start'}-{kmax or 'end'})"

    print(f"\nGroup {group}: {len(input_files)} input kernel files{range_str}")
    print(f"  Range: {input_files[0][0]} .. {input_files[-1][0]}")

    # Find all deviation directories
    group_result_dir = f"{results_base}/{group}"
    if not os.path.exists(group_result_dir):
        print(f"[WARN] No results directory: {group_result_dir}")
        return

    devs = []
    for d in os.listdir(group_result_dir):
        if d.startswith("deviation_"):
            try:
                devs.append(int(d.split("_")[1]))
            except ValueError:
                pass
    devs.sort()

    if deviation is not None:
        devs = [deviation]

    if not devs:
        print(f"[WARN] No deviation subdirectories in {group_result_dir}")
        return

    for dev in devs:
        result_dir = f"{group_result_dir}/deviation_{dev}"
        if not os.path.exists(result_dir):
            print(f"\n  Deviation {dev}: directory not found")
            continue

        missing = []
        empty = []
        complete = []

        for knum, fname in input_files:
            base = fname.replace(".txt", "")
            expected = f"{result_dir}/{base}_result.txt"

            if not os.path.exists(expected):
                missing.append((knum, fname))
            elif os.path.getsize(expected) == 0:
                empty.append((knum, fname))
            else:
                complete.append((knum, fname))

        total = len(input_files)
        pct = 100.0 * len(complete) / total if total > 0 else 0

        print(f"\n  Deviation {dev}:")
        print(f"    Complete : {len(complete)}/{total}  ({pct:.1f}%)")
        print(f"    Missing  : {len(missing)}/{total}")
        print(f"    Empty    : {len(empty)}/{total}")

        if missing:
            ranges = _compress_ranges([m[0] for m in missing])
            print(f"    Missing kernels: {ranges}")

        if empty:
            ranges = _compress_ranges([e[0] for e in empty])
            print(f"    Empty kernels  : {ranges}")

        if not missing and not empty:
            print(f"    [OK] All {total} results present ✓")

    # Also check for range-specific summary files
    _check_summaries(group_result_dir, devs, kmin, kmax)


def _check_summaries(group_result_dir, devs, kmin, kmax):
    """Check for summary files from parallel runs."""
    for dev in devs:
        result_dir = f"{group_result_dir}/deviation_{dev}"
        if not os.path.exists(result_dir):
            continue

        summaries = [f for f in os.listdir(result_dir)
                     if f.startswith("SUMMARY_brief")]
        if len(summaries) > 1:
            print(f"\n  Deviation {dev} — found {len(summaries)} summary files (parallel run):")
            for s in sorted(summaries):
                print(f"    • {s}")


def _compress_ranges(nums):
    """Compress [1,2,3,5,7,8,9] → '1-3, 5, 7-9'"""
    if not nums:
        return ""
    nums = sorted(nums)
    ranges = []
    start = end = nums[0]
    for n in nums[1:]:
        if n == end + 1:
            end = n
        else:
            ranges.append(f"{start}-{end}" if start != end else str(start))
            start = end = n
    ranges.append(f"{start}-{end}" if start != end else str(start))
    return ", ".join(ranges)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check APM results completeness"
    )
    parser.add_argument("groups", type=int, nargs="+",
                        help="Group number(s) to check")
    parser.add_argument("--deviation", "-d", type=int, default=None,
                        help="Check specific deviation only")
    parser.add_argument("--kmin", type=int, default=None,
                        help="Starting kernel number (e.g. 57)")
    parser.add_argument("--kmax", type=int, default=None,
                        help="Ending kernel number (e.g. 83)")
    parser.add_argument("--results-dir", default="Results_further",
                        help="Results base directory")
    parser.add_argument("--input-dir", default="kernel_output",
                        help="Input base directory")

    args = parser.parse_args()

    for g in args.groups:
        check_group(g, args.deviation, args.input_dir, args.results_dir,
                    args.kmin, args.kmax)
    print()
