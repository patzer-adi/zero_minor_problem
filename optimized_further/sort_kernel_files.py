#!/usr/bin/env python3
"""
sort_kernel_files.py
────────────────────
Sorts kernel output files into strict numerical (serial) order.

MPI writes kernel files in whatever order processes finish:
    kernel_38_18.txt, kernel_38_32.txt, kernel_38_1.txt, ...

This script renames them to zero-padded serial order:
    kernel_38_0001.txt, kernel_38_0002.txt, kernel_38_0003.txt, ...

The renaming preserves the CONTENT — it only changes filenames so that
lexicographic sort == numeric sort. Each _RN.txt companion is also renamed.

Usage:
    python3 sort_kernel_files.py 38                       # sort group 38
    python3 sort_kernel_files.py 38 39 40                 # sort groups 38-40
    python3 sort_kernel_files.py 38 --base kernel_output  # custom base dir
    python3 sort_kernel_files.py 38 --dry-run             # preview only
"""

import os
import re
import sys
import argparse
import shutil


def sort_kernel_files(group, input_base="kernel_output", dry_run=False):
    """
    Sort kernel files for a given group into zero-padded serial order.

    Files are sorted by the numeric suffix:
        kernel_38_1.txt  (suffix=1)
        kernel_38_2.txt  (suffix=2)
        ...
        kernel_38_100.txt (suffix=100)

    Renamed to:
        kernel_38_0001.txt
        kernel_38_0002.txt
        ...
        kernel_38_0100.txt

    The companion _RN.txt files are also renamed correspondingly.
    """
    folder = os.path.join(input_base, str(group))

    if not os.path.exists(folder):
        print(f"[ERROR] {folder} not found")
        return False

    # Match kernel files: kernel_<group>_<number>.txt  (NOT _RN.txt)
    pattern = re.compile(rf"kernel_{group}_(\d+)\.txt$")

    files = []
    for f in os.listdir(folder):
        m = pattern.match(f)
        if m:
            files.append((int(m.group(1)), f))

    if not files:
        print(f"[ERROR] No kernel files found in {folder}")
        return False

    # Sort by the numeric suffix
    files.sort(key=lambda x: x[0])

    total = len(files)
    pad_width = len(str(total))  # e.g., 100 files → 4 digits (0001..0100)
    if pad_width < 4:
        pad_width = 4

    print(f"\nGroup {group}: found {total} kernel files")
    print(f"  Directory : {folder}")
    print(f"  Padding   : {pad_width} digits")
    print(f"  Range     : {files[0][0]} .. {files[-1][0]}")

    # Check if already sorted (files might already be zero-padded)
    already_sorted = True
    for new_idx, (old_num, fname) in enumerate(files, start=1):
        new_name = f"kernel_{group}_{new_idx:0{pad_width}d}.txt"
        if fname != new_name:
            already_sorted = False
            break

    if already_sorted:
        print(f"  [OK] Already in serial order. Nothing to do.")
        return True

    # Two-pass rename to avoid collisions (old name might clash with new name)
    # Pass 1: rename to temporary names
    print(f"\n  Pass 1: Rename to temp names...")
    temp_map = []
    for new_idx, (old_num, fname) in enumerate(files, start=1):
        old_path = os.path.join(folder, fname)
        temp_name = f"__tmp_sort_{new_idx:0{pad_width}d}.txt"
        temp_path = os.path.join(folder, temp_name)

        # Also handle _RN companion
        rn_old = fname.replace(".txt", "_RN.txt")
        rn_old_path = os.path.join(folder, rn_old)
        rn_temp_name = f"__tmp_sort_{new_idx:0{pad_width}d}_RN.txt"
        rn_temp_path = os.path.join(folder, rn_temp_name)

        has_rn = os.path.exists(rn_old_path)

        if dry_run:
            print(f"    [DRY] {fname} → {temp_name}")
        else:
            os.rename(old_path, temp_path)
            if has_rn:
                os.rename(rn_old_path, rn_temp_path)

        temp_map.append((new_idx, temp_name, has_rn, rn_temp_name))

    # Pass 2: rename temp names to final serial names
    print(f"  Pass 2: Rename to final serial names...")
    renamed = 0
    for new_idx, temp_name, has_rn, rn_temp_name in temp_map:
        temp_path = os.path.join(folder, temp_name)
        final_name = f"kernel_{group}_{new_idx:0{pad_width}d}.txt"
        final_path = os.path.join(folder, final_name)

        rn_final_name = f"kernel_{group}_{new_idx:0{pad_width}d}_RN.txt"
        rn_final_path = os.path.join(folder, rn_final_name)
        rn_temp_path = os.path.join(folder, rn_temp_name)

        if dry_run:
            print(f"    [DRY] {temp_name} → {final_name}")
        else:
            os.rename(temp_path, final_path)
            if has_rn:
                os.rename(rn_temp_path, rn_final_path)
            renamed += 1

    action = "Would rename" if dry_run else "Renamed"
    print(f"\n  [DONE] {action} {renamed} files to serial order "
          f"0001..{total:0{pad_width}d}")
    return True


def verify_sorted(group, input_base="kernel_output"):
    """Verify that files are in correct serial order after sorting."""
    folder = os.path.join(input_base, str(group))
    if not os.path.exists(folder):
        print(f"[ERROR] {folder} not found")
        return False

    pattern = re.compile(rf"kernel_{group}_(\d+)\.txt$")
    files = []
    for f in sorted(os.listdir(folder)):
        m = pattern.match(f)
        if m:
            files.append((int(m.group(1)), f))

    if not files:
        print(f"[ERROR] No kernel files in {folder}")
        return False

    # Check consecutive numbering
    ok = True
    for i, (num, fname) in enumerate(files, start=1):
        if num != i:
            print(f"  [MISMATCH] Expected index {i}, got {num} ({fname})")
            ok = False

    if ok:
        print(f"  [VERIFIED] Group {group}: {len(files)} files, "
              f"1..{len(files)} in order ✓")
    return ok


# ── CLI ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sort kernel output files into serial order"
    )
    parser.add_argument("groups", type=int, nargs="+",
                        help="Group number(s) to sort, e.g. 38 39 40")
    parser.add_argument("--base", default="kernel_output",
                        help="Base directory (default: kernel_output)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview renames without executing")
    parser.add_argument("--verify", action="store_true",
                        help="Only verify (don't rename)")

    args = parser.parse_args()

    for g in args.groups:
        if args.verify:
            verify_sorted(g, args.base)
        else:
            sort_kernel_files(g, args.base, args.dry_run)
            if not args.dry_run:
                verify_sorted(g, args.base)
        print()
