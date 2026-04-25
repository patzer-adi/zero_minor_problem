#!/usr/bin/env python3
"""
check_and_move.py
─────────────────
Interactive script to:
  1. Check which input files in kernel_output/<group>/ are missing
     corresponding _result.txt files in Results_brahma_2/<group>/deviation_<x>/
  2. Optionally move those missing input files (+ their _RN counterparts)
     into  kernel_output_missed/<group>/  for re-processing.

Usage:
    python3 check_and_move.py              # interactive prompts
    python3 check_and_move.py 38 4         # group=38, deviation=4, then prompts
"""

import os
import sys
import shutil


def get_input_files(input_dir, group):
    """Return sorted list of input matrix files (excluding _RN files)."""
    return sorted([
        f for f in os.listdir(input_dir)
        if f.startswith(f"kernel_{group}_")
        and f.endswith(".txt")
        and not f.endswith("_RN.txt")
    ])


def check_deviation(group, deviation,
                     input_base="kernel_output",
                     results_base="Results_brahma_2"):
    """
    Check which input files are missing results.
    Returns (missing, empty, complete) lists of filenames.
    """
    input_dir  = f"{input_base}/{group}"
    result_dir = f"{results_base}/{group}/deviation_{deviation}"

    # ── validate input dir ──
    if not os.path.exists(input_dir):
        print(f"[ERROR] Input dir not found: {input_dir}")
        return None, None, None

    input_files = get_input_files(input_dir, group)

    if not input_files:
        print(f"[ERROR] No input files found in {input_dir}")
        return None, None, None

    # ── validate result dir ──
    if not os.path.exists(result_dir):
        print(f"[ERROR] Result dir not found: {result_dir}")
        print(f"        Deviation {deviation} was never run.")
        return None, None, None

    # ── classify each file ──
    missing  = []
    empty    = []
    complete = []

    for fname in input_files:
        base     = fname.replace(".txt", "")
        expected = f"{result_dir}/{base}_result.txt"

        if not os.path.exists(expected):
            missing.append(fname)
        elif os.path.getsize(expected) == 0:
            empty.append(fname)
        else:
            complete.append(fname)

    # ── report ──
    total = len(input_files)
    print(f"\nGroup {group} | Deviation {deviation}")
    print("=" * 50)
    print(f"  Input files  : {total}")
    print(f"  Complete     : {len(complete)}")
    print(f"  Missing      : {len(missing)}")
    print(f"  Empty        : {len(empty)}")

    if missing:
        print(f"\n  MISSING results:")
        for f in missing:
            print(f"    - {f}")

    if empty:
        print(f"\n  EMPTY result files:")
        for f in empty:
            print(f"    - {f}")

    if not missing and not empty:
        print(f"\n  [OK] All {total} results present.")
    else:
        print(f"\n  [INCOMPLETE] {len(missing) + len(empty)} files need re-run.")

    return missing, empty, complete


def move_missing_files(group, missing_files, empty_files,
                       input_base="kernel_output",
                       missed_base="kernel_output_missed"):
    """
    Move the input files (and their _RN counterparts) for missing/empty
    results into  kernel_output_missed/<group>/.
    """
    input_dir  = f"{input_base}/{group}"
    missed_dir = f"{missed_base}/{group}"

    # Create the output directory
    os.makedirs(missed_dir, exist_ok=True)

    # Combine missing + empty into one list to move
    to_move = sorted(set(missing_files + empty_files))

    moved_count = 0
    for fname in to_move:
        src = os.path.join(input_dir, fname)

        # Also grab the corresponding _RN file if it exists
        rn_fname = fname.replace(".txt", "_RN.txt")
        rn_src   = os.path.join(input_dir, rn_fname)

        # Move the main file
        if os.path.exists(src):
            shutil.move(src, os.path.join(missed_dir, fname))
            moved_count += 1
            print(f"  moved: {fname}")
        else:
            print(f"  skip (not found): {fname}")

        # Move the _RN file
        if os.path.exists(rn_src):
            shutil.move(rn_src, os.path.join(missed_dir, rn_fname))
            print(f"  moved: {rn_fname}")

    print(f"\n[DONE] Moved {moved_count} input files → {missed_dir}/")
    print(f"       You can now re-run the kernel on this folder.")


# ── interactive main ──────────────────────────────────────────
if __name__ == "__main__":

    # ── Step 1: get group & deviation ──
    if len(sys.argv) > 2:
        group     = int(sys.argv[1])
        deviation = int(sys.argv[2])
    else:
        try:
            group     = int(input("Enter group number  (e.g. 38): "))
            deviation = int(input("Enter deviation     (e.g.  4): "))
        except (ValueError, EOFError):
            print("Invalid input. Exiting.")
            sys.exit(1)

    # ── Step 2: check ──
    missing, empty, complete = check_deviation(group, deviation)

    if missing is None:
        sys.exit(1)

    # Nothing to move
    if not missing and not empty:
        sys.exit(0)

    # ── Step 3: ask to move ──
    total_bad = len(missing) + len(empty)
    print(f"\nMove {total_bad} missing/empty input file(s) "
          f"to kernel_output_missed/{group}/ ?")
    choice = input("  [y/N]: ").strip().lower()

    if choice in ("y", "yes"):
        move_missing_files(group, missing, empty)
    else:
        print("No files moved.")
