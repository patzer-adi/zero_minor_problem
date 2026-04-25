"""
analyze_results.py
==================
Parses all SUMMARY_detailed*.txt files under Results_combined/ and generates
research-quality visualizations.

Output folder layout:
  plots/
    global/                         <- cross-group summary plots (1, 2, 6, 7, 8)
    group_<G>/
      deviation_<D>/                <- per (group, deviation) plots (3, 4, 5, 9)

Plots:
  1. Min deviation per group (line)
  2. Hit ratio by deviation (line, when SUMMARY_brief available)
  3. Anchor s distribution (histogram, % of hits)          [per group/dev]
  4. Principal submatrix report (text)                      [per group/dev]
  5. Index recurrence (bar)                                 [per group/dev]
  6. Time to first hit (boxplot, global)
  7. Time vs Group (scatter, global)
  8. Time vs Minors Tested (scatter, global)
  9. Selected matrix indices (scatter row vs col)           [per group/dev]
"""

import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — works on all machines
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR    = Path("/Users/adityagowari/Programs/crypto_proj/zero_minor_problem")
RESULTS_DIR = BASE_DIR / "Results_combined"
OUTPUT_DIR  = BASE_DIR / "Analysis_Scripts" / "plots"
GLOBAL_DIR  = OUTPUT_DIR / "global"

for d in [OUTPUT_DIR, GLOBAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Prime lookup: group (bit-size) -> prime
# ---------------------------------------------------------------------------
PRIMES = {
    10: 1021,               11: 2029,               12: 4079,
    13: 8111,               14: 16273,              15: 32749,
    16: 65413,              17: 131071,              18: 262103,
    19: 524257,             20: 1048573,             21: 2097147,
    22: 4194217,            23: 8388449,             24: 16777099,
    25: 33554393,           26: 44923183,            27: 134217689,
    28: 268435399,          29: 536870909,           30: 1073741789,
    31: 2147483647,         32: 4294967291,          33: 8589934583,
    34: 17179869143,        35: 34359738337,         36: 68719476503,
    37: 137438953097,       38: 274877906837,        39: 549755813657,
    40: 1099511627689,      41: 2199023255531,       42: 4398046511093,
    43: 8796093022151,      44: 17592186044399,      45: 35184372088777,
    46: 70368744177643,     47: 140737488355213,     48: 281474976710597,
    49: 562949953421231,    50: 1125899906842597,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", palette="tab10")

def save(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {path.relative_to(BASE_DIR)}")

def spec_dir(grp, dev):
    d = OUTPUT_DIR / f"group_{grp}" / f"deviation_{dev}"
    d.mkdir(parents=True, exist_ok=True)
    return d

# ---------------------------------------------------------------------------
# Data source whitelist
# Explicit mapping: which sub-folder inside Results_combined to use for each
# group.  Anything NOT listed here is ignored (35.old, 38.old, 39.old, group
# 33 from Results_hits_one_check_till_100, etc.)
# ---------------------------------------------------------------------------
SOURCES = {
    # group : relative sub-path inside Results_combined
    25: "Results_hits_one_check_till_100/25",
    26: "Results_hits_one_check_till_100/26",
    27: "Results_hits_one_check_till_100/27",
    28: "Results_hits_one_check_till_100/28",
    29: "Results_hits_one_check_till_100/29",
    30: "Results_hits_one_check_till_100/30",
    31: "Results_hits_one_check_till_100/31",
    32: "Results_hits_one_check_till_100/32",
    # 33 from Results_hits_one_check_till_100 is EXCLUDED
    33: "Parambrahma_data_20April/Results_brahma_2/33",
    34: "Parambrahma_data_20April/Results_brahma_2/34",
    35: "Parambrahma_data_20April/Results_brahma_2/35",   # NOT 35.old
    36: "Parambrahma_data_20April/Results_brahma/36",
    37: "Parambrahma_data_20April/Results_brahma/37",
    38: "Parambrahma_data_20April/Results_brahma/38",     # NOT 38.old
    39: "Results_further/39",
    40: "Results_further/40",
}

def iter_summary_files():
    """Yield (group, dev, path) only from the whitelisted source folders."""
    for grp, rel in SOURCES.items():
        src = RESULTS_DIR / rel
        if not src.exists():
            print(f"  [MISSING] {rel}")
            continue
        for path in sorted(src.rglob("SUMMARY_detailed*.txt")):
            dev_str = path.parts[-2]
            if not dev_str.startswith("deviation_"):
                continue
            try:
                dev = int(dev_str.split("_")[1])
            except ValueError:
                continue
            yield grp, dev, path

# ---------------------------------------------------------------------------
# Step 1 — Parse SUMMARY_detailed into DataFrame
# ---------------------------------------------------------------------------
print("\n=== PARSING SUMMARY FILES ===")
rows = []
for grp, dev, path in iter_summary_files():
    text = path.read_text(errors="replace")
    for block in text.split("--- Zero Minor"):
        s_m  = re.search(r"Principal block s:\s*(\d+)", block)
        r_m  = re.search(r"Row indices \[.+?\]\s*:\s*([\d ]+)", block)
        c_m  = re.search(r"Col indices \[.+?\]\s*:\s*([\d ]+)", block)
        t_m  = re.search(r"Time found\s*:\s*([\d.]+)", block)
        mn_m = re.search(r"Minors tested\s*:\s*([\d.]+)", block)
        fn_m = re.search(r"\[\d+/\d+\] (.+\.txt)", block)

        if not (s_m and r_m and c_m):
            continue

        row_idx = list(map(int, r_m.group(1).split()))
        col_idx = list(map(int, c_m.group(1).split()))

        rows.append({
            "group"        : grp,
            "dev"          : dev,
            "s"            : int(s_m.group(1)),
            "row_idx"      : tuple(row_idx),
            "col_idx"      : tuple(col_idx),
            "time_ms"      : float(t_m.group(1))  if t_m  else None,
            "minors_tested": float(mn_m.group(1)) if mn_m else None,
            "matrix"       : fn_m.group(1).strip() if fn_m else None,
            "is_principal" : row_idx == col_idx,
        })

if not rows:
    print("[ERROR] No Zero Minor data found.")
    sys.exit(1)

df = pd.DataFrame(rows)
groups = sorted(df["group"].unique())
print(f"Loaded {len(df)} Zero Minor hits across groups: {[int(g) for g in groups]}")

# ---------------------------------------------------------------------------
# Step 2 — Aggregate matrix/zero_minor totals from the same whitelisted files
# ---------------------------------------------------------------------------
summary_rows = []
for grp, dev, path in iter_summary_files():
    text = path.read_text(errors="replace")
    mat_m = re.search(r"Matrices\s*:\s*(\d+)", text)
    zm_m  = re.search(r"Zero minors\s*:\s*(\d+)", text)
    if mat_m and zm_m:
        summary_rows.append({
            "group"       : grp,
            "dev"         : dev,
            "matrices"    : int(mat_m.group(1)),
            "zero_minors" : int(zm_m.group(1)),
        })

df_summary = (
    pd.DataFrame(summary_rows)
    .groupby(["group", "dev"], as_index=False)
    .sum()
)
if not df_summary.empty:
    df_summary["hit_ratio"] = df_summary["zero_minors"] / df_summary["matrices"]

# ---------------------------------------------------------------------------
# GLOBAL PLOTS
# ---------------------------------------------------------------------------

# ---- 0a. KEY FINDING: Min deviation for 100% hit rate (step function) --
print("\n=== GLOBAL PLOTS ===")
print("0a. Key finding — min deviation for 100% hit rate...")
if not df_summary.empty:
    # Find the minimum deviation where hit_ratio == 1.0 for each group
    full_hit = df_summary[df_summary["hit_ratio"] >= 1.0]
    min_full  = full_hit.groupby("group")["dev"].min().reset_index()
    min_full.columns = ["group", "min_dev_100pct"]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(min_full["group"], min_full["min_dev_100pct"],
                  color=["#2c7bb6" if g <= 32 else "#d7191c" for g in min_full["group"]],
                  edgecolor="black", width=0.7)
    # Annotate bars
    for bar, (_, row) in zip(bars, min_full.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"dev={int(row['min_dev_100pct'])}",
                ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xlabel("Group (prime bit-size)")
    ax.set_ylabel("Minimum Deviation for 100% Hit Rate")
    ax.set_title("Minimum Deviation Guaranteeing a Zero Minor in Every Matrix\n"
                 "(Blue = groups 25–32 at dev 4;  Red = groups 33–40 at dev 5)")
    ax.set_xticks(min_full["group"])
    ax.set_yticks(sorted(min_full["min_dev_100pct"].unique()))
    ax.set_ylim(0, min_full["min_dev_100pct"].max() + 1)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    # Draw threshold line
    ax.axhline(y=4, color="#2c7bb6", linestyle="--", alpha=0.6, label="dev = 4 (groups 25–32)")
    ax.axhline(y=5, color="#d7191c", linestyle="--", alpha=0.6, label="dev = 5 (groups 33–40)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, GLOBAL_DIR / "0a_key_finding_min_dev_100pct.png")

# ---- 0b. Hit rate heatmap: group × deviation ----------------------------
print("0b. Hit rate heatmap (group × deviation)...")
if not df_summary.empty:
    pivot = df_summary.pivot(index="group", columns="dev", values="hit_ratio").fillna(0)
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"dev {d}" for d in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"Group {g}" for g in pivot.index])
    ax.set_xlabel("Deviation")
    ax.set_ylabel("Group (prime bit-size)")
    ax.set_title("Hit Rate Heatmap (green = 100%, red = 0%)\nEach cell: zero minors found / matrices tested")
    # Annotate each cell with the ratio
    for i, grp in enumerate(pivot.index):
        for j, dev in enumerate(pivot.columns):
            val = pivot.loc[grp, dev]
            ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                    fontsize=8, color="black" if 0.2 < val < 0.85 else "white")
    plt.colorbar(im, ax=ax, label="Hit Ratio")
    fig.tight_layout()
    save(fig, GLOBAL_DIR / "0b_hit_rate_heatmap.png")

# ---- 1. Min deviation per group (meaningful threshold) -----------------
# Use hit_ratio >= 0.10 so 1-8 stray hits at dev 2 don't distort the curve.
print("1. Min deviation per group (hit_ratio >= 10%)...")
if not df_summary.empty:
    meaningful = df_summary[df_summary["hit_ratio"] >= 0.10]
    min_dev_meaningful = meaningful.groupby("group")["dev"].min().reset_index()
    min_dev_meaningful.columns = ["group", "dev"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(min_dev_meaningful["group"], min_dev_meaningful["dev"],
            marker="o", linewidth=2, color="#2c7bb6")
    for _, row in min_dev_meaningful.iterrows():
        ax.annotate(f"dev {int(row['dev'])}",
                    xy=(row["group"], row["dev"]),
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=7, color="#555")
    ax.set_xlabel("Group (prime bit-size)")
    ax.set_ylabel("Minimum Deviation (≥ 10% hit rate)")
    ax.set_title("Minimum Meaningful Deviation per Group\n"
                 "(first deviation where ≥10% of matrices have a zero minor)")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    save(fig, GLOBAL_DIR / "1_min_dev_per_group.png")
# ---- 0c. Overall row-index heatmap: group × matrix index ---------------
print("0c. Overall row-index frequency heatmap...")
# Build a 2D array: rows = groups, cols = matrix index 0..max_idx
max_idx = max(i for tup in df["row_idx"] for i in tup)
grp_list = sorted(df["group"].unique())
heat = pd.DataFrame(0, index=grp_list,
                    columns=range(max_idx + 1), dtype=float)
for grp in grp_list:
    sub = df[df["group"] == grp]
    all_idx = [i for tup in sub["row_idx"] for i in tup]
    counts = Counter(all_idx)
    total = len(sub)  # normalise by number of hits
    for idx, cnt in counts.items():
        heat.loc[grp, idx] = cnt / total  # fraction of hits containing this index

fig, ax = plt.subplots(figsize=(16, 7))
im = ax.imshow(heat.values, aspect="auto", cmap="YlOrRd", vmin=0)
ax.set_yticks(range(len(grp_list)))
ax.set_yticklabels([f"G{g}" for g in grp_list], fontsize=8)
ax.set_xlabel("Matrix Row Index")
ax.set_ylabel("Group")
ax.set_title("Row Index Frequency Heatmap (across all deviations)\n"
             "Each cell = fraction of hits for that group containing this index")
plt.colorbar(im, ax=ax, label="Fraction of hits")
fig.tight_layout()
save(fig, GLOBAL_DIR / "0c_row_index_heatmap.png")

print("2. Hit ratio by deviation...")
if not df_summary.empty:
    fig, ax = plt.subplots(figsize=(10, 5))
    for grp in groups:
        sub = df_summary[df_summary["group"] == grp].sort_values("dev")
        if sub.empty:
            continue
        ax.plot(sub["dev"], sub["hit_ratio"], marker="o", label=f"Group {grp}")
    ax.set_xlabel("Deviation")
    ax.set_ylabel("Hit Ratio  (zero minors / matrices)")
    ax.set_title("Hit Ratio by Deviation Level, per Group")
    ax.legend(title="Group", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    save(fig, GLOBAL_DIR / "2_hit_ratio_by_deviation.png")
else:
    print("  [SKIP] No aggregated matrix/zero_minor counts found.")

# ---- 6. Time to first hit — global boxplot ----------------------------
print("6. Time to first hit (boxplot)...")
df_t = df.dropna(subset=["time_ms"])
if not df_t.empty:
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=df_t, x="group", y="time_ms", hue="dev", palette="Set2", ax=ax)
    if df_t["time_ms"].max() / (df_t["time_ms"].min() + 1e-9) > 100:
        ax.set_yscale("log")
        ax.set_ylabel("Time to First Hit (ms) — log scale")
    else:
        ax.set_ylabel("Time to First Hit (ms)")
    ax.set_xlabel("Group")
    ax.set_title("Time to Find First Zero Minor Hit, by Group and Deviation")
    ax.legend(title="Deviation", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    save(fig, GLOBAL_DIR / "6_time_to_hit_boxplot.png")
else:
    print("  [SKIP] No time_ms data.")

# ---- 7. Time vs Group scatter ------------------------------------------
print("7. Time vs Group (scatter)...")
if not df_t.empty:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df_t, x="group", y="time_ms", hue="dev",
                    palette="viridis", alpha=0.65, s=40, ax=ax)
    if df_t["time_ms"].max() / (df_t["time_ms"].min() + 1e-9) > 100:
        ax.set_yscale("log")
        ax.set_ylabel("Time (ms) — log scale")
    else:
        ax.set_ylabel("Time (ms)")
    ax.set_xlabel("Group (prime bit-size)")
    ax.set_title("Time to First Hit vs Prime Group")
    ax.legend(title="Deviation", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    save(fig, GLOBAL_DIR / "7_scatter_time_vs_group.png")

# ---- 8. Time vs Minors Tested scatter ---------------------------------
print("8. Time vs Minors Tested (scatter)...")
df_tm = df.dropna(subset=["time_ms", "minors_tested"])
if not df_tm.empty:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df_tm, x="minors_tested", y="time_ms", hue="group",
                    palette="coolwarm", alpha=0.65, s=40, ax=ax)
    if df_tm["time_ms"].max() / (df_tm["time_ms"].min() + 1e-9) > 100:
        ax.set_yscale("log"); ax.set_ylabel("Time (ms) — log scale")
    else:
        ax.set_ylabel("Time (ms)")
    if df_tm["minors_tested"].max() / (df_tm["minors_tested"].min() + 1) > 100:
        ax.set_xscale("log"); ax.set_xlabel("Minors Tested — log scale")
    else:
        ax.set_xlabel("Minors Tested")
    ax.set_title("Time vs Number of Minors Tested")
    ax.legend(title="Group", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    save(fig, GLOBAL_DIR / "8_scatter_time_vs_minors.png")

# ---------------------------------------------------------------------------
# PER-GROUP / PER-DEVIATION PLOTS
# ---------------------------------------------------------------------------
print("\n=== PER-GROUP / PER-DEVIATION PLOTS ===")

for grp in groups:
    devs = sorted(df[df["group"] == grp]["dev"].unique())
    for dev in devs:
        sub = df[(df["group"] == grp) & (df["dev"] == dev)]
        if sub.empty:
            continue

        d = spec_dir(grp, dev)
        tag = f"Group {grp}  Dev {dev}"
        print(f"  Processing {tag}  ({len(sub)} hits)...")

        # ---- 3. Anchor s distribution (histogram, % of hits) -----------
        s_vals = sub["s"].values
        bins   = range(int(s_vals.min()), int(s_vals.max()) + 2)
        w      = [100.0 / len(sub)] * len(sub)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(s_vals, bins=list(bins), align="left", rwidth=0.8,
                color="#4e91d2", edgecolor="black", weights=w)
        ax.set_xlabel("Anchor  s")
        ax.set_ylabel("% of Hits")
        ax.set_ylim(0, 105)
        ax.set_title(f"{tag}: Distribution of Principal Anchor  s")
        if len(bins) > 1:
            ax.set_xticks(list(bins)[:-1])
        save(fig, d / "3_anchor_s.png")

        # ---- 4. Principal submatrix check (text file) ------------------
        pct = sub["is_principal"].mean() * 100
        prime = PRIMES.get(int(grp), "unknown")
        with open(d / "4_principal_check.txt", "w") as f:
            f.write(f"Group {grp}, Deviation {dev}\n")
            f.write("=" * 40 + "\n")
            f.write(f"Prime (p)          : {prime}\n")
            f.write(f"Prime bit-size     : {grp} bits\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total hits         : {len(sub)}\n")
            f.write(f"Principal (row==col): {sub['is_principal'].sum()}  ({pct:.2f}%)\n")
            f.write(f"Non-principal       : {(~sub['is_principal']).sum()}  ({100-pct:.2f}%)\n")

        # ---- 5. Index recurrence (bar) ----------------------------------
        all_idx = [i for tup in sub["row_idx"] for i in tup]
        counts  = Counter(all_idx)
        if counts:
            idx_k, idx_v = zip(*sorted(counts.items()))
            fig, ax = plt.subplots(figsize=(max(10, len(idx_k) // 2), 5))
            ax.bar(idx_k, idx_v, color="#e07b54", edgecolor="black")
            ax.set_xlabel("Matrix Row Index")
            ax.set_ylabel("Times Appearing in a Hit")
            ax.set_title(f"{tag}: Row Index Recurrence")
            if max(idx_k) - min(idx_k) <= 60:
                ax.set_xticks(range(min(idx_k), max(idx_k) + 1))
            fig.tight_layout()
            save(fig, d / "5_idx_recurrence.png")

        # ---- 9. Scatter: selected row vs col indices -------------------
        px, py = [], []
        for r_tup, c_tup in zip(sub["row_idx"], sub["col_idx"]):
            for r in r_tup:
                for c in c_tup:
                    px.append(c)
                    py.append(r)

        if px:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(px, py, alpha=0.12, s=14, color="#6a0dad")
            ax.set_xlabel("Column Index")
            ax.set_ylabel("Row Index")
            ax.invert_yaxis()
            ax.set_title(f"{tag}: Matrix Indices of Zero Minor Hits")
            ax.grid(True, linestyle="--", alpha=0.3)
            save(fig, d / "9_scatter_indices.png")

# ---------------------------------------------------------------------------
# EXPORT DATASETS
# ---------------------------------------------------------------------------
print("\n=== EXPORTING DATASETS ===")
DATA_DIR = BASE_DIR / "Analysis_Scripts" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Flatten row_idx / col_idx tuples to comma-separated strings for CSV
df_export = df.copy()
df_export["row_idx"] = df_export["row_idx"].apply(lambda t: ",".join(map(str, t)))
df_export["col_idx"] = df_export["col_idx"].apply(lambda t: ",".join(map(str, t)))
df_export["prime"]   = df_export["group"].map(PRIMES)

# 1. Full hits dataset (every Zero Minor hit as one row)
full_path = DATA_DIR / "all_hits.csv"
df_export.to_csv(full_path, index=False)
print(f"  saved -> Analysis_Scripts/data/all_hits.csv  ({len(df_export)} rows)")

# 2. Summary dataset (one row per group/deviation: hit count, time stats)
agg = (
    df.groupby(["group", "dev"])
    .agg(
        prime        = ("group", lambda x: PRIMES.get(int(x.iloc[0]), None)),
        total_hits   = ("s", "count"),
        time_min_ms  = ("time_ms", "min"),
        time_max_ms  = ("time_ms", "max"),
        time_mean_ms = ("time_ms", "mean"),
        minors_mean  = ("minors_tested", "mean"),
        principal_pct= ("is_principal", lambda x: round(x.mean() * 100, 2)),
        anchor_s_mode= ("s", lambda x: int(x.mode().iloc[0])),
    )
    .reset_index()
)
if not df_summary.empty:
    agg = agg.merge(
        df_summary[["group", "dev", "matrices", "zero_minors", "hit_ratio"]],
        on=["group", "dev"], how="left"
    )
summary_path = DATA_DIR / "summary_per_group_dev.csv"
agg.to_csv(summary_path, index=False)
print(f"  saved -> Analysis_Scripts/data/summary_per_group_dev.csv  ({len(agg)} rows)")

# 3. Per-group CSV files inside each group folder
for grp in sorted(df["group"].unique()):
    sub = df_export[df_export["group"] == grp]
    grp_path = DATA_DIR / f"group_{grp}_hits.csv"
    sub.to_csv(grp_path, index=False)
    print(f"  saved -> Analysis_Scripts/data/group_{grp}_hits.csv  ({len(sub)} rows)")

# ---------------------------------------------------------------------------
print(f"\n=== DONE — all plots in {OUTPUT_DIR.relative_to(BASE_DIR)} ===")
print(f"=== DONE — all data  in {DATA_DIR.relative_to(BASE_DIR)} ===\n")
