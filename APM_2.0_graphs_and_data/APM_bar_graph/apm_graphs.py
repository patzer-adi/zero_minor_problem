import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from math import comb
import re
import warnings
warnings.filterwarnings('ignore')

# ── 1. Parse Excel ──────────────────────────────────────────────────────────
xl   = pd.ExcelFile("/mnt/user-data/uploads/APM_Summary_2_.xlsx")
raw  = xl.parse("APM Summary", header=None)

# locate every block header row (contains "APM Data")
header_rows = raw[raw[0].astype(str).str.contains("APM Data", na=False)].index.tolist()

blocks = []
for i, hrow in enumerate(header_rows):
    title   = str(raw.iloc[hrow, 0])
    m       = re.search(r'Bit\s+(\d+).*?=\s*([\d,]+).*?n\s*=\s*(\d+)', title)
    bit     = int(m.group(1))
    prime   = int(m.group(2).replace(",",""))
    n       = int(m.group(3))

    # data rows start 2 after header (skip column-name row)
    start   = hrow + 2
    end     = header_rows[i+1] if i+1 < len(header_rows) else len(raw)

    chunk   = raw.iloc[start:end].dropna(subset=[0])
    chunk   = chunk[chunk[0].apply(lambda x: str(x).replace('.','',1).isdigit())].copy()
    chunk   = chunk.reset_index(drop=True)
    df = pd.DataFrame({
        "deviation":           chunk[0].astype(int).values,
        "minor_size":          chunk[1].astype(str).values,
        "hits":                chunk[2].astype(str).values,
        "total_minors_tested": chunk[3].astype(float).values,
        "total_zero_minors":   chunk[4].astype(float).values,
        "hit_ratio_pct":       chunk[5].astype(str).str.replace("%","").astype(float).values,
    })
    df["hits_num"]   = df["hits"].str.split("/").str[0].astype(int)
    df["avg_minors"] = df["total_minors_tested"] / 100.0   # avg per kernel
    df["bit"]        = bit
    df["prime"]      = prime
    df["n"]          = n
    blocks.append(df)

data = pd.concat(blocks, ignore_index=True)
bits_present = sorted(data["bit"].unique())
print(f"Loaded bits: {bits_present}")

# ── 2. Helper: worst-case minors for APM at given deviation on n×n matrix ──
def worst_case_apm(n, deviation):
    """
    APM with 'deviation' deviations on an n×n matrix A.
    Principal minor size pm goes from 2 to n-deviation.
    For each pm, we test C(n-pm, deviation)^2 almost-principal minors.
    Worst case = max over all pm of C(n-pm, deviation)^2
    Total worst case = sum over pm of C(n-pm, deviation)^2
    """
    total = 0
    worst = 0
    for pm in range(2, n - deviation + 1):
        c = comb(n - pm, deviation)
        combos = c * c
        total += combos
        worst = max(worst, combos)
    return total, worst

# ── 3. For each bit: find top-hit deviation, build table ────────────────────
print("\n=== Top-3 deviations per bit (by hit_ratio) ===")
summary_rows = []
for bit in bits_present:
    sub  = data[data["bit"] == bit].copy()
    n    = sub["n"].iloc[0]
    top3 = sub.nlargest(3, "hit_ratio_pct")[["deviation","minor_size","hits_num",
                                              "avg_minors","hit_ratio_pct"]]
    top_dev  = top3.iloc[0]["deviation"]
    top_devs = sorted(top3["deviation"].tolist())

    # also include top-1 ±1
    best_dev = int(top3.iloc[0]["deviation"])
    candidates = sorted(set([best_dev-1, best_dev, best_dev+1]) &
                        set(sub["deviation"].tolist()))

    for dev in candidates:
        row_data = sub[sub["deviation"]==dev].iloc[0]
        wc_total, wc_single = worst_case_apm(n, dev)
        # worst case for FULL matrix (n_full = 3*bit at offset=1)
        n_full = 3 * bit
        wc_full_total, _ = worst_case_apm(n_full, dev)
        summary_rows.append({
            "bit":           bit,
            "n (offset)":    n,
            "n_full":        n_full,
            "deviation":     dev,
            "minor_size":    row_data["minor_size"],
            "hits":          row_data["hits"],
            "avg_minors":    row_data["avg_minors"],
            "hit_ratio_%":   row_data["hit_ratio_pct"],
            "wc_offset_mat": wc_total,
            "wc_full_mat":   wc_full_total,
            "reduction":     f"{wc_full_total/wc_total:.1f}x" if wc_total>0 else "N/A"
        })

summary = pd.DataFrame(summary_rows)
print(summary[["bit","n (offset)","deviation","hits","avg_minors",
               "hit_ratio_%","wc_offset_mat","wc_full_mat","reduction"]].to_string(index=False))

# ── 4. GRAPH 1: Hit Ratio vs Deviation for all bits ────────────────────────
fig, axes = plt.subplots(3, 4, figsize=(18, 13))
axes = axes.flatten()
colors = plt.cm.tab10.colors

for idx, bit in enumerate(bits_present):
    ax  = axes[idx]
    sub = data[data["bit"]==bit].copy()
    n   = sub["n"].iloc[0]

    ax.bar(sub["deviation"], sub["hit_ratio_pct"],
           color=colors[idx % 10], alpha=0.75, edgecolor="black", linewidth=0.5)
    ax.plot(sub["deviation"], sub["hit_ratio_pct"],
            "k-o", markersize=4, linewidth=1)

    # mark top deviation
    best = sub.loc[sub["hit_ratio_pct"].idxmax()]
    ax.axvline(best["deviation"], color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax.annotate(f"peak\ndev={int(best['deviation'])}\n{best['hit_ratio_pct']:.0f}%",
                xy=(best["deviation"], best["hit_ratio_pct"]),
                xytext=(best["deviation"]+0.5, best["hit_ratio_pct"]-8),
                fontsize=7, color="red")

    ax.set_title(f"{bit}-bit  |  n={n}  |  matrix {n}×{n}", fontsize=9, fontweight="bold")
    ax.set_xlabel("Deviation", fontsize=8)
    ax.set_ylabel("Hit Ratio (%)", fontsize=8)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.tick_params(labelsize=7)

for idx in range(len(bits_present), len(axes)):
    axes[idx].set_visible(False)

fig.suptitle("APM Hit Ratio vs Deviation  —  Offset 0.2 Matrices", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/graph1_hit_ratio_vs_deviation.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved graph1_hit_ratio_vs_deviation.png")

# ── 5. GRAPH 2: Avg Minors Tested vs Deviation for all bits ─────────────────
fig, axes = plt.subplots(3, 4, figsize=(18, 13))
axes = axes.flatten()

for idx, bit in enumerate(bits_present):
    ax  = axes[idx]
    sub = data[data["bit"]==bit].copy()
    n   = sub["n"].iloc[0]

    ax.bar(sub["deviation"], sub["avg_minors"],
           color=colors[idx % 10], alpha=0.75, edgecolor="black", linewidth=0.5)

    best = sub.loc[sub["hit_ratio_pct"].idxmax()]
    ax.axvline(best["deviation"], color="red", linestyle="--", linewidth=1, alpha=0.8,
               label=f"best dev={int(best['deviation'])}")

    ax.set_title(f"{bit}-bit  |  n={n}  |  matrix {n}×{n}", fontsize=9, fontweight="bold")
    ax.set_xlabel("Deviation", fontsize=8)
    ax.set_ylabel("Avg Minors Tested / kernel", fontsize=8)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e9:.1f}B" if x>=1e9 else
                                                                     f"{x/1e6:.1f}M" if x>=1e6 else
                                                                     f"{x/1e3:.0f}K"))
    ax.legend(fontsize=7)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.tick_params(labelsize=7)

for idx in range(len(bits_present), len(axes)):
    axes[idx].set_visible(False)

fig.suptitle("APM Avg Minors Tested per Kernel vs Deviation  —  Offset 0.2 Matrices",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/graph2_avg_minors_vs_deviation.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved graph2_avg_minors_vs_deviation.png")

# ── 6. GRAPH 3: Worst-case minors — offset matrix vs full matrix ─────────────
fig, ax = plt.subplots(figsize=(13, 7))

for bit in bits_present:
    sub = data[data["bit"]==bit]
    n   = sub["n"].iloc[0]
    best_dev = int(sub.loc[sub["hit_ratio_pct"].idxmax(), "deviation"])

    deviations = range(2, n)
    wc_offset  = [worst_case_apm(n, d)[0] for d in deviations]
    n_full     = 3 * bit
    wc_full    = [worst_case_apm(n_full, d)[0] for d in deviations]

    ax.semilogy(list(deviations), wc_offset, "-o", markersize=4, linewidth=1.5,
                label=f"{bit}-bit offset(n={n})")

ax.set_xlabel("Deviation", fontsize=11)
ax.set_ylabel("Worst-case total minors (log scale)", fontsize=11)
ax.set_title("Worst-case APM minors: offset 0.2 matrix vs deviation", fontsize=13, fontweight="bold")
ax.legend(fontsize=8, ncol=2)
ax.grid(True, which="both", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/graph3_worst_case_offset.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved graph3_worst_case_offset.png")

# ── 7. GRAPH 4: Offset matrix vs Full matrix worst case comparison ───────────
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()

for idx, bit in enumerate(bits_present[:8]):
    ax     = axes[idx]
    sub    = data[data["bit"]==bit]
    n      = sub["n"].iloc[0]
    n_full = 3 * bit

    deviations = list(range(2, min(n, 10)))
    wc_off  = [float(worst_case_apm(n, d)[0]) for d in deviations]
    wc_full = [float(min(worst_case_apm(n_full, d)[0], 1e18)) for d in deviations]

    x = np.arange(len(deviations))
    w = 0.35
    ax.bar(x - w/2, wc_off,  w, label=f"offset n={n}",    color="#4C72B0", alpha=0.8)
    ax.bar(x + w/2, wc_full, w, label=f"full n={n_full}", color="#DD8452", alpha=0.8)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(deviations, fontsize=7)
    ax.set_title(f"{bit}-bit", fontsize=9, fontweight="bold")
    ax.set_xlabel("Deviation", fontsize=8)
    ax.set_ylabel("Worst-case minors", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.tick_params(labelsize=7)

for idx in range(len(bits_present[:8]), len(axes)):
    axes[idx].set_visible(False)

fig.suptitle("Worst-case minors: offset 0.2 matrix (n) vs full matrix (3n)  —  log scale",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/graph4_offset_vs_full_worstcase.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved graph4_offset_vs_full_worstcase.png")

# ── 8. GRAPH 5: Top deviation summary across all bits ───────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

bits_list, best_devs, best_ratios, best_avg_minors = [], [], [], []
for bit in bits_present:
    sub      = data[data["bit"]==bit]
    best_row = sub.loc[sub["hit_ratio_pct"].idxmax()]
    bits_list.append(bit)
    best_devs.append(int(best_row["deviation"]))
    best_ratios.append(best_row["hit_ratio_pct"])
    best_avg_minors.append(best_row["avg_minors"])

x  = np.arange(len(bits_list))
w  = 0.35
b1 = ax.bar(x - w/2, best_ratios, w, color="#4C72B0", alpha=0.85, label="Hit ratio at best deviation (%)")
ax2 = ax.twinx()
b2 = ax2.bar(x + w/2, best_devs, w, color="#DD8452", alpha=0.85, label="Best deviation value")

ax.set_xticks(x)
ax.set_xticklabels([f"{b}-bit\nn={data[data['bit']==b]['n'].iloc[0]}" for b in bits_list], fontsize=9)
ax.set_ylabel("Hit Ratio at best deviation (%)", fontsize=10, color="#4C72B0")
ax2.set_ylabel("Best deviation value", fontsize=10, color="#DD8452")
ax.set_title("Best deviation and hit ratio across all bit sizes  —  offset 0.2 matrices",
             fontsize=12, fontweight="bold")

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, labels1+labels2, fontsize=9, loc="upper left")
ax.grid(axis="y", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/graph5_best_deviation_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved graph5_best_deviation_summary.png")

# ── 9. Save summary table as CSV ────────────────────────────────────────────
summary.to_csv("/mnt/user-data/outputs/apm_summary_table.csv", index=False)
print("Saved apm_summary_table.csv")
print("\nDone. All outputs in /mnt/user-data/outputs/")
