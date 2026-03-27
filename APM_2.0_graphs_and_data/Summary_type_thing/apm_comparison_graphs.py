"""
APM Comparison Line Graphs
==========================
For each bit: shows the top deviation (best hit ratio) and its ±1 neighbours.

For each of those 3 deviations the script plots:
  1. Hit ratio (%)                      — from the Excel (APM_Summary data)
  2. Avg minors tested / kernel         — from the Excel
  3. Worst-case minors: offset matrix   — C(n_offset, dev+2)  (professor's formula)
  4. Worst-case minors: full matrix     — C(n_full,   dev+2)  (professor's formula)

All four panels are combined per-bit into one figure.
A final summary figure overlays all bits on one chart.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import re
import warnings
from math import comb

warnings.filterwarnings("ignore")

# ── 1. Load offset-matrix summary (already computed) ─────────────────────────
csv_path  = "/mnt/user-data/uploads/apm_summary_table.csv"
xlsx_path = "/mnt/user-data/uploads/APM_Summary_2_.xlsx"

offset_df = pd.read_csv(csv_path)

# ── 2. Parse full per-deviation data from Excel ───────────────────────────────
xl  = pd.ExcelFile(xlsx_path)
raw = xl.parse("APM Summary", header=None)

header_rows = raw[raw[0].astype(str).str.contains("APM Data", na=False)].index.tolist()

blocks = []
for i, hrow in enumerate(header_rows):
    title = str(raw.iloc[hrow, 0])
    m     = re.search(r"Bit\s+(\d+).*?=\s*([\d,]+).*?n\s*=\s*(\d+)", title)
    bit   = int(m.group(1))
    prime = int(m.group(2).replace(",", ""))
    n     = int(m.group(3))

    start = hrow + 2
    end   = header_rows[i + 1] if i + 1 < len(header_rows) else len(raw)
    chunk = raw.iloc[start:end].dropna(subset=[0])
    chunk = chunk[chunk[0].apply(lambda x: str(x).replace(".", "", 1).isdigit())].copy()
    chunk = chunk.reset_index(drop=True)

    df = pd.DataFrame({
        "deviation":      chunk[0].astype(int).values,
        "minor_size":     chunk[1].astype(str).values,
        "hits_str":       chunk[2].astype(str).values,
        "total_minors":   chunk[3].astype(float).values,
        "hit_ratio_pct":  chunk[5].astype(str).str.replace("%", "").astype(float).values,
    })
    df["hits_num"]   = df["hits_str"].str.split("/").str[0].astype(int)
    df["avg_minors"] = df["total_minors"] / 100.0
    df["bit"]        = bit
    df["prime"]      = prime
    df["n"]          = n
    df["n_full"]     = 3 * bit
    blocks.append(df)

all_data = pd.concat(blocks, ignore_index=True)
bits     = sorted(all_data["bit"].unique())

# ── 3. Professor's worst-case formula: C(n, deviation + 2) ───────────────────
#       Reasoning: smallest principal minor = 2×2, so you choose
#       (deviation + 2) row/column indices from n available.
def wc_simple(n, deviation):
    k = deviation + 2
    if k > n:
        return 0
    return comb(n, k)

# ── 4. Build per-bit summary (top deviation ± 1) ─────────────────────────────
summary_rows = []
for bit in bits:
    sub = all_data[all_data["bit"] == bit].copy()
    n   = int(sub["n"].iloc[0])
    n_f = int(sub["n_full"].iloc[0])
    prime = int(sub["prime"].iloc[0])

    best_dev = int(sub.loc[sub["hit_ratio_pct"].idxmax(), "deviation"])
    candidates = sorted(
        set([best_dev - 1, best_dev, best_dev + 1]) & set(sub["deviation"].tolist())
    )

    for dev in candidates:
        row = sub[sub["deviation"] == dev].iloc[0]
        summary_rows.append({
            "bit":          bit,
            "prime":        prime,
            "n_offset":     n,
            "n_full":       n_f,
            "deviation":    dev,
            "is_best":      (dev == best_dev),
            "hits":         row["hits_str"],
            "hit_ratio_%":  row["hit_ratio_pct"],
            "avg_minors":   row["avg_minors"],
            "wc_offset":    wc_simple(n, dev),
            "wc_full":      wc_simple(n_f, dev),
        })

summary = pd.DataFrame(summary_rows)
print("=== Summary table (top dev ±1) ===")
print(summary[["bit", "prime", "n_offset", "n_full", "deviation",
               "hits", "hit_ratio_%", "avg_minors", "wc_offset", "wc_full"]].to_string(index=False))


# ── 5. Colour palette ─────────────────────────────────────────────────────────
COLORS = plt.cm.tab20.colors

# ── 6. Per-bit 4-panel figure ─────────────────────────────────────────────────
print("\nGenerating per-bit figures …")

for bit in bits:
    sub = summary[summary["bit"] == bit].sort_values("deviation")
    if sub.empty:
        continue

    devs        = sub["deviation"].tolist()
    hit_ratios  = sub["hit_ratio_%"].tolist()
    avg_minors  = sub["avg_minors"].tolist()
    wc_offset   = sub["wc_offset"].tolist()
    wc_full     = sub["wc_full"].tolist()
    prime       = int(sub["prime"].iloc[0])
    n_off       = int(sub["n_offset"].iloc[0])
    n_full      = int(sub["n_full"].iloc[0])
    best_dev    = int(sub[sub["is_best"]]["deviation"].iloc[0])

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"{bit}-bit  |  Prime = {prime:,}  |  Offset matrix n={n_off}×{n_off}  |  Full matrix n={n_full}×{n_full}",
        fontsize=12, fontweight="bold", y=1.01,
    )
    color = COLORS[bits.index(bit) % len(COLORS)]

    # ── Panel A: Hit Ratio ────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(devs, hit_ratios, "-o", color=color, linewidth=2.5, markersize=9, zorder=3)
    for d, v in zip(devs, hit_ratios):
        ax.annotate(f"{v:.0f}%", (d, v), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9, fontweight="bold")
    ax.axvline(best_dev, color="red", linestyle="--", linewidth=1.5,
               label=f"best dev={best_dev}", alpha=0.8)
    ax.set_title("Hit Ratio (%)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Deviation", fontsize=9)
    ax.set_ylabel("Hit Ratio (%)", fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_xticks(devs)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

    # ── Panel B: Avg Minors Tested ────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(devs, avg_minors, "-s", color=color, linewidth=2.5, markersize=9, zorder=3)
    for d, v in zip(devs, avg_minors):
        lbl = (f"{v/1e9:.2f}B" if v >= 1e9 else
               f"{v/1e6:.1f}M" if v >= 1e6 else f"{v/1e3:.0f}K")
        ax.annotate(lbl, (d, v), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9, fontweight="bold")
    ax.axvline(best_dev, color="red", linestyle="--", linewidth=1.5,
               label=f"best dev={best_dev}", alpha=0.8)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: (f"{x/1e9:.1f}B" if x >= 1e9 else
                      f"{x/1e6:.0f}M" if x >= 1e6 else f"{x/1e3:.0f}K")))
    ax.set_title("Avg Minors Tested / Kernel", fontsize=10, fontweight="bold")
    ax.set_xlabel("Deviation", fontsize=9)
    ax.set_ylabel("Avg Minors / kernel", fontsize=9)
    ax.set_xticks(devs)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

    # ── Panel C: Worst-case (offset vs full), log scale ──────────────────────
    ax = axes[1, 0]
    ax.semilogy(devs, wc_offset, "-o", color="#4C72B0", linewidth=2.5, markersize=9,
                label=f"Offset n={n_off}  →  C(n,dev+2)")
    ax.semilogy(devs, wc_full,   "-^", color="#DD8452", linewidth=2.5, markersize=9,
                label=f"Full   n={n_full}  →  C(n,dev+2)")
    ax.axvline(best_dev, color="red", linestyle="--", linewidth=1.5, alpha=0.8)

    for d, vo, vf in zip(devs, wc_offset, wc_full):
        ax.annotate(f"{vo:.1e}", (d, vo), textcoords="offset points",
                    xytext=(-20, 6), fontsize=7.5, color="#4C72B0")
        ax.annotate(f"{vf:.1e}", (d, vf), textcoords="offset points",
                    xytext=(-20, -14), fontsize=7.5, color="#DD8452")

    ax.set_title("Worst-case: C(n, dev+2)  —  offset vs full (log)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Deviation", fontsize=9)
    ax.set_ylabel("C(n, dev+2)  [log scale]", fontsize=9)
    ax.set_xticks(devs)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    # ── Panel D: Reduction factor ─────────────────────────────────────────────
    ax = axes[1, 1]
    reductions = [vf / vo if vo > 0 else 0 for vo, vf in zip(wc_offset, wc_full)]
    bars = ax.bar([str(d) for d in devs], reductions, color=color, alpha=0.8,
                  edgecolor="black", linewidth=0.7)
    for bar, r in zip(bars, reductions):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                f"{r:.1e}×", ha="center", va="bottom", fontsize=9, fontweight="bold")
    best_idx = devs.index(best_dev)
    bars[best_idx].set_edgecolor("red")
    bars[best_idx].set_linewidth(2.5)
    ax.set_title("Reduction Factor  (full / offset)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Deviation", fontsize=9)
    ax.set_ylabel("wc_full / wc_offset", fontsize=9)
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    out = f"/mnt/user-data/outputs/per_bit_{bit}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ── 7. Summary line graph: wc_offset & wc_full for all bits at best dev ───────
print("\nGenerating summary line graphs …")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Worst-case C(n, dev+2): offset vs full matrix — best deviation per bit",
             fontsize=13, fontweight="bold")

best_rows = summary[summary["is_best"]].sort_values("bit").reset_index(drop=True)

# left: log-scale line
ax = axes[0]
ax.semilogy(best_rows["bit"], best_rows["wc_offset"], "-o", color="#4C72B0",
            linewidth=2.5, markersize=9, label="Offset matrix  C(n_offset, dev+2)")
ax.semilogy(best_rows["bit"], best_rows["wc_full"],   "-^", color="#DD8452",
            linewidth=2.5, markersize=9, label="Full matrix    C(n_full, dev+2)")

for _, r in best_rows.iterrows():
    ax.annotate(f"dev={int(r['deviation'])}", (r["bit"], r["wc_offset"]),
                textcoords="offset points", xytext=(4, 6), fontsize=7.5, color="#4C72B0")

ax.set_xlabel("Bit size", fontsize=11)
ax.set_ylabel("C(n, dev+2)  [log scale]", fontsize=11)
ax.set_title("Worst-case minors at best deviation (log)", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, which="both", linestyle="--", alpha=0.4)

# right: hit ratio at best deviation
ax = axes[1]
ax.plot(best_rows["bit"], best_rows["hit_ratio_%"], "-D", color="green",
        linewidth=2.5, markersize=9, label="Hit Ratio at best deviation")
for _, r in best_rows.iterrows():
    ax.annotate(f"{r['hit_ratio_%']:.0f}%\ndev={int(r['deviation'])}",
                (r["bit"], r["hit_ratio_%"]),
                textcoords="offset points", xytext=(4, 5), fontsize=8, color="darkgreen")
ax.set_xlabel("Bit size", fontsize=11)
ax.set_ylabel("Hit Ratio (%)", fontsize=11)
ax.set_ylim(0, 105)
ax.set_title("Hit Ratio (%) at best deviation across all bits", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/summary_all_bits.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved summary_all_bits.png")


# ── 8. Multi-bit overlay: wc_offset vs wc_full across ±1 deviations ──────────
fig, ax = plt.subplots(figsize=(15, 7))

x_positions = []
x_labels    = []
x           = 0

for bit in bits:
    sub = summary[summary["bit"] == bit].sort_values("deviation")
    if sub.empty:
        continue
    color = COLORS[bits.index(bit) % len(COLORS)]
    devs_b = sub["deviation"].tolist()
    for j, (_, row) in enumerate(sub.iterrows()):
        marker = "*" if row["is_best"] else ("^" if j == len(devs_b) - 1 else "v")
        ax.semilogy(x, row["wc_offset"], marker, color=color, markersize=11 if row["is_best"] else 8,
                    zorder=3)
        ax.semilogy(x, row["wc_full"], "x", color=color, markersize=11 if row["is_best"] else 8,
                    markeredgewidth=2, zorder=3)
        x_positions.append(x)
        x_labels.append(f"{bit}b\ndev={int(row['deviation'])}" + ("★" if row["is_best"] else ""))
        x += 1
    x += 0.5  # gap between bit groups

# legend proxies
from matplotlib.lines import Line2D
legend_elems = [
    Line2D([0], [0], marker="*", color="gray", markersize=11, linestyle="None", label="Best dev (offset)"),
    Line2D([0], [0], marker="x", color="gray", markersize=10, markeredgewidth=2, linestyle="None", label="Best dev (full)"),
    Line2D([0], [0], marker="^", color="gray", markersize=8, linestyle="None", label="best+1 or best-1 (offset)"),
]
ax.legend(handles=legend_elems, fontsize=9, loc="upper left")
ax.set_xticks(x_positions)
ax.set_xticklabels(x_labels, fontsize=7, rotation=0)
ax.set_ylabel("C(n, dev+2)  [log scale]", fontsize=11)
ax.set_title("Worst-case C(n, dev+2) for best deviation ±1 — all bits\n"
             "Circles = offset matrix, × = full matrix, ★ = best deviation",
             fontsize=12, fontweight="bold")
ax.grid(True, which="both", linestyle="--", alpha=0.35)

# color legend per bit
for bit in bits:
    color = COLORS[bits.index(bit) % len(COLORS)]
    ax.plot([], [], "-", color=color, linewidth=3, label=f"{bit}-bit")
ax.legend(fontsize=8, ncol=4, loc="lower right")

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/overlay_all_bits_wc.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved overlay_all_bits_wc.png")


# ── 9. Save enhanced summary CSV ─────────────────────────────────────────────
summary.to_csv("/mnt/user-data/outputs/apm_top_devs_summary.csv", index=False)
print("  Saved apm_top_devs_summary.csv")
print("\nDone. All outputs in /mnt/user-data/outputs/")
