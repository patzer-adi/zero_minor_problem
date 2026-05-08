"""
Generate 3 clean, simple graphs for APM 3.5 data.
Output: APM_2.0_graphs_and_data/graph_3.5/
"""

import os, csv, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
BASE    = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.path.dirname(BASE)
CSV_IN  = os.path.join(ROOT, "Analysis_Scripts", "data", "summary_per_group_dev.csv")
OUT_DIR = os.path.join(BASE, "graph_3.5")
os.makedirs(OUT_DIR, exist_ok=True)

# ── load data ─────────────────────────────────────────────────────────────────
rows = []
with open(CSV_IN, newline="") as f:
    for r in csv.DictReader(f):
        rows.append({
            "group":     int(r["group"]),
            "dev":       int(r["dev"]),
            "hits":      int(r["total_hits"]),
            "total":     int(r["matrices"]),
            "hit_ratio": float(r["hit_ratio"]) * 100,
        })

groups = sorted(set(r["group"] for r in rows))
devs   = sorted(set(r["dev"]   for r in rows))
lookup = {(r["group"], r["dev"]): r for r in rows}

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       10,
    "axes.grid":       True,
    "grid.color":      "#cccccc",
    "grid.linewidth":  0.6,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":      150,
})

# ══════════════════════════════════════════════════════════════════════════════
# GRAPH 1 — HEATMAP (grayscale)
# ══════════════════════════════════════════════════════════════════════════════
print("Graph 1: Heatmap …")

heat = np.full((len(groups), len(devs)), np.nan)
for gi, g in enumerate(groups):
    for di, d in enumerate(devs):
        if (g, d) in lookup:
            heat[gi, di] = lookup[(g, d)]["hit_ratio"]

fig, ax = plt.subplots(figsize=(8, 7))

im = ax.imshow(heat, aspect="auto", cmap="Blues", vmin=0, vmax=100,
               origin="upper", interpolation="nearest")

for gi in range(len(groups)):
    for di in range(len(devs)):
        val = heat[gi, di]
        if not np.isnan(val):
            color = "white" if val > 60 else "#1a1a1a"
            ax.text(di, gi, f"{val:.0f}%", ha="center", va="center",
                    fontsize=8.5, color=color)
        else:
            ax.text(di, gi, "—", ha="center", va="center",
                    fontsize=9, color="#aaaaaa")

ax.set_xticks(range(len(devs)))
ax.set_xticklabels([f"Dev {d}" for d in devs], fontsize=10)
ax.set_yticks(range(len(groups)))
ax.set_yticklabels([f"Bit {g}" for g in groups], fontsize=9)

cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Hit Ratio (%)", fontsize=10)

ax.set_title("Hit Ratio — Prime Bit Group × Deviation", fontsize=13, pad=12)
ax.set_xlabel("Deviation", fontsize=11)
ax.set_ylabel("Prime Bit Group", fontsize=11)
ax.grid(False)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "1_heatmap_hit_ratio.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print("  ✓ 1_heatmap_hit_ratio.png")

# ══════════════════════════════════════════════════════════════════════════════
# GRAPH 2 — MULTI-LINE CHART
# ══════════════════════════════════════════════════════════════════════════════
print("Graph 2: Multi-line chart …")

fig, ax = plt.subplots(figsize=(10, 6))

# blue shades from light to dark
blue_cmap = plt.cm.get_cmap("Blues")
shades_vals = np.linspace(0.3, 0.9, len(groups))

for i, g in enumerate(groups):
    xs = [d for d in devs if (g, d) in lookup]
    ys = [lookup[(g, d)]["hit_ratio"] for d in xs]
    if not xs:
        continue
    c = blue_cmap(shades_vals[i])
    ax.plot(xs, ys, "o-", color=c, linewidth=1.6, markersize=5,
            label=f"Bit {g}")

ax.axhline(100, color="black", linewidth=1.0, linestyle="--", alpha=0.4,
           label="100%")

ax.set_xticks(devs)
ax.set_xticklabels([f"Dev {d}" for d in devs])
ax.set_ylim(-5, 110)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))

ax.set_title("Hit Ratio vs Deviation (per Prime Bit Group)", fontsize=13, pad=12)
ax.set_xlabel("Deviation", fontsize=11)
ax.set_ylabel("Hit Ratio (%)", fontsize=11)
ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.5)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "2_multiline_hit_ratio_vs_deviation.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print("  ✓ 2_multiline_hit_ratio_vs_deviation.png")

# ══════════════════════════════════════════════════════════════════════════════
# GRAPH 3 — GROUPED BAR CHART
# ══════════════════════════════════════════════════════════════════════════════
print("Graph 3: Grouped bar chart …")

blue_bar_cmap = plt.cm.get_cmap("Blues")
bar_shades = np.linspace(0.3, 0.85, len(devs))

fig, ax = plt.subplots(figsize=(13, 6))

bar_w      = 0.18
n_devs     = len(devs)
x_centers  = np.arange(len(groups))

for di, d in enumerate(devs):
    offsets = x_centers + (di - n_devs / 2 + 0.5) * bar_w
    heights = [lookup[(g, d)]["hit_ratio"] if (g, d) in lookup else 0
               for g in groups]
    c = blue_bar_cmap(bar_shades[di])
    ax.bar(offsets, heights, width=bar_w * 0.85,
           color=c, label=f"Dev {d}",
           edgecolor="white", linewidth=0.5)

ax.axhline(100, color="black", linewidth=0.9, linestyle="--", alpha=0.4)

ax.set_xticks(x_centers)
ax.set_xticklabels([f"Bit {g}" for g in groups], rotation=30, ha="right")
ax.set_ylim(0, 115)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))

ax.set_title("Hit Ratio per Group × Deviation", fontsize=13, pad=12)
ax.set_xlabel("Prime Bit Group", fontsize=11)
ax.set_ylabel("Hit Ratio (%)", fontsize=11)
ax.legend(loc="upper right", fontsize=9, framealpha=0.5)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "3_grouped_bar_hit_ratio.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print("  ✓ 3_grouped_bar_hit_ratio.png")

print(f"\nDone → {OUT_DIR}")
