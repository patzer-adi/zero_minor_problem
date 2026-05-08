"""
Generate 3 clean graphs for APM 2.0 data from APM_2.0_Summary.xlsx:
  1. Heatmap       — Hit Ratio by (Prime Bit Group × Deviation/Minor Size)
  2. Multi-Line    — Hit Ratio vs Deviation, one line per group
  3. Grouped Bar   — Hit Ratio per Deviation per Group

Output: APM_2.0_graphs_and_data/graph_2.0/
"""

import os, re, warnings
import numpy as np
import openpyxl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
BASE    = os.path.dirname(os.path.abspath(__file__))
XLSX_IN = os.path.join(BASE, "APM_2.0_Summary.xlsx")
OUT_DIR = os.path.join(BASE, "graph_2.0")
os.makedirs(OUT_DIR, exist_ok=True)

# ── parse Excel ───────────────────────────────────────────────────────────────
wb = openpyxl.load_workbook(XLSX_IN)
ws = wb.active

rows = []
current_group = None
for row in ws.iter_rows(values_only=True):
    a = row[0]
    # title row → extract group
    if isinstance(a, str) and "Prime Bit" in a:
        m = re.search(r"Prime Bit (\d+)", a)
        if m:
            current_group = int(m.group(1))
    # data row → integer deviation in col A
    elif current_group and isinstance(a, int):
        dev, minor, hits_str, total, zeros, ratio_str = row
        # parse hit ratio: "54.00%" → 54.0
        ratio = float(str(ratio_str).replace("%", "").strip())
        rows.append({
            "group":     current_group,
            "dev":       dev,
            "minor":     minor,     # e.g. "8×8"
            "hit_ratio": ratio,
        })

groups = sorted(set(r["group"] for r in rows))
devs   = sorted(set(r["dev"]   for r in rows))
lookup = {(r["group"], r["dev"]): r for r in rows}

# ── shared style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.grid":         True,
    "grid.color":        "#cccccc",
    "grid.linewidth":    0.6,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})

blue_cmap = plt.cm.get_cmap("Blues")

# ══════════════════════════════════════════════════════════════════════════════
# GRAPH 1 — HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
print("Graph 1: Heatmap …")

heat = np.full((len(groups), len(devs)), np.nan)
for gi, g in enumerate(groups):
    for di, d in enumerate(devs):
        if (g, d) in lookup:
            heat[gi, di] = lookup[(g, d)]["hit_ratio"]

fig, ax = plt.subplots(figsize=(14, 7))

im = ax.imshow(heat, aspect="auto", cmap="Blues", vmin=0, vmax=100,
               origin="upper", interpolation="nearest")

for gi in range(len(groups)):
    for di in range(len(devs)):
        val = heat[gi, di]
        if not np.isnan(val):
            color = "white" if val > 60 else "#1a1a1a"
            ax.text(di, gi, f"{val:.0f}%", ha="center", va="center",
                    fontsize=7.5, color=color)
        else:
            ax.text(di, gi, "—", ha="center", va="center",
                    fontsize=9, color="#aaaaaa")

# x-axis: show deviation + minor size label
xlabels = []
for d in devs:
    # find a minor label for this dev from any group
    for g in groups:
        if (g, d) in lookup:
            xlabels.append(f"Dev {d}\n({lookup[(g,d)]['minor']})")
            break
    else:
        xlabels.append(f"Dev {d}")

ax.set_xticks(range(len(devs)))
ax.set_xticklabels(xlabels, fontsize=8)
ax.set_yticks(range(len(groups)))
ax.set_yticklabels([f"Bit {g}" for g in groups], fontsize=9)

cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
cbar.set_label("Hit Ratio (%)", fontsize=10)

ax.set_title("APM 2.0 — Hit Ratio Heatmap\n(Prime Bit Group × Deviation / Minor Size)",
             fontsize=13, pad=12)
ax.set_xlabel("Deviation (Minor Size)", fontsize=11)
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

shades_vals = np.linspace(0.3, 0.9, len(groups))

fig, ax = plt.subplots(figsize=(13, 6.5))

for i, g in enumerate(groups):
    xs = [d for d in devs if (g, d) in lookup]
    ys = [lookup[(g, d)]["hit_ratio"] for d in xs]
    if not xs:
        continue
    c = blue_cmap(shades_vals[i])
    ax.plot(xs, ys, "o-", color=c, linewidth=1.7, markersize=5,
            label=f"Bit {g}")

ax.axhline(100, color="black", linewidth=1.0, linestyle="--", alpha=0.35,
           label="100%")

ax.set_xticks(devs)
# show minor size on x-axis too
minor_labels = []
for d in devs:
    for g in groups:
        if (g, d) in lookup:
            minor_labels.append(f"{d}\n({lookup[(g,d)]['minor']})")
            break
    else:
        minor_labels.append(str(d))

ax.set_xticklabels(minor_labels, fontsize=8)
ax.set_ylim(-5, 110)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))

ax.set_title("APM 2.0 — Hit Ratio vs Deviation (per Prime Bit Group)", fontsize=13, pad=12)
ax.set_xlabel("Deviation (Minor Size)", fontsize=11)
ax.set_ylabel("Hit Ratio (%)", fontsize=11)
ax.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.5)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "2_multiline_hit_ratio_vs_deviation.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print("  ✓ 2_multiline_hit_ratio_vs_deviation.png")

# ══════════════════════════════════════════════════════════════════════════════
# GRAPH 3 — GROUPED BAR CHART
# ══════════════════════════════════════════════════════════════════════════════
print("Graph 3: Grouped bar chart …")

# Too many devs for grouped bars — use one bar per group, x=deviation
# Plot each group as a separate line-of-bars; groups on x, devs as bar groups
# Since there are up to 16 devs, cluster by group with bar per dev subset
# Strategy: one sub-plot per group (small multiples), clean and readable

n_groups = len(groups)
ncols = 4
nrows = (n_groups + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.2),
                         sharey=True)
axes = axes.flatten()

bar_shades = np.linspace(0.25, 0.85, len(devs))
dev_to_shade = {d: blue_cmap(bar_shades[i]) for i, d in enumerate(devs)}

for idx, g in enumerate(groups):
    ax = axes[idx]
    g_rows = [(r["dev"], r["hit_ratio"]) for r in rows if r["group"] == g]
    g_devs = [x[0] for x in g_rows]
    g_hits = [x[1] for x in g_rows]
    colors = [dev_to_shade[d] for d in g_devs]

    bars = ax.bar(range(len(g_devs)), g_hits, color=colors,
                  edgecolor="white", linewidth=0.5)
    ax.axhline(100, color="black", linewidth=0.8, linestyle="--", alpha=0.3)

    ax.set_xticks(range(len(g_devs)))
    ax.set_xticklabels([str(d) for d in g_devs], fontsize=7)
    ax.set_ylim(0, 110)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_title(f"Bit {g}", fontsize=10, fontweight="bold", pad=4)
    ax.set_xlabel("Deviation", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", color="#cccccc", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# hide unused subplots
for idx in range(len(groups), len(axes)):
    axes[idx].set_visible(False)

fig.suptitle("APM 2.0 — Hit Ratio per Deviation (per Prime Bit Group)",
             fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout(pad=1.5)
fig.savefig(os.path.join(OUT_DIR, "3_bar_hit_ratio_per_group.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print("  ✓ 3_bar_hit_ratio_per_group.png")

print(f"\nDone → {OUT_DIR}")
