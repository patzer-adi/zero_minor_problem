from math import comb, floor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

# ── Data ────────────────────────────────────────────────────────────────────
data = [
    (25,  75, {0.20: 15, 0.25: 18, 0.30: 22, 0.35: 26, 0.40: 30, 0.50: 37}),
    (26,  78, {0.20: 15, 0.25: 19, 0.30: 23, 0.35: 27, 0.40: 31, 0.50: 39}),
    (27,  82, {0.20: 16, 0.25: 20, 0.30: 24, 0.35: 28, 0.40: 32, 0.50: 40}),
    (28,  89, {0.20: 16, 0.25: 21, 0.30: 25, 0.35: 29, 0.40: 33, 0.50: 42}),
    (29,  87, {0.20: 17, 0.25: 21, 0.30: 26, 0.35: 30, 0.40: 34, 0.50: 43}),
    (30,  90, {0.20: 18, 0.25: 22, 0.30: 26, 0.35: 31, 0.40: 36, 0.50: 45}),
    (31,  93, {0.20: 18, 0.25: 23, 0.30: 27, 0.35: 32, 0.40: 37, 0.50: 46}),
    (32,  96, {0.20: 19, 0.25: 24, 0.30: 28, 0.35: 33, 0.40: 38, 0.50: 48}),
    (33,  99, {0.20: 19, 0.25: 24, 0.30: 29, 0.35: 34, 0.40: 39, 0.50: 49}),
    (34, 102, {0.20: 20, 0.25: 25, 0.30: 30, 0.35: 35, 0.40: 40, 0.50: 51}),
    (35, 105, {0.20: 21, 0.25: 26, 0.30: 30, 0.35: 36, 0.40: 42, 0.50: 52}),
]

offsets = [0.20, 0.25, 0.30, 0.35, 0.40, 0.50]

# Distinct colors for each antioffset (used consistently across all plots)
OFFSET_COLORS = {
    0.20: '#E63946',  # red
    0.25: '#F4A261',  # orange
    0.30: '#2A9D8F',  # teal
    0.35: '#457B9D',  # steel blue
    0.40: '#6A4C93',  # purple
    0.50: '#2D6A4F',  # dark green
}

# C(full) always plotted in a dark neutral so it contrasts with offset colors
FULL_COLOR = '#1A1A2E'   # near-black navy

out_dir = "graphs"
os.makedirs(out_dir, exist_ok=True)

# ── Helper ───────────────────────────────────────────────────────────────────
def d_limit(n_offset):
    """Plot up to floor(n_offset/2) - 2, minimum 1."""
    return max(1, floor(n_offset / 2) - 2)

def build_series(pool_full, pool_off, d_max):
    ds      = list(range(1, d_max + 1))
    c_full  = [comb(pool_full, d) for d in ds]
    c_off   = [comb(pool_off,  d) for d in ds]
    return ds, c_full, c_off

def style_log_ax(ax, ds, c_full, c_off, label_full, label_off, color_off, title):
    ax.semilogy(ds, c_full, color=FULL_COLOR,  linewidth=2.2, marker='o',
                markersize=5, label=label_full, zorder=3)
    ax.semilogy(ds, c_off,  color=color_off,   linewidth=2.2, marker='s',
                markersize=5, linestyle='--', label=label_off, zorder=3)
    ax.set_title(title, fontsize=9, fontweight='bold', pad=4)
    ax.set_xlabel("Deviation d", fontsize=8)
    ax.set_ylabel("C(n,d)  [log scale]", fontsize=8)
    ax.legend(fontsize=7, framealpha=0.85)
    ax.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.6)
    ax.set_xticks(ds)
    ax.tick_params(axis='both', labelsize=7)

# ── Per-sheet individual graphs ───────────────────────────────────────────────
for bits, r_full, offset_map in data:
    pool_full = r_full - 2
    sheet_dir = os.path.join(out_dir, f"{bits}bit")
    os.makedirs(sheet_dir, exist_ok=True)

    for offset_val in offsets:
        n_offset  = offset_map[offset_val]
        pool_off  = n_offset - 2
        d_max     = d_limit(n_offset)
        ds, c_full, c_off = build_series(pool_full, pool_off, d_max)

        color_off   = OFFSET_COLORS[offset_val]
        label_full  = f"C({pool_full}, d)  —  full {r_full}×{r_full}"
        label_off   = f"C({pool_off}, d)  —  offset {n_offset}×{n_offset}"
        title       = (f"{bits}-bit prime  |  antioffset {offset_val:.2f}  "
                       f"|  {n_offset}×{n_offset} vs {r_full}×{r_full}  "
                       f"(d = 1 … {d_max})")

        fig, ax = plt.subplots(figsize=(8, 4.5))
        style_log_ax(ax, ds, c_full, c_off, label_full, label_off, color_off, title)
        fig.tight_layout()

        fname = os.path.join(sheet_dir,
                             f"{bits}bit_antioffset_{str(offset_val).replace('.','p')}.png")
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  saved: {fname}")

# ── Per-sheet combined graph (all 6 offsets on one canvas) ───────────────────
for bits, r_full, offset_map in data:
    pool_full = r_full - 2
    sheet_dir = os.path.join(out_dir, f"{bits}bit")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"{bits}-bit prime  —  all antioffsets  (log scale, d up to ⌊n/2⌋−2)",
                 fontsize=12, fontweight='bold', y=1.01)
    axes_flat = axes.flatten()

    for idx, offset_val in enumerate(offsets):
        n_offset  = offset_map[offset_val]
        pool_off  = n_offset - 2
        d_max     = d_limit(n_offset)
        ds, c_full, c_off = build_series(pool_full, pool_off, d_max)

        color_off  = OFFSET_COLORS[offset_val]
        label_full = f"C({pool_full},d) full {r_full}×{r_full}"
        label_off  = f"C({pool_off},d) offset {n_offset}×{n_offset}"
        title      = f"antioffset {offset_val:.2f}  |  {n_offset}×{n_offset} vs {r_full}×{r_full}"

        style_log_ax(axes_flat[idx], ds, c_full, c_off,
                     label_full, label_off, color_off, title)

    fig.tight_layout()
    fname = os.path.join(sheet_dir, f"{bits}bit_combined.png")
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"saved combined: {fname}")

# ── Master combined: all bits × all offsets on one mega canvas ───────────────
n_bits   = len(data)   # 11
n_off    = len(offsets) # 6
# Layout: rows = bits (11), cols = offsets (6)  → 11×6 subplots

fig, axes = plt.subplots(n_bits, n_off, figsize=(n_off * 4, n_bits * 3.2))
fig.suptitle("All bit primes × all antioffsets  —  C(full,d) vs C(offset,d)  [log scale]",
             fontsize=14, fontweight='bold', y=1.005)

for row_idx, (bits, r_full, offset_map) in enumerate(data):
    pool_full = r_full - 2
    for col_idx, offset_val in enumerate(offsets):
        n_offset  = offset_map[offset_val]
        pool_off  = n_offset - 2
        d_max     = d_limit(n_offset)
        ds, c_full, c_off = build_series(pool_full, pool_off, d_max)

        ax         = axes[row_idx][col_idx]
        color_off  = OFFSET_COLORS[offset_val]
        label_full = f"C({pool_full},d)"
        label_off  = f"C({pool_off},d)"
        title      = f"{bits}-bit | ao {offset_val:.2f} | d≤{d_max}"

        ax.semilogy(ds, c_full, color=FULL_COLOR, linewidth=1.6, marker='o',
                    markersize=3.5, label=label_full, zorder=3)
        ax.semilogy(ds, c_off, color=color_off, linewidth=1.6, marker='s',
                    markersize=3.5, linestyle='--', label=label_off, zorder=3)
        ax.set_title(title, fontsize=7, fontweight='bold', pad=3)
        ax.set_xticks(ds)
        ax.tick_params(axis='both', labelsize=6)
        ax.grid(True, which='both', linestyle=':', linewidth=0.4, alpha=0.5)
        ax.legend(fontsize=5.5, framealpha=0.8)

        # Column headers (top row only)
        if row_idx == 0:
            ax.set_xlabel(f"antioffset {offset_val:.2f}", fontsize=8,
                          fontweight='bold', labelpad=12)
            ax.xaxis.set_label_position('top')
        # Row labels (first column only)
        if col_idx == 0:
            ax.set_ylabel(f"{bits}-bit\n(R={r_full})", fontsize=7, fontweight='bold')

fig.tight_layout()
master_path = os.path.join(out_dir, "MASTER_all_bits_all_offsets.png")
fig.savefig(master_path, dpi=130, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved master: {master_path}")
print("Done.")
