"""
APM Per-Bit Logarithmic Line Graphs
====================================
For each prime bit: one clean log-scale line graph comparing
worst-case C(n, dev+2) for offset matrix vs full matrix
at the top deviation and its ±1 neighbours.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Load data ─────────────────────────────────────────────────────────────────
summary = pd.read_csv("/mnt/user-data/outputs/apm_top_devs_summary.csv")
bits    = sorted(summary["bit"].unique())

COLORS = plt.cm.tab20.colors

# ── One log-scale line graph per bit ─────────────────────────────────────────
for bit in bits:
    sub      = summary[summary["bit"] == bit].sort_values("deviation").reset_index(drop=True)
    prime    = int(sub["prime"].iloc[0])
    n_off    = int(sub["n_offset"].iloc[0])
    n_full   = int(sub["n_full"].iloc[0])
    best_dev = int(sub[sub["is_best"]]["deviation"].iloc[0])

    devs      = sub["deviation"].tolist()
    wc_offset = sub["wc_offset"].tolist()
    wc_full   = sub["wc_full"].tolist()

    color = COLORS[bits.index(bit) % len(COLORS)]

    fig, ax = plt.subplots(figsize=(9, 6))

    # ── Lines ──────────────────────────────────────────────────────────────────
    ax.semilogy(devs, wc_offset, "-o", color="#4C72B0", linewidth=2.5,
                markersize=10, label=f"Offset matrix  n={n_off}  →  C({n_off}, dev+2)",
                zorder=3)
    ax.semilogy(devs, wc_full,   "-^", color="#DD8452", linewidth=2.5,
                markersize=10, label=f"Full matrix    n={n_full}  →  C({n_full}, dev+2)",
                zorder=3)

    # ── Best-deviation vertical line ───────────────────────────────────────────
    ax.axvline(best_dev, color="red", linestyle="--", linewidth=1.8, alpha=0.85,
               label=f"Best deviation = {best_dev}")

    # ── Annotate each point ────────────────────────────────────────────────────
    for d, vo, vf in zip(devs, wc_offset, wc_full):
        is_best = (d == best_dev)
        weight  = "bold" if is_best else "normal"
        # offset line labels (above)
        ax.annotate(f"C({n_off},{d+2})\n={vo:.2e}",
                    xy=(d, vo),
                    xytext=(0, 14), textcoords="offset points",
                    ha="center", fontsize=7.5, color="#2255AA", fontweight=weight)
        # full line labels (below)
        ax.annotate(f"C({n_full},{d+2})\n={vf:.2e}",
                    xy=(d, vf),
                    xytext=(0, -28), textcoords="offset points",
                    ha="center", fontsize=7.5, color="#BB5500", fontweight=weight)

    # ── Star marker on best deviation ─────────────────────────────────────────
    best_row = sub[sub["is_best"]].iloc[0]
    ax.plot(best_dev, best_row["wc_offset"], "*", color="#4C72B0",
            markersize=18, zorder=5)
    ax.plot(best_dev, best_row["wc_full"],   "*", color="#DD8452",
            markersize=18, zorder=5)

    # ── Shade region between the two lines ────────────────────────────────────
    ax.fill_between(devs,
                    [np.log10(v) for v in wc_offset],   # semilogy doesn't fill directly
                    [np.log10(v) for v in wc_full],
                    alpha=0)   # invisible — we use a twin for shading

    # Add a subtle fill on the actual log axis via polygon
    ax_twin = ax.twinx()
    ax_twin.fill_between(devs, wc_offset, wc_full,
                         alpha=0.07, color="gray")
    ax_twin.set_yscale("log")
    ax_twin.set_ylim(ax.get_ylim())
    ax_twin.set_yticks([])
    ax_twin.set_yticklabels([])

    # ── Formatting ─────────────────────────────────────────────────────────────
    ax.set_xticks(devs)
    ax.set_xticklabels(
        [f"dev={d}\n{'★ best' if d == best_dev else ('+1' if d == best_dev+1 else '-1')}"
         for d in devs],
        fontsize=9
    )

    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda y, _: f"$10^{{{int(np.log10(y))}}}$" if y > 0 else "0")
    )
    ax.grid(True, which="both", linestyle="--", alpha=0.45)
    ax.set_xlabel("Deviation  (best ±1)", fontsize=11)
    ax.set_ylabel("Worst-case minors   C(n, dev+2)   [log scale]", fontsize=11)

    ax.set_title(
        f"{bit}-bit prime  |  P = {prime:,}\n"
        f"Offset matrix n={n_off}×{n_off}  vs  Full matrix n={n_full}×{n_full}",
        fontsize=12, fontweight="bold"
    )

    # Hit ratio annotation box in corner
    hr_rows = sub[["deviation", "hit_ratio_%", "hits"]].copy()
    info_lines = ["Deviation | Hits | Hit Ratio"]
    for _, r in hr_rows.iterrows():
        star = " ★" if int(r["deviation"]) == best_dev else ""
        info_lines.append(f"  dev={int(r['deviation'])}{star}   {r['hits']}   {r['hit_ratio_%']:.0f}%")
    info_text = "\n".join(info_lines)
    ax.text(0.98, 0.03, info_text,
            transform=ax.transAxes,
            fontsize=8, verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                      edgecolor="gray", alpha=0.9),
            fontfamily="monospace")

    ax.legend(fontsize=9, loc="upper left")

    plt.tight_layout()
    out = f"/mnt/user-data/outputs/logplot_{bit}bit.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")

print("\nAll per-bit log graphs saved!")
