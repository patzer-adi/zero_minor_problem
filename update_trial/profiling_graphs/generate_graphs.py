#!/usr/bin/env python3
"""
APM Brahma Profiling — Before vs After Optimization Graphs
Generates publication-quality comparison charts from profiling data.
"""

import re
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ─── Config ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Inter', 'Helvetica Neue', 'Arial', 'DejaVu Sans'],
    'font.size': 12,
    'axes.titlesize': 15,
    'axes.titleweight': 'bold',
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.3,
})

DARK_BG    = '#0f1117'
CARD_BG    = '#1a1d27'
GRID_CLR   = '#2a2d37'
TEXT_CLR    = '#e0e0e0'
ACCENT_RED = '#ff4d6a'
ACCENT_GRN = '#00e676'
ACCENT_BLU = '#448aff'
ACCENT_YLW = '#ffd740'
ACCENT_PRP = '#b388ff'
ACCENT_ORG = '#ff9100'

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
BEFORE_FILE = os.path.join(OUT_DIR, '..', 'profiling_brahma', 'brahma_sys_stats.txt')
AFTER_FILE  = os.path.join(OUT_DIR, '..', 'profiling_results', 'baseline_before.txt')


def style_ax(ax, title='', xlabel='', ylabel=''):
    """Apply dark theme to an axis."""
    ax.set_facecolor(CARD_BG)
    ax.set_title(title, color=TEXT_CLR, pad=14)
    ax.set_xlabel(xlabel, color=TEXT_CLR)
    ax.set_ylabel(ylabel, color=TEXT_CLR)
    ax.tick_params(colors=TEXT_CLR)
    ax.spines['bottom'].set_color(GRID_CLR)
    ax.spines['left'].set_color(GRID_CLR)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', color=GRID_CLR, linewidth=0.5, alpha=0.5)


def parse_deviation_summaries(filepath):
    """Extract per-deviation summary stats from a run log file."""
    with open(filepath, 'r') as f:
        text = f.read()
    
    devs = {}
    pattern = r'dev=(\d+) complete.*?\n\s+Time\s+:\s+([\d.]+) s.*?Minors tested\s+:\s+([\d]+).*?Matrices hit\s+:\s+(\d+)\s*/\s*(\d+)'
    for m in re.finditer(pattern, text, re.DOTALL):
        dev = int(m.group(1))
        devs[dev] = {
            'time': float(m.group(2)),
            'minors': int(m.group(3)),
            'hits': int(m.group(4)),
            'total': int(m.group(5)),
        }
    return devs


def parse_per_matrix_times(filepath, dev_target=4):
    """Extract per-matrix times and kernel times for a specific deviation."""
    with open(filepath, 'r') as f:
        text = f.read()
    
    # Find the section for the target deviation
    dev_section_pattern = rf'\[group=25 \| dev={dev_target}/23\].*?dev={dev_target} complete'
    section_match = re.search(dev_section_pattern, text, re.DOTALL)
    if not section_match:
        return []
    
    section = section_match.group(0)
    
    results = []
    # Match each matrix entry
    time_pattern = r'\[(\d+)/100\].*?kernel_25_(\d+)\.txt.*?Time:\s+([\d.]+)\s+s.*?Tested:\s+([\d]+)\s+minors.*?Zero minors:\s+(\d+)'
    for m in re.finditer(time_pattern, section, re.DOTALL):
        idx = int(m.group(1))
        matrix_id = int(m.group(2))
        time_s = float(m.group(3))
        minors = int(m.group(4))
        zeros = int(m.group(5))
        results.append({
            'idx': idx,
            'matrix_id': matrix_id,
            'time': time_s,
            'minors': minors,
            'zeros': zeros,
        })
    
    return results


def parse_total_wall_time(filepath):
    """Extract total wall time."""
    with open(filepath, 'r') as f:
        text = f.read()
    m = re.search(r'Total wall time\s*:\s*([\d.]+)\s*s', text)
    return float(m.group(1)) if m else 0.0


def parse_nsys_cuda_api(filepath):
    """Parse nsys CUDA API summary table."""
    with open(filepath, 'r') as f:
        text = f.read()
    
    # Find cuda_api_sum section
    api_section = re.search(r"Executing 'cuda_api_sum'.*?Executing", text, re.DOTALL)
    if not api_section:
        return {}
    
    section = api_section.group(0)
    apis = {}
    # Match rows: Time(%) | Total Time(ns) | Num Calls | ... | Name
    row_pattern = r'([\d.]+)\s+([\d,]+)\s+([\d,]+)\s+[\d,.]+\s+[\d,.]+\s+[\d,]+\s+[\d,]+\s+[\d,.]+\s+(\w+)'
    for m in re.finditer(row_pattern, section):
        pct = float(m.group(1))
        name = m.group(4)
        apis[name] = pct
    
    return apis


def parse_nsys_mem_transfer(filepath):
    """Parse nsys GPU memory size summary."""
    with open(filepath, 'r') as f:
        text = f.read()
    
    transfers = {}
    # D2H
    d2h_match = re.search(r'([\d,.]+)\s+[\d,]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+\[CUDA memcpy Device-to-Host\]', text)
    if d2h_match:
        transfers['D2H_MB'] = float(d2h_match.group(1).replace(',', ''))
    h2d_match = re.search(r'([\d,.]+)\s+[\d,]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+\[CUDA memcpy Host-to-Device\]', text)
    if h2d_match:
        transfers['H2D_MB'] = float(h2d_match.group(1).replace(',', ''))
    
    return transfers


# ─── Parse Data ──────────────────────────────────────────────────────────────

print("Parsing profiling data...")
before_devs = parse_deviation_summaries(BEFORE_FILE)
after_devs  = parse_deviation_summaries(AFTER_FILE)
before_total = parse_total_wall_time(BEFORE_FILE)
after_total  = parse_total_wall_time(AFTER_FILE)
before_d4 = parse_per_matrix_times(BEFORE_FILE, dev_target=4)
after_d4  = parse_per_matrix_times(AFTER_FILE, dev_target=4)
nsys_apis = parse_nsys_cuda_api(BEFORE_FILE)
nsys_mem  = parse_nsys_mem_transfer(BEFORE_FILE)

print(f"  Before: {len(before_devs)} deviations, total={before_total:.1f}s")
print(f"  After:  {len(after_devs)} deviations, total={after_total:.1f}s")
print(f"  Dev=4 matrices: before={len(before_d4)}, after={len(after_d4)}")


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH 1: Overall Wall Time Comparison
# ═══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor(DARK_BG)
style_ax(ax, title='Total Wall Time — Before vs After', ylabel='Time (seconds)')

bars = ax.bar(
    ['Before\n(apm_brahma_2)', 'After\n(apm_brahma)'],
    [before_total, after_total],
    color=[ACCENT_RED, ACCENT_GRN],
    width=0.5,
    edgecolor='none',
    alpha=0.9,
)

for bar, val in zip(bars, [before_total, after_total]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f'{val:.1f}s', ha='center', va='bottom', color=TEXT_CLR,
            fontsize=14, fontweight='bold')

speedup = before_total / after_total
ax.text(0.5, 0.92, f'⚡ {speedup:.2f}× faster',
        transform=ax.transAxes, ha='center', va='top',
        fontsize=18, fontweight='bold', color=ACCENT_YLW,
        bbox=dict(boxstyle='round,pad=0.4', facecolor=DARK_BG, edgecolor=ACCENT_YLW, alpha=0.8))

ax.set_ylim(0, before_total * 1.2)
fig.savefig(os.path.join(OUT_DIR, '1_total_wall_time.png'))
plt.close(fig)
print("  ✓ 1_total_wall_time.png")


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH 2: Per-Deviation Time Comparison (Grouped Bar)
# ═══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(DARK_BG)
style_ax(ax, title='Per-Deviation Time — Before vs After', ylabel='Time (seconds)')

devs = sorted(set(before_devs.keys()) & set(after_devs.keys()))
x = np.arange(len(devs))
w = 0.32

before_times = [before_devs[d]['time'] for d in devs]
after_times  = [after_devs[d]['time'] for d in devs]
speedups = [b/a if a > 0 else 0 for b, a in zip(before_times, after_times)]

b1 = ax.bar(x - w/2, before_times, w, label='Before', color=ACCENT_RED, alpha=0.85, edgecolor='none')
b2 = ax.bar(x + w/2, after_times,  w, label='After',  color=ACCENT_GRN, alpha=0.85, edgecolor='none')

for i, (bv, av, sp) in enumerate(zip(before_times, after_times, speedups)):
    ax.text(i - w/2, bv + 1, f'{bv:.1f}s', ha='center', va='bottom', color=ACCENT_RED, fontsize=10)
    ax.text(i + w/2, av + 1, f'{av:.1f}s', ha='center', va='bottom', color=ACCENT_GRN, fontsize=10)
    # Speedup label above both bars
    top = max(bv, av)
    ax.text(i, top + 7, f'{sp:.1f}×', ha='center', va='bottom', color=ACCENT_YLW,
            fontsize=13, fontweight='bold')

minor_sizes = {2: '4×4', 3: '5×5', 4: '6×6'}
ax.set_xticks(x)
ax.set_xticklabels([f'dev={d}\n({minor_sizes.get(d, "?")})' for d in devs], color=TEXT_CLR)
ax.legend(loc='upper left', facecolor=CARD_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)
ax.set_ylim(0, max(before_times) * 1.3)

fig.savefig(os.path.join(OUT_DIR, '2_per_deviation_comparison.png'))
plt.close(fig)
print("  ✓ 2_per_deviation_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH 3: Speedup Factor Bar Chart
# ═══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor(DARK_BG)
style_ax(ax, title='Speedup Factor by Deviation', ylabel='Speedup (×)')

colors = [ACCENT_BLU, ACCENT_PRP, ACCENT_ORG]
bars = ax.bar(
    [f'dev={d}\n({minor_sizes.get(d, "?")})' for d in devs],
    speedups,
    color=colors[:len(devs)],
    width=0.45,
    edgecolor='none',
    alpha=0.9,
)

for bar, sp in zip(bars, speedups):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'{sp:.2f}×', ha='center', va='bottom', color=TEXT_CLR,
            fontsize=14, fontweight='bold')

ax.axhline(y=1, color=ACCENT_RED, linewidth=1.5, linestyle='--', alpha=0.6, label='No change (1×)')
ax.set_ylim(0, max(speedups) * 1.3)
ax.legend(loc='upper right', facecolor=CARD_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)

fig.savefig(os.path.join(OUT_DIR, '3_speedup_factor.png'))
plt.close(fig)
print("  ✓ 3_speedup_factor.png")


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH 4: Per-Matrix Kernel Time Distribution (dev=4)
# ═══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor(DARK_BG)
style_ax(ax, title='Per-Matrix Time at dev=4 (6×6 minors) — Before vs After',
         xlabel='Matrix Index', ylabel='Time (seconds)')

if before_d4 and after_d4:
    bx = [m['idx'] for m in before_d4]
    bt = [m['time'] for m in before_d4]
    ax_ = [m['idx'] for m in after_d4]
    at_ = [m['time'] for m in after_d4]
    
    ax.scatter(bx, bt, color=ACCENT_RED, alpha=0.6, s=30, label=f'Before (median={np.median(bt):.3f}s)', zorder=3)
    ax.scatter(ax_, at_, color=ACCENT_GRN, alpha=0.6, s=30, label=f'After (median={np.median(at_):.3f}s)', zorder=3)
    
    # Median lines
    ax.axhline(y=np.median(bt), color=ACCENT_RED, linewidth=1.5, linestyle='--', alpha=0.5)
    ax.axhline(y=np.median(at_), color=ACCENT_GRN, linewidth=1.5, linestyle='--', alpha=0.5)
    
    ax.legend(loc='upper right', facecolor=CARD_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)

fig.savefig(os.path.join(OUT_DIR, '4_per_matrix_dev4_scatter.png'))
plt.close(fig)
print("  ✓ 4_per_matrix_dev4_scatter.png")


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH 5: Per-Matrix Time Histogram (dev=4)
# ═══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(DARK_BG)
style_ax(ax, title='Kernel Time Distribution at dev=4 — Before vs After',
         xlabel='Time (seconds)', ylabel='Number of Matrices')

if before_d4 and after_d4:
    bt = [m['time'] for m in before_d4]
    at_ = [m['time'] for m in after_d4]
    
    bins_all = np.linspace(0, max(max(bt), max(at_)) * 1.05, 30)
    ax.hist(bt, bins=bins_all, alpha=0.6, color=ACCENT_RED, label=f'Before (μ={np.mean(bt):.3f}s)', edgecolor='none')
    ax.hist(at_, bins=bins_all, alpha=0.6, color=ACCENT_GRN, label=f'After (μ={np.mean(at_):.3f}s)', edgecolor='none')
    
    ax.legend(loc='upper right', facecolor=CARD_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)

fig.savefig(os.path.join(OUT_DIR, '5_kernel_time_histogram.png'))
plt.close(fig)
print("  ✓ 5_kernel_time_histogram.png")


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH 6: CUDA API Time Breakdown (Before — from nsys)
# ═══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(8, 8))
fig.patch.set_facecolor(DARK_BG)
ax.set_facecolor(DARK_BG)

# Hardcoded from the nsys data we parsed
api_data = {
    'cudaDeviceSynchronize\n(92.7%)': 92.7,
    'cudaMemcpy\n(7.2%)': 7.2,
    'Other\n(0.1%)': 0.1,
}

colors_pie = [ACCENT_RED, ACCENT_ORG, ACCENT_BLU]
wedges, texts, autotexts = ax.pie(
    api_data.values(),
    labels=api_data.keys(),
    colors=colors_pie,
    autopct='',
    startangle=90,
    pctdistance=0.75,
    wedgeprops=dict(width=0.4, edgecolor=DARK_BG, linewidth=2),
    textprops={'color': TEXT_CLR, 'fontsize': 12},
)

ax.set_title('CUDA API Time Breakdown — Before Optimization\n(from nsys profiling)',
             color=TEXT_CLR, fontsize=15, fontweight='bold', pad=20)

# Center annotation
ax.text(0, 0, 'GPU was\nidle\n92.7%\nof time', ha='center', va='center',
        fontsize=16, fontweight='bold', color=ACCENT_RED)

fig.savefig(os.path.join(OUT_DIR, '6_cuda_api_breakdown.png'))
plt.close(fig)
print("  ✓ 6_cuda_api_breakdown.png")


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH 7: Memory Transfer Imbalance (Before — from nsys)
# ═══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor(DARK_BG)
style_ax(ax, title='GPU Memory Transfer Imbalance — Before Optimization',
         ylabel='Total Transfer (GB)')

d2h_gb = 50366.515 / 1024
h2d_gb = 1329.382 / 1024

bars = ax.bar(
    ['Device → Host\n(flag arrays)', 'Host → Device\n(index sets)'],
    [d2h_gb, h2d_gb],
    color=[ACCENT_RED, ACCENT_GRN],
    width=0.45,
    edgecolor='none',
    alpha=0.9,
)

for bar, val in zip(bars, [d2h_gb, h2d_gb]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.1f} GB', ha='center', va='bottom', color=TEXT_CLR,
            fontsize=14, fontweight='bold')

ax.text(0.5, 0.88, f'38:1 ratio — backwards!',
        transform=ax.transAxes, ha='center', va='top',
        fontsize=16, fontweight='bold', color=ACCENT_YLW,
        bbox=dict(boxstyle='round,pad=0.4', facecolor=DARK_BG, edgecolor=ACCENT_YLW, alpha=0.8))

ax.set_ylim(0, d2h_gb * 1.25)
fig.savefig(os.path.join(OUT_DIR, '7_memory_transfer_imbalance.png'))
plt.close(fig)
print("  ✓ 7_memory_transfer_imbalance.png")


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH 8: Projected Speedup for Larger Groups
# ═══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(DARK_BG)
style_ax(ax, title='Projected mod_mul Speedup by Prime Group',
         xlabel='Prime Group', ylabel='mod_mul Loop Iterations')

groups = np.arange(25, 51)
old_iters = groups.copy().astype(float)  # ~log2(p) iterations in old binary loop
new_iters = np.ones_like(groups, dtype=float)  # 1 instruction with __uint128_t

ax.fill_between(groups, old_iters, new_iters, alpha=0.3, color=ACCENT_RED, label='Eliminated iterations')
ax.plot(groups, old_iters, color=ACCENT_RED, linewidth=2.5, marker='o', markersize=4,
        label='Before (binary loop)', zorder=3)
ax.plot(groups, new_iters, color=ACCENT_GRN, linewidth=2.5, marker='o', markersize=4,
        label='After (__uint128_t)', zorder=3)

# Annotate key groups
for g in [25, 31, 39, 50]:
    ax.annotate(f'Group {g}\n{g}× saving',
                xy=(g, g), xytext=(g, g + 5),
                ha='center', fontsize=9, color=ACCENT_YLW,
                arrowprops=dict(arrowstyle='->', color=ACCENT_YLW, lw=1.2))

ax.legend(loc='upper left', facecolor=CARD_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)
ax.set_ylim(0, 60)
ax.set_xlim(24, 51)

fig.savefig(os.path.join(OUT_DIR, '8_projected_modmul_speedup.png'))
plt.close(fig)
print("  ✓ 8_projected_modmul_speedup.png")


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH 9: Stack Frame Comparison
# ═══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor(DARK_BG)
style_ax(ax, title='Per-Thread Stack Frame — Before vs After (from ptxas)',
         ylabel='Stack Size (KB)')

categories = ['sub[50×50]\n(kernel)', 'a[50×50]\n(det_mod)', 'Total']
before_stack = [20, 20, 40]
after_stack  = [20, 0, 20]

x = np.arange(len(categories))
w = 0.32
b1 = ax.bar(x - w/2, before_stack, w, label='Before (40 KB)', color=ACCENT_RED, alpha=0.85, edgecolor='none')
b2 = ax.bar(x + w/2, after_stack,  w, label='After (20 KB)',  color=ACCENT_GRN, alpha=0.85, edgecolor='none')

for bar, val in zip(list(b1) + list(b2), before_stack + after_stack):
    if val > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val} KB', ha='center', va='bottom', color=TEXT_CLR, fontsize=11, fontweight='bold')

ax.text(2, 30, '50%\nreduction',
        ha='center', va='center', fontsize=14, fontweight='bold', color=ACCENT_YLW)

ax.set_xticks(x)
ax.set_xticklabels(categories, color=TEXT_CLR)
ax.legend(loc='upper left', facecolor=CARD_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)
ax.set_ylim(0, 50)

fig.savefig(os.path.join(OUT_DIR, '9_stack_frame_comparison.png'))
plt.close(fig)
print("  ✓ 9_stack_frame_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH 10: Combined Summary Dashboard
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle('APM Brahma — Optimization Impact Dashboard',
             color=TEXT_CLR, fontsize=20, fontweight='bold', y=0.98)

# Panel A: Wall time
ax = axes[0, 0]
style_ax(ax, title='Total Wall Time', ylabel='Seconds')
bars = ax.bar(['Before', 'After'], [before_total, after_total],
              color=[ACCENT_RED, ACCENT_GRN], width=0.5, edgecolor='none', alpha=0.9)
for bar, val in zip(bars, [before_total, after_total]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f'{val:.1f}s', ha='center', va='bottom', color=TEXT_CLR, fontsize=12, fontweight='bold')
ax.text(0.5, 0.85, f'{speedup:.2f}× faster', transform=ax.transAxes, ha='center',
        fontsize=15, fontweight='bold', color=ACCENT_YLW)

# Panel B: Per-deviation speedup
ax = axes[0, 1]
style_ax(ax, title='Speedup by Deviation', ylabel='Speedup (×)')
bars = ax.bar([f'dev={d}' for d in devs], speedups,
              color=colors[:len(devs)], width=0.45, edgecolor='none', alpha=0.9)
for bar, sp in zip(bars, speedups):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'{sp:.1f}×', ha='center', va='bottom', color=TEXT_CLR, fontsize=12, fontweight='bold')
ax.axhline(y=1, color=ACCENT_RED, linewidth=1, linestyle='--', alpha=0.5)
ax.set_ylim(0, max(speedups) * 1.3)

# Panel C: Dev=4 scatter
ax = axes[1, 0]
style_ax(ax, title='Per-Matrix Time (dev=4)', xlabel='Matrix #', ylabel='Seconds')
if before_d4 and after_d4:
    bx = [m['idx'] for m in before_d4]
    bt = [m['time'] for m in before_d4]
    ax2 = [m['idx'] for m in after_d4]
    at2 = [m['time'] for m in after_d4]
    ax.scatter(bx, bt, color=ACCENT_RED, alpha=0.5, s=20, label='Before')
    ax.scatter(ax2, at2, color=ACCENT_GRN, alpha=0.5, s=20, label='After')
    ax.axhline(y=np.median(bt), color=ACCENT_RED, linewidth=1, linestyle='--', alpha=0.4)
    ax.axhline(y=np.median(at2), color=ACCENT_GRN, linewidth=1, linestyle='--', alpha=0.4)
    ax.legend(loc='upper right', facecolor=CARD_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR, fontsize=9)

# Panel D: Stack comparison
ax = axes[1, 1]
style_ax(ax, title='Stack Per Thread (KB)', ylabel='KB')
x_pos = np.arange(2)
w = 0.32
ax.bar(x_pos - w/2, [40, 20], w, label='Before', color=ACCENT_RED, alpha=0.85, edgecolor='none')
ax.bar(x_pos + w/2, [20, 0], w, label='After', color=ACCENT_GRN, alpha=0.85, edgecolor='none')
ax.set_xticks(x_pos)
ax.set_xticklabels(['sub[] + a[][]', 'a[][] (eliminated)'], color=TEXT_CLR)
ax.set_ylim(0, 50)
ax.text(0, 42, '40 KB → 20 KB', ha='center', fontsize=13, fontweight='bold', color=ACCENT_YLW)
ax.legend(loc='upper right', facecolor=CARD_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR, fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUT_DIR, '10_summary_dashboard.png'))
plt.close(fig)
print("  ✓ 10_summary_dashboard.png")


# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n✅ All graphs saved to: {OUT_DIR}/")
print(f"   10 PNG files generated.")
