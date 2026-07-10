"""
generate_figures.py
===================
Figure generation script for:
"Resource Efficiency of Predictive Coding versus Backpropagation
in Matched Nonlinear Classification Networks: A Sample-Compute Trade-off Analysis"

Requires:
    - confirmatory_results.csv in the same directory
    - matplotlib, numpy, pandas

Usage:
    python3 generate_figures.py

Outputs (written to figures/ subdirectory):
    fig1_pareto_confirmatory.png/.pdf
    fig2_learning_curves_confirmatory.png/.pdf
    fig3_budget_confirmatory.png/.pdf
    fig4_compute_ratio_confirmatory.png/.pdf
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import os
import warnings
warnings.filterwarnings('ignore')

# ── OUTPUT DIRECTORY ──────────────────────────────────────────────────────────
os.makedirs('figures', exist_ok=True)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
df = pd.read_csv('confirmatory_results.csv')

# Cumulative pure training time per run
cum_time = df.groupby(
    ['algorithm','num_layers','train_size','seed','inference_steps']
)['wall_time_s'].sum().reset_index()
cum_time.rename(columns={'wall_time_s':'cum_train_time'}, inplace=True)

ep30 = df[df['epoch']==30][
    ['algorithm','num_layers','train_size','seed','inference_steps','test_acc']
].copy()
merged = ep30.merge(cum_time,
                    on=['algorithm','num_layers','train_size','seed','inference_steps'])

bp_mean = merged[merged['algorithm']=='BP'].groupby(['num_layers','train_size']).agg(
    acc=('test_acc','mean'), acc_std=('test_acc','std'),
    cum_time=('cum_train_time','mean')).reset_index()

pc_mean = merged[merged['algorithm']=='PC'].groupby(
    ['num_layers','train_size','inference_steps']).agg(
    acc=('test_acc','mean'), acc_std=('test_acc','std'),
    cum_time=('cum_train_time','mean')).reset_index()

# ── GLOBAL STYLE ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['Times New Roman', 'DejaVu Serif'],
    'font.size':         10,
    'axes.titlesize':    11,
    'axes.labelsize':    10,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'legend.fontsize':   9,
    'figure.dpi':        180,
    'axes.linewidth':    0.8,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'xtick.direction':   'out',
    'ytick.direction':   'out',
    'xtick.major.size':  4,
    'ytick.major.size':  4,
    'axes.grid':         True,
    'grid.alpha':        0.25,
    'grid.linewidth':    0.5,
})

DEPTHS         = [2, 4, 6]
TRAIN_SIZES    = [1000, 5000]
BUDGETS        = [10, 20, 50]
depth_colors   = {2:'#2166ac', 4:'#d6604d', 6:'#4dac26'}
depth_labels   = {2:'2 hidden layers', 4:'4 hidden layers', 6:'6 hidden layers'}
budget_markers = {10:'o', 20:'s', 50:'^'}
budget_ls      = {10:'-', 20:'--', 50:'-.'}


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — PARETO MAP
# x = cumulative pure training time (log scale)
# y = mean final evaluation accuracy
# PC points: marker shape = T, colour = depth
# BP: horizontal dashed reference lines, labelled on rightmost panel only
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
fig.subplots_adjust(wspace=0.08, left=0.08, right=0.80,
                    top=0.82, bottom=0.22)

for ax, depth in zip(axes, DEPTHS):
    col  = depth_colors[depth]
    bp_d = bp_mean[bp_mean['num_layers']==depth]
    pc_d = pc_mean[pc_mean['num_layers']==depth]

    for _, row in bp_d.iterrows():
        ax.axhline(row['acc'], color='#aaaaaa', lw=1.0, ls='--', zorder=1)

    for ts in TRAIN_SIZES:
        pc_ts = pc_d[pc_d['train_size']==ts].sort_values('inference_steps')
        ax.plot(pc_ts['cum_time'], pc_ts['acc'],
                color=col, lw=0.7, alpha=0.35, zorder=2)
        for _, r in pc_ts.iterrows():
            T = int(r['inference_steps'])
            ax.scatter(r['cum_time'], r['acc'], color=col,
                       marker=budget_markers[T], s=55, zorder=5,
                       edgecolors='white', linewidths=0.6)

    ax.set_xscale('log')
    ax.set_ylim(0.45, 1.00)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel('Cumulative pure training time (s, log scale)', fontsize=9.5)
    ax.set_title(depth_labels[depth], fontsize=11, pad=6)
    if depth == 2:
        ax.set_ylabel('Mean final evaluation accuracy', fontsize=10)

# BP labels on right side panel only
bp_label_ax = fig.add_axes([0.815, 0.30, 0.001, 0.45])
bp_label_ax.axis('off')
for ts, ypos in zip(TRAIN_SIZES, [0.78, 0.32]):
    bp_label_ax.text(0.5, ypos, f'BP  n={ts:,}',
                     va='center', ha='left', fontsize=8.5,
                     color='#666666', transform=bp_label_ax.transAxes)

marker_handles = [
    Line2D([0],[0], marker=budget_markers[T], color='#444', lw=0,
           markersize=7, label=f'PC   T = {T}')
    for T in BUDGETS
] + [Line2D([0],[0], color='#aaa', lw=1.2, ls='--', label='BP baseline')]

fig.legend(handles=marker_handles, loc='lower center', ncol=4,
           frameon=False, fontsize=9, bbox_to_anchor=(0.44, 0.02))

fig.suptitle(
    'Figure 1.  Accuracy–compute trade-off map for the confirmatory grid\n'
    'Each marker = one (depth, training size, T) configuration, mean across 5 seeds. '
    'Dashed lines = matched BP baselines.',
    fontsize=9.5, y=0.99, va='top')

plt.savefig('figures/fig1_pareto_confirmatory.png', bbox_inches='tight', dpi=180)
plt.savefig('figures/fig1_pareto_confirmatory.pdf', bbox_inches='tight')
plt.close()
print("Figure 1 saved.")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — COMPUTE RATIO BARS
# Grouped bars: one group per training size, one bar per T
# Height = cumulative PC time / cumulative BP time
# Annotated with time ratio (bold) and accuracy gap (grey)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
fig.subplots_adjust(wspace=0.32, left=0.07, right=0.97,
                    top=0.78, bottom=0.20)

x          = np.arange(len(TRAIN_SIZES))
width      = 0.22
offsets    = {10: -width, 20: 0, 50: width}
bar_colors = {10:'#4393c3', 20:'#2166ac', 50:'#053061'}

for ax, depth in zip(axes, DEPTHS):
    bp_d = bp_mean[bp_mean['num_layers']==depth]
    pc_d = pc_mean[pc_mean['num_layers']==depth]

    y_max = 0
    bar_data = {}
    for T in BUDGETS:
        ratios, gaps = [], []
        for ts in TRAIN_SIZES:
            bp_r  = bp_d[bp_d['train_size']==ts]
            pc_r  = pc_d[(pc_d['train_size']==ts) & (pc_d['inference_steps']==T)]
            ratio = pc_r['cum_time'].values[0] / bp_r['cum_time'].values[0]
            gap   = bp_r['acc'].values[0] - pc_r['acc'].values[0]
            ratios.append(ratio)
            gaps.append(gap)
            y_max = max(y_max, ratio)
        bar_data[T] = (ratios, gaps)

    ax.set_ylim(0, y_max * 1.48)

    for T in BUDGETS:
        ratios, gaps = bar_data[T]
        ax.bar(x + offsets[T], ratios, width=width*0.92,
               color=bar_colors[T], alpha=0.85,
               edgecolor='white', linewidth=0.4, label=f'T = {T}')
        for i, (r, g) in enumerate(zip(ratios, gaps)):
            ax.text(x[i]+offsets[T], r + y_max*0.02,
                    f'{r:.0f}×',
                    ha='center', va='bottom',
                    fontsize=7.5, fontweight='bold', color='#111')
            ax.text(x[i]+offsets[T], r + y_max*0.15,
                    f'−{g:.2f}',
                    ha='center', va='bottom',
                    fontsize=6.8, color='#444')

    ax.axhline(1, color='#333', lw=0.9, ls='--', zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f'n={ts:,}' for ts in TRAIN_SIZES], fontsize=9)
    ax.set_title(depth_labels[depth], fontsize=11, pad=6)
    if depth == 2:
        ax.set_ylabel('Cumulative time ratio  PC / BP', fontsize=10)
    ax.legend(fontsize=8, frameon=True, framealpha=0.9,
              edgecolor='#cccccc', loc='upper left', ncol=3,
              columnspacing=0.8)

fig.text(0.50, 0.09,
         '— — —  Dashed line = parity (PC / BP = 1)',
         ha='center', va='top', fontsize=8.5, color='#444', style='italic')

fig.suptitle(
    'Figure 2.  Cumulative pure training time ratio (PC / BP) by inference budget\n'
    'Numbers above bars: time ratio (bold) and accuracy gap BP − PC (grey). '
    'Dashed line = parity.',
    fontsize=9.5, y=0.99, va='top')

plt.savefig('figures/fig2_compute_ratio_confirmatory.png', bbox_inches='tight', dpi=180)
plt.savefig('figures/fig2_compute_ratio_confirmatory.pdf', bbox_inches='tight')
plt.close()
print("Figure 2 saved.")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — BUDGET EFFECT
# Panel (a): accuracy vs T, one line per depth, solid=n=1000, dotted=n=5000
# Panel (b): cumulative training time vs T, same structure
# NO BP baselines in panel (a)
# Single shared legend above both panels
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
fig.subplots_adjust(wspace=0.35, left=0.09, right=0.97,
                    top=0.78, bottom=0.18)

pc_fix_1000 = pc_mean[pc_mean['train_size']==1000]
pc_fix_5000 = pc_mean[pc_mean['train_size']==5000]

for depth in DEPTHS:
    col  = depth_colors[depth]
    sub1 = pc_fix_1000[pc_fix_1000['num_layers']==depth].sort_values('inference_steps')
    sub5 = pc_fix_5000[pc_fix_5000['num_layers']==depth].sort_values('inference_steps')

    axes[0].plot(sub1['inference_steps'], sub1['acc'],
                 color=col, marker='o', markersize=7, lw=1.8, zorder=5)
    axes[0].plot(sub5['inference_steps'], sub5['acc'],
                 color=col, marker='s', markersize=5, lw=1.0,
                 ls=':', alpha=0.55, zorder=4)

    axes[1].plot(sub1['inference_steps'], sub1['cum_time'],
                 color=col, marker='o', markersize=7, lw=1.8, zorder=5)
    axes[1].plot(sub5['inference_steps'], sub5['cum_time'],
                 color=col, marker='s', markersize=5, lw=1.0,
                 ls=':', alpha=0.55, zorder=4)

axes[0].set_xticks(BUDGETS)
axes[0].set_ylim(0.45, 0.95)
axes[0].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
axes[0].set_xlabel('Inference steps  T', fontsize=10)
axes[0].set_ylabel('Mean final evaluation accuracy', fontsize=10)
axes[0].set_title('(a)  Accuracy vs. inference budget', fontsize=11, pad=6)

axes[1].set_xticks(BUDGETS)
axes[1].set_xlabel('Inference steps  T', fontsize=10)
axes[1].set_ylabel('Cumulative pure training time (s)', fontsize=10)
axes[1].set_title('(b)  Cumulative training time vs. inference budget', fontsize=11, pad=6)

# Shared legend above panels
legend_handles = []
legend_labels  = []
for depth in DEPTHS:
    legend_handles.append(
        Line2D([0],[0], color=depth_colors[depth], lw=2.0,
               marker='o', markersize=6))
    legend_labels.append(depth_labels[depth])
legend_handles.append(Line2D([0],[0], color='none'))
legend_labels.append('')
legend_handles.append(
    Line2D([0],[0], color='#555', lw=1.8, marker='o', markersize=6))
legend_labels.append('n = 1,000  (solid)')
legend_handles.append(
    Line2D([0],[0], color='#555', lw=1.2, marker='s', markersize=5,
           ls=':', alpha=0.7))
legend_labels.append('n = 5,000  (dotted)')

fig.legend(handles=legend_handles, labels=legend_labels,
           loc='upper center', ncol=4,
           frameon=True, framealpha=0.95, edgecolor='#cccccc',
           fontsize=8.5, bbox_to_anchor=(0.53, 0.97),
           columnspacing=1.2, handlelength=2.0)

fig.suptitle(
    'Figure 3.  Effect of PC inference budget on evaluation accuracy '
    'and cumulative pure training time',
    fontsize=10, y=1.01, va='bottom')

plt.savefig('figures/fig3_budget_confirmatory.png', bbox_inches='tight', dpi=180)
plt.savefig('figures/fig3_budget_confirmatory.pdf', bbox_inches='tight')
plt.close()
print("Figure 3 saved.")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — LEARNING CURVES
# 2x3 grid: rows = training sizes, columns = depths
# BP (black) and PC T=10/20/50 (coloured) across all 30 epochs
# Shaded bands = +/- 1 s.d. across 5 seeds
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharey=False)
fig.subplots_adjust(wspace=0.22, hspace=0.42,
                    left=0.08, right=0.97, top=0.90, bottom=0.12)

for col_idx, depth in enumerate(DEPTHS):
    for row_idx, ts in enumerate(TRAIN_SIZES):
        ax  = axes[row_idx, col_idx]
        col = depth_colors[depth]

        bp_sub = df[(df['algorithm']=='BP') &
                    (df['num_layers']==depth) &
                    (df['train_size']==ts)]
        bp_ep  = bp_sub.groupby('epoch')['test_acc'].agg(['mean','std']).reset_index()
        ax.plot(bp_ep['epoch'], bp_ep['mean'],
                color='#111111', lw=2.0, label='BP', zorder=6)
        ax.fill_between(bp_ep['epoch'],
                        bp_ep['mean']-bp_ep['std'],
                        bp_ep['mean']+bp_ep['std'],
                        color='#111111', alpha=0.10, zorder=5)

        for T in BUDGETS:
            pc_sub = df[(df['algorithm']=='PC') &
                        (df['num_layers']==depth) &
                        (df['train_size']==ts) &
                        (df['inference_steps']==T)]
            pc_ep  = pc_sub.groupby('epoch')['test_acc'].agg(['mean','std']).reset_index()
            ax.plot(pc_ep['epoch'], pc_ep['mean'],
                    color=col, lw=1.4, ls=budget_ls[T],
                    marker=budget_markers[T], markersize=3,
                    markevery=5,
                    label=f'PC  T={T}', zorder=4, alpha=0.9)
            ax.fill_between(pc_ep['epoch'],
                            pc_ep['mean']-pc_ep['std'],
                            pc_ep['mean']+pc_ep['std'],
                            color=col, alpha=0.08, zorder=3)

        ax.set_xlim(1, 30)
        ax.set_ylim(0.40, 1.00)
        ax.yaxis.set_major_formatter(
            ticker.PercentFormatter(xmax=1, decimals=0))
        ax.set_xlabel('Epoch', fontsize=9.5)
        if col_idx == 0:
            ax.set_ylabel('Mean evaluation accuracy', fontsize=9.5)
        ax.set_title(f'{depth_labels[depth]},  n = {ts:,}',
                     fontsize=10, pad=5)

        if row_idx == 0 and col_idx == 0:
            ax.legend(fontsize=8, frameon=True, framealpha=0.9,
                      edgecolor='#cccccc', loc='lower right', ncol=2)

fig.suptitle(
    'Figure 4.  Learning curves: mean evaluation accuracy vs. epoch\n'
    'Shaded bands = ±1 s.d. across 5 seeds. '
    'BP (black); PC at T ∈ {10, 20, 50} (coloured lines).',
    fontsize=10, y=0.98, va='top')

plt.savefig('figures/fig4_learning_curves_confirmatory.png',
            bbox_inches='tight', dpi=180)
plt.savefig('figures/fig4_learning_curves_confirmatory.pdf',
            bbox_inches='tight')
plt.close()
print("Figure 4 saved.")
print("\nAll figures saved to figures/")
