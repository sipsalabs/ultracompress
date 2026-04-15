"""
Generate publication-quality figures for the FRR paper.

Creates:
  1. Figure 1: T10 training curves (0.6B vs 1.7B, random vs real text)
  2. Figure 2: Scaling behavior bar chart
  3. Figure 3: Compression-quality Pareto frontier
  4. Figure 4: Inference speed comparison

Usage:
  python generate_paper_figures.py
  python generate_paper_figures.py --output-dir paper_figures
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = 'paper_figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    '0.6b_random': '#2196F3',
    '0.6b_real': '#1565C0',
    '1.7b_random': '#FF9800',
    '1.7b_real': '#E65100',
    'frr': '#4CAF50',
    'baseline': '#9E9E9E',
    'hwi': '#9C27B0',
    'bitnet': '#F44336',
    'genome': '#795548',
}


# ══════════════════════════════════════════════════════════════════════
# DATA (from actual experiments — verified results)
# ══════════════════════════════════════════════════════════════════════

# Training curves: (step, T10%) 
curves_t10 = {
    '0.6B Random': {
        'steps': [0, 10_000, 15_000, 25_000, 50_000, 100_000],
        't10':   [18, 52,     56,     60,     63,     65],
        'color': COLORS['0.6b_random'],
        'style': '--',
    },
    '0.6B Real Text': {
        'steps': [0, 15_000],
        't10':   [18, 60],
        'color': COLORS['0.6b_real'],
        'style': '-',
    },
    '1.7B Random': {
        'steps': [0, 15_000, 50_000, 100_000],
        't10':   [21, 61,     64,     67],
        'color': COLORS['1.7b_random'],
        'style': '--',
    },
    '1.7B Real Text': {
        'steps': [0, 5_000, 10_000, 15_000, 20_000],
        't10':   [21, 61.4,  62.4,   61.0,   60.3],
        'color': COLORS['1.7b_real'],
        'style': '-',
        'marker': 'o',
    },
}

# Compression vs quality data
compression_quality = [
    ('FRR 0.6B (100K)', 60, 65, COLORS['frr'], 'o'),
    ('FRR 1.7B (100K)', 52, 67, COLORS['1.7b_real'], 's'),
    ('FRR 1.7B (10K real)', 52, 62.4, COLORS['1.7b_real'], '^'),
    ('HWI', 76, 57, COLORS['hwi'], 'D'),
    ('BitNet Ternary', 6, 57, COLORS['bitnet'], 'v'),
    ('Genome + Hidden', 37, 63, COLORS['genome'], 'P'),
    ('FRR + Q2 + Entropy', 959, 63.5, '#E91E63', '*'),
]

# Inference speed
inference_data = {
    'seq_lens': [32, 128, 256],
    'teacher': [613, 2624, 5223],
    'frr': [2073, 8041, 16403],
}

# Scaling table
scaling_data = {
    'labels': ['0.6B\n15K', '0.6B\n50K', '0.6B\n100K', '1.7B\n15K', '1.7B\n100K', '1.7B\n10K RT'],
    't10': [56, 63, 65, 61, 67, 62.4],
    'is_real_text': [False, False, False, False, False, True],
}


# ══════════════════════════════════════════════════════════════════════
# FIGURE 1: Training Curves
# ══════════════════════════════════════════════════════════════════════
def fig1_training_curves():
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, data in curves_t10.items():
        marker = data.get('marker', '')
        ax.plot(
            [s / 1000 for s in data['steps']], data['t10'],
            color=data['color'], linestyle=data['style'],
            marker=marker, markersize=5, linewidth=2,
            label=name,
        )

    # Annotate the T10 peak and decline for 1.7B real text
    ax.annotate(
        '62.4% peak\n(10K steps)',
        xy=(10, 62.4), xytext=(25, 58),
        fontsize=9, ha='center',
        arrowprops=dict(arrowstyle='->', color='gray', lw=1.2),
    )

    ax.set_xlabel('Training Steps (×1000)')
    ax.set_ylabel('Top-10 Agreement (%)')
    ax.set_title('FRR Training Curves: Random vs Real Text Distillation')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_ylim(15, 72)
    ax.set_xlim(-2, 105)

    path = os.path.join(OUTPUT_DIR, 'fig1_training_curves.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════
# FIGURE 2: Scaling Behavior
# ══════════════════════════════════════════════════════════════════════
def fig2_scaling():
    fig, ax = plt.subplots(figsize=(8, 5))

    labels = scaling_data['labels']
    t10s = scaling_data['t10']
    is_rt = scaling_data['is_real_text']
    x = np.arange(len(labels))

    colors = [COLORS['1.7b_real'] if rt else ('#FF9800' if '1.7B' in l else '#2196F3')
              for l, rt in zip(labels, is_rt)]

    bars = ax.bar(x, t10s, color=colors, edgecolor='white', linewidth=0.5, width=0.7)

    # Add value labels on bars
    for bar, val in zip(bars, t10s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add legend manually
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2196F3', label='0.6B (random tokens)'),
        Patch(facecolor='#FF9800', label='1.7B (random tokens)'),
        Patch(facecolor=COLORS['1.7b_real'], label='1.7B (real text)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9)

    ax.set_xlabel('Model Scale / Training Steps')
    ax.set_ylabel('Top-10 Agreement (%)')
    ax.set_title('FRR Quality Scales with Model Size and Training Data')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 75)

    path = os.path.join(OUTPUT_DIR, 'fig2_scaling.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════
# FIGURE 3: Compression-Quality Pareto Frontier
# ══════════════════════════════════════════════════════════════════════
def fig3_pareto():
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, compression, t10, color, marker in compression_quality:
        ax.scatter(compression, t10, c=color, marker=marker, s=120,
                   edgecolors='white', linewidths=0.5, zorder=5, label=name)

    # Draw Pareto frontier
    points = sorted(compression_quality, key=lambda x: x[1])
    pareto_x, pareto_y = [], []
    max_t10 = 0
    for _, comp, t10, _, _ in sorted(compression_quality, key=lambda x: x[1]):
        if t10 >= max_t10:
            pareto_x.append(comp)
            pareto_y.append(t10)
            max_t10 = t10

    # Add reference lines
    ax.axhline(y=63, color='gray', linestyle=':', alpha=0.4, label='Genome baseline (63%)')
    ax.axvline(x=4, color='gray', linestyle=':', alpha=0.4)
    ax.text(4.5, 55, 'Prior SOTA\n(2-4x)', fontsize=8, color='gray', ha='left')

    ax.set_xscale('log')
    ax.set_xlabel('Compression Ratio (×)')
    ax.set_ylabel('Top-10 Agreement (%)')
    ax.set_title('Compression vs Quality: FRR Pareto Frontier')
    ax.legend(loc='lower left', fontsize=8, framealpha=0.9)
    ax.set_xlim(3, 1500)
    ax.set_ylim(50, 72)

    path = os.path.join(OUTPUT_DIR, 'fig3_pareto.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════
# FIGURE 4: Inference Speedup
# ══════════════════════════════════════════════════════════════════════
def fig4_inference():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    seq_lens = inference_data['seq_lens']
    teacher = inference_data['teacher']
    frr = inference_data['frr']
    speedups = [f / t for f, t in zip(frr, teacher)]

    x = np.arange(len(seq_lens))
    width = 0.35

    # Left: throughput comparison
    bars1 = ax1.bar(x - width / 2, teacher, width, label='Teacher (0.6B)',
                    color=COLORS['baseline'], edgecolor='white')
    bars2 = ax1.bar(x + width / 2, frr, width, label='FRR (60x)',
                    color=COLORS['frr'], edgecolor='white')

    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Tokens/sec')
    ax1.set_title('Inference Throughput')
    ax1.set_xticks(x)
    ax1.set_xticklabels(seq_lens)
    ax1.legend(framealpha=0.9)

    # Add value labels
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=8)

    # Right: speedup
    ax2.bar(x, speedups, color=COLORS['frr'], edgecolor='white', width=0.5)
    for i, (xi, s) in enumerate(zip(x, speedups)):
        ax2.text(xi, s + 0.05, f'{s:.2f}×', ha='center', va='bottom',
                 fontsize=11, fontweight='bold')

    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Speedup (×)')
    ax2.set_title('FRR Speedup over Teacher')
    ax2.set_xticks(x)
    ax2.set_xticklabels(seq_lens)
    ax2.set_ylim(0, 4)
    ax2.axhline(y=1, color='gray', linestyle=':', alpha=0.5)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig4_inference.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════
# FIGURE 5: Temperature Annealing Effect
# ══════════════════════════════════════════════════════════════════════
def fig5_temperature():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # 1.7B real text data: step, T10, temperature
    steps = [0, 5000, 10000, 15000, 20000]
    t10 = [21.4, 61.4, 62.4, 61.0, 60.3]
    temps = [5.0, 4.8, 4.5, 4.2, 4.0]
    loss = [561.9, 41.3, 37.6, 37.4, 37.2]

    # Left: T10 and temperature vs steps
    color_t10 = COLORS['1.7b_real']
    color_temp = '#9E9E9E'

    ax1.plot([s / 1000 for s in steps], t10, 'o-', color=color_t10, linewidth=2,
             markersize=6, label='T10 Agreement')
    ax1_twin = ax1.twinx()
    ax1_twin.plot([s / 1000 for s in steps], temps, 's--', color=color_temp,
                  linewidth=1.5, markersize=5, alpha=0.7, label='Temperature')

    ax1.set_xlabel('Steps (×1000)')
    ax1.set_ylabel('T10 (%)', color=color_t10)
    ax1_twin.set_ylabel('Temperature', color=color_temp)
    ax1.set_title('T10 vs Temperature During Training')
    ax1.tick_params(axis='y', labelcolor=color_t10)
    ax1_twin.tick_params(axis='y', labelcolor=color_temp)

    # Mark the peak
    ax1.annotate('Peak: 62.4%', xy=(10, 62.4), xytext=(15, 65),
                 fontsize=9, ha='center',
                 arrowprops=dict(arrowstyle='->', color='gray', lw=1))

    # Add legend combining both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=9)

    # Right: Loss vs T10 (showing misalignment)
    ax2.scatter(loss[1:], t10[1:], c=[temps[i] for i in range(1, len(temps))],
                cmap='RdYlBu', s=100, edgecolors='black', linewidths=0.5, zorder=5)
    for i in range(1, len(steps)):
        ax2.annotate(f'{steps[i]//1000}K', xy=(loss[i], t10[i]),
                     xytext=(5, 5), textcoords='offset points', fontsize=8)

    ax2.set_xlabel('Training Loss')
    ax2.set_ylabel('T10 Agreement (%)')
    ax2.set_title('Loss-Quality Misalignment')
    ax2.invert_xaxis()

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='RdYlBu', norm=plt.Normalize(vmin=min(temps), vmax=max(temps)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2, label='Temperature')

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig5_temperature.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════
# Generate all figures
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("GENERATING PAPER FIGURES")
    print("=" * 60)

    figs = [
        ("Fig 1: Training Curves", fig1_training_curves),
        ("Fig 2: Scaling", fig2_scaling),
        ("Fig 3: Pareto Frontier", fig3_pareto),
        ("Fig 4: Inference Speed", fig4_inference),
        ("Fig 5: Temperature Effect", fig5_temperature),
    ]

    paths = []
    for name, fn in figs:
        print(f"\n{name}...")
        try:
            path = fn()
            paths.append(path)
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\n{'=' * 60}")
    print(f"Generated {len(paths)} figures in {OUTPUT_DIR}/")
    print("=" * 60)
