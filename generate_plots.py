"""
Generate publication-quality plots from experiment logs.
Outputs PNG files for the arxiv paper.
"""
import re
import os

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not installed. Run: pip install matplotlib")


def parse_steps(filepath):
    """Extract (step, t1, t10) from a log file."""
    points = []
    if not os.path.exists(filepath):
        return points
    with open(filepath, 'r', errors='replace') as f:
        for line in f:
            m = re.search(r'Step (\d+).*?T1=(\d+)%.*?T10=(\d+)%', line)
            if m:
                points.append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
    return points


def plot_training_curves():
    """Plot T10 vs training steps for different configurations."""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    curves = {
        '0.6B 50K': 'long_train_output.log',
        '0.6B 100K': '100k_train_output.log',
        '1.7B 15K': 'scaling_1.7b_output.log',
        '1.7B 50K': '1.7b_50k_output.log',
    }

    for label, logfile in curves.items():
        points = parse_steps(logfile)
        if points:
            steps = [p[0] for p in points if p[0] > 0]
            t10s = [p[2] for p in points if p[0] > 0]
            if steps:
                ax.plot(steps, t10s, 'o-', label=label, markersize=4)

    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Top-10 Token Agreement (%)', fontsize=12)
    ax.set_title('FRR Quality vs Training Duration', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(40, 75)

    plt.tight_layout()
    plt.savefig('plot_training_curves.png', dpi=150)
    print("Saved plot_training_curves.png")
    plt.close()


def plot_compression_comparison():
    """Plot quality vs compression ratio for all methods."""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # (compression, t10, label)
    methods = [
        (4, 89, 'GPTQ Q2'),
        (60, 63, 'FRR 50K'),
        (48, 61, 'FRR 1.7B'),
        (239, 53, 'FRR-PHM'),
        (959, 53, 'FRR+Q2 E2E'),
        (76, 57, 'HWI'),
        (2, 95, 'MobileLLM (est.)'),
        (2, 93, 'Relaxed RT (est.)'),
    ]

    ours = [(c, t, l) for c, t, l in methods if 'FRR' in l or 'HWI' in l]
    others = [(c, t, l) for c, t, l in methods if 'FRR' not in l and 'HWI' not in l]

    for c, t, l in ours:
        ax.scatter(c, t, s=100, zorder=5, c='red')
        ax.annotate(l, (c, t), textcoords="offset points", xytext=(5, 5), fontsize=8)

    for c, t, l in others:
        ax.scatter(c, t, s=60, zorder=4, c='blue', alpha=0.6)
        ax.annotate(l, (c, t), textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.set_xscale('log')
    ax.set_xlabel('Compression Ratio (log scale)', fontsize=12)
    ax.set_ylabel('Top-10 Token Agreement (%)', fontsize=12)
    ax.set_title('Quality vs Compression: FRR vs Competitors', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('plot_compression_comparison.png', dpi=150)
    print("Saved plot_compression_comparison.png")
    plt.close()


def plot_scaling():
    """Plot quality vs model size to show scaling trend."""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    # (model_size_B, t10_at_15K, compression)
    sizes = [0.6, 1.7]
    t10s = [56, 61]
    compressions = [60, 48]

    ax.plot(sizes, t10s, 'ro-', markersize=10, linewidth=2)
    for i, (s, t, c) in enumerate(zip(sizes, t10s, compressions)):
        ax.annotate(f'{t}% T10\n{c}x compression', (s, t),
                   textcoords="offset points", xytext=(10, -15), fontsize=9)

    # Projected
    ax.plot([8], [66], 'r^', markersize=10, alpha=0.5)
    ax.annotate('8B (projected)\n~66% T10\n~32x', (8, 66),
               textcoords="offset points", xytext=(10, -15), fontsize=9, alpha=0.5)

    ax.set_xlabel('Model Size (Billion Parameters)', fontsize=12)
    ax.set_ylabel('Top-10 Agreement at 15K Steps (%)', fontsize=12)
    ax.set_title('FRR Quality Scales with Model Size', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 10)
    ax.set_ylim(50, 75)

    plt.tight_layout()
    plt.savefig('plot_scaling.png', dpi=150)
    print("Saved plot_scaling.png")
    plt.close()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if not HAS_MPL:
        print("Install matplotlib: pip install matplotlib")
        exit(1)

    plot_training_curves()
    plot_compression_comparison()
    plot_scaling()
    print("\nAll plots generated!")
