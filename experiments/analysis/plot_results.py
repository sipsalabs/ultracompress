"""
Parse training log files and generate publication-quality plots.

Reads from log files or live terminal output saved to .log files.
Generates:
  1. Loss curves
  2. T1/T10 progression
  3. Experiment comparisons (selective student)
  4. Records timeline

Usage:
  python plot_results.py                              # Plot all available logs
  python plot_results.py --log 1.7b_real_text_100k_output.log
  python plot_results.py --log selective_student_test_output.log
"""
import argparse
import os
import re
import sys
from dataclasses import dataclass, field

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class TrainPoint:
    step: int
    loss: float
    t1: float
    t10: float
    elapsed: float
    temperature: float | None = None
    new_best: bool = False


@dataclass
class Experiment:
    name: str
    points: list[TrainPoint] = field(default_factory=list)
    trainable_params: int = 0


# ── Parsing ───────────────────────────────────────────────────────────
STEP_PATTERN = re.compile(
    r'Step\s+(\d+)/(\d+):\s+loss=([\d.]+)\s+'
    r'T1=([\d.]+)%\s+T10=([\d.]+)%\s+'
    r'(?:T=([\d.]+)\s+)?'
    r'\((\d+)s\)'
    r'(\s+\*\*\* NEW BEST \*\*\*)?'
)

EXP_PATTERN = re.compile(
    r'(?:Experiment \d+:|EXPERIMENT \d+:)\s*(.*?)$', re.MULTILINE
)

PARAMS_PATTERN = re.compile(r'Trainable params:\s+([\d,]+)')

FINAL_PATTERN = re.compile(
    r'FINAL:\s+T1=([\d.]+)%\s+T10=([\d.]+)%\s+\((\d+)s\)'
)


def parse_log(text: str) -> list[Experiment]:
    """Parse training log and extract experiments with data points."""
    experiments = []
    current_exp = Experiment(name="Training")

    lines = text.split('\n')
    for line in lines:
        # Check for new experiment
        exp_match = EXP_PATTERN.search(line)
        if exp_match:
            if current_exp.points:
                experiments.append(current_exp)
            current_exp = Experiment(name=exp_match.group(1).strip())
            continue

        # Check for params
        params_match = PARAMS_PATTERN.search(line)
        if params_match:
            current_exp.trainable_params = int(params_match.group(1).replace(',', ''))

        # Check for training step
        step_match = STEP_PATTERN.search(line)
        if step_match:
            point = TrainPoint(
                step=int(step_match.group(1)),
                loss=float(step_match.group(3)),
                t1=float(step_match.group(4)),
                t10=float(step_match.group(5)),
                temperature=float(step_match.group(6)) if step_match.group(6) else None,
                elapsed=float(step_match.group(7)),
                new_best=bool(step_match.group(8)),
            )
            current_exp.points.append(point)

    if current_exp.points:
        experiments.append(current_exp)

    return experiments


def parse_log_file(path: str) -> list[Experiment]:
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        return parse_log(f.read())


# ── Plotting ──────────────────────────────────────────────────────────
COLORS = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800', '#00BCD4']


def plot_training_progress(experiments: list[Experiment], title: str, out_path: str):
    """Plot T1, T10, and loss for one or more experiments."""
    n_exp = len(experiments)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    for i, exp in enumerate(experiments):
        steps = [p.step for p in exp.points]
        t1s = [p.t1 for p in exp.points]
        t10s = [p.t10 for p in exp.points]
        losses = [p.loss for p in exp.points]
        color = COLORS[i % len(COLORS)]
        label = exp.name if n_exp > 1 else None

        axes[0].plot(steps, t1s, 'o-', color=color, label=label, markersize=4)
        axes[1].plot(steps, t10s, 's-', color=color, label=label, markersize=4)
        axes[2].plot(steps[1:], losses[1:], '-', color=color, label=label, linewidth=1.5)

    # T1
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('T1 (%)')
    axes[0].set_title('Top-1 Agreement')
    axes[0].axhline(y=46, color='red', linestyle='--', alpha=0.5, label='Record (46%)')
    axes[0].grid(True, alpha=0.3)
    if n_exp > 1:
        axes[0].legend(fontsize=8)

    # T10
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('T10 (%)')
    axes[1].set_title('Top-10 Agreement')
    axes[1].axhline(y=67, color='red', linestyle='--', alpha=0.5, label='Record (67%)')
    axes[1].grid(True, alpha=0.3)
    if n_exp > 1:
        axes[1].legend(fontsize=8)

    # Loss
    axes[2].set_xlabel('Step')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('Training Loss')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    if n_exp > 1:
        axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_comparison_bar(experiments: list[Experiment], out_path: str):
    """Bar chart comparing final T1/T10 across experiments."""
    if len(experiments) < 2:
        return

    names = [exp.name for exp in experiments]
    t1_finals = [exp.points[-1].t1 if exp.points else 0 for exp in experiments]
    t10_finals = [exp.points[-1].t10 if exp.points else 0 for exp in experiments]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, t1_finals, width, label='T1 (%)', color='#2196F3')
    ax.bar(x + width / 2, t10_finals, width, label='T10 (%)', color='#FF5722')

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Selective Student — Experiment Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, (v1, v10) in enumerate(zip(t1_finals, t10_finals)):
        ax.text(i - width / 2, v1 + 0.5, f'{v1:.1f}%', ha='center', va='bottom', fontsize=9)
        ax.text(i + width / 2, v10 + 0.5, f'{v10:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_records_timeline(out_path: str):
    """Plot the records timeline from STATUS.md data."""
    records = {
        'Day 1 FRR': {'t1': 44, 't10': 62},
        'Day 2 1.7B random': {'t1': 35, 't10': 67},
        'Day 2 1.7B real 50K': {'t1': 46, 't10': 62},
        'Day 4 1.7B real 10K': {'t1': 47, 't10': 62.4},
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    x = list(range(len(records)))
    names = list(records.keys())
    t1s = [r['t1'] for r in records.values()]
    t10s = [r['t10'] for r in records.values()]

    ax.plot(x, t1s, 'o-', color='#2196F3', label='T1 (%)', markersize=8)
    ax.plot(x, t10s, 's-', color='#FF5722', label='T10 (%)', markersize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('UltraCompress — Records Timeline')
    ax.legend()
    ax.grid(True, alpha=0.3)

    for i, (v1, v10) in enumerate(zip(t1s, t10s)):
        ax.annotate(f'{v1}%', (i, v1), textcoords="offset points", xytext=(0, 10),
                    ha='center', fontsize=9, color='#2196F3')
        ax.annotate(f'{v10}%', (i, v10), textcoords="offset points", xytext=(0, -15),
                    ha='center', fontsize=9, color='#FF5722')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", nargs="*", help="Specific log files to plot")
    args = parser.parse_args()

    print("=" * 60)
    print("ULTRACOMPRESS — Results Visualization")
    print("=" * 60)

    os.makedirs("plots", exist_ok=True)

    # Auto-discover log files
    log_files = args.log if args.log else [
        f for f in os.listdir('.')
        if f.endswith('_output.log') or f.endswith('_log.txt')
    ]

    for log_file in log_files:
        if not os.path.exists(log_file):
            print(f"  Skipping {log_file} (not found)")
            continue

        print(f"\nParsing: {log_file}")
        experiments = parse_log_file(log_file)
        if not experiments:
            print(f"  No training data found in {log_file}")
            continue

        for exp in experiments:
            print(f"  Found: {exp.name} ({len(exp.points)} points, "
                  f"{exp.trainable_params:,} params)")

        base = os.path.splitext(log_file)[0]
        plot_training_progress(
            experiments,
            title=base.replace('_', ' ').title(),
            out_path=f"plots/{base}_curves.png",
        )

        if len(experiments) >= 2:
            plot_comparison_bar(experiments, f"plots/{base}_comparison.png")

    # Always generate records timeline
    print("\nGenerating records timeline...")
    plot_records_timeline("plots/records_timeline.png")

    print("\nDone! All plots in plots/")


if __name__ == "__main__":
    main()
