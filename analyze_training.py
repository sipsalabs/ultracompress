"""Training curve analysis and visualization for UltraCompress FRR experiments.

Parses training logs, computes statistics, fits convergence curves, and generates
publication-quality plots (saved as PNG). Runs CPU-only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


@dataclass
class TrainingPoint:
    step: int
    loss: float
    t1: float
    t10: float
    temp: float | None = None
    elapsed: float | None = None


@dataclass
class Experiment:
    name: str
    points: list[TrainingPoint] = field(default_factory=list)
    total_steps: int = 0
    params: int = 0


# ---------------------------------------------------------------------------
# Hardcoded data (from terminal output)
# ---------------------------------------------------------------------------

def get_1_7b_real_text() -> Experiment:
    """1.7B real text 100K training data (through 35K)."""
    exp = Experiment(name="1.7B Real Text 100K", total_steps=100_000, params=29_380_636)
    data = [
        (0, 561.92, 5.0, 21.4, 5.0, 12),
        (5000, 41.33, 32.0, 61.4, 4.8, 644),
        (10000, 37.56, 47.0, 62.4, 4.5, 1276),
        (15000, 37.44, 41.0, 61.0, 4.2, 1928),
        (20000, 37.23, 33.0, 60.3, 4.0, 2582),
        (25000, 37.94, 40.0, 57.2, 3.8, 3226),
        (30000, 38.49, 37.0, 61.4, 3.5, 3874),
        (35000, 38.94, 42.0, 61.3, 3.2, 4532),
    ]
    for step, loss, t1, t10, temp, elapsed in data:
        exp.points.append(TrainingPoint(step, loss, t1, t10, temp, elapsed))
    return exp


def get_selective_baseline() -> Experiment:
    """Selective Student Exp 1 — Standard KL baseline (0.6B)."""
    exp = Experiment(name="Baseline (Standard KL)", total_steps=15_000, params=7_350_300)
    data = [
        (0, 606.58, 4.0, 18.7, None, 11),
        (3000, 56.33, 47.0, 58.4, None, 420),
        (6000, 54.26, 42.0, 57.1, None, 918),
        (9000, 69.40, 39.0, 57.5, None, 1360),
        (12000, 58.13, 53.0, 59.8, None, 1919),
        (14999, 48.16, 42.0, 58.9, None, 2566),
    ]
    for step, loss, t1, t10, temp, elapsed in data:
        exp.points.append(TrainingPoint(step, loss, t1, t10, temp, elapsed))
    return exp


def get_selective_trustgate() -> Experiment:
    """Selective Student Exp 2 — TrustGate (0.6B)."""
    exp = Experiment(name="TrustGate (Selective)", total_steps=15_000, params=7_350_621)
    data = [
        (0, 9.84, 5.0, 19.6, None, 10),
        (3000, 0.86, 44.0, 49.7, None, 700),
        (6000, 1.19, 43.0, 55.2, None, 1337),
        (9000, 0.94, 49.0, 59.1, None, 1962),
        (12000, 1.01, 43.0, 62.2, None, 2515),
    ]
    for step, loss, t1, t10, temp, elapsed in data:
        exp.points.append(TrainingPoint(step, loss, t1, t10, temp, elapsed))
    return exp


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

LOG_PATTERN = re.compile(
    r"Step\s+(\d+)/(\d+):\s+loss=([\d.]+)\s+T1=([\d.]+)%\s+T10=([\d.]+)%"
    r"(?:\s+T=([\d.]+))?"
    r"(?:\s+\((\d+)s\))?"
)


def parse_log_file(path: Path) -> Experiment:
    """Parse a training log file into an Experiment."""
    exp = Experiment(name=path.stem)
    text = path.read_text(encoding="utf-8", errors="replace")
    for m in LOG_PATTERN.finditer(text):
        step = int(m.group(1))
        total = int(m.group(2))
        loss = float(m.group(3))
        t1 = float(m.group(4))
        t10 = float(m.group(5))
        temp = float(m.group(6)) if m.group(6) else None
        elapsed = float(m.group(7)) if m.group(7) else None
        exp.points.append(TrainingPoint(step, loss, t1, t10, temp, elapsed))
        exp.total_steps = total
    return exp


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def compute_statistics(exp: Experiment) -> dict:
    """Compute summary statistics for an experiment."""
    if not exp.points:
        return {}

    t10_vals = [p.t10 for p in exp.points if p.step > 0]
    t1_vals = [p.t1 for p in exp.points if p.step > 0]
    loss_vals = [p.loss for p in exp.points if p.step > 0]

    best_t10 = max(exp.points, key=lambda p: p.t10)
    best_t1 = max(exp.points, key=lambda p: p.t1)

    # Compute plateau stats (last 50% of data)
    n = len(t10_vals)
    plateau_t10 = t10_vals[n // 2:]

    stats = {
        "name": exp.name,
        "total_points": len(exp.points),
        "best_t10": best_t10.t10,
        "best_t10_step": best_t10.step,
        "best_t1": best_t1.t1,
        "best_t1_step": best_t1.step,
        "final_t10": exp.points[-1].t10,
        "final_t1": exp.points[-1].t1,
        "final_loss": exp.points[-1].loss,
        "t10_mean": float(np.mean(t10_vals)),
        "t10_std": float(np.std(t10_vals)),
        "t10_plateau_mean": float(np.mean(plateau_t10)) if plateau_t10 else 0,
        "t10_plateau_std": float(np.std(plateau_t10)) if plateau_t10 else 0,
        "t1_mean": float(np.mean(t1_vals)),
        "t1_std": float(np.std(t1_vals)),
        "loss_trend": "decreasing" if len(loss_vals) > 1 and loss_vals[-1] < loss_vals[0] else "non-monotonic",
    }

    # Convergence rate: steps to reach 90% of best T10
    threshold = best_t10.t10 * 0.9
    for p in exp.points:
        if p.t10 >= threshold:
            stats["steps_to_90pct"] = p.step
            break

    # Time efficiency
    if exp.points[-1].elapsed and exp.points[-1].elapsed > 0:
        stats["steps_per_second"] = exp.points[-1].step / exp.points[-1].elapsed
        stats["total_hours"] = exp.points[-1].elapsed / 3600

    return stats


def compare_experiments(exp1: Experiment, exp2: Experiment) -> dict:
    """Compare two experiments at matched steps."""
    steps1 = {p.step: p for p in exp1.points}
    steps2 = {p.step: p for p in exp2.points}
    common = sorted(set(steps1.keys()) & set(steps2.keys()))

    comparisons = []
    for s in common:
        p1, p2 = steps1[s], steps2[s]
        comparisons.append({
            "step": s,
            "t10_diff": p2.t10 - p1.t10,
            "t1_diff": p2.t1 - p1.t1,
            "exp1_t10": p1.t10,
            "exp2_t10": p2.t10,
        })

    return {
        "exp1": exp1.name,
        "exp2": exp2.name,
        "common_steps": len(common),
        "comparisons": comparisons,
        "avg_t10_diff": float(np.mean([c["t10_diff"] for c in comparisons])) if comparisons else 0,
    }


def predict_convergence(exp: Experiment, target_steps: int | None = None) -> dict:
    """Fit a simple exponential saturation curve to predict T10 at target_steps."""
    steps = np.array([p.step for p in exp.points if p.step > 0], dtype=float)
    t10 = np.array([p.t10 for p in exp.points if p.step > 0], dtype=float)

    if len(steps) < 3:
        return {"error": "Need at least 3 data points"}

    # Fit: T10(s) = A * (1 - exp(-s/tau)) + offset
    # Linearize: log(A - T10) = log(A) - s/tau
    # Use iterative approach with A estimate from max observed
    a_est = max(t10) * 1.05  # slight overestimate
    y = np.log(np.maximum(a_est - t10, 0.1))

    # Linear fit: y = b - s/tau
    coeffs = np.polyfit(steps, y, 1)
    tau = -1.0 / coeffs[0] if coeffs[0] != 0 else 1e6
    b = coeffs[1]
    a_fit = np.exp(b) + t10.min()

    target = target_steps or exp.total_steps
    predicted_t10 = a_fit * (1 - np.exp(-target / max(tau, 1))) if tau > 0 else t10[-1]
    predicted_t10 = min(predicted_t10, 100.0)

    return {
        "model": "exponential_saturation",
        "a_asymptote": float(a_fit),
        "tau_steps": float(tau),
        "predicted_t10_at_target": float(predicted_t10),
        "target_steps": target,
        "current_t10": float(t10[-1]),
        "current_step": int(steps[-1]),
        "confidence": "low" if len(steps) < 5 else "medium",
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training_curves(experiments: list[Experiment], output: Path) -> None:
    """Plot T10, T1, and loss curves for multiple experiments."""
    if not HAS_MPL:
        print("matplotlib not available, skipping plots")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("UltraCompress FRR Training Analysis", fontsize=14, fontweight="bold")

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]

    # T10 curve
    ax = axes[0, 0]
    for i, exp in enumerate(experiments):
        steps = [p.step for p in exp.points]
        t10 = [p.t10 for p in exp.points]
        ax.plot(steps, t10, "o-", color=colors[i % len(colors)], label=exp.name, markersize=5)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Top-10 Agreement (%)")
    ax.set_title("Top-10 Agreement")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhspan(55, 65, alpha=0.1, color="green", label="Target zone")

    # T1 curve
    ax = axes[0, 1]
    for i, exp in enumerate(experiments):
        steps = [p.step for p in exp.points]
        t1 = [p.t1 for p in exp.points]
        ax.plot(steps, t1, "o-", color=colors[i % len(colors)], label=exp.name, markersize=5)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Top-1 Agreement (%)")
    ax.set_title("Top-1 Agreement")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Loss curve
    ax = axes[1, 0]
    for i, exp in enumerate(experiments):
        steps = [p.step for p in exp.points]
        loss = [p.loss for p in exp.points]
        ax.plot(steps, loss, "o-", color=colors[i % len(colors)], label=exp.name, markersize=5)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Temperature (1.7B only) or comparison delta
    ax = axes[1, 1]
    has_temp = False
    for i, exp in enumerate(experiments):
        temps = [(p.step, p.temp) for p in exp.points if p.temp is not None]
        if temps:
            has_temp = True
            steps, temp_vals = zip(*temps)
            ax.plot(steps, temp_vals, "o-", color=colors[i % len(colors)], label=f"{exp.name} temp", markersize=5)

    if has_temp:
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Temperature")
        ax.set_title("Temperature Annealing")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        # Show T10 difference between experiments
        if len(experiments) >= 2:
            exp1, exp2 = experiments[0], experiments[1]
            steps1 = {p.step: p.t10 for p in exp1.points}
            steps2 = {p.step: p.t10 for p in exp2.points}
            common = sorted(set(steps1.keys()) & set(steps2.keys()))
            if common:
                diffs = [steps2[s] - steps1[s] for s in common]
                ax.bar(range(len(common)), diffs, color=["#4CAF50" if d > 0 else "#FF5722" for d in diffs])
                ax.set_xticks(range(len(common)))
                ax.set_xticklabels([f"{s // 1000}K" for s in common], fontsize=8)
                ax.set_xlabel("Training Step")
                ax.set_ylabel("T10 Difference (%)")
                ax.set_title(f"T10 Delta: {exp2.name} − {exp1.name}")
                ax.axhline(y=0, color="black", linewidth=0.5)
                ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {output}")
    plt.close()


def plot_trustgate_trajectory(baseline: Experiment, trustgate: Experiment, output: Path) -> None:
    """Plot the TrustGate vs Baseline trajectory with CI bands."""
    if not HAS_MPL:
        print("matplotlib not available, skipping plots")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("TrustGate vs Standard KL Baseline — Selective Student Experiment", fontsize=13, fontweight="bold")

    # Left: T10 curves with CI bands
    for exp, color, label in [(baseline, "#2196F3", "Baseline"), (trustgate, "#FF5722", "TrustGate")]:
        steps = [p.step for p in exp.points]
        t10 = [p.t10 for p in exp.points]
        ax1.plot(steps, t10, "o-", color=color, label=label, markersize=7, linewidth=2)
        # CI band: ±9.5% at n=100
        ci = 9.5
        t10_arr = np.array(t10)
        ax1.fill_between(steps, t10_arr - ci, t10_arr + ci, alpha=0.1, color=color)

    ax1.set_xlabel("Training Step", fontsize=11)
    ax1.set_ylabel("Top-10 Agreement (%)", fontsize=11)
    ax1.set_title("T10 with 95% CI bands (n=100)")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 80)

    # Right: Delta plot
    steps_b = {p.step: p.t10 for p in baseline.points}
    steps_t = {p.step: p.t10 for p in trustgate.points}
    common = sorted(set(steps_b.keys()) & set(steps_t.keys()))
    if common:
        deltas = [steps_t[s] - steps_b[s] for s in common]
        colors = ["#4CAF50" if d > 0 else "#FF5722" for d in deltas]
        bars = ax2.bar(range(len(common)), deltas, color=colors, width=0.6, edgecolor="black", linewidth=0.5)
        ax2.set_xticks(range(len(common)))
        ax2.set_xticklabels([f"{s // 1000}K" for s in common], fontsize=10)
        ax2.set_xlabel("Training Step", fontsize=11)
        ax2.set_ylabel("T10 Difference (TrustGate − Baseline)", fontsize=11)
        ax2.set_title("TrustGate Advantage")
        ax2.axhline(y=0, color="black", linewidth=1)
        ax2.axhspan(-9.5, 9.5, alpha=0.08, color="gray", label="Within CI (n=100)")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, axis="y")

        for bar, d in zip(bars, deltas):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (0.3 if d > 0 else -1.5),
                     f"{d:+.1f}%", ha="center", va="bottom" if d > 0 else "top", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {output}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="UltraCompress FRR Training Analyzer")
    parser.add_argument("--log", type=Path, help="Parse a training log file")
    parser.add_argument("--builtin", action="store_true", help="Analyze built-in experiment data")
    parser.add_argument("--plot", action="store_true", help="Generate plots (requires matplotlib)")
    parser.add_argument("--predict", action="store_true", help="Predict convergence")
    parser.add_argument("--compare", action="store_true", help="Compare TrustGate vs baseline")
    parser.add_argument("--output-dir", type=Path, default=Path("paper_figures"), help="Output directory for plots")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()

    if not args.builtin and not args.log:
        args.builtin = True

    args.output_dir.mkdir(exist_ok=True)
    experiments: list[Experiment] = []

    if args.builtin:
        experiments = [get_1_7b_real_text(), get_selective_baseline(), get_selective_trustgate()]
    elif args.log:
        experiments = [parse_log_file(args.log)]

    # Statistics
    all_stats = []
    for exp in experiments:
        stats = compute_statistics(exp)
        all_stats.append(stats)
        if not args.json:
            print(f"\n{'=' * 60}")
            print(f"  {exp.name}")
            print(f"{'=' * 60}")
            print(f"  Best T10:  {stats['best_t10']:.1f}% (step {stats['best_t10_step']})")
            print(f"  Best T1:   {stats['best_t1']:.1f}% (step {stats['best_t1_step']})")
            print(f"  Final T10: {stats['final_t10']:.1f}%")
            print(f"  Final T1:  {stats['final_t1']:.1f}%")
            print(f"  T10 mean:  {stats['t10_mean']:.1f}% ± {stats['t10_std']:.1f}%")
            print(f"  Plateau:   {stats['t10_plateau_mean']:.1f}% ± {stats['t10_plateau_std']:.1f}%")
            if "steps_to_90pct" in stats:
                print(f"  90% of best reached at step {stats['steps_to_90pct']}")
            if "steps_per_second" in stats:
                print(f"  Speed:     {stats['steps_per_second']:.1f} steps/sec ({stats['total_hours']:.1f}h elapsed)")

    # Convergence prediction
    if args.predict:
        if not args.json:
            print(f"\n{'=' * 60}")
            print("  CONVERGENCE PREDICTIONS")
            print(f"{'=' * 60}")
        predictions = []
        for exp in experiments:
            pred = predict_convergence(exp)
            predictions.append(pred)
            if not args.json:
                if "error" in pred:
                    print(f"  {exp.name}: {pred['error']}")
                else:
                    print(f"  {exp.name}:")
                    print(f"    Asymptote estimate: {pred['a_asymptote']:.1f}%")
                    print(f"    Time constant: {pred['tau_steps']:.0f} steps")
                    print(f"    T10 at {pred['target_steps']}: {pred['predicted_t10_at_target']:.1f}% (confidence: {pred['confidence']})")

    # Comparison
    if args.compare:
        baseline = get_selective_baseline()
        trustgate = get_selective_trustgate()
        comp = compare_experiments(baseline, trustgate)
        if not args.json:
            print(f"\n{'=' * 60}")
            print(f"  COMPARISON: {comp['exp1']} vs {comp['exp2']}")
            print(f"{'=' * 60}")
            for c in comp["comparisons"]:
                sign = "+" if c["t10_diff"] > 0 else ""
                print(f"  Step {c['step']:>5}: {c['exp1_t10']:.1f}% vs {c['exp2_t10']:.1f}% ({sign}{c['t10_diff']:.1f}%)")
            print(f"  Average T10 delta: {comp['avg_t10_diff']:+.1f}%")
            print(f"  NOTE: Differences <10% are NOT statistically significant at n=100")

    # Plots
    if args.plot:
        plot_training_curves(experiments, args.output_dir / "training_curves.png")
        baseline = get_selective_baseline()
        trustgate = get_selective_trustgate()
        plot_trustgate_trajectory(baseline, trustgate, args.output_dir / "trustgate_trajectory.png")

    # JSON output
    if args.json:
        result = {
            "statistics": all_stats,
            "experiments": [
                {"name": exp.name, "points": [
                    {"step": p.step, "loss": p.loss, "t1": p.t1, "t10": p.t10, "temp": p.temp}
                    for p in exp.points
                ]} for exp in experiments
            ],
        }
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
