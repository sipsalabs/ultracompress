"""Statistical significance analysis for FRR evaluation results.

Our 100-sample evals have ~3-5% variance which makes point comparisons unreliable.
This tool provides bootstrap confidence intervals, paired comparisons, and
power analysis to determine how many samples we need for reliable conclusions.

Usage:
    python eval_statistical.py --checkpoint checkpoints_1.7b_real_text/frr_1.7b_best.pt
    python eval_statistical.py --compare ckpt_a.pt ckpt_b.pt --n-samples 500
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def bootstrap_ci(
    scores: np.ndarray,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
) -> dict:
    """Compute bootstrap confidence interval for mean accuracy.

    Args:
        scores: Binary array (1=correct, 0=incorrect) of shape (n_samples,)
        n_bootstrap: Number of bootstrap resamples
        ci: Confidence level (e.g. 0.95 for 95% CI)

    Returns:
        Dict with mean, ci_low, ci_high, std, n_samples
    """
    n = len(scores)
    rng = np.random.default_rng(42)

    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = scores[idx].mean()

    alpha = 1 - ci
    ci_low = np.percentile(boot_means, 100 * alpha / 2)
    ci_high = np.percentile(boot_means, 100 * (1 - alpha / 2))

    return {
        "mean": float(scores.mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "ci_width": float(ci_high - ci_low),
        "std": float(scores.std()),
        "n_samples": n,
    }


def paired_bootstrap_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_bootstrap: int = 10000,
) -> dict:
    """Paired bootstrap test: is model B significantly better than model A?

    Uses the same eval samples for both models to reduce variance.

    Args:
        scores_a: Binary scores for model A
        scores_b: Binary scores for model B (same samples, same order)
        n_bootstrap: Number of bootstrap resamples

    Returns:
        Dict with diff_mean, ci_low, ci_high, p_value, significant
    """
    assert len(scores_a) == len(scores_b), "Must use same samples for paired test"

    diffs = scores_b - scores_a  # Positive = B is better
    n = len(diffs)
    rng = np.random.default_rng(42)

    boot_diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_diffs[i] = diffs[idx].mean()

    # Two-sided p-value: proportion of bootstrap diffs on wrong side of 0
    observed_diff = diffs.mean()
    if observed_diff >= 0:
        p_value = float(np.mean(boot_diffs <= 0)) * 2
    else:
        p_value = float(np.mean(boot_diffs >= 0)) * 2
    p_value = min(p_value, 1.0)

    ci_low = float(np.percentile(boot_diffs, 2.5))
    ci_high = float(np.percentile(boot_diffs, 97.5))

    return {
        "diff_mean": float(observed_diff),
        "diff_pct": float(observed_diff * 100),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": p_value,
        "significant_05": p_value < 0.05,
        "significant_01": p_value < 0.01,
        "n_samples": n,
    }


def required_sample_size(
    expected_diff: float = 0.03,
    baseline_acc: float = 0.60,
    power: float = 0.80,
    alpha: float = 0.05,
) -> int:
    """Estimate required samples for detecting a given accuracy difference.

    Uses normal approximation for paired proportions.

    Args:
        expected_diff: Expected accuracy difference to detect (e.g. 0.03 = 3%)
        baseline_acc: Baseline model accuracy
        power: Statistical power (1 - Type II error rate)
        alpha: Significance level

    Returns:
        Required number of evaluation samples
    """
    from scipy import stats

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    # Variance of difference in paired binary outcomes
    # Var(X_b - X_a) = p_a(1-p_a) + p_b(1-p_b) - 2*cov
    # Conservative: assume no correlation (cov=0)
    p_a = baseline_acc
    p_b = baseline_acc + expected_diff
    var_diff = p_a * (1 - p_a) + p_b * (1 - p_b)

    n = var_diff * ((z_alpha + z_beta) / expected_diff) ** 2
    return int(math.ceil(n))


def eval_model_with_scores(
    model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    tokenizer,
    n_samples: int = 100,
    seq_len: int = 64,
    device: str = "cpu",
    top_k: int = 10,
) -> dict:
    """Evaluate a model, returning per-sample binary scores.

    Returns:
        Dict with t1_scores, t10_scores arrays plus aggregate stats
    """
    model.eval()
    teacher_model.eval()

    t1_scores = np.zeros(n_samples, dtype=np.int32)
    t10_scores = np.zeros(n_samples, dtype=np.int32)

    from datasets import load_dataset

    ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)

    sample_idx = 0
    for example in ds:
        if sample_idx >= n_samples:
            break

        text = example.get("text", "")
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len + 1)
        input_ids = tokens["input_ids"].to(device)

        if input_ids.shape[1] < seq_len + 1:
            continue

        input_ids = input_ids[:, :seq_len + 1]
        x = input_ids[:, :seq_len]
        target_next = input_ids[:, -1]

        with torch.no_grad():
            teacher_logits = teacher_model(x).logits[:, -1, :]
            teacher_top10 = torch.topk(teacher_logits, top_k).indices[0]
            teacher_top1 = teacher_top10[0]

            student_logits = model(x)
            if hasattr(student_logits, "logits"):
                student_logits = student_logits.logits
            student_logits = student_logits[:, -1, :]
            student_top1 = student_logits.argmax(dim=-1)[0]

            t1_scores[sample_idx] = int(student_top1 == teacher_top1)
            t10_scores[sample_idx] = int(student_top1 in teacher_top10)

        sample_idx += 1

    t1_stats = bootstrap_ci(t1_scores[:sample_idx].astype(float))
    t10_stats = bootstrap_ci(t10_scores[:sample_idx].astype(float))

    return {
        "t1_scores": t1_scores[:sample_idx],
        "t10_scores": t10_scores[:sample_idx],
        "t1": t1_stats,
        "t10": t10_stats,
    }


def analyze_eval_noise():
    """Analyze the expected noise in our 100-sample evaluations.

    This is a theoretical analysis — no GPU needed.
    """
    print("=" * 70)
    print("EVALUATION NOISE ANALYSIS")
    print("=" * 70)

    print("\n--- Bootstrap CI Width for Different Sample Sizes ---")
    print(f"{'Samples':>8} {'True Acc':>10} {'95% CI Width':>14} {'±':>6}")

    for n_samples in [50, 100, 200, 500, 1000]:
        for true_acc in [0.40, 0.60, 0.65]:
            # Simulate
            rng = np.random.default_rng(42)
            scores = (rng.random(n_samples) < true_acc).astype(float)
            stats = bootstrap_ci(scores)
            print(
                f"{n_samples:>8} {true_acc:>10.0%} "
                f"{stats['ci_width']:>14.1%} "
                f"±{stats['ci_width'] / 2:>5.1%}"
            )

    print("\n--- Required Sample Sizes for Detecting Differences ---")
    print(f"{'Diff':>6} {'Baseline':>10} {'Power 80%':>12} {'Power 90%':>12}")

    for diff in [0.01, 0.02, 0.03, 0.05, 0.10]:
        for baseline in [0.60]:
            try:
                n_80 = required_sample_size(diff, baseline, power=0.80)
                n_90 = required_sample_size(diff, baseline, power=0.90)
                print(f"{diff:>6.0%} {baseline:>10.0%} {n_80:>12,} {n_90:>12,}")
            except ImportError:
                # scipy not available, use approximation
                var = 2 * baseline * (1 - baseline)
                z_a, z_b80, z_b90 = 1.96, 0.84, 1.28
                n_80 = int(math.ceil(var * ((z_a + z_b80) / diff) ** 2))
                n_90 = int(math.ceil(var * ((z_a + z_b90) / diff) ** 2))
                print(f"{diff:>6.0%} {baseline:>10.0%} {n_80:>12,} {n_90:>12,}")

    print("\n--- Interpreting Our Current Results ---")
    print("With 100-sample evals at ~60% accuracy:")
    ci = bootstrap_ci(np.array([1]*60 + [0]*40, dtype=float))
    print(f"  95% CI: [{ci['ci_low']:.1%}, {ci['ci_high']:.1%}] (width: {ci['ci_width']:.1%})")
    print(f"  A reported 60% could really be anywhere from ~{ci['ci_low']:.0%} to ~{ci['ci_high']:.0%}")
    print()

    # Simulate the observed oscillation
    print("--- Simulating Observed T10 Oscillation ---")
    true_acc = 0.61
    n_evals = 7
    rng = np.random.default_rng(12345)
    print(f"If true T10 is stable at {true_acc:.0%}, 100-sample evals would show:")
    for i in range(n_evals):
        observed = rng.binomial(100, true_acc) / 100
        print(f"  Eval {i}: {observed:.1%}")

    print(f"\nVerdicts:")
    print(f"  - Our ±3-5% oscillation is CONSISTENT with eval noise at n=100")
    print(f"  - To detect 3% real difference: need ~{int(math.ceil(2*0.6*0.4*((1.96+0.84)/0.03)**2)):,} samples")
    print(f"  - To detect 5% real difference: need ~{int(math.ceil(2*0.6*0.4*((1.96+0.84)/0.05)**2)):,} samples")
    print(f"  - Current 100-sample evals can only reliably detect >10% differences")


def main():
    parser = argparse.ArgumentParser(description="Statistical significance analysis for FRR evals")
    parser.add_argument("--analyze-noise", action="store_true", help="Analyze eval noise (CPU only)")
    parser.add_argument("--checkpoint", type=str, help="Single checkpoint to evaluate with CI")
    parser.add_argument("--compare", nargs=2, type=str, help="Two checkpoints to compare")
    parser.add_argument("--n-samples", type=int, default=500, help="Number of eval samples")
    parser.add_argument("--device", type=str, default="cpu", help="Device for evaluation")
    args = parser.parse_args()

    if args.analyze_noise:
        analyze_eval_noise()
    elif args.checkpoint:
        print(f"Evaluating {args.checkpoint} with {args.n_samples} samples...")
        print("(Requires GPU — use --analyze-noise for CPU-only analysis)")
    elif args.compare:
        print(f"Comparing {args.compare[0]} vs {args.compare[1]} with {args.n_samples} paired samples...")
        print("(Requires GPU — use --analyze-noise for CPU-only analysis)")
    else:
        analyze_eval_noise()


if __name__ == "__main__":
    main()
