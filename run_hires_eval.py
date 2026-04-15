"""High-resolution evaluation with proper statistical confidence.

Standard evals use 100 samples (±9.5% CI). This script uses 500-1000 samples
for reliable comparisons, with paired evaluation across checkpoints.

Usage:
    python run_hires_eval.py --checkpoint checkpoints_1.7b_real_text/frr_1.7b_best.pt --teacher 1.7b
    python run_hires_eval.py --compare ckpt_a.pt ckpt_b.pt --teacher 0.6b --n-samples 500
    python run_hires_eval.py --all-checkpoints checkpoints_1.7b_real_text/ --teacher 1.7b
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_teacher(scale: str, device: str = "cuda:0"):
    """Load teacher model and tokenizer."""
    model_name = {
        "0.6b": "Qwen/Qwen3-0.6B",
        "1.7b": "Qwen/Qwen3-1.7B",
    }[scale.lower()]

    print(f"Loading teacher {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, trust_remote_code=True
    ).to(device)
    teacher.eval()

    config = teacher.config
    hidden = config.hidden_size
    num_heads = config.num_attention_heads
    head_dim = hidden // num_heads
    ffn_hidden = config.intermediate_size
    vocab = config.vocab_size

    print(f"  Hidden: {hidden}, Heads: {num_heads}, HeadDim: {head_dim}, Vocab: {vocab}")
    return teacher, tokenizer, {
        "hidden_size": hidden,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "ffn_hidden": ffn_hidden,
        "vocab_size": vocab,
    }


def load_frr_model(checkpoint_path: str, teacher_config: dict, device: str = "cuda:0"):
    """Load FRR model from checkpoint."""
    from ultracompress.moonshot import FractalModel

    model = FractalModel(
        hidden_dim=teacher_config["hidden_size"],
        n_heads=teacher_config["num_heads"],
        n_scales=4,
        iters_per_scale=7,
        vocab_size=teacher_config["vocab_size"],
    ).to(device)

    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  FRR loaded: {n_params:,} params from {checkpoint_path}")
    return model


def sys_path_hack():
    """Ensure frr_model is importable."""
    import sys
    project_root = str(Path(__file__).parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def get_eval_samples(tokenizer, n_samples: int, seq_len: int = 64, seed: int = 42):
    """Get deterministic eval samples from FineWeb-Edu.

    Returns list of (input_ids, target_next_token) tuples.
    Using a fixed seed ensures paired comparison across checkpoints.
    """
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)

    samples = []
    for example in ds:
        if len(samples) >= n_samples:
            break

        text = example.get("text", "")
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len + 1)
        input_ids = tokens["input_ids"]

        if input_ids.shape[1] < seq_len + 1:
            continue

        input_ids = input_ids[:, :seq_len + 1]
        samples.append(input_ids)

    print(f"  Collected {len(samples)} eval samples")
    return samples


def evaluate_model(
    model,
    teacher,
    samples: list,
    device: str = "cuda:0",
    top_k: int = 10,
    batch_size: int = 8,
) -> dict:
    """Evaluate model on pre-collected samples, returning per-sample scores."""
    n = len(samples)
    t1_scores = np.zeros(n, dtype=np.int32)
    t10_scores = np.zeros(n, dtype=np.int32)

    model.eval()
    teacher.eval()

    for i in range(0, n, batch_size):
        batch_end = min(i + batch_size, n)
        batch_ids = torch.cat([s[:, :-1] for s in samples[i:batch_end]], dim=0).to(device)

        with torch.no_grad(), torch.amp.autocast("cuda"):
            teacher_logits = teacher(batch_ids).logits[:, -1, :]
            student_logits = model(batch_ids)
            if hasattr(student_logits, "logits"):
                student_logits = student_logits.logits
            student_logits = student_logits[:, -1, :]

        for j in range(batch_end - i):
            t_logits = teacher_logits[j]
            s_logits = student_logits[j]

            teacher_topk = torch.topk(t_logits, top_k).indices
            teacher_top1 = teacher_topk[0]
            student_top1 = s_logits.argmax()

            idx = i + j
            t1_scores[idx] = int(student_top1 == teacher_top1)
            t10_scores[idx] = int(student_top1 in teacher_topk)

        if (i // batch_size) % 10 == 0:
            current = min(batch_end, n)
            t1_so_far = t1_scores[:current].mean() * 100
            t10_so_far = t10_scores[:current].mean() * 100
            print(f"    [{current}/{n}] T1={t1_so_far:.1f}% T10={t10_so_far:.1f}%")

    return {
        "t1_scores": t1_scores,
        "t10_scores": t10_scores,
        "t1_mean": float(t1_scores.mean()),
        "t10_mean": float(t10_scores.mean()),
    }


def bootstrap_ci(scores: np.ndarray, n_bootstrap: int = 10000, ci: float = 0.95) -> dict:
    """Bootstrap confidence interval."""
    rng = np.random.default_rng(42)
    n = len(scores)
    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = scores[idx].mean()

    alpha = 1 - ci
    return {
        "mean": float(scores.mean()) * 100,
        "ci_low": float(np.percentile(boot_means, 100 * alpha / 2)) * 100,
        "ci_high": float(np.percentile(boot_means, 100 * (1 - alpha / 2))) * 100,
        "ci_width": float(np.percentile(boot_means, 100 * (1 - alpha / 2)) -
                         np.percentile(boot_means, 100 * alpha / 2)) * 100,
        "n": n,
    }


def paired_test(scores_a: np.ndarray, scores_b: np.ndarray, n_bootstrap: int = 10000) -> dict:
    """Paired bootstrap comparison."""
    diffs = scores_b.astype(float) - scores_a.astype(float)
    n = len(diffs)
    rng = np.random.default_rng(42)

    boot_diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_diffs[i] = diffs[idx].mean()

    observed = diffs.mean()
    if observed >= 0:
        p_value = float(np.mean(boot_diffs <= 0)) * 2
    else:
        p_value = float(np.mean(boot_diffs >= 0)) * 2
    p_value = min(p_value, 1.0)

    return {
        "diff_pct": float(observed) * 100,
        "ci_low": float(np.percentile(boot_diffs, 2.5)) * 100,
        "ci_high": float(np.percentile(boot_diffs, 97.5)) * 100,
        "p_value": p_value,
        "significant": p_value < 0.05,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, help="Single checkpoint to evaluate")
    parser.add_argument("--compare", nargs=2, type=str, help="Two checkpoints to compare (paired)")
    parser.add_argument("--all-checkpoints", type=str, help="Directory of checkpoints to sweep")
    parser.add_argument("--teacher", type=str, default="0.6b", choices=["0.6b", "1.7b"])
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default="hires_eval_results.json")
    args = parser.parse_args()

    teacher, tokenizer, config = load_teacher(args.teacher, args.device)
    samples = get_eval_samples(tokenizer, args.n_samples)

    results = {"teacher": args.teacher, "n_samples": len(samples), "evaluations": []}

    if args.checkpoint:
        print(f"\n--- Evaluating {args.checkpoint} ---")
        model = load_frr_model(args.checkpoint, config, args.device)
        scores = evaluate_model(model, teacher, samples, args.device)

        t1_ci = bootstrap_ci(scores["t1_scores"])
        t10_ci = bootstrap_ci(scores["t10_scores"])

        print(f"\nResults ({len(samples)} samples):")
        print(f"  T1:  {t1_ci['mean']:.1f}% [{t1_ci['ci_low']:.1f}%, {t1_ci['ci_high']:.1f}%] (±{t1_ci['ci_width']/2:.1f}%)")
        print(f"  T10: {t10_ci['mean']:.1f}% [{t10_ci['ci_low']:.1f}%, {t10_ci['ci_high']:.1f}%] (±{t10_ci['ci_width']/2:.1f}%)")

        results["evaluations"].append({
            "checkpoint": args.checkpoint,
            "t1": t1_ci,
            "t10": t10_ci,
        })
        del model
        torch.cuda.empty_cache()

    elif args.compare:
        print(f"\n--- Paired comparison ---")
        model_a = load_frr_model(args.compare[0], config, args.device)
        scores_a = evaluate_model(model_a, teacher, samples, args.device)
        del model_a
        torch.cuda.empty_cache()

        model_b = load_frr_model(args.compare[1], config, args.device)
        scores_b = evaluate_model(model_b, teacher, samples, args.device)
        del model_b
        torch.cuda.empty_cache()

        t10_test = paired_test(scores_a["t10_scores"], scores_b["t10_scores"])
        t1_test = paired_test(scores_a["t1_scores"], scores_b["t1_scores"])

        print(f"\nPaired comparison ({len(samples)} shared samples):")
        print(f"  Model A: {args.compare[0]}")
        print(f"  Model B: {args.compare[1]}")
        print(f"  T10 diff: {t10_test['diff_pct']:+.1f}% [{t10_test['ci_low']:+.1f}%, {t10_test['ci_high']:+.1f}%]")
        print(f"  T10 p-value: {t10_test['p_value']:.4f} {'***' if t10_test['significant'] else 'n.s.'}")
        print(f"  T1  diff: {t1_test['diff_pct']:+.1f}% [{t1_test['ci_low']:+.1f}%, {t1_test['ci_high']:+.1f}%]")
        print(f"  T1  p-value: {t1_test['p_value']:.4f} {'***' if t1_test['significant'] else 'n.s.'}")

        results["comparison"] = {
            "model_a": args.compare[0],
            "model_b": args.compare[1],
            "t10": t10_test,
            "t1": t1_test,
        }

    elif args.all_checkpoints:
        ckpt_dir = Path(args.all_checkpoints)
        ckpts = sorted(ckpt_dir.glob("*.pt"))
        print(f"\n--- Evaluating {len(ckpts)} checkpoints from {ckpt_dir} ---")

        all_scores = {}
        for ckpt in ckpts:
            print(f"\n  {ckpt.name}:")
            model = load_frr_model(str(ckpt), config, args.device)
            scores = evaluate_model(model, teacher, samples, args.device)

            t1_ci = bootstrap_ci(scores["t1_scores"])
            t10_ci = bootstrap_ci(scores["t10_scores"])

            print(f"    T1:  {t1_ci['mean']:.1f}% ±{t1_ci['ci_width']/2:.1f}%")
            print(f"    T10: {t10_ci['mean']:.1f}% ±{t10_ci['ci_width']/2:.1f}%")

            all_scores[ckpt.name] = scores
            results["evaluations"].append({
                "checkpoint": ckpt.name,
                "t1": t1_ci,
                "t10": t10_ci,
            })

            del model
            torch.cuda.empty_cache()

        # Paired comparisons between consecutive checkpoints
        ckpt_names = list(all_scores.keys())
        if len(ckpt_names) >= 2:
            print(f"\n--- Paired comparisons (consecutive checkpoints) ---")
            for i in range(len(ckpt_names) - 1):
                a, b = ckpt_names[i], ckpt_names[i + 1]
                t10_test = paired_test(all_scores[a]["t10_scores"], all_scores[b]["t10_scores"])
                sig = "***" if t10_test["significant"] else "n.s."
                print(f"  {a} → {b}: T10 {t10_test['diff_pct']:+.1f}% (p={t10_test['p_value']:.3f}) {sig}")

    # Save results
    output_path = Path(args.output)
    # Convert numpy arrays for JSON serialization
    def clean_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean_for_json(v) for v in obj]
        return obj

    with open(output_path, "w") as f:
        json.dump(clean_for_json(results), f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
