"""High-resolution evaluation with proper statistical confidence.

Standard evals use 100 samples (±9.5% CI). This script uses 500-1000 samples
for reliable comparisons, with paired evaluation across checkpoints.

Uses the SAME cached MiniTransformer teacher as training for consistency.

Usage:
    python run_hires_eval.py --checkpoint checkpoints_1.7b_real_text/frr_1.7b_best.pt --teacher 1.7b
    python run_hires_eval.py --compare ckpt_a.pt ckpt_b.pt --teacher 0.6b --n-samples 500
    python run_hires_eval.py --all-checkpoints checkpoints_1.7b_real_text/ --teacher 1.7b
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from ultracompress.inference import ModelConfig, MiniTransformer
from ultracompress.moonshot import FractalModel
from transformers import AutoTokenizer
from datasets import load_dataset

# ── Teacher configs (must match training scripts) ─────────────────────
TEACHER_CONFIGS = {
    '0.6b': {
        'cache': 'qwen3_0.6b_cache.pt',
        'hf_name': 'Qwen/Qwen3-0.6B',
        'hidden': 1024, 'n_heads': 16, 'n_kv_heads': 8, 'n_layers': 28,
        'intermediate_size': 3072, 'vocab_size': 151936, 'head_dim': 128,
    },
    '1.7b': {
        'cache': 'qwen3_1.7b_cache.pt',
        'hf_name': 'Qwen/Qwen3-1.7B',
        'hidden': 2048, 'n_heads': 16, 'n_kv_heads': 8, 'n_layers': 28,
        'intermediate_size': 6144, 'vocab_size': 151936, 'head_dim': 128,
    },
}

HF_TO_GGUF = {
    'self_attn.q_proj.weight': 'attn_q.weight',
    'self_attn.k_proj.weight': 'attn_k.weight',
    'self_attn.v_proj.weight': 'attn_v.weight',
    'self_attn.o_proj.weight': 'attn_output.weight',
    'self_attn.q_norm.weight': 'attn_q_norm.weight',
    'self_attn.k_norm.weight': 'attn_k_norm.weight',
    'input_layernorm.weight': 'attn_norm.weight',
    'post_attention_layernorm.weight': 'ffn_norm.weight',
    'mlp.gate_proj.weight': 'ffn_gate.weight',
    'mlp.up_proj.weight': 'ffn_up.weight',
    'mlp.down_proj.weight': 'ffn_down.weight',
}


def load_teacher(scale: str, device: str = "cuda:0"):
    """Load cached MiniTransformer teacher (same as training)."""
    cfg = TEACHER_CONFIGS[scale.lower()]
    print(f"Loading cached teacher ({scale})...")
    wd = torch.load(cfg['cache'], weights_only=True)

    gd = {}
    gd['token_embd.weight'] = wd['model.embed_tokens.weight'].float()
    gd['output_norm.weight'] = wd.get('model.norm.weight', torch.ones(cfg['hidden'])).float()
    gd['output.weight'] = wd.get('lm_head.weight', gd['token_embd.weight']).float()
    for li in range(cfg['n_layers']):
        for h, g in HF_TO_GGUF.items():
            k = f'model.layers.{li}.{h}'
            if k in wd:
                gd[f'blk.{li}.{g}'] = wd[k].float()
    del wd

    model_cfg = ModelConfig(
        n_layers=cfg['n_layers'], n_heads=cfg['n_heads'], n_kv_heads=cfg['n_kv_heads'],
        hidden_size=cfg['hidden'], intermediate_size=cfg['intermediate_size'],
        vocab_size=cfg['vocab_size'], head_dim=cfg['head_dim'],
    )
    teacher = MiniTransformer(model_cfg, device)
    teacher.load_weights(gd)
    teacher.embed_weight = teacher.embed_weight.to(device)
    if teacher.lm_head is not None:
        teacher.lm_head = teacher.lm_head.to(device)

    embed_w = gd['token_embd.weight'].to(device)
    norm_w = gd['output_norm.weight'].to(device)
    lm_head_w = gd['output.weight'].to(device)
    del gd

    tokenizer = AutoTokenizer.from_pretrained(cfg['hf_name'], trust_remote_code=True)
    print(f"  Hidden: {cfg['hidden']}, Heads: {cfg['n_heads']}, HeadDim: {cfg['head_dim']}, Vocab: {cfg['vocab_size']}")
    return teacher, tokenizer, cfg, embed_w, norm_w, lm_head_w


def load_frr_model(checkpoint_path: str, cfg: dict, embed_w, norm_w, lm_head_w, device: str = "cuda:0"):
    """Load FRR model from checkpoint, inferring ff_mult from weight shapes."""
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "model_state_dict" in state:
        state = state["model_state_dict"]

    hidden = cfg["hidden"]
    ff_mult = 1
    per_layer_mod = False
    for key, tensor in state.items():
        if "ffn" in key.lower() or "ff" in key.lower():
            max_dim = max(tensor.shape)
            if max_dim > hidden:
                ff_mult = max_dim // hidden
            break
    if "layer_gamma" in state:
        per_layer_mod = True

    model = FractalModel(
        hidden_dim=hidden,
        n_heads=cfg["n_heads"],
        n_scales=4,
        iters_per_scale=7,
        vocab_size=cfg["vocab_size"],
        ff_mult=ff_mult,
        embed_weight=embed_w,
        lm_head_weight=lm_head_w,
        norm_weight=norm_w,
        per_layer_mod=per_layer_mod,
    ).to(device)

    model.load_state_dict(state, strict=False)
    model.eval()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mod_type = "per-layer" if per_layer_mod else "per-scale"
    print(f"  FRR loaded: {trainable:,} trainable params (ff_mult={ff_mult}, {mod_type})")
    return model


def get_eval_samples(tokenizer, n_samples: int, seq_len: int = 32, seed: int = 42):
    """Get deterministic eval samples from FineWeb-Edu.

    Uses seq_len=32 to match training eval. Returns list of input_ids tensors.
    Using a fixed seed ensures paired comparison across checkpoints.
    """
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)

    samples = []
    for example in ds:
        if len(samples) >= n_samples:
            break

        text = example.get("text", "")
        if len(text) < 200:
            continue
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
        input_ids = tokens["input_ids"]

        if input_ids.shape[1] < seq_len:
            continue

        input_ids = input_ids[:, :seq_len]
        samples.append(input_ids)

    print(f"  Collected {len(samples)} eval samples")
    return samples


def evaluate_model(
    model,
    teacher,
    n_layers: int,
    samples: list,
    device: str = "cuda:0",
    top_k: int = 10,
) -> dict:
    """Evaluate model on pre-collected samples using cached MiniTransformer teacher.

    Uses same T10 metric as training: |student_top10 ∩ teacher_top10| / 10.
    Evaluates at last position only (matching training eval).
    """
    n = len(samples)
    t1_scores = np.zeros(n, dtype=np.float64)
    t10_scores = np.zeros(n, dtype=np.float64)

    model.eval()

    for i in range(n):
        tokens = samples[i].to(device)

        with torch.no_grad():
            tl = teacher.forward(tokens, max_layers=n_layers)
            sl = model(tokens)

            # Last position logits (same as training eval)
            t_logits = tl[0, -1].float()
            s_logits = sl[0, -1].float()

            # T1: top-1 match
            t1_scores[i] = int(t_logits.argmax() == s_logits.argmax())

            # T10: overlap of top-K sets (same metric as training)
            t_topk = set(t_logits.topk(top_k).indices.tolist())
            s_topk = set(s_logits.topk(top_k).indices.tolist())
            t10_scores[i] = len(t_topk & s_topk) / top_k

        if (i + 1) % 50 == 0 or i == n - 1:
            t1_so_far = t1_scores[:i+1].mean() * 100
            t10_so_far = t10_scores[:i+1].mean() * 100
            print(f"    [{i+1}/{n}] T1={t1_so_far:.1f}% T10={t10_so_far:.1f}%")

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

    teacher, tokenizer, config, embed_w, norm_w, lm_head_w = load_teacher(args.teacher, args.device)
    samples = get_eval_samples(tokenizer, args.n_samples)
    n_layers = config['n_layers']

    results = {"teacher": args.teacher, "n_samples": len(samples), "evaluations": []}

    if args.checkpoint:
        print(f"\n--- Evaluating {args.checkpoint} ---")
        model = load_frr_model(args.checkpoint, config, embed_w, norm_w, lm_head_w, args.device)
        scores = evaluate_model(model, teacher, n_layers, samples, args.device)

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
        model_a = load_frr_model(args.compare[0], config, embed_w, norm_w, lm_head_w, args.device)
        scores_a = evaluate_model(model_a, teacher, n_layers, samples, args.device)
        del model_a
        torch.cuda.empty_cache()

        model_b = load_frr_model(args.compare[1], config, embed_w, norm_w, lm_head_w, args.device)
        scores_b = evaluate_model(model_b, teacher, n_layers, samples, args.device)
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
            model = load_frr_model(str(ckpt), config, embed_w, norm_w, lm_head_w, args.device)
            scores = evaluate_model(model, teacher, n_layers, samples, args.device)

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
