#!/usr/bin/env python3
"""
UltraCompress v6 — Calibrated PQ: Minimize OUTPUT error, not WEIGHT error.

This is the key integration that should solve the error compounding problem.

Standard PQ: minimizes ||W - W'|| (weight error)
Calibrated PQ: minimizes ||X*W - X*W'|| (output error)

With importance weighting, errors concentrate in dimensions the model doesn't
use (near-zero activations), so they don't compound through layers.

Usage:
    python run_calibrated_pq.py                        # Default balanced
    python run_calibrated_pq.py --mode quality          # Higher quality
    python run_calibrated_pq.py --mode extreme          # 0.016 BPW
    python run_calibrated_pq.py --refine-steps 200      # More gradient refinement
    python run_calibrated_pq.py --max-layers 8          # Test more layers
"""

import argparse
import sys
import os
import torch
import torch.nn.functional as F
import time
import numpy as np

os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultracompress.product_quantize import product_quantize
from ultracompress.calibrated_pq import calibrated_product_quantize, collect_activation_stats
from ultracompress.ultra_pq import refine_codebooks_gradient, estimate_entropy_bpw
from ultracompress.metrics import compute_quality
from ultracompress.gguf_loader import load_ollama_model, find_ollama_model_path
from ultracompress.inference import parse_gguf_config, MiniTransformer, compare_layer_outputs


CONFIGS = {
    "extreme":      {"M": 4,  "K": 2,   "G": 2048, "label": "Calibrated Binary PQ (0.016 BPW)"},
    "balanced":     {"M": 8,  "K": 4,   "G": 256,  "label": "Calibrated PQ (0.13 BPW)"},
    "quality":      {"M": 16, "K": 16,  "G": 256,  "label": "Calibrated High-K PQ (0.36 BPW)"},
    "conservative": {"M": 4,  "K": 256, "G": 32,   "label": "Calibrated Conservative PQ (1.6 BPW)"},
    "int4":         {"M": None, "K": None, "G": None, "label": "INT4 baseline (no calibration)"},
    "standard_pq":  {"M": 8,  "K": 4,   "G": 256,  "label": "Standard PQ (no calibration, for comparison)"},
}


def compress_weight_calibrated_pq(tensor, importance, M, K, G, n_iter=20, refine_steps=100):
    """Compress with calibration-aware PQ + optional gradient refinement."""
    if tensor.ndim < 2 or tensor.numel() < G * 2:
        return tensor

    try:
        # Step 1: Calibrated PQ (importance-weighted k-means)
        cpq = calibrated_product_quantize(
            tensor, importance,
            n_subvectors=M, codebook_size=K,
            group_size=G, n_iter=n_iter,
            importance_power=0.5,
        )

        # Step 2: Gradient refinement of codebooks
        if refine_steps > 0:
            cpq = refine_codebooks_gradient(cpq, tensor, n_steps=refine_steps, lr=0.01)

        return cpq.decompress().reshape(tensor.shape)
    except Exception as e:
        # Fallback to standard PQ
        try:
            pq = product_quantize(tensor, n_subvectors=M, codebook_size=K,
                                  group_size=G, n_iter=n_iter)
            return pq.decompress().reshape(tensor.shape)
        except Exception:
            return tensor


def compress_weight_standard_pq(tensor, M, K, G, n_iter=20):
    """Standard PQ for comparison."""
    if tensor.ndim < 2 or tensor.numel() < G * 2:
        return tensor
    try:
        pq = product_quantize(tensor, n_subvectors=M, codebook_size=K,
                              group_size=G, n_iter=n_iter)
        return pq.decompress().reshape(tensor.shape)
    except Exception:
        return tensor


def compress_weight_int4(tensor, group_size=128):
    """INT4 baseline."""
    from ultracompress.quantize import quantize_absmax
    if tensor.ndim < 2 or tensor.numel() < 256:
        return tensor
    q = quantize_absmax(tensor, bits=4, group_size=group_size)
    return q.decompress().reshape(tensor.shape)


def main():
    parser = argparse.ArgumentParser(description="Calibrated PQ inference test")
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--mode", default="balanced", choices=list(CONFIGS.keys()))
    parser.add_argument("--max-layers", type=int, default=4)
    parser.add_argument("--refine-steps", type=int, default=100,
                        help="Gradient refinement steps (0 to disable)")
    parser.add_argument("--compare", action="store_true",
                        help="Also run standard PQ for side-by-side comparison")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = CONFIGS[args.mode]
    device = args.device

    print("=" * 70)
    print(f"  UltraCompress v6 — Calibrated PQ")
    print(f"  {config['label']}")
    print(f"  Gradient refinement: {args.refine_steps} steps")
    print("=" * 70)

    # Parse model architecture
    model_path = find_ollama_model_path(args.model)
    if not model_path:
        print(f"Model '{args.model}' not found. Make sure Ollama has it pulled.")
        return

    arch = parse_gguf_config(model_path)
    print(f"  Model: {args.model}")
    print(f"  Layers: {arch.n_layers}, Hidden: {arch.hidden_size}")
    print(f"  Testing first {args.max_layers} layers")
    print()

    # Load weights
    print("Loading weights...")
    original_weights = {}
    weight_list = []  # For activation stats
    for name, tensor in load_ollama_model(args.model, device="cpu"):
        layer_idx = -1
        if "blk." in name:
            try:
                layer_idx = int(name.split(".")[1])
            except (ValueError, IndexError):
                pass
            if layer_idx >= args.max_layers:
                continue
        original_weights[name] = tensor
        weight_list.append((name, tensor))
    print(f"  Loaded {len(original_weights)} tensors")

    # ---- Step 1: Collect activation statistics ----
    if config["M"] is not None and args.mode != "standard_pq":
        print("\nCollecting activation statistics (calibration)...")
        t0 = time.time()
        activation_stats = collect_activation_stats(
            weight_list,
            n_calibration_tokens=256,
            seq_len=32,
            vocab_size=arch.vocab_size,
            device=device,
        )
        print(f"  Collected importance weights for {len(activation_stats)} tensors ({time.time()-t0:.1f}s)")

        # Show importance distribution for a few layers
        for name in list(activation_stats.keys())[:3]:
            imp = activation_stats[name]
            nonzero_ratio = (imp > imp.max() * 0.01).float().mean().item()
            print(f"  {name}: {nonzero_ratio*100:.0f}% dimensions are important (>1% of max)")
    else:
        activation_stats = {}

    # ---- Step 2: Compress with calibrated PQ ----
    print(f"\nCompressing with {config['label']}...")
    compressed_weights = {}
    weight_cosines = []
    t_start = time.time()

    for name, tensor in original_weights.items():
        if tensor.ndim >= 2 and tensor.numel() >= 256:
            if config["M"] is None:
                # INT4 baseline
                recon = compress_weight_int4(tensor)
            elif args.mode == "standard_pq":
                # Standard PQ (no calibration)
                recon = compress_weight_standard_pq(
                    tensor, config["M"], config["K"], config["G"]
                )
            else:
                # Calibrated PQ
                importance = activation_stats.get(name)
                if importance is None:
                    # No stats for this tensor — use uniform importance
                    in_dim = tensor.shape[1] if tensor.ndim >= 2 else tensor.shape[-1]
                    importance = torch.ones(in_dim, device=tensor.device)

                recon = compress_weight_calibrated_pq(
                    tensor, importance,
                    config["M"], config["K"], config["G"],
                    refine_steps=args.refine_steps,
                )

            # Measure weight quality
            w = tensor.float().to(device)
            r = recon.float().to(device)
            if r.shape != w.shape:
                r = r.reshape(w.shape)
            q = compute_quality(w, r)
            weight_cosines.append(q["cosine_sim"])
            compressed_weights[name] = recon.cpu()
        else:
            compressed_weights[name] = tensor

    elapsed = time.time() - t_start
    avg_weight_cos = np.mean(weight_cosines) if weight_cosines else 0
    print(f"  Compressed {len(weight_cosines)} weight matrices in {elapsed:.1f}s")
    print(f"  Avg weight cosine: {avg_weight_cos:.6f}")

    # ---- Step 3: Inference comparison ----
    print(f"\nRunning inference comparison ({args.max_layers} layers)...")
    test_input = torch.randint(0, arch.vocab_size, (1, 32), device=device)

    results = compare_layer_outputs(
        original_weights, compressed_weights,
        arch, test_input, device=device,
        max_layers=args.max_layers,
    )

    # Display results
    print(f"\n{'Layer':<20} {'Cosine Sim':>12} {'Rel Error':>12}")
    print("-" * 50)
    layer_cosines = []
    for key, val in results.items():
        if isinstance(val, dict) and "cosine_sim" in val:
            rel_err = val.get("relative_error", "N/A")
            cos = val["cosine_sim"]
            layer_cosines.append(cos)
            if isinstance(rel_err, float):
                print(f"{key:<20} {cos:>12.8f} {rel_err:>12.8f}")
            else:
                print(f"{key:<20} {cos:>12.8f}")

    print()
    if "top10_agreement" in results:
        print(f"  Top-10 token agreement: {results['top10_agreement']*100:.0f}%")
    if "top1_match" in results:
        print(f"  Top-1 prediction match: {'YES' if results['top1_match'] else 'NO'}")

    # Summary
    print()
    print("=" * 70)
    print(f"  RESULTS: {config['label']}")
    print(f"  + Gradient refinement: {args.refine_steps} steps")
    print("=" * 70)
    print(f"  Weight-level cosine:     {avg_weight_cos:.6f}")
    if layer_cosines:
        avg_act = np.mean(layer_cosines)
        min_act = min(layer_cosines)
        print(f"  Activation-level cosine: {avg_act:.6f}")
        print(f"  Min activation cosine:   {min_act:.6f}")
        print(f"  Layer degradation:       {avg_weight_cos - avg_act:.6f} "
              f"(weight cos - activation cos)")
    if "logits" in results and isinstance(results["logits"], dict):
        print(f"  Logit cosine:            {results['logits']['cosine_sim']:.6f}")
    if "top10_agreement" in results:
        print(f"  Top-10 agreement:        {results['top10_agreement']*100:.0f}%")
    if "top1_match" in results:
        print(f"  Top-1 match:             {'YES' if results['top1_match'] else 'NO'}")
    print()

    # ---- Step 4: Side-by-side comparison with standard PQ ----
    if args.compare and config["M"] is not None and args.mode != "standard_pq":
        print("=" * 70)
        print("  COMPARISON: Standard PQ (no calibration, no refinement)")
        print("=" * 70)

        std_compressed = {}
        std_cosines = []
        for name, tensor in original_weights.items():
            if tensor.ndim >= 2 and tensor.numel() >= 256:
                recon = compress_weight_standard_pq(
                    tensor, config["M"], config["K"], config["G"]
                )
                w = tensor.float().to(device)
                r = recon.float().to(device)
                if r.shape != w.shape:
                    r = r.reshape(w.shape)
                q = compute_quality(w, r)
                std_cosines.append(q["cosine_sim"])
                std_compressed[name] = recon.cpu()
            else:
                std_compressed[name] = tensor

        std_results = compare_layer_outputs(
            original_weights, std_compressed,
            arch, test_input, device=device,
            max_layers=args.max_layers,
        )

        std_layer_cos = []
        for key, val in std_results.items():
            if isinstance(val, dict) and "cosine_sim" in val:
                std_layer_cos.append(val["cosine_sim"])

        std_avg_w = np.mean(std_cosines) if std_cosines else 0
        std_avg_a = np.mean(std_layer_cos) if std_layer_cos else 0

        print(f"  Standard PQ weight cosine:     {std_avg_w:.6f}")
        print(f"  Standard PQ activation cosine: {std_avg_a:.6f}")
        print()
        print(f"  Calibrated PQ weight cosine:     {avg_weight_cos:.6f}")
        cal_avg_a = np.mean(layer_cosines) if layer_cosines else 0
        print(f"  Calibrated PQ activation cosine: {cal_avg_a:.6f}")
        print()
        improvement_w = avg_weight_cos - std_avg_w
        improvement_a = cal_avg_a - std_avg_a
        print(f"  Weight improvement:     {improvement_w:+.6f}")
        print(f"  Activation improvement: {improvement_a:+.6f}")
        if improvement_a > 0:
            print(f"  >>> Calibration helps! Activation quality improved by {improvement_a:.6f}")
        elif improvement_a < -0.001:
            print(f"  >>> Calibration didn't help here. May need more refinement steps or real calibration data.")
        else:
            print(f"  >>> Similar performance — calibration is neutral at this config.")
    print()
    print("Next steps:")
    print("  python run_calibrated_pq.py --compare              # Side-by-side vs standard PQ")
    print("  python run_calibrated_pq.py --refine-steps 500     # More gradient refinement")
    print("  python run_calibrated_pq.py --mode quality         # Higher BPW config")
    print("  python run_calibrated_pq.py --max-layers 8         # Test more layers")


if __name__ == "__main__":
    main()
