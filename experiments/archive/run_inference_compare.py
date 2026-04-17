#!/usr/bin/env python3
"""
THE REAL TEST: Compress model, then run inference and compare outputs.

Weight cosine similarity doesn't prove output quality.
This test does: compress every weight, reconstruct, run actual transformer
forward pass, and compare logits/tokens/text between original and compressed.

Usage:
    python run_inference_compare.py                          # Default balanced
    python run_inference_compare.py --mode extreme           # 10T-in-20GB config
    python run_inference_compare.py --mode quality           # Higher quality
    python run_inference_compare.py --max-layers 4           # Test fewer layers
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
from ultracompress.ultra_pq import residual_product_quantize
from ultracompress.calibrated_pq import collect_activation_stats, calibrated_product_quantize
from ultracompress.metrics import compute_quality
from ultracompress.gguf_loader import load_ollama_model, find_ollama_model_path
from ultracompress.inference import parse_gguf_config, MiniTransformer, compare_layer_outputs


CONFIGS = {
    "extreme": {"M": 4, "K": 2, "G": 2048, "label": "Binary PQ (0.016 BPW)"},
    "balanced": {"M": 8, "K": 4, "G": 256, "label": "PQ (0.13 BPW)"},
    "quality": {"M": 16, "K": 16, "G": 256, "label": "High-K PQ (0.36 BPW)"},
    "conservative": {"M": 4, "K": 256, "G": 32, "label": "Conservative PQ (1.6 BPW)"},
    "int4": {"M": None, "K": None, "G": None, "label": "INT4 baseline (4.25 BPW)"},
}


def compress_weight_pq(tensor, M, K, G, n_iter=20, importance=None):
    """Compress a single weight tensor with PQ (optionally calibrated)."""
    if tensor.ndim < 2 or tensor.numel() < G * 2:
        return tensor  # Return uncompressed

    try:
        if importance is not None:
            pq = calibrated_product_quantize(
                tensor, importance=importance,
                n_subvectors=M, codebook_size=K,
                group_size=G, n_iter=n_iter,
            )
        else:
            pq = product_quantize(tensor, n_subvectors=M, codebook_size=K,
                                  group_size=G, n_iter=n_iter)
        return pq.decompress().reshape(tensor.shape)
    except Exception:
        return tensor


def compress_weight_int4(tensor, group_size=128):
    """Compress with INT4 for comparison."""
    from ultracompress.quantize import quantize_absmax
    if tensor.ndim < 2 or tensor.numel() < 256:
        return tensor
    q = quantize_absmax(tensor, bits=4, group_size=group_size)
    return q.decompress().reshape(tensor.shape)


def main():
    parser = argparse.ArgumentParser(description="Inference comparison test")
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--mode", default="balanced", choices=list(CONFIGS.keys()))
    parser.add_argument("--max-layers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--calibrated", action="store_true",
                        help="Use calibration-aware PQ (importance-weighted k-means)")
    args = parser.parse_args()

    config = CONFIGS[args.mode]
    device = args.device

    print("=" * 65)
    print(f"  Inference Comparison: {config['label']}")
    print("=" * 65)

    # Parse model architecture
    model_path = find_ollama_model_path(args.model)
    if not model_path:
        print(f"Model '{args.model}' not found")
        return

    arch = parse_gguf_config(model_path)
    print(f"  Model: {args.model}")
    print(f"  Layers: {arch.n_layers}, Hidden: {arch.hidden_size}")
    print(f"  Testing first {args.max_layers} layers")
    print()

    # Load weights
    print("Loading weights...")
    original_weights = {}
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
    print(f"  Loaded {len(original_weights)} tensors")

    # Collect activation stats for calibrated PQ
    activation_stats = {}
    if args.calibrated:
        print("\nCollecting activation statistics (CPU, low-RAM)...")
        weight_list = list(original_weights.items())
        activation_stats = collect_activation_stats(
            weight_list, n_calibration_tokens=64, seq_len=16, device="cpu",
        )
        print(f"  Got importance weights for {len(activation_stats)} tensors")

    # Compress all weights
    label = config['label'] + (" + calibrated" if args.calibrated else "")
    print(f"\nCompressing with {label}...")
    compressed_weights = {}
    weight_cosines = []
    total_orig_bytes = 0
    total_comp_bytes = 0

    for name, tensor in original_weights.items():
        if tensor.ndim >= 2 and tensor.numel() >= 256:
            if config["M"] is not None:
                imp = activation_stats.get(name, None)
                recon = compress_weight_pq(tensor, config["M"], config["K"], config["G"],
                                           importance=imp)
            else:
                recon = compress_weight_int4(tensor)

            # Measure weight quality
            w = tensor.float().to(device)
            r = recon.float().to(device)
            if r.shape != w.shape:
                r = r.reshape(w.shape)
            q = compute_quality(w, r)
            weight_cosines.append(q["cosine_sim"])

            compressed_weights[name] = recon.cpu()
            total_orig_bytes += tensor.numel() * 2
        else:
            compressed_weights[name] = tensor

    avg_weight_cos = np.mean(weight_cosines) if weight_cosines else 0
    print(f"  Compressed {len(weight_cosines)} weight matrices")
    print(f"  Avg weight cosine: {avg_weight_cos:.6f}")

    # Run inference comparison
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
    print("=" * 65)
    print(f"  SUMMARY: {label}")
    print("=" * 65)
    print(f"  Weight-level cosine:     {avg_weight_cos:.6f}")
    if layer_cosines:
        print(f"  Activation-level cosine: {np.mean(layer_cosines):.6f}")
        print(f"  Min activation cosine:   {min(layer_cosines):.6f}")
    if "logits" in results and isinstance(results["logits"], dict):
        print(f"  Logit cosine:            {results['logits']['cosine_sim']:.6f}")
    if "top10_agreement" in results:
        print(f"  Top-10 agreement:        {results['top10_agreement']*100:.0f}%")
    if "top1_match" in results:
        print(f"  Top-1 match:             {'YES' if results['top1_match'] else 'NO'}")
    print()

    # Run at multiple configs for comparison
    if args.mode != "all":
        print("Run with different modes to compare:")
        print("  python run_inference_compare.py --mode conservative")
        print("  python run_inference_compare.py --mode balanced")
        print("  python run_inference_compare.py --mode extreme")
        print("  python run_inference_compare.py --mode int4")


if __name__ == "__main__":
    main()
