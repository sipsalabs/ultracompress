#!/usr/bin/env python3
"""
UltraCompress v5 — Full Model Compression Benchmark

Runs the complete compression pipeline on a model and shows projections.

Usage:
    python run_ultra.py                           # Default: qwen3:4b, balanced
    python run_ultra.py --model qwen3:8b          # Different model
    python run_ultra.py --mode extreme             # 10T-in-20GB mode (lowest BPW)
    python run_ultra.py --mode quality             # Best quality mode
    python run_ultra.py --mode balanced            # Best quality-per-bit
    python run_ultra.py --rpq 3                    # Residual PQ with 3 levels
    python run_ultra.py --source Qwen/Qwen3-8B    # FP16 from HuggingFace
"""

import argparse
import sys
import os
import torch
import time

os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultracompress.product_quantize import product_quantize
from ultracompress.ultra_pq import (
    residual_product_quantize, refine_codebooks_gradient,
    estimate_entropy_bpw, GlobalCodebookManager,
)
from ultracompress.calibrated_pq import collect_activation_stats, calibrated_product_quantize
from ultracompress.metrics import compute_quality
from ultracompress.gguf_loader import load_ollama_model


# Preset configurations
PRESETS = {
    "extreme": {
        "description": "10T-in-20GB mode — absolute minimum BPW",
        "pq_config": (4, 2, 2048),   # 0.014 BPW, 0.90 cosine
        "rpq_levels": 1,
        "target_cosine": 0.85,
    },
    "balanced": {
        "description": "Best quality-per-bit tradeoff",
        "pq_config": (8, 4, 256),    # 0.13 BPW, 0.915 cosine
        "rpq_levels": 1,
        "target_cosine": 0.90,
    },
    "quality": {
        "description": "Highest quality with RPQ",
        "pq_config": (8, 4, 256),
        "rpq_levels": 3,
        "rpq_configs": [
            (8, 4, 256),    # Level 1: bulk
            (8, 8, 128),    # Level 2: detail
            (8, 16, 64),    # Level 3: fine
        ],
        "target_cosine": 0.95,
    },
    "10t_quality": {
        "description": "10T target with best quality",
        "pq_config": (8, 4, 1024),   # 0.036 BPW
        "rpq_levels": 2,
        "rpq_configs": [
            (8, 4, 1024),
            (8, 4, 512),
        ],
        "target_cosine": 0.90,
    },
}


def run_benchmark(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preset = PRESETS[args.mode]

    print("=" * 65)
    print(f"  UltraCompress v5 -- {preset['description']}")
    print("=" * 65)
    print(f"  Model: {args.model}")
    print(f"  Mode: {args.mode}")
    print(f"  Device: {device}")
    print(f"  PQ config: M={preset['pq_config'][0]} K={preset['pq_config'][1]} G={preset['pq_config'][2]}")
    print(f"  RPQ levels: {preset['rpq_levels']}")
    print(f"  Calibrated: {args.calibrated}")
    print()

    # Load weights
    print("Loading weights...")
    weights = []
    for name, tensor in load_ollama_model(args.model, max_tensors=args.max_tensors):
        weights.append((name, tensor))
    print(f"Loaded {len(weights)} tensors")

    # Collect activation stats for calibrated PQ
    activation_stats = {}
    if args.calibrated:
        print("Collecting activation statistics for calibrated PQ...")
        # Use CPU for calibration to save RAM — importance weights are small
        cal_device = "cpu"
        activation_stats = collect_activation_stats(
            weights, n_calibration_tokens=64, seq_len=16, device=cal_device,
        )
        print(f"  Got importance weights for {len(activation_stats)} tensors")
        print()

    # Compress
    M, K, G = preset["pq_config"]
    n_iter = args.n_iter
    rpq_levels = preset["rpq_levels"]
    rpq_configs = preset.get("rpq_configs", [(M, K, G)])

    total_orig = 0
    total_comp = 0
    total_params = 0
    cosines = []
    methods = {}
    t_start = time.time()

    print()
    print(f"{'Tensor':42s} {'Method':22s} {'Cos':>8} {'BPW':>7} {'Time':>5}")
    print("-" * 90)

    for name, tensor in weights:
        if tensor.ndim < 2 or tensor.numel() < 256:
            continue

        t0 = time.time()

        try:
            if rpq_levels > 1:
                rpq = residual_product_quantize(
                    tensor, n_levels=rpq_levels,
                    level_configs=rpq_configs[:rpq_levels],
                    n_iter=n_iter,
                )
                recon = rpq.decompress().to(device)
                w = tensor.float().to(device)
                if recon.shape != w.shape:
                    recon = recon.reshape(w.shape)
                quality = compute_quality(w, recon)
                comp_bytes = rpq.storage_bytes()
                method = f"rpq{rpq_levels}_m{M}k{K}g{G}"
            elif args.calibrated and name in activation_stats:
                pq = calibrated_product_quantize(
                    tensor, importance=activation_stats[name],
                    n_subvectors=M, codebook_size=K,
                    group_size=G, n_iter=n_iter,
                )
                recon = pq.decompress().to(device)
                w = tensor.float().to(device)
                if recon.shape != w.shape:
                    recon = recon.reshape(w.shape)
                quality = compute_quality(w, recon)
                comp_bytes = pq.storage_bytes()
                method = f"cal_pq_m{M}k{K}g{G}"
            else:
                pq = product_quantize(
                    tensor, n_subvectors=M, codebook_size=K,
                    group_size=G, n_iter=n_iter,
                )
                recon = pq.decompress().to(device)
                w = tensor.float().to(device)
                if recon.shape != w.shape:
                    recon = recon.reshape(w.shape)
                quality = compute_quality(w, recon)
                comp_bytes = pq.storage_bytes()
                method = f"pq_m{M}k{K}g{G}"

        except Exception as e:
            # Fallback to smaller G
            try:
                fallback_G = min(G, 128)
                while fallback_G > 4 and tensor.numel() < fallback_G * 2:
                    fallback_G //= 2
                pq = product_quantize(
                    tensor, n_subvectors=min(M, fallback_G),
                    codebook_size=K, group_size=fallback_G, n_iter=n_iter,
                )
                recon = pq.decompress().to(device)
                w = tensor.float().to(device)
                if recon.shape != w.shape:
                    recon = recon.reshape(w.shape)
                quality = compute_quality(w, recon)
                comp_bytes = pq.storage_bytes()
                method = f"pq_m{min(M,fallback_G)}k{K}g{fallback_G}"
            except Exception:
                continue

        dt = time.time() - t0
        orig_bytes = tensor.numel() * 2
        bpw = comp_bytes * 8 / tensor.numel()

        total_orig += orig_bytes
        total_comp += comp_bytes
        total_params += tensor.numel()
        cosines.append(quality["cosine_sim"])
        methods[method] = methods.get(method, 0) + 1

        sys.stdout.write(
            f"{name:42s} {method:22s} {quality['cosine_sim']:>8.5f} {bpw:>7.4f} {dt:>4.1f}s\n"
        )
        sys.stdout.flush()

    elapsed = time.time() - t_start
    avg_bpw = (total_comp * 8) / total_params if total_params > 0 else 0
    avg_cos = sum(cosines) / len(cosines) if cosines else 0
    min_cos = min(cosines) if cosines else 0

    print()
    print("=" * 65)
    print(f"  RESULTS")
    print("=" * 65)
    print(f"  Tensors:     {len(cosines)}")
    print(f"  Time:        {elapsed:.0f}s")
    print(f"  Methods:     {methods}")
    print(f"  Avg BPW:     {avg_bpw:.4f}")
    print(f"  Avg cosine:  {avg_cos:.5f}")
    print(f"  Min cosine:  {min_cos:.5f}")
    print(f"  Ratio:       {total_orig/max(total_comp,1):.0f}x from FP16")
    print()
    print(f"  PROJECTIONS at {avg_bpw:.4f} BPW:")
    print(f"  {'Model':25s} {'FP16':>8} {'Compressed':>11} {'Ratio':>7} {'20GB?':>6}")
    print(f"  {'-'*60}")

    for model_name, size_b in [
        ("8B", 8), ("32B", 32), ("70B", 70), ("235B", 235),
        ("405B", 405), ("671B", 671), ("1T", 1000),
        ("10T", 10000), ("100T", 100000),
    ]:
        fp16_gb = size_b * 2
        comp_gb = size_b * 1e9 * avg_bpw / 8 / 1e9
        ratio = fp16_gb / max(comp_gb, 0.001)
        fits = "YES" if comp_gb <= 20 else "NO"
        print(f"  {model_name:25s} {fp16_gb:>6} GB {comp_gb:>8.1f} GB {ratio:>6.0f}x   {fits}")

    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="UltraCompress v5")
    parser.add_argument("--model", default="qwen3:4b", help="Ollama model name")
    parser.add_argument("--mode", default="balanced",
                        choices=list(PRESETS.keys()),
                        help="Compression preset")
    parser.add_argument("--max-tensors", type=int, default=None)
    parser.add_argument("--n-iter", type=int, default=20, help="k-means iterations")
    parser.add_argument("--rpq", type=int, default=None, help="Override RPQ levels")
    parser.add_argument("--calibrated", action="store_true",
                        help="Use calibration-aware PQ (importance-weighted k-means)")

    args = parser.parse_args()

    if args.rpq is not None:
        PRESETS[args.mode]["rpq_levels"] = args.rpq

    run_benchmark(args)


if __name__ == "__main__":
    main()
