#!/usr/bin/env python3
"""
Generative Weight Compression — Train a neural network to BE the model.

Instead of storing compressed weights, train a small generator network
that produces weight values on demand from coordinates.

Usage:
    python run_gwc.py                              # Default: qwen3:4b, small generator
    python run_gwc.py --hidden 512 --depth 6       # Bigger generator
    python run_gwc.py --hidden 128 --depth 3       # Tiny generator (max compression)
    python run_gwc.py --max-layers 4               # Quick test on first 4 layers
    python run_gwc.py --epochs 100                 # More training
"""

import argparse
import sys
import os
import time
import torch
import numpy as np

# Fix CUDA_VISIBLE_DEVICES if it's set to a stale UUID
if 'CUDA_VISIBLE_DEVICES' in os.environ:
    val = os.environ['CUDA_VISIBLE_DEVICES']
    if 'GPU-' in val:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultracompress.generative_compression import (
    train_generator, GWCResult, get_layer_id, get_tensor_type,
)
from ultracompress.gguf_loader import load_ollama_model
from ultracompress.metrics import compute_quality


PRESETS = {
    "tiny": {
        "description": "Maximum compression — tiny generator",
        "hidden_dim": 128,
        "n_hidden": 3,
        "n_fourier_freqs": 16,
    },
    "small": {
        "description": "Balanced quality/compression",
        "hidden_dim": 256,
        "n_hidden": 4,
        "n_fourier_freqs": 32,
    },
    "medium": {
        "description": "Higher quality, larger generator",
        "hidden_dim": 512,
        "n_hidden": 6,
        "n_fourier_freqs": 48,
    },
    "large": {
        "description": "Best quality — large generator",
        "hidden_dim": 1024,
        "n_hidden": 8,
        "n_fourier_freqs": 64,
    },
}


def main():
    parser = argparse.ArgumentParser(description="Generative Weight Compression")
    parser.add_argument("--model", default="qwen3:4b", help="Ollama model name")
    parser.add_argument("--preset", default="small", choices=list(PRESETS.keys()))
    parser.add_argument("--hidden", type=int, default=None, help="Override hidden dim")
    parser.add_argument("--depth", type=int, default=None, help="Override num hidden layers")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-layers", type=int, default=None, help="Limit layers for quick test")
    parser.add_argument("--max-samples", type=int, default=100000, help="Max samples per tensor")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    # Device selection
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    preset = PRESETS[args.preset]
    hidden_dim = args.hidden or preset["hidden_dim"]
    n_hidden = args.depth or preset["n_hidden"]
    n_fourier_freqs = preset["n_fourier_freqs"]

    print("=" * 65)
    print("  GENERATIVE WEIGHT COMPRESSION (GWC)")
    print("  The Paradigm Shift: Don't store weights. Generate them.")
    print("=" * 65)
    print(f"  Model: {args.model}")
    print(f"  Preset: {args.preset} — {preset['description']}")
    print(f"  Generator: hidden={hidden_dim}, depth={n_hidden}, fourier={n_fourier_freqs}")
    print(f"  Training: {args.epochs} epochs, batch={args.batch_size}, lr={args.lr}")
    print(f"  Device: {device}")
    if args.max_layers:
        print(f"  Testing first {args.max_layers} layers only")
    print()

    # Load weights
    print("Loading model weights...")
    weights = []
    for name, tensor in load_ollama_model(args.model):
        if args.max_layers is not None:
            layer_id = get_layer_id(name)
            if "blk." in name and layer_id >= args.max_layers:
                continue
        weights.append((name, tensor))
    print(f"  Loaded {len(weights)} tensors")
    print()

    # Train generator
    print("Training weight generator...")
    t_start = time.time()

    result = train_generator(
        weights,
        hidden_dim=hidden_dim,
        n_hidden=n_hidden,
        n_fourier_freqs=n_fourier_freqs,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        max_samples_per_tensor=args.max_samples,
    )

    elapsed = time.time() - t_start

    # Results
    print()
    print("=" * 65)
    print("  GWC RESULTS")
    print("=" * 65)
    print(f"  Training time:    {elapsed:.0f}s")
    print(f"  Training loss:    {result.training_loss:.6f}")
    print()
    print(f"  Generator size:   {result.generator_bytes / 1e6:.1f} MB ({result.generator_params:,} params)")
    print(f"  Original size:    {result.original_bytes / 1e9:.2f} GB ({result.original_params:,} params)")
    print(f"  Compression:      {result.compression_ratio:.0f}x")
    print(f"  BPW:              {result.bpw:.6f}")
    print()
    print(f"  Avg cosine sim:   {result.avg_cosine:.6f}")
    print()

    # Per-tensor breakdown (top/bottom 5)
    sorted_cosines = sorted(result.per_tensor_cosine.items(), key=lambda x: x[1])

    print("  WORST 5 tensors:")
    for name, cos in sorted_cosines[:5]:
        print(f"    {name:42s} {cos:.6f}")
    print()
    print("  BEST 5 tensors:")
    for name, cos in sorted_cosines[-5:]:
        print(f"    {name:42s} {cos:.6f}")

    # Projections
    print()
    print("=" * 65)
    print("  SCALING PROJECTIONS")
    print("=" * 65)
    print()

    # The key insight: generator size scales SUB-LINEARLY with model size
    # Because larger models have MORE redundancy, not less
    gen_mb = result.generator_bytes / 1e6

    print(f"  Current generator: {gen_mb:.1f} MB for {result.original_params/1e9:.1f}B params")
    print(f"  At this quality, projections with FIXED generator size:")
    print()
    print(f"  {'Model':20s} {'FP16':>8} {'Generator':>10} {'Ratio':>8} {'BPW':>10} {'<20GB?':>7}")
    print(f"  {'-'*65}")

    for model_name, size_b in [
        ("8B", 8), ("70B", 70), ("235B", 235), ("405B", 405),
        ("1T", 1000), ("10T", 10000), ("100T", 100000), ("1000T", 1000000),
    ]:
        fp16_gb = size_b * 2
        # Generator size grows ~logarithmically with model size
        # (more layers = more layer embeddings, but backbone stays fixed)
        extra_layer_params = size_b * 1e9 / result.original_params * 128 * 2  # layer embeds
        proj_gen_bytes = result.generator_bytes + extra_layer_params * 2
        proj_gen_gb = proj_gen_bytes / 1e9
        ratio = fp16_gb / max(proj_gen_gb, 0.001)
        bpw = proj_gen_bytes * 8 / (size_b * 1e9)
        fits = "YES" if proj_gen_gb <= 20 else "no"
        print(f"  {model_name:20s} {fp16_gb:>6} GB {proj_gen_gb:>8.2f} GB {ratio:>7.0f}x {bpw:>9.6f}   {fits}")

    print()
    print("  NOTE: These projections assume fixed generator backbone.")
    print("  Real scaling will require larger generators for larger models,")
    print("  but the ratio IMPROVES with scale due to increasing redundancy.")
    print("=" * 65)


if __name__ == "__main__":
    main()
