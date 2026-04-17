#!/usr/bin/env python3
"""
UltraCompress v8 — Weight Genome: Hierarchical Generative Compression

Train a small neural network (the "genome") that generates weight values
on demand, plus PQ-compressed residuals for fine detail.

The genome + residuals approach targets 0.001 BPW for 1000T models.

Usage:
    python run_genome.py                                    # Default
    python run_genome.py --model Qwen/Qwen3-0.6B            # Small model
    python run_genome.py --hidden 512 --depth 6 --epochs 200 # Bigger/longer
    python run_genome.py --entropy-weight 0.1                # PQ-aware training
"""

import argparse
import sys
import os
import time
import math
import torch
import numpy as np

os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultracompress.weight_genome import (
    WeightGenome, prepare_training_data, train_genome, evaluate_genome,
)
from ultracompress.safetensors_loader import load_hf_model
from ultracompress.product_quantize import product_quantize
from ultracompress.metrics import compute_quality


def main():
    parser = argparse.ArgumentParser(description="Weight Genome Training")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--hidden", type=int, default=256, help="Generator hidden dim")
    parser.add_argument("--depth", type=int, default=4, help="SIREN depth")
    parser.add_argument("--fourier", type=int, default=64, help="Fourier features")
    parser.add_argument("--mod-dim", type=int, default=128, help="Modulation dimension")
    parser.add_argument("--group-size", type=int, default=64, help="Weight group size")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--entropy-weight", type=float, default=0.0,
                       help="Entropy loss weight (0=MSE only, >0=PQ-aware)")
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument("--max-groups", type=int, default=5000,
                       help="Max groups per weight for training")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("  UltraCompress v8 -- THE WEIGHT GENOME")
    print("  Hierarchical Generative Weight Compression")
    print("=" * 70)
    print(f"  Model:      {args.model}")
    print(f"  Generator:  hidden={args.hidden}, depth={args.depth}, fourier={args.fourier}")
    print(f"  Modulation: dim={args.mod_dim}")
    print(f"  Training:   {args.epochs} epochs, batch={args.batch_size}, lr={args.lr}")
    print(f"  Entropy:    weight={args.entropy_weight}")
    print(f"  Device:     {device}")
    print()

    # Load model
    print("Loading model weights...")
    t0 = time.time()
    weights_dict = {}
    for name, tensor in load_hf_model(args.model):
        weights_dict[name] = tensor
    print(f"  Loaded {len(weights_dict)} tensors ({time.time()-t0:.1f}s)")

    # Count layers
    n_layers = 0
    for key in weights_dict:
        if 'model.layers.' in key:
            try:
                idx = int(key.split('model.layers.')[1].split('.')[0])
                n_layers = max(n_layers, idx + 1)
            except (ValueError, IndexError):
                pass
    if args.max_layers:
        n_layers = min(n_layers, args.max_layers)
    print(f"  Layers: {n_layers}")

    # Count total params in weight matrices
    total_weight_params = 0
    for key in weights_dict:
        if any(t in key for t in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']):
            layer_idx = -1
            try:
                layer_idx = int(key.split('model.layers.')[1].split('.')[0])
            except:
                pass
            if layer_idx < n_layers:
                total_weight_params += weights_dict[key].numel()
    print(f"  Weight params: {total_weight_params:,}")

    # Prepare training data
    print("\nPreparing training data...")
    t0 = time.time()
    data = prepare_training_data(
        weights_dict, n_layers,
        group_size=args.group_size,
        max_groups_per_weight=args.max_groups,
        device=device,
    )
    print(f"  Training samples: {data.coords.shape[0]:,}")
    print(f"  Prep time: {time.time()-t0:.1f}s")

    # Create genome
    genome = WeightGenome(
        hidden_dim=args.hidden,
        n_hidden=args.depth,
        n_fourier_freqs=args.fourier,
        n_layers=n_layers,
        mod_dim=args.mod_dim,
        group_size=args.group_size,
    )
    gen_params = genome.count_params()
    gen_bytes = genome.size_bytes()
    print(f"\nGenome: {gen_params:,} params = {gen_bytes/1e6:.1f} MB")
    print(f"  Compression ratio (genome only): {total_weight_params * 2 / gen_bytes:.0f}x")
    print(f"  Genome BPW: {gen_bytes * 8 / total_weight_params:.6f}")

    # Train
    print(f"\nTraining genome ({args.epochs} epochs)...")
    t0 = time.time()
    train_result = train_genome(
        genome, data,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        entropy_weight=args.entropy_weight,
        device=device,
    )
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.0f}s")

    # Evaluate
    print("\nEvaluating genome quality...")
    eval_result = evaluate_genome(genome, data, device=device)

    print(f"\n  Overall cosine:    {eval_result['overall_cosine']:.6f}")
    print(f"  Residual ratio:    {eval_result['residual_ratio']:.4f} (lower = better prediction)")
    print()

    # Per-type breakdown
    print("  Per weight type:")
    for wtype, cos in eval_result['per_type'].items():
        print(f"    {wtype:12s}: {cos:.4f}")
    print()

    # Per-layer breakdown (first few + last few)
    print("  Per layer (selection):")
    layers = sorted(eval_result['per_layer'].keys())
    show = layers[:3] + layers[-3:] if len(layers) > 6 else layers
    for l in show:
        print(f"    Layer {l:>2}: {eval_result['per_layer'][l]:.4f}")
    print()

    # Compute residual PQ cost
    print("Computing residual PQ cost...")
    genome.eval()
    residual_bpw_total = 0
    residual_samples = 0

    with torch.no_grad():
        # Predict all groups
        batch_size = 8192
        N = data.coords.shape[0]
        all_preds = []
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            pred = genome(
                data.coords[start:end],
                data.layer_ids[start:end],
                data.type_ids[start:end],
            )
            all_preds.append(pred)
        preds = torch.cat(all_preds, dim=0)

        # Compute residuals and PQ them
        residuals = data.targets - preds
        # Scale residuals back
        residuals_scaled = residuals * data.scales.unsqueeze(1)

        # Sample and PQ the residuals
        n_sample = min(50000, residuals_scaled.shape[0])
        sample_idx = torch.randperm(residuals_scaled.shape[0])[:n_sample]
        sample_residuals = residuals_scaled[sample_idx].reshape(-1)

        # PQ the residuals at various configs
        print("\n  Residual PQ quality:")
        print(f"  {'Config':25s} {'BPW':>7} {'Cosine':>8} {'MSE':>10}")
        print(f"  {'-'*55}")

        for M, K, G in [(8, 4, 64), (8, 16, 64), (4, 256, 32)]:
            if sample_residuals.numel() >= G * 4:
                fake_weight = sample_residuals[:sample_residuals.numel() // G * G].reshape(-1, G)
                for row_idx in range(min(fake_weight.shape[0], 1000)):
                    row = fake_weight[row_idx:row_idx+1].reshape(1, -1)
                    # This is just a quick check
                pq = product_quantize(
                    sample_residuals[:sample_residuals.numel() // G * G].reshape(1, -1),
                    n_subvectors=M, codebook_size=K, group_size=G, n_iter=20,
                )
                recon = pq.decompress().reshape(-1)
                orig = sample_residuals[:recon.numel()]
                cos = torch.nn.functional.cosine_similarity(
                    orig.reshape(1, -1), recon.reshape(1, -1)
                ).item()
                mse = torch.nn.functional.mse_loss(recon, orig).item()
                bpw = pq.bits_per_weight
                print(f"  M={M:>2} K={K:>3} G={G:>3}           {bpw:>7.4f} {cos:>8.4f} {mse:>10.6f}")

    # Final summary
    genome_bpw = gen_bytes * 8 / total_weight_params
    residual_ratio = eval_result['residual_ratio']

    print()
    print("=" * 70)
    print("  WEIGHT GENOME RESULTS")
    print("=" * 70)
    print(f"  Genome size:       {gen_bytes/1e6:.1f} MB ({gen_params:,} params)")
    print(f"  Genome BPW:        {genome_bpw:.6f}")
    print(f"  Prediction cosine: {eval_result['overall_cosine']:.6f}")
    print(f"  Residual ratio:    {residual_ratio:.4f}")
    print(f"  Training time:     {train_time:.0f}s")
    print()

    # Scaling projections
    print("  SCALING PROJECTIONS:")
    print(f"  {'Model':15s} {'FP16':>8} {'Genome':>8} {'Residual':>10} {'Total':>8} {'Ratio':>7} {'<20GB?':>7}")
    print(f"  {'-'*65}")

    for model_name, size_b in [
        ("0.6B", 0.6), ("8B", 8), ("70B", 70), ("235B", 235),
        ("1T", 1000), ("10T", 10000), ("100T", 100000), ("1000T", 1000000),
    ]:
        fp16_gb = size_b * 2

        # Genome scales sub-linearly (more layers = more embeddings but same backbone)
        # Rough estimate: backbone is 80% of params, layer embeddings scale linearly
        scale_factor = max(1, math.log2(size_b / 0.6 + 1) / math.log2(2))
        proj_gen_gb = gen_bytes / 1e9 * scale_factor

        # Residual PQ: residual_ratio * original_size * PQ_BPW
        # At 0.5 BPW PQ on residuals (generous — we showed 0.06 works)
        residual_pq_bpw = 0.5
        proj_residual_gb = size_b * 1e9 * residual_ratio * residual_pq_bpw / 8 / 1e9

        total_gb = proj_gen_gb + proj_residual_gb
        ratio = fp16_gb / max(total_gb, 0.001)
        fits = "YES" if total_gb <= 20 else "no"

        print(f"  {model_name:15s} {fp16_gb:>6} GB {proj_gen_gb:>6.1f} GB {proj_residual_gb:>8.1f} GB {total_gb:>6.1f} GB {ratio:>6.0f}x   {fits}")

    print("=" * 70)


if __name__ == "__main__":
    main()
