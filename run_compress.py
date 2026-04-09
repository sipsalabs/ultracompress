#!/usr/bin/env python3
"""
UltraCompress v2 — Near-Zero Degradation LLM Compression

Usage:
    python run_compress.py                              # Default: qwen3:4b, quality-first
    python run_compress.py --model qwen3:8b             # Larger model
    python run_compress.py --target-bpw 0.5             # Custom BPW target
    python run_compress.py --sweep                      # Quality/size sweep
    python run_compress.py --inference                   # Full inference comparison
    python run_compress.py --inference --max-layers 4    # Test first 4 layers
"""

import argparse
import sys
import os
import torch
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultracompress.pipeline import PipelineConfig, compress_model, compress_weight
from ultracompress.gguf_loader import (
    load_ollama_model, find_ollama_model_path, list_gguf_tensors
)
from ultracompress.safetensors_loader import load_model
from ultracompress.metrics import ModelCompressionReport


def print_header():
    print("=" * 65)
    print("  UltraCompress v2 — Near-Zero Degradation LLM Compression")
    print("=" * 65)


def load_model_info(args):
    """Find model and print info."""
    model_path = find_ollama_model_path(args.model)
    if model_path is None:
        print(f"\nERROR: Could not find Ollama model '{args.model}'")
        os.system("ollama list")
        return None, None, None

    file_size_gb = os.path.getsize(model_path) / 1e9
    tensor_list = list_gguf_tensors(model_path)
    total_params = sum(t["n_params"] for t in tensor_list)

    print(f"\nModel:        {args.model}")
    print(f"GGUF file:    {file_size_gb:.2f} GB")
    print(f"Tensors:      {len(tensor_list)}")
    print(f"Parameters:   {total_params:,} ({total_params/1e9:.2f}B)")
    print(f"FP16 size:    {total_params * 2 / 1e9:.2f} GB")

    return model_path, tensor_list, total_params


def load_weights(args, device="cpu"):
    """Load model weights from Ollama."""
    named_weights = []
    for name, tensor in load_ollama_model(
        args.model, max_tensors=args.max_tensors,
        name_filter=args.filter, device=device,
    ):
        named_weights.append((name, tensor))
    return named_weights


def run_compression(args):
    """Run v2 compression pipeline."""
    print_header()
    model_path, tensor_list, total_params = load_model_info(args)
    if model_path is None:
        return

    config = PipelineConfig(
        target_bpw=args.target_bpw,
        target_cosine_sim=args.min_cosine,
        energy_target=args.energy_target,
        enable_residual=args.residual,
        residual_levels=args.residual_levels,
        group_size=args.group_size,
        codebook_size=args.codebook_size,
        max_rank=args.max_rank,
        use_sigma_delta=args.sigma_delta,
        enable_vq=args.vq,
        vq_codebook_size=args.vq_codebook_size,
        vq_group_sizes=tuple(args.vq_group_sizes),
        vq_residual_levels=args.vq_levels,
        vq_n_iter=args.vq_iter,
        device=args.device,
        min_tensor_size=args.min_tensor_size,
    )

    print(f"\nLoading weights (max={args.max_tensors or 'all'})...")
    named_weights = load_weights(args)
    print(f"Loaded {len(named_weights)} tensors")

    start_time = time.time()
    report = compress_model(named_weights, config)
    elapsed = time.time() - start_time

    print(report.summary())
    print(f"  Time elapsed:         {elapsed:.1f}s")

    # Show quality distribution
    if report.layers:
        cosines = [l.cosine_sim for l in report.layers]
        print(f"\n  Quality Distribution:")
        print(f"    Min cosine sim:    {min(cosines):.6f}")
        print(f"    Max cosine sim:    {max(cosines):.6f}")
        print(f"    Mean cosine sim:   {sum(cosines)/len(cosines):.6f}")
        print(f"    Layers >= 0.999:   {sum(1 for c in cosines if c >= 0.999)}/{len(cosines)}")
        print(f"    Layers >= 0.99:    {sum(1 for c in cosines if c >= 0.99)}/{len(cosines)}")
        print(f"    Layers >= 0.95:    {sum(1 for c in cosines if c >= 0.95)}/{len(cosines)}")

    # Extrapolations toward the 10T goal
    for size in [8, 32, 70, 235, 405, 671, 1000, 10000]:
        print(report.extrapolate(size))

    # Per-layer detail
    if len(report.layers) > 1:
        sorted_layers = sorted(report.layers, key=lambda l: l.cosine_sim)
        print("\n  Worst layers (lowest cosine sim):")
        for l in sorted_layers[:5]:
            print(f"    {l.name:50s} cos={l.cosine_sim:.6f}  {l.bits_per_weight:.3f} BPW")
        print("\n  Best layers (highest cosine sim):")
        for l in sorted_layers[-5:]:
            print(f"    {l.name:50s} cos={l.cosine_sim:.6f}  {l.bits_per_weight:.3f} BPW")

    return report


def run_sweep(args):
    """Sweep BPW and energy targets to map the quality/size frontier."""
    print_header()
    print("  MODE: Quality/Size Sweep\n")

    model_path, tensor_list, total_params = load_model_info(args)
    if model_path is None:
        return

    print(f"\nLoading weights...")
    named_weights = load_weights(args)
    print(f"Loaded {len(named_weights)} tensors\n")

    # Sweep configurations
    configs = [
        # (energy_target, residual, sigma_delta, codebook_size, label)
        (0.999,  False, False, 256,   "SVD 99.9% only"),
        (0.999,  False, True,  256,   "SVD 99.9% + sigma-delta"),
        (0.999,  False, True,  4096,  "SVD 99.9% + SD + CB4096"),
        (0.9999, False, True,  4096,  "SVD 99.99% + SD + CB4096"),
        (0.999,  True,  True,  4096,  "Residual(3) + SD + CB4096"),
        (0.9999, True,  True,  4096,  "Residual(3) 99.99% + SD + CB"),
    ]

    print(f"{'Config':40s} {'BPW':>8} {'Ratio':>8} {'Cosine':>10} {'RelErr':>10} {'235B GB':>10}")
    print("-" * 90)

    for energy, residual, sd, cb_size, label in configs:
        config = PipelineConfig(
            target_bpw=0.5,
            energy_target=energy,
            enable_residual=residual,
            residual_levels=3 if residual else 1,
            use_sigma_delta=sd,
            codebook_size=cb_size,
            device=args.device,
            min_tensor_size=args.min_tensor_size,
        )

        report = compress_model(named_weights, config)
        bpw = report.avg_bits_per_weight
        size_235b = (235e9 * bpw) / 8 / 1e9

        print(f"{label:40s} {bpw:>8.3f} {report.overall_ratio:>7.0f}x "
              f"{report.avg_cosine_sim:>10.6f} {report.avg_relative_error:>10.6f} "
              f"{size_235b:>9.1f}")


def run_inference_test(args):
    """Full inference comparison: original vs compressed weights."""
    print_header()
    print("  MODE: Inference Comparison\n")

    from ultracompress.inference import (
        parse_gguf_config, MiniTransformer, compare_layer_outputs
    )
    from ultracompress.pipeline import PipelineConfig, compress_weight
    from ultracompress.profiler import profile_layer

    model_path = find_ollama_model_path(args.model)
    if model_path is None:
        print(f"ERROR: Model '{args.model}' not found")
        return

    # Parse architecture
    print("Parsing model architecture...")
    config = parse_gguf_config(model_path)
    print(f"  Layers: {config.n_layers}, Heads: {config.n_heads}, Hidden: {config.hidden_size}")
    print(f"  KV Heads: {config.n_kv_heads}, Intermediate: {config.intermediate_size}")
    print(f"  Vocab: {config.vocab_size}, Head dim: {config.head_dim}")

    # Load ALL weights into dict
    max_layers = args.max_layers or 2
    print(f"\nLoading weights for first {max_layers} layers...")

    original_weights = {}
    for name, tensor in load_ollama_model(args.model, device="cpu"):
        # Only load what we need
        layer_idx = -1
        if "blk." in name:
            try:
                layer_idx = int(name.split(".")[1])
            except (ValueError, IndexError):
                pass
            if layer_idx >= max_layers:
                continue
        original_weights[name] = tensor

    print(f"  Loaded {len(original_weights)} tensors")

    # Compress all weight matrices
    print(f"\nCompressing weights...")
    pipe_config = PipelineConfig(
        target_bpw=args.target_bpw,
        target_cosine_sim=args.min_cosine,
        energy_target=args.energy_target,
        enable_residual=args.residual,
        residual_levels=args.residual_levels,
        use_sigma_delta=args.sigma_delta,
        codebook_size=args.codebook_size,
        device=args.device,
    )

    compressed_weights = {}
    for name, tensor in original_weights.items():
        if tensor.ndim >= 2 and tensor.numel() >= 1024:
            result, layer = compress_weight(name, tensor, pipe_config)

            # Decompress for inference
            with torch.no_grad():
                if layer.residual_codebooks:
                    recon = torch.zeros(tensor.shape[0], tensor.shape[1] if tensor.ndim > 1 else 1,
                                       device=pipe_config.device)
                    for cb_U, cb_V in layer.residual_codebooks:
                        recon = recon + cb_U.decompress() @ cb_V.decompress()
                    compressed_weights[name] = recon.cpu().reshape(tensor.shape)
                elif layer.codebook_U is not None:
                    U_r = layer.codebook_U.decompress()
                    V_r = layer.codebook_V.decompress()
                    compressed_weights[name] = (U_r @ V_r).cpu().reshape(tensor.shape)
                elif layer.binarized is not None:
                    compressed_weights[name] = layer.binarized.decompress().cpu()
                else:
                    compressed_weights[name] = tensor
            print(f"  {name:50s} cos={result.cosine_sim:.6f} {result.bits_per_weight:.3f} BPW")
        else:
            compressed_weights[name] = tensor

    # Compare layer outputs
    print(f"\nRunning inference comparison ({max_layers} layers)...")
    test_input = torch.randint(0, config.vocab_size, (1, 32), device=args.device)

    results = compare_layer_outputs(
        original_weights, compressed_weights,
        config, test_input, device=args.device,
        max_layers=max_layers,
    )

    print(f"\n{'Layer':<20} {'Cosine Sim':>12} {'Rel Error':>12}")
    print("-" * 50)
    for key, val in results.items():
        if isinstance(val, dict) and "cosine_sim" in val:
            rel_err = val.get("relative_error", "N/A")
            if isinstance(rel_err, float):
                print(f"{key:<20} {val['cosine_sim']:>12.8f} {rel_err:>12.8f}")
            else:
                print(f"{key:<20} {val['cosine_sim']:>12.8f} {'':>12}")

    if "top10_agreement" in results:
        print(f"\nTop-10 token agreement: {results['top10_agreement']*100:.0f}%")
    if "top1_match" in results:
        print(f"Top-1 prediction match: {'YES' if results['top1_match'] else 'NO'}")

    print("\n" + "=" * 65)


def main():
    parser = argparse.ArgumentParser(description="UltraCompress v2")
    parser.add_argument("--model", default="qwen3:4b", help="Ollama model name")
    parser.add_argument("--target-bpw", type=float, default=0.5)
    parser.add_argument("--min-cosine", type=float, default=0.999, help="Minimum cosine similarity")
    parser.add_argument("--energy-target", type=float, default=0.999)
    parser.add_argument("--max-tensors", type=int, default=None)
    parser.add_argument("--filter", type=str, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--codebook-size", type=int, default=4096)
    parser.add_argument("--max-rank", type=int, default=1024)
    parser.add_argument("--min-tensor-size", type=int, default=1024)
    parser.add_argument("--residual", action="store_true", default=True, help="Multi-level residual")
    parser.add_argument("--no-residual", action="store_false", dest="residual")
    parser.add_argument("--residual-levels", type=int, default=3)
    parser.add_argument("--sigma-delta", action="store_true", default=True)
    parser.add_argument("--no-sigma-delta", action="store_false", dest="sigma_delta")

    # Vector quantization
    parser.add_argument("--vq", action="store_true", default=True, help="Enable VQ path")
    parser.add_argument("--no-vq", action="store_false", dest="vq")
    parser.add_argument("--vq-codebook-size", type=int, default=256)
    parser.add_argument("--vq-group-sizes", type=int, nargs="+", default=[16, 12, 8])
    parser.add_argument("--vq-levels", type=int, default=2, help="Residual VQ levels")
    parser.add_argument("--vq-iter", type=int, default=15, help="k-means iterations")

    # Modes
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--inference", action="store_true")
    parser.add_argument("--max-layers", type=int, default=None, help="Max layers for inference test")

    args = parser.parse_args()

    if args.sweep:
        run_sweep(args)
    elif args.inference:
        run_inference_test(args)
    else:
        run_compression(args)


if __name__ == "__main__":
    main()
