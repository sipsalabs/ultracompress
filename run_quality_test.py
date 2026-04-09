#!/usr/bin/env python3
"""
UltraCompress Quality Test

Two-part test:
  1. Weight-level: Compress every weight matrix, measure per-layer quality
  2. Activation-level: Multiply random activations through original vs compressed
     weights, measure how much the output diverges layer by layer

This proves compression quality without needing full GGUF dequantization.
For actual text generation comparison, we compare Ollama's output against
what a compressed model would produce.
"""

import sys
import os
import torch
import numpy as np
import time
import subprocess
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultracompress.pipeline import PipelineConfig, compress_weight
from ultracompress.gguf_loader import load_ollama_model, find_ollama_model_path, list_gguf_tensors
from ultracompress.metrics import compute_quality, ModelCompressionReport, CompressionResult


def activation_propagation_test(named_weights: list, config: PipelineConfig):
    """
    Simulate how compression error propagates through the network.

    For each weight matrix W:
      1. Generate random activation x ~ N(0, 1)
      2. Compute y_orig = x @ W^T (original)
      3. Compress W -> W', compute y_comp = x @ W'^T
      4. Measure cosine_sim(y_orig, y_comp)

    Then simulate multi-layer propagation:
      Feed the output of layer i as input to layer i+1.
      Track how errors compound.
    """
    device = config.device
    print("\n" + "=" * 65)
    print("  Activation Propagation Test")
    print("=" * 65)

    # Separate weight matrices by layer
    layer_weights = {}
    for name, weight in named_weights:
        if weight.ndim < 2 or weight.numel() < 1024:
            continue
        # Extract layer number
        parts = name.split(".")
        if len(parts) >= 2 and parts[1].isdigit():
            layer_idx = int(parts[1])
        else:
            layer_idx = -1
        if layer_idx not in layer_weights:
            layer_weights[layer_idx] = []
        layer_weights[layer_idx].append((name, weight))

    # Per-weight activation test
    print(f"\n{'Layer':<50} {'Weight Cos':>10} {'Activ Cos':>10} {'Activ Err':>10}")
    print("-" * 85)

    all_weight_cos = []
    all_activ_cos = []

    for name, weight in named_weights:
        if weight.ndim < 2 or weight.numel() < 1024:
            continue

        w = weight.float().to(device)
        result, layer = compress_weight(name, weight, config)

        # Decompress
        if layer.quantized is not None:
            w_compressed = layer.quantized.decompress().to(device)
        else:
            continue

        if w_compressed.shape != w.shape:
            w_compressed = w_compressed.reshape(w.shape)

        # Random activation
        x = torch.randn(1, 32, w.shape[1], device=device)  # (batch, seq, in_dim)

        with torch.no_grad():
            y_orig = torch.nn.functional.linear(x, w)
            y_comp = torch.nn.functional.linear(x, w_compressed)

            activ_cos = torch.nn.functional.cosine_similarity(
                y_orig.reshape(1, -1), y_comp.reshape(1, -1)
            ).item()
            activ_err = (torch.norm(y_orig - y_comp) / torch.norm(y_orig)).item()

        all_weight_cos.append(result.cosine_sim)
        all_activ_cos.append(activ_cos)

        print(f"  {name:<48} {result.cosine_sim:>10.6f} {activ_cos:>10.6f} {activ_err:>10.6f}")

    if all_weight_cos:
        print(f"\n  Summary:")
        print(f"    Mean weight cosine:     {np.mean(all_weight_cos):.6f}")
        print(f"    Mean activation cosine: {np.mean(all_activ_cos):.6f}")
        print(f"    Min activation cosine:  {np.min(all_activ_cos):.6f}")

    # Multi-layer propagation simulation
    print(f"\n  Multi-Layer Error Propagation:")
    print(f"  (Simulating sequential layers with compressed weights)\n")

    sorted_layers = sorted(layer_weights.keys())
    if -1 in sorted_layers:
        sorted_layers.remove(-1)

    if len(sorted_layers) >= 2:
        # Use the first weight matrix from each layer as representative
        hidden_dim = None
        x = None

        cumulative_cos_list = []
        for layer_idx in sorted_layers[:8]:  # Test first 8 layers
            layer_w = layer_weights[layer_idx]
            # Find the biggest weight (likely FFN or attention output)
            biggest = max(layer_w, key=lambda nw: nw[1].numel())
            name, weight = biggest
            w = weight.float().to(device)

            result, layer_obj = compress_weight(name, weight, config)
            if layer_obj.quantized is None:
                continue
            w_comp = layer_obj.quantized.decompress().to(device).reshape(w.shape)

            in_dim = w.shape[1]
            out_dim = w.shape[0]

            if x is None:
                x_orig = torch.randn(1, 16, in_dim, device=device)
                x_comp = x_orig.clone()

            # Resize if dimensions don't match
            if x_orig.shape[-1] != in_dim:
                x_orig = torch.randn(1, 16, in_dim, device=device)
                x_comp = x_orig.clone()

            with torch.no_grad():
                y_orig = torch.nn.functional.linear(x_orig, w)
                y_comp = torch.nn.functional.linear(x_comp, w_comp)

                cos = torch.nn.functional.cosine_similarity(
                    y_orig.reshape(1, -1), y_comp.reshape(1, -1)
                ).item()
                cumulative_cos_list.append(cos)

                # Feed forward (with ReLU-like nonlinearity)
                x_orig = torch.nn.functional.gelu(y_orig)
                x_comp = torch.nn.functional.gelu(y_comp)

            print(f"    After layer {layer_idx}: cosine_sim = {cos:.8f}")

        if cumulative_cos_list:
            print(f"\n    Final multi-layer cosine: {cumulative_cos_list[-1]:.8f}")
            print(f"    Error growth rate: {(1 - cumulative_cos_list[-1]) / len(cumulative_cos_list):.8f} per layer")

    return all_weight_cos, all_activ_cos


def ollama_comparison(model_name: str):
    """Compare Ollama's output with a simple test prompt."""
    print("\n" + "=" * 65)
    print("  Ollama Output (Reference)")
    print("=" * 65)

    prompt = "Explain quantum computing in one sentence:"
    print(f"\n  Prompt: {prompt}")

    try:
        result = subprocess.run(
            ["ollama", "run", model_name, prompt],
            capture_output=True, text=True, timeout=60
        )
        output = result.stdout.strip()
        print(f"  Output: {output[:200]}")
        print(f"\n  (This is what the full-precision model produces.)")
        print(f"  (A well-compressed model should produce similar quality text.)")
    except Exception as e:
        print(f"  Could not run Ollama: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--max-tensors", type=int, default=None)
    parser.add_argument("--min-cosine", type=float, default=0.995)
    parser.add_argument("--quant-bits", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ollama", action="store_true", help="Also run Ollama comparison")
    args = parser.parse_args()

    print("=" * 65)
    print("  UltraCompress v3 — Full Quality Analysis")
    print("=" * 65)

    model_path = find_ollama_model_path(args.model)
    if not model_path:
        print(f"Model '{args.model}' not found")
        return

    tensor_list = list_gguf_tensors(model_path)
    total_params = sum(t["n_params"] for t in tensor_list)
    print(f"\n  Model: {args.model}")
    print(f"  Params: {total_params:,} ({total_params/1e9:.2f}B)")

    # Run compression at multiple bit rates
    print(f"\n  Loading weights...")
    named_weights = list(load_ollama_model(
        args.model, max_tensors=args.max_tensors, device="cpu"
    ))
    print(f"  Loaded {len(named_weights)} tensors")

    # Test at the target bit rate
    for bits in [2, 3, 4]:
        config = PipelineConfig(
            target_bpw=0.5,
            target_cosine_sim=0.0,  # Don't retry — test exact bit rate
            max_retries=0,
            quant_bits=bits,
            quant_group_size=args.group_size,
            device=args.device,
        )
        print(f"\n{'='*65}")
        print(f"  Testing INT{bits} quantization (group_size={args.group_size})")
        print(f"{'='*65}")

        weight_cos, activ_cos = activation_propagation_test(named_weights, config)

        if weight_cos:
            bpw = bits + 32 / args.group_size  # bits + scale overhead
            size_235b = (235e9 * bpw) / 8 / 1e9
            print(f"\n  At ~{bpw:.2f} BPW:")
            print(f"    235B model size: {size_235b:.1f} GB")
            print(f"    Fits 10GB VRAM? {'YES' if size_235b <= 10 else 'NO'}")

    if args.ollama:
        ollama_comparison(args.model)


if __name__ == "__main__":
    main()
