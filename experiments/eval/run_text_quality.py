#!/usr/bin/env python3
"""
UltraCompress v8 — Text Generation Quality Validation

THE REAL TEST: Does the compound pipeline produce usable text?

Compresses weights using the compound pipeline (with proper
sequential activation propagation), then runs actual inference
through the MiniTransformer and compares token-level quality.

Usage:
    python run_text_quality.py                              # Default
    python run_text_quality.py --bpw 0.05 --layers 8        # More compression
"""

import argparse
import sys
import os
import time
import math
import torch
import torch.nn.functional as F
import numpy as np

os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultracompress.safetensors_loader import load_hf_model
from ultracompress.inference import ModelConfig, MiniTransformer, compare_layer_outputs
from run_compound import (
    find_closest_config, parse_layer_weights, compress_layer_full,
    get_layer_bpw_multiplier,
)


def main():
    parser = argparse.ArgumentParser(description="Text Quality Validation")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--bpw", type=float, default=0.5)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--max-tokens", type=int, default=30)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    actual_bpw, (M, K, G) = find_closest_config(args.bpw)

    print("=" * 70)
    print("  UltraCompress v8 -- TEXT GENERATION QUALITY TEST")
    print("=" * 70)
    print(f"  Model:   {args.model}")
    print(f"  Config:  M={M} K={K} G={G} ({actual_bpw:.4f} BPW)")
    print(f"  Layers:  {args.layers}")
    print()

    # Load model
    print("Loading model weights...")
    weights_dict = {}
    for name, tensor in load_hf_model(args.model):
        weights_dict[name] = tensor

    # Detect architecture
    n_layers = 0
    for key in weights_dict:
        if 'model.layers.' in key:
            try:
                idx = int(key.split('model.layers.')[1].split('.')[0])
                n_layers = max(n_layers, idx + 1)
            except:
                pass
    max_layers = min(args.layers, n_layers)

    layer0 = parse_layer_weights(weights_dict, 0)
    hidden_size = layer0['q_proj'].shape[1]
    head_dim = layer0['q_norm'].shape[0] if 'q_norm' in layer0 else 128
    n_heads = layer0['q_proj'].shape[0] // head_dim
    n_kv_heads = layer0['k_proj'].shape[0] // head_dim

    config = ModelConfig(
        n_layers=max_layers, n_heads=n_heads, n_kv_heads=n_kv_heads,
        hidden_size=hidden_size,
        intermediate_size=layer0['gate_proj'].shape[0],
        vocab_size=weights_dict['model.embed_tokens.weight'].shape[0],
        head_dim=head_dim,
    )
    print(f"  {max_layers} layers, hidden={hidden_size}, heads={n_heads}")

    # HF -> GGUF name mapping
    hf_to_gguf = {
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
    type_to_hf = {
        'q_proj': 'self_attn.q_proj.weight',
        'k_proj': 'self_attn.k_proj.weight',
        'v_proj': 'self_attn.v_proj.weight',
        'o_proj': 'self_attn.o_proj.weight',
        'gate_proj': 'mlp.gate_proj.weight',
        'up_proj': 'mlp.up_proj.weight',
        'down_proj': 'mlp.down_proj.weight',
    }

    def build_gguf(wd, max_l):
        """Convert HF names to GGUF names for MiniTransformer."""
        gd = {}
        if 'model.embed_tokens.weight' in wd:
            gd['token_embd.weight'] = wd['model.embed_tokens.weight']
        if 'model.norm.weight' in wd:
            gd['output_norm.weight'] = wd['model.norm.weight']
        if 'lm_head.weight' in wd:
            gd['output.weight'] = wd['lm_head.weight']
        for li in range(max_l):
            for hf_s, gguf_s in hf_to_gguf.items():
                hf_k = f'model.layers.{li}.{hf_s}'
                gguf_k = f'blk.{li}.{gguf_s}'
                if hf_k in wd:
                    gd[gguf_k] = wd[hf_k]
        return gd

    # =============================================
    # Step 1: Compress with compound pipeline
    # =============================================
    print(f"\nCompressing {max_layers} layers with compound pipeline...")
    compressed_dict = dict(weights_dict)  # Start with copy

    embed = weights_dict['model.embed_tokens.weight'].float().to(device)
    tokens = torch.randint(0, embed.shape[0], (32, 64), device=device)
    activations = F.embedding(tokens, embed).reshape(-1, embed.shape[1])

    for layer_idx in range(max_layers):
        layer_w = parse_layer_weights(weights_dict, layer_idx)
        if not layer_w:
            continue

        results, activations, comp_weights = compress_layer_full(
            layer_w, activations, M, K, G,
            n_iter=20, refine_steps=args.steps, lr=args.lr,
            device=device, n_heads=n_heads, n_kv_heads=n_kv_heads,
            layer_idx=layer_idx,
        )

        # Store compressed weights
        prefix = f'model.layers.{layer_idx}.'
        for wtype, recon_tensor in comp_weights.items():
            hf_suffix = type_to_hf.get(wtype)
            if hf_suffix:
                compressed_dict[prefix + hf_suffix] = recon_tensor

        ocos_vals = [s['ocos_ref'] for _, s in results]
        print(f"    Layer {layer_idx}: avg ocos={np.mean(ocos_vals):.4f}")

        torch.cuda.empty_cache()

    # =============================================
    # Step 2: Run inference comparison
    # =============================================
    print("\nBuilding MiniTransformer models...")
    orig_gguf = build_gguf(weights_dict, max_layers)
    comp_gguf = build_gguf(compressed_dict, max_layers)

    prompt_tokens = torch.randint(100, 5000, (1, 16), device=device)
    print(f"  Prompt: {prompt_tokens[0, :8].tolist()}...")

    # Use compare_layer_outputs for layer-by-layer comparison
    print("\nLayer-by-layer comparison (MiniTransformer with real attention):")
    comp_results = compare_layer_outputs(
        orig_gguf, comp_gguf, config,
        prompt_tokens, device=device, max_layers=max_layers,
    )

    print(f"\n  {'Layer':<15} {'Cosine':>10} {'Rel Error':>10}")
    print(f"  {'-'*38}")
    for key, val in comp_results.items():
        if isinstance(val, dict) and 'cosine_sim' in val:
            cos = val['cosine_sim']
            err = val.get('relative_error', 'N/A')
            if isinstance(err, float):
                print(f"  {key:<15} {cos:>10.6f} {err:>10.6f}")
            else:
                print(f"  {key:<15} {cos:>10.6f}")

    if 'top10_agreement' in comp_results:
        print(f"\n  Top-10 agreement: {comp_results['top10_agreement']*100:.0f}%")
    if 'top1_match' in comp_results:
        print(f"  Top-1 match:      {'YES' if comp_results['top1_match'] else 'NO'}")

    # Token generation comparison
    print("\nToken generation comparison:")
    orig_model = MiniTransformer(config, device)
    orig_model.load_weights(orig_gguf)
    comp_model = MiniTransformer(config, device)
    comp_model.load_weights(comp_gguf)

    with torch.no_grad():
        orig_gen = orig_model.generate(prompt_tokens.to(device), max_new=args.max_tokens, temperature=0.01)
        comp_gen = comp_model.generate(prompt_tokens.to(device), max_new=args.max_tokens, temperature=0.01)

    min_len = min(len(orig_gen), len(comp_gen))
    if min_len > 0:
        matches = sum(1 for a, b in zip(orig_gen[:min_len], comp_gen[:min_len]) if a == b)
        print(f"  Token agreement: {matches}/{min_len} ({matches/min_len*100:.0f}%)")
    print(f"  Original:   {orig_gen[:15]}")
    print(f"  Compressed: {comp_gen[:15]}")

    # Summary
    print()
    print("=" * 70)
    logit_key = 'logits'
    logit_cos = comp_results.get(logit_key, {}).get('cosine_sim', 0)
    top10 = comp_results.get('top10_agreement', 0)
    top1 = comp_results.get('top1_match', False)

    print(f"  Logit cosine:     {logit_cos:.6f}")
    print(f"  Top-10 agreement: {top10*100:.0f}%")
    print(f"  Top-1 match:      {'YES' if top1 else 'NO'}")
    if min_len > 0:
        print(f"  Token agreement:  {matches/min_len*100:.0f}%")

    if logit_cos > 0.9 and top10 > 0.5:
        print("\n  VERDICT: GOOD — compression preserves text quality")
    elif logit_cos > 0.5:
        print("\n  VERDICT: PARTIAL — some quality preserved, needs more refinement")
    else:
        print("\n  VERDICT: POOR — quality not preserved, investigation needed")
    print("=" * 70)


if __name__ == "__main__":
    main()
