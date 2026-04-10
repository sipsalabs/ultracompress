#!/usr/bin/env python3
"""
UltraCompress v8 — Compound Output-Aware Compression Pipeline

The breakthrough: track activations through the FULL transformer layer
(attention + FFN + residual) so ALL weights get output-aware refinement.

Previous run_output_aware.py only refined 2/7 weights per layer because
it couldn't track activations through attention. This pipeline does a
simplified but complete forward pass, giving every weight matrix the
activation context needed for output-aware gradient refinement.

Results from v7 (partial coverage):
  2/7 weights refined → 1.0000 cosine by layer 6 at 0.32 BPW
  But 5/7 unrefined → 0.39 cosine (avg dragged to 0.54)

Expected from v8 (full coverage):
  7/7 weights refined → target 0.99+ avg cosine

Usage:
    python run_compound.py                              # Default
    python run_compound.py --model Qwen/Qwen3-0.6B      # Small model
    python run_compound.py --bpw 0.5 --layers 8          # More layers
    python run_compound.py --bpw 0.19 --layers 4         # Extreme
    python run_compound.py --steps 1000                   # More refinement
    python run_compound.py --mixed-precision               # Higher BPW for first/last layers
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

from ultracompress.product_quantize import product_quantize
from ultracompress.ultra_pq import refine_codebooks_gradient
from ultracompress.safetensors_loader import load_hf_model
from ultracompress.metrics import compute_quality


# BPW presets: (M, K, G)
BPW_CONFIGS = {
    0.016: (8, 4, 1024),
    0.05:  (8, 4, 512),
    0.094: (8, 4, 256),
    0.125: (8, 4, 256),
    0.1875:(8, 8, 128),
    0.25:  (8, 16, 128),
    0.5:   (8, 16, 64),
    0.75:  (8, 64, 64),
    1.0:   (4, 256, 32),
    1.5:   (8, 256, 32),
}


def find_closest_config(target_bpw):
    best_bpw = min(BPW_CONFIGS.keys(), key=lambda x: abs(x - target_bpw))
    return best_bpw, BPW_CONFIGS[best_bpw]


def get_layer_bpw_multiplier(layer_idx, n_layers, mixed_precision=False):
    """Mixed precision: give first/last layers more bits.

    Layer 0 has no upstream error context for error reversal, so it
    needs higher quality PQ to avoid the v_proj bottleneck.
    Last layer feeds directly into the output head.
    """
    if not mixed_precision:
        return 1.0
    if layer_idx == 0:
        return 4.0  # 4x more bits for first layer
    if layer_idx == n_layers - 1:
        return 2.0  # 2x more for last layer
    if layer_idx == 1:
        return 2.0  # 2x for second layer (still building context)
    return 1.0


def rms_norm(x, weight, eps=1e-6):
    """RMSNorm forward pass."""
    variance = x.float().pow(2).mean(-1, keepdim=True)
    x_normed = x.float() * torch.rsqrt(variance + eps)
    return (x_normed * weight.float()).to(x.dtype)


def compress_and_refine(weight, activations, M, K, G, n_iter, refine_steps, lr, device):
    """PQ compress + output-aware gradient refinement.

    Returns: (compressed_weight, pq, stats)
    """
    w = weight.float().to(device)

    # k-means PQ
    pq = product_quantize(w, n_subvectors=M, codebook_size=K, group_size=G, n_iter=n_iter)

    recon_km = pq.decompress().reshape(w.shape).to(device)
    wcos_km = F.cosine_similarity(w.reshape(1, -1), recon_km.reshape(1, -1)).item()

    # Output-aware refinement
    if activations is not None and refine_steps > 0:
        pq_ref = refine_codebooks_gradient(
            pq, w, n_steps=refine_steps, lr=lr, activations=activations,
        )
    else:
        pq_ref = pq

    recon_ref = pq_ref.decompress().reshape(w.shape).to(device)
    wcos_ref = F.cosine_similarity(w.reshape(1, -1), recon_ref.reshape(1, -1)).item()

    # Output cosine (if activations available)
    ocos_km = wcos_km
    ocos_ref = wcos_ref
    if activations is not None and activations.shape[-1] == w.shape[1]:
        true_out = activations @ w.t()
        out_km = activations @ recon_km.t()
        out_ref = activations @ recon_ref.t()
        ocos_km = F.cosine_similarity(true_out.reshape(1, -1), out_km.reshape(1, -1)).item()
        ocos_ref = F.cosine_similarity(true_out.reshape(1, -1), out_ref.reshape(1, -1)).item()

    stats = {
        'wcos_km': wcos_km, 'wcos_ref': wcos_ref,
        'ocos_km': ocos_km, 'ocos_ref': ocos_ref,
        'bpw': pq_ref.bits_per_weight,
        'params': weight.numel(),
        'comp_bytes': pq_ref.storage_bytes(),
    }

    return recon_ref, pq_ref, stats


def parse_layer_weights(weights_dict, layer_idx):
    """Extract the 7 weight matrices for a transformer layer."""
    prefix = f'model.layers.{layer_idx}.'
    result = {}
    mapping = {
        'q_proj': 'self_attn.q_proj.weight',
        'k_proj': 'self_attn.k_proj.weight',
        'v_proj': 'self_attn.v_proj.weight',
        'o_proj': 'self_attn.o_proj.weight',
        'gate_proj': 'mlp.gate_proj.weight',
        'up_proj': 'mlp.up_proj.weight',
        'down_proj': 'mlp.down_proj.weight',
    }
    norms = {
        'input_layernorm': 'input_layernorm.weight',
        'post_attention_layernorm': 'post_attention_layernorm.weight',
        'q_norm': 'self_attn.q_norm.weight',
        'k_norm': 'self_attn.k_norm.weight',
    }
    for key, suffix in mapping.items():
        full = prefix + suffix
        if full in weights_dict:
            result[key] = weights_dict[full]
    for key, suffix in norms.items():
        full = prefix + suffix
        if full in weights_dict:
            result[key] = weights_dict[full]
    return result


def simplified_attention(q, k, v, n_heads, n_kv_heads):
    """Simplified multi-head attention (no RoPE, no causal mask).

    This is sufficient for generating activations for output-aware PQ.
    We don't need perfect attention — we need realistic activation
    distributions so the gradient refinement knows which output
    dimensions matter.
    """
    B, T, _ = q.shape
    head_dim_q = q.shape[-1] // n_heads
    head_dim_kv = k.shape[-1] // n_kv_heads

    q = q.reshape(B, T, n_heads, head_dim_q).transpose(1, 2)
    k = k.reshape(B, T, n_kv_heads, head_dim_kv).transpose(1, 2)
    v = v.reshape(B, T, n_kv_heads, head_dim_kv).transpose(1, 2)

    # GQA: repeat k/v
    if n_kv_heads < n_heads:
        repeat = n_heads // n_kv_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)

    # Truncate Q to KV dim for dot product if needed
    if head_dim_q != head_dim_kv:
        q_attn = q[..., :head_dim_kv]
    else:
        q_attn = q

    attn = torch.matmul(q_attn.float(), k.float().transpose(-2, -1))
    attn = attn / math.sqrt(head_dim_kv)
    attn = F.softmax(attn, dim=-1)
    out = torch.matmul(attn, v.float())
    return out.transpose(1, 2).reshape(B, T, -1)


def compress_layer_full(
    layer_weights, activations, M, K, G, n_iter, refine_steps, lr, device,
    n_heads, n_kv_heads, layer_idx,
):
    """Compress ALL weights in a transformer layer with full activation tracking.

    The data flow:
      x → layernorm → q/k/v projections → attention → o_proj → residual
        → layernorm → gate/up → silu(gate)*up → down → residual

    At each weight, we have the correct input activations for output-aware refinement.
    """
    results = []
    x = activations  # (B*T, hidden_dim) flattened sequences

    # We need 3D for attention: reshape to (B, T, hidden)
    # Use batch_size=1, T=all tokens
    T = x.shape[0]
    hidden = x.shape[1]
    x_3d = x.unsqueeze(0)  # (1, T, hidden)

    # === ATTENTION BLOCK ===
    # 1. Input layernorm
    if 'input_layernorm' in layer_weights:
        x_norm_3d = rms_norm(x_3d, layer_weights['input_layernorm'].to(device))
    else:
        x_norm_3d = x_3d
    x_norm = x_norm_3d.reshape(-1, hidden)  # Back to 2D for weight compression

    # 2. Q projection — activations = x_norm
    print(f"    q_proj ", end="", flush=True)
    if 'q_proj' in layer_weights:
        q_recon, _, stats = compress_and_refine(
            layer_weights['q_proj'], x_norm, M, K, G, n_iter, refine_steps, lr, device,
        )
        results.append(('q_proj', stats))
        q = (x_norm @ q_recon.t()).unsqueeze(0)  # (1, T, q_dim)
        print(f"ocos={stats['ocos_ref']:.4f} ", end="", flush=True)
    else:
        q = x_norm_3d

    # 3. K projection — activations = x_norm
    print(f"k_proj ", end="", flush=True)
    if 'k_proj' in layer_weights:
        k_recon, _, stats = compress_and_refine(
            layer_weights['k_proj'], x_norm, M, K, G, n_iter, refine_steps, lr, device,
        )
        results.append(('k_proj', stats))
        k = (x_norm @ k_recon.t()).unsqueeze(0)
        print(f"ocos={stats['ocos_ref']:.4f} ", end="", flush=True)
    else:
        k = x_norm_3d

    # 4. V projection — activations = x_norm
    print(f"v_proj ", end="", flush=True)
    if 'v_proj' in layer_weights:
        v_recon, _, stats = compress_and_refine(
            layer_weights['v_proj'], x_norm, M, K, G, n_iter, refine_steps, lr, device,
        )
        results.append(('v_proj', stats))
        v = (x_norm @ v_recon.t()).unsqueeze(0)
        print(f"ocos={stats['ocos_ref']:.4f} ", end="", flush=True)
    else:
        v = x_norm_3d

    # 5. Attention computation (simplified — no RoPE, no causal mask)
    with torch.no_grad():
        attn_out = simplified_attention(q, k, v, n_heads, n_kv_heads)
    attn_out_2d = attn_out.reshape(-1, attn_out.shape[-1])

    # 6. O projection — activations = attention output
    print(f"o_proj ", end="", flush=True)
    if 'o_proj' in layer_weights:
        o_recon, _, stats = compress_and_refine(
            layer_weights['o_proj'], attn_out_2d, M, K, G, n_iter, refine_steps, lr, device,
        )
        results.append(('o_proj', stats))
        o_out = attn_out_2d @ o_recon.t()
        print(f"ocos={stats['ocos_ref']:.4f}", flush=True)
    else:
        o_out = attn_out_2d

    # Residual connection
    h = x + o_out  # (T, hidden)
    h_3d = h.unsqueeze(0)

    # === FFN BLOCK ===
    # 7. Post-attention layernorm
    if 'post_attention_layernorm' in layer_weights:
        h_norm_3d = rms_norm(h_3d, layer_weights['post_attention_layernorm'].to(device))
    else:
        h_norm_3d = h_3d
    h_norm = h_norm_3d.reshape(-1, hidden)

    # 8. Gate projection — activations = h_norm
    print(f"    gate   ", end="", flush=True)
    if 'gate_proj' in layer_weights:
        gate_recon, _, stats = compress_and_refine(
            layer_weights['gate_proj'], h_norm, M, K, G, n_iter, refine_steps, lr, device,
        )
        results.append(('gate_proj', stats))
        gate = h_norm @ gate_recon.t()
        print(f"ocos={stats['ocos_ref']:.4f} ", end="", flush=True)
    else:
        gate = h_norm

    # 9. Up projection — activations = h_norm
    print(f"up ", end="", flush=True)
    if 'up_proj' in layer_weights:
        up_recon, _, stats = compress_and_refine(
            layer_weights['up_proj'], h_norm, M, K, G, n_iter, refine_steps, lr, device,
        )
        results.append(('up_proj', stats))
        up = h_norm @ up_recon.t()
        print(f"ocos={stats['ocos_ref']:.4f} ", end="", flush=True)
    else:
        up = h_norm

    # 10. SiLU(gate) * up → hidden
    with torch.no_grad():
        ffn_hidden = F.silu(gate) * up

    # 11. Down projection — activations = ffn_hidden
    print(f"down ", end="", flush=True)
    if 'down_proj' in layer_weights:
        down_recon, _, stats = compress_and_refine(
            layer_weights['down_proj'], ffn_hidden, M, K, G, n_iter, refine_steps, lr, device,
        )
        results.append(('down_proj', stats))
        down_out = ffn_hidden @ down_recon.t()
        print(f"ocos={stats['ocos_ref']:.4f}", flush=True)
    else:
        down_out = ffn_hidden

    # Residual connection
    output = h + down_out  # (T, hidden)

    return results, output


def run_pipeline(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    actual_bpw, (M, K, G) = find_closest_config(args.bpw)
    computed_bpw = M * math.log2(K) / G

    print("=" * 70)
    print(f"  UltraCompress v8 — Compound Output-Aware Pipeline")
    print("=" * 70)
    print(f"  Model:    {args.model}")
    print(f"  Target:   {args.bpw:.3f} BPW")
    print(f"  Config:   M={M} K={K} G={G} => {computed_bpw:.4f} BPW")
    print(f"  Refine:   {args.steps} steps, lr={args.lr}")
    print(f"  Device:   {device}")
    print()

    # Load model
    print("Loading model weights...")
    t_load = time.time()
    weights_dict = {}
    for name, tensor in load_hf_model(args.model):
        weights_dict[name] = tensor
    print(f"  Loaded {len(weights_dict)} tensors ({time.time()-t_load:.1f}s)")

    # Detect architecture
    n_layers = 0
    for key in weights_dict:
        if 'model.layers.' in key:
            try:
                idx = int(key.split('model.layers.')[1].split('.')[0])
                n_layers = max(n_layers, idx + 1)
            except (ValueError, IndexError):
                pass

    max_layers = min(args.layers, n_layers) if args.layers else n_layers

    # Detect head counts from weight shapes
    layer0 = parse_layer_weights(weights_dict, 0)
    hidden_size = layer0['q_proj'].shape[1] if 'q_proj' in layer0 else 1024
    q_out = layer0['q_proj'].shape[0] if 'q_proj' in layer0 else hidden_size
    k_out = layer0['k_proj'].shape[0] if 'k_proj' in layer0 else hidden_size

    # Infer head counts
    head_dim = 128  # Common default
    if 'q_norm' in layer0:
        head_dim = layer0['q_norm'].shape[0]
    n_heads = q_out // head_dim
    n_kv_heads = k_out // head_dim

    print(f"  Layers: {n_layers} total, compressing {max_layers}")
    print(f"  Hidden: {hidden_size}, Heads: {n_heads}, KV Heads: {n_kv_heads}")

    # Generate initial activations from embedding
    embed_key = 'model.embed_tokens.weight'
    if embed_key not in weights_dict:
        print("ERROR: No embedding found!")
        return
    embed = weights_dict[embed_key].float().to(device)
    print(f"  Embedding: {embed.shape}")

    tokens = torch.randint(0, embed.shape[0], (args.n_tokens, args.seq_len), device=device)
    activations = F.embedding(tokens, embed).reshape(-1, embed.shape[1])
    print(f"  Activations: {activations.shape}")
    print()

    # Compress layer by layer
    all_results = []
    total_params = 0
    total_comp_bytes = 0
    layer_avg_ocos = []

    for layer_idx in range(max_layers):
        t0 = time.time()
        layer_weights = parse_layer_weights(weights_dict, layer_idx)
        if not layer_weights:
            continue

        # Mixed precision: adjust BPW per layer
        bpw_mult = get_layer_bpw_multiplier(layer_idx, max_layers, args.mixed_precision)
        if bpw_mult != 1.0:
            layer_bpw = min(args.bpw * bpw_mult, 1.5)  # Cap at 1.5 BPW
            _, (lM, lK, lG) = find_closest_config(layer_bpw)
            print(f"  Layer {layer_idx} (mixed: {bpw_mult:.0f}x -> {layer_bpw:.3f} BPW, M={lM} K={lK} G={lG}):")
        else:
            lM, lK, lG = M, K, G
            print(f"  Layer {layer_idx}:")

        results, activations = compress_layer_full(
            layer_weights, activations, lM, lK, lG,
            n_iter=20, refine_steps=args.steps, lr=args.lr,
            device=device, n_heads=n_heads, n_kv_heads=n_kv_heads,
            layer_idx=layer_idx,
        )

        layer_ocos = []
        for name, stats in results:
            total_params += stats['params']
            total_comp_bytes += stats['comp_bytes']
            all_results.append((f"L{layer_idx}.{name}", stats))
            layer_ocos.append(stats['ocos_ref'])

        avg_ocos = np.mean(layer_ocos) if layer_ocos else 0
        layer_avg_ocos.append(avg_ocos)
        dt = time.time() - t0
        print(f"    -> Layer {layer_idx} avg output cosine: {avg_ocos:.4f} ({dt:.1f}s)")
        print()

        # Cleanup
        del layer_weights
        torch.cuda.empty_cache()

    # Final summary
    avg_bpw = (total_comp_bytes * 8) / total_params if total_params > 0 else 0
    all_ocos = [s['ocos_ref'] for _, s in all_results]
    all_wcos = [s['wcos_ref'] for _, s in all_results]

    print("=" * 70)
    print(f"  RESULTS — {max_layers} layers, {len(all_results)} weights")
    print("=" * 70)
    print(f"  Avg BPW:             {avg_bpw:.4f}")
    print(f"  Weight cosine avg:   {np.mean(all_wcos):.6f}")
    print(f"  Output cosine avg:   {np.mean(all_ocos):.6f}")
    print(f"  Output cosine min:   {min(all_ocos):.6f}")
    print(f"  Compression ratio:   {total_params * 2 / max(total_comp_bytes, 1):.0f}x from FP16")
    print()

    # Per-layer progression
    print("  Layer progression (avg output cosine):")
    for i, avg in enumerate(layer_avg_ocos):
        bar = "#" * int(avg * 40)
        print(f"    Layer {i:>2}: {avg:.4f} |{bar}")
    print()

    # Per-weight-type breakdown
    print("  Per weight type (avg across layers):")
    type_cos = {}
    for name, stats in all_results:
        wtype = name.split('.')[-1]
        type_cos.setdefault(wtype, []).append(stats['ocos_ref'])
    for wtype in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
        if wtype in type_cos:
            avg = np.mean(type_cos[wtype])
            print(f"    {wtype:12s}: {avg:.4f}")
    print()

    # Projections
    print(f"  PROJECTIONS at {avg_bpw:.4f} BPW:")
    print(f"  {'Model':20s} {'FP16':>8} {'Compressed':>11} {'Ratio':>7} {'20GB?':>6}")
    print(f"  {'-'*55}")
    for model_name, size_b in [
        ("8B", 8), ("70B", 70), ("235B", 235), ("405B", 405),
        ("671B", 671), ("1T", 1000), ("10T", 10000),
        ("100T", 100000), ("1000T", 1000000),
    ]:
        fp16_gb = size_b * 2
        comp_gb = size_b * 1e9 * avg_bpw / 8 / 1e9
        ratio = fp16_gb / max(comp_gb, 0.001)
        fits = "YES" if comp_gb <= 20 else "NO"
        print(f"  {model_name:20s} {fp16_gb:>6} GB {comp_gb:>8.1f} GB {ratio:>6.0f}x   {fits}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="UltraCompress v8 — Compound Pipeline")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B",
                       help="HuggingFace model ID")
    parser.add_argument("--bpw", type=float, default=0.5,
                       help="Target bits per weight")
    parser.add_argument("--layers", type=int, default=4,
                       help="Number of layers to compress")
    parser.add_argument("--steps", type=int, default=500,
                       help="Gradient refinement steps per weight")
    parser.add_argument("--lr", type=float, default=0.005,
                       help="Learning rate for refinement")
    parser.add_argument("--n-tokens", type=int, default=64,
                       help="Calibration token sequences")
    parser.add_argument("--seq-len", type=int, default=128,
                       help="Sequence length for calibration")
    parser.add_argument("--mixed-precision", action="store_true",
                       help="Use higher BPW for first/last layers")

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
