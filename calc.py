#!/usr/bin/env python3
"""
UltraCompress Calculator — estimate compression before running.

Usage:
  python calc.py --model Qwen/Qwen3-0.6B
  python calc.py --params 70B --layers 80 --hidden 8192
  python calc.py --params 100T --layers 200 --hidden 16384
"""
import argparse


def calc(total_params, n_layers, hidden_size, vocab_size=151936,
         n_heads=None, ff_mult=1, phm=False):
    """Calculate FRR compression ratios."""
    if n_heads is None:
        n_heads = hidden_size // 128

    # FRR block size
    head_dim = hidden_size // n_heads
    q_params = hidden_size * hidden_size
    k_params = hidden_size * (hidden_size // 2)  # GQA estimate
    v_params = k_params
    o_params = hidden_size * hidden_size
    attn = q_params + k_params + v_params + o_params

    if ff_mult == 1:
        ffn = 3 * hidden_size * hidden_size  # gate + up + down
    else:
        ffn = 3 * hidden_size * hidden_size * ff_mult

    norms = hidden_size * 2
    block = attn + ffn + norms

    if phm:
        block = block // 4  # PHM gives 4x reduction

    # Modulation
    mod = n_layers * hidden_size * 3  # gamma + beta + iter_scale

    # Embedding (shared, not compressed)
    embed = vocab_size * hidden_size * 2  # input + output (or tied)

    # Layer params
    layer_params = total_params - embed
    if layer_params < 0:
        layer_params = total_params * 0.95  # estimate

    # Compression ratios
    frr_ratio = layer_params / (block + mod)
    frr_size_fp32 = (block + mod) * 4
    frr_size_fp16 = (block + mod) * 2

    # With quantization pipeline
    q4_ratio = frr_ratio * 8  # FP32 -> 4bit
    q2_ratio = frr_ratio * 16  # FP32 -> 2bit
    q2_entropy_ratio = q2_ratio * 6  # 6x from entropy coding on Q2

    # Full model sizes
    total_fp16 = total_params * 2
    frr_total = frr_size_fp16 + embed * 2  # block FP16 + embed FP16

    return {
        'block_params': block,
        'mod_params': mod,
        'embed_params': embed,
        'layer_params': layer_params,
        'frr_ratio': frr_ratio,
        'frr_size_fp16': frr_size_fp16,
        'frr_total': frr_total,
        'total_fp16': total_fp16,
        'q4_ratio': q4_ratio,
        'q2_ratio': q2_ratio,
        'q2_entropy_ratio': q2_entropy_ratio,
        'phm': phm,
    }


def format_size(bytes_val):
    if bytes_val >= 1e12:
        return f"{bytes_val/1e12:.1f} TB"
    elif bytes_val >= 1e9:
        return f"{bytes_val/1e9:.1f} GB"
    elif bytes_val >= 1e6:
        return f"{bytes_val/1e6:.1f} MB"
    else:
        return f"{bytes_val/1e3:.1f} KB"


def parse_params(s):
    s = s.upper().replace(',', '')
    if s.endswith('T'):
        return int(float(s[:-1]) * 1e12)
    elif s.endswith('B'):
        return int(float(s[:-1]) * 1e9)
    elif s.endswith('M'):
        return int(float(s[:-1]) * 1e6)
    return int(s)


# Known models
KNOWN_MODELS = {
    'Qwen/Qwen3-0.6B': (751_632_384, 28, 1024),
    'Qwen/Qwen3-1.7B': (2_031_739_904, 28, 2048),
    'Qwen/Qwen3-4B': (4_000_000_000, 36, 3584),
    'Qwen/Qwen3-8B': (8_000_000_000, 32, 4096),
    'meta-llama/Llama-3.1-8B': (8_000_000_000, 32, 4096),
    'meta-llama/Llama-3.1-70B': (70_000_000_000, 80, 8192),
    'meta-llama/Llama-3.1-405B': (405_000_000_000, 126, 16384),
}


def main():
    parser = argparse.ArgumentParser(description="UltraCompress Calculator")
    parser.add_argument('--model', help='Known model name (e.g., Qwen/Qwen3-8B)')
    parser.add_argument('--params', help='Total params (e.g., 8B, 70B, 100T)')
    parser.add_argument('--layers', type=int, help='Number of layers')
    parser.add_argument('--hidden', type=int, help='Hidden size')
    parser.add_argument('--phm', action='store_true', help='Use PHM (4x fewer block params)')
    args = parser.parse_args()

    if args.model and args.model in KNOWN_MODELS:
        total, layers, hidden = KNOWN_MODELS[args.model]
        name = args.model
    elif args.params:
        total = parse_params(args.params)
        layers = args.layers or 32
        hidden = args.hidden or 4096
        name = f"{args.params} model"
    else:
        # Default: show all known models
        print(f"{'='*70}")
        print(f"  ULTRACOMPRESS COMPRESSION CALCULATOR")
        print(f"{'='*70}")
        print(f"\n  {'Model':<30} {'Original':>10} {'FRR':>10} {'FRR+Q2':>10} {'FRR+Q2+E':>10}")
        print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        for model_name, (t, l, h) in KNOWN_MODELS.items():
            r = calc(t, l, h)
            short = model_name.split('/')[-1]
            print(f"  {short:<30} {format_size(r['total_fp16']):>10} "
                  f"{format_size(r['frr_size_fp16']):>10} "
                  f"{format_size(r['frr_size_fp16']/16):>10} "
                  f"{format_size(r['frr_size_fp16']/96):>10}")

        # 100T projection
        r = calc(100_000_000_000_000, 200, 16384)
        print(f"  {'100T (projected)':<30} {format_size(r['total_fp16']):>10} "
              f"{format_size(r['frr_size_fp16']):>10} "
              f"{format_size(r['frr_size_fp16']/16):>10} "
              f"{format_size(r['frr_size_fp16']/96):>10}")

        print(f"\n  WARNING: These are THEORETICAL MAXIMUMS assuming quality holds.")
        print(f"  Actual quality depends on training. Proven results so far:")
        print(f"    0.6B: 63% T10 at 60x (50K steps) -- PROVEN")
        print(f"    1.7B: 60% T10 at 48x (20K steps) -- PROVEN, still climbing")
        print(f"    E2E:  53% T10 at 959x -- PROVEN end-to-end")
        print(f"    8B+:  NOT YET TESTED -- quality unknown at this scale")
        print(f"  The compression ratio math is real. The quality at scale is NOT proven.")
        return

    r = calc(total, layers, hidden, phm=args.phm)

    print(f"\n{'='*60}")
    print(f"  COMPRESSION ESTIMATE: {name}")
    print(f"{'='*60}")
    print(f"  Original model: {total:,} params ({format_size(r['total_fp16'])} FP16)")
    print(f"  Layers: {layers}, Hidden: {hidden}")
    print(f"  {'PHM enabled (4x fewer block params)' if args.phm else ''}")
    print(f"\n  FRR block: {r['block_params']:,} params ({format_size(r['block_params']*2)} FP16)")
    print(f"  Modulation: {r['mod_params']:,} params")
    print(f"\n  --- Compression Ratios ---")
    print(f"  FRR only (FP16):     {r['frr_ratio']:.0f}x -> {format_size(r['frr_size_fp16'])}")
    print(f"  FRR + Q4:            {r['q4_ratio']:.0f}x -> {format_size(r['frr_size_fp16']/8)}")
    print(f"  FRR + Q2:            {r['q2_ratio']:.0f}x -> {format_size(r['frr_size_fp16']/16)}")
    print(f"  FRR + Q2 + entropy:  {r['q2_entropy_ratio']:.0f}x -> {format_size(r['frr_size_fp16']/96)}")
    print(f"\n  Note: embeddings ({format_size(r['embed_params']*2)}) shared from base model")


if __name__ == '__main__':
    main()
