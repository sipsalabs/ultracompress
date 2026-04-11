#!/usr/bin/env python3
"""
UltraCompress CLI — Compress any model to a genome.

Usage:
    python ultracompress.py compress --model Qwen/Qwen3-0.6B --target-size 20MB
    python ultracompress.py compress --model Qwen/Qwen3-8B --sd 128 --steps 50000
    python ultracompress.py run --genome genome_sd128_28L.pt --prompt "Hello world"
    python ultracompress.py info --genome genome_sd128_28L.pt
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def cmd_compress(args):
    """Compress a model into a genome."""
    import torch
    from ultracompress.inference import ModelConfig
    from ultracompress.genome_compressor import GenomeCompressor

    print(f"UltraCompress — Genome Compression Engine")
    print(f"Model: {args.model}")
    print(f"Target: sd={args.sd}, {args.steps} steps")
    print()

    # Load model
    if args.model.endswith('.pt'):
        wd = torch.load(args.model, weights_only=True)
    else:
        from ultracompress.safetensors_loader import load_hf_model
        wd = {}
        for name, tensor in load_hf_model(args.model):
            wd[name] = tensor

    # Detect config
    n_layers = 0
    hidden = 0
    for key in wd:
        if 'model.layers.' in key:
            try:
                idx = int(key.split('model.layers.')[1].split('.')[0])
                n_layers = max(n_layers, idx + 1)
            except: pass
        if 'q_proj.weight' in key and hidden == 0:
            hidden = wd[key].shape[1]

    # Infer architecture
    layer0_q = [k for k in wd if 'layers.0' in k and 'q_proj.weight' in k]
    layer0_k = [k for k in wd if 'layers.0' in k and 'k_proj.weight' in k]
    head_dim = 128
    for k in wd:
        if 'q_norm' in k and 'layers.0' in k:
            head_dim = wd[k].shape[0]
            break

    n_heads = wd[layer0_q[0]].shape[0] // head_dim if layer0_q else 16
    n_kv = wd[layer0_k[0]].shape[0] // head_dim if layer0_k else 8
    intermediate = 0
    for k in wd:
        if 'gate_proj.weight' in k and 'layers.0' in k:
            intermediate = wd[k].shape[0]
            break

    vocab = wd['model.embed_tokens.weight'].shape[0] if 'model.embed_tokens.weight' in wd else 151936

    config = ModelConfig(
        n_layers=n_layers, n_heads=n_heads, n_kv_heads=n_kv,
        hidden_size=hidden, intermediate_size=intermediate,
        vocab_size=vocab, head_dim=head_dim,
    )

    print(f"Detected: {n_layers} layers, {hidden} hidden, {n_heads} heads")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    compressor = GenomeCompressor(model_weights=wd, model_config=config, device=device)

    if args.hybrid:
        result = compressor.compress_progressive(
            small_dim=args.sd, n_heads=min(args.sd // 32, 8),
            n_layers=n_layers, steps_per_layer=args.steps // n_layers,
            batch_size=args.batch_size, lr=args.lr,
            eval_samples=50, verbose=True,
        )
    else:
        cache = compressor.build_cache(n_samples=args.cache_size, batch_size=16, n_layers=n_layers)
        result = compressor.compress_from_cache(
            cache, small_dim=args.sd, n_heads=min(args.sd // 32, 8),
            n_steps=args.steps, batch_size=args.batch_size,
            lr=args.lr, verbose=True,
        )

    output = args.output or f"genome_{os.path.basename(args.model)}_sd{args.sd}.pt"
    result.genome.save_genome(output)

    print(f"\nSaved: {output}")
    print(f"Size: {result.genome_size_mb:.1f} MB ({result.compression_ratio:.0f}x compression)")
    print(f"Quality: Top1={result.top1_accuracy*100:.0f}% Top10={result.top10_overlap*100:.0f}%")


def cmd_run(args):
    """Run inference with a genome model."""
    import torch
    from ultracompress.genome_compressor import GenomeModel

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    genome_data = torch.load(args.genome, weights_only=True)
    cfg = genome_data['config']

    # Need original model's embed/head weights
    if args.weights:
        wd = torch.load(args.weights, weights_only=True)
    else:
        cache_file = 'qwen3_0.6b_cache.pt'
        if os.path.exists(cache_file):
            wd = torch.load(cache_file, weights_only=True)
        else:
            print("Need --weights to specify model weights for embedding/LM head")
            return

    embed = wd['model.embed_tokens.weight'].float().to(device)
    norm_w = wd.get('model.norm.weight', torch.ones(cfg['big_dim'])).float().to(device)
    head_w = wd.get('lm_head.weight', embed).float().to(device)

    genome = GenomeModel(
        vocab_size=embed.shape[0], big_dim=cfg['big_dim'],
        small_dim=cfg['small_dim'], n_heads=cfg['n_heads'],
        n_layers=cfg['n_layers'],
        embed_weight=embed, lm_head_weight=head_w, norm_weight=norm_w,
    ).to(device)
    genome.load_genome(args.genome)
    genome.eval()

    # Tokenize if possible
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B', trust_remote_code=True)
        tokens = torch.tensor([tok.encode(args.prompt)], device=device)
    except:
        tokens = torch.randint(100, 5000, (1, 16), device=device)

    print(f"Genome: {genome.genome_param_count()*2/1e6:.1f} MB")
    print(f"Generating from: {args.prompt}")
    print()

    with torch.no_grad():
        for _ in range(args.max_tokens):
            logits = genome(tokens)
            next_token = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            tokens = torch.cat([tokens, next_token], dim=1)

    try:
        print(tok.decode(tokens[0].tolist()))
    except:
        print(f"Tokens: {tokens[0].tolist()}")


def cmd_info(args):
    """Show info about a genome file."""
    import torch
    data = torch.load(args.genome, weights_only=True)
    cfg = data['config']
    state = data['genome_state']
    params = sum(v.numel() for v in state.values())

    print(f"Genome: {args.genome}")
    print(f"  Layers: {cfg['n_layers']}")
    print(f"  Small dim: {cfg['small_dim']}")
    print(f"  Heads: {cfg['n_heads']}")
    print(f"  Parameters: {params:,}")
    print(f"  Size: {params*2/1e6:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="UltraCompress — Genome Compression Engine")
    sub = parser.add_subparsers(dest='command')

    # Compress
    p = sub.add_parser('compress', help='Compress a model into a genome')
    p.add_argument('--model', required=True, help='HuggingFace model ID or .pt file')
    p.add_argument('--sd', type=int, default=128, help='Genome bottleneck dimension')
    p.add_argument('--steps', type=int, default=30000, help='Training steps')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--cache-size', type=int, default=5000)
    p.add_argument('--hybrid', action='store_true', help='Use progressive+cached hybrid')
    p.add_argument('--output', help='Output genome file path')

    # Run
    p = sub.add_parser('run', help='Run inference with a genome')
    p.add_argument('--genome', required=True, help='Genome .pt file')
    p.add_argument('--weights', help='Original model weights for embed/head')
    p.add_argument('--prompt', default='The meaning of life is')
    p.add_argument('--max-tokens', type=int, default=30)

    # Info
    p = sub.add_parser('info', help='Show genome info')
    p.add_argument('--genome', required=True)

    args = parser.parse_args()
    if args.command == 'compress':
        cmd_compress(args)
    elif args.command == 'run':
        cmd_run(args)
    elif args.command == 'info':
        cmd_info(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
