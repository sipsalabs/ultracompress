#!/usr/bin/env python3
"""
Full E2E compression: FRR model -> pipeline -> final compressed file.

Takes a trained FRR checkpoint and applies the full compression stack:
  1. Extract FRR block weights
  2. Hadamard rotation (lossless)
  3. SVD manifold projection (lossless w/ residual)
  4. Q2 quantization (lossy)
  5. Entropy coding (lossless)
  6. Package into .ucz file

Usage:
  python compress_e2e.py --frr frr_100k_best.pt --output compressed.ucz
"""
import lib.unbuffered
import argparse
import torch
import os
import sys
import time
import json
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="E2E FRR compression")
    parser.add_argument('--frr', required=True, help='Trained FRR checkpoint (.pt)')
    parser.add_argument('--output', default='compressed.ucz', help='Output file')
    parser.add_argument('--qbits', type=int, default=2, help='Quantization bits (2, 4, 8)')
    parser.add_argument('--base-model', default='Qwen/Qwen3-0.6B', help='Base model for embeddings')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"{'='*60}")
    print(f"  E2E FRR COMPRESSION")
    print(f"  Input: {args.frr}")
    print(f"  Output: {args.output}")
    print(f"  Quantization: Q{args.qbits}")
    print(f"{'='*60}")

    # Load FRR
    from ultracompress.moonshot import FractalModel

    print("\nLoading base model embeddings...")
    wd = torch.load('qwen3_0.6b_cache.pt', weights_only=True)
    embed_w = wd['model.embed_tokens.weight'].float()
    norm_w = wd.get('model.norm.weight', torch.ones(1024)).float()
    lm_head_w = wd.get('lm_head.weight', embed_w).float()

    model = FractalModel(
        hidden_dim=1024, n_heads=16, n_scales=4, iters_per_scale=7,
        vocab_size=151936, ff_mult=1,
        embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w
    )
    model.load_state_dict(torch.load(args.frr, map_location='cpu'))
    print(f"Loaded FRR: {sum(p.numel() for p in model.parameters()):,} params")

    # Extract components
    block_sd = {k: v for k, v in model.block.state_dict().items()}
    modulation = {
        'scale_gamma': model.scale_gamma.data,
        'scale_beta': model.scale_beta.data,
        'iter_scale': model.iter_scale.data,
    }

    block_params = sum(v.numel() for v in block_sd.values())
    mod_params = sum(v.numel() for v in modulation.values())
    print(f"Block: {block_params:,} params")
    print(f"Modulation: {mod_params:,} params")

    # Apply pipeline to block weights
    from ultracompress.ultimate_pipeline import UltimatePipeline, UltimatePipelineConfig

    print(f"\nApplying compression pipeline (Q{args.qbits})...")
    pipe_config = UltimatePipelineConfig(
        quant_bits=args.qbits,
        residual_bits=min(args.qbits, 2),
        rank_fraction=0.8 if args.qbits >= 4 else 0.5,
    )
    pipe = UltimatePipeline(pipe_config)
    compressed_block = pipe.compress(block_sd)
    pipe.report()

    # Package into .ucz
    print(f"\nPackaging into {args.output}...")
    t0 = time.time()

    with zipfile.ZipFile(args.output, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Manifest
        manifest = {
            'format': 'ultracompress_frr',
            'version': '2.0',
            'architecture': {
                'type': 'frr',
                'hidden_dim': 1024,
                'n_heads': 16,
                'n_scales': 4,
                'iters_per_scale': 7,
                'ff_mult': 1,
                'vocab_size': 151936,
            },
            'compression': {
                'frr_compression': f'{block_params}x',
                'quant_bits': args.qbits,
                'pipeline': 'hadamard_svd_quant_entropy',
            },
            'base_model': args.base_model,
            'created': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        zf.writestr('manifest.json', json.dumps(manifest, indent=2))

        # Compressed block (binary blob from pipeline)
        import pickle
        zf.writestr('block.compressed', pickle.dumps(compressed_block))

        # Modulation params (tiny, store as FP16)
        for name, tensor in modulation.items():
            buf = tensor.half().numpy().tobytes()
            zf.writestr(f'modulation/{name}.bin', buf)

    file_size = os.path.getsize(args.output)
    original_size = (block_params + mod_params) * 4  # FP32
    teacher_size = 751_632_384 * 2  # Qwen3-0.6B FP16

    print(f"\n{'='*60}")
    print(f"  COMPRESSION RESULT")
    print(f"  Output: {args.output} ({file_size/1e6:.2f} MB)")
    print(f"  FRR block (FP32): {block_params*4/1e6:.1f} MB")
    print(f"  Compressed block: {file_size/1e6:.2f} MB")
    print(f"  Teacher (FP16): {teacher_size/1e6:.0f} MB")
    print(f"  Total compression: {teacher_size/file_size:.0f}x")
    print(f"  (Note: embeddings shared from base model, not stored)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
