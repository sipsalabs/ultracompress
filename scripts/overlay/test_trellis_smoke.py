"""Trellis quantizer smoke test.

Runs `trellis_quantize_weight` on a small synthetic 2048x2048 weight in three
configurations and prints rel-Frobenius error / per-row cosine for each:

  * trellis(block=0)     — vanilla QTIP path: RHT + per-row std-norm + Viterbi.
                           No GSQ-style per-input-block scaling. With proper
                           std-calibration (matching the codebook output std),
                           even the vanilla path beats scalar baseline on
                           Gaussian inputs.
  * trellis(block=64)    — composed: RHT + per-block(64) absmax scale (in the
                           rotated basis) + per-row std-norm + Viterbi +
                           un-block-scale + un-RHT. The per-block layer adds
                           outlier-robustness for non-Gaussian (real LLM)
                           weights; on iid-Gaussian smoke the gain over
                           vanilla is small because RHT already Gaussianized.
  * scalar(block=64)     — per-block absmax scalar (Cure-A4). Reference baseline.

Memory: capped at a 2048x2048 fp32 tensor (16 MB) to avoid VRAM contention with
the 32B GSQ run already on cuda:0/cuda:1. Defaults to CPU; pass --cuda1 to run
on cuda:1 if you've confirmed it has headroom.

Pass criterion: trellis(block=64) rel-Frob STRICTLY LESS than scalar baseline
rel-Frob. Stretch goal: trellis(block=64) rel-Frob < 0.10.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

import torch

from quantizers.trellis import trellis_quantize_weight


def per_row_cos(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    a_n = A / A.norm(dim=1, keepdim=True).clamp(min=1e-12)
    b_n = B / B.norm(dim=1, keepdim=True).clamp(min=1e-12)
    return (a_n * b_n).sum(dim=1)


def rel_frob(W: torch.Tensor, Wq: torch.Tensor) -> float:
    return float((W - Wq).norm() / W.norm().clamp(min=1e-12))


def baseline_per_block_scalar(W: torch.Tensor, bpw: int, block: int) -> torch.Tensor:
    """Per-block absmax scalar quant — matches block_scalar_quantize_weight in
    scaling_curve_runner.py. Used as the comparison baseline."""
    half = 2 ** bpw // 2
    rows, cols = W.shape
    pad = (block - cols % block) % block
    W_pad = torch.nn.functional.pad(W, (0, pad)) if pad else W
    Wb = W_pad.view(rows, -1, block)
    rm = Wb.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    Q = (Wb / rm * half).round().clamp(-half, half - 1) / half * rm
    return Q.view(rows, -1)[:, :cols].contiguous()


def run_trellis(W: torch.Tensor, bpw: int, group_size: int, block_size: int,
                use_rht: bool, seed: int, label: str) -> tuple[float, float, float]:
    t0 = time.time()
    Wq = trellis_quantize_weight(
        W, bpw=bpw,
        group_size=group_size,
        block_size=block_size,
        use_rht=use_rht,
        seed=seed,
    )
    elapsed = time.time() - t0
    rel = rel_frob(W, Wq)
    cos = per_row_cos(W, Wq)
    print(f'[{label:22s}] rel-Frob={rel:.4f}  '
          f'cos mean={cos.mean().item():.4f}  '
          f'cos min={cos.min().item():.4f}  '
          f'cos p50={cos.median().item():.4f}  '
          f'time={elapsed:.1f}s')
    return rel, float(cos.mean()), elapsed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--rows', type=int, default=2048)
    ap.add_argument('--cols', type=int, default=2048)
    ap.add_argument('--bpw', type=int, default=3)
    ap.add_argument('--group_size', type=int, default=64)
    ap.add_argument('--block_size', type=int, default=64,
                    help='Per-block absmax scaling along input axis (GSQ-style). '
                         '0 = vanilla QTIP path (per-row only).')
    ap.add_argument('--cuda1', action='store_true',
                    help='Use cuda:1 (only if you confirmed it has headroom; '
                         'otherwise CPU is fine for a 2048x2048 prototype).')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no_rht', action='store_true',
                    help='Disable random Hadamard transform (worse error baseline).')
    args = ap.parse_args()

    if args.cuda1:
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            print('[smoke] cuda:1 requested but unavailable; falling back to CPU.')
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:1')
    else:
        device = torch.device('cpu')

    print(f'[smoke] device={device}, shape=({args.rows},{args.cols}), '
          f'bpw={args.bpw}, group_size={args.group_size}, '
          f'block_size={args.block_size}, '
          f'rht={not args.no_rht}')

    torch.manual_seed(args.seed)
    # Gaussian-ish weight with a moderate outlier mix — proxies real LLM weights.
    W = torch.randn(args.rows, args.cols, device=device, dtype=torch.float32)
    W += 0.05 * torch.sign(W) * (torch.rand_like(W) ** 8)

    # ---- 1. Vanilla QTIP trellis (no per-block scaling) ----
    rel_v, _, _ = run_trellis(
        W, bpw=args.bpw, group_size=args.group_size,
        block_size=0, use_rht=not args.no_rht, seed=args.seed,
        label='trellis(block=0)',
    )

    # ---- 2. Per-block + RHT + trellis (the new compose path) ----
    rel_b, _, _ = run_trellis(
        W, bpw=args.bpw, group_size=args.group_size,
        block_size=args.block_size, use_rht=not args.no_rht, seed=args.seed,
        label=f'trellis(block={args.block_size})',
    )

    # ---- 3. Baseline: per-block scalar at same bpw ----
    Wq_s = baseline_per_block_scalar(W, args.bpw, block=args.block_size)
    rel_s = rel_frob(W, Wq_s)
    cos_s = per_row_cos(W, Wq_s)
    print(f'[{"scalar(block=" + str(args.block_size) + ")":22s}] '
          f'rel-Frob={rel_s:.4f}  '
          f'cos mean={cos_s.mean().item():.4f}  '
          f'cos min={cos_s.min().item():.4f}')

    # ---- Summary ratios ----
    print()
    print(f'[ratio  ] trellis(block=0)         / scalar = {rel_v / rel_s:.3f}  '
          f'(vanilla QTIP vs scalar baseline)')
    print(f'[ratio  ] trellis(block={args.block_size:>3d})        '
          f'/ scalar = {rel_b / rel_s:.3f}  '
          f'(composed vs scalar baseline)')
    print(f'[ratio  ] trellis(block={args.block_size:>3d})        '
          f'/ trellis(block=0) = {rel_b / rel_v:.3f}  '
          f'(composition speedup over vanilla)')

    # Pass criterion: trellis(block=block_size) MUST beat scalar baseline.
    # Stretch: rel-Frob < 0.10.
    beats_scalar = rel_b < rel_s
    stretch = rel_b < 0.10
    if beats_scalar and stretch:
        verdict = 'PASS (stretch: rel-Frob < 0.10)'
        rc = 0
    elif beats_scalar:
        verdict = 'PASS (beats scalar baseline; stretch < 0.10 not met)'
        rc = 0
    else:
        verdict = 'FAIL (trellis(block) did NOT beat scalar baseline)'
        rc = 1
    print()
    print(f'[result ] {verdict}')
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
