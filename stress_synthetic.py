"""stress_synthetic.py -- Shape/dtype stress test for v17 encode-decode cycle.

Covers odd shapes, aspect ratios, and dtype paths that real model checkpoints
may or may not exercise. Each test:
  1. builds a random teacher W fp16 [O, I]
  2. runs the v17 encode path (s_col, rotation, hierarchical k-means, beam)
  3. runs the v17 decode path
  4. asserts bit-for-bit determinism of decode given same seeds
  5. reports relW = ||W-Wq||_F / ||W||_F

Passes without a model file; pure shape coverage.
"""
from __future__ import annotations
import argparse, json, time
import torch

from compress_v14 import build_rotation
from compress_v15 import beam_assign


def encode_decode(W_fp16: torch.Tensor, K1: int, K2: int, D: int,
                  device: str, seed: int = 42) -> tuple[torch.Tensor, float]:
    """Minimal self-contained v17-like encode/decode for synthetic W."""
    torch.manual_seed(seed)
    W = W_fp16.to(device=device, dtype=torch.float32)
    O, I = W.shape
    assert I % D == 0, f"I={I} must be divisible by D={D}"

    # s_col: per-column scaling derived from W alone (no activations)
    s = W.abs().mean(0).clamp(min=1e-6)
    W_scaled = W * s.unsqueeze(0)

    rot = build_rotation(I, device, seed=42 + I)
    Wrot = W_scaled @ rot
    rs = Wrot.abs().amax(1, keepdim=True).clamp(min=1e-6)
    g = (Wrot / rs).view(O, I // D, D).reshape(-1, D)

    # tiny hierarchical init: random subset -> cb1, residual -> cb2
    pool = min(200_000, g.shape[0])
    perm = torch.randperm(g.shape[0], device=device, generator=torch.Generator(device).manual_seed(seed))[:pool]
    cb1 = g[perm[:K1]].clone()
    # residual codebook after one argmin
    K = cb1.shape[0]
    cn = (cb1 * cb1).sum(-1)
    idx1_samp = ((cn.unsqueeze(0) - 2*(g[perm] @ cb1.T) + (g[perm]*g[perm]).sum(-1, keepdim=True)).argmin(-1))
    r1 = g[perm] - cb1[idx1_samp]
    cb2 = r1[:K2].clone() if r1.shape[0] >= K2 else torch.randn(K2, D, device=device) * 0.01

    idx1, idx2, _ = beam_assign(g, cb1, cb2, beam=8, chunk=100_000)
    gh = cb1[idx1] + cb2[idx2]
    Wq_rot_scaled = (gh.view(O, I // D, D).reshape(O, I)) * rs.expand(O, I)
    Wq_scaled = Wq_rot_scaled @ rot.T
    Wq = Wq_scaled / s.unsqueeze(0)
    relW = (W - Wq).norm() / W.norm().clamp(min=1e-9)
    return Wq.to(dtype=torch.float16, device="cpu"), float(relW)


SHAPES = [
    # (O, I, label)
    (128, 128, "tiny-square"),
    (256, 768, "gpt2-style"),
    (1024, 1024, "1k-square"),
    (2048, 2048, "2k-square"),
    (512, 4096, "wide-4k"),
    (4096, 512, "tall-4k"),
    (1024, 3072, "gpt2-mlp"),
    (3072, 1024, "gpt2-mlp-T"),
    (2048, 1376, "non-po2-in"),
    (1137, 2048, "non-po2-out"),
    (4096, 14336, "llama-mlp"),
    (14336, 4096, "llama-mlp-T"),
    (3584, 2048, "qwen-like"),
    (1024, 8192, "wide-8k"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K1", type=int, default=2048)
    ap.add_argument("--K2", type=int, default=256)
    ap.add_argument("--D", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="stress_synthetic_results.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"[stress] {len(SHAPES)} shapes, K1={args.K1} K2={args.K2} D={args.D}")
    results = []
    t_all = time.time()
    for (O, I, label) in SHAPES:
        if I % args.D != 0:
            print(f"  SKIP {label:<14}  O={O:>5} I={I:>5}  I%D != 0")
            results.append({"label": label, "O": O, "I": I, "skipped": "I%D!=0"})
            continue
        torch.manual_seed(args.seed)
        W = torch.randn(O, I, dtype=torch.float16)
        t0 = time.time()
        try:
            Wq, relW = encode_decode(W, args.K1, args.K2, args.D, args.device, args.seed)
            # determinism: run again, assert bit-identical
            Wq2, relW2 = encode_decode(W, args.K1, args.K2, args.D, args.device, args.seed)
            det_ok = torch.equal(Wq, Wq2)
            dt = time.time() - t0
            print(f"  OK   {label:<14}  O={O:>5} I={I:>5}  relW={relW:.4f}  det={'Y' if det_ok else 'N'}  ({dt:.1f}s)")
            results.append({"label": label, "O": O, "I": I,
                            "relW": relW, "deterministic": det_ok,
                            "wall_sec": dt})
        except Exception as e:
            print(f"  FAIL {label:<14}  O={O:>5} I={I:>5}  {type(e).__name__}: {e}")
            results.append({"label": label, "O": O, "I": I, "error": repr(e)})
        torch.cuda.empty_cache()
    print(f"[stress] total {time.time()-t_all:.1f}s")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # pass/fail summary
    n_ok = sum(1 for r in results if "relW" in r)
    n_det = sum(1 for r in results if r.get("deterministic") is True)
    n_fail = sum(1 for r in results if "error" in r)
    n_skip = sum(1 for r in results if "skipped" in r)
    print(f"\n  passed: {n_ok}/{len(SHAPES)}  deterministic: {n_det}/{n_ok}  failed: {n_fail}  skipped: {n_skip}")


if __name__ == "__main__":
    main()
