"""
screen_v16.py — Pre-screen for Claim 13 (marginal-error bit allocation).

Before spending 5+ minutes on a full-body v16 run, answer two empirical
questions on the full 1.409B body but restricted to single roles:

  Q1 (o_proj, the tail):  does raising bits on o_proj actually lower its
     rel-W max? Sweep (K1, K2) in {(2048,256), (4096,256), (4096,512),
     (8192,256)}.
  Q2 (down_proj, the easiest): can we drop bits on down_proj without
     hurting its rel-W? Sweep (K1,K2) in {(2048,256), (1024,256),
     (512,256), (2048,128)}.

If Q1 shows o_proj rel-W max drops ≥5% going 2048→4096, and Q2 shows
down_proj rel-W stays within 2% at lower K, Claim 13 is alive. Otherwise
abort and pick a different lever.

Reuses v15's beam-search + weighted EM + role-bank machinery.
"""
from __future__ import annotations
import argparse
import gc
import math
import time

import torch

from universal_v9 import kmeans_init
from compress_v14 import (
    ROLE_PATTERNS, _role_of, build_rotation, collect_body_linears,
    _weighted_cb_update,
)
from compress_v15 import beam_assign


def _chunked_argmin(X: torch.Tensor, C: torch.Tensor, bs: int = 300_000) -> torch.Tensor:
    out = torch.empty(X.shape[0], dtype=torch.long, device=X.device)
    C_nrm = (C * C).sum(-1)
    for s in range(0, X.shape[0], bs):
        e = min(s + bs, X.shape[0])
        d = C_nrm.unsqueeze(0) - 2.0 * (X[s:e] @ C.T) + (X[s:e] * X[s:e]).sum(-1, keepdim=True)
        out[s:e] = d.argmin(-1)
        del d
    return out


def fit_single_role(W_orig_cpu: dict, role: str, K1: int, K2: int, D: int,
                    iters: int, beam: int, device: str,
                    pool_sz: int = 200_000, kmeans_iters: int = 6):
    """Fit (cb1, cb2) for a single role using v15 protocol; return rel-W stats."""
    dims = sorted({W.shape[1] for n, W in W_orig_cpu.items()
                   if _role_of(n) == role})
    rots = {I: build_rotation(I, device, seed=42 + I) for I in dims}

    # build per-tensor metadata for this role
    meta = []  # (name, offset_in_bank, n_chunks, O, I)
    g_list, rs_list = [], []
    offset = 0
    for name, W_cpu in W_orig_cpu.items():
        if _role_of(name) != role or W_cpu.shape[1] % D != 0:
            continue
        W = W_cpu.to(device)
        Wrot = W @ rots[W.shape[1]]
        O, I = Wrot.shape
        rs = Wrot.abs().amax(1, keepdim=True).clamp(min=1e-6)
        g = (Wrot / rs).view(O, I // D, D).reshape(-1, D)
        rs_ch = rs.expand(O, I // D).reshape(-1)
        g_list.append(g); rs_list.append(rs_ch)
        meta.append((name, offset, g.shape[0], O, I))
        offset += g.shape[0]
        del W, Wrot

    G = torch.cat(g_list, 0); RS = torch.cat(rs_list, 0); rs_sq = RS ** 2
    del g_list, rs_list
    torch.cuda.empty_cache()

    # init (greedy residual)
    pool = min(pool_sz, G.shape[0])
    cb1 = kmeans_init(G[torch.randperm(G.shape[0], device=device)[:pool]],
                      K1, iters=kmeans_iters)
    idx1 = _chunked_argmin(G, cb1)
    R1 = G - cb1[idx1]
    cb2 = kmeans_init(R1[torch.randperm(R1.shape[0], device=device)[:pool]],
                      K2, iters=kmeans_iters)
    idx2 = _chunked_argmin(R1, cb2)
    del R1
    torch.cuda.empty_cache()

    def eval_stats():
        rws = []
        for name, off, n, O, I in meta:
            gh = cb1[idx1[off:off+n]] + cb2[idx2[off:off+n]]
            W = W_orig_cpu[name].to(device)
            Wq_rot = gh.view(O, I // D, D).reshape(O, I) \
                     * RS[off:off+n].view(O, I // D).repeat_interleave(D, dim=1)
            Wq = Wq_rot @ rots[W.shape[1]].T
            rws.append(((W - Wq).pow(2).mean() / W.pow(2).mean()).item())
            del W, Wq, Wq_rot
        return rws

    # EM loop with beam assignment (Claim 11) + weighted update (Claim 9)
    for it in range(iters):
        i1, i2, _ = beam_assign(G, cb1, cb2, beam=beam)
        idx1, idx2 = i1, i2
        cb1 = _weighted_cb_update(G, idx1, rs_sq, K1, D)
        R1 = G - cb1[idx1]
        cb2 = _weighted_cb_update(R1, idx2, rs_sq, K2, D)
        del R1
    rws = eval_stats()
    bpw = (math.log2(K1) + math.log2(K2)) / D
    return {
        "role": role, "K1": K1, "K2": K2, "D": D, "bpw": bpw,
        "mean": sum(rws) / len(rws), "max": max(rws),
        "n_chunks": G.shape[0], "n_tensors": len(meta),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="qwen3_1.7b_cache.pt")
    p.add_argument("--D", type=int, default=8)
    p.add_argument("--iters", type=int, default=6)
    p.add_argument("--beam", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default="v16_screen.pt")
    args = p.parse_args()

    raw = collect_body_linears(args.teacher)
    W_orig_cpu = {n: v.float() for n, v in raw.items() if v.shape[1] % args.D == 0}
    del raw; gc.collect()

    # Q1: o_proj — can we spend more bits productively?
    o_configs = [(2048, 256), (4096, 256), (4096, 512), (8192, 256)]
    # Q2: down_proj — can we save bits cheaply?
    d_configs = [(2048, 256), (1024, 256), (512, 256), (2048, 128)]

    results = {"o_proj": [], "down_proj": []}
    t0 = time.time()
    print(f"\n=== Q1: o_proj bit sweep ===  (baseline 2.375 bpw, v15 max 0.0662)\n")
    for K1, K2 in o_configs:
        t = time.time()
        r = fit_single_role(W_orig_cpu, "o_proj", K1, K2, args.D,
                            iters=args.iters, beam=args.beam, device=args.device)
        r["wall"] = time.time() - t
        results["o_proj"].append(r)
        print(f"  K1={K1:>5d} K2={K2:>4d}  bpw={r['bpw']:.3f}  "
              f"mean={r['mean']:.4f}  max={r['max']:.4f}   ({r['wall']:.0f}s)")
        torch.cuda.empty_cache(); gc.collect()

    print(f"\n=== Q2: down_proj bit sweep ===  (baseline 2.375 bpw, v15 max 0.0598)\n")
    for K1, K2 in d_configs:
        t = time.time()
        r = fit_single_role(W_orig_cpu, "down_proj", K1, K2, args.D,
                            iters=args.iters, beam=args.beam, device=args.device)
        r["wall"] = time.time() - t
        results["down_proj"].append(r)
        print(f"  K1={K1:>5d} K2={K2:>4d}  bpw={r['bpw']:.3f}  "
              f"mean={r['mean']:.4f}  max={r['max']:.4f}   ({r['wall']:.0f}s)")
        torch.cuda.empty_cache(); gc.collect()

    # Decision logic
    o_base = results["o_proj"][0]
    d_base = results["down_proj"][0]
    print("\n=== Decision ===")
    for r in results["o_proj"][1:]:
        drop = (o_base["max"] - r["max"]) / o_base["max"] * 100
        verdict = "ALIVE" if drop >= 5.0 else "dead"
        print(f"  o_proj  {o_base['K1']}->{r['K1']} (K2 {o_base['K2']}->{r['K2']}):  "
              f"max {o_base['max']:.4f}->{r['max']:.4f}  drop {drop:+.1f}%  [{verdict}]")
    for r in results["down_proj"][1:]:
        rise = (r["max"] - d_base["max"]) / d_base["max"] * 100
        verdict = "ALIVE" if rise <= 2.0 else "dead"
        print(f"  down_proj {d_base['K1']}->{r['K1']} (K2 {d_base['K2']}->{r['K2']}):  "
              f"max {d_base['max']:.4f}->{r['max']:.4f}  rise {rise:+.1f}%  [{verdict}]")

    print(f"\nTotal wall: {time.time()-t0:.0f}s")
    torch.save(results, args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
