"""
compress_v15.py — Beam-Search Joint Residual PQ + Entropy-Coded Indices.

Two independent, composable novelties on top of v14 (Claims 7+8+9+10):

CLAIM 11 (Beam-search joint residual assignment, "BSJR"):
  Standard residual PQ greedily picks idx1 = argmin_k ||g - cb1[k]||^2, then
  idx2 = argmin_k ||(g - cb1[idx1]) - cb2[k]||^2. The joint minimum of
  ||g - cb1[i1] - cb2[i2]||^2 over (i1, i2) in K1 x K2 is NOT in general
  achieved by the greedy path — the individually best i1 can leave a
  residual that is hard for cb2 to represent. We keep the top-B candidates
  for idx1 (B ~ 8) and for each compute the best idx2, then select the
  joint pair with minimum reconstruction error.
     - Zero bit-cost change (indices still 19 bits/chunk at K1=2048, K2=256).
     - Strict (weak) MSE improvement for every chunk.
     - Runs inside the weighted-EM loop (Claim 9), so codebooks adapt to
       the beam-search assignment statistics — a different fixed point.

CLAIM 12 (Entropy-coded index stream):
  After training, the empirical distribution over cb1 atoms is non-uniform
  (k-means always has "popular" and "rare" atoms, amplified by the
  row-scale weighting of Claim 9). True bits/weight is
      (H(idx1) + H(idx2)) / D
  where H is the Shannon entropy of the empirical index distribution. A
  Huffman / ANS coder on the idx stream realizes this bound within <1%.
  The raw-log2-K account is a strict overestimate.
     - Zero fidelity cost (same codebooks, same indices, same decode).
     - Typical saving: 5-15% bits/weight, model- and role-dependent.
     - We report per-role entropies, average, and effective bits/weight.

Empirical stacking target (Qwen3-1.7B body, K1=2048 K2=256 D=8):
     v14: rel-W mean 0.0679  max 0.0759  @ 2.375 raw bpw
     v15: rel-W mean ?       max ?       @ ?     entropy bpw
Expected: 2-5% fidelity gain + 5-15% bpw reduction = free double win.

Usage:
  python compress_v15.py --teacher qwen3_1.7b_cache.pt --K1 2048 --K2 256 --D 8 \
                         --iters 8 --beam 8 --out v15_result.pt
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


# ---------------------------------------------------------------------------
# Claim 11: beam-search joint residual assignment
# ---------------------------------------------------------------------------
def beam_assign(G: torch.Tensor, cb1: torch.Tensor, cb2: torch.Tensor,
                beam: int = 8, chunk: int = 200_000
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """For each row g in G, find (i1, i2) minimizing ||g - cb1[i1] - cb2[i2]||^2
    by keeping the top-`beam` candidates for i1 and picking the best joint pair.

    Returns (idx1, idx2, total_sq_err_per_chunk).
    """
    N = G.shape[0]
    device = G.device
    idx1_out = torch.empty(N, dtype=torch.long, device=device)
    idx2_out = torch.empty(N, dtype=torch.long, device=device)
    err_out = torch.empty(N, dtype=torch.float32, device=device)

    # cb1 norms used for fast argmin via -2*g.cb1 + ||cb1||^2
    cb1_nrm = (cb1 * cb1).sum(-1)  # [K1]
    cb2_nrm = (cb2 * cb2).sum(-1)  # [K2]

    # Adaptive chunk: dominant buffer is d1 = [chunk, K1] fp32.
    # Keep it <= ~1.5 GB so ultra-tier K1 up to 16384 fits on 32GB cards.
    K1 = cb1.shape[0]
    max_chunk = max(16_000, int(400_000_000 // max(K1, 1)))
    chunk = min(chunk, max_chunk)

    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        Gc = G[s:e]                              # [n, D]
        # distances to cb1: d1[i,k] = ||Gc[i] - cb1[k]||^2
        d1 = cb1_nrm.unsqueeze(0) - 2.0 * (Gc @ cb1.T) + (Gc * Gc).sum(-1, keepdim=True)
        # top-B cb1 candidates (smallest d1)
        topB_vals, topB_idx = torch.topk(d1, beam, dim=1, largest=False)  # [n, B]
        del d1

        best_err = torch.full((e - s,), float("inf"), device=device)
        best_i1 = torch.empty(e - s, dtype=torch.long, device=device)
        best_i2 = torch.empty(e - s, dtype=torch.long, device=device)
        for b in range(beam):
            i1_b = topB_idx[:, b]                # [n]
            r = Gc - cb1[i1_b]                   # [n, D]
            d2 = cb2_nrm.unsqueeze(0) - 2.0 * (r @ cb2.T) + (r * r).sum(-1, keepdim=True)
            v2, i2_b = d2.min(-1)                # [n]
            improve = v2 < best_err
            best_err = torch.where(improve, v2, best_err)
            best_i1 = torch.where(improve, i1_b, best_i1)
            best_i2 = torch.where(improve, i2_b, best_i2)
            del r, d2, v2, i2_b
        idx1_out[s:e] = best_i1
        idx2_out[s:e] = best_i2
        err_out[s:e] = best_err
    return idx1_out, idx2_out, err_out


# ---------------------------------------------------------------------------
# Claim 12: Shannon-entropy index cost
# ---------------------------------------------------------------------------
def index_entropy(idx: torch.Tensor, K: int) -> float:
    """Shannon entropy (base 2) of an index stream over alphabet of size K."""
    cnt = torch.bincount(idx, minlength=K).float()
    p = cnt / cnt.sum().clamp(min=1.0)
    p = p[p > 0]
    return float(-(p * p.log2()).sum().item())


# ---------------------------------------------------------------------------
# v15 main compression
# ---------------------------------------------------------------------------
def v15_compress(teacher_pt: str, K1: int, K2: int, D: int,
                 iters: int = 8, beam: int = 8, device: str = "cuda:0",
                 pool_sz: int = 200_000, kmeans_iters: int = 6):
    raw = collect_body_linears(teacher_pt)
    W_orig_cpu = {n: v.float() for n, v in raw.items() if v.shape[1] % D == 0}
    del raw

    dims = sorted({W.shape[1] for W in W_orig_cpu.values()})
    rots = {I: build_rotation(I, device, seed=42 + I) for I in dims}

    # --- per-role chunk pools ---
    per_tensor_meta: list[tuple[str, str, int, int, int, int]] = []
    bank_g: dict[str, list[torch.Tensor]] = {r: [] for r in ROLE_PATTERNS}
    bank_rs: dict[str, list[torch.Tensor]] = {r: [] for r in ROLE_PATTERNS}
    bank_offset: dict[str, int] = {r: 0 for r in ROLE_PATTERNS}

    for name, W_cpu in W_orig_cpu.items():
        role = _role_of(name)
        W = W_cpu.to(device)
        Wrot = W @ rots[W.shape[1]]
        O, I = Wrot.shape
        rs = Wrot.abs().amax(1, keepdim=True).clamp(min=1e-6)
        g = (Wrot / rs).view(O, I // D, D).reshape(-1, D)
        rs_chunk = rs.expand(O, I // D).reshape(-1)
        bank_g[role].append(g)
        bank_rs[role].append(rs_chunk)
        per_tensor_meta.append((name, role, bank_offset[role], g.shape[0], O, I))
        bank_offset[role] += g.shape[0]
        del W, Wrot

    banks: dict[str, dict] = {}
    for role in ROLE_PATTERNS:
        if not bank_g[role]:
            continue
        G = torch.cat(bank_g[role], 0)
        RS = torch.cat(bank_rs[role], 0)
        banks[role] = {"G": G, "RS": RS, "rs_sq": RS ** 2}
    del bank_g, bank_rs
    torch.cuda.empty_cache(); gc.collect()

    def _chunked_argmin(X: torch.Tensor, C: torch.Tensor, bs: int = 300_000) -> torch.Tensor:
        out = torch.empty(X.shape[0], dtype=torch.long, device=X.device)
        C_nrm = (C * C).sum(-1)
        for s in range(0, X.shape[0], bs):
            e = min(s + bs, X.shape[0])
            d = C_nrm.unsqueeze(0) - 2.0 * (X[s:e] @ C.T) + (X[s:e] * X[s:e]).sum(-1, keepdim=True)
            out[s:e] = d.argmin(-1)
            del d
        return out

    # --- per-role init (greedy residual, same as v14; chunked to avoid OOM) ---
    for role, b in banks.items():
        G = b["G"]
        pool = min(pool_sz, G.shape[0])
        cb1 = kmeans_init(G[torch.randperm(G.shape[0], device=device)[:pool]],
                          K1, iters=kmeans_iters)
        idx1 = _chunked_argmin(G, cb1)
        R1 = G - cb1[idx1]
        cb2 = kmeans_init(R1[torch.randperm(R1.shape[0], device=device)[:pool]],
                          K2, iters=kmeans_iters)
        idx2 = _chunked_argmin(R1, cb2)
        del R1
        b["cb1"], b["cb2"], b["idx1"], b["idx2"] = cb1, cb2, idx1, idx2
        torch.cuda.empty_cache()
        print(f"[v15] init role={role:10s}  n_chunks={G.shape[0]:>10d}")

    def eval_metric() -> tuple[list[float], dict[str, list[float]]]:
        rws_all: list[float] = []
        rws_by_role: dict[str, list[float]] = {r: [] for r in banks}
        for name, role, off, n, O, I in per_tensor_meta:
            b = banks[role]
            gh = b["cb1"][b["idx1"][off:off + n]] + b["cb2"][b["idx2"][off:off + n]]
            rs = b["RS"][off:off + n]
            W = W_orig_cpu[name].to(device)
            Wq_rot = (gh.view(O, I // D, D).reshape(O, I)) \
                     * rs.view(O, I // D).repeat_interleave(D, dim=1)
            Wq = Wq_rot @ rots[W.shape[1]].T
            r = ((W - Wq).pow(2).mean() / W.pow(2).mean()).item()
            rws_all.append(r); rws_by_role[role].append(r)
            del W, Wq, Wq_rot
        return rws_all, rws_by_role

    rws0, _ = eval_metric()
    print(f"\n[v15] init (greedy): mean {sum(rws0)/len(rws0):.4f}  max {max(rws0):.4f}")

    # --- EM loop with beam-search assignment (Claim 11) + weighted update (Claim 9) ---
    history = [rws0]
    for it in range(iters):
        t = time.time()
        for role, b in banks.items():
            # BSJR joint assignment
            i1, i2, _ = beam_assign(b["G"], b["cb1"], b["cb2"], beam=beam)
            b["idx1"], b["idx2"] = i1, i2
            # Claim 9 weighted update: first cb1 on G, then cb2 on residual
            b["cb1"] = _weighted_cb_update(b["G"], b["idx1"], b["rs_sq"], K1, D)
            R1 = b["G"] - b["cb1"][b["idx1"]]
            b["cb2"] = _weighted_cb_update(R1, b["idx2"], b["rs_sq"], K2, D)
            del R1
        rws, rws_by_role = eval_metric()
        history.append(rws)
        print(f"[v15] iter {it+1}/{iters}: mean {sum(rws)/len(rws):.4f} "
              f"max {max(rws):.4f} | {time.time()-t:.0f}s  (beam={beam})")

    # --- Claim 12: entropy accounting ---
    print("\n[v15] final per-role breakdown + entropy:")
    log2K1 = math.log2(K1); log2K2 = math.log2(K2)
    raw_bpw = (log2K1 + log2K2) / D
    ent_by_role: dict[str, dict] = {}
    total_weighted_H1 = 0.0; total_weighted_H2 = 0.0; total_n = 0
    for role, b in banks.items():
        H1 = index_entropy(b["idx1"], K1)
        H2 = index_entropy(b["idx2"], K2)
        n = b["G"].shape[0]
        ent_by_role[role] = {"H1": H1, "H2": H2, "n_chunks": n,
                             "entropy_bpw": (H1 + H2) / D}
        total_weighted_H1 += H1 * n
        total_weighted_H2 += H2 * n
        total_n += n
        rs = rws_by_role[role]
        print(f"       {role:10s} mean {sum(rs)/len(rs):.4f}  max {max(rs):.4f}  "
              f"H1 {H1:5.2f}/{log2K1:.0f}  H2 {H2:5.2f}/{log2K2:.0f}  "
              f"ebpw {(H1+H2)/D:.3f}")
    avg_H1 = total_weighted_H1 / total_n
    avg_H2 = total_weighted_H2 / total_n
    entropy_bpw = (avg_H1 + avg_H2) / D
    savings = (raw_bpw - entropy_bpw) / raw_bpw * 100
    print(f"\n[v15] raw      bpw: {raw_bpw:.4f}")
    print(f"[v15] entropy  bpw: {entropy_bpw:.4f}   "
          f"(avg H1 {avg_H1:.3f}, H2 {avg_H2:.3f})")
    print(f"[v15] bpw savings : {savings:.1f}% "
          f"(Claim 12, zero fidelity cost)")

    return {
        "banks": {r: {"cb1": b["cb1"].cpu(), "cb2": b["cb2"].cpu()} for r, b in banks.items()},
        "idx1_by_role": {r: b["idx1"].cpu() for r, b in banks.items()},
        "idx2_by_role": {r: b["idx2"].cpu() for r, b in banks.items()},
        "K1": K1, "K2": K2, "D": D, "iters": iters, "beam": beam,
        "rel_w_init_mean": sum(rws0) / len(rws0),
        "rel_w_init_max": max(rws0),
        "rel_w_final_mean": sum(rws) / len(rws),
        "rel_w_final_max": max(rws),
        "final_by_role": {r: {"mean": sum(v)/len(v), "max": max(v), "n": len(v)}
                          for r, v in rws_by_role.items()},
        "history": history,
        "raw_bpw": raw_bpw,
        "entropy_bpw": entropy_bpw,
        "bpw_savings_pct": savings,
        "entropy_by_role": ent_by_role,
        "rot_sizes": dims,
        "roles": list(banks.keys()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="qwen3_1.7b_cache.pt")
    p.add_argument("--K1", type=int, default=2048)
    p.add_argument("--K2", type=int, default=256)
    p.add_argument("--D", type=int, default=8)
    p.add_argument("--iters", type=int, default=8)
    p.add_argument("--beam", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default="v15_result.pt")
    args = p.parse_args()

    t0 = time.time()
    r = v15_compress(args.teacher, args.K1, args.K2, args.D,
                     iters=args.iters, beam=args.beam, device=args.device)
    r["wall_sec"] = time.time() - t0
    # don't save the massive idx tensors by default — blow up the .pt file
    # Keep codebooks + stats; indices can be re-derived from codebooks + W.
    idx_bytes = sum(t.numel() * 8 for t in r["idx1_by_role"].values())  # long is 8B
    idx_bytes += sum(t.numel() * 8 for t in r["idx2_by_role"].values())
    print(f"[v15] (indices would add {idx_bytes/1e6:.0f} MB to artifact; dropping)")
    r_save = {k: v for k, v in r.items() if k not in ("idx1_by_role", "idx2_by_role")}
    torch.save(r_save, args.out)
    print(f"\nSaved {args.out}")
    print(f"  init  mean {r['rel_w_init_mean']:.4f}  max {r['rel_w_init_max']:.4f}")
    print(f"  final mean {r['rel_w_final_mean']:.4f}  max {r['rel_w_final_max']:.4f}")
    gain_mean = (r["rel_w_init_mean"] - r["rel_w_final_mean"]) / r["rel_w_init_mean"] * 100
    gain_max = (r["rel_w_init_max"] - r["rel_w_final_max"]) / r["rel_w_init_max"] * 100
    print(f"  gain  mean {gain_mean:.1f}%   max {gain_max:.1f}%")
    print(f"  raw bpw {r['raw_bpw']:.3f}  entropy bpw {r['entropy_bpw']:.3f}  "
          f"savings {r['bpw_savings_pct']:.1f}%")
    print(f"  wall  {r['wall_sec']:.0f}s  (beam={args.beam})")


if __name__ == "__main__":
    main()
