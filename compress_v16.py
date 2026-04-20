"""
compress_v16.py — Asymmetric Per-Role Codebook Capacity (Claim 13).

Screen (screen_v16.py) established empirically on Qwen3-1.7B body:
  o_proj :  +bits are HIGHLY productive (2048->4096 drops max 15.3%,
             4096+512 drops max 27.8%).
  others :  bits are NOT wasted — cutting ANY role hurts it by 17%+.

So the correct lever is NOT bit-shifting, it is ASYMMETRIC CAPACITY:
give o_proj a larger codebook, hold all other roles at the baseline.
Cost: o_proj is 8.3% of body params; +0.25 bpw on that role only costs
+0.021 bpw globally (0.9% bpw increase) for a predicted 27.8% rel-W
max improvement.

CLAIM 13: Per-role asymmetric codebook capacity. Under a role-banked
universal-codebook family (Claim 10), each role's (K1_r, K2_r) is chosen
independently to minimize per-role reconstruction error at its own
fidelity/bit trade-off, driven by a pre-screen that measures
    rel-W_r(K1, K2)
on that role alone (scales as n_r * K * D, not N * K * D) and picks the
(K1_r, K2_r) on each role's Pareto frontier that matches the global
minimax target. The per-weight bit cost becomes
    bpw_global = sum_r (log2 K1_r + log2 K2_r) / D  *  (n_params_r / N)
which is 2.396 bpw for the v16 allocation below vs 2.375 for uniform.

Default allocation (from screen):
    q_proj, k_proj, v_proj, gate_proj, up_proj, down_proj : K1=2048 K2=256
    o_proj                                                 : K1=4096 K2=512

Composes on top of Claims 7+8+9+10+11+12.
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
from compress_v15 import beam_assign, index_entropy


DEFAULT_ROLE_K: dict[str, tuple[int, int]] = {
    "q_proj":    (2048, 256),
    "k_proj":    (2048, 256),
    "v_proj":    (2048, 256),
    "o_proj":    (4096, 512),   # <-- asymmetric upgrade
    "gate_proj": (2048, 256),
    "up_proj":   (2048, 256),
    "down_proj": (2048, 256),
}


def _chunked_argmin(X: torch.Tensor, C: torch.Tensor, bs: int = 300_000) -> torch.Tensor:
    out = torch.empty(X.shape[0], dtype=torch.long, device=X.device)
    C_nrm = (C * C).sum(-1)
    for s in range(0, X.shape[0], bs):
        e = min(s + bs, X.shape[0])
        d = C_nrm.unsqueeze(0) - 2.0 * (X[s:e] @ C.T) + (X[s:e] * X[s:e]).sum(-1, keepdim=True)
        out[s:e] = d.argmin(-1)
        del d
    return out


def v16_compress(teacher_pt: str, role_K: dict, D: int,
                 iters: int = 8, beam: int = 8, device: str = "cuda:0",
                 pool_sz: int = 200_000, kmeans_iters: int = 6):
    raw = collect_body_linears(teacher_pt)
    W_orig_cpu = {n: v.float() for n, v in raw.items() if v.shape[1] % D == 0}
    del raw

    dims = sorted({W.shape[1] for W in W_orig_cpu.values()})
    rots = {I: build_rotation(I, device, seed=42 + I) for I in dims}

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
        bank_g[role].append(g); bank_rs[role].append(rs_chunk)
        per_tensor_meta.append((name, role, bank_offset[role], g.shape[0], O, I))
        bank_offset[role] += g.shape[0]
        del W, Wrot

    banks: dict[str, dict] = {}
    for role in ROLE_PATTERNS:
        if not bank_g[role]:
            continue
        G = torch.cat(bank_g[role], 0); RS = torch.cat(bank_rs[role], 0)
        banks[role] = {"G": G, "RS": RS, "rs_sq": RS ** 2,
                       "K1": role_K[role][0], "K2": role_K[role][1]}
    del bank_g, bank_rs
    torch.cuda.empty_cache(); gc.collect()

    # init per-role (greedy residual) with role-specific K1, K2
    for role, b in banks.items():
        G, K1, K2 = b["G"], b["K1"], b["K2"]
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
        print(f"[v16] init role={role:10s}  K1={K1:>4d} K2={K2:>3d}  "
              f"n_chunks={G.shape[0]:>10d}  "
              f"bpw_r={(math.log2(K1)+math.log2(K2))/D:.3f}")

    def eval_metric() -> tuple[list[float], dict[str, list[float]], dict[str, int]]:
        rws_all: list[float] = []
        rws_by_role: dict[str, list[float]] = {r: [] for r in banks}
        params_by_role: dict[str, int] = {r: 0 for r in banks}
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
            params_by_role[role] += O * I
            del W, Wq, Wq_rot
        return rws_all, rws_by_role, params_by_role

    rws0, _, params_by_role = eval_metric()
    print(f"\n[v16] init (greedy): mean {sum(rws0)/len(rws0):.4f}  max {max(rws0):.4f}")

    history = [rws0]
    for it in range(iters):
        t = time.time()
        for role, b in banks.items():
            K1, K2 = b["K1"], b["K2"]
            i1, i2, _ = beam_assign(b["G"], b["cb1"], b["cb2"], beam=beam)
            b["idx1"], b["idx2"] = i1, i2
            b["cb1"] = _weighted_cb_update(b["G"], b["idx1"], b["rs_sq"], K1, D)
            R1 = b["G"] - b["cb1"][b["idx1"]]
            b["cb2"] = _weighted_cb_update(R1, b["idx2"], b["rs_sq"], K2, D)
            del R1
        rws, rws_by_role, _ = eval_metric()
        history.append(rws)
        print(f"[v16] iter {it+1}/{iters}: mean {sum(rws)/len(rws):.4f} "
              f"max {max(rws):.4f} | {time.time()-t:.0f}s")

    # Global bpw: sum over roles, weighted by param share
    N_total = sum(params_by_role.values())
    global_bpw = sum(
        (math.log2(b["K1"]) + math.log2(b["K2"])) / D * params_by_role[role] / N_total
        for role, b in banks.items()
    )

    print("\n[v16] final per-role breakdown:")
    print(f"{'role':<12} {'K1':>5} {'K2':>4} {'bpw_r':>6} {'mean':>7} "
          f"{'max':>7} {'H1/K1':>10} {'H2/K2':>10}")
    for role, b in banks.items():
        K1, K2 = b["K1"], b["K2"]
        H1 = index_entropy(b["idx1"], K1)
        H2 = index_entropy(b["idx2"], K2)
        rs = rws_by_role[role]
        print(f"{role:<12} {K1:>5d} {K2:>4d} "
              f"{(math.log2(K1)+math.log2(K2))/D:>6.3f} "
              f"{sum(rs)/len(rs):>7.4f} {max(rs):>7.4f} "
              f"{H1:>5.2f}/{math.log2(K1):<4.0f} {H2:>5.2f}/{math.log2(K2):<4.0f}")
    print(f"\n[v16] global bpw (param-weighted): {global_bpw:.4f}")
    print(f"[v16] vs uniform v15 baseline 2.3750 bpw: "
          f"{(global_bpw - 2.375):+.4f} ({(global_bpw/2.375 - 1)*100:+.2f}%)")

    return {
        "banks": {r: {"cb1": b["cb1"].cpu(), "cb2": b["cb2"].cpu(),
                      "K1": b["K1"], "K2": b["K2"]} for r, b in banks.items()},
        "role_K": role_K, "D": D, "iters": iters, "beam": beam,
        "rel_w_init_mean": sum(rws0) / len(rws0),
        "rel_w_init_max": max(rws0),
        "rel_w_final_mean": sum(rws) / len(rws),
        "rel_w_final_max": max(rws),
        "final_by_role": {r: {"mean": sum(v)/len(v), "max": max(v), "n": len(v)}
                          for r, v in rws_by_role.items()},
        "global_bpw": global_bpw,
        "params_by_role": params_by_role,
        "history": history,
        "rot_sizes": dims,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="qwen3_1.7b_cache.pt")
    p.add_argument("--D", type=int, default=8)
    p.add_argument("--iters", type=int, default=8)
    p.add_argument("--beam", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default="v16_result.pt")
    args = p.parse_args()

    t0 = time.time()
    r = v16_compress(args.teacher, DEFAULT_ROLE_K, args.D,
                     iters=args.iters, beam=args.beam, device=args.device)
    r["wall_sec"] = time.time() - t0
    torch.save(r, args.out)
    print(f"\nSaved {args.out}")
    print(f"  init  mean {r['rel_w_init_mean']:.4f}  max {r['rel_w_init_max']:.4f}")
    print(f"  final mean {r['rel_w_final_mean']:.4f}  max {r['rel_w_final_max']:.4f}")
    print(f"  global bpw {r['global_bpw']:.4f}  (v15 was 2.3750)")
    print(f"  wall  {r['wall_sec']:.0f}s  (beam={args.beam})")


if __name__ == "__main__":
    main()
