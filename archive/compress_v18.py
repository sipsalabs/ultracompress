"""
compress_v18.py — Two-Sided Saliency Conditioning (Claim 15).

Extends Claim 14 (v17) by scaling BOTH input columns AND output rows:

    W_tilde = diag(u) · W · diag(s)
    s[i] = sigma2_in[i]^alpha      (input-column scale, Claim 14)
    u[o] = sigma2_out[o]^beta      (output-row scale, NEW)

Decode:
    W_q = diag(1/u) · decode(W_tilde) · diag(1/s)

Rationale:
- Input scaling (Claim 14) handles the diagonal of the input Hessian
  E[X X^T], which sets the OUTPUT error's sensitivity to column noise.
- Output scaling handles the diagonal of the output covariance
  E[Y Y^T], which sets the DOWNSTREAM loss's sensitivity to row noise
  (since Y is the next block's input after LN + residual).
- Together they capture both sides of the forward-Jacobian diagonal,
  effectively producing an input-AND-output whitening before rotation.
- AWQ (Lin et al. 2023) uses output scaling only; GPTQ uses input
  Hessian only. Neither combines the two with a product codebook.

Storage overhead: fp16 u (O dims) + fp16 s (I dims) per Linear.
~2x of v17's +0.006 bpw = +0.012 bpw.

Composes with all prior claims (rotation, role banks, weighted EM,
beam search, asymmetric capacity).
"""
from __future__ import annotations
import argparse, gc, math, time
import torch

from universal_v9 import kmeans_init
from compress_v14 import (
    ROLE_PATTERNS, _role_of, build_rotation, collect_body_linears,
    _weighted_cb_update,
)
from compress_v15 import beam_assign
from compress_v16 import DEFAULT_ROLE_K, _chunked_argmin


def v18_compress(teacher_pt: str, actio_pt: str, role_K: dict, D: int,
                 alpha: float = 0.25, beta: float = 0.25,
                 iters: int = 6, beam: int = 8, device: str = "cuda:0",
                 pool_sz: int = 200_000, kmeans_iters: int = 6,
                 eps: float = 1e-4):
    raw = collect_body_linears(teacher_pt)
    W_orig_cpu = {n: v.float() for n, v in raw.items() if v.shape[1] % D == 0}
    del raw

    print(f"[v18] loading activation cache {actio_pt}")
    act = torch.load(actio_pt, map_location="cpu", weights_only=False)
    sig_in = act["sigma2_in"]; sig_out = act["sigma2_out"]

    s_col: dict[str, torch.Tensor] = {}
    u_row: dict[str, torch.Tensor] = {}
    miss_i = miss_o = 0
    for name, W in W_orig_cpu.items():
        O, I = W.shape
        if name in sig_in:
            s_col[name] = (sig_in[name].clamp(min=eps)).pow(alpha)
        else:
            s_col[name] = torch.ones(I); miss_i += 1
        if name in sig_out:
            u_row[name] = (sig_out[name].clamp(min=eps)).pow(beta)
        else:
            u_row[name] = torch.ones(O); miss_o += 1
    if miss_i or miss_o:
        print(f"[v18] WARNING: {miss_i} in-miss, {miss_o} out-miss")

    print(f"[v18] alpha={alpha} beta={beta}  "
          f"avg s-ratio {sum((s.max()/s.mean()).item() for s in s_col.values())/len(s_col):.2f}  "
          f"avg u-ratio {sum((u.max()/u.mean()).item() for u in u_row.values())/len(u_row):.2f}")

    dims = sorted({W.shape[1] for W in W_orig_cpu.values()})
    rots = {I: build_rotation(I, device, seed=42 + I) for I in dims}

    per_tensor_meta = []
    bank_g: dict[str, list[torch.Tensor]] = {r: [] for r in ROLE_PATTERNS}
    bank_rs: dict[str, list[torch.Tensor]] = {r: [] for r in ROLE_PATTERNS}
    bank_offset: dict[str, int] = {r: 0 for r in ROLE_PATTERNS}

    for name, W_cpu in W_orig_cpu.items():
        role = _role_of(name)
        W = W_cpu.to(device)
        s = s_col[name].to(device); u = u_row[name].to(device)
        # W_tilde = diag(u) · W · diag(s)
        W_sc = u.unsqueeze(1) * W * s.unsqueeze(0)
        Wrot = W_sc @ rots[W.shape[1]]
        O, I = Wrot.shape
        rs = Wrot.abs().amax(1, keepdim=True).clamp(min=1e-6)
        g = (Wrot / rs).view(O, I // D, D).reshape(-1, D)
        rs_chunk = rs.expand(O, I // D).reshape(-1)
        bank_g[role].append(g); bank_rs[role].append(rs_chunk)
        per_tensor_meta.append((name, role, bank_offset[role], g.shape[0], O, I))
        bank_offset[role] += g.shape[0]
        del W, W_sc, Wrot

    banks: dict[str, dict] = {}
    for role in ROLE_PATTERNS:
        if not bank_g[role]: continue
        G = torch.cat(bank_g[role], 0); RS = torch.cat(bank_rs[role], 0)
        banks[role] = {"G": G, "RS": RS, "rs_sq": RS ** 2,
                       "K1": role_K[role][0], "K2": role_K[role][1]}
    del bank_g, bank_rs
    torch.cuda.empty_cache(); gc.collect()

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
        print(f"[v18] init role={role:10s}  K1={K1:>4d} K2={K2:>3d}  "
              f"n_chunks={G.shape[0]:>10d}")

    def eval_metric():
        """rel-W in original (unscaled) space."""
        rws_all = []; rws_by_role = {r: [] for r in banks}
        params_by_role = {r: 0 for r in banks}
        for name, role, off, n, O, I in per_tensor_meta:
            b = banks[role]
            gh = b["cb1"][b["idx1"][off:off+n]] + b["cb2"][b["idx2"][off:off+n]]
            rs = b["RS"][off:off+n]
            s = s_col[name].to(device); u = u_row[name].to(device)
            Wq_rot_tilde = gh.view(O, I//D, D).reshape(O, I) \
                           * rs.view(O, I//D).repeat_interleave(D, dim=1)
            Wq_tilde = Wq_rot_tilde @ rots[I].T
            # unfold: W_q = (1/u) · W_tilde · (1/s)
            Wq = Wq_tilde / u.unsqueeze(1) / s.unsqueeze(0)
            W = W_orig_cpu[name].to(device)
            r = ((W - Wq).pow(2).mean() / W.pow(2).mean()).item()
            rws_all.append(r); rws_by_role[role].append(r)
            params_by_role[role] += O * I
        return rws_all, rws_by_role, params_by_role

    rws0, _, params_by_role = eval_metric()
    print(f"\n[v18] init (greedy): orig-space rel-W mean "
          f"{sum(rws0)/len(rws0):.4f}  max {max(rws0):.4f}")

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
        print(f"[v18] iter {it+1}/{iters}: mean {sum(rws)/len(rws):.4f} "
              f"max {max(rws):.4f} | {time.time()-t:.0f}s")

    N_total = sum(params_by_role.values())
    global_bpw = sum(
        (math.log2(b["K1"]) + math.log2(b["K2"])) / D * params_by_role[role] / N_total
        for role, b in banks.items()
    )
    n_scale_bits = (sum(s.numel() for s in s_col.values())
                    + sum(u.numel() for u in u_row.values())) * 16
    n_total_params = sum(W_orig_cpu[n].numel() for n in W_orig_cpu)
    overhead_bpw = n_scale_bits / n_total_params
    print(f"\n[v18] overhead {n_scale_bits/8/1e6:.2f} MB (+{overhead_bpw:.4f} bpw)")

    print("\n[v18] final per-role breakdown:")
    for role, b in banks.items():
        rs = rws_by_role[role]
        print(f"  {role:<12} K1={b['K1']:>5d} K2={b['K2']:>4d}  "
              f"mean {sum(rs)/len(rs):.4f}  max {max(rs):.4f}")

    return {
        "banks": {r: {"cb1": b["cb1"].cpu(), "cb2": b["cb2"].cpu(),
                      "K1": b["K1"], "K2": b["K2"]} for r, b in banks.items()},
        "s_col": {n: s.cpu() for n, s in s_col.items()},
        "u_row": {n: u.cpu() for n, u in u_row.items()},
        "role_K": role_K, "D": D, "iters": iters, "beam": beam,
        "alpha": alpha, "beta": beta,
        "rel_w_final_mean": sum(rws)/len(rws),
        "rel_w_final_max": max(rws),
        "final_by_role": {r: {"mean": sum(v)/len(v), "max": max(v), "n": len(v)}
                          for r, v in rws_by_role.items()},
        "global_bpw": global_bpw, "overhead_bpw": overhead_bpw,
        "rot_sizes": dims,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="qwen3_1.7b_cache.pt")
    p.add_argument("--actio", default="v18_activations_io.pt")
    p.add_argument("--alpha", type=float, default=0.25)
    p.add_argument("--beta", type=float, default=0.25)
    p.add_argument("--D", type=int, default=8)
    p.add_argument("--iters", type=int, default=6)
    p.add_argument("--beam", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default="v18_result.pt")
    args = p.parse_args()

    t0 = time.time()
    r = v18_compress(args.teacher, args.actio, DEFAULT_ROLE_K, args.D,
                     alpha=args.alpha, beta=args.beta,
                     iters=args.iters, beam=args.beam, device=args.device)
    r["wall_sec"] = time.time() - t0
    torch.save(r, args.out)
    print(f"\nSaved {args.out}")
    print(f"  alpha={args.alpha} beta={args.beta}  "
          f"total bpw {r['global_bpw']+r['overhead_bpw']:.4f}")
    print(f"  mean {r['rel_w_final_mean']:.4f}  max {r['rel_w_final_max']:.4f}")
    print(f"  wall {r['wall_sec']:.0f}s")


if __name__ == "__main__":
    main()
