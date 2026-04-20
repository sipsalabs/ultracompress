"""
compress_v17.py — Activation-Aware Input-Column Rescaling (Claim 14).

DISCOVERY that triggered this claim (see eval_v16_ppl.py):
  v16 achieves rel-W mean 0.0577 / max 0.0600 at 2.396 bpw, but its
  downstream PPL on WikiText-103 is 2099× baseline. The reconstruction
  loss has been optimizing the WRONG objective: L2 of (W - W_q), which
  treats every input dim of W equally. Measured INPUT activation
  variances (cache_activations.py, 32 WikiText windows):
     q/k/v inputs:  max/mean ratio ≈ 120×
     mlp inputs  :  max/mean ratio ≈ 290×
  Quantization noise in high-sigma columns therefore causes much larger
  output-space error than noise in low-sigma columns, but rel-W sees
  them as identical.

CLAIM 14: Activation-aware input-column rescaling for universal PQ.
Given per-input-dim activation variance sigma2[i] measured on calibration
data, fold s[i] = sigma2[i] ** alpha into W as
    W' = W * s[None, :]
Apply the full v16 stack to W'. At decode, W_q = W_q' / s[None, :] (or
equivalently, inference pre-scales X by 1/s before the Linear). Choose
alpha in [0, 0.5]; alpha=0 reduces to v16, alpha=0.5 is AWQ with
sigma-weighted scaling, alpha=0.25 is a damped variant that we find is
near-optimal for universal-PQ reconstruction.

Storage overhead: per Linear, I fp16 scales = 196 * 3072 * 2B ≈ 1.2 MB
on a 420 MB body = 0.3 % overhead.

Key differences from AWQ:
  - AWQ scales per-OUTPUT-row; we scale per-INPUT-column (columns carry
    activation variance, rows do not).
  - AWQ applies uniform-integer quantization; we apply product-codebook
    residual PQ with role banks, beam search, and weighted EM.
  - The scaling composes cleanly with Claim 8 rotation because rotation
    acts on the *same* input axis: R @ (W * s_col)^T = (R * s_col[None, :]) @ W^T
    fold... no, we apply rotation AFTER the scaling to W. The rotation
    whitens residual off-diagonal activation covariance; the scaling
    handles the diagonal component. Both compose without conflict.

Usage:
  python compress_v17.py --v17act v17_activations.pt --alpha 0.25 \
                         --teacher qwen3_1.7b_cache.pt --iters 6 --beam 8 \
                         --out v17_result.pt
"""
from __future__ import annotations
import argparse, gc, math, time

import torch

from universal_v9 import kmeans_init
from compress_v14 import (
    ROLE_PATTERNS, _role_of, build_rotation, collect_body_linears,
    _weighted_cb_update,
)
from compress_v15 import beam_assign, index_entropy
from compress_v16 import DEFAULT_ROLE_K, _chunked_argmin


def _key_to_cache(name: str) -> str:
    """Map compress_v14.collect_body_linears key ('layers.i.xxx') to HF key
    ('model.layers.i.xxx.weight')."""
    # teacher cache keys already have 'model.' prefix and '.weight' suffix;
    # collect_body_linears returns them as-is.
    return name if name.startswith("model.") else f"model.{name}"


def v17_compress(teacher_pt: str, act_pt: str, role_K: dict, D: int,
                 alpha: float = 0.25,
                 iters: int = 8, beam: int = 8, device: str = "cuda:0",
                 pool_sz: int = 200_000, kmeans_iters: int = 6,
                 eps: float = 1e-4):
    raw = collect_body_linears(teacher_pt)
    W_orig_cpu = {n: v.float() for n, v in raw.items() if v.shape[1] % D == 0}
    del raw

    print(f"[v17] loading activation cache {act_pt}")
    act = torch.load(act_pt, map_location="cpu", weights_only=False)
    # Build per-tensor s_col
    s_col: dict[str, torch.Tensor] = {}
    missing = 0
    for name in W_orig_cpu:
        k = name  # compress_v14 keys already 'model.layers...weight'
        if k in act:
            s = (act[k].clamp(min=eps)).pow(alpha)
            s_col[name] = s
        else:
            missing += 1
            s_col[name] = torch.ones(W_orig_cpu[name].shape[1])
    if missing:
        print(f"[v17] WARNING: {missing} tensors missing activation data; "
              f"using s=1 (no rescaling)")
    print(f"[v17] alpha={alpha}  "
          f"s ratio max/mean avg over tensors: "
          f"{sum((s.max()/s.mean()).item() for s in s_col.values())/len(s_col):.2f}")

    dims = sorted({W.shape[1] for W in W_orig_cpu.values()})
    rots = {I: build_rotation(I, device, seed=42 + I) for I in dims}

    per_tensor_meta = []
    bank_g: dict[str, list[torch.Tensor]] = {r: [] for r in ROLE_PATTERNS}
    bank_rs: dict[str, list[torch.Tensor]] = {r: [] for r in ROLE_PATTERNS}
    bank_offset: dict[str, int] = {r: 0 for r in ROLE_PATTERNS}

    for name, W_cpu in W_orig_cpu.items():
        role = _role_of(name)
        W = W_cpu.to(device)
        s = s_col[name].to(device)
        W_scaled = W * s.unsqueeze(0)              # [O, I], scaled
        Wrot = W_scaled @ rots[W.shape[1]]
        O, I = Wrot.shape
        rs = Wrot.abs().amax(1, keepdim=True).clamp(min=1e-6)
        g = (Wrot / rs).view(O, I // D, D).reshape(-1, D)
        rs_chunk = rs.expand(O, I // D).reshape(-1)
        bank_g[role].append(g); bank_rs[role].append(rs_chunk)
        per_tensor_meta.append((name, role, bank_offset[role], g.shape[0], O, I))
        bank_offset[role] += g.shape[0]
        del W, W_scaled, Wrot

    banks: dict[str, dict] = {}
    for role in ROLE_PATTERNS:
        if not bank_g[role]:
            continue
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
        print(f"[v17] init role={role:10s}  K1={K1:>4d} K2={K2:>3d}  "
              f"n_chunks={G.shape[0]:>10d}")

    def eval_metric():
        """rel-W is computed on the ORIGINAL (unscaled) W for apples-to-apples
        comparison with v16. The substitution path does the same unscaling."""
        rws_all = []
        rws_by_role = {r: [] for r in banks}
        params_by_role = {r: 0 for r in banks}
        for name, role, off, n, O, I in per_tensor_meta:
            b = banks[role]
            gh = b["cb1"][b["idx1"][off:off + n]] + b["cb2"][b["idx2"][off:off + n]]
            rs = b["RS"][off:off + n]
            s = s_col[name].to(device)
            Wq_rot_scaled = (gh.view(O, I // D, D).reshape(O, I)) \
                            * rs.view(O, I // D).repeat_interleave(D, dim=1)
            Wq_scaled = Wq_rot_scaled @ rots[I].T
            Wq = Wq_scaled / s.unsqueeze(0)
            W = W_orig_cpu[name].to(device)
            r = ((W - Wq).pow(2).mean() / W.pow(2).mean()).item()
            rws_all.append(r); rws_by_role[role].append(r)
            params_by_role[role] += O * I
            del W, Wq, Wq_scaled, Wq_rot_scaled
        return rws_all, rws_by_role, params_by_role

    rws0, _, params_by_role = eval_metric()
    print(f"\n[v17] init (greedy, on W*s): "
          f"orig-space rel-W mean {sum(rws0)/len(rws0):.4f}  max {max(rws0):.4f}")

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
        print(f"[v17] iter {it+1}/{iters}: mean {sum(rws)/len(rws):.4f} "
              f"max {max(rws):.4f} | {time.time()-t:.0f}s")

    N_total = sum(params_by_role.values())
    global_bpw = sum(
        (math.log2(b["K1"]) + math.log2(b["K2"])) / D * params_by_role[role] / N_total
        for role, b in banks.items()
    )
    # add fp16 per-input-dim scale overhead: N_linears * avg_I * 16 bits / total_params
    n_scale_bits = sum(s.numel() for s in s_col.values()) * 16
    n_total_params = sum(W_orig_cpu[n].numel() for n in W_orig_cpu)
    overhead_bpw = n_scale_bits / n_total_params
    print(f"\n[v17] s_col overhead: {n_scale_bits/8/1e6:.2f} MB  "
          f"(+{overhead_bpw:.4f} bpw)")

    print("\n[v17] final per-role breakdown (orig-space rel-W):")
    for role, b in banks.items():
        K1, K2 = b["K1"], b["K2"]
        rs = rws_by_role[role]
        print(f"  {role:<12} K1={K1:>5d} K2={K2:>4d}  "
              f"mean {sum(rs)/len(rs):.4f}  max {max(rs):.4f}")

    return {
        "banks": {r: {"cb1": b["cb1"].cpu(), "cb2": b["cb2"].cpu(),
                      "K1": b["K1"], "K2": b["K2"]} for r, b in banks.items()},
        "s_col": {n: s.cpu() for n, s in s_col.items()},
        "role_K": role_K, "D": D, "iters": iters, "beam": beam, "alpha": alpha,
        "rel_w_init_mean": sum(rws0) / len(rws0),
        "rel_w_init_max": max(rws0),
        "rel_w_final_mean": sum(rws) / len(rws),
        "rel_w_final_max": max(rws),
        "final_by_role": {r: {"mean": sum(v)/len(v), "max": max(v), "n": len(v)}
                          for r, v in rws_by_role.items()},
        "global_bpw": global_bpw, "overhead_bpw": overhead_bpw,
        "params_by_role": params_by_role,
        "history": history, "rot_sizes": dims,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="qwen3_1.7b_cache.pt")
    p.add_argument("--v17act", default="v17_activations.pt")
    p.add_argument("--alpha", type=float, default=0.25)
    p.add_argument("--D", type=int, default=8)
    p.add_argument("--iters", type=int, default=6)
    p.add_argument("--beam", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default="v17_result.pt")
    args = p.parse_args()

    t0 = time.time()
    r = v17_compress(args.teacher, args.v17act, DEFAULT_ROLE_K, args.D,
                     alpha=args.alpha, iters=args.iters, beam=args.beam,
                     device=args.device)
    r["wall_sec"] = time.time() - t0
    torch.save(r, args.out)
    print(f"\nSaved {args.out}")
    print(f"  alpha={args.alpha}  global bpw (w/overhead) "
          f"{r['global_bpw'] + r['overhead_bpw']:.4f}")
    print(f"  final mean {r['rel_w_final_mean']:.4f}  max {r['rel_w_final_max']:.4f}")
    print(f"  wall {r['wall_sec']:.0f}s")


if __name__ == "__main__":
    main()
