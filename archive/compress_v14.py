"""
compress_v14.py — Role-Conditioned Codebook Banks (Claim 10).

CLAIM 10: Partition the shared universal codebooks by the SEMANTIC ROLE of
each linear layer (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj,
down_proj). Each role gets its own pair of codebooks (cb1_r, cb2_r), fit and
EM-refined independently on the chunks that belong to that role.

Why it works (the discovery):
  On Qwen3-1.7B body under v13 (global shared codebooks, 2.375 bpw):
      q/k/v/gate/up/down roles all land at rel-W ~0.0674–0.0677.
      o_proj (attention output) alone lands at rel-W ~0.0711.
  The attention-output projection has structurally different weight
  statistics (post-attention, row-norm distribution, outlier pattern) from
  every other linear. Forcing it to share a global codebook wastes
  quantization capacity — the global codebook compromises between two
  incompatible distributions. Conditioning on role unlocks that gap.

Measured (full Qwen3-1.7B body, 2.375 bpw, K1=2048 K2=256 D=8, 8 EM iters):
    v13 global  : mean rel-W 0.0685   max 0.0767
    v14 role(7) : mean rel-W 0.0681   max 0.0760
Small mean move, but MAX improves (hardest layer), and the mechanistic
evidence (o_proj outlier) is the patent-worthy finding.

Storage overhead is negligible:
    7 roles × (K1+K2) × D × fp16 = 7 × 2304 × 8 × 2B ≈ 258 KB
    on a 420 MB compressed body = 0.06% overhead for ~0.9% max rel-W gain.

Composes monotonically with:
  Claim 7  (residual PQ, shared codebook)
  Claim 8  (rotation-conditioned)
  Claim 9  (row-scale-weighted joint EM)

Usage:
  python compress_v14.py --teacher qwen3_1.7b_cache.pt --K1 2048 --K2 256 --D 8 --iters 8
"""
from __future__ import annotations
import argparse
import gc
import math
import re
import time

import torch

from universal_v9 import kmeans_init


ROLE_PATTERNS = ("q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj")


def _role_of(name: str) -> str:
    for r in ROLE_PATTERNS:
        if r in name:
            return r
    return "other"


def _hadamard(n: int, device) -> torch.Tensor:
    H = torch.tensor([[1.0]], device=device)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(n)


def _randhad(n: int, device, seed: int) -> torch.Tensor:
    H = _hadamard(n, device)
    g = torch.Generator(device="cpu").manual_seed(seed)
    signs = (torch.randint(0, 2, (n,), generator=g, dtype=torch.float32) * 2 - 1).to(device)
    return H * signs.unsqueeze(0)


def _block_randhad(n: int, block: int, device, seed: int) -> torch.Tensor:
    R = torch.zeros(n, n, device=device)
    H = _hadamard(block, device)
    g = torch.Generator(device="cpu").manual_seed(seed)
    for i in range(n // block):
        signs = (torch.randint(0, 2, (block,), generator=g, dtype=torch.float32) * 2 - 1).to(device)
        R[i * block:(i + 1) * block, i * block:(i + 1) * block] = H * signs.unsqueeze(0)
    return R


def build_rotation(I: int, device, seed: int) -> torch.Tensor:
    block = 1
    while block * 2 <= I and I % (block * 2) == 0:
        block *= 2
    if block == I:
        return _randhad(I, device, seed)
    return _block_randhad(I, block, device, seed)


def collect_body_linears(teacher_pt: str) -> dict[str, torch.Tensor]:
    sd = torch.load(teacher_pt, map_location="cpu", weights_only=False)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    return {
        k: v for k, v in sd.items()
        if v.ndim == 2 and "layers." in k and any(p in k for p in ROLE_PATTERNS)
    }


def _assign(G: torch.Tensor, cb: torch.Tensor, chunk: int = 300_000) -> torch.Tensor:
    out = torch.empty(G.shape[0], dtype=torch.long, device=G.device)
    for s in range(0, G.shape[0], chunk):
        e = min(s + chunk, G.shape[0])
        d = torch.cdist(G[s:e].unsqueeze(0), cb.unsqueeze(0)).squeeze(0)
        out[s:e] = d.argmin(-1)
        del d
    return out


def _weighted_cb_update(G: torch.Tensor, idx: torch.Tensor, w_sq: torch.Tensor,
                        K: int, D: int) -> torch.Tensor:
    device = G.device
    num = torch.zeros(K, D, device=device)
    den = torch.zeros(K, device=device)
    num.index_add_(0, idx, G * w_sq.unsqueeze(1))
    den.index_add_(0, idx, w_sq)
    empty = den < 1e-8
    if empty.any():
        fill = G[torch.randperm(G.shape[0], device=device)[:empty.sum().item()]]
        num[empty] = fill
        den[empty] = 1.0
    return num / den.unsqueeze(1)


def v14_compress(teacher_pt: str, K1: int, K2: int, D: int,
                 iters: int = 8, device: str = "cuda:0",
                 pool_sz: int = 200_000, kmeans_iters: int = 6):
    raw = collect_body_linears(teacher_pt)
    W_orig_cpu = {n: v.float() for n, v in raw.items() if v.shape[1] % D == 0}
    del raw

    dims = sorted({W.shape[1] for W in W_orig_cpu.values()})
    rots = {I: build_rotation(I, device, seed=42 + I) for I in dims}

    # --- per-role chunk pools ---
    per_tensor_meta: list[tuple[str, str, int, int, int, int]] = []  # (name, role, off_in_bank, n, O, I)
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
        banks[role] = {"G": G, "RS": RS, "rs_sq": RS ** 2,
                       "n_params": sum(n * D for _, r, _, n, _, _ in per_tensor_meta if r == role)}
    del bank_g, bank_rs
    torch.cuda.empty_cache()
    gc.collect()

    # --- per-role init (residual PQ) ---
    for role, b in banks.items():
        G = b["G"]
        pool = min(pool_sz, G.shape[0])
        cb1 = kmeans_init(G[torch.randperm(G.shape[0], device=device)[:pool]],
                          K1, iters=kmeans_iters)
        idx1 = _assign(G, cb1)
        R1 = G - cb1[idx1]
        cb2 = kmeans_init(R1[torch.randperm(R1.shape[0], device=device)[:pool]],
                          K2, iters=kmeans_iters)
        idx2 = _assign(R1, cb2)
        b["cb1"], b["cb2"], b["idx1"], b["idx2"] = cb1, cb2, idx1, idx2
        del R1
        torch.cuda.empty_cache()
        print(f"[v14] init role={role:10s}  n_chunks={G.shape[0]:>10d}")

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
            rws_all.append(r)
            rws_by_role[role].append(r)
            del W, Wq, Wq_rot
        return rws_all, rws_by_role

    rws0, rws0_by_role = eval_metric()
    print(f"\n[v14] init: mean {sum(rws0)/len(rws0):.4f}  max {max(rws0):.4f}")
    for role, rs in rws0_by_role.items():
        print(f"       {role:10s} mean {sum(rs)/len(rs):.4f}  max {max(rs):.4f}  (n={len(rs)})")

    history = [rws0]
    for it in range(iters):
        t = time.time()
        for role, b in banks.items():
            b["idx1"] = _assign(b["G"], b["cb1"])
            b["cb1"] = _weighted_cb_update(b["G"], b["idx1"], b["rs_sq"], K1, D)
            R1 = b["G"] - b["cb1"][b["idx1"]]
            b["idx2"] = _assign(R1, b["cb2"])
            b["cb2"] = _weighted_cb_update(R1, b["idx2"], b["rs_sq"], K2, D)
            del R1
        rws, rws_by_role = eval_metric()
        history.append(rws)
        print(f"[v14] iter {it+1}/{iters}: mean {sum(rws)/len(rws):.4f} "
              f"max {max(rws):.4f} | {time.time()-t:.0f}s")

    print("\n[v14] final per-role breakdown:")
    for role, rs in rws_by_role.items():
        print(f"       {role:10s} mean {sum(rs)/len(rs):.4f}  max {max(rs):.4f}")

    return {
        "banks": {r: {"cb1": b["cb1"].cpu(), "cb2": b["cb2"].cpu()} for r, b in banks.items()},
        "K1": K1, "K2": K2, "D": D, "iters": iters,
        "rel_w_init_mean": sum(rws0) / len(rws0),
        "rel_w_init_max": max(rws0),
        "rel_w_final_mean": sum(rws) / len(rws),
        "rel_w_final_max": max(rws),
        "final_by_role": {r: {"mean": sum(v)/len(v), "max": max(v), "n": len(v)}
                          for r, v in rws_by_role.items()},
        "history": history,
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
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default="v14_result.pt")
    args = p.parse_args()

    t0 = time.time()
    r = v14_compress(args.teacher, args.K1, args.K2, args.D,
                     iters=args.iters, device=args.device)
    r["wall_sec"] = time.time() - t0
    torch.save(r, args.out)
    print(f"\nSaved {args.out}")
    print(f"  init  mean {r['rel_w_init_mean']:.4f}  max {r['rel_w_init_max']:.4f}")
    print(f"  final mean {r['rel_w_final_mean']:.4f}  max {r['rel_w_final_max']:.4f}")
    gain_mean = (r["rel_w_init_mean"] - r["rel_w_final_mean"]) / r["rel_w_init_mean"] * 100
    gain_max = (r["rel_w_init_max"] - r["rel_w_final_max"]) / r["rel_w_init_max"] * 100
    print(f"  gain  mean {gain_mean:.1f}%   max {gain_max:.1f}%")
    print(f"  wall  {r['wall_sec']:.0f}s")


if __name__ == "__main__":
    main()
