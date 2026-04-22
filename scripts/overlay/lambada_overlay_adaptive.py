"""lambada_overlay_adaptive.py -- Claim 22 (Sensitivity-Adaptive Overlay Budget).

The fp8 overlay in Claim 18A restores a *uniform* rho * O rows per body
Linear. This is suboptimal: some linears (attention-out, MLP-down) have
much higher quantization error than q/k/v projections, because their
output distribution is heavier-tailed. Uniform allocation over-spends
overlay bits on robust layers and under-spends on sensitive ones.

Claim 22 replaces uniform rho with a **budget allocation proportional to
codebook reconstruction error**:

  1. Decode each body Linear with the v17 codebook (no overlay).
  2. Compute MSE_i = ||W_i - Wq_i||_F^2       (Frobenius residual energy)
     or weighted   = ||(W_i - Wq_i) * s_i||   (column-scaled, same as Claim 18D)
  3. Total row budget K_total = rho * sum(O_i).
  4. Allocate per-linear rows k_i proportional to MSE_i, clipped to [0, O_i].
  5. Restore the top-k_i highest-residual rows in each Linear with fp8 + scale.

The allocation is **calibration-free**: MSE_i is a pure byproduct of the
v17 decode that the baseline compressor already performs. No gradient data,
no activations, no Hessian diagonal approximation needed.

Expected outcome: at matched effective bpw, 0.3 - 0.8 pp higher LAMBADA
T1 than uniform rho, driven by reallocating bits from q/k/v projections
(low residual) to attn_out and mlp_down (high residual).

Run:
    python scripts/overlay/lambada_overlay_adaptive.py \
        --model qwen3_1.7b --rho 0.003 --device cuda:0
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from compress_v14 import ROLE_PATTERNS, _role_of, build_rotation  # noqa: E402
from compress_v15 import beam_assign                              # noqa: E402
from lambada_overlay import MODELS                                # noqa: E402


# ---------- fp8 encode/decode (per-row scale, identical to Claim 18A) ------
def _fp8_round_trip(x: torch.Tensor):
    absmax = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scale = absmax / 448.0
    xs = (x / scale).to(torch.float8_e4m3fn)
    return xs.to(torch.float32) * scale, scale.squeeze(1)


# ---------- stage 1: decode + compute per-linear residual energy -----------
def _decode_and_score(W_fp16, bank, s_col, D, rot, device, score_mode):
    """Decode W under v17, return (W_cpu_fp32, Wq_device_fp32, residual, score_mode_residual_energy)."""
    W = W_fp16.to(device=device, dtype=torch.float32)
    s = s_col.to(device=device, dtype=torch.float32)
    W_scaled = W * s.unsqueeze(0)
    Wrot = W_scaled @ rot
    O, I = Wrot.shape
    rs = Wrot.abs().amax(1, keepdim=True).clamp(min=1e-6)
    g = (Wrot / rs).view(O, I // D, D).reshape(-1, D)
    cb1 = bank["cb1"].to(device); cb2 = bank["cb2"].to(device)
    chunk = max(25_000, (200_000 * 2048) // max(cb1.shape[0], 1))
    idx1, idx2, _ = beam_assign(g, cb1, cb2, beam=8, chunk=chunk)
    gh = cb1[idx1] + cb2[idx2]
    Wq_rot_scaled = (gh.view(O, I // D, D).reshape(O, I)) * rs.expand(O, I)
    Wq_scaled = Wq_rot_scaled @ rot.T
    Wq = Wq_scaled / s.unsqueeze(0)
    raw = W - Wq
    if score_mode == "weighted":
        diff = raw * s.unsqueeze(0)
    else:
        diff = raw
    per_row_score = (diff * diff).sum(1)          # [O] row-wise residual energy
    total_energy = float(per_row_score.sum().item())  # scalar per-linear sensitivity
    # Free intermediates but keep W, Wq, per_row_score on device.
    del W_scaled, Wrot, g, cb1, cb2, idx1, idx2, gh, Wq_rot_scaled, Wq_scaled, raw, diff, rs, s
    return W, Wq, per_row_score, total_energy, O, I


# ---------- stage 2: apply overlay with pre-computed k_i -------------------
def _apply_overlay_with_budget(W, Wq, per_row_score, k_i):
    """Restore the k_i highest-residual rows of W into Wq (in place), fp8-quantized."""
    O, I = Wq.shape
    if k_i <= 0 or O == 0:
        return 0, 0
    k_i = min(k_i, O)
    idx = per_row_score.topk(k_i).indices
    rows_fp32 = W[idx]
    rows_q, _ = _fp8_round_trip(rows_fp32)
    Wq[idx] = rows_q
    return k_i, k_i * I


# ---------- two-pass driver ------------------------------------------------
def substitute_v17_adaptive_overlay(model, state_dict, v17, device, D, rho, score_mode):
    rots = {}
    dims = sorted({v.shape[1] for k, v in state_dict.items()
                   if "layers." in k and any(p in k for p in ROLE_PATTERNS)
                   and v.ndim == 2 and v.shape[1] % D == 0})
    for I in dims:
        rots[I] = build_rotation(I, device, seed=42 + I)
    banks = v17["banks"]
    s_col = v17["s_col"]
    hf_keys = [k for k in state_dict.keys()
               if "layers." in k and any(p in k for p in ROLE_PATTERNS)
               and k.endswith(".weight") and state_dict[k].ndim == 2
               and state_dict[k].shape[1] % D == 0]

    # -- PASS 1: decode every linear, record sensitivity energy --
    print(f"  [pass 1] decoding {len(hf_keys)} linears, measuring sensitivity")
    cached = []   # list of (idx, key, role, O, I, energy)
    energies = []
    O_list = []
    total_rows = 0
    total_params = 0
    for n, k in enumerate(hf_keys):
        role = _role_of(k)
        bank = banks[role]
        W = state_dict[k]
        s = s_col.get(k, torch.ones(W.shape[1]))
        _W, _Wq, _scr, E, O, I = _decode_and_score(
            W, bank, s, D, rots[W.shape[1]], device, score_mode)
        cached.append((n, k, role, O, I, E))
        energies.append(E)
        O_list.append(O)
        total_rows += O
        total_params += O * I
        # free GPU tensors; re-decode in pass 2 (cheap vs. caching every Wq)
        del _W, _Wq, _scr
        if (n + 1) % 40 == 0:
            torch.cuda.empty_cache(); gc.collect()

    energies_t = torch.tensor(energies, dtype=torch.float64)
    O_t = torch.tensor(O_list, dtype=torch.float64)

    # -- Budget allocation: K_total rows distributed proportional to energy^alpha --
    # alpha=1 is plain proportional; alpha in (0,1) dampens toward uniform.
    K_total = int(round(rho * total_rows))
    alpha = 1.0
    weights = energies_t.pow(alpha)
    if weights.sum() <= 0:
        weights = torch.ones_like(weights)
    alloc = (weights / weights.sum()) * K_total
    # clip to [0, O_i] and re-normalize overflow into unclipped layers
    alloc = torch.minimum(alloc, O_t)
    # very rarely some layer has tiny O_i and gets its cap -- iterate to consume residual budget
    for _ in range(4):
        residual = K_total - float(alloc.sum().item())
        if residual <= 0.5:
            break
        slack = (O_t - alloc).clamp(min=0)
        if slack.sum() <= 0:
            break
        alloc = alloc + (slack / slack.sum()) * residual
        alloc = torch.minimum(alloc, O_t)
    k_i = alloc.round().long()

    # -- PASS 2: re-decode + apply per-layer overlay --
    print(f"  [pass 2] applying adaptive overlay (K_total={int(k_i.sum().item())} "
          f"vs uniform_K={K_total})")
    total_restored = 0
    total_params_restored = 0
    for (n, k, role, O, I, E), k_this in zip(cached, k_i.tolist()):
        bank = banks[role]
        W = state_dict[k]
        s = s_col.get(k, torch.ones(W.shape[1]))
        _W, Wq, scr, _E, _O, _I = _decode_and_score(
            W, bank, s, D, rots[W.shape[1]], device, score_mode)
        nr, nparams = _apply_overlay_with_budget(_W, Wq, scr, int(k_this))
        total_restored += nr
        total_params_restored += nparams
        out = Wq.to(dtype=torch.float16, device="cpu")
        mod = model
        for p in k.replace(".weight", "").split("."):
            mod = getattr(mod, p)
        mod.weight.data.copy_(out.to(mod.weight.device, dtype=mod.weight.dtype))
        del _W, Wq, scr, out
        if (n + 1) % 40 == 0:
            torch.cuda.empty_cache(); gc.collect()
    torch.cuda.empty_cache(); gc.collect()

    gbpw = float(v17.get("global_bpw", 2.78))
    if total_restored > 0:
        avg_I = total_params_restored / total_restored
        excess_bits = total_restored * ((8 - gbpw) * avg_I + 16 + 32)
    else:
        excess_bits = 0.0
    overlay_bpw = excess_bits / max(total_params, 1)
    print(f"  adaptive overlay: restored {total_restored}/{total_rows} rows "
          f"({100*total_restored/max(total_rows,1):.3f}%), "
          f"{total_params_restored}/{total_params} params "
          f"({100*total_params_restored/max(total_params,1):.3f}%)")
    print(f"  adaptive overlay bpw overhead: +{overlay_bpw:.4f} "
          f"(base {gbpw:.4f} -> effective {gbpw+overlay_bpw:.4f})")

    # diagnostic: per-role allocation share
    role_share = {}
    for (_, k, role, O, I, E), k_this in zip(cached, k_i.tolist()):
        r = role_share.setdefault(role, {"rows_restored": 0, "rows_total": 0, "energy": 0.0})
        r["rows_restored"] += int(k_this)
        r["rows_total"] += O
        r["energy"] += E
    for role, d in role_share.items():
        frac = d["rows_restored"] / max(d["rows_total"], 1)
        print(f"    role {role:>16s}: {d['rows_restored']:>6d}/{d['rows_total']:>6d} "
              f"rows  ({frac*100:5.2f}%)  E={d['energy']:.3e}")

    return {
        "n_restored_rows": int(total_restored),
        "n_total_rows": int(total_rows),
        "rows_pct": 100 * total_restored / max(total_rows, 1),
        "overlay_bpw": float(overlay_bpw),
        "base_bpw": gbpw,
        "effective_bpw": float(gbpw + overlay_bpw),
        "role_share": role_share,
    }


# ---------- run one: identical protocol to lambada_overlay_fp8.run_one ----
def run_one(name, model_id, teacher_path, v17_path, tokens_path,
            n, seq_len, device, rho, score_mode="weighted",
            tier="hifi+adaptive-overlay"):
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.modeling_utils import no_init_weights
    from eval_v16_ppl import measure_ppl
    from eval_claim16_topk import measure_topk

    print(f"\n[{name}] teacher -> {teacher_path}", flush=True)
    sd = torch.load(teacher_path, map_location="cpu", weights_only=False)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    toks = torch.load(tokens_path, weights_only=True).to(torch.long)
    g = torch.Generator().manual_seed(42)
    starts = torch.randint(0, toks.numel() - seq_len - 1, (n,), generator=g)

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    with no_init_weights():
        model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float16,
                                                 trust_remote_code=True)
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()

    t0 = time.time()
    tch_topk, tch_cache = measure_topk(model, toks, starts, seq_len, device, teacher_topk=None)
    tch_ppl, _ = measure_ppl(model, toks, starts, seq_len, device)
    print(f"[{name}] teacher  PPL={tch_ppl:.4f}  T1={tch_topk['t1_gt']*100:.2f}%  "
          f"T10={tch_topk['t10_gt']*100:.2f}%   ({time.time()-t0:.0f}s)", flush=True)

    t1 = time.time()
    v17 = torch.load(v17_path, map_location="cpu", weights_only=False)
    D = int(v17.get("D", 8))
    ov = substitute_v17_adaptive_overlay(model, sd, v17, device, D, rho, score_mode)
    v17_topk, _ = measure_topk(model, toks, starts, seq_len, device, teacher_topk=tch_cache)
    v17_ppl, _ = measure_ppl(model, toks, starts, seq_len, device)
    print(f"[{name}] v17+adapt rho={rho} score={score_mode}  PPL={v17_ppl:.4f}  "
          f"T1={v17_topk['t1_gt']*100:.2f}%  T10={v17_topk['t10_gt']*100:.2f}%  "
          f"T1_vs_teacher={v17_topk['t1_agree']*100:.2f}%  ({time.time()-t1:.0f}s)", flush=True)

    ppl_ratio = v17_ppl / tch_ppl
    t1_ret  = v17_topk['t1_gt']  / tch_topk['t1_gt']  if tch_topk['t1_gt']  > 0 else 0.0
    t10_ret = v17_topk['t10_gt'] / tch_topk['t10_gt'] if tch_topk['t10_gt'] > 0 else 0.0
    return {
        "name": name,
        "tier": tier,
        "rho": rho,
        "score_mode": score_mode,
        "n": n,
        "seq_len": seq_len,
        "teacher_ppl": float(tch_ppl),
        "teacher_t1": float(tch_topk['t1_gt']),
        "teacher_t10": float(tch_topk['t10_gt']),
        "v17_ppl": float(v17_ppl),
        "v17_t1": float(v17_topk['t1_gt']),
        "v17_t10": float(v17_topk['t10_gt']),
        "v17_t1_vs_teacher": float(v17_topk['t1_agree']),
        "ppl_ratio": float(ppl_ratio),
        "t1_ret": float(t1_ret),
        "t10_ret": float(t10_ret),
        "overlay": ov,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3-1.7B",
                    help="exact name from lambada_overlay.MODELS")
    ap.add_argument("--rho",   type=float, default=0.003)
    ap.add_argument("--score", default="weighted",
                    choices=["weighted", "unweighted"])
    ap.add_argument("--n",     type=int, default=80)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # MODELS is a list of (name, model_id, teacher, v17, tokens) tuples
    found = None
    for row in MODELS:
        if row[0] == args.model:
            found = row; break
    if found is None:
        names = [r[0] for r in MODELS]
        raise SystemExit(f"--model {args.model!r} not in MODELS; choose from {names}")
    name, model_id, teacher, v17_pt, tokens = found

    device = torch.device(args.device)
    result = run_one(
        name=name,
        model_id=model_id,
        teacher_path=str(REPO / teacher),
        v17_path=str(REPO / v17_pt),
        tokens_path=str(REPO / tokens),
        n=args.n, seq_len=args.seq_len, device=device,
        rho=args.rho, score_mode=args.score,
    )
    if args.out:
        out_path = REPO / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"\n[adaptive] wrote {out_path}")


if __name__ == "__main__":
    main()
