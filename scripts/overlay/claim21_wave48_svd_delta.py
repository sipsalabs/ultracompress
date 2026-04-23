"""claim21_wave48_svd_delta.py -- wave 48.

LOW-RANK SVD DELTA COMPRESSION.

Waves 45-47 exploited the *magnitude* structure of ft - base (small
values → good for entropy coding + low-bit quant). Wave 48 exploits
its *rank* structure: fine-tune deltas from SFT / RLHF are empirically
well-approximated by low-rank matrices (this is the well-known LoRA
assumption, applied post-hoc to already-trained full fine-tunes).

For each matching 2-D linear weight, we compute the delta
`D = ft_fp32 - base_fp32`, run a randomized truncated SVD at a fixed
fraction of rank `r = max(1, round(FRAC * min(m, n)))`, and store
`U_r (m, r)` and `Vt_r (r, n)` with singular values absorbed into U.
That is, `D ≈ Uhat @ Vt` where `Uhat[:, j] = U[:, j] * s[j]`.

We use `torch.svd_lowrank`, which runs randomized SVD on GPU and is
dramatically faster than full SVD for large matrices.

We measure:

  * raw bytes (bf16 delta baseline, bf16 U+Vt at rank r)
  * brotli-11 of bf16 U+Vt
  * rel-Frobenius ||Uhat @ Vt - D||_F / ||D||_F per tensor
  * cohort-aggregate rank-reduction ratio (params in Uhat+Vt) / (params in D)

This is pure post-hoc; we do not retrain or fine-tune anything.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import brotli
import numpy as np
import torch
from safetensors.torch import load_file as safe_load
from huggingface_hub import snapshot_download


def load_hf_state_dict(repo_id: str) -> dict:
    local = Path(snapshot_download(
        repo_id,
        allow_patterns=["*.safetensors", "*.safetensors.index.json"]))
    shards = sorted(local.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors in {local}")
    sd = {}
    for s in shards:
        sd.update(safe_load(str(s)))
    return sd


def matching_linear_keys(sd_base: dict, sd_ft: dict) -> list[str]:
    keys = []
    for k, v in sd_base.items():
        if not k.endswith(".weight") or v.ndim != 2:
            continue
        if k not in sd_ft or sd_ft[k].shape != v.shape:
            continue
        keys.append(k)
    return sorted(keys)


def rank_for(shape: tuple[int, int], frac: float) -> int:
    m, n = shape
    return max(1, int(round(frac * min(m, n))))


def svd_lowrank_delta(D_fp32_gpu: torch.Tensor, rank: int
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomized truncated SVD on GPU.

    Returns (Uhat, Vt) where Uhat = U * S broadcasted, both fp32 on GPU.
    """
    # svd_lowrank returns U (m, q), S (q,), V (n, q). Uses q ≥ rank + niter
    # oversamples internally.  q=rank is fine for modest ranks.
    U, S, V = torch.svd_lowrank(D_fp32_gpu, q=rank, niter=4)
    Uhat = U * S.unsqueeze(0)
    Vt = V.transpose(0, 1).contiguous()
    return Uhat, Vt


def bf16_bytes_of(x: torch.Tensor) -> bytes:
    return (x.to(torch.bfloat16).contiguous()
             .view(torch.int16).cpu().numpy().tobytes())


def brotli_bytes(b: bytes) -> int:
    return len(brotli.compress(b, quality=11))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--ft", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frac", type=float, default=0.125,
                    help="rank fraction: r = round(frac * min(m,n))")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    t0 = time.time()
    print(f"== wave 48 SVD-DELTA base={args.base} ft={args.ft} "
          f"frac={args.frac} device={args.device} ==")

    sd_b = load_hf_state_dict(args.base)
    sd_f = load_hf_state_dict(args.ft)
    keys = matching_linear_keys(sd_b, sd_f)
    print(f"  matched 2-D linear weights: {len(keys)}")

    dev = torch.device(args.device
                       if torch.cuda.is_available()
                       else "cpu")
    print(f"  using device: {dev}")

    bf16_parts: list[bytes] = []
    lr_parts:   list[bytes] = []
    per_tensor: list[dict] = []
    relerr_num = 0.0
    relerr_den = 0.0
    n_params = 0
    n_params_lowrank = 0

    for k in keys:
        Wb = sd_b[k].to(torch.float32)
        Wf = sd_f[k].to(torch.float32)
        # Mask bf16 equivalent for byte-exact comparability with wave 45:
        Wb = Wb.to(torch.bfloat16).to(torch.float32)
        Wf = Wf.to(torch.bfloat16).to(torch.float32)
        D = (Wf - Wb).to(dev)
        m, n = D.shape
        r = rank_for((m, n), args.frac)

        n_params += D.numel()
        n_params_lowrank += r * (m + n)

        # bf16 baseline bytes (for comparison)
        bf16_parts.append(bf16_bytes_of(D.cpu()))

        # Low-rank SVD
        Uhat, Vt = svd_lowrank_delta(D, r)
        # Reconstruction error (fp32 on GPU for accuracy)
        D_hat = Uhat @ Vt
        num = float(((D_hat - D) ** 2).sum())
        den = float((D ** 2).sum())
        relerr_num += num
        relerr_den += den

        # Serialize as bf16 bytes; scale absorbed into Uhat so decoder
        # just computes Uhat @ Vt.
        lr_parts.append(bf16_bytes_of(Uhat.cpu()))
        lr_parts.append(bf16_bytes_of(Vt.cpu()))

        per_tensor.append({
            "key": k, "shape": [m, n], "rank": r,
            "rel_frob": (num / den) ** 0.5 if den > 0 else 0.0,
        })

        # Release GPU memory aggressively
        del Wb, Wf, D, Uhat, Vt, D_hat
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    bf16 = b"".join(bf16_parts)
    lr   = b"".join(lr_parts)

    print(f"  {n_params:,} delta params  →  "
          f"{n_params_lowrank:,} low-rank params  "
          f"({100.0 * n_params_lowrank / n_params:.2f}%)")

    t = time.time(); br_bf16 = brotli_bytes(bf16); t_b = time.time() - t
    print(f"  br-11 bf16 delta : {br_bf16:,} B ({t_b:.0f}s)")
    t = time.time(); br_lr = brotli_bytes(lr); t_l = time.time() - t
    print(f"  br-11 lowrank bf16: {br_lr:,} B ({t_l:.0f}s)")

    relerr = (relerr_num / relerr_den) ** 0.5

    out = {
        "claim": 21,
        "wave": 48,
        "experiment": "svd_lowrank_delta",
        "rank_frac": args.frac,
        "base_repo": args.base,
        "ft_repo":   args.ft,
        "n_params":          n_params,
        "n_params_lowrank":  n_params_lowrank,
        "lowrank_param_fraction": n_params_lowrank / n_params,
        "sha256": {
            "bf16":    hashlib.sha256(bf16).hexdigest(),
            "lowrank": hashlib.sha256(lr).hexdigest(),
        },
        "raw_bytes": {
            "bf16_delta":        len(bf16),
            "lowrank_bf16_U_Vt": len(lr),
        },
        "brotli_11_bytes": {
            "bf16_delta":        br_bf16,
            "lowrank_bf16_U_Vt": br_lr,
        },
        "rel_frobenius_reconstruction_error": relerr,
        "ratio_bf16_delta_over_lowrank": br_bf16 / br_lr,
        "wall_seconds_total": time.time() - t0,
    }
    print()
    print("  HEADLINE:")
    print(f"    br-11 bf16 delta / br-11 lowrank bf16 = "
          f"{out['ratio_bf16_delta_over_lowrank']:.3f}x  "
          f"(relerr {relerr:.4e}, frac {args.frac})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    os.replace(tmp, out_path)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
