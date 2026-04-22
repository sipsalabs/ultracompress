"""claim21_row_order_invariance.py -- Claim 21 row-order decomposition.

Decomposes the measured Claim 21 savings into two components:

  (a) per-row intrinsic compressibility  -- the local fp8 byte distribution
      inside each restored row; should be order-invariant.
  (b) cross-row ordering gain            -- additional savings a coder extracts
      from the *sequence* in which restored rows are laid out (e.g. the
      sorted-ascending row indices form a low-delta, easy-to-code sequence).

For each restored body-Linear we collect rows independently, then for three
orderings of the rows within each linear we re-concat the three streams
(fp8, idx, scale) and measure compressed size with three strong coders
(zstd-9, lzma-6, brotli-11):

  - 'sorted'    : original ascending-index ordering (current Claim 21 default)
  - 'shuffled'  : Fisher-Yates shuffle with fixed seed
  - 'reversed'  : indices in descending order

Expected result:
  fp8 savings   : ~invariant across orderings (per-row local byte structure)
  scale savings : ~invariant across orderings
  idx savings   : HIGHLY order-dependent (sorted -> small positive deltas,
                  shuffled -> near-uniform random, reversed -> small NEGATIVE
                  deltas which re-encode almost as well)

Implication: the bulk of Claim 21 fp8/scale savings comes from per-row
intrinsic structure, NOT from clever row-ordering. The index-stream
savings, by contrast, ARE a direct consequence of sorted ordering -
which is the natural emitted order and has zero cost.

Emits:
  results/claim21_row_order_invariance_<model>_rho<rho>.json
  results/claim21_row_order_invariance.txt (aggregated table over models)
"""
from __future__ import annotations

import argparse
import bz2 as _bz2
import gc
import json
import lzma as _lzma
import sys
import time
import zlib as _zlib
from pathlib import Path

import numpy as np
import torch
import zstandard as _zstd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from compress_v14 import ROLE_PATTERNS, _role_of, build_rotation  # noqa: E402
from compress_v15 import beam_assign                              # noqa: E402
from entropy_code_overlay import (                                # noqa: E402
    MODEL_CONFIGS,
    _fp8_encode,
)

try:
    import brotli as _brotli
    HAVE_BROTLI = True
except ImportError:
    HAVE_BROTLI = False


def collect_rows_per_linear(W_fp16, role, bank, s_col, D, rot, device, rho):
    """Like select_overlay_rows but returns (sorted_idx, rows_fp8 [K,I], scales [K])
    with NO flattening. These are the per-linear-level pieces we re-order."""
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
    diff = raw * s.unsqueeze(0)
    score = (diff * diff).sum(1)
    K = max(1, int(round(rho * O))) if rho > 0 else 0
    if K == 0:
        return (torch.empty(0, dtype=torch.long),
                torch.empty(0, I, dtype=torch.uint8),
                torch.empty(0, dtype=torch.float16))
    idx = score.topk(K).indices
    idx_sorted, order = idx.sort()
    rows_fp32 = W[idx_sorted]
    rows_fp8, scales = _fp8_encode(rows_fp32)
    return idx_sorted.cpu(), rows_fp8.cpu(), scales.cpu()


def pack_streams_with_order(per_linear, order_name: str, seed: int):
    """Assemble (fp8_bytes, idx_delta_bytes, scale_bytes) for a given per-linear
    row ordering. `per_linear` is a list of (idx [K], rows [K,I] uint8, scl [K] fp16)."""
    fp8_buf = bytearray()
    idx_buf = bytearray()
    scl_buf = bytearray()

    rng = np.random.default_rng(seed)
    for (idx, rows, scl) in per_linear:
        K = int(idx.numel())
        if K == 0:
            continue
        idx_np = idx.numpy().astype(np.int32)
        rows_np = rows.numpy()   # [K, I] uint8
        scl_np = scl.numpy()     # [K] uint16 (fp16 reinterpreted)

        if order_name == "sorted":
            order = np.arange(K)
        elif order_name == "shuffled":
            order = np.arange(K)
            rng.shuffle(order)
        elif order_name == "reversed":
            order = np.arange(K - 1, -1, -1)
        else:
            raise ValueError(order_name)

        idx_ord = idx_np[order]
        rows_ord = rows_np[order]
        scl_ord = scl_np[order]

        # delta-encode indices as int32 deltas, pack as int32 so negatives
        # (for reversed ordering) encode cleanly
        if K > 0:
            deltas = np.empty(K, dtype=np.int32)
            deltas[0] = idx_ord[0]
            deltas[1:] = np.diff(idx_ord).astype(np.int32)
            idx_buf += deltas.tobytes()
            fp8_buf += rows_ord.tobytes()
            scl_buf += scl_ord.tobytes()

    return bytes(fp8_buf), bytes(idx_buf), bytes(scl_buf)


CODECS = {
    "zstd-9":    lambda b: _zstd.ZstdCompressor(level=9).compress(b),
    "lzma-6":    lambda b: _lzma.compress(b, preset=6),
    "brotli-11": (lambda b: _brotli.compress(b, quality=11)) if HAVE_BROTLI else None,
}


def compress_all(fp8_b, idx_b, scl_b):
    out = {}
    for cn, fn in CODECS.items():
        if fn is None:
            continue
        out[cn] = {
            "fp8_bytes":   len(fn(fp8_b)),
            "idx_bytes":   len(fn(idx_b)),
            "scale_bytes": len(fn(scl_b)),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3_1.7b", choices=list(MODEL_CONFIGS))
    ap.add_argument("--rho",   type=float, default=0.010)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed",   type=int, default=0x21)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.out is None:
        args.out = f"results/claim21_row_order_invariance_{args.model}_rho{args.rho}.json"

    teacher_pt, v17_pt = MODEL_CONFIGS[args.model]
    teacher_path = REPO / teacher_pt
    v17_path = REPO / v17_pt
    assert teacher_path.exists(), teacher_path
    assert v17_path.exists(), v17_path

    device = torch.device(args.device)
    print(f"[row-order] model={args.model} rho={args.rho} device={device} seed={args.seed}")

    sd = torch.load(teacher_path, map_location="cpu", weights_only=False)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    v17 = torch.load(v17_path, map_location="cpu", weights_only=False)
    D = int(v17.get("D", 8))
    banks = v17["banks"]
    s_col = v17["s_col"]

    hf_keys = [k for k in sd.keys()
               if "layers." in k and any(p in k for p in ROLE_PATTERNS)
               and k.endswith(".weight") and sd[k].ndim == 2
               and sd[k].shape[1] % D == 0]
    dims = sorted({sd[k].shape[1] for k in hf_keys})
    rots = {I: build_rotation(I, device, seed=42 + I) for I in dims}
    print(f"  body linears: {len(hf_keys)}")

    per_linear = []
    t0 = time.time()
    for n, k in enumerate(hf_keys):
        role = _role_of(k)
        bank = banks[role]
        W = sd[k]
        O, I = W.shape
        s = s_col.get(k, torch.ones(I))
        idx, rows, scl = collect_rows_per_linear(
            W, role, bank, s, D, rots[I], device, args.rho)
        per_linear.append((idx, rows, scl))
        if (n + 1) % 40 == 0:
            torch.cuda.empty_cache(); gc.collect()
            print(f"  [{n+1}/{len(hf_keys)}]  t={time.time()-t0:.0f}s")

    # Per-ordering stream packing + codec measurement
    results = {}
    raw_sizes = None
    for order_name in ("sorted", "shuffled", "reversed"):
        fp8_b, idx_b, scl_b = pack_streams_with_order(per_linear, order_name, args.seed)
        if raw_sizes is None:
            raw_sizes = {"fp8": len(fp8_b), "idx": len(idx_b), "scale": len(scl_b)}
        t1 = time.time()
        codec_results = compress_all(fp8_b, idx_b, scl_b)
        print(f"[{order_name:<8}]  streams: fp8={len(fp8_b):>12,}  idx={len(idx_b):>10,}  scl={len(scl_b):>10,}  (codec-elapsed {time.time()-t1:.0f}s)")
        for cn, d in codec_results.items():
            total = d["fp8_bytes"] + d["idx_bytes"] + d["scale_bytes"]
            print(f"    {cn:<9}  fp8={d['fp8_bytes']:>10,}  idx={d['idx_bytes']:>8,}  scl={d['scale_bytes']:>8,}  total={total:>10,}")
        results[order_name] = codec_results

    out_json = {
        "claim": 21,
        "mode": "row_order_invariance",
        "model": args.model,
        "rho": args.rho,
        "seed": args.seed,
        "raw_sizes": raw_sizes,
        "by_ordering": results,
    }
    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_json, indent=2))
    print(f"\n[wrote] {out_path}")


if __name__ == "__main__":
    main()
