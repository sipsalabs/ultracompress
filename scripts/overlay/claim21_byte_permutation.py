"""claim21_byte_permutation.py -- order-0 vs. higher-order context test.

We reuse the exact Claim 21 payload (sorted ordering) from the stream-
independence experiment and then SHUFFLE the fp8 byte buffer uniformly
at random at the byte level. Two observables per codec:

  A) savings (original fp8_bytes)       -- baseline
  B) savings (byte-permuted fp8_bytes)  -- destroys local structure
                                           but preserves order-0 stats

Order-0 Shannon H is invariant under byte permutation. So the gap
between (A) and (B) is exactly the higher-order-context savings the
coder was extracting. If (A) - (B) is large, the coder is finding
context. If (A) - (B) ~= 0, the coder is operating at order-0 only.

We also permute idx_delta and scale for completeness.

Emits: results/claim21_byte_permutation_<model>_rho<rho>.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from compress_v14 import ROLE_PATTERNS, _role_of, build_rotation  # noqa: E402
from entropy_code_overlay import MODEL_CONFIGS                    # noqa: E402
from claim21_row_order_invariance import (                        # noqa: E402
    collect_rows_per_linear,
    pack_streams_with_order,
    CODECS,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3_1.7b", choices=list(MODEL_CONFIGS))
    ap.add_argument("--rho",   type=float, default=0.010)
    ap.add_argument("--seed",  type=int,   default=0x21)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = f"results/claim21_byte_permutation_{args.model}_rho{args.rho}.json"

    teacher_pt, v17_pt = MODEL_CONFIGS[args.model]
    device = torch.device(args.device)
    print(f"[byte-perm] model={args.model} rho={args.rho} seed={args.seed} device={device}")

    sd = torch.load(REPO / teacher_pt, map_location="cpu", weights_only=False)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    v17 = torch.load(REPO / v17_pt, map_location="cpu", weights_only=False)
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
    for k in hf_keys:
        role = _role_of(k)
        bank = banks[role]
        W = sd[k]
        I = W.shape[1]
        s = s_col.get(k, torch.ones(I))
        idx, rows, scl = collect_rows_per_linear(
            W, role, bank, s, D, rots[I], device, args.rho)
        per_linear.append((idx, rows, scl))

    t0 = time.time()
    fp8_b, idx_b, scl_b = pack_streams_with_order(per_linear, "sorted", seed=0)
    print(f"[byte-perm] pack: {time.time()-t0:.1f}s  "
          f"fp8={len(fp8_b):,} idx={len(idx_b):,} scl={len(scl_b):,}")

    rng = np.random.default_rng(args.seed)
    def byte_shuffle(b):
        arr = np.frombuffer(b, dtype=np.uint8).copy()
        rng.shuffle(arr)
        return arr.tobytes()

    t0 = time.time()
    fp8_perm = byte_shuffle(fp8_b)
    idx_perm = byte_shuffle(idx_b)
    scl_perm = byte_shuffle(scl_b)
    print(f"[byte-perm] shuffle: {time.time()-t0:.1f}s")

    # Sanity: same order-0 distribution. Check byte histogram equality.
    def hist(b):
        return np.bincount(np.frombuffer(b, dtype=np.uint8), minlength=256)
    for name, orig, perm in (("fp8", fp8_b, fp8_perm),
                              ("idx_delta", idx_b, idx_perm),
                              ("scale", scl_b, scl_perm)):
        h1 = hist(orig); h2 = hist(perm)
        assert np.array_equal(h1, h2), f"hist mismatch {name}"
    print("[byte-perm] histograms match (order-0 preserved)")

    results = {
        "claim": 21,
        "experiment": "byte_permutation",
        "model": args.model,
        "rho": args.rho,
        "seed": args.seed,
        "raw_sizes": {"fp8": len(fp8_b), "idx_delta": len(idx_b), "scale": len(scl_b)},
        "by_codec": {},
    }

    for codec_name, codec_fn in CODECS.items():
        if codec_fn is None:
            continue
        per_stream = {}
        for stream_name, raw, perm in (("fp8", fp8_b, fp8_perm),
                                         ("idx_delta", idx_b, idx_perm),
                                         ("scale", scl_b, scl_perm)):
            t0 = time.time()
            orig_cmp = len(codec_fn(raw))
            t_orig = time.time() - t0
            t0 = time.time()
            perm_cmp = len(codec_fn(perm))
            t_perm = time.time() - t0

            orig_pct = 100.0 * (len(raw) - orig_cmp) / len(raw)
            perm_pct = 100.0 * (len(raw) - perm_cmp) / len(raw)
            context_gap = orig_pct - perm_pct  # savings lost when order destroyed
            per_stream[stream_name] = {
                "orig_bytes": orig_cmp,
                "perm_bytes": perm_cmp,
                "orig_pct": orig_pct,
                "perm_pct": perm_pct,
                "context_gap_pp": context_gap,
                "orig_s": t_orig,
                "perm_s": t_perm,
            }
            print(
                f"  [{codec_name:<10} / {stream_name:<10}] "
                f"orig {orig_pct:6.3f}%  perm {perm_pct:6.3f}%  "
                f"context = {context_gap:+.3f} pp"
            )
        results["by_codec"][codec_name] = per_stream

    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[wrote] {out_path}")


if __name__ == "__main__":
    main()
