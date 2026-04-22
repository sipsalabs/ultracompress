"""claim21_stream_independence.py -- direct concat-vs-split experiment.

Tests whether Claim 21's 3-stream decomposition (fp8 | idx_delta |
scale) leaves any cross-stream information on the table.

For one (model, rho) we collect the EXACT same Claim 21 payload as the
row_order_invariance script (sorted ordering). We then measure, for
each strong entropy coder:

  A) SPLIT  :  sum_s | codec(stream_s) |     (Claim 21's emission)
  B) CONCAT :  | codec(fp8 || idx_delta || scale) |     (alternative)

If the 3 streams are statistically independent, CONCAT should be
LARGER than SPLIT (because a single coder sees a mixture distribution,
and order-0 Shannon is strictly increasing in mixture dispersion).

If the streams share structure, CONCAT could be SMALLER than SPLIT.

Emits:
  results/claim21_stream_independence_<model>_rho<rho>.json
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
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.out is None:
        args.out = f"results/claim21_stream_independence_{args.model}_rho{args.rho}.json"

    teacher_pt, v17_pt = MODEL_CONFIGS[args.model]
    device = torch.device(args.device)
    print(f"[stream-indep] model={args.model} rho={args.rho} device={device}")

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

    print(f"[stream-indep] {len(per_linear)} linears with restored rows")
    t0 = time.time()
    fp8_b, idx_b, scl_b = pack_streams_with_order(per_linear, "sorted", seed=0)
    print(f"[stream-indep] pack: {time.time()-t0:.1f}s  "
          f"fp8={len(fp8_b):,} idx={len(idx_b):,} scl={len(scl_b):,}")

    concat_b = fp8_b + idx_b + scl_b
    raw_bytes = len(concat_b)

    results = {
        "claim": 21,
        "experiment": "stream_independence",
        "model": args.model,
        "rho": args.rho,
        "n_linears": len(per_linear),
        "raw_sizes": {
            "fp8": len(fp8_b),
            "idx_delta": len(idx_b),
            "scale": len(scl_b),
            "concat": raw_bytes,
        },
        "by_codec": {},
    }

    for codec_name, codec_fn in CODECS.items():
        if codec_fn is None:
            continue
        t0 = time.time()
        f_cmp = codec_fn(fp8_b)
        i_cmp = codec_fn(idx_b)
        s_cmp = codec_fn(scl_b)
        split_total = len(f_cmp) + len(i_cmp) + len(s_cmp)
        t_split = time.time() - t0

        t0 = time.time()
        c_cmp = codec_fn(concat_b)
        concat_total = len(c_cmp)
        t_concat = time.time() - t0

        split_vs_raw = 100.0 * (raw_bytes - split_total) / raw_bytes
        concat_vs_raw = 100.0 * (raw_bytes - concat_total) / raw_bytes
        concat_vs_split = 100.0 * (concat_total - split_total) / split_total

        results["by_codec"][codec_name] = {
            "split_bytes": {
                "fp8": len(f_cmp), "idx_delta": len(i_cmp), "scale": len(s_cmp),
                "total": split_total,
            },
            "concat_bytes": concat_total,
            "split_vs_raw_pct":    split_vs_raw,
            "concat_vs_raw_pct":   concat_vs_raw,
            "concat_vs_split_pct": concat_vs_split,
            "split_encode_s":  t_split,
            "concat_encode_s": t_concat,
        }
        print(
            f"[{codec_name:<10}] raw={raw_bytes:>12,}  "
            f"split={split_total:>12,} ({split_vs_raw:6.3f}%)  "
            f"concat={concat_total:>12,} ({concat_vs_raw:6.3f}%)  "
            f"concat-split={concat_vs_split:+.3f}%"
        )

    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[wrote] {out_path}")


if __name__ == "__main__":
    main()
