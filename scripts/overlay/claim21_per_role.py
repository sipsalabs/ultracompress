"""claim21_per_role.py -- savings breakdown by linear role.

Split the Claim 21 payload by the 7 transformer linear roles
(q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj),
pack per-role streams with sorted order, and compress each role
separately. Tests whether the ~18% brotli-11 savings is uniform
across attention/MLP roles or concentrated in a subset.

Emits: results/claim21_per_role_<model>_rho<rho>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

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

REPORT_CODECS = ("zstd-9", "lzma-6", "brotli-11")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_CONFIGS))
    ap.add_argument("--rho",   type=float, default=0.010)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = f"results/claim21_per_role_{args.model}_rho{args.rho}.json"

    teacher_pt, v17_pt = MODEL_CONFIGS[args.model]
    device = torch.device(args.device)
    print(f"[per-role] model={args.model} rho={args.rho}")

    sd = torch.load(REPO / teacher_pt, map_location="cpu", weights_only=False)
    if "state_dict" in sd: sd = sd["state_dict"]
    v17 = torch.load(REPO / v17_pt, map_location="cpu", weights_only=False)
    D = int(v17.get("D", 8))
    banks = v17["banks"]; s_col = v17["s_col"]

    hf_keys = [k for k in sd.keys()
               if "layers." in k and any(p in k for p in ROLE_PATTERNS)
               and k.endswith(".weight") and sd[k].ndim == 2
               and sd[k].shape[1] % D == 0]
    dims = sorted({sd[k].shape[1] for k in hf_keys})
    rots = {I: build_rotation(I, device, seed=42 + I) for I in dims}

    per_role: dict[str, list] = defaultdict(list)
    for k in hf_keys:
        role = _role_of(k); bank = banks[role]
        W = sd[k]; I = W.shape[1]
        s = s_col.get(k, torch.ones(I))
        idx, rows, scl = collect_rows_per_linear(
            W, role, bank, s, D, rots[I], device, args.rho)
        per_role[role].append((idx, rows, scl))

    results = {
        "claim": 21, "experiment": "per_role",
        "model": args.model, "rho": args.rho,
        "roles": {},
    }
    for role in sorted(per_role.keys()):
        linears = per_role[role]
        fp8_b, idx_b, scl_b = pack_streams_with_order(linears, "sorted", seed=0)
        raw = len(fp8_b) + len(idx_b) + len(scl_b)
        entry = {
            "n_linears": len(linears),
            "raw_bytes": {"fp8": len(fp8_b), "idx_delta": len(idx_b),
                          "scale": len(scl_b), "total": raw},
            "codecs": {},
        }
        print(f"\n--- role = {role}  n_linears={len(linears)}  raw={raw:,} B ---")
        for c in REPORT_CODECS:
            fn = CODECS[c]
            if fn is None: continue
            fp8_c = len(fn(fp8_b))
            idx_c = len(fn(idx_b))
            scl_c = len(fn(scl_b))
            total_c = fp8_c + idx_c + scl_c
            saved_pct = 100.0 * (raw - total_c) / raw
            entry["codecs"][c] = {
                "fp8_bytes": fp8_c, "idx_bytes": idx_c, "scale_bytes": scl_c,
                "total_bytes": total_c,
                "saved_pct": saved_pct,
            }
            print(f"  {c:<12} fp8 {fp8_c:>10,}  idx {idx_c:>6,}  "
                  f"scl {scl_c:>6,}  saved = {saved_pct:6.3f}%")
        results["roles"][role] = entry

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[wrote] {out}")


if __name__ == "__main__":
    main()
