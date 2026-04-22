"""claim21_block_shuffle.py -- characteristic context scale per stream.

Generalize the byte-permutation test: instead of permuting individual
bytes, chop the stream into fixed-size blocks and permute those
blocks. This preserves all structure UP TO the block size. As the
block size grows, any local structure shorter than the block is
preserved; so savings should recover monotonically toward the
as-emitted value.

The block size at which savings are recovered to within codec noise
is a direct measurement of the characteristic context scale the
coder is exploiting on that stream.

Order-0 preservation: at every block size the byte histogram is
identical (asserted).

Runs for one model (qwen3_1.7b) at rho=0.010. Block sizes:
  1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, full (= as-emitted).

Emits: results/claim21_block_shuffle_<model>_rho<rho>.json
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

BLOCK_SIZES = (1, 4, 16, 64, 256, 1024, 4096, 16384, 65536)


def block_shuffle(b: bytes, block: int, rng: np.random.Generator) -> bytes:
    if block <= 1:
        arr = np.frombuffer(b, dtype=np.uint8).copy()
        rng.shuffle(arr)
        return arr.tobytes()
    arr = np.frombuffer(b, dtype=np.uint8)
    n = len(arr)
    n_full = n // block
    head = arr[:n_full * block].reshape(n_full, block).copy()
    tail = arr[n_full * block:].tobytes()
    perm = rng.permutation(n_full)
    return head[perm].tobytes() + tail


def hist(b: bytes) -> np.ndarray:
    return np.bincount(np.frombuffer(b, dtype=np.uint8), minlength=256)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3_1.7b", choices=list(MODEL_CONFIGS))
    ap.add_argument("--rho",   type=float, default=0.010)
    ap.add_argument("--seed",  type=int,   default=0x21)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = f"results/claim21_block_shuffle_{args.model}_rho{args.rho}.json"

    teacher_pt, v17_pt = MODEL_CONFIGS[args.model]
    device = torch.device(args.device)
    print(f"[block-shuffle] model={args.model} rho={args.rho} device={device}")

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

    fp8_b, idx_b, scl_b = pack_streams_with_order(per_linear, "sorted", seed=0)
    print(f"[block-shuffle] raw: fp8={len(fp8_b):,} idx={len(idx_b):,} scl={len(scl_b):,}")

    streams = {"fp8": fp8_b, "idx_delta": idx_b, "scale": scl_b}
    h0 = {name: hist(b) for name, b in streams.items()}

    results = {
        "claim": 21,
        "experiment": "block_shuffle",
        "model": args.model,
        "rho": args.rho,
        "seed": args.seed,
        "block_sizes": list(BLOCK_SIZES) + ["full"],
        "raw_sizes": {name: len(b) for name, b in streams.items()},
        "by_stream": {},
    }

    rng = np.random.default_rng(args.seed)

    for stream_name, raw in streams.items():
        print(f"\n--- stream {stream_name}  ({len(raw):,} B) ---")
        per_block = {}
        # "full" = as-emitted (no shuffle)
        row = {"orig_encoded": {}}
        for codec_name, codec_fn in CODECS.items():
            if codec_fn is None:
                continue
            t0 = time.time()
            cmp = len(codec_fn(raw))
            row["orig_encoded"][codec_name] = {
                "bytes": cmp,
                "pct": 100.0 * (len(raw) - cmp) / len(raw),
                "s": time.time() - t0,
            }
        per_block["full"] = row

        for block in BLOCK_SIZES:
            perm_b = block_shuffle(raw, block, rng)
            assert np.array_equal(hist(perm_b), h0[stream_name]), \
                f"hist mismatch at block={block}"
            entry = {}
            for codec_name, codec_fn in CODECS.items():
                if codec_fn is None:
                    continue
                t0 = time.time()
                cmp = len(codec_fn(perm_b))
                entry[codec_name] = {
                    "bytes": cmp,
                    "pct": 100.0 * (len(raw) - cmp) / len(raw),
                    "s": time.time() - t0,
                }
            per_block[str(block)] = entry
            line_parts = []
            for c in ("zstd-9", "lzma-6", "brotli-11"):
                if c in entry:
                    line_parts.append(f"{c}={entry[c]['pct']:6.3f}%")
            print(f"  block={block:>6}  " + "  ".join(line_parts))
        # as-emitted
        line_parts = []
        for c in ("zstd-9", "lzma-6", "brotli-11"):
            if c in per_block["full"]["orig_encoded"]:
                line_parts.append(f"{c}={per_block['full']['orig_encoded'][c]['pct']:6.3f}%")
        print(f"  block=  full  " + "  ".join(line_parts))
        results["by_stream"][stream_name] = per_block

    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[wrote] {out_path}")


if __name__ == "__main__":
    main()
