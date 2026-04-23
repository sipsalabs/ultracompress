"""claim21_finetune_delta.py -- wave 45.

FINE-TUNE DELTA STORAGE PROBE.

First-order question before building a full overlay+quant pipeline on
deltas: does the raw delta (instruct - base, in bf16) compress
dramatically better than either endpoint under state-of-the-art
lossless coders alone?

If yes, the enterprise story is enormous: every company serving N
fine-tunes of a base model can store N-1 of them for a fraction of the
base. Combined with Claim-21 on the base and on each delta, the
total-fleet storage multiplies the headline savings by N.

For each (base, fine_tune) pair, on all shared 2-D linear weights:
  A. brotli-11(base bf16 bytes)
  B. brotli-11(fine_tune bf16 bytes)
  C. brotli-11(delta bf16 bytes)    where delta = ft - base
  D. raw sizes, bpB, ratios

We load HF safetensors directly from the local HF cache; this does not
depend on any pre-baked teacher_pt / v17_pt artifact, so new fine-tunes
can be dropped in without re-running any prior waves.
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
    """Materialize the full state_dict from an HF repo's safetensors.

    Supports single-file and sharded checkpoints.
    """
    local = snapshot_download(repo_id, allow_patterns=["*.safetensors",
                                                        "*.safetensors.index.json"])
    local = Path(local)
    shards = sorted(local.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors in {local}")
    sd = {}
    for s in shards:
        part = safe_load(str(s))
        sd.update(part)
    return sd


def matching_linear_keys(sd_base: dict, sd_ft: dict) -> list[str]:
    keys = []
    for k, v in sd_base.items():
        if not k.endswith(".weight"):
            continue
        if v.ndim != 2:
            continue
        if k not in sd_ft:
            continue
        if sd_ft[k].shape != v.shape:
            continue
        # Include every shape-matched 2-D weight (layers, embeddings,
        # lm_head). Embeddings and lm_head do move under instruction
        # tuning and belong in the delta storyy
        # valid to include. We include everything 2-D that matches.
        keys.append(k)
    return sorted(keys)


def bf16_concat(sd: dict, keys: list[str]) -> tuple[bytes, int]:
    parts = []
    n = 0
    for k in keys:
        W = sd[k].to(torch.bfloat16).contiguous()
        parts.append(W.view(torch.int16).numpy().tobytes())
        n += W.numel()
    return b"".join(parts), n


def delta_bf16_concat(sd_base: dict, sd_ft: dict, keys: list[str]
                     ) -> tuple[bytes, int]:
    """Compute (ft - base) in bf16 space and concatenate bytes.

    We cast each tensor to bf16 first (matching the storage form the
    overlay will see), then subtract in float32 for numerical safety,
    then cast back to bf16. This is the bf16 delta stream.
    """
    parts = []
    n = 0
    for k in keys:
        Wb = sd_base[k].to(torch.bfloat16).to(torch.float32)
        Wf = sd_ft[k].to(torch.bfloat16).to(torch.float32)
        D = (Wf - Wb).to(torch.bfloat16).contiguous()
        parts.append(D.view(torch.int16).numpy().tobytes())
        n += D.numel()
    return b"".join(parts), n


def gzip_bytes(b: bytes) -> int:
    import gzip
    return len(gzip.compress(b, compresslevel=9))


def zstd_bytes(b: bytes, level: int = 19) -> int:
    import zstandard as zstd
    return len(zstd.ZstdCompressor(level=level).compress(b))


def brotli_bytes(b: bytes) -> int:
    return len(brotli.compress(b, quality=11))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True,
                    help="HF repo id of the base model (e.g. allenai/OLMo-2-0425-1B)")
    ap.add_argument("--ft", required=True,
                    help="HF repo id of the fine-tuned model "
                         "(e.g. allenai/OLMo-2-0425-1B-Instruct)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-gzip-zstd", action="store_true",
                    help="Only run brotli-11 (saves wall time for "
                         "first-pass probes on multi-GB streams).")
    args = ap.parse_args()

    t0 = time.time()
    print(f"== claim21 wave 45 FT-DELTA  base={args.base}  ft={args.ft} ==")

    print("  loading base state_dict...")
    sd_b = load_hf_state_dict(args.base)
    print(f"    {len(sd_b)} tensors")
    print("  loading fine-tune state_dict...")
    sd_f = load_hf_state_dict(args.ft)
    print(f"    {len(sd_f)} tensors")

    keys = matching_linear_keys(sd_b, sd_f)
    print(f"  matched 2-D linear weights: {len(keys)}")

    print("  concatenating bf16 streams...")
    bf_base, n_b = bf16_concat(sd_b, keys)
    bf_ft,   n_f = bf16_concat(sd_f, keys)
    bf_dl,   n_d = delta_bf16_concat(sd_b, sd_f, keys)
    assert n_b == n_f == n_d, (n_b, n_f, n_d)
    nparams = n_b
    nbytes = len(bf_base)
    print(f"    {nparams:,} params  {nbytes:,} bf16 bytes per stream")

    # quick sanity: delta should have many more small-magnitude entries
    dl_i16 = np.frombuffer(bf_dl, dtype=np.int16)
    base_i16 = np.frombuffer(bf_base, dtype=np.int16)
    print(f"    |delta| i16  mean={np.mean(np.abs(dl_i16.astype(np.int64))):.1f}  "
          f"max={np.max(np.abs(dl_i16.astype(np.int64)))}")
    print(f"    |base|  i16  mean={np.mean(np.abs(base_i16.astype(np.int64))):.1f}  "
          f"max={np.max(np.abs(base_i16.astype(np.int64)))}")

    out = {
        "claim": 21,
        "wave": 45,
        "experiment": "finetune_delta_storage",
        "base_repo": args.base,
        "ft_repo": args.ft,
        "n_params": nparams,
        "bf16_bytes_per_stream": nbytes,
        "sha256": {
            "base_bf16":  hashlib.sha256(bf_base).hexdigest(),
            "ft_bf16":    hashlib.sha256(bf_ft).hexdigest(),
            "delta_bf16": hashlib.sha256(bf_dl).hexdigest(),
        },
        "compressed_bytes": {},
    }

    def compress_and_record(label: str, raw: bytes):
        rec = {"raw_bytes": len(raw)}
        t = time.time(); b = brotli_bytes(raw); rec["brotli_11"] = {
            "bytes": b, "bpB": 8.0*b/len(raw), "wall_seconds": time.time()-t}
        print(f"    {label:<12} brotli-11  {b:>12,} B  "
              f"({rec['brotli_11']['bpB']:.4f} bpB)  "
              f"({rec['brotli_11']['wall_seconds']:.0f}s)")
        if not args.skip_gzip_zstd:
            t = time.time(); g = gzip_bytes(raw); rec["gzip_9"] = {
                "bytes": g, "bpB": 8.0*g/len(raw), "wall_seconds": time.time()-t}
            print(f"    {label:<12} gzip-9     {g:>12,} B  "
                  f"({rec['gzip_9']['bpB']:.4f} bpB)  "
                  f"({rec['gzip_9']['wall_seconds']:.0f}s)")
            t = time.time(); z = zstd_bytes(raw, 19); rec["zstd_19"] = {
                "bytes": z, "bpB": 8.0*z/len(raw), "wall_seconds": time.time()-t}
            print(f"    {label:<12} zstd-19    {z:>12,} B  "
                  f"({rec['zstd_19']['bpB']:.4f} bpB)  "
                  f"({rec['zstd_19']['wall_seconds']:.0f}s)")
        return rec

    print("  compressing base bf16 ...")
    out["compressed_bytes"]["base"]  = compress_and_record("base",  bf_base)
    print("  compressing fine-tune bf16 ...")
    out["compressed_bytes"]["ft"]    = compress_and_record("ft",    bf_ft)
    print("  compressing delta bf16 ...")
    out["compressed_bytes"]["delta"] = compress_and_record("delta", bf_dl)

    br_base  = out["compressed_bytes"]["base"]["brotli_11"]["bytes"]
    br_ft    = out["compressed_bytes"]["ft"]["brotli_11"]["bytes"]
    br_delta = out["compressed_bytes"]["delta"]["brotli_11"]["bytes"]
    out["headline_ratios"] = {
        "ft_over_delta_brotli11":  br_ft / br_delta,
        "base_over_delta_brotli11": br_base / br_delta,
        "delta_over_ft_brotli11":  br_delta / br_ft,
        "raw_bf16_over_brotli11_ft":    nbytes / br_ft,
        "raw_bf16_over_brotli11_delta": nbytes / br_delta,
    }
    out["wall_seconds_total"] = time.time() - t0

    print()
    print("  HEADLINE RATIOS:")
    for k, v in out["headline_ratios"].items():
        print(f"    {k:<36}  {v:.3f}")

    # Atomic write so any external watcher (e.g. wave45_finisher) never
    # sees a partial JSON on a stable-size poll.
    out_path = Path(args.out)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    os.replace(tmp_path, out_path)
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
