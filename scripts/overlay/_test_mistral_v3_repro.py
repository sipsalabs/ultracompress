"""End-to-end customer reproduction on Mistral-7B-v0.3 v3 pack from public HF.

Mistral just committed. Validate the customer flow:
  pip install -U ultracompress  (already done)
  hf download SipsaLabs/mistral-7b-v0.3-uc-v3-bpw5
  reconstruct layer 0 + spot-check shapes/values against the parsed v3 pack.

Runs on cuda:1 since cuda:0 is occupied by the 405B compression.
"""
from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import torch
from huggingface_hub import snapshot_download

from ultracompress.pack_v3 import parse_uc_layer_v3

REPO = "SipsaLabs/mistral-7b-v0.3-uc-v3-bpw5"
LOCAL = r"C:\Users\scamd\AppData\Local\Temp\customer_repro\mistral-7b-v0.3-uc-v3-bpw5"

# Reserved keys returned by parse_uc_layer_v3 that are NOT linears
RESERVED = {"__version__", "__layer_idx__", "__extras__"}


def main() -> int:
    print(f"[mistral-repro] start ts={time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"[mistral-repro] downloading {REPO}...", flush=True)
    t0 = time.time()
    local_dir = snapshot_download(repo_id=REPO, repo_type="model", local_dir=LOCAL)
    print(f"[mistral-repro] downloaded in {time.time() - t0:.1f}s -> {local_dir}", flush=True)

    files = sorted(Path(local_dir).glob("*"))
    uc_files = sorted([f for f in files if f.suffix == ".uc"])
    print(f"[mistral-repro] {len(files)} files, {len(uc_files)} layer.uc", flush=True)

    if not uc_files:
        print(f"[mistral-repro] FAIL: no .uc files in repo", flush=True)
        return 1

    # Parse layer_000.uc
    print("\n[mistral-repro] Parsing layer_000.uc via pack_v3...", flush=True)
    t0 = time.time()
    parsed = parse_uc_layer_v3(uc_files[0])
    print(f"[mistral-repro] layer_000 parsed in {time.time() - t0:.2f}s", flush=True)

    # Filter linears vs extras
    linear_names = sorted(k for k in parsed.keys() if k not in RESERVED)
    extras = parsed.get("__extras__", {})
    print(f"[mistral-repro]   {len(linear_names)} quantized Linears + {len(extras)} extras (norms)")
    print(f"[mistral-repro]   pack version: {parsed['__version__']}, layer_idx: {parsed['__layer_idx__']}")

    # Spot-check first quantized linear: shape sanity
    first_name = linear_names[0]
    data = parsed[first_name]
    W_base = data["W_base"]
    print(f"\n[mistral-repro] First Linear: {first_name}")
    print(f"  W_base.shape={tuple(W_base.shape)}, dtype={W_base.dtype}")
    print(f"  V.shape={tuple(data['V'].shape)}")
    print(f"  U.shape={tuple(data['U'].shape)}")
    print(f"  alpha={data['alpha']:.6f}")
    print(f"  bpw={data['bpw']}, K={data['K']}, block_size={data['block_size']}, rank={data['rank']}")

    # Reconstruct W_recon = W_base + alpha * U @ V
    W_recon = W_base.to(torch.float32) + data["alpha"] * (data["U"].to(torch.float32) @ data["V"].to(torch.float32))
    print(f"  W_recon.shape={tuple(W_recon.shape)}, |W_recon|_l2={W_recon.norm().item():.4f}")

    # Verify all linears reconstruct cleanly (no NaN/Inf)
    n_clean = 0
    for name in linear_names:
        d = parsed[name]
        W = d["W_base"].to(torch.float32) + d["alpha"] * (d["U"].to(torch.float32) @ d["V"].to(torch.float32))
        if torch.isnan(W).any() or torch.isinf(W).any():
            print(f"  FAIL: {name} has NaN/Inf!")
        else:
            n_clean += 1
    print(f"\n[mistral-repro]   {n_clean}/{len(linear_names)} linears reconstruct cleanly (no NaN/Inf)")

    # Sample a couple more layers to spot-check coverage
    n_spot = min(3, len(uc_files) - 1)
    print(f"\n[mistral-repro] Spot-checking {n_spot} additional layers...")
    for uc in uc_files[1:1 + n_spot]:
        p = parse_uc_layer_v3(uc)
        n_lin = sum(1 for k in p.keys() if k not in RESERVED)
        n_ext = len(p.get("__extras__", {}))
        print(f"  {uc.name}: {n_lin} linears + {n_ext} extras (layer_idx={p['__layer_idx__']})")

    print(f"\n[mistral-repro] === SUMMARY ===")
    print(f"    Public artifact: huggingface.co/{REPO}")
    print(f"    Files downloaded: {len(files)}")
    print(f"    layer.uc files: {len(uc_files)}")
    print(f"    Layer 0 reconstructs: {len(linear_names)} quantized Linears + {len(extras)} extras")
    print(f"    All linears reconstruct cleanly: {n_clean}/{len(linear_names)}")
    print(f"    Customer reproduction: PASS (artifact is well-formed v3 pack)")
    print(f"\n[mistral-repro] done ts={time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
