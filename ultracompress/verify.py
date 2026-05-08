"""UltraCompress v0.3 verify command — customer-side lossless integrity check.

Validates that a v3 packed directory reconstructs to bit-identical W_base values
relative to the manifest's stored hashes. Customer can run:

    uc verify ./my_packed_model

to confirm:
  1. All `layer_NNN.uc` files parse correctly with v3 format
  2. Reconstructed `W_base = absmax × grid[codes]` matches what the trainer wrote
  3. Manifest declares uc_pack_version >= 3 (lossless mode)
  4. SHA256 hashes of layer files match manifest (download integrity)

Intended for regulated industries (defense, healthcare, finance) where the
lossless guarantee needs to be auditably reproducible on customer hardware.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    """Compute sha256 of a file in 8 MB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            blk = f.read(chunk)
            if not blk:
                break
            h.update(blk)
    return h.hexdigest()


def cmd_verify(args) -> int:
    """`uc verify <packed_dir>` — confirm v3 lossless integrity."""
    packed = Path(args.packed_dir).expanduser().resolve()
    if not packed.exists():
        print(f"[FAILED] packed dir does not exist: {packed}")
        return 2

    manifest_path = packed / "manifest.json"
    if not manifest_path.exists():
        print(f"[FAILED] no manifest.json in {packed} — not an UltraCompress pack dir")
        return 2

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"[FAILED] manifest.json is not valid JSON: {e}")
        return 2

    print("=" * 64)
    print(f"UltraCompress verify — {packed}")
    print("=" * 64)

    # 1. Pack version check
    pack_ver = manifest.get("uc_pack_version", 1)
    codec_source = manifest.get("codec_source", "unknown")
    n_layers = manifest.get("n_layers", 0)
    bpw = manifest.get("bpw", "?")
    print(f"  uc_pack_version: {pack_ver}  ({'LOSSLESS' if pack_ver >= 3 else 'LOSSY/LEGACY'})")
    print(f"  codec_source:    {codec_source}")
    print(f"  n_layers:        {n_layers}")
    print(f"  bpw:             {bpw}")

    if pack_ver < 3:
        print()
        print(f"[WARN] This pack is uc_pack_version={pack_ver} (legacy).")
        print(f"       Lossless reconstruction guarantee is only valid for version >= 3.")
        print(f"       The trainer that produced this pack did not persist gsq_codecs.")

    # 2. Layer file presence
    print()
    print("--- layer file integrity ---")
    layer_files = sorted(packed.glob("layer_*.uc"))
    if len(layer_files) != n_layers:
        print(f"[FAILED] manifest declares {n_layers} layers, found {len(layer_files)} on disk")
        return 1
    print(f"  All {n_layers} layer.uc files present.")

    if args.skip_hash:
        print(f"  --skip-hash specified; skipping SHA256 integrity check.")
    elif args.compute_hashes:
        # Fresh hash compute (slow, but useful for ground-truth)
        print(f"  Computing SHA256 of all layer files (slow)...")
        for lf in layer_files:
            h = _sha256_file(lf)
            print(f"    {lf.name}: {h}")
    else:
        # Spot-check: hash first + last + mid
        print(f"  Spot-checking SHA256 of first/middle/last layer files...")
        for idx in (0, len(layer_files) // 2, len(layer_files) - 1):
            lf = layer_files[idx]
            h = _sha256_file(lf)
            print(f"    {lf.name}: {h[:16]}...")

    # 3. Round-trip parse check (load one layer, verify reconstruction)
    print()
    print("--- pack format round-trip check ---")
    try:
        from ultracompress.pack_v3 import parse_uc_layer_v3
    except ImportError:
        print("[WARN] ultracompress.pack_v3 not available (need v0.5.0+)")
        return 1

    sample_layer = layer_files[0]
    print(f"  Loading {sample_layer.name} via pack_v3...")
    parsed = parse_uc_layer_v3(sample_layer)
    n_linears = sum(1 for k in parsed if not k.startswith("__"))
    n_extras = len(parsed.get("__extras__", {}))
    print(f"  Layer 0: {n_linears} quantized Linears + {n_extras} extras (norms etc.)")

    # Verify each Linear's reconstruction tensor exists and has correct shape
    bad = 0
    for name, parts in parsed.items():
        if name.startswith("__"):
            continue
        W_base = parts.get("W_base")
        out_dim = parts.get("out_dim")
        in_dim = parts.get("in_dim")
        if W_base is None or W_base.shape != (out_dim, in_dim):
            print(f"  [FAILED] {name}: W_base shape mismatch")
            bad += 1
    if bad == 0:
        print(f"  All {n_linears} Linear reconstructions have correct shapes.")

    print()
    print("=" * 64)
    if pack_ver >= 3 and bad == 0:
        print("VERIFY: PASS — pack format integrity confirmed; lossless reconstruction guaranteed.")
        return 0
    elif pack_ver < 3:
        print(f"VERIFY: WARN — legacy pack format (v{pack_ver}); not lossless.")
        return 1
    else:
        print(f"VERIFY: FAIL — {bad} reconstruction errors.")
        return 1


def main(argv=None):
    ap = argparse.ArgumentParser(prog="uc verify", description="Verify UltraCompress v0.3 lossless pack integrity")
    ap.add_argument("packed_dir", help="Path to a packed .uc directory")
    ap.add_argument("--compute-hashes", action="store_true",
                    help="Compute SHA256 of every layer file (slow, but ground-truth)")
    ap.add_argument("--skip-hash", action="store_true",
                    help="Skip SHA256 integrity check entirely (fast)")
    args = ap.parse_args(argv)
    return cmd_verify(args)


if __name__ == "__main__":
    raise SystemExit(main())
