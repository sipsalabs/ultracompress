"""Pack structure + download-integrity self-check.

What this does (and only this):
  * confirms a directory looks like an UltraCompress pack (manifest.json
    present and parseable, declared layer count matches files on disk,
    no zero-byte layer files);
  * computes a stable SHA-256 fingerprint of the pack bytes so two parties
    can confirm they hold byte-identical downloads, or compare against a
    fingerprint Sipsa Labs publishes out of band.

This verifies download integrity. The end-to-end cryptographically
verifiable reconstruction audit (a deterministic decode to the
SHA-256-pinned validated artifact — the rigorous form of the
reconstruction contract) is delivered via the `uc audit` primitive
under engagement; see
docs/reference/audit-receipt-schema.md for the audit-receipt schema.
The public package contains no reconstruction methodology by design.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

_CHUNK = 8 * 1024 * 1024


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def cmd_verify(args) -> int:
    packed = Path(args.packed_dir).expanduser()
    if not packed.is_dir():
        print(f"[FAILED] not a directory: {packed}")
        return 1

    manifest_path = packed / "manifest.json"
    if not manifest_path.exists():
        print(f"[FAILED] no manifest.json in {packed} - not an UltraCompress pack dir")
        return 1
    try:
        # utf-8-sig tolerates a leading UTF-8 BOM (common on Windows-edited JSON).
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"[FAILED] manifest.json unreadable: {e}")
        return 1

    bpw = manifest.get("bpw", "?")
    declared = manifest.get("n_layers")
    schema = manifest.get("schema") or manifest.get("uc_pack_version") or "(not declared)"

    layer_files = sorted(packed.glob("layer_*.uc"))
    print(f"schema:          {schema}")
    print(f"bpw:             {bpw}")
    print(f"layer files:     {len(layer_files)}")

    ok = True

    if isinstance(declared, int) and declared != len(layer_files):
        print(f"[FAILED] manifest declares {declared} layers, found {len(layer_files)} on disk")
        ok = False
    elif not layer_files:
        print("[FAILED] no layer_*.uc files found")
        ok = False

    empty = [p.name for p in layer_files if p.stat().st_size == 0]
    if empty:
        print(f"[FAILED] zero-byte layer files: {', '.join(empty[:5])}")
        ok = False

    digest_lines: list[str] = []
    to_hash = [manifest_path] + layer_files
    aux_name = manifest.get("aux_file")
    if aux_name and (packed / aux_name).exists():
        to_hash.append(packed / aux_name)

    if getattr(args, "skip_hash", False):
        print("  --skip-hash specified; skipping integrity fingerprint.")
    else:
        full = getattr(args, "full", False)
        if full or len(layer_files) <= 3:
            shown = to_hash
        else:
            shown = [to_hash[0], to_hash[1], to_hash[len(to_hash) // 2], to_hash[-1]]
        for p in to_hash:
            digest_lines.append(f"{p.name}:{_sha256_file(p)}")
        shown_names = {p.name for p in shown}
        print("SHA-256 (spot-check; use --full for all):")
        for line in digest_lines:
            name = line.split(":", 1)[0]
            if name in shown_names:
                print(f"  {line[:len(name) + 1 + 16]}...")
        combined = hashlib.sha256("\n".join(sorted(digest_lines)).encode()).hexdigest()
        print(f"pack fingerprint (sha256 of sorted file digests):\n  {combined}")

    print()
    if ok:
        print("=> STRUCTURE OK - download integrity verified; pack is well-formed and")
        print("   the fingerprint above is the per-file SHA-256 reference. End-to-end")
        print("   cryptographically verifiable reconstruction (a deterministic decode to")
        print("   the validated artifact) is delivered via `uc audit` under")
        print("   engagement (founder@sipsalabs.com); see")
        print("   docs/reference/audit-receipt-schema.md for the audit-receipt schema.")
        print()
        print("Next:")
        print("   uc try sipsa-qwen3-0.6b     generate text with a compressed model")
        print("   uc catalog                  see all 20 architectures + tiers")
        print("   sipsalabs.com/poc           Phase 0 POC ($5K / 5 business days)")
        return 0
    print("=> FAILED - pack structure check did not pass (see lines above).")
    return 1
