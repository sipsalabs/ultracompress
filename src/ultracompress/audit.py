"""`uc audit <pack_dir>` — emits a JSON audit receipt the customer can hand
to their compliance, procurement, or security team.

What the receipt records:
  * the pack's full manifest (parsed verbatim),
  * SHA-256 of every layer file plus manifest plus any declared aux file,
  * the structural checks (`uc verify` already runs) recorded as booleans,
  * a PII-free fingerprint of the host class the audit ran on,
  * a stable `pack_fingerprint_sha256` identical to `uc verify`'s.

What the receipt deliberately does NOT do: claim bit-identical
reconstruction of model weights. That claim is provided by Sipsa Labs
under engagement; the public package ships no reconstruction methodology
and therefore cannot make the claim from a customer's machine. The audit
receipt is a structural and integrity artifact.

The schema is documented at `docs/reference/audit-receipt-schema.md` and
versioned by `schema_version`. Field renames bump major; adding fields
does not.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

from . import __version__

_CHUNK = 8 * 1024 * 1024
_SCHEMA_VERSION = "1.0"


def _looks_like_hf_repo_id(s: str) -> bool:
    """True when the arg looks like an HF repo id ('Org/model') rather than a
    local path — the most common new-user mistake. Mirrors verify.py."""
    if not s or s.startswith((".", "/", "\\")) or ":" in s or s.endswith((".uc", ".pt")):
        return False
    parts = s.split("/")
    return len(parts) == 2 and all(p and not p.isspace() for p in parts)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _host_block() -> dict:
    """Build the `host` block: machine class identifiers, zero PII.

    Deliberately excluded: hostname, username, home dir, MAC, IP, GPU id,
    drive serial, BIOS UUID. See audit-receipt-schema.md for the full
    list of fields this is *not* allowed to include.
    """
    os_name = platform.system()
    # platform.release() returns e.g. "10.0.26200" on Windows or "5.15.0-91" on Linux.
    # Trim to the major version family on Linux to avoid leaking kernel patch
    # level (a soft PII-ish identifier in a small fleet).
    raw_release = platform.release() or ""
    if os_name == "Linux":
        # First numeric token only ("5.15.0-91-generic" -> "5").
        os_release = raw_release.split(".")[0] if raw_release else ""
    elif os_name == "Windows":
        # platform.win32_ver()[0] returns "11" / "10" on modern Windows.
        try:
            os_release = platform.win32_ver()[0] or raw_release
        except Exception:  # noqa: BLE001
            os_release = raw_release
    else:
        os_release = raw_release

    cpu_arch = platform.machine() or ""
    cpu_count = os.cpu_count() or 0
    fingerprint_seed = f"{os_name}|{os_release}|{cpu_arch}|{cpu_count}"
    host_fp = hashlib.sha256(fingerprint_seed.encode("utf-8")).hexdigest()

    return {
        "os": os_name,
        "os_release": os_release,
        "python_version": sys.version.split()[0],
        "cpu_arch": cpu_arch,
        "cpu_count": cpu_count,
        "host_fingerprint": host_fp,
    }


def _build_receipt(packed: Path) -> tuple[dict, bool]:
    """Build the audit receipt dict. Returns (receipt, passing).

    The receipt is always returned (even on failure) so the failure is
    auditable. `passing` is False iff any structural check failed.
    """
    audited_at = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    receipt: dict = {
        "schema_version": _SCHEMA_VERSION,
        "tool": {"name": "ultracompress", "version": __version__},
        "audited_at_utc": audited_at,
        "pack": {
            "path": str(packed.resolve()),
            "manifest": None,
            "declared_layer_count": None,
            "observed_layer_file_count": 0,
            "bytes_on_disk": 0,
        },
        "host": _host_block(),
        "checks": {
            "manifest_present": False,
            "structure_ok": False,
            "declared_layer_count_matches": False,
            "no_zero_byte_files": False,
        },
        "files": [],
        "pack_fingerprint_sha256": "",
    }

    if not packed.is_dir():
        return receipt, False

    manifest_path = packed / "manifest.json"
    if not manifest_path.exists():
        return receipt, False

    receipt["checks"]["manifest_present"] = True

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, OSError):
        return receipt, False

    receipt["pack"]["manifest"] = manifest
    declared = manifest.get("n_layers")
    receipt["pack"]["declared_layer_count"] = (
        int(declared) if isinstance(declared, int) else None
    )

    layer_files = sorted(packed.glob("layer_*.uc"))
    receipt["pack"]["observed_layer_file_count"] = len(layer_files)

    # Files to hash: manifest + all layers + declared aux (if present on disk).
    to_hash = [manifest_path] + layer_files
    aux_name = manifest.get("aux_file")
    if aux_name and (packed / aux_name).exists():
        to_hash.append(packed / aux_name)

    file_entries: list[dict] = []
    bytes_on_disk = 0
    zero_byte = False
    for p in to_hash:
        size = p.stat().st_size
        bytes_on_disk += size
        if p.name.startswith("layer_") and size == 0:
            zero_byte = True
        file_entries.append({
            "name": p.name,
            "bytes": size,
            "sha256": _sha256_file(p),
        })
    # Stable order so the receipt is deterministic.
    file_entries.sort(key=lambda e: e["name"])
    receipt["files"] = file_entries
    receipt["pack"]["bytes_on_disk"] = bytes_on_disk

    # Stable pack fingerprint — identical to `uc verify`'s.
    digest_lines = [f"{e['name']}:{e['sha256']}" for e in file_entries]
    pack_fp = hashlib.sha256("\n".join(sorted(digest_lines)).encode()).hexdigest()
    receipt["pack_fingerprint_sha256"] = pack_fp

    layer_count_ok = (
        receipt["pack"]["declared_layer_count"] is None
        or receipt["pack"]["declared_layer_count"] == len(layer_files)
    )
    receipt["checks"]["declared_layer_count_matches"] = layer_count_ok
    receipt["checks"]["no_zero_byte_files"] = (not zero_byte) and bool(layer_files)
    receipt["checks"]["structure_ok"] = (
        receipt["checks"]["manifest_present"]
        and receipt["checks"]["declared_layer_count_matches"]
        and receipt["checks"]["no_zero_byte_files"]
    )

    passing = receipt["checks"]["structure_ok"]
    return receipt, passing


def cmd_audit(args) -> int:
    """Entry point bound from cli.py.

    Args attributes used:
      * packed_dir (positional): the pack directory to audit.
      * output (optional --output): receipt path. Default: alongside the pack
        as `<pack>.audit.json`, or stdout if `--stdout`.
      * stdout (optional --stdout): write to stdout instead of a file.
      * quiet (optional --quiet): suppress the human-readable banner.
    """
    raw = str(args.packed_dir)
    packed = Path(raw).expanduser()

    # Guard the bad path BEFORE building/writing anything. Without this, a
    # missing pack falls through to writing "<pack>.audit.json" whose parent
    # may be a drive root (e.g. C:\missing -> C:\), and mkdir() on the root
    # raises PermissionError — a raw traceback. Mirror `uc verify`'s clean UX.
    if not packed.exists():
        if _looks_like_hf_repo_id(raw):
            local_name = raw.split("/", 1)[1]
            print(f"[FAILED] '{raw}' looks like a HuggingFace repo id, not a local directory.")
            print()
            print("To audit a Sipsa pack from HuggingFace, download it first:")
            print(f"  hf download {raw} --local-dir ./{local_name}")
            print(f"  uc audit ./{local_name}")
            return 2
        print(f"[FAILED] pack directory does not exist: {packed}")
        print("         Pass the path to a downloaded UltraCompress pack directory")
        print("         (the folder containing manifest.json and layer_*.uc).")
        return 2
    if not packed.is_dir():
        print(f"[FAILED] not a directory: {packed}")
        print("         `uc audit` takes a pack directory, not a file.")
        return 2

    receipt, passing = _build_receipt(packed)
    receipt_json = json.dumps(receipt, indent=2, sort_keys=False)

    if getattr(args, "stdout", False):
        # Receipt to stdout; banner to stderr so callers can pipe the JSON.
        print(receipt_json)
        out_path: Path | None = None
    else:
        out_path = (
            Path(args.output).expanduser()
            if getattr(args, "output", None)
            else packed.with_name(packed.name + ".audit.json")
        )
        try:
            if out_path.parent and not out_path.parent.exists():
                out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(receipt_json + "\n", encoding="utf-8")
        except OSError as exc:
            # Unwritable destination (permission, read-only volume, drive root):
            # fail cleanly and fall back to stdout so the receipt is not lost.
            print(f"[FAILED] could not write receipt to {out_path}: {exc}")
            print("         Use --stdout, or --output <path> to a writable location.")
            print()
            print(receipt_json)
            return 1

    if not getattr(args, "quiet", False):
        print()
        print(f"audit schema:    {receipt['schema_version']}")
        print(f"audited at:      {receipt['audited_at_utc']}")
        print(f"pack path:       {receipt['pack']['path']}")
        print(f"pack files:      {len(receipt['files'])}")
        print(f"pack bytes:      {receipt['pack']['bytes_on_disk']}")
        print(f"fingerprint:     {receipt['pack_fingerprint_sha256']}")
        print()
        print("Structural checks:")
        for k, v in receipt["checks"].items():
            print(f"  {k:<32}  {'PASS' if v else 'FAIL'}")
        print()
        if out_path is not None:
            print(f"Receipt written to: {out_path}")
        print()
        if passing:
            print("=> AUDIT PASS - the receipt above documents this pack on this host.")
            print("   The receipt records pack structure + per-file SHA-256 + a")
            print("   PII-free host fingerprint. It is NOT a reconstruction proof.")
            print("   For a counter-signed reconstruction certificate under NDA,")
            print("   contact founder@sipsalabs.com with the pack_fingerprint_sha256.")
        else:
            print("=> AUDIT FAIL - the receipt records which structural checks did")
            print("   not pass. Hand the receipt to Sipsa Labs (founder@sipsalabs.com)")
            print("   if you believe the pack should have audited cleanly.")
        print()

    return 0 if passing else 1
