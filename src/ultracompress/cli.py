"""UltraCompress public CLI.

Commands: `uc verify <pack_dir>`, `uc info`, `uc version`.

The public package is intentionally minimal: pack-structure and
download-integrity checking plus project information. The compression and
reconstruction methodology is patent-pending and is not distributed here.
"""
from __future__ import annotations

import argparse
import sys

from . import __version__

_INFO = """\
UltraCompress {ver} - public CLI

Near-lossless 5-bit transformer compression (~1% perplexity vs the bf16
reference; lossy). Published model artifacts decode deterministically to a
SHA-256-pinned validated artifact, giving reproducible, cryptographically
verifiable reconstruction; that reconstruction and the codec are
patent-pending and provided by Sipsa Labs under engagement.

Quick start (30 seconds, no GPU, no signup):
  uc try sipsa-qwen3-0.6b    generate text with a compressed model

This public package provides:
  uc try [model]         generate text against a Sipsa-hosted model
  uc catalog             list the full compressed-model catalog + tiers
  uc verify <pack_dir>   pack structure + download-integrity self-check
  uc audit <pack_dir>    emit a JSON audit receipt (structure + SHA-256, not a
                         reconstruction proof) for compliance / procurement
  uc info                this message
  uc version             print version

It contains no compression/reconstruction methodology by design.

Artifacts : https://huggingface.co/SipsaLabs
Project   : https://sipsalabs.com
Contact   : founder@sipsalabs.com   (technical due diligence under NDA)
License   : BUSL-1.1 + Additional Use Grant
"""


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="uc", description="UltraCompress public CLI")
    p.add_argument("-V", "--version", action="version", version=f"ultracompress {__version__}")
    sub = p.add_subparsers(dest="command")

    v = sub.add_parser("verify", help="Pack structure + download-integrity self-check")
    v.add_argument("packed_dir", help="Path to a downloaded UltraCompress pack directory")
    v.add_argument("--full", action="store_true", help="Print every file digest, not a spot-check")
    v.add_argument("--skip-hash", action="store_true", help="Skip the integrity fingerprint")

    a = sub.add_parser(
        "audit",
        help="Emit a JSON audit receipt for compliance / procurement / security review.",
    )
    a.add_argument("packed_dir", help="Path to a downloaded UltraCompress pack directory")
    a.add_argument(
        "--output",
        help="Receipt output path. Default: <pack>.audit.json alongside the pack.",
    )
    a.add_argument(
        "--stdout",
        action="store_true",
        help="Write the receipt JSON to stdout instead of a file.",
    )
    a.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the human-readable banner; only write/print the JSON.",
    )

    t = sub.add_parser("try", help="Generate text against a Sipsa-hosted compressed model")
    t.add_argument(
        "model",
        nargs="?",
        default=None,
        help="Model id (e.g. sipsa-qwen3-0.6b). Defaults to sipsa-qwen3-0.6b.",
    )
    t.add_argument(
        "--prompt",
        help="Override the default demo prompt with your own text.",
    )
    t.add_argument(
        "--key",
        help="Bearer key (sk-sps-...). Falls back to $SIPSA_API_KEY.",
    )
    t.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Cap response length (default 220).",
    )

    sub.add_parser("catalog", help="List the full compressed-model catalog + tiers")
    sub.add_parser("info", help="What this package is + contact/links")
    sub.add_parser("version", help="Print version")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv if argv is not None else sys.argv[1:])

    if args.command == "verify":
        from .verify import cmd_verify
        return cmd_verify(args)
    if args.command == "audit":
        from .audit import cmd_audit
        return cmd_audit(args)
    if args.command == "try":
        from .try_cmd import cmd_try
        return cmd_try(args)
    if args.command == "catalog":
        from .catalog import cmd_catalog
        return cmd_catalog(args)
    if args.command == "version":
        print(__version__)
        return 0
    if args.command == "info":
        print(_INFO.format(ver=__version__))
        return 0

    print(_INFO.format(ver=__version__))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
