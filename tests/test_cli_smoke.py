"""Smoke tests for the actual public CLI surface (verify / try / catalog / info / version)."""
from __future__ import annotations

import json
import subprocess
import sys


def test_version_prints():
    from ultracompress import __version__
    assert __version__.startswith("0.6.")


def test_uc_version_command():
    r = subprocess.run([sys.executable, "-m", "ultracompress", "version"],
                       capture_output=True, text=True, timeout=15)
    assert r.returncode == 0
    assert "0.6." in r.stdout


def test_uc_info_command():
    r = subprocess.run([sys.executable, "-m", "ultracompress", "info"],
                       capture_output=True, text=True, timeout=15)
    assert r.returncode == 0
    assert "UltraCompress" in r.stdout


def test_uc_verify_missing_dir():
    r = subprocess.run([sys.executable, "-m", "ultracompress", "verify", "/nonexistent/path"],
                       capture_output=True, text=True, timeout=15)
    assert r.returncode != 0


def test_uc_verify_bom_tolerant_manifest(tmp_path):
    """v0.6.19+ verify must parse UTF-8-BOM-prefixed manifests."""
    p = tmp_path / "pack"
    p.mkdir()
    (p / "manifest.json").write_bytes(
        b"\xef\xbb\xbf" + json.dumps({"schema": "sipsa-uc", "bpw": 5, "n_layers": 1}).encode("utf-8"))
    (p / "layer_000.uc").write_bytes(b"fake-layer-content")
    r = subprocess.run([sys.executable, "-m", "ultracompress", "verify", str(p), "--skip-hash"],
                       capture_output=True, text=True, timeout=15)
    assert r.returncode == 0
    assert "schema:" in r.stdout
    assert "sipsa-uc" in r.stdout
