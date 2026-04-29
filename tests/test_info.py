"""Tests for ultracompress_cli.info."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from ultracompress_cli import info


def test_read_metadata_returns_none_for_missing_path(tmp_path: Path) -> None:
    """A directory without an ultracompress.json returns None."""
    result = info.read_artifact_metadata(tmp_path)
    assert result is None


def test_read_metadata_loads_manifest(tmp_path: Path) -> None:
    """A directory with a valid ultracompress.json returns the parsed dict."""
    manifest = {
        "model_id": "sipsalabs/qwen3-1.7b-uc2p79",
        "base_model": "Qwen/Qwen3-1.7B",
        "method": "track-a-row-overlay",
        "bpw": 2.798,
        "size_bytes": 1_098_421_760,
        "license": "research-free",
    }
    (tmp_path / "ultracompress.json").write_text(json.dumps(manifest))
    result = info.read_artifact_metadata(tmp_path)
    assert result is not None
    assert result["bpw"] == pytest.approx(2.798)
    assert result["base_model"] == "Qwen/Qwen3-1.7B"


def test_read_metadata_handles_corrupt_json(tmp_path: Path) -> None:
    """A corrupt ultracompress.json should not crash."""
    (tmp_path / "ultracompress.json").write_text("{not_valid_json")
    result = info.read_artifact_metadata(tmp_path)
    assert result is None


def test_read_metadata_accepts_file_path(tmp_path: Path) -> None:
    """Passing the json file directly should also work."""
    manifest = {"bpw": 2.5}
    fp = tmp_path / "ultracompress.json"
    fp.write_text(json.dumps(manifest))
    result = info.read_artifact_metadata(fp)
    assert result == manifest


def test_summarize_artifact_accepts_string_numeric_fields() -> None:
    """String-valued numeric fields from manifests should not crash rendering."""
    output = StringIO()
    console = Console(file=output, force_terminal=False, width=100)
    info.summarize_artifact({"bpw": "2.798", "ratio": "5.72"}, console)
    rendered = output.getvalue()
    assert "2.798" in rendered
    assert "5.72" in rendered


def test_verify_artifact_files_accepts_matching_manifest(tmp_path: Path) -> None:
    """Declared files with matching size and SHA-256 should produce no warnings."""
    model_file = tmp_path / "model.safetensors"
    model_file.write_bytes(b"weights")
    manifest = {
        "files": {
            "model.safetensors": {
                "size_bytes": len(b"weights"),
                "sha256": "9a129038d9a00aed0cf6a7ea059ca50a813449061ab87848cf1a13eafdf33b2c",
            }
        }
    }
    assert info.verify_artifact_files(tmp_path, manifest) == []


def test_verify_artifact_files_reports_missing_and_unsafe_paths(tmp_path: Path) -> None:
    """Manifest verification should flag missing files and path traversal."""
    manifest = {
        "files": {
            "missing.safetensors": {"size_bytes": 1},
            "../outside.safetensors": {"size_bytes": 1},
        }
    }
    issues = info.verify_artifact_files(tmp_path, manifest)
    assert "Missing file: missing.safetensors" in issues
    assert "Unsafe manifest file path: ../outside.safetensors" in issues
