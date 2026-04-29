"""Tests for ultracompress_cli.info."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

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
