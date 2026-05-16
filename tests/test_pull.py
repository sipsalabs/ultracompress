"""Tests for ultracompress_cli.pull (download CLI plumbing).

We don't actually hit the HF Hub in tests — the wrapper is patched.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from ultracompress_cli import pull


def test_pull_model_invokes_snapshot_download(tmp_path: Path) -> None:
    """The pull wrapper should call huggingface_hub.snapshot_download with the right args."""
    with patch("huggingface_hub.snapshot_download") as mock_snap:
        mock_snap.return_value = str(tmp_path)
        result = pull.pull_model("sipsalabs/qwen3-1.7b-uc-v3-bpw5", tmp_path)
    mock_snap.assert_called_once()
    _, kwargs = mock_snap.call_args
    assert kwargs.get("repo_id") == "sipsalabs/qwen3-1.7b-uc-v3-bpw5"
    assert Path(kwargs.get("local_dir")) == tmp_path.resolve()
    assert isinstance(result, Path)


def test_pull_model_with_revision(tmp_path: Path) -> None:
    """When revision is set, snapshot_download should receive it."""
    with patch("huggingface_hub.snapshot_download") as mock_snap:
        mock_snap.return_value = str(tmp_path)
        pull.pull_model("sipsalabs/qwen3-1.7b-uc-v3-bpw5", tmp_path, revision="abc123")
    _, kwargs = mock_snap.call_args
    assert kwargs.get("revision") == "abc123"


def test_pull_model_propagates_hub_error(tmp_path: Path) -> None:
    """Hub errors should propagate so the CLI can exit cleanly."""
    import pytest

    with patch("huggingface_hub.snapshot_download", side_effect=ConnectionError("hub unreachable")):
        with pytest.raises(ConnectionError):
            pull.pull_model("sipsalabs/qwen3-1.7b-uc-v3-bpw5", tmp_path)
