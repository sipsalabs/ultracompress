"""Tests for ultracompress_cli.listing."""
from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from ultracompress_cli import listing


@pytest.fixture
def fake_hf_models() -> list[Any]:
    """Return a small fixture of objects shaped like huggingface_hub ModelInfo."""

    class _Model:
        def __init__(self, modelId: str, downloads: int = 0) -> None:
            self.modelId = modelId
            self.downloads = downloads
            self.likes = 0
            self.card_data = {
                "base_model": "Qwen/Qwen3-1.7B" if "qwen" in modelId else "?",
                "bpw": 2.798,
                "size_human": "1.04GB",
            }

    return [
        _Model("sipsalabs/qwen3-1.7b-uc2p79", downloads=42),
        _Model("sipsalabs/llama2-7b-uc2p79", downloads=17),
    ]


def test_list_published_models_returns_list_when_hub_unreachable() -> None:
    """If the HF Hub call fails, we should return [] instead of crashing."""
    with patch("huggingface_hub.list_models", side_effect=ConnectionError("boom")):
        result = listing.list_published_models()
    assert isinstance(result, list)
    assert result == []


def test_list_published_models_passes_through_models(fake_hf_models: list[Any]) -> None:
    """The HF list_models result should be normalized into our dict shape."""
    with patch("huggingface_hub.list_models", return_value=fake_hf_models):
        result = listing.list_published_models()
    assert all(r["modelId"].startswith("sipsalabs/") for r in result)
    assert len(result) == 2
    for row in result:
        assert "bpw" in row
        assert "modelId" in row
        assert "downloads" in row


def test_list_published_models_handles_empty_response() -> None:
    """When the Hub returns no models, we should return []."""
    with patch("huggingface_hub.list_models", return_value=[]):
        result = listing.list_published_models()
    assert result == []
