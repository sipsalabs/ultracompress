"""List pre-compressed UltraCompress models on the Hugging Face Hub."""

from __future__ import annotations

from typing import Any

from . import HF_COLLECTION_TAG, HF_ORG


def list_published_models() -> list[dict[str, Any]]:
    """Return metadata for every pre-compressed UltraCompress model on HF Hub.

    We filter the HF Hub by the author (`HF_ORG`) and the tag (`HF_COLLECTION_TAG`).
    Each returned dict has fields: modelId, base_model, bpw, size_human, downloads.
    """
    try:
        from huggingface_hub import list_models
    except ImportError as e:
        raise RuntimeError("huggingface_hub not installed; `pip install ultracompress`") from e

    out: list[dict[str, Any]] = []
    try:
        # huggingface_hub >=0.26 prefers `filter`; fall back to `tags=` for older versions.
        try:
            raw = list_models(author=HF_ORG, filter=HF_COLLECTION_TAG, limit=100)
        except TypeError:
            raw = list_models(author=HF_ORG, tags=[HF_COLLECTION_TAG], limit=100)
    except Exception:
        # Hub unreachable; return empty list rather than crash
        return out

    for m in raw:
        card = getattr(m, "card_data", None) or {}
        out.append(
            {
                "modelId": m.modelId,
                "base_model": card.get("base_model", "?"),
                "bpw": float(card.get("bpw", 0.0)),
                "size_human": card.get("size_human", "?"),
                "downloads": int(getattr(m, "downloads", 0) or 0),
                "likes": int(getattr(m, "likes", 0) or 0),
            }
        )
    # Sort by bpw ascending, then downloads descending
    out.sort(key=lambda d: (d["bpw"], -d["downloads"]))
    return out
