"""Pull a pre-compressed UltraCompress model from the Hugging Face Hub."""

from __future__ import annotations

from pathlib import Path


def pull_model(model_id: str, output_dir: Path, revision: str | None = None) -> Path:
    """Download a compressed-model repo from the Hugging Face Hub.

    Uses `huggingface_hub.snapshot_download`, which supports resumption
    and parallel file downloads.

    Returns the local directory path.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise RuntimeError("huggingface_hub not installed; run `pip install ultracompress`") from e

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    local_dir = snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
    )
    return Path(local_dir)
