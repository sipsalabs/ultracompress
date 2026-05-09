"""Unit tests for the v0.2 self-contained aux_weights.uc format.

Covers:
- serialize -> parse round-trip preserves tensor values + shapes + dtypes
- weight-tied lm_head sentinel handling
- SHA-256 stability under deterministic re-pack
- backward compat: parser rejects bad magic / unsupported version
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from ultracompress.aux_pack import (
    DEFAULT_AUX_KEYS,
    UCAX_MAGIC,
    UCAX_VERSION,
    collect_aux_tensors_from_model,
    load_aux_into_model,
    parse_aux_weights,
    serialize_aux_weights,
    write_aux_weights,
)


def _sample_tensors() -> dict[str, torch.Tensor]:
    """Dict mimicking the embed/norm/lm_head tensors of a small transformer."""
    g = torch.Generator().manual_seed(42)
    return {
        "model.embed_tokens.weight": torch.randn((128, 32), generator=g, dtype=torch.float32).to(torch.bfloat16),
        "model.norm.weight": torch.ones((32,), dtype=torch.bfloat16) * 1.5,
        "lm_head.weight": torch.randn((128, 32), generator=g, dtype=torch.float32).to(torch.bfloat16),
    }


def test_roundtrip_preserves_values_and_dtype(tmp_path: Path) -> None:
    """Write -> read yields bit-exact tensors and unchanged dtypes."""
    src = _sample_tensors()
    out_path = tmp_path / "aux_weights.uc"
    meta = write_aux_weights(out_path, src)

    assert out_path.exists()
    assert out_path.stat().st_size == meta["size_bytes"]
    assert meta["n_tensors"] == 3
    assert sorted(meta["keys"]) == sorted(src.keys())

    parsed = parse_aux_weights(out_path)
    for k, t_in in src.items():
        assert k in parsed, f"key {k} missing from parsed aux"
        t_out = parsed[k]
        assert t_out.dtype == t_in.dtype, f"{k}: dtype changed {t_in.dtype} -> {t_out.dtype}"
        assert t_out.shape == t_in.shape
        assert torch.equal(t_out, t_in), f"{k}: bit-exact round-trip failed"


def test_serialize_is_deterministic() -> None:
    """Same dict -> same bytes (sorted keys, no nondeterministic float ordering)."""
    src = _sample_tensors()
    blob_a = serialize_aux_weights(src)
    blob_b = serialize_aux_weights(src)
    assert blob_a == blob_b, "serialization is not deterministic"
    sha_a = hashlib.sha256(blob_a).hexdigest()
    sha_b = hashlib.sha256(blob_b).hexdigest()
    assert sha_a == sha_b


def test_empty_dict_raises() -> None:
    """Refuse to serialize an empty aux file (would defeat the format's purpose)."""
    with pytest.raises(ValueError, match="cannot be empty"):
        serialize_aux_weights({})


def test_bad_magic_rejected(tmp_path: Path) -> None:
    """Parser rejects files without UCAX magic — protects against pointing
    at a layer.uc file by mistake."""
    bad = tmp_path / "bad.uc"
    bad.write_bytes(b"UCL\x00" + b"\x00" * 100)
    with pytest.raises(ValueError, match="UCAX"):
        parse_aux_weights(bad)


def test_unsupported_version_rejected(tmp_path: Path) -> None:
    """Forward-compat: old v1 loader refuses a future v2 file rather than
    silently mis-parsing."""
    import struct as _struct
    bad = tmp_path / "future.uc"
    # UCAX magic + version=99 + n_tensors=0 + 4 reserved
    bad.write_bytes(UCAX_MAGIC + _struct.pack("<HH", 99, 0) + b"\x00" * 4)
    with pytest.raises(ValueError, match="version 99"):
        parse_aux_weights(bad)


class _TinyModel(torch.nn.Module):
    """Stub that exposes the same state_dict layout as a transformers HF model."""

    def __init__(self, vocab: int = 32, hidden: int = 8, tie_lm_head: bool = False):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(vocab, hidden)
        self.model.norm = torch.nn.LayerNorm(hidden)
        self.lm_head = torch.nn.Linear(hidden, vocab, bias=False)
        if tie_lm_head:
            self.lm_head.weight = self.model.embed_tokens.weight


def test_collect_skips_missing_keys() -> None:
    """If a key isn't in the model state_dict, collect silently skips it
    (defensive — different architectures expose different submodules)."""
    m = _TinyModel()
    out = collect_aux_tensors_from_model(m, keys=("model.embed_tokens.weight", "nonexistent.weight"))
    assert "model.embed_tokens.weight" in out
    assert "nonexistent.weight" not in out
    assert "__tied_lm_head__" not in out  # untied case


def test_collect_marks_tied_lm_head() -> None:
    """Weight-tied lm_head -> sentinel set, no duplicate bytes stored."""
    m = _TinyModel(tie_lm_head=True)
    out = collect_aux_tensors_from_model(m)
    assert "__tied_lm_head__" in out, "tied case must produce sentinel"
    # The state_dict will deduplicate the tied weight, so lm_head.weight is
    # already absent or aliased — the sentinel is the auditable signal.


def test_load_aux_into_model_round_trips() -> None:
    """write -> parse -> load round-trip injects values into a fresh model."""
    src_model = _TinyModel(tie_lm_head=False)
    # Set known sentinel values so we can verify the load.
    with torch.no_grad():
        src_model.model.embed_tokens.weight.fill_(0.5)
        src_model.model.norm.weight.fill_(2.0)
        src_model.lm_head.weight.fill_(0.25)

    aux = collect_aux_tensors_from_model(src_model)
    blob = serialize_aux_weights(aux)
    parsed = parse_aux_weights_from_bytes(blob)

    fresh = _TinyModel(tie_lm_head=False)
    n_loaded = load_aux_into_model(fresh, parsed)
    assert n_loaded == 3
    assert torch.allclose(fresh.model.embed_tokens.weight, torch.full_like(fresh.model.embed_tokens.weight, 0.5))
    assert torch.allclose(fresh.model.norm.weight, torch.full_like(fresh.model.norm.weight, 2.0))
    assert torch.allclose(fresh.lm_head.weight, torch.full_like(fresh.lm_head.weight, 0.25))


def test_load_aux_re_ties_lm_head() -> None:
    """When sentinel is present, load_aux_into_model re-ties lm_head to embed."""
    src_model = _TinyModel(tie_lm_head=True)
    with torch.no_grad():
        src_model.model.embed_tokens.weight.fill_(0.7)
    aux = collect_aux_tensors_from_model(src_model)
    blob = serialize_aux_weights(aux)
    parsed = parse_aux_weights_from_bytes(blob)

    fresh = _TinyModel(tie_lm_head=False)  # start untied
    load_aux_into_model(fresh, parsed)
    # After load, lm_head.weight should be the SAME tensor object as embed
    # (data_ptr identity, not just value equality).
    assert fresh.lm_head.weight.data_ptr() == fresh.model.embed_tokens.weight.data_ptr(), \
        "tied sentinel did not re-tie lm_head to embed_tokens"


def parse_aux_weights_from_bytes(blob: bytes) -> dict[str, torch.Tensor]:
    """Helper: parse_aux_weights but from in-memory bytes (avoids tmp file)."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".uc", delete=False) as f:
        f.write(blob)
        path = Path(f.name)
    try:
        return parse_aux_weights(path)
    finally:
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
