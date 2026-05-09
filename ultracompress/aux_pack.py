"""UltraCompress v0.2 self-contained format — `aux_weights.uc` builder/reader.

Solves the v3 dependency problem: customer downloads a packed dir, but to
reconstruct an inference-ready model they ALSO need `embed_tokens.weight`,
`model.norm.weight`, and `lm_head.weight` from the original bf16 safetensors
on HuggingFace. That defeats the compression benefit on first-download.

v3.5 fix: pack the model-level non-Linear weights into a single `aux_weights.uc`
file at the pack root. Customer downloads ONE directory, runs reconstruction,
gets a working model — no second HF download required.

File format (`aux_weights.uc`):
  MAGIC(4 bytes "UCAX")
  version(u16, =1)
  n_tensors(u16)
  reserved(u32)
  Then n_tensors blobs, each:
    name_len(u16) + name(utf-8) +
    n_dims(u8) + dims(u32 each) +
    dtype_tag(u8) + raw_bytes
  (Reuses `_serialize_extra` from `ultracompress.pack` so dtype handling
  matches per-layer extras.)

Backward compatibility:
- v3 packs without `aux_weights.uc` continue to load via the existing
  HF-download fallback path. The new manifest field `aux_file` is None for
  legacy packs and the loader treats absence as "fall back to HF".
- The aux file gets its own SHA-256 in the manifest, joining the existing
  cryptographic-provenance chain used by `uc verify`.
"""
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .pack import UC_MAGIC, _DTYPE_TAGS, _TAG_DTYPES, _serialize_extra

# Magic + version for the aux file. Distinct from UC_MAGIC ("UCL\x00") so a
# truncated/corrupted layer file is never mis-parsed as an aux file.
UCAX_MAGIC = b"UCAX"
UCAX_VERSION = 1

# Default model-level keys we always pull from a transformers HF model.
# Order matters only for deterministic packing (same hash on re-pack).
DEFAULT_AUX_KEYS: tuple[str, ...] = (
    "model.embed_tokens.weight",
    "model.norm.weight",
    "lm_head.weight",
)


def serialize_aux_weights(tensors: dict[str, torch.Tensor]) -> bytes:
    """Serialize a {name: tensor} dict into the aux_weights.uc binary blob.

    Uses `_serialize_extra` per tensor for dtype-tagged framing identical to
    the per-layer extras format. Tensors are written in sorted key order so
    re-packing the same dict yields a byte-identical file (hash-stable).
    """
    if not tensors:
        raise ValueError("aux_weights cannot be empty — pass at least 1 tensor")

    body_blobs: list[bytes] = [
        _serialize_extra(name, t) for name, t in sorted(tensors.items())
    ]
    n_tensors = len(body_blobs)

    header = (
        UCAX_MAGIC
        + struct.pack("<HH", UCAX_VERSION, n_tensors)
        + b"\x00" * 4  # reserved
    )
    return header + b"".join(body_blobs)


def write_aux_weights(out_path: Path, tensors: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Write tensors to `out_path` and return packing metadata.

    Returns:
      {
        'path': str,
        'size_bytes': int,
        'sha256': str (hex),
        'n_tensors': int,
        'keys': sorted list of str,
        'version': UCAX_VERSION,
      }
    """
    blob = serialize_aux_weights(tensors)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(blob)

    return {
        "path": out_path.name,
        "size_bytes": len(blob),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "n_tensors": len(tensors),
        "keys": sorted(tensors.keys()),
        "version": UCAX_VERSION,
    }


def parse_aux_weights(aux_path: Path) -> dict[str, torch.Tensor]:
    """Parse an aux_weights.uc binary file back into a {name: tensor} dict.

    Raises ValueError on bad magic / unsupported version. Returns tensors on
    CPU; caller is responsible for moving them to the inference device.
    """
    buf = aux_path.read_bytes()
    offset = 0

    magic = buf[offset:offset + 4]
    offset += 4
    if magic != UCAX_MAGIC:
        raise ValueError(
            f"Not a UCAX aux file: {aux_path} (magic={magic!r}, expected {UCAX_MAGIC!r})"
        )

    (version, n_tensors) = struct.unpack("<HH", buf[offset:offset + 4])
    offset += 4
    offset += 4  # reserved

    if version != UCAX_VERSION:
        raise ValueError(
            f"Unsupported UCAX version {version} in {aux_path} "
            f"(this loader supports v{UCAX_VERSION})"
        )

    out: dict[str, torch.Tensor] = {}
    for _ in range(n_tensors):
        (name_len,) = struct.unpack("<H", buf[offset:offset + 2])
        offset += 2
        name = buf[offset:offset + name_len].decode("utf-8")
        offset += name_len

        (n_dims,) = struct.unpack("<B", buf[offset:offset + 1])
        offset += 1
        dims: list[int] = []
        for __ in range(n_dims):
            (d,) = struct.unpack("<I", buf[offset:offset + 4])
            offset += 4
            dims.append(d)

        (dtype_tag,) = struct.unpack("<B", buf[offset:offset + 1])
        offset += 1
        dtype = _TAG_DTYPES[dtype_tag]

        n_elems = 1
        for d in dims:
            n_elems *= d

        if dtype == torch.float32:
            n_bytes = n_elems * 4
            arr = np.frombuffer(buf[offset:offset + n_bytes], dtype=np.float32).copy()
            t = (
                torch.from_numpy(arr).view(*dims)
                if dims
                else torch.from_numpy(arr).reshape(())
            )
        else:  # bf16 / fp16 — 2 bytes
            n_bytes = n_elems * 2
            arr = np.frombuffer(buf[offset:offset + n_bytes], dtype=np.uint16).copy()
            t = (
                torch.from_numpy(arr).view(dtype).view(*dims)
                if dims
                else torch.from_numpy(arr).view(dtype).reshape(())
            )
        offset += n_bytes
        out[name] = t

    return out


def collect_aux_tensors_from_model(
    model: torch.nn.Module, keys: tuple[str, ...] | list[str] = DEFAULT_AUX_KEYS,
) -> dict[str, torch.Tensor]:
    """Pull the named tensors from a loaded HF model's state_dict.

    Skips keys that don't exist in the model (e.g. weight-tied lm_head where
    `lm_head.weight` may be aliased to `model.embed_tokens.weight`). Tensors
    are detached, moved to CPU, and copied (so the caller can free the model
    immediately). Tensors are stored in their NATIVE dtype to preserve
    precision — bf16 inputs stay bf16 in the aux file.

    Weight-tied lm_head: many small models (Qwen3-1.7B-Base, SmolLM2,
    Phi-3-Mini, Llama3 small) tie `lm_head.weight` to `embed_tokens.weight`.
    PyTorch's `state_dict()` exposes BOTH keys with the same `data_ptr`. We
    detect this via pointer equality on the live state_dict, drop the
    duplicate from disk, and emit a `__tied_lm_head__` sentinel so the loader
    can re-tie on reload (no duplicate bytes stored).
    """
    sd = model.state_dict()
    embed_key = "model.embed_tokens.weight"
    head_key = "lm_head.weight"

    # Detect tied case BEFORE cloning (clone breaks data_ptr equality).
    is_tied = (
        embed_key in sd and head_key in sd
        and sd[embed_key].data_ptr() == sd[head_key].data_ptr()
    )

    out: dict[str, torch.Tensor] = {}
    for k in keys:
        if k not in sd:
            continue
        # Skip the tied lm_head copy — we'll re-tie on load via sentinel.
        if is_tied and k == head_key:
            continue
        out[k] = sd[k].detach().cpu().clone()

    if is_tied and embed_key in out:
        out["__tied_lm_head__"] = torch.tensor([1], dtype=torch.uint8)

    return out


def load_aux_into_model(
    model: torch.nn.Module, aux_tensors: dict[str, torch.Tensor],
) -> int:
    """Inject aux tensors into a fresh HF model skeleton in-place.

    Returns the number of tensors actually loaded. Re-ties `lm_head.weight`
    to `embed_tokens.weight` when the `__tied_lm_head__` sentinel is present
    in the aux dict.

    Note: re-tying happens AFTER load_state_dict so the lm_head Parameter
    object is reassigned (not just its data copied), preserving the
    aliasing relationship that downstream code may rely on for memory
    accounting and gradient sharing.
    """
    sd = model.state_dict()
    n_loaded = 0
    tied = "__tied_lm_head__" in aux_tensors
    for k, t in aux_tensors.items():
        if k.startswith("__"):
            continue
        if k not in sd:
            continue
        target = sd[k]
        # Match dtype/device of the destination parameter so the load is a
        # straight copy (no extra cast).
        sd[k] = t.to(device=target.device, dtype=target.dtype)
        n_loaded += 1
    model.load_state_dict(sd, strict=False)

    if tied:
        # Walk the module tree to find embed_tokens (commonly at
        # `model.embed_tokens`, but transformers occasionally reorganizes —
        # e.g., Mixtral-8x7B exposes it the same way; SSMs may not).
        embed = None
        for parent_name in ("model", "transformer", "backbone"):
            parent = getattr(model, parent_name, None)
            if parent is not None:
                embed = getattr(parent, "embed_tokens", None) or getattr(parent, "wte", None)
                if embed is not None:
                    break
        head = getattr(model, "lm_head", None)
        if embed is not None and head is not None and hasattr(head, "weight"):
            # Reassign the Parameter OBJECT (not just its data) so the alias
            # is preserved at the attribute-reference level.
            head.weight = embed.weight  # type: ignore[assignment]

    return n_loaded


__all__ = [
    "UCAX_MAGIC",
    "UCAX_VERSION",
    "DEFAULT_AUX_KEYS",
    "serialize_aux_weights",
    "write_aux_weights",
    "parse_aux_weights",
    "collect_aux_tensors_from_model",
    "load_aux_into_model",
]
