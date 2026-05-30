# uc-pack-v4 format reference

uc-pack-v4 extends uc-pack-v3 to cover convolutional weight tensors (Conv2d / Conv1d) in
addition to the Linear tensors handled by v3. This lets the format support diffusion models
(e.g. SDXL, Flux, ControlNet) and audio frontends (e.g. Whisper encoders), whose weights are
3D / 4D rather than 2D.

## compatibility

- Backward-compatible: v4 readers handle v3 files; v3 readers reject v4 files via a
  version-field mismatch (graceful failure).
- The manifest (`ultracompress.json`) gains additive, descriptive fields for convolutional
  layers (a layer-type tag and the original tensor shape). v3 manifests are treated as all-Linear.
- `pack_format_version` advances to `"4.0"`.

## public vs. NDA-gated

This document describes only the *existence* and *compatibility* of the v4 format. The per-layer
binary payload layout and the compression / reconstruction pipeline are part of Sipsa Labs'
patent-pending codec and are **NDA-gated** — available to partners under engagement, not
published here.

The public CLI reads v4 packs transparently via `uc verify` / `uc inspect`; consumers never
need the internal layout.

## verification

`uc verify <pack>` and `uc inspect <pack>` operate on v4 packs exactly as on v3 — structure +
per-file SHA-256 + pack fingerprint. See `docs/reference/manifest-schema.md` for the public
manifest surface.
