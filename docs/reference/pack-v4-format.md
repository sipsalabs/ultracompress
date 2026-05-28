# uc-pack-v4 format reference

extends uc-pack-v3 with support for Conv2d and Conv1d weight tensors. backward-compatible: v4 readers handle v3 files; v3 readers reject v4 files (version field mismatch).

## motivation

v3 compresses `nn.Linear` weight matrices (2D tensors). diffusion models (SDXL, Flux, ControlNet) and audio frontends (Whisper encoder) contain `nn.Conv2d` and `nn.Conv1d` layers that are 4D and 3D tensors respectively. v4 adds a well-defined reshape-quantize-restore pipeline for these layer types.

## schema diff from v3

### manifest (`ultracompress.json`)

new fields per packed layer entry:

| field | type | description |
|---|---|---|
| `layer_type` | string | `"Linear"`, `"Conv2d"`, or `"Conv1d"` |
| `layer_type_id` | integer | `0` = Linear, `1` = Conv2d, `2` = Conv1d |
| `original_shape` | array[int] | original tensor shape before reshape (e.g. `[512, 256, 3, 3]` for Conv2d) |
| `reshaped_shape` | array[int] | the 2D shape used for quantization (e.g. `[512, 2304]`) |

existing fields (`out_dim`, `in_dim`, `block_size`, `bpw`, `rank`, `codec`) remain unchanged and describe the reshaped 2D form.

### binary per-layer record

extends the v3 per-weight-module record with:

| offset (relative to name field end) | size | field | notes |
|---|---|---|---|
| +0 | 1 byte | `layer_type` (u8) | 0/1/2 per above |
| +1 | as before | `out_dim` (u32), `in_dim` (u32) | these are the reshaped 2D dims |
| +9 | 1 byte | `original_ndim` (u8) | 2 for Linear, 3 for Conv1d, 4 for Conv2d |
| +10 | 4 * ndim bytes | `original_shape` (u32 per dim) | only present when ndim > 2 |

remainder of the record (the codec-internal payload) is identical to v3. Refer to the v3 binary format specification (NDA-gated; available to partners under engagement).

### format version

| field | v3.6 value | v4 value |
|---|---|---|
| `pack_format_version` | `"3.6"` | `"4.0"` |
| binary version (u16 in file header) | `4` | `5` |

## Conv2d quantization pipeline

1. **reshape:** `(out_c, in_c, kH, kW)` -> `(out_c, in_c * kH * kW)`
2. **block size selection:**
   - 1x1 kernel: standard block sizing (128 if divisible, else 64, else per-row)
   - 3x3+ kernel: `block_size = in_channels` (kernel-aligned scaling)
   - if `in_channels > 256`: largest divisor of `in_channels` that is <= 256
3. **quantize:** standard per-block symmetric scalar quantization on the 2D form
4. **correction:** low-rank `V @ U` correction on the 2D form (same as Linear)
5. **pack:** serialize the 2D quantized form + reshape metadata
6. **restore:** on load, dequantize the 2D form, reshape back to `(out_c, in_c, kH, kW)`

Conv1d follows the same pipeline with `(out_c, in_c, k)` -> `(out_c, in_c * k)`.

## block size rationale

for a 3x3 conv with `in_channels=256`, flattening produces rows of length `256 * 9 = 2304`. a standard block-128 would group elements from different spatial positions (different kernel taps), mixing scales. setting `block_size = in_channels = 256` means each block contains all channels at one spatial position, preserving per-position dynamic range.

empirical result: kernel-aligned block-256 on SDXL UNet's largest Conv2d (1280x2560x3x3) gives relative error 0.058, vs 0.208 for per-row quantization (3.6x improvement).

## backward compatibility

- v4 reader checks binary version field. if version <= 4 (v3.6), reads as v3 (all layers assumed Linear).
- v3 reader encountering version 5 (v4) should reject gracefully.
- manifest `layer_type` field is additive; v3 manifests implicitly have `layer_type = "Linear"` for all entries.

## implementation

internal: `.private/pack_v4.py`
public schema only: this document.
