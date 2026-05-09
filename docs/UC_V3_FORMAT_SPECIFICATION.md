# UltraCompress v3 Binary Pack Format — Specification

**Document status:** Normative
**Version:** v3 (file-header `version = 3`)
**Editor:** Sipsa Labs, Inc.
**Date:** 2026-05-08
**Reference implementation:** [`ultracompress/pack_v3.py`](https://github.com/sipsalabs/ultracompress/blob/master/ultracompress/pack_v3.py)
**Audience:** ML systems engineers, format auditors, regulated-industry reviewers, independent researchers reproducing published results.

---

## 1. Scope

This document specifies the byte-level layout of the **UltraCompress v3 layer pack file** (`.uc`). A v3 pack is the on-disk container that stores a single transformer layer's quantized weights, low-rank correction matrices, and supporting tensors in a compact, deterministic, and bit-reversible form.

This specification covers **only the binary container format**. The following are explicitly **out of scope** and are not described in this document:

- The training procedure that produced the weights or the codec.
- The loss function, calibration data, distillation recipe, or any per-Linear bitwidth allocation.
- The Hessian-aware code-assignment heuristics or any internal trainer logic.
- Quantization-aware fine-tuning details.

The format is the audit surface; the trainer is a separate, proprietary surface. Two implementations producing byte-identical v3 files MAY use entirely different training pipelines.

This specification is written so that a competent ML systems engineer can implement an independent reader in any language (C, C++, Rust, Go) using only the contents of this document.

## 2. Notation and Conventions

- All multi-byte integer fields are **little-endian**, encoded as defined by Python's `struct` module with the `<` prefix.
- `u8`, `u16`, `u32` denote unsigned little-endian integers of 1, 2, and 4 bytes.
- `bf16` denotes IEEE 754 bfloat16 (1 sign / 8 exponent / 7 mantissa). Stored as 2 raw bytes; the reader reinterprets the bytes as bfloat16.
- `fp32` denotes IEEE 754 binary32. 4 bytes, little-endian.
- `utf-8` denotes a sequence of bytes interpreted as UTF-8; no terminator, length is given by a preceding `u16`.
- "MUST", "SHOULD", and "MAY" follow RFC 2119 conventions.
- Tables list fields in the order they appear in the byte stream. The `Offset` column is relative to the start of the enclosing record (file or per-Linear blob), with `*` indicating an offset that depends on a preceding variable-length field.

## 3. File Structure Overview

A v3 `.uc` file represents a **single layer** of a transformer model and consists of:

1. A fixed-size **file header** (16 bytes).
2. `n_linears` **per-Linear records**, concatenated.
3. `n_extras` **extras records** (non-quantized tensors such as norms), concatenated.

There is no trailer, no index, no padding between records. Records are read sequentially from the start; the parser walks the byte stream and emits a dictionary of named tensors.

A complete model is represented as a directory of `layer_NNN.uc` files plus a `manifest.json`. The manifest is a separate JSON document (not part of this binary specification) and is described informally in §11.

## 4. File Header (16 bytes)

| Offset | Size | Type | Field | Description |
|--------|------|------|-------|-------------|
| 0 | 4 | bytes | `MAGIC` | ASCII `U`, `C`, `L`, `\x00` (`UC_MAGIC = b"UCL\x00"`). Readers MUST reject the file if the magic does not match exactly. |
| 4 | 2 | u16 | `version` | Format version. For this specification, MUST equal `3`. |
| 6 | 2 | u16 | `layer_idx` | Zero-based transformer layer index (0..65534). Used to identify which layer this file represents in a multi-layer pack. |
| 8 | 2 | u16 | `n_linears` | Number of per-Linear records that follow the header (0..65535). |
| 10 | 2 | u16 | `n_extras` | Number of extras records that follow the per-Linear records (0..65535). |
| 12 | 4 | bytes | `reserved` | Reserved. Writers MUST emit four `\x00` bytes. Readers MUST tolerate any value but SHOULD log a warning if non-zero. |

The header is exactly 16 bytes. After parsing the header, the read cursor is at byte offset 16.

## 5. Per-Linear Record

After the header, exactly `n_linears` per-Linear records appear consecutively. Each record describes one quantized `nn.Linear` module: its name, dimensions, the quantization codec (block size, bits per weight, K-cluster grid), the per-block scales, the bit-packed code stream, the V18-C low-rank correction matrices (V and U), and an optional bias.

### 5.1 Per-Linear Header

The Linear header is variable-length because of the embedded UTF-8 name.

| Offset | Size | Type | Field | Description |
|--------|------|------|-------|-------------|
| 0 | 2 | u16 | `name_len` | Length of `name` in bytes (1..65535). |
| 2 | `name_len` | utf-8 | `name` | Fully qualified Linear name within the layer (e.g., `self_attn.q_proj`, `mlp.down_proj`). No null terminator. MUST be valid UTF-8. |
| 2+`name_len` | 4 | u32 | `out_dim` | Output dimension of the Linear (rows of W). 1..(2^32 - 1). |
| 6+`name_len` | 4 | u32 | `in_dim` | Input dimension of the Linear (columns of W). MUST be a multiple of `block_size` (writers MUST pad upstream if not). |
| 10+`name_len` | 2 | u16 | `block_size` | GSQ block width along the input dimension. Typical value: 64. |
| 12+`name_len` | 1 | u8 | `bpw` | Bits per weight in the packed code stream. Range 1..8. v3 production uses `5`. |
| 13+`name_len` | 1 | u8 | `rank` | Rank of the V18-C low-rank correction (typical: 32). 1..255. |
| 14+`name_len` | 2 | u16 | `K` | Number of clusters in the K-cluster grid. For `bpw = 5` this MUST equal `32`. In general `K` MUST equal `1 << bpw`. |

The per-Linear header is `16 + name_len` bytes. The fields that follow are sized in terms of `out_dim`, `in_dim`, `block_size`, `bpw`, `rank`, and `K`.

For the remainder of §5, "the cursor" refers to the byte offset within the current per-Linear record, starting at `16 + name_len` immediately after the header.

### 5.2 Alpha Scalar

The `alpha` scalar gates the contribution of the V18-C correction term during reconstruction (see §6). v3 stores `alpha` as `fp32`; v2 stored it as `bf16` and was found to lose enough precision to cause measurable downstream perplexity regression, so v3 widened the field.

| Cursor offset | Size | Type | Field | Description |
|---------------|------|------|-------|-------------|
| `cursor` | 4 | fp32 | `alpha` | Single-precision scalar gain on the V18-C correction term. |

After this field the cursor advances by 4 bytes.

### 5.3 K-Cluster Grid

| Cursor offset | Size | Type | Field | Description |
|---------------|------|------|-------|-------------|
| `cursor` | `K * 4` | fp32[K] | `grid` | Per-Linear K-cluster grid, stored as a flat `fp32` array of `K` entries. Indexed directly by codes (see §6 and §7). |

For `K = 32` (the v3 production setting) this is exactly 128 bytes. The grid is per-Linear: writers MUST emit a fresh `grid` for every record.

The cursor advances by `K * 4` bytes.

### 5.4 Per-Block Absmax Scales

Each Linear is divided into `n_blocks` column blocks of width `block_size`:

```
n_blocks = ceil(in_dim / block_size)
```

For each output row and each block, a single fp32 scale (the absolute maximum of the original block, or equivalently a derived per-block normalization factor) is stored.

| Cursor offset | Size | Type | Field | Description |
|---------------|------|------|-------|-------------|
| `cursor` | `out_dim * n_blocks * 4` | fp32[out_dim, n_blocks] | `absmax` | Per-block scale matrix in row-major order: row `i`, block `j` is at element `i * n_blocks + j`. |

Total bytes: `out_dim * n_blocks * 4`. The cursor advances accordingly.

### 5.5 Packed Codes

The integer codes form a flat sequence of length `n_weights = out_dim * in_dim`. Each code is a non-negative integer in `[0, K)`. Codes are bit-packed with `bpw` bits per code and stored in a contiguous byte array. The full bit-packing scheme is specified in §7.

| Cursor offset | Size | Type | Field | Description |
|---------------|------|------|-------|-------------|
| `cursor` | `n_packed_bytes` | u8[n_packed_bytes] | `packed_codes` | Bit-packed codes, where `n_packed_bytes = ceil(n_weights * bpw / 8)`. |

The cursor advances by `n_packed_bytes`.

The decoded code stream is to be reshaped to `(out_dim, n_blocks, block_size)` for indexing the grid (see §6).

### 5.6 V Matrix (V18-C correction, low-rank right factor)

| Cursor offset | Size | Type | Field | Description |
|---------------|------|------|-------|-------------|
| `cursor` | `rank * in_dim * 4` | fp32[rank, in_dim] | `V` | Right factor of the V18-C low-rank correction, row-major. |

`V` is stored as `fp32` (not `bf16`). The cursor advances by `rank * in_dim * 4`.

### 5.7 U Matrix (V18-C correction, low-rank left factor)

| Cursor offset | Size | Type | Field | Description |
|---------------|------|------|-------|-------------|
| `cursor` | `out_dim * rank * 4` | fp32[out_dim, rank] | `U` | Left factor of the V18-C low-rank correction, row-major. |

`U` is stored as `fp32`. The cursor advances by `out_dim * rank * 4`.

### 5.8 Optional Bias

| Cursor offset | Size | Type | Field | Description |
|---------------|------|------|-------|-------------|
| `cursor` | 1 | u8 | `bias_present` | `1` if a bias vector follows; `0` if not. No other values are valid. |
| `cursor + 1` | `out_dim * 2` (if present) | bf16[out_dim] | `bias` | Bias vector stored as `bf16` (raw u16 reinterpreted). Present iff `bias_present == 1`. |

If `bias_present == 0`, the cursor advances by 1 byte and the next per-Linear record (or extras section) begins immediately.

If `bias_present == 1`, the cursor advances by `1 + out_dim * 2` bytes.

This concludes one per-Linear record. The next record (if any) begins at the new cursor position.

### 5.9 Optional AWQ Scale (V4-A Extension)

When the V4-A AWQ-style channel rescaling patch is active (`UC_AWQ_SCALING=1` at compression time), the per-Linear codec dict in the `.pt` layer artifact contains an additional field `awq_scale` of shape `(in_dim,)` in `fp32`. This field records the per-input-channel activation-aware scale vector `s` used during quantization (arxiv:2306.00978).

**Reconstruction with AWQ scale present:**

```
W_base[i, c] = grid[ codes[i, c // B, c % B] ] * absmax[i, c // B] / awq_scale[c]
```

The `awq_scale` field is **optional**: if absent, reconstruction proceeds per the standard formula in S6. If present, the division by `awq_scale` MUST be applied element-wise along the input dimension after the standard `grid * absmax` product.

**Note:** The `awq_scale` field is stored in the per-Linear codec dict of the `.pt` training artifact, not in the binary `.uc` pack file. A future revision of this specification (v4) may promote it to the binary format. For v3, the `.uc` binary layout is unchanged; `awq_scale` travels with the trainer artifact only.

## 6. Reconstruction Formula (Audit Trail)

Given the parsed fields `(grid, codes, absmax, alpha, V, U, bias?)` for one Linear, the original weight matrix `W ∈ R^(out_dim × in_dim)` and effective forward operator are reconstructed as follows.

**Step 1 — Quantized base reconstruction.**

```
W_base[i, c] = grid[ codes[i, c // block_size, c % block_size] ] * absmax[i, c // block_size]
```

Equivalently, with `codes` reshaped to `(out_dim, n_blocks, block_size)`:

```
W_base = (grid[codes] * absmax[:, :, None]).reshape(out_dim, in_dim)
```

The `grid[...]` lookup MUST be performed in `fp32`. The multiplication by `absmax` MUST be performed in `fp32`. The result MAY be cast to `bf16` for downstream use; the on-disk representation is bit-exact regardless.

**Step 2 — V18-C low-rank correction.**

The effective Linear weight used at inference is:

```
W_eff = W_base + alpha * (U @ V)
```

where `U` is `(out_dim, rank)` and `V` is `(rank, in_dim)`. Both `U` and `V` are loaded from the file in `fp32`. The product `U @ V` is computed in `fp32`. The scaling by `alpha` is `fp32 * fp32`.

**Step 3 — Optional bias.**

If `bias` is present (see §5.8), the Linear's forward operation is `y = x @ W_eff.T + bias`. Otherwise `y = x @ W_eff.T`. The bias is reinterpreted from `bf16` to whatever compute precision the runtime uses.

**Audit-trail formula (single line):**

```
W = grid[codes] * absmax + alpha * (U @ V)
```

This is the canonical formula a third-party auditor SHOULD implement to verify a v3 pack against an independently obtained reference weight tensor.

## 7. Bit-Packing Scheme

### 7.1 Code Domain

Each code `c` is a non-negative integer in `[0, K)`, where `K = 1 << bpw`. There are `n_weights = out_dim * in_dim` codes per Linear, in row-major order (row 0 columns 0..in_dim-1, then row 1 columns 0..in_dim-1, and so on).

### 7.2 Signed/Unsigned Offset (Internal Encoding)

For historical compatibility with the v2 packer, the on-disk bit stream represents codes in **signed** form internally. The mapping is:

```
bias = 1 << (bpw - 1)        # for bpw = 5, bias = 16
signed_code   = unsigned_code - bias       # writer
unsigned_code = signed_code   + bias       # reader
```

For `bpw = 5`, `signed_code ∈ [-16, 15]`, `unsigned_code ∈ [0, 31]`. The signed→unsigned offset is identity-preserving and bit-reversible: any well-formed packed stream round-trips to the exact same unsigned codes. Implementations MAY skip the intermediate signed representation entirely, provided they emit the same byte sequence; the bit layout below (§7.3) is the normative interface.

### 7.3 Bit Layout

Codes are written into a flat bit stream, **LSB-first within each code**, and **codes are written sequentially** with no inter-code padding. The stream is then byte-packed **LSB-first**: bit 0 of byte 0 is bit 0 of code 0; bit 1 of byte 0 is bit 1 of code 0; ...; bit `bpw - 1` of byte 0 is the most significant bit of code 0; bit `bpw` of the stream is bit 0 of code 1; and so on.

The total number of bits is `n_weights * bpw`. The total number of bytes is:

```
n_packed_bytes = (n_weights * bpw + 7) // 8
```

The trailing fractional byte (if `n_weights * bpw` is not a multiple of 8) is zero-padded in its high-order bits.

For `bpw = 5`: every group of 8 codes occupies exactly 5 bytes (40 bits). For arbitrary `n_weights`, codes do **not** align to byte boundaries except at multiples of 8.

A reference Python implementation is given in [`ultracompress/pack.py`](https://github.com/sipsalabs/ultracompress/blob/master/ultracompress/pack.py) as `_bitpack` / `_bitunpack`. The implementation uses `numpy.packbits(..., bitorder='little')`; an independent reader MAY use `numpy.unpackbits(..., bitorder='little')` to invert.

## 8. Extras Serialization

Following the `n_linears` per-Linear records, the file contains exactly `n_extras` **extras records**. Extras carry layer-level tensors that are not quantized: layer norms, embeddings (when stored per-layer), routing-gate weights for non-target Linears, and any other non-target state-dict entries the runtime needs.

Each extras record has the following layout.

| Cursor offset | Size | Type | Field | Description |
|---------------|------|------|-------|-------------|
| `cursor` | 2 | u16 | `name_len` | Length of `name` in bytes (1..65535). |
| `cursor + 2` | `name_len` | utf-8 | `name` | Fully qualified tensor name within the layer. |
| `cursor + 2 + name_len` | 1 | u8 | `n_dims` | Number of tensor dimensions (0..255). `0` denotes a scalar. |
| ... | `n_dims * 4` | u32[n_dims] | `dims` | Tensor shape, one `u32` per dimension, in C-order. |
| ... | 1 | u8 | `dtype_tag` | Dtype tag (see §8.1). |
| ... | `n_bytes` | bytes | `raw` | Raw tensor bytes in row-major C-order; size depends on `dtype_tag` and shape (see §8.2). |

### 8.1 Dtype Tags

| Tag | Dtype | Element size |
|-----|-------|--------------|
| 0 | `bfloat16` | 2 bytes |
| 1 | `float32` | 4 bytes |
| 2 | `float16` | 2 bytes |

Writers MUST emit one of the three defined tags. Readers MUST reject unknown tags.

### 8.2 Raw Bytes Sizing

Let `n_elems = product(dims)` (with `n_elems = 1` for a scalar).

- `dtype_tag = 0` (bf16): `n_bytes = n_elems * 2`. Stored as raw `u16` (reinterpret as `bfloat16`).
- `dtype_tag = 1` (fp32): `n_bytes = n_elems * 4`.
- `dtype_tag = 2` (fp16): `n_bytes = n_elems * 2`. Stored as raw `u16` (reinterpret as `float16`).

After the raw bytes, the cursor advances by `n_bytes` and the next extras record (if any) begins immediately.

The end of the last extras record MUST coincide with the end of the file. Any trailing bytes are a format error.

## 9. Determinism Guarantees

For a fixed v3 pack file, the reconstruction:

```
W = grid[codes] * absmax + alpha * (U @ V)
```

produces a **bit-identical** `fp32` tensor across:

1. CPython 3.10+ with NumPy ≥ 1.24 and PyTorch ≥ 2.1.
2. Any CPU architecture supporting IEEE 754 binary32 with round-to-nearest-even semantics (x86-64, aarch64, RISC-V).
3. Linux, Windows, macOS.

Bit-identical reconstruction holds because every operation in the formula is one of:

- A direct table lookup (`grid[codes]`).
- An IEEE 754 binary32 multiplication (`* absmax`, `* alpha`).
- A `fp32` matrix multiplication (`U @ V`) with deterministic accumulation order.
- An IEEE 754 binary32 addition.

The optional cast to `bf16` for downstream forward execution is **not** part of the reconstruction guarantee; readers that require bit-identical bf16 outputs MUST use the same rounding mode as the reference implementation (`torch.Tensor.to(torch.bfloat16)`, which uses round-to-nearest-even).

GPU reconstruction (e.g., on CUDA via PyTorch) is bit-identical for the elementwise lookup and multiplication steps, but the `U @ V` matmul is **not guaranteed** to produce bit-identical results across CUDA kernel selections. Auditors who require strictly bit-identical results across GPUs SHOULD perform the matmul on CPU (`torch.matmul(U.cpu(), V.cpu())`) and compare against the reference.

The reference implementation guarantees that `parse_uc_layer_v3` followed by re-packing via `pack_layer_v3` yields a byte-identical `.uc` file (round-trip stability), provided the input state was itself produced by the v3 trainer.

## 10. Verification Protocol (`uc verify`)

The reference verifier `uc verify <packed_dir>` (see [`ultracompress/verify.py`](https://github.com/sipsalabs/ultracompress/blob/master/ultracompress/verify.py)) performs the following checks. Independent implementations SHOULD perform an equivalent set.

1. **Manifest parse.** Load `<packed_dir>/manifest.json`. Reject if missing or non-JSON.
2. **Pack-version gate.** Read `manifest.uc_pack_version`. If `< 3`, emit a warning that lossless reconstruction is not guaranteed.
3. **Layer file presence.** Enumerate `layer_*.uc` in the directory. Verify the count matches `manifest.n_layers`.
4. **SHA-256 integrity.** For each layer file (or a representative subset in spot-check mode), compute SHA-256 over the entire file in 8 MiB chunks. Compare against manifest-recorded hashes if available, or simply emit the digest for downstream verification.
5. **Structural reconstruction.** Parse a sample layer with `parse_uc_layer_v3`. For each Linear, verify that:
    - `W_base` was reconstructed and has shape `(out_dim, in_dim)`.
    - `V` has shape `(rank, in_dim)`.
    - `U` has shape `(out_dim, rank)`.
    - `bias` (if present) has shape `(out_dim,)`.
6. **NaN/Inf check.** For the reconstructed `W_base`, `V`, `U`, and `bias`, verify there are no NaN or Inf values. Auditors SHOULD apply this check across all layers, not just the sample.

A pack passes verification iff all checks succeed and `uc_pack_version >= 3`.

## 11. Manifest (Informative)

The companion `manifest.json` is a JSON document with the following fields. It is **not part of the binary specification** and is described here for convenience only.

| Field | Type | Description |
|-------|------|-------------|
| `format` | string | Legacy compatibility tag. Always `"uc-pack-v1"`. |
| `uc_pack_version` | int | Same value as the per-file `version` field. MUST be `3` for v3 packs. |
| `vu_dtype` | string | `"fp32"` for v3. |
| `codec_source` | string | `"trainer-persisted"` for v3. v2 used `"reverse-derived"`. |
| `bpw` | int | Bits per weight. Typical: `5`. |
| `block_size` | int | GSQ block width. Typical: `64`. |
| `n_layers` | int | Number of `layer_NNN.uc` files in the directory. |
| `total_input_bytes`, `total_output_bytes`, `overall_shrink_ratio` | int / float | Aggregate sizing. |
| `layers` | array | Per-layer sub-records: `{layer_idx, n_linears_packed, n_extras_packed, output_size_bytes, linears: [...], extras: [...]}`. |

## 12. Backward Compatibility

### 12.1 v2 → v3 Differences

The v3 format is **not** a superset of v2. A v3 reader SHOULD reject files where the file-header `version` is not exactly `3`. A v2 reader cannot parse v3 files because the per-Linear record layout differs.

The semantic differences from v2:

- **`alpha` widened from bf16 to fp32.** v2's bf16 `alpha` lost enough precision to cause a measurable PPL regression on Qwen3-1.7B (PPL_r ≈ 1.13 in the worst case). v3 uses fp32, eliminating this regression.
- **V/U widened from bf16 to fp32.** Identified 2026-05-07 as the cause of a PPL_r ≈ 1.22 regression. The format-overhead cost is approximately 1% (V/U total < 2 MB per layer at `rank = 32`).
- **`grid` is stored explicitly.** v2 reverse-derived the grid from the dequantized `W_base`, assuming a uniform symmetric grid `{-15, ..., 15} / 15`. v3 stores the trainer-learned K-cluster grid directly (`grid_K * 4` bytes per Linear), which makes round-trip lossless when the trainer used a learned (k-means) grid rather than the uniform grid.
- **`codes` are stored as the trainer's actual unsigned indices** (after the signed-offset round-trip, see §7.2), not re-derived. This eliminates the v2 dequant-then-requant lossy step.

### 12.2 Detection

To detect v3 vs v2 from the file header alone, read bytes 4..6 as a little-endian `u16`. v2 files have `version = 2`; v3 files have `version = 3`. The magic and header layout are otherwise identical (16 bytes total), but per-Linear records are not interchangeable.

### 12.3 Migration

There is no in-place conversion from v2 to v3. v2 packs were produced by trainers that did not persist `gsq_codecs`; the original learned grid is unrecoverable from the v2 file. To produce a v3 pack, the model MUST be re-compressed with the v3 trainer (which persists `gsq_codecs` to `state['gsq_codecs']` in each `layer_NNN.pt` artifact). Re-compressing typically takes a few hours per 8B-class model on a single GPU.

## 13. Format Limits

| Quantity | Encoded as | Max value | Practical implication |
|----------|------------|-----------|----------------------|
| `version` | u16 | 65 535 | Room for many future format revisions. |
| `layer_idx` | u16 | 65 535 | Up to 65 535 layers per model. |
| `n_linears` per layer | u16 | 65 535 | Far exceeds any current architecture (e.g., dense Qwen3-72B has 7 target Linears per layer). |
| `n_extras` per layer | u16 | 65 535 | Same. |
| `name_len` (per-Linear and extras) | u16 | 65 535 bytes | Sufficient for any realistic UTF-8 module name. |
| `out_dim`, `in_dim` | u32 | 4 294 967 295 | Up to ~4 billion per dimension; far exceeds any plausible Linear. |
| `block_size` | u16 | 65 535 | Typical: 64. |
| `bpw` | u8 | 255; effective range 1..8 | `K = 1 << bpw`, so `bpw <= 8` keeps `K` in u16 range. v3 production uses `5`. |
| `rank` | u8 | 255 | Typical: 32. |
| `K` | u16 | 65 535 | Fixed at `32` for v3 (`bpw = 5`). MUST equal `1 << bpw`. |
| `n_dims` (extras) | u8 | 255 | Trivial bound for any tensor. |
| `dtype_tag` (extras) | u8 | enumerated: 0, 1, 2 | bf16, fp32, fp16. Other values are format errors. |

## 14. Conformance

A reader is conformant with this specification if it correctly parses any well-formed v3 file and produces, for every Linear, the reconstructed `W_base` tensor described in §6 with bit-exact `fp32` results.

A writer is conformant if every `.uc` file it produces is parsable by a conformant reader and yields the same in-memory state (codes, grid, absmax, V, U, alpha, bias) as the writer recorded.

Test vectors and a reference parser are provided in the reference implementation linked at the top of this document.

---

## Appendix A — Python Reference Decoder

The following Python reads a v3 `.uc` layer file and returns one decoded Linear. It mirrors `parse_uc_layer_v3` in `ultracompress/pack_v3.py` and is provided as normative pseudocode.

```python
import struct
import numpy as np
import torch

UC_MAGIC = b"UCL\x00"
UC_VERSION_V3 = 3

DTYPE_TAGS = {0: torch.bfloat16, 1: torch.float32, 2: torch.float16}

def bitunpack(packed: np.ndarray, n_weights: int, bpw: int) -> np.ndarray:
    """Inverse of bit-pack. Returns int8 codes in signed range [-K/2, K/2-1]."""
    bias = 1 << (bpw - 1)
    bits = np.unpackbits(packed.astype(np.uint8), bitorder="little")
    bits = bits[: n_weights * bpw].reshape(n_weights, bpw)
    pow2 = (1 << np.arange(bpw, dtype=np.uint16)).astype(np.uint16)
    codes_unsigned = (bits.astype(np.uint16) * pow2).sum(axis=1)
    return (codes_unsigned.astype(np.int16) - bias).astype(np.int8)

def parse_uc_layer_v3(path):
    buf = open(path, "rb").read()
    o = 0

    # ---- File header (16 bytes) ----
    assert buf[o:o + 4] == UC_MAGIC, "bad magic"
    o += 4
    version, layer_idx, n_linears, n_extras = struct.unpack("<HHHH", buf[o:o + 8])
    o += 8 + 4  # skip 4 reserved bytes
    assert version == UC_VERSION_V3, f"expected v3, got {version}"

    out = {"__version__": version, "__layer_idx__": layer_idx}

    # ---- Per-Linear records ----
    for _ in range(n_linears):
        (name_len,) = struct.unpack("<H", buf[o:o + 2]); o += 2
        name = buf[o:o + name_len].decode("utf-8"); o += name_len
        out_dim, in_dim = struct.unpack("<II", buf[o:o + 8]); o += 8
        block_size, bpw, rank = struct.unpack("<HBB", buf[o:o + 4]); o += 4
        (K,) = struct.unpack("<H", buf[o:o + 2]); o += 2

        # alpha (fp32)
        alpha = float(np.frombuffer(buf[o:o + 4], dtype=np.float32)[0]); o += 4

        # grid (K fp32)
        grid = torch.from_numpy(
            np.frombuffer(buf[o:o + K * 4], dtype=np.float32).copy()
        )
        o += K * 4

        # absmax (out_dim * n_blocks fp32)
        n_blocks = (in_dim + block_size - 1) // block_size
        absmax = torch.from_numpy(
            np.frombuffer(buf[o:o + out_dim * n_blocks * 4], dtype=np.float32).copy()
        ).view(out_dim, n_blocks)
        o += out_dim * n_blocks * 4

        # packed codes
        n_weights = out_dim * in_dim
        n_packed = (n_weights * bpw + 7) // 8
        packed = np.frombuffer(buf[o:o + n_packed], dtype=np.uint8).copy()
        o += n_packed
        bias_off = 1 << (bpw - 1)
        codes_signed = bitunpack(packed, n_weights, bpw)
        codes = torch.from_numpy(
            (codes_signed.astype(np.int16) + bias_off).astype(np.int16)
        ).reshape(out_dim, n_blocks, block_size)

        # V (rank × in_dim fp32), U (out_dim × rank fp32)
        V = torch.from_numpy(
            np.frombuffer(buf[o:o + rank * in_dim * 4], dtype=np.float32).copy()
        ).view(rank, in_dim)
        o += rank * in_dim * 4
        U = torch.from_numpy(
            np.frombuffer(buf[o:o + out_dim * rank * 4], dtype=np.float32).copy()
        ).view(out_dim, rank)
        o += out_dim * rank * 4

        # bias
        (bias_present,) = struct.unpack("<B", buf[o:o + 1]); o += 1
        bias_t = None
        if bias_present:
            bias_arr = np.frombuffer(buf[o:o + out_dim * 2], dtype=np.uint16).copy()
            bias_t = torch.from_numpy(bias_arr).view(torch.bfloat16)
            o += out_dim * 2

        # Reconstruct W (audit-trail formula)
        W_base = (grid[codes.long()] * absmax.unsqueeze(-1)).reshape(out_dim, in_dim)
        # W_eff = W_base + alpha * (U @ V)  -- compute lazily at inference time

        out[name] = dict(
            alpha=alpha, W_base=W_base, V=V, U=U, bias=bias_t,
            grid=grid, codes=codes, absmax=absmax,
            out_dim=out_dim, in_dim=in_dim,
            block_size=block_size, bpw=bpw, rank=rank, K=K,
        )

    # ---- Extras ----
    extras = {}
    for _ in range(n_extras):
        (name_len,) = struct.unpack("<H", buf[o:o + 2]); o += 2
        name = buf[o:o + name_len].decode("utf-8"); o += name_len
        (n_dims,) = struct.unpack("<B", buf[o:o + 1]); o += 1
        dims = []
        for __ in range(n_dims):
            (d,) = struct.unpack("<I", buf[o:o + 4]); o += 4
            dims.append(d)
        (dtype_tag,) = struct.unpack("<B", buf[o:o + 1]); o += 1
        dtype = DTYPE_TAGS[dtype_tag]
        n_elems = 1
        for d in dims:
            n_elems *= d
        if dtype == torch.float32:
            arr = np.frombuffer(buf[o:o + n_elems * 4], dtype=np.float32).copy()
            t = torch.from_numpy(arr).view(*dims) if dims else torch.from_numpy(arr).reshape(())
            o += n_elems * 4
        else:
            arr = np.frombuffer(buf[o:o + n_elems * 2], dtype=np.uint16).copy()
            t = torch.from_numpy(arr).view(dtype)
            t = t.view(*dims) if dims else t.reshape(())
            o += n_elems * 2
        extras[name] = t

    out["__extras__"] = extras
    assert o == len(buf), "trailing bytes after extras"
    return out
```

## Appendix B — C-like Pseudocode for Downstream Ports

The following pseudocode is suitable for porting to C, C++, Rust, Go, or any language with raw-byte and IEEE 754 support. It assumes a little-endian host (the format itself is little-endian; on a big-endian host, byte-swap each multi-byte field after reading).

```c
/* ---- File header ---- */
struct uc_file_header {
    uint8_t  magic[4];       /* "UCL\0" */
    uint16_t version;        /* MUST equal 3 */
    uint16_t layer_idx;
    uint16_t n_linears;
    uint16_t n_extras;
    uint8_t  reserved[4];    /* writers emit zeros */
};                            /* sizeof = 16 */

/* ---- Per-Linear header (fields in stream order, NOT a packed struct) ---- */
/*   uint16_t name_len;                                                     */
/*   uint8_t  name[name_len];           // utf-8, no NUL                    */
/*   uint32_t out_dim;                                                      */
/*   uint32_t in_dim;                   // multiple of block_size           */
/*   uint16_t block_size;                                                   */
/*   uint8_t  bpw;                      // 1..8                             */
/*   uint8_t  rank;                                                         */
/*   uint16_t K;                        // == (1 << bpw)                    */

/* ---- Per-Linear payload (in stream order) ---- */
/*   float    alpha;                            // 4 bytes                  */
/*   float    grid[K];                          // K * 4 bytes              */
/*   float    absmax[out_dim][n_blocks];        // out_dim*n_blocks*4 bytes */
/*   uint8_t  packed_codes[n_packed_bytes];                                 */
/*   float    V[rank][in_dim];                                              */
/*   float    U[out_dim][rank];                                             */
/*   uint8_t  bias_present;                                                 */
/*   uint16_t bias[out_dim];   // bf16 raw bits, only if bias_present == 1  */

/* where:                                                                   */
/*   n_blocks       = (in_dim + block_size - 1) / block_size                */
/*   n_weights      = (size_t)out_dim * (size_t)in_dim                      */
/*   n_packed_bytes = (n_weights * bpw + 7) / 8                             */

/* ---- Bit unpack (LSB-first, sequential codes) ---- */
static void bit_unpack(const uint8_t *packed, size_t n_weights,
                       int bpw, uint8_t *codes_out /* >= n_weights */) {
    int bias = 1 << (bpw - 1);
    size_t bit_cursor = 0;
    int mask = (1 << bpw) - 1;
    for (size_t w = 0; w < n_weights; ++w) {
        unsigned int v = 0;
        for (int b = 0; b < bpw; ++b) {
            size_t byte_i = (bit_cursor + b) >> 3;
            int    bit_i  = (bit_cursor + b) & 7;
            unsigned int bit = (packed[byte_i] >> bit_i) & 1u;
            v |= bit << b;
        }
        v &= mask;
        /* on-stream encoding is signed; recover unsigned index (see Sec 7.2) */
        int signed_code = (int)v - bias;
        codes_out[w] = (uint8_t)(signed_code + bias);   /* == v */
        bit_cursor += bpw;
    }
}

/* ---- Reconstruction (audit-trail formula) ---- */
/*                                                                          */
/* For each row i in [0, out_dim):                                          */
/*   For each col c in [0, in_dim):                                         */
/*     int b = c / block_size;                                              */
/*     int k = c % block_size;                                              */
/*     uint8_t code = codes[i * in_dim + c];      // == codes[i][b][k]      */
/*     float w_base_ic = grid[code] * absmax[i][b];                         */
/*     /* W_base[i][c] = w_base_ic */                                       */
/*                                                                          */
/* Then for the effective weight at inference:                              */
/*   W_eff[i][j] = W_base[i][j] + alpha * sum_r U[i][r] * V[r][j]           */
/*                                                                          */
/* If bias_present:                                                         */
/*   y[i] = W_eff[i] @ x + bf16_to_fp32(bias[i])                            */

/* ---- Extras record (in stream order) ---- */
/*   uint16_t name_len;                                                     */
/*   uint8_t  name[name_len];                                               */
/*   uint8_t  n_dims;                                                       */
/*   uint32_t dims[n_dims];                                                 */
/*   uint8_t  dtype_tag;     // 0=bf16, 1=fp32, 2=fp16                      */
/*   uint8_t  raw[n_bytes];  // n_bytes = product(dims) * elem_size(dtype)  */
```

---

## License

The **format specification** in this document is licensed under the **Apache License 2.0**. Independent implementations of v3 readers and writers MAY freely incorporate any portion of this specification.

The **reference codec implementation** (`ultracompress/pack_v3.py`, `ultracompress/pack.py`, `ultracompress/verify.py`) is distributed under the **Sipsa Labs Research Evaluation License v1.0**. See the project root for the full license text.

## References

- Reference packer/parser source: <https://github.com/sipsalabs/ultracompress/blob/master/ultracompress/pack_v3.py>
- Reference verifier source: <https://github.com/sipsalabs/ultracompress/blob/master/ultracompress/verify.py>
- Bit-packing primitives: <https://github.com/sipsalabs/ultracompress/blob/master/ultracompress/pack.py>

## Patent Disclosure

The compression methodology associated with this format is the subject of:

- USPTO provisional patent application **64/049,511** (filed 2026-04-25)
- USPTO provisional patent application **64/049,517** (filed 2026-04-25)
- Five supplementary USPTO provisional patent applications (filed 2026-05-09)

Patent claims cover the training methodology, codec construction, per-Linear bitwidth allocation, and related techniques — **not** the binary format described in this document. Independent implementations of the binary format (readers, writers, verifiers) are unaffected by these patents.
