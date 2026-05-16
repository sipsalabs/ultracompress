# `ultracompress.json` — manifest schema reference

Every UltraCompress artifact ships with a JSON manifest at `<artifact>/ultracompress.json`. The manifest is the authoritative record of what was compressed, how, and at what fidelity.

## Schema (v1.0)

```json
{
  "schema_version": "1.0",
  "model_id": "string (HF Hub repo id)",
  "base_model": "string (HF Hub repo id of the FP16 source)",
  "method": "string (uc-v3 lossless 5-bit pack)",
  "method_version": "string (semver)",
  "bpw": "number (effective bits per weight, includes any overhead)",
  "size_bytes": "integer (total artifact size on disk)",
  "files": {
    "<filename>": {
      "sha256": "string (hex digest of the file)",
      "size_bytes": "integer"
    }
  },
  "license": "string (license identifier)",
  "patents": ["array of strings"],
  "created_at": "string (ISO-8601 timestamp)",
  "tooling_version": "string (which internal tool produced this artifact)",
  "notes": "string (optional, human-readable)"
}
```

## Field reference

### `schema_version`

The version of THIS schema. Currently `1.0`. Bumped when we make a breaking change to the manifest structure.

### `model_id`

Hugging Face Hub repository ID of this artifact. Format: `<org>/<name>`. Example: `sipsalabs/<model-id>`.

### `base_model`

The FP16 source model from which this artifact was derived. Example: `Qwen/Qwen3-1.7B`.

### `method`

The compression method applied. Currently:

| Value | Description |
|---|---|
| `uc-v3` | Patent-pending lossless 5-bit pack format (bit-identical reconstruction) |

Future methods will be added as new strings; readers should accept unknown values gracefully. Internal method specifics are NDA-gated.

### `method_version`

Semver version of the method. Bumped when the method itself changes (not just the pipeline that runs it). Example: `1.2.0`.

### `bpw`

Effective bits per weight. This is the **total artifact size in bits** divided by **the number of weights in the model** — including any overhead for codebooks, scales, zero-points, etc. Round to 3 decimal places.

Example: a 1.7B-parameter model at 5 bpw means total artifact is approximately `1.7e9 × 5 / 8 = 1.06 GB` (plus tokenizer, config files, etc.).

### `size_bytes`

Total bytes on disk for the artifact directory.

### `files`

A dictionary mapping filename → SHA-256 + size. Every meaningful file in the artifact (excluding the manifest itself) is listed. Used for tamper detection.

### `license`

License identifier. Currently one of:

| Value | Meaning |
|---|---|
| `sipsalabs-research-eval-1.0` | Sipsa Labs Research and Evaluation License v1.0 |
| `sipsalabs-commercial` | Per-customer commercial license (terms in the contract, not the manifest) |

### `patents`

Patent status of the method. Always populated. Example: `["patent pending"]`.

### `created_at`

ISO-8601 timestamp of when the artifact was produced. UTC.

### `tooling_version`

Identifier of the internal tool that produced this artifact. Used for traceability when bugs surface. Example: `ultracompress-internal-publishing/0.3.2`.

### `notes`

Optional free-form text. Used for human-readable annotations like "rebuilt on 2026-05-12 to fix tokenizer hash mismatch."

## Validation

The CLI verifies the manifest at `uc info` time:

1. Schema is valid JSON
2. All declared files exist in the artifact directory
3. SHA-256 of each declared file matches the manifest
4. `bpw` × `weight_count` ≈ `model.safetensors` size (with some tolerance for codebook overhead)

If any verification fails, `uc info` prints a warning but does not exit non-zero (the manifest may still be useful even if files are partially corrupt).

## Forward compatibility

When you encounter a manifest with `schema_version` > what your CLI version supports:

1. The CLI will print a warning
2. Known fields will still be parsed and displayed
3. Unknown fields will be passed through in `--json` mode

We commit to never removing or repurposing existing fields within a major schema version.

## Example

```json
{
  "schema_version": "1.0",
  "model_id": "sipsalabs/<model-id>",
  "base_model": "Qwen/Qwen3-1.7B",
  "method": "uc-v3",
  "method_version": "3.0.0",
  "bpw": 5,
  "size_bytes": 1116691072,
  "files": {
    "model.safetensors": {
      "sha256": "a3f5c8b9d2e1f0a4b7c6d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9",
      "size_bytes": 1109953968
    },
    "config.json": {
      "sha256": "b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5",
      "size_bytes": 4276
    },
    "tokenizer.json": {
      "sha256": "c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6",
      "size_bytes": 2734851
    }
  },
  "license": "sipsalabs-research-eval-1.0",
  "patents": ["patent pending"],
  "created_at": "2026-04-22T18:32:11Z",
  "tooling_version": "ultracompress-internal-publishing/0.3.2",
  "notes": "Initial public release"
}
```

## See also

- [`uc info`](../commands/info.md)
- [Compression methods](../concepts/compression-methods.md)
- [Reproducibility](../concepts/reproducibility.md)
- For per-model licensing, contact `legal@sipsalabs.com`.
