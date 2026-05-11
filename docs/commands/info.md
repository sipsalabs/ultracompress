# `uc info`

Display the provenance manifest of a downloaded UltraCompress artifact.

## Synopsis

```
uc info <path>
```

## Arguments

| Argument | Required | Description |
|---|---|---|
| `<path>` | yes | Path to a directory or `ultracompress.json` file produced by `uc pull` |

## Output

```
UltraCompress v0.1.0  · https://sipsalabs.com
Extreme compression for large language models. Patent pending — USPTO 64/049,511 + 64/049,517

UltraCompress artifact: sipsalabs/<model-id>
─────────────────────────────────────────────────
Base model:   Qwen/Qwen3-1.7B
Method:       row-overlay-quantization (RoQ) v1
Bits/weight:  2.798
Size:         1.04 GB
SHA-256:      a3f5c8...   (verified ✓)
License:      research-free; commercial requires separate license
Patents:      USPTO 64/049,511 (filed 2026-04-25)

Files:
  model.safetensors      1116691072 bytes
  tokenizer.json            2734851 bytes
  config.json                  4276 bytes
  ultracompress.json           1842 bytes
```

## Manifest schema

The `ultracompress.json` file is a JSON document with the following fields. See [Manifest schema](../reference/manifest-schema.md) for the formal schema.

```json
{
  "schema_version": "1.0",
  "model_id": "sipsalabs/<model-id>",
  "base_model": "Qwen/Qwen3-1.7B",
  "method": "track-a-row-overlay",
  "method_version": "1.0",
  "bpw": 2.798,
  "size_bytes": 1116691072,
  "files": {
    "model.safetensors": {
      "sha256": "a3f5c8...",
      "size_bytes": 1116691072
    }
  },
  "license": "sipsalabs-research-eval-1.0",
  "patents": ["USPTO 64/049,511"],
  "created_at": "2026-04-22T18:32:11Z",
  "tooling_version": "ultracompress-internal-publishing/0.3.2"
}
```

## Behavior

- Locates `ultracompress.json` either in the directory you passed, or treats the path as the manifest file directly.
- Verifies `model.safetensors` SHA-256 against the manifest if both are present.
- Returns 0 on success; 1 if no manifest found.

## Examples

```bash
# Inspect a downloaded artifact
uc info ./models/sipsalabs_<model-id>

# Inspect a manifest file directly
uc info ./models/sipsalabs_<model-id>/ultracompress.json
```

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Manifest found and printed |
| 1 | No manifest found at the given path |
| 2 | Invalid arguments (Click default) |

## Failure modes

- **Path doesn't exist** → exit 2 with Click's standard error
- **Path is a directory but contains no `ultracompress.json`** → exit 1 with "No ultracompress metadata found at `<path>`"
- **Path is `ultracompress.json` but corrupt** → exit 1 with parse error
- **Path is a manifest but the referenced `model.safetensors` SHA mismatches** → prints with "verified ✗" warning but still exits 0 (you should `uc pull` again)

## See also

- [`uc pull`](pull.md) — download an artifact in the first place
- [`uc bench`](bench.md) — benchmark the artifact on downstream tasks
- [Manifest schema](../reference/manifest-schema.md) — formal schema documentation
