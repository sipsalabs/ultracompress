# `uc pull`

Download a pre-compressed model artifact from the Hugging Face Hub to local disk.

## Synopsis

```
uc pull <model-id> [-o <output-dir>] [--revision <ref>]
```

## Arguments

| Argument | Required | Description |
|---|---|---|
| `<model-id>` | yes | Hugging Face Hub model ID (e.g., `sipsalabs/<model-id>`) |

## Options

| Option | Default | Description |
|---|---|---|
| `-o, --output PATH` | `./models/<model-id-underscored>` | Where to save the downloaded artifact |
| `--revision REF` | (latest) | Specific commit SHA, branch, or tag to download |

## Output

```
UltraCompress v0.6.11  · https://sipsalabs.com
Extreme compression for large language models. Patent pending

→ Pulling sipsalabs/<model-id> to models/sipsalabs_<model-id>
⠋ downloading...
OK  Saved to models/sipsalabs_<model-id>
Inspect with: uc info models/sipsalabs_<model-id>
```

## Behavior

- Wraps `huggingface_hub.snapshot_download` to fetch the entire repository to local disk.
- Creates the output directory if it doesn't exist.
- If you've authenticated with `huggingface-cli login`, your token is used; otherwise the public API is queried.
- Resumes interrupted downloads automatically.
- Verifies SHA-256 of the resulting `model.safetensors` against the manifest after download (when present).

## Examples

```bash
# Default location
uc pull sipsalabs/<model-id>

# Custom output directory
uc pull sipsalabs/<model-id> -o ~/models/qwen3-1.7b-uc

# Pin to a specific commit
uc pull sipsalabs/<model-id> --revision abc123def456

# Inspect what came down
uc info ~/models/qwen3-1.7b-uc
```

## Disk space

Plan for the artifact size shown in `uc list`. A 7B-parameter model at 5 bpw is approximately 4.4 GB on disk; a 1.7B model is approximately 1.1 GB. The download is **incremental** — interrupted pulls resume from the last completed file.

## Network behavior

- Default timeout: 30 seconds per file
- Retries: 3 with exponential backoff
- Concurrent file downloads: 8 (configurable via `HF_HUB_DOWNLOAD_TIMEOUT` and `HF_HUB_PARALLEL_DOWNLOAD_THREADS` env vars)
- Mirror support: set `HF_ENDPOINT=https://hf-mirror.com` for the China mirror

## Exit codes

| Code | Meaning |
|---|---|
| 0 | OK |
| 1 | Download failed (network, auth, or model not found) |
| 2 | Invalid arguments (Click default) |

## See also

- [`uc list`](list.md) — find the model ID before pulling
- [`uc info`](info.md) — verify the artifact after pulling
- [Hugging Face Hub authentication docs](https://huggingface.co/docs/huggingface_hub/en/quick-start#login-command)
