# Downloading a compressed model

> **Note:** `uc pull` was removed in v0.6.21. The current public CLI surface
> is `uc verify | uc try | uc catalog | uc info | uc version`. Use the flow
> below instead.

To get a Sipsa-compressed model onto local disk, use the standard Hugging
Face download path (no Sipsa-specific command is needed). Then use
`uc verify` to validate the SHA-256 download-integrity / manifest check on your own
hardware.

## Synopsis

```bash
# 1. Discover available packs
uc catalog

# 2. Download from HuggingFace
huggingface-cli download SipsaLabs/<repo-id> --local-dir ./<repo-id>

# 3. Verify download integrity against the published SHA-256 manifest
uc verify ./<repo-id>
```

You can also use the Python API:

```python
from huggingface_hub import snapshot_download
snapshot_download("SipsaLabs/<repo-id>", local_dir="./<repo-id>")
```

## Examples

```bash
# Free-tier pack: download + verify
huggingface-cli download SipsaLabs/qwen3-0.6b-uc-v3-bpw5 --local-dir ./qwen3-0.6b
uc verify ./qwen3-0.6b

# Inspect what came down
uc info ./qwen3-0.6b
```

## Behavior

- `huggingface-cli download` resumes interrupted downloads automatically.
- If you've authenticated with `huggingface-cli login`, your token is used;
  otherwise the public API is queried.
- Gated packs (`request` / `POC` tiers in `uc catalog`) require granted
  access on the HuggingFace repo first; see <https://sipsalabs.com/access>.
- `uc verify` checks the SHA-256 manifest end-to-end against the on-disk
  artifact: if the downloaded bytes do not match the published manifest, it fails.

## Disk space

Plan for the artifact size shown in `uc catalog`. A 7B-parameter model at
5 bits per weight is approximately 4.4 GB on disk; a 1.7B model is
approximately 1.1 GB. Downloads are incremental — interrupted runs resume
from the last completed file.

## Network behavior

- Default timeout: 30 seconds per file
- Retries: 3 with exponential backoff
- Concurrent file downloads: 8 (configurable via `HF_HUB_DOWNLOAD_TIMEOUT`
  and `HF_HUB_PARALLEL_DOWNLOAD_THREADS` env vars)
- Mirror support: set `HF_ENDPOINT=https://hf-mirror.com` for the China
  mirror

## Exit codes (`uc verify`)

| Code | Meaning |
|---|---|
| 0 | OK — downloaded bytes match the published SHA-256 manifest |
| 1 | Verification failed |
| 2 | Invalid arguments |

## See also

- [`uc catalog`](list.md) — discover the model ID
- [`uc info`](info.md) — inspect manifest fields
- [Hugging Face Hub authentication docs](https://huggingface.co/docs/huggingface_hub/en/quick-start#login-command)
