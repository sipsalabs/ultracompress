# Environment variables reference

UltraCompress respects a small set of environment variables. They are intentionally minimal; the CLI is not configurable through a sprawling configuration file.

## UltraCompress-specific

| Variable | Default | Effect |
|---|---|---|
| `UC_HF_ORG` | `sipsalabs` | Override the Hugging Face Hub organization queried by `uc list`. Useful for testing forks or mirrors. |
| `UC_HF_COLLECTION_TAG` | `ultracompress` | Override the Hugging Face Hub tag filter used by `uc list`. |
| `UC_BENCH_SEED` | `42` | Override the deterministic seed used by `uc bench`. |
| `UC_BENCH_OUTPUT_DIR` | `./bench-results` | Override the default `--output-dir` for `uc bench`. |
| `UC_VERBOSE` | unset | If set to `1`, print extra diagnostic logs to stderr. |
| `UC_NO_BANNER` | unset | If set to `1`, suppress the `_banner()` printout (useful for scripted use). |

## Inherited from `huggingface_hub`

UltraCompress uses `huggingface_hub` for Hub access; the standard Hub env vars all apply.

| Variable | Effect |
|---|---|
| `HF_HOME` | Cache root (default: `~/.cache/huggingface`) |
| `HF_ENDPOINT` | Override the API endpoint (use `https://hf-mirror.com` for the China mirror) |
| `HF_TOKEN` | Hugging Face access token (alternative to `huggingface-cli login`) |
| `HF_HUB_DOWNLOAD_TIMEOUT` | Per-file download timeout in seconds (default 30) |
| `HF_HUB_PARALLEL_DOWNLOAD_THREADS` | Concurrent download threads (default 8) |
| `HF_HUB_OFFLINE` | If set to `1`, never call the Hub; serve only from local cache |
| `HF_HUB_DISABLE_PROGRESS_BARS` | Hide tqdm progress bars |

## Inherited from `transformers` (when running `uc bench`)

| Variable | Effect |
|---|---|
| `TRANSFORMERS_OFFLINE` | If set to `1`, never call the Hub from `transformers` |
| `TRANSFORMERS_CACHE` | Override the model cache directory |

## Inherited from PyTorch

| Variable | Effect |
|---|---|
| `CUDA_VISIBLE_DEVICES` | Limit which GPUs are visible to PyTorch |
| `PYTORCH_CUDA_ALLOC_CONF` | Memory allocator tuning (e.g., `max_split_size_mb:512`) |

## CI-relevant

| Variable | Effect |
|---|---|
| `CI` | Most CLI tools detect `CI=true` and disable color/animation; UltraCompress respects this for `rich` output |
| `NO_COLOR` | If set, all colored output is disabled (per https://no-color.org) |

## Examples

### Use a different HF Hub org for testing

```bash
export UC_HF_ORG=your-org
uc list   # queries hf.co/your-org instead of hf.co/sipsalabs
```

### Run completely offline

```bash
export HF_HUB_OFFLINE=1
uc info ./models/sipsalabs_<model-id>   # works as long as files are cached
```

### Quiet mode for scripts

```bash
export UC_NO_BANNER=1
uc list --json | jq '.'   # clean JSON output without the banner
```

### Limit GPUs for `uc bench`

```bash
export CUDA_VISIBLE_DEVICES=0   # use only the first GPU
uc bench ./models/sipsalabs_<model-id> --tasks hellaswag --limit 100
```
