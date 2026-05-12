# Configuration

UltraCompress is intentionally configuration-light. Most behavior is controlled via command-line flags or [environment variables](../reference/env-vars.md). There is no global config file by default.

## Configuration precedence

When the same setting is specified in multiple places, the order from highest to lowest priority is:

1. Command-line flag (e.g., `--device cuda:1`)
2. Environment variable (e.g., `UC_DEVICE=cuda:1`)
3. User config file at `~/.config/ultracompress/config.toml` (when supported, v0.1.1+)
4. Built-in defaults

## User config file (v0.1.1+ planned)

Optional config file at `~/.config/ultracompress/config.toml` (or `%APPDATA%\ultracompress\config.toml` on Windows):

```toml
[hub]
org = "sipsalabs"
collection_tag = "ultracompress"

[bench]
default_tasks = "hellaswag,arc_challenge"
default_limit = 500
default_batch_size = 8
default_device = "cuda:0"
default_output_dir = "./bench-results"

[cli]
no_banner = false
verbose = false
```

This is supported in v0.1.1 and later. v0.1.0 ignores it.

## Hugging Face Hub authentication

For private models or higher rate limits:

```bash
pip install -U huggingface_hub
huggingface-cli login
# Paste your token from https://huggingface.co/settings/tokens
```

Or set `HF_TOKEN`:

```bash
export HF_TOKEN=hf_...
```

The token is read by all HF Hub-using commands automatically.

## CUDA / device selection

Default device is `cuda:0`. To use a different GPU:

```bash
uc bench ./models/sipsalabs_<model-id> --device cuda:1
```

For CPU-only:

```bash
uc bench ./models/sipsalabs_<model-id> --device cpu --limit 50
```

(CPU is ~100× slower; useful for smoke testing only.)

To restrict visible GPUs at the OS level:

```bash
export CUDA_VISIBLE_DEVICES=0,1
uc bench ./models/sipsalabs_<model-id> --device cuda:0
```

## Output directory

By default `uc pull` saves to `./models/<model-id-underscored>`. Override:

```bash
uc pull sipsalabs/<model-id> -o ~/models/qwen3-uc
```

By default `uc bench` saves to `./bench-results`. Override:

```bash
uc bench ./models/sipsalabs_<model-id> -o /tmp/bench-runs/run-001
```

## Quiet / scripted mode

Suppress the brand banner:

```bash
export UC_NO_BANNER=1
uc list --json | jq '.'   # clean JSON output
```

## Verbose mode

Enable diagnostic logging:

```bash
export UC_VERBOSE=1
uc bench ./models/sipsalabs_<model-id> --tasks hellaswag --limit 50
# Logs go to stderr
```

## Cache locations

| Cache | Default location | Override |
|---|---|---|
| Hugging Face Hub | `~/.cache/huggingface` | `HF_HOME` |
| Pulled models | `./models/` | `-o` flag on `uc pull` |
| Benchmark results | `./bench-results/` | `-o` flag on `uc bench` |

## What we don't support yet

- Project-local `pyproject.toml` configuration
- Workspace-style multi-config (different defaults per directory)
- TOML config encryption
- Profile-based configurations (e.g., "production" vs "research" profiles)

These may land in v0.2 if customer demand justifies. Open an issue with your specific use case.
