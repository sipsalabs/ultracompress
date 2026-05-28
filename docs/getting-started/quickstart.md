# Quickstart

Five minutes from `pip install` to running benchmarks on a pre-compressed model.

## 1. Install

```bash
pip install "ultracompress[torch]"
```

(See [Install](install.md) for alternatives.)

## 2. Browse the catalog

```bash
uc list
```

You'll see a table of pre-compressed models published by Sipsa Labs:

```
                          Pre-compressed models from Hugging Face Hub
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┓
┃ Model ID                    ┃ Base               ┃    bpw ┃   Size ┃ Downloads┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━┩
│ sipsalabs/<model-id>        │ Qwen/Qwen3-1.7B    │    5.0 │ ~1.2GB │      ... │
│ sipsalabs/llama-3.1-70b-... │ meta-llama/Ll...   │    5.0 │ ~44 GB │      ... │
│ sipsalabs/mistral-7b-...    │ mistralai/Mis...   │    5.0 │ ~4.5GB │      ... │
└─────────────────────────────┴────────────────────┴────────┴────────┴──────────┘
```

## 3. Download a model

```bash
huggingface-cli download SipsaLabs/<repo-id> --local-dir ./<repo-id>
```

Or use the Python API:

```python
from huggingface_hub import snapshot_download
snapshot_download("SipsaLabs/<repo-id>", local_dir="./<repo-id>")
```

## 4. Inspect the artifact

```bash
uc info ./models/sipsalabs_<model-id>
```

You'll see the provenance manifest:

```
UltraCompress artifact: sipsalabs/<model-id>
─────────────────────────────────────────────────
Base model:   Qwen/Qwen3-1.7B
Bits/weight:  5.0
Size:         ~1.2 GB
SHA-256:      a3f5c8...   (verified ✓)
License:      research-free; commercial requires separate license
Patents:      USPTO provisional applications on file; method specifics under NDA
```

## 5. Verify the pack against your own runtime

The public CLI in v0.6.x is intentionally minimal: `verify`, `try`, `catalog`, `info`, `audit`, `version`. Downstream-task benchmarking is the caller's responsibility — point your existing `lm-eval-harness` (or any other evaluator) at the reconstructed model and compare against the published numbers in `docs/benchmarks.json`.

```bash
uc verify ./<repo-id>        # SHA-256 + manifest integrity
uc info ./<repo-id>          # provenance + bpw + base model
```

That's it — you're up and running.

## Next steps

- **Use the model in your inference stack** → [Integration guides](../integration/llamacpp.md)
- **Understand the method** → [Compression methods](../concepts/compression-methods.md)
- **Plan a commercial deployment** → Email `legal@sipsalabs.com`
- **Hit a snag?** → [Open an issue](https://github.com/sipsalabs/ultracompress/issues)

## Programmatic use

The CLI is the supported public surface for v0.1. Programmatic access (`from ultracompress_cli import ...`) is intentionally minimal until v0.2; the API will stabilize once the patent-pending methods are formally published. If you need programmatic access today, use the Hugging Face Hub API directly:

```python
from huggingface_hub import snapshot_download
local_dir = snapshot_download("sipsalabs/<model-id>")
```

Then load with `transformers` or your preferred runtime.
