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
│ sipsalabs/<model-id> │ Qwen/Qwen3-1.7B    │  2.798 │ 1.04GB │      ... │
│ sipsalabs/llama2-7b-uc2p79  │ meta-llama/Ll...   │  2.798 │ 4.20GB │      ... │
│ sipsalabs/mistral-7b-uc2p79 │ mistralai/Mis...   │  2.798 │ 4.21GB │      ... │
└─────────────────────────────┴────────────────────┴────────┴────────┴──────────┘
```

## 3. Download a model

```bash
uc pull sipsalabs/<model-id>
```

The model lands in `./models/sipsalabs_<model-id>/`.

## 4. Inspect the artifact

```bash
uc info ./models/sipsalabs_<model-id>
```

You'll see the provenance manifest:

```
UltraCompress artifact: sipsalabs/<model-id>
─────────────────────────────────────────────────
Base model:   Qwen/Qwen3-1.7B
Method:       row-overlay-quantization (RoQ) v1
Bits/weight:  2.798
Size:         1.04 GB
SHA-256:      a3f5c8...   (verified ✓)
License:      research-free; commercial requires separate license
Patents:      USPTO 64/049,511 (filed 2026-04-25)
```

## 5. Run a benchmark

```bash
uc bench ./models/sipsalabs_<model-id> --tasks hellaswag --limit 500
```

This runs the `lm-eval-harness` HellaSwag task with 500 samples. Output:

```
                    Benchmark results
┏━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Task      ┃   acc   ┃ acc_norm       ┃    stderr ┃
┡━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ hellaswag │ 51.20%  │     67.60%     │  +/-2.23% │
└───────────┴─────────┴────────────────┴───────────┘
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
