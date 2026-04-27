# Integration — llama.cpp / GGUF

Native llama.cpp / GGUF support is on the v0.2 roadmap (target Q3 2026). Until then, the integration story is two paths:

1. Use the UltraCompress reference loader in Python (current path)
2. Convert UltraCompress artifacts to GGUF format using a converter script we'll publish

This page describes both.

## Current path (v0.1, mid-2026)

```bash
pip install "ultracompress[torch]"
uc pull sipsalabs/<model-id>
```

Inference via the Python reference loader:

```python
from ultracompress_cli.loader import load_model  # available v0.1.1+
from transformers import AutoTokenizer

local = "./models/sipsalabs_<model-id>"
tokenizer = AutoTokenizer.from_pretrained(local)
model = load_model(local).cuda()

# Standard transformers generate
out = model.generate(**tokenizer("Hello", return_tensors="pt").to("cuda"))
print(tokenizer.decode(out[0]))
```

This works for evaluation, prototyping, and integration testing. It is **not** a production-grade inference path on llama.cpp's hardware.

## v0.2 path (Q3 2026 roadmap)

We will ship `uc export --format gguf` that converts an UltraCompress artifact to GGUF format compatible with `llama.cpp`:

```bash
uc export ./models/sipsalabs_<model-id> --format gguf -o qwen3-uc.gguf
./llama-cli -m qwen3-uc.gguf -p "Hello"
```

The exported GGUF file will:

- Inflate the UltraCompress weights into one of llama.cpp's native quantization formats (likely Q3_K_S or a new format we contribute)
- Preserve the `ultracompress.json` provenance via a custom GGUF metadata field (`general.ultracompress.bpw`, `general.ultracompress.method`, `general.ultracompress.patents`)
- Be a drop-in replacement for any other GGUF model in your llama.cpp pipeline

## Why we don't ship llama.cpp natively today

The UltraCompress weight format is not a llama.cpp-native quantization scheme. To run efficiently on llama.cpp, we would need to either:

- Contribute new ggml quantization types upstream (slow, ~6-12 months of upstream review)
- Inflate to existing types at export time (faster, but loses some compression efficiency)

We're pursuing both paths. The export-to-existing-types path lands first (v0.2, Q3 2026); upstream contribution lands eventually.

## Memory footprint after GGUF export

Inflating UltraCompress's 2.798-bpw representation to llama.cpp's nearest existing type:

| UltraCompress source | llama.cpp target | Memory after inflation | Compression vs FP16 |
|---|---|---|---|
| 2.798 bpw | Q3_K_S (~3.4 bpw) | ~720 MB for 1.7B | 4.7× |
| 2.798 bpw | IQ3_XS (~2.9 bpw) | ~620 MB for 1.7B | 5.5× |

We give up some of the on-disk compression efficiency in exchange for native llama.cpp inference speed. This is the right tradeoff for almost all production deployment scenarios; for **storage**-bound use cases (e.g., mobile distribution where the artifact is downloaded once), keep the native UltraCompress format.

## What you can do today (mid-2026)

- Use the Python loader for evaluation + prototyping
- Run `uc bench` to compare compressed vs FP16 quality on your tasks
- Open a GitHub issue with your specific llama.cpp use case so we prioritize the export path correctly

## What you'll be able to do post-Q3 2026

- `uc export --format gguf` to convert artifacts
- Direct `llama-cli` / `llama-server` use on the exported GGUF
- All standard llama.cpp tooling: `llama-bench`, `llama-quantize` (for re-quantizing), `llama-perplexity`

## See also

- [Integration with Hugging Face Transformers](transformers.md)
- [Integration with vLLM](vllm.md)
- [Integration with TensorRT-LLM](tensorrt-llm.md)
- [Manifest schema](../reference/manifest-schema.md)
- [llama.cpp upstream](https://github.com/ggerganov/llama.cpp)
