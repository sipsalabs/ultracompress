# UltraCompress

**Compression infrastructure for trained transformers. Single 32GB consumer GPU. 1.7B → 405B parameters. Dense and MoE.**

UltraCompress takes any HuggingFace transformer checkpoint and produces a `.uc` artifact at sub-1.5% perplexity degradation, on a single consumer GPU, regardless of whether the source model fits in GPU memory.

The 9-architecture matrix as of 2026-05-07:

| Model | Params | Type | Baseline PPL | Compressed PPL | PPL_r |
|---|---|---|---|---|---|
| Qwen3-1.7B | 1.7B | dense | 16.116 | 16.263 | 1.0091 |
| Mistral-7B-v0.3 | 7.2B | dense | 6.443 | 6.525 | 1.0126 |
| Llama-3.1-8B | 8.0B | dense | 8.265 | 8.324 | 1.0071 |
| Llama-3.1-70B | 70B | dense | 6.118 | 6.173 | 1.0090 |
| Hermes-3-Llama-3.1-405B | 405B | dense | 4.910 | 4.945 | 1.0071 |
| Qwen3-235B-A22B | 235B | MoE 128 | 8.095 | 8.125 | 1.0038 |
| Mixtral-8x22B-v0.1 | 141B | MoE 8 | 5.145 | 5.176 | 1.0061 |
| Mixtral-8x7B-v0.1 | 46.7B | MoE 8 | 6.004 | 6.026 | 1.0037 |
| Phi-3.5-MoE-instruct | 42B | MoE 16 | 6.513 | 6.521 | **1.0013** |

Mean PPL_r: **1.0066**. All 9 PASS the ≤1.013 stretch goal.

---

## Install

```bash
pip install ultracompress
```

## Compress (single command, single GPU)

```bash
uc compress \
  --hf-id Qwen/Qwen3-8B \
  --bpw 5 \
  --rank 32 \
  --device cuda:0 \
  --output ./qwen3-8b.uc
```

Streams the source model from disk one decoder layer at a time. Caches teacher hidden states. Trains a per-layer V18-C correction. Writes a single `.uc` directory of layer-shaped artifacts. Peak VRAM bounded by one decoder layer + activations.

## Load and run

```python
import ultracompress as uc
from transformers import AutoModelForCausalLM

skeleton = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", torch_dtype="float16")
compressed = uc.load("./qwen3-8b.uc", skeleton)
out = compressed.generate(input_ids, max_new_tokens=128)
```

## What's supported

- **Architectures**: Qwen3, Qwen2/2.5, Mistral, Llama, Mixtral, Phi-3, Phi-MoE, Qwen3-MoE.
- **Scale**: from 1.7B to 405B dense; from 42B to 235B MoE; tested on a single 32GB GPU end-to-end.
- **Hardware**: any CUDA GPU with at least 16 GB VRAM. Tested on RTX 5090 (32 GB).
- **Calibration**: 64 prompts × 1024 tokens FineWeb-edu by default; bring your own corpus with `--calibration-tokens path.pt`.

## Method (high level)

1. **Stream-compress (Phase 1)** — load each decoder layer one at a time from local safetensors shards, run a teacher forward pass to cache the next-layer hidden state on CPU, free the layer.
2. **Per-layer V18-C training (Phase 2)** — for each layer in turn: load weights, apply 5-bit GSQ scalar quantization, wrap each Linear with a low-rank V18-C correction (`y = (alpha * Wq + V@U) @ x + b`), train rank-32 corrections via hidden-MSE for 200 steps with SVD warm-start.
3. **Streaming-teacher PPL** — baseline-quality measurement uses the same per-layer streaming pipeline so the comparison is exact.

For the full pipeline see `scripts/overlay/stream_compress_e2e.py`.

## License

Apache 2.0. Patent provisionals 64/049,511 and 64/049,517 filed at the USPTO 2026-04-25.

## Contact

founder@sipsalabs.com — Sipsa Labs, Inc.
