# UltraCompress Quickstart

Get a compressed LLM running on your machine in under 5 minutes.

---

## Install

```bash
pip install ultracompress
```

Requires Python 3.10+. PyTorch with CUDA is needed for GPU inference and evaluation.

---

## Pull a compressed model

The smallest/fastest checkpoint to start with:

```bash
ultracompress pull SipsaLabs/qwen3-8b-streaming-bpw5
```

This downloads the streaming-compressed Qwen3-8B checkpoint from Hugging Face. Download size is roughly 5 GB.

Other available models:

```bash
ultracompress pull SipsaLabs/qwen3-14b-streaming-bpw5
ultracompress pull SipsaLabs/qwen3-32b-streaming-bpw5
ultracompress pull SipsaLabs/qwen2.5-72b-streaming-bpw5
```

The 32B and 72B checkpoints are larger downloads (roughly 20 GB and 45 GB respectively). Start with 8B.

---

## Run a smoke perplexity eval

Clone the repo to get the evaluation scripts:

```bash
git clone https://github.com/sipsalabs/ultracompress.git
cd ultracompress
```

Run the compressed-model evaluator:

```bash
python `uc verify` \
    --model SipsaLabs/qwen3-8b-streaming-bpw5 \
    --n_eval 50 \
    --seq_len 128
```

Expected output (within run-to-run variance):

```
PPL ratio: ~1.028x
Peak VRAM: ~2.26 GB
```

For the full streaming compression pipeline (compress from scratch):

```bash
python (production trainer, patent-protected) \
    --model qwen3-8b --bpw 5 --block_size 64 --rank 32 \
    --train_steps 200 --n_calib 100 --n_eval 50
```

This takes about 9 minutes on an RTX 5090. It will produce a result JSON under `scripts/overlay/artifacts/`.

---

## What do these compressed checkpoints contain

Each checkpoint is a directory with:

- **Per-layer `.pt` files** -- the compressed weight tensors and learned correction overlays for each transformer layer, saved independently.
- **`manifest.json`** -- metadata listing the base model, bits-per-weight, block size, correction rank, and the eval metrics at compression time.
- **Scaffold weights are NOT included.** At inference time, the base model's embedding layer, final layer norm, and language model head are loaded from the original Hugging Face model (e.g., `Qwen/Qwen3-8B`). The compressed checkpoint replaces only the transformer body. This keeps download sizes small and avoids redistributing unmodified weights.

The streaming compression recipe: scalar quantization scalar 5 bpw + per-block (B=64) absmax normalization + correction overlay low-rank low-rank correction overlay + KL distillation pass per layer. Each layer is compressed independently, which is why peak VRAM stays bounded by roughly one transformer layer regardless of total model depth.

---

## Reference numbers

| Model | PPL ratio | Peak VRAM | Status |
|---|---:|---:|---|
| Qwen3-8B | 1.028x | 2.26 GB | PROD |
| Qwen3-14B | 1.011x | 3.37 GB | PROD |
| Qwen3-32B | 1.037x | 4.85 GB | PROD |
| Qwen2.5-72B | 1.016x | 8.98 GB | PROD |

---

## License

Compressed model checkpoints are released under the **Sipsa Labs Research Evaluation License v1.0** (`sipsa-labs-research-evaluation-v1.0`). Research and evaluation use is permitted. Production deployment of the compression mechanism on your own models requires a separate commercial license -- contact founder@sipsalabs.com.

The base model weights (embedding, LM head, layer norm) are subject to the original model's license (Qwen: Apache 2.0).

Patent pending: USPTO applications 64/049,511 and 64/049,517, plus May 2026 supplements.

---

## Issues and support

File issues at: https://github.com/sipsalabs/ultracompress/issues

For pilot/commercial inquiries: founder@sipsalabs.com

---

*The 32B and 72B checkpoints are significantly larger downloads. If you're on a metered connection, start with the 8B checkpoint.*

Codec internals + training procedure are patent-protected (USPTO 64/049,511 + 64/049,517).
