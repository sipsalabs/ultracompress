# UltraCompress Quickstart

Get a compressed LLM running on your machine in under 5 minutes.

---

## Install

```bash
pip install ultracompress
```

Requires Python 3.10+. PyTorch with CUDA is needed for GPU inference and evaluation.

---

## Download a compressed model

The smallest/fastest checkpoint to start with:

```bash
huggingface-cli download SipsaLabs/qwen3-8b-uc-v3-bpw5 --local-dir ./qwen3-8b
```

This downloads the 5-bit compressed Qwen3-8B pack from Hugging Face. Download size is roughly 5 GB.

Other available models (see `uc catalog` for the full live list):

```bash
huggingface-cli download SipsaLabs/qwen3-14b-uc-v3-bpw5 --local-dir ./qwen3-14b
huggingface-cli download SipsaLabs/phi-3-mini-4k-instruct-uc-v3-bpw5 --local-dir ./phi-3-mini
huggingface-cli download SipsaLabs/qwen3-0.6b-uc-v3-bpw5 --local-dir ./qwen3-0.6b
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

The full streaming pipeline that produces a fresh compressed pack is NDA-gated. Customer-side `uc verify` is the public CLI surface; for engagement access, contact founder@sipsalabs.com.

---

## What do these compressed checkpoints contain

Each checkpoint is a directory with:

- **Per-layer `.pt` files** -- the compressed-weight tensors for each transformer layer, saved independently. Per-layer codec internals are NDA-gated.
- **`manifest.json`** -- public auditor-facing metadata: base model, bits-per-weight, per-layer SHA-256 hashes, and the eval metrics at compression time. Internal codec geometry is NDA-gated.
- **Scaffold weights are NOT included.** At inference time, the base model's embedding layer, final layer norm, and language model head are loaded from the original Hugging Face model (e.g., `Qwen/Qwen3-8B`). The compressed checkpoint replaces only the transformer body. This keeps download sizes small and avoids redistributing unmodified weights.

The streaming compression pipeline compresses each transformer layer independently in a single GPU residency window, which is why peak VRAM stays bounded by roughly one transformer layer regardless of total model depth. Internal codec specifics are NDA-gated.

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

Patent pending: two USPTO provisional applications on file, plus supplemental filings. Claim text under NDA.

---

## Issues and support

File issues at: https://github.com/sipsalabs/ultracompress/issues

For pilot/commercial inquiries: founder@sipsalabs.com

---

*The 32B and 72B checkpoints are significantly larger downloads. If you're on a metered connection, start with the 8B checkpoint.*

Codec internals and the procedure used to produce a compressed pack are patent-protected; USPTO provisional applications are on file and full claim text is available under NDA.
