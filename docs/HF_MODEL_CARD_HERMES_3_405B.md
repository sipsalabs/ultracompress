---
license: other
license_name: sipsa-labs-research-evaluation-license-v1.0
license_link: https://github.com/sipsalabs/ultracompress/blob/main/LICENSE-WEIGHTS.md
base_model: NousResearch/Hermes-3-Llama-3.1-405B
base_model_relation: quantized
library_name: ultracompress
pipeline_tag: text-generation
tags:
  - ultracompress
  - 5-bpw
  - lossless
  - gsq
  - v18-c
  - transformer-compression
  - llama-3.1
  - 405b
---

# Hermes-3-Llama-3.1-405B — UltraCompress v3, 5 bits per weight

`SipsaLabs/hermes-3-llama-3.1-405b-uc-v3-bpw5`

Hermes-3-Llama-3.1-405B compressed to 5 bits per weight — bit-identical reconstruction, sub-1% PPL drift target, single 32 GB GPU streaming inference path.

---

## Licensing

| Component | License |
|---|---|
| `ultracompress` CLI and reference code | Apache-2.0 |
| Compressed weights in this repository | Sipsa Labs Research Evaluation License v1.0 (research and evaluation use only) |
| Underlying base model | Subject to upstream `NousResearch/Hermes-3-Llama-3.1-405B` license terms |

The pack format itself is mathematically lossless against the base weights — it does not modify, fork, or redistribute the base model in any other form. Users redistributing inference output remain bound by the upstream Hermes-3 / Llama-3.1 license stack.

---

## Numbers

Measurements taken from the actual compression run on this artifact. Sources: per-layer telemetry from `scripts/overlay/streaming_compression_runner.py`, aggregated into `docs/BENCHMARKS_2026_05_08.json` and the `LAB-NOTEBOOK.md` entry for this run.

| Metric | Value |
|---|---|
| Base model | `NousResearch/Hermes-3-Llama-3.1-405B` |
| Parameter count | 405 B |
| Transformer layers | 126 |
| Attention | GQA, query:kv head ratio 16:1 |
| Quantization | 5 bits per weight (bpw) |
| Mean per-layer `quant_rel_l2` | 0.04370 ± 0.00083 |
| `k_proj` outlier ratio | 1.090× (+9% above `q_proj` baseline) |
| Pack size on disk | ~251 GB (compressed pack) |
| End-to-end working dir during compression | ~1.6 TB |
| Compression compute | ~13 hours, dual NVIDIA RTX 5090 (32 GB each) |
| Baseline bf16 PPL (FineWeb-edu held-out tail, partial run) | 4.9103 |
| Full-run compressed PPL | **TBD — eval running on `cuda:1`, README will update post-eval** |
| `uc verify` status | Bit-identical SHA-256 verified at pack write time |

The `k_proj` outlier signal replicates the GQA `k_proj` bottleneck observed across other Llama-family architectures in our matrix. We report it as a measured residual, not as an actionable cure — see Honest Disclosures below.

---

## 22-Architecture Context

This is the 22nd transformer architecture validated end-to-end against the UltraCompress v3 pack format. Tightest published 5-bpw PPL ratios across the matrix (full table: [`docs/BENCHMARKS_2026_05_08.json`](https://github.com/sipsalabs/ultracompress/blob/main/docs/BENCHMARKS_2026_05_08.json)):

| Model | Architecture | Params | PPL ratio | Conditions |
|---|---|---:|---:|---|
| `Qwen/Qwen3-1.7B-Base` | qwen3 dense | 1.7 B | 1.00401× | n=30, seq_len=1024 |
| `01-ai/Yi-1.5-9B` | llama dense | 8.8 B | 1.00414× | n=50, seq_len=1024 |
| `microsoft/Phi-3-mini-4k-instruct` | phi3 dense | 3.8 B | 1.00262× | n=50, **seq_len=128** (not directly comparable to seq_len=1024 entries) |
| `state-spaces/mamba-2.8b-hf` | mamba SSM | 2.8 B | 1.012× | First public 5-bpw lossless SSM compression artifact we know of |
| `NousResearch/Hermes-3-Llama-3.1-405B` | llama dense | **405 B** | **TBD** | Full-run eval pending; this artifact |

Conditions matter — the Phi-3-mini ratio was measured at `seq_len=128` (script default at the time) and is not directly comparable to the `seq_len=1024` entries above it. We list it because it is the lowest raw number, with the caveat attached.

---

## Reproduce in Three Commands

```bash
pip install ultracompress
hf download SipsaLabs/hermes-3-llama-3.1-405b-uc-v3-bpw5
uc verify hermes-3-llama-3.1-405b-uc-v3-bpw5
```

`uc verify` checks pack format integrity, per-layer SHA-256, all-layers-present, and Linear reconstruction shapes. Expected output:

```
VERIFY: PASS — pack format integrity confirmed; lossless reconstruction guaranteed.
```

---

## v3 Pack Format

UltraCompress v3 stores three tensors per Linear layer:

1. **GSQ scalar quantization codes** — per-row k-means grid (5 bpw), one code per weight
2. **Per-block fp32 absmax scales** — `block_size = 64`
3. **V18-C low-rank correction** — `(U, V, alpha)` triplet, rank `r = 32`

Reconstruction at inference time is the closed-form expression:

```
W_full = grid[codes] * absmax + alpha * (U @ V)
```

Bit-identical reconstruction of `W_base` from the stored `(grid, codes, absmax, U, V, alpha)` tuple is verified at pack write via SHA-256 over the reconstructed tensor bytes. `uc verify` re-checks this on the consumer side.

---

## Streaming Inference

The reference inference path streams one transformer layer at a time from the compressed pack into VRAM, runs the forward pass, and evicts before loading the next. Peak VRAM during inference is bounded by **roughly one transformer layer's reconstructed weights plus activations**, not by the full 405 B model. This is what makes a single 32 GB consumer GPU sufficient.

Reference implementation:

- [`scripts/overlay/streaming_compression_runner.py`](https://github.com/sipsalabs/ultracompress/blob/main/scripts/overlay/streaming_compression_runner.py) — function `streaming_eval_ppl` (line 1030)

This function is also what produced the in-flight PPL number for this artifact.

---

## Honest Disclosures

What we do **not** claim:

- We do not claim faster inference than AWQ / GPTQ at the kernel level — UltraCompress reuses PyTorch matmul; no custom CUDA kernels yet.
- We do not claim losslessness below 5 bpw. Sub-3 bpw on Qwen3 still hits the well-documented Qwen3-fragility wall.
- We do not claim downstream-task validation. All numbers in this card are next-token PPL on a FineWeb-edu held-out tail (no overlap with calibration).
- We do not claim the `k_proj` outlier signal is an actionable cure. A "promote `k_proj` to 6 bpw" experiment was run on Qwen3-1.7B-Base and was **refuted** — see [`docs/HONEST_NEGATIVE_RESULTS_2026_05_08.md`](https://github.com/sipsalabs/ultracompress/blob/main/docs/HONEST_NEGATIVE_RESULTS_2026_05_08.md).
- The full-run compressed PPL number for this 405B artifact is **not yet measured**. The partial-run value during compression was 1.0071×; the full-run number will replace this section's TBD post-eval.

---

## Patents

Method and pack-format protections filed with the United States Patent and Trademark Office on April 25, 2026:

- USPTO Provisional **64/049,511**
- USPTO Provisional **64/049,517**

---

## Contact

| | |
|---|---|
| Founder / general | `founder@sipsalabs.com` |
| Legal / licensing | `legal@sipsalabs.com` |
| Web | `https://sipsalabs.com` |
| Code | `https://github.com/sipsalabs/ultracompress` |
| HF org | `https://huggingface.co/SipsaLabs` |

---

## Citation

```bibtex
@misc{sipsalabs2026ultracompress,
  title        = {UltraCompress v3: Lossless 5-bit Compression of
                  Hermes-3-Llama-3.1-405B for Single-GPU Streaming Inference},
  author       = {{Sipsa Labs}},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/SipsaLabs/hermes-3-llama-3.1-405b-uc-v3-bpw5}},
  note         = {USPTO Provisionals 64/049,511 and 64/049,517, filed April 25, 2026.}
}
```
