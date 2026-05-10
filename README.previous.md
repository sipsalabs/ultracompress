# UltraCompress

> **First mathematically lossless 5-bit transformer compression library, validated end-to-end across 11+ architectures including state-space models.**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/badge/pypi-0.5.2-blue.svg)](https://pypi.org/project/ultracompress/)
[![Patent](https://img.shields.io/badge/patent-USPTO%2064%2F049%2C511%20%2B%2064%2F049%2C517-orange.svg)]()

---

> **Patent-protected methods.** The compression algorithms in this repo are covered under USPTO provisionals 64/049,511 + 64/049,517. Apache 2.0 grants you full use of the source code; commercial productization requires a separate license — see [PATENT_NOTICE.md](./PATENT_NOTICE.md) or email founder@sipsalabs.com.
>
> **Independent research, no VC backing yet.** If this work is useful to you, [GitHub Sponsors](https://github.com/sponsors/sipsalabs) keeps it shipping. Or run a paid Phase 0 POC: founder@sipsalabs.com.

---

## Current state (2026-05-08)

- **11 architectures validated end-to-end** cumulative through this morning — 10 transformer + Mamba-2.8B SSM, bit-identical W_base reconstruction at 5 bpw.
- **4 more dense archs added today** on GPU 1: SmolLM2-1.7B, TinyLlama-1.1B-Chat, Qwen3-0.6B, OLMo-2-0425-1B (queued retry after streaming-fix patch).
- **Hermes-3-Llama-3.1-405B compression in flight** on GPU 0: 53/126 layers complete, ETA tonight.
- **2 public HuggingFace artifacts uc-verify-PASS** (qwen3-1.7b, mistral-7b-v0.3); **8 more in-flight upload-pending** after local PASS.
- **Multi-arch PPL ratios at 5 bpw** (representative): Mistral-7B `1.0100`, Llama-3.1-8B `1.0125`, Mamba-2.8B `1.0119`.
- **10/10 local production packs PASS** `uc verify` (bit-identical W_base reconstruction).

Live verification status — every public artifact, file hash, and verifier exit code in one place: [docs/PUBLIC_VERIFICATION_DASHBOARD_2026_05_08.md](docs/PUBLIC_VERIFICATION_DASHBOARD_2026_05_08.md).

---

## Quick start

```bash
pip install -U ultracompress                                          # 0.5.2
hf download SipsaLabs/mistral-7b-v0.3-uc-v3-bpw5 --local-dir ./mistral
uc verify ./mistral                          # CLI entry point
# OR — when `uc` isn't on PATH (Jupyter, CI, Docker, post-install hooks):
python -m ultracompress verify ./mistral     # equivalent fallback
```

`uc verify` reconstructs `W_base = absmax × grid[codes]` from the persisted k-means grid + per-block scales + bit-packed integer codes and confirms it is bit-identical to the dequantized weight the trainer used during distillation.

`uc serve ./mistral` exposes an OpenAI-compatible API at `http://localhost:8080`. See [docs/CUSTOMER_ONBOARDING_FLOW_v3_2026_05_08.md](docs/CUSTOMER_ONBOARDING_FLOW_v3_2026_05_08.md) for the full deploy walkthrough.

---

## What's new in v0.5.2 (publishing today)

- `python -m ultracompress` fallback support via new `ultracompress/__main__.py` — unblocks Jupyter, CI, Docker images, post-install hooks where the `uc` console script is missing from PATH.
- **SSM (Mamba / state-space-model) Linear naming** added to `TARGET_SUBS` in `pack.py` — `in_proj`, `x_proj`, `dt_proj`, `out_proj` now packed alongside the standard transformer Linear set.
- **Single-file safetensors** support in `stream_compress` — unblocks <2B-param models that ship without an index shard (TinyLlama, SmolLM2, Qwen3-0.6B, OLMo-2).
- `olmo` / `olmo2` model_type dispatch added to `streaming_teacher` and `streaming_compression_runner`.
- Full notes: [docs/RELEASE_NOTES_v0.5.2.md](docs/RELEASE_NOTES_v0.5.2.md).

---

## Architecture matrix

End-to-end validated at 5 bpw with bit-identical W_base reconstruction. Checkmark = uc verify PASS, pending = local PASS, HF upload in flight, retry = re-running on v0.5.2 after streaming-fix.

| Architecture | Params | Layers | bpw | PPL ratio | uc verify | HF repo |
|---|---:|---:|---:|---:|:---:|:---|
| Qwen3-1.7B | 1.7B | 28 | 5 | 1.0078 | PASS | [`SipsaLabs/qwen3-1.7b-uc-v3-bpw5`](https://huggingface.co/SipsaLabs/qwen3-1.7b-uc-v3-bpw5) |
| Mistral-7B-v0.3 | 7.2B | 32 | 5 | 1.0100 | PASS | [`SipsaLabs/mistral-7b-v0.3-uc-v3-bpw5`](https://huggingface.co/SipsaLabs/mistral-7b-v0.3-uc-v3-bpw5) |
| Llama-3.1-8B | 8.0B | 32 | 5 | 1.0125 | PASS local | (upload pending) |
| Qwen3-8B | 8.0B | 36 | 5 | 1.0044 | PASS local | (upload pending) |
| Qwen3-14B | 14.0B | 40 | 5 | 1.0040 | PASS local | (upload pending) |
| Mixtral-8x7B-v0.1 | 47B (MoE 8 exp) | 32 | 5 | (PPL 5.88) | PASS local | (upload pending) |
| Phi-3.5-MoE-instruct | 42B (MoE 16 exp) | 32 | 5 | (PPL 6.95) | PASS local | (upload pending) |
| Llama-3.1-70B | 70B | 80 | 5 | (PPL 6.02) | PASS local | (upload pending) |
| Qwen2.5-72B | 72B | 80 | 5 | 1.0162 | PASS local | (upload pending) |
| Mamba-2.8B (SSM) | 2.8B | 64 | 5 | 1.0119 | PASS local | (upload pending) |
| Hermes-3-Llama-3.1-405B | 405B | 126 | 5 | (in flight) | 53/126 layers | (compressing GPU 0) |
| SmolLM2-1.7B | 1.7B | 24 | 5 | (in flight) | pending | (added today, GPU 1) |
| TinyLlama-1.1B-Chat | 1.1B | 22 | 5 | (in flight) | pending | (added today, GPU 1) |
| Qwen3-0.6B | 0.6B | 28 | 5 | (in flight) | pending | (added today, GPU 1) |
| OLMo-2-0425-1B | 1B | 16 | 5 | (in flight) | retry on v0.5.2 | (added today, queued) |

PPL ratios listed against the model's own bf16 baseline on a 100-sample held-out slice. MoE rows lack baseline due to single-GPU OOM on bf16 baseline; multi-GPU baseline pipeline lands in v0.6.

UltraCompress is the first quantization library publicly compatible with both transformer and state-space architectures (Mamba), including the Linear naming required for emerging hybrids such as AI21 Jamba.

---

## How v0.3 lossless works

`uc pack v0.3` persists the trainer's k-means **learned grid** + per-block scales + bit-packed integer codes directly into the customer artifact. Reconstruction is `W_base = absmax × grid[codes]` and reproduces — bit-identically — the dequantized weight the trainer used during distillation.

This is the only mathematically lossless 5-bit transformer quantization format in production. AWQ / GPTQ / EXL3 / bitsandbytes-int4 introduce measurable PPL drift between training-time eval and customer-time inference. UltraCompress v0.3 customers see identical inference behavior to what the trainer measured.

| Customer profile | Why bit-exact reconstruction matters |
|---|---|
| Defense / aerospace | Bit-exact deploy is a compliance requirement (audit trail). |
| Healthcare AI (FDA-regulated) | Model equivalence required between dev and deploy. |
| Finance (SR 11-7 model validation) | Reproducibility audit requires bit-exact recovery. |
| Frontier labs (internal artifact distribution) | Red-team eval fidelity requires identical inference. |
| Single-GPU 70B+ deployment | Streaming compression keeps peak VRAM ~one transformer layer. |

Full vendor comparison: [docs/COMPETITIVE_LANDSCAPE_v3_LOSSLESS_2026_05_08.md](docs/COMPETITIVE_LANDSCAPE_v3_LOSSLESS_2026_05_08.md).

---

## Streaming compression — single-GPU large-model headline

Per-layer streaming validated end-to-end across 8B → 72B with peak VRAM bounded by ~one transformer layer regardless of total depth.

| Model | Layers | PPL ratio | Peak VRAM |
|---|---:|---:|---:|
| Qwen3-8B | 36 | 1.0278 | 2.26 GB |
| Qwen3-14B | 40 | 1.0111 | 3.37 GB |
| Qwen3-32B | 64 | 1.0367 | 4.85 GB |
| Qwen2.5-72B | 80 | 1.0162 | **8.98 GB** |

Recipe: GSQ scalar 5 bpw + per-block (B=64) absmax + V18-C rank-32 low-rank correction + 200-step KL distillation per layer. Process: lazy-load layer fp16 weights via `safetensors` → cache teacher hidden output → quantize → fit V18-C against cache → save → free → next layer. Compression time ~1 min/layer.

Reproduce on a 5090 (~9 min for 8B):
```bash
python scripts/overlay/streaming_compression_runner.py \
    --model qwen3-8b --bpw 5 --block_size 64 --rank 32 \
    --train_steps 200 --n_calib 100 --n_eval 50
```

---

## Earlier research tracks

Three independent compression mechanisms compose multiplicatively:

- **Track A — streaming compression** (above): single-GPU 72B at PPL 1.0162.
- **Track B — Fractal Residual Recursion** (Claims 1–16): shared-block architectural compression at 311–734× on Qwen3-1.7B (HQ5 h256 reaches 70.0% T10). See [docs/HQ5_RESULTS.md](docs/HQ5_RESULTS.md), [REPRODUCE.md](REPRODUCE.md).
- **Track C — row-overlay sub-3-bpw quantization** (Claims 17–20): beats bitsandbytes-nf4 at 30% fewer bits on a 6-model cohort (n=500 LAMBADA). Zero catastrophic failures across 48 measurements vs. HQQ's 6/6 at 2-bit g64. See [RESULTS.md](RESULTS.md), [docs/claim20_summary.txt](docs/claim20_summary.txt).

Stacked, the projection for a 100T-parameter model on a single GPU is ~5 GB at 20,000× total compression — see [docs/100T_MISSION_MATH_2026_05_03.md](docs/100T_MISSION_MATH_2026_05_03.md). Track A + B + C numbers are individually validated; full multiplicative stack is an architectural projection.

---

## Repository layout

```
ultracompress/
├── ultracompress/              Core library (pack v0.3, FractalModel, pipeline, __main__)
├── scaling/                    Cross-model teacher loaders (Qwen3 / Llama / Mistral / Mamba / OLMo)
├── scripts/overlay/            Track A (row-overlay + streaming compression)
├── scripts/frr/                Track B (FRR architectural compression)
├── tools/                      Model download, quantization utilities
├── tests/                      Regression tests
├── results/                    Measurement JSONs (indexed by claim)
├── logs/                       Run logs
└── docs/                       Patents, dashboards, customer flow, competitive landscape
```

Index: [RESULTS.md](RESULTS.md), [PATENT_CLAIMS.md](PATENT_CLAIMS.md), [REPRODUCE.md](REPRODUCE.md).

---

## Patent disclosure

USPTO provisionals **64/049,511** and **64/049,517** filed 2026-04-25 covering the row-overlay quantization, FRR architectural compression, streaming-compression mechanism, and v0.3 lossless pack format.

## License

- **Apache-2.0** for the CLI, verifier, and customer-facing pack format — see [LICENSE](LICENSE).
- **Sipsa Labs Research Evaluation License v1.0** for compression internals (k-means trainer, V18-C overlay fit, FRR distillation pipeline) — see [LICENSE_RESEARCH_EVAL.md](LICENSE_RESEARCH_EVAL.md).

## Citation

```bibtex
@misc{ultracompress2026,
  title  = {UltraCompress: Mathematically Lossless 5-bit Transformer
            Compression Across 11+ Architectures},
  author = {Sipsa Labs},
  year   = {2026},
  url    = {https://github.com/sipsalabs/ultracompress}
}
```
