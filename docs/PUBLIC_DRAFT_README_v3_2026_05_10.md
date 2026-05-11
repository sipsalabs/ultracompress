# UltraCompress

> **Mathematically lossless reconstruction of `W_base` for 5-bit transformer weights — 22 architectures validated, Hermes-3-405B at PPL ratio 1.0066x on a single 32 GB consumer GPU.**

[![PyPI](https://img.shields.io/badge/pypi-0.5.5-blue.svg)](https://pypi.org/project/ultracompress/)
[![License](https://img.shields.io/badge/license-BUSL--1.1-orange.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Patent](https://img.shields.io/badge/patent--pending-USPTO%2064%2F049%2C511%20%2B%2064%2F049%2C517-orange.svg)](PATENT_NOTICE.md)

---

## Verify a record in 30 seconds

```bash
pip install ultracompress
hf download SipsaLabs/qwen3-8b-uc-v3-bpw5 --local-dir ./qwen3-8b-uc
uc verify ./qwen3-8b-uc       # SHA-256 bit-identical reconstruction PASS
uc bench  ./qwen3-8b-uc       # TTFT, tokens/sec, decode-TPS, peak VRAM
```

That `uc verify` step is the whole guarantee. It rebuilds `W_base` for every Linear in the model and checks each one against a SHA-256 manifest packed at training time. If a single byte drifts, the verifier fails closed. There is no "trust me" step.

---

## What this is

UltraCompress is a compression layer for trained transformer language models. You give it a Hugging Face checkpoint; it produces a `.uc` artifact at roughly 5 bits per weight that reconstructs `W_base` bit-identically and runs on standard PyTorch `Linear`-shaped operations. The artifact is portable, the verifier is open source, and the reconstruction is cryptographically pinned.

The headline:

- **Hermes-3-Llama-3.1-405B at 5 bpw — PPL ratio 1.0066x vs streaming bf16 teacher.** Single 32 GB consumer GPU (RTX 5090). 250 GB pack, 126 layers. `n_eval=50, seq_len=1024, seed=42`. Same per-layer streaming pipeline used for baseline and compressed sides. First 405B-class artifact compressed and verifiable on a single 32 GB GPU that we know of.
- **22 architectures validated end-to-end** at 5 bpw — Qwen3 (0.6B → 14B), Qwen2.5, Llama-3.1, Mistral, Mixtral (8x7B + 8x22B), Phi-3 / Phi-3.5-MoE, OLMo-2 (Allen Institute), SmolLM2 (HuggingFace), Yi-1.5, TinyLlama, and Mamba-2.8B (the first lossless 5-bit state-space-model artifact we know of).
- **40+ public artifacts** in the [SipsaLabs HuggingFace org](https://huggingface.co/SipsaLabs).

## Tightest verified PPL ratios (BENCHMARKS_2026_05_10.json)

| Model | Params | PPL ratio | Note |
|---|---|---|---|
| Phi-3-mini-4k-instruct | 3.8B | **1.00262x** | seq_len=128 (not apples-to-apples with rest) |
| Mixtral-8x7B-v0.1 | 47B (MoE, 13B active) | **1.00368x** | best MoE result |
| Qwen3-1.7B-Base | 1.7B | **1.00401x** | small-decoder record |
| Qwen3-14B | 14B | **1.00403x** | essentially tied with small-decoder record at 14B scale |
| Yi-1.5-9B | 8.8B | **1.00414x** | >8B record |
| Qwen3-8B | 8.0B | **1.00440x** | 8B-class record |
| Hermes-3-Llama-3.1-405B | 405B | **1.0066x** | 32 GB GPU, 250 GB pack, n=50 seq=1024 |

Eval methodology: 30 prompts × 1024 tokens of held-out FineWeb-edu (Hermes-3-405B uses 50 prompts × 1024 tokens), seed=42, on a single 32 GB consumer GPU. Full table in [BENCHMARKS](https://sipsalabs.com/benchmarks).

## Why this matters

5-bit is the band where lossy quantizers (AWQ, GPTQ, EXL3, HQQ) typically post 1.01x-1.05x perplexity drift and trade away tail behavior. UltraCompress packs at the same nominal bit budget but reconstructs `W_base` bit-identically per a SHA-256 manifest, so the customer's verifier proves the math instead of trusting a benchmark report. A search of the public Hugging Face Hub on 2026-05-09 returned zero other 5-bit lossless transformer artifacts.

## Try the hosted API

[sipsalabs.com/inference](https://sipsalabs.com/inference) — OpenAI-compatible endpoint serving the same 5-bit lossless artifacts you can verify here. $5 of free credits, no card required. -44% on Hermes-3-405B vs Together AI list price.

## License

[BUSL-1.1](LICENSE) with a 4-year Change Date to Apache 2.0. Free for non-production use, evaluation, and research; commercial production use requires a license — `commercial@sipsalabs.com`.

The `0.5.x` series is permanently Apache 2.0 and remains in the [`apache-0.5.x`](https://github.com/sipsalabs/ultracompress/tree/apache-0.5.x) branch. New work lands under BUSL-1.1.

## Patent

USPTO provisionals **64/049,511** and **64/049,517** filed 2026-04-25 — see [PATENT_NOTICE.md](PATENT_NOTICE.md) for scope. Patent-pending status preserves first-publish credit; it does not grant a license.

## Phase 0 POC

Running 70B+ in production? [3 paid POC slots for Q3 2026](https://sipsalabs.com/poc) — $5,000, one week, three customer-picked models, verified packs + benchmark report + SHA-256 manifest. You keep the artifacts and the serving rights.

## Links

- **PyPI**: [pypi.org/project/ultracompress/](https://pypi.org/project/ultracompress/)
- **HuggingFace org**: [huggingface.co/SipsaLabs](https://huggingface.co/SipsaLabs)
- **Site**: [sipsalabs.com](https://sipsalabs.com)
- **Hosted inference**: [sipsalabs.com/inference](https://sipsalabs.com/inference)
- **Benchmarks**: [sipsalabs.com/benchmarks](https://sipsalabs.com/benchmarks)
- **Phase 0 POC**: [sipsalabs.com/poc](https://sipsalabs.com/poc)
- **Contact**: `founder@sipsalabs.com`

---

Sipsa Labs, Inc. — Build the impossible.
