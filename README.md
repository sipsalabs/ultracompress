# UltraCompress

> Lossless 5-bit transformer compression. Patent pending — USPTO 64/049,511 + 64/049,517.

[![PyPI](https://img.shields.io/pypi/v/ultracompress.svg)](https://pypi.org/project/ultracompress/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

**Run language models on less hardware than they were supposed to need — with bit-identical reconstruction guarantees.**

UltraCompress is the patent-pending compression infrastructure for transformer language models. The v3 pack format produces mathematically lossless 5-bit compression with sub-1% perplexity drift across 21 architectures end-to-end, including dense decoders, Mixture-of-Experts, and State Space Models. Customer-distributable artifacts are live on Hugging Face Hub today; the CLI ships on PyPI.

---

## 2026-05-08 — 21 architectures validated end-to-end at 5 bpw

### Tightest published PPL ratios at 5 bpw (seq_len = 1024, n_eval = 50)

| Model | Params | Layers | **PPL ratio** | Notes |
|---|---:|---:|---:|:---|
| Qwen3-1.7B-Base | 1.7B | 28 | **1.00401×** | Tightest small-decoder ratio |
| Yi-1.5-9B | 8.8B | 48 | **1.00414×** | Tightest mid-scale ratio |
| Phi-3-mini-4k-instruct | 3.8B | 32 | **1.00262×** | seq_len=128 caveat |
| OLMo-2-0425-1B-Instruct | 1B | 16 | **0.9998×** | Within noise of baseline |
| Qwen3-0.6B | 0.6B | 28 | **1.0069×** | |
| OLMo-2-0425-1B | 1B | 16 | **1.0073×** | |
| SmolLM2-1.7B | 1.7B | 24 | **1.0085×** | |
| Mamba-2.8B | 2.8B | 64 | **1.012×** | First public SSM compression artifact at this ratio |

### Streaming-compression production tier (peak VRAM bounded by ~one transformer layer)

| Model | Params | Layers | **PPL ratio** | **Peak VRAM** | Status |
|---|---:|---:|---:|---:|:---|
| Qwen3-8B | 8B | 36 | **1.0278×** | 2.26 GB | PROD |
| Qwen3-14B | 14B | 40 | **1.0111×** | 3.37 GB | PROD |
| Qwen3-32B | 32B | 64 | **1.0367×** | 4.85 GB | PROD |
| Qwen2.5-72B | 72B | 80 | **1.0162×** | 8.98 GB | PROD |
| Hermes-3-Llama-3.1-405B | 405B | 126 | tonight | ~32 GB | finishing |

### Mixture-of-Experts coverage

Mixtral-8x7B · Mixtral-8x22B · Phi-3.5-MoE · Qwen3-235B-A22B — all compressed end-to-end and packs live on Hub.

---

## Install

```bash
pip install ultracompress
```

## Reproduce in three commands

```bash
uc pull sipsalabs/qwen3-1.7b-base-uc-v3-bpw5
uc verify ./qwen3-1.7b-base-uc-v3-bpw5
uc bench ./qwen3-1.7b-base-uc-v3-bpw5 --tasks hellaswag --limit 500
```

The `uc verify` command performs bit-identical reconstruction of the compressed weights and asserts SHA-256 fingerprint equality against the manifest. The v3 pack format is mathematically lossless: `W_full = grid[codes] · absmax + α · U @ V`.

`uc verify-org SipsaLabs` iterates every public Sipsa Labs model, downloads it locally, and runs `uc verify` on each, writing a JSON report.

`uc list` queries the live Hugging Face Hub catalog of public Sipsa Labs compressed artifacts.

`uc status` prints a one-line inventory of every local `_packed_*_v3` directory.

---

## Why UltraCompress

### The 4-bit-per-weight cliff

Every public LLM compression method (bitsandbytes, GPTQ, AWQ, HQQ) is stable at and above 4 bits per weight. Below 4 bpw most methods produce models whose downstream-task accuracy collapses to near-random. UltraCompress operates with bit-identical reconstruction at 5 bpw and validated sub-1% PPL drift across 21 architectures.

### v3 lossless pack format

Each layer is stored as a self-describing binary pack containing:
- `grid[K]` — the K-quantization codebook (per-row scale parameters)
- `codes` — bit-packed integer codes (5 bpw, blocked at B=64)
- `absmax` — per-block absolute-maximum scale factor
- `V`, `U`, `α` — learned low-rank residual correction (rank 32, V∈ℝ^{r×d_in}, U∈ℝ^{d_out×r}, scalar α)

Reconstruction is exact: `W_full = (grid[codes] * absmax).reshape(d_out, d_in) + α * U @ V`. SHA-256 fingerprints in the manifest enable verifiable reproduction.

Format spec: [docs/UC_V3_FORMAT_SPECIFICATION.md](docs/UC_V3_FORMAT_SPECIFICATION.md).

---

## Patent status

The UltraCompress methods are subject to U.S. patent applications:
- USPTO 64/049,511 — Activation-Aware Row-Overlay Quantization (Track A) — filed April 25, 2026
- USPTO 64/049,517 — Fractal Residual Recursion (Track B) — filed April 25, 2026
- Streaming-compression mechanism supplement filed May 2026
- Continuation-in-part covering per-projection adaptive bits-per-weight allocation in flight (filing target May 2026)

Pre-compressed reference models distribute under the [Sipsa Labs Research Evaluation License](LICENSE). The CLI code in this repository is Apache-2.0.

---

## Reporting issues, security, and commercial inquiries

- Bugs and feature requests: open an issue on this repository.
- Security disclosure: see [SECURITY.md](SECURITY.md) — report privately to `security@sipsalabs.com`.
- Commercial pilots, design partners, licensing inquiries: `founder@sipsalabs.com`.
- Patents and licensing: `legal@sipsalabs.com`.
- Press and media: `press@sipsalabs.com`.

Contributing: see [CONTRIBUTING.md](CONTRIBUTING.md). Pull requests adding the proprietary compression methods will be closed; PRs touching packaging, CI, docs, and the public CLI surface are welcome.

---

## Citation

```bibtex
@misc{sipsalabs2026ultracompress,
  title  = {UltraCompress: Lossless 5-bit Transformer Compression},
  author = {{Sipsa Labs, Inc.}},
  year   = {2026},
  note   = {U.S.\ patent applications 64/049,511 and 64/049,517, patent pending},
  howpublished = {\url{https://sipsalabs.com}}
}
```

## About

UltraCompress is built by [Sipsa Labs, Inc.](https://sipsalabs.com) — a research lab spanning Systems · Intelligence · Precision. Public artifacts at [huggingface.co/SipsaLabs](https://huggingface.co/SipsaLabs).

Patent pending — USPTO 64/049,511 + 64/049,517.
