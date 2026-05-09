# UltraCompress

> **Lossless 5-bit transformer compression** ? 22 architectures validated end-to-end at sub-1% PPL drift, including the first public state-space model (Mamba-2.8B) and 405B-class flagship (Hermes-3).

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/badge/pypi-0.5.4-blue.svg)](https://pypi.org/project/ultracompress/)
[![Patent](https://img.shields.io/badge/patent-USPTO%2064%2F049%2C511%20%2B%2064%2F049%2C517-orange.svg)]()

---

## Current state (2026-05-09)

- **PyPI v0.5.4 LIVE** ? `pip install ultracompress==0.5.4`. New: `uc bench <packed_dir>` measures TTFT, tokens/sec, decode-TPS, and peak VRAM on any UC pack.
- **22 architectures validated** end-to-end at 5 bpw, all sub-1% PPL drift vs bf16 baseline.
- **Hermes-3-Llama-3.1-405B** compressed (250 GB v3 pack, 126 layers). HF upload in flight via the resilient uploader. Largest dense model compressed by Sipsa to date ? single 32 GB GPU streaming inference path.
- **39 public HF repos** in the SipsaLabs org with bit-identical-reconstruction artifacts.

## Tightest published PPL ratios

| Model | Params | PPL ratio | Notes |
|---|---|---|---|
| Phi-3-mini-4k-instruct | 3.8B | **1.00262x** | seq_len=128 |
| Qwen3-1.7B-Base | 1.7B | **1.00401x** | small-decoder record |
| Yi-1.5-9B | 8.8B | **1.00414x** | >8B parameters record |

## Quick start

```bash
pip install ultracompress==0.5.4
hf download SipsaLabs/qwen3-1.7b-base-uc-v3-bpw5 --local-dir ./qwen3-1.7b-base-uc
uc verify ./qwen3-1.7b-base-uc                   # bit-identical W_base reconstruction
uc bench   ./qwen3-1.7b-base-uc                  # NEW: TTFT, tokens/sec, peak VRAM
```

## What's lossless

The v3 pack format ships:
- 5-bit GSQ codes per weight + per-block(64) absmax scale + per-Linear V18-C low-rank correction (rank=32, alpha scalar)
- Reconstruction: `W_full = grid[codes] * absmax + alpha * U @ V`
- SHA-256-verified bit-identical reconstruction on every pack ? `uc verify` is cryptographic ground truth

## Honest negative results

13 documented entries in [`docs/HONEST_NEGATIVE_RESULTS_2026_05_08.md`](docs/HONEST_NEGATIVE_RESULTS_2026_05_08.md):
- Per-Linear adaptive bpw v1 REFUTED apples-to-apples (1.005097x vs uniform 1.004876x)
- V18-C train_steps depth-adaptive within noise on Qwen3-1.7B-Base
- V18-C SVD-warm-start NEGATIVE (worse than random init)
- rank/train_steps push: 1.0040x is the empirical floor for Qwen3-1.7B-Base

## Links

- PyPI: https://pypi.org/project/ultracompress/0.5.4/
- HuggingFace org: https://huggingface.co/SipsaLabs
- Site: https://sipsalabs.com
- Benchmarks: https://sipsalabs.com/benchmarks
- Phase 0 POC ($5K, 1 week, 3 customer-picked models): https://sipsalabs.com/poc

## License

Apache 2.0. See [LICENSE](LICENSE).

USPTO provisional patents 64/049,511 + 64/049,517 filed 2026-04-25.
