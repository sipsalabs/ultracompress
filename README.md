# UltraCompress

> Extreme compression for large language models. Patent pending — USPTO 64/049,511 + 64/049,517

[![PyPI](https://img.shields.io/pypi/v/ultracompress.svg)](https://pypi.org/project/ultracompress/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Run large language models on less hardware.
UltraCompress publishes pre-compressed reference language models at sub-3 bits per weight — **30% smaller than bitsandbytes NF4** with **zero catastrophic failures** on a 6-model head-to-head cohort. The underlying methods are patent pending; this CLI lets you **download pre-compressed reference models** and run them locally.

> **v0.1 alpha**: pre-compressed reference models are uploading to Hugging Face Hub throughout April–May 2026. Run `uc list` for the live catalog. Examples below show expected post-launch usage.

## Install

```bash
pip install ultracompress
```

## Quickstart

```bash
# List pre-compressed models available on the official Hugging Face Hub
uc list

# Download a pre-compressed model
uc pull sipsalabs/<model-id>

# Inspect what's in a compressed artifact
uc info ./models/<model-id>

# Benchmark the compressed model against the fp16 teacher
uc bench ./models/<model-id> --tasks hellaswag --limit 500
```

## What's available today (v0.1 — alpha)

- `uc list` — browse pre-compressed models from our Hugging Face Hub collection.
- `uc pull <model-id>` — download a pre-compressed model locally.
- `uc info <path>` — inspect the compression metadata of an artifact.
- `uc eval <path> --tasks <list>` — run downstream benchmarks via `lm-eval-harness` on the compressed model.

## What's coming (v0.2 — Q3 2026)

- `uc compress <hf-model-id> --bpw 2.8` — self-compression (gated on patent prosecution timeline).
- `uc serve <path>` — inference server with OpenAI-compatible API.
- `uc export --format gguf` — export to llama.cpp GGUF format.
- `uc export --format coreml` — export to Apple CoreML for on-device inference.

## Why UltraCompress

On a 6-model × 8-method × 500-sample head-to-head benchmark:

| Method | Bits per weight | Cohort median T1 retention | Catastrophic failures |
|---|---:|---:|---:|
| bitsandbytes int8 | 8.000 | 99.75% | 0/6 |
| bitsandbytes nf4 | 4.000 | 98.31% | 0/6 |
| HQQ 4-bit g64 | 4.500 | 97.72% | 0/6 |
| **UltraCompress 2.8 bpw** | **2.798** | **95.63%** | **0/6** |
| HQQ 3-bit g64 | 3.500 | 72.46% | 1/6 |
| HQQ 2-bit g64 | 2.500 | 3.46% | 6/6 |

UltraCompress is the only sub-3-bpw method on this cohort that produces zero catastrophic failures.

## Patent status

The UltraCompress compression methods are the subject of pending U.S. patent applications. Pre-compressed models are distributed under a separate licensing arrangement described in [LICENSE](LICENSE). The CLI code in this repository is Apache-2.0.

## Reporting issues, security, and commercial inquiries

- Bugs and feature requests: open an issue.
- Security vulnerabilities: see [SECURITY.md](SECURITY.md) — report privately to `security@sipsalabs.com`.
- Commercial / design-partner / pilot inquiries: `founder@sipsalabs.com`.
- Patent / licensing: `legal@sipsalabs.com`.

Contributing: see [CONTRIBUTING.md](CONTRIBUTING.md). Changes that touch packaging, CI, docs, and the public CLI surface are very welcome. Pull requests adding the proprietary compression methods will be closed.

## Citation

```bibtex
@misc{sipsalabs2026ultracompress,
  title        = {UltraCompress: Extreme Compression for Large Language Models},
  author       = {{Sipsa Labs, Inc.}},
  year         = {2026},
  note         = {U.S.\ patent applications 64/049,511 and 64/049,517, patent pending},
  howpublished = {\url{https://sipsalabs.com}}
}
```

## About

UltraCompress is built by [Sipsa Labs](https://sipsalabs.com) — a research lab spanning Systems · Intelligence · Precision.

Patent pending — USPTO 64/049,511 + 64/049,517.
