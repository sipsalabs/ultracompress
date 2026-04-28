# UltraCompress

> Extreme compression for large language models.

[![PyPI](https://img.shields.io/pypi/v/ultracompress.svg)](https://pypi.org/project/ultracompress/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/sipsalabs/ultracompress/blob/main/LICENSE)
[![CI](https://github.com/sipsalabs/ultracompress/actions/workflows/ci.yml/badge.svg)](https://github.com/sipsalabs/ultracompress/actions)

UltraCompress shrinks transformer language models below the 4-bits-per-weight floor that has stumped every prior open-source method, with **zero catastrophic failures** on a 6-model head-to-head benchmark.

!!! info "v0.1 alpha"
    Pre-compressed reference models are uploading to the Hugging Face Hub throughout April–May 2026. Run `uc list` for the live catalog at any time.

The compression methods are the subject of pending U.S. patent applications (USPTO 64/049,511 and 64/049,517, filed 2026-04-25). This CLI is the open-source distribution layer; pre-compressed reference models roll out through the [`sipsalabs`](https://huggingface.co/sipsalabs) organization on the Hugging Face Hub through April–May 2026 — `uc list` shows the live catalog at any time.

## Install

=== "pip"

    ```bash
    pip install ultracompress
    ```

=== "uv"

    ```bash
    uv add ultracompress
    ```

=== "From source"

    ```bash
    git clone https://github.com/sipsalabs/ultracompress.git
    cd ultracompress
    pip install -e ".[dev]"
    ```

## 60-second quickstart

```bash
# Browse the official catalog
uc list

# Download a pre-compressed model
uc pull sipsalabs/<model-id>

# Inspect what's in it
uc info ./models/<model-id>

# Run downstream benchmarks
uc bench ./models/<model-id> --tasks hellaswag --limit 500
```

## What's in a pre-compressed artifact

Each artifact is a directory with:

- `model.safetensors` — quantized weights in our compressed format
- `ultracompress.json` — provenance manifest (bpw, base model ID, SHA-256 of weights, license, method version)
- `tokenizer/` — pre-loaded tokenizer matching the base model
- `LICENSE` — the per-model license (research-free or commercial-paid; contact `legal@sipsalabs.com`)

## Why we exist

The published methods most teams use (bitsandbytes, GPTQ, AWQ, HQQ) all hit a wall at 4 bits per weight. Below 4 bits they collapse with catastrophic quality loss. We push past the wall:

| Method | Bits per weight | Quality retention (cohort median) | Catastrophic failures |
|---|---:|---:|---:|
| bitsandbytes int8 | 8.000 | 99.75% | 0/6 |
| bitsandbytes NF4 | 4.000 | 98.31% | 0/6 |
| HQQ 4-bit g64 | 4.500 | 97.72% | 0/6 |
| **UltraCompress 2.8 bpw** | **2.798** | **95.63%** | **0/6** |
| HQQ 3-bit g64 | 3.500 | 72.46% | 1/6 |
| HQQ 2-bit g64 | 2.500 | 3.46% | 6/6 |

Source: 6-model × 8-method × 500-sample head-to-head benchmark on WikiText-103 perplexity ratio, deterministic seed, full SHA-256 verification manifest available under NDA.

## Where to go next

- **First time here?** [Quickstart](getting-started/quickstart.md)
- **Want to understand the methods?** [Compression methods overview](concepts/compression-methods.md)
- **Need to integrate with your inference stack?** [Integration guides](integration/llamacpp.md)
- **Looking for a specific model?** Run `uc list` for the live catalog.
- **Deploying in a commercial product?** Email `legal@sipsalabs.com`.

## Status

UltraCompress is in **public alpha** as of v0.1.0 (April 2026). The CLI is stable for `list`, `pull`, `info`, `bench`. Self-compression (`uc compress <model>`) is intentionally not yet shipped — it depends on the patent-pending compression methods being formally protected. Targeted v0.2 release: late Q3 2026.

## Stay in touch

- **Website**: [sipsalabs.com](https://sipsalabs.com)
- **GitHub**: [github.com/sipsalabs/ultracompress](https://github.com/sipsalabs/ultracompress)
- **Hugging Face**: [huggingface.co/sipsalabs](https://huggingface.co/sipsalabs)
- **PyPI**: [pypi.org/project/ultracompress](https://pypi.org/project/ultracompress/)
- **Twitter**: [@sipsalabs](https://x.com/sipsalabs)
- **Email**: `founder@sipsalabs.com` for commercial / partnership inquiries
