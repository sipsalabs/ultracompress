# UltraCompress

> Extreme compression for large language models.

[![PyPI](https://img.shields.io/pypi/v/ultracompress.svg)](https://pypi.org/project/ultracompress/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/sipsalabs/ultracompress/blob/main/LICENSE)
[![CI](https://github.com/sipsalabs/ultracompress/actions/workflows/ci.yml/badge.svg)](https://github.com/sipsalabs/ultracompress/actions)

UltraCompress produces **near-lossless 5-bit** transformer packs (~1% perplexity vs the bf16 reference; lossy) with **reproducible, cryptographically verifiable reconstruction** — a deterministic decode to the SHA-256-pinned validated artifact (a ~1% PPL reconstruction of the bf16 source, not a bit-identical copy of it) — 22 architectures shipped end-to-end, all 22 PPL-verified end-to-end against their bf16 baseline (plus 1 ViT cosine-verified).

!!! info "v0.6.11"
    Pre-compressed reference models are published on the Hugging Face Hub. Run `uc list` for the live catalog at any time.

The compression methods are the subject of pending USPTO provisional applications filed in April 2026. This CLI is the open-source distribution layer; pre-compressed reference models roll out through the [`sipsalabs`](https://huggingface.co/sipsalabs) organization on the Hugging Face Hub through April–May 2026 — `uc list` shows the live catalog at any time.

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
uc catalog

# Download a pre-compressed model (via HuggingFace CLI)
huggingface-cli download SipsaLabs/<repo-id> --local-dir ./<repo-id>

# Verify download integrity against the SHA-256 manifest
uc verify ./<repo-id>

# Inspect manifest
uc info ./<repo-id>
```

## What's in a pre-compressed artifact

Each artifact is a directory with:

- `model.safetensors` — quantized weights in our compressed format
- `ultracompress.json` — provenance manifest (bpw, base model ID, SHA-256 of weights, license, method version)
- `tokenizer/` — pre-loaded tokenizer matching the base model
- `LICENSE` — the per-model license (research-free or commercial-paid; contact `legal@sipsalabs.com`)

## Why we exist

The published methods most teams use (bitsandbytes, GPTQ, AWQ, HQQ) are all lossy — they drift relative to the original weights, with no verifiable contract on what you get back. UltraCompress ships a *near-lossless* 5-bit pack (~1% perplexity vs the bf16 reference; also lossy) with a reproducible, cryptographically verifiable reconstruction: a deterministic decode to the SHA-256-pinned validated artifact, verifiable against its manifest.

| Method | Bits per weight | Reconstruction | Catastrophic failures |
|---|---:|---:|---:|
| bitsandbytes int8 | 8.000 | lossy | 0/6 |
| bitsandbytes NF4 | 4.000 | lossy | 0/6 |
| HQQ 4-bit g64 | 4.500 | lossy | 0/6 |
| **UltraCompress 5 bpw** | **5.000** | **near-lossless (~1% PPL), verifiable** | **0/6** |
| HQQ 3-bit g64 | 3.500 | lossy | 1/6 |
| HQQ 2-bit g64 | 2.500 | lossy | 6/6 |

22 architectures PPL-verified end-to-end against their bf16 baseline (FineWeb-edu held-out tail, seq_len=1024, seed=42), plus 1 ViT cosine-verified. Every published number traces to a JSON receipt.

## Where to go next

- **First time here?** [Quickstart](getting-started/quickstart.md)
- **Want to understand the methods?** [Compression methods overview](concepts/compression-methods.md)
- **Need to integrate with your inference stack?** [Integration guides](integration/llamacpp.md)
- **Looking for a specific model?** Run `uc list` for the live catalog.
- **Deploying in a commercial product?** Email `legal@sipsalabs.com`.

## Status

UltraCompress is public as of v0.6.11. The CLI is stable for `list`, `pull`, `info`, `bench`, `verify`, `pack`. Self-compression (`uc compress <model>`) is intentionally not yet shipped — it depends on the patent-pending compression methods being formally protected. Targeted release: late Q3 2026.

## Stay in touch

- **Website**: [sipsalabs.com](https://sipsalabs.com)
- **GitHub**: [github.com/sipsalabs/ultracompress](https://github.com/sipsalabs/ultracompress)
- **Hugging Face**: [huggingface.co/sipsalabs](https://huggingface.co/sipsalabs)
- **PyPI**: [pypi.org/project/ultracompress](https://pypi.org/project/ultracompress/)
- **Twitter**: [@sipsalabs](https://x.com/sipsalabs)
- **Email**: `founder@sipsalabs.com` for commercial / partnership inquiries
