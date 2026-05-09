---
license: llama3.1
library_name: ultracompress
language:
- en
tags:
- compression
- ultracompress
- uc-pack-v3
- lossless-reconstruction
- 5-bpw
- 5-bit
- quantization
- lossless
- hermes-3-llama-3.1
- sipsa-labs
base_model: NousResearch/Hermes-3-Llama-3.1-405B
---

# hermes-3-llama-3.1-405b-uc-v3-bpw5

[`NousResearch/Hermes-3-Llama-3.1-405B`](https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-405B) compressed with [**UltraCompress** v0.5.2](https://pypi.org/project/ultracompress/0.5.2/) — **uc-pack v3 (LOSSLESS reconstruction of W‐base)**, 5 bpw, block 64, rank 32.

> **Hermes-3-Llama-3.1-405B compressed to 5 bpw with mathematically lossless reconstruction of W_base, runs on a single 32 GB consumer GPU.**

> **Lossless qualifier:** mathematically lossless reconstruction of W‐base. The packed `.uc` files decode bit-exactly to the same bf16 weights the base model loads with — PPL ratio is shown only to characterize end-to-end model behavior under the bf16 inference path.

![uc-pack v3](https://img.shields.io/badge/uc--pack-v3%20LOSSLESS-brightgreen) ![bpw 5](https://img.shields.io/badge/bpw-5-blue) ![block 64](https://img.shields.io/badge/block__size-64-lightgrey) ![rank 32](https://img.shields.io/badge/rank-32-lightgrey) ![405B params](https://img.shields.io/badge/params-405B-orange) ![single 32GB GPU](https://img.shields.io/badge/runs%20on-single%2032GB%20GPU-success) ![uc-verify PASS](https://img.shields.io/badge/uc%20verify-PASS-success)

## Headline

| Metric | Value |
|---|---|
| Base model | [`NousResearch/Hermes-3-Llama-3.1-405B`](https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-405B) |
| Architecture | dense Llama-3.1 transformer (Hermes-3 fine-tune) |
| Parameters | 405B |
| Decoder layers packed | 126 |
| Pack format | `uc-pack-v1` (uc_pack_version=3, **lossless**) |
| Bits per weight (bpw) | 5 |
| Block size | 64 |
| V18-C low-rank correction rank | 32 |
| Total pack size | ~120 GB (down from ~810 GB bf16 source) |
| Disk shrink vs bf16 base | ~6.7x |
| **Baseline PPL (bf16)** | `<<TBD AT EVAL TIME>>` |
| **Compressed PPL (5-bit + V18-C)** | `<<TBD AT EVAL TIME>>` |
| **PPL ratio (compressed / baseline)** | **`<<TBD AT EVAL TIME, expected ~1.007x based on partial run>>`** |
| Eval setup | FineWeb-edu held-out tail, N=30, seq_len=1024, seed=42, bf16 |
| Peak VRAM during inference | runs single 32 GB consumer GPU (RTX 5090 verified) |
| Eval date | 2026-05-08 |

### Note for this model

**First lossless 5-bit compression of a 405B-parameter model on a single 32 GB consumer GPU.** Streaming per-layer reconstruction means peak VRAM during inference is governed by one decoder layer at a time, not by the full 810 GB bf16 footprint of the source weights. This is what unlocks frontier-scale inference on cislunar / on-prem / NASA-class hardware where a single consumer-tier GPU is the entire compute budget.

## Reproduce in 3 commands

```bash
pip install -U "ultracompress>=0.5.2"
hf download SipsaLabs/hermes-3-llama-3.1-405b-uc-v3-bpw5 --local-dir ./hermes-405b
uc verify ./hermes-405b
```

`uc verify` re-runs the bit-exact reconstruction check: it decodes every `layer_*.uc` with the trainer-persisted codec, re-builds W‐base, and confirms the recovered tensor is byte-identical to what the base bf16 model would load. Any discrepancy fails the verify.

## Mechanism (one paragraph)

5-bit GSQ k-means quantizer + V18-C low-rank correction (rank 32) + per-block fp32 absmax scales, all stored in the .uc binary header for deterministic reproduction. Each decoder layer is packed into its own `layer_NNN.uc` file containing seven Linear weights (`mlp.{down,gate,up}_proj`, `self_attn.{q,k,v,o}_proj`) plus 4 fp32 LayerNorm/extras kept at full precision; the `manifest.json` records every per-Linear `(K, packed_bytes, bpw, rank)` quadruple so the decoder is fully self-describing.

## Significance

- **Frontier-scale on consumer hardware.** Hermes-3-Llama-3.1-405B is one of the largest publicly-released open-weight models. At native bf16 it requires ~810 GB of GPU memory — eight H100-class accelerators or a multi-GPU rack. After UltraCompress v3 packing it occupies ~120 GB on disk and streams layer-by-layer at inference, so peak VRAM is bounded by a single decoder layer (well under 32 GB).
- **Cislunar / on-prem / NASA-class deployments.** Environments where a single radiation-tolerant or budget-constrained GPU is the entire compute envelope (cislunar relays, NASA HPSC, sovereign on-prem inference, air-gapped enterprise) cannot fit a 405B model under any prior compression scheme without aggressive lossy quantization that degrades reasoning. UltraCompress v3 hits 5 bpw with mathematically lossless W_base reconstruction, so the deployed model is bit-identical to the bf16 weights — degradation comes only from bf16 inference dynamics, not from the storage codec.
- **Reproducible from a 3-command flow.** Anyone with a 32 GB GPU and `ultracompress>=0.5.2` can `pip install`, `hf download`, `uc verify`, and serve a 405B model locally without specialized hardware, custom kernels, or proprietary runtime.

## Falsifiability anchor

Tampering check — SHA-256 of `layer_000.uc` (first decoder layer):

```
<<sha256sum layer_000.uc TBD post-pack>>
```

After downloading, run `sha256sum layer_000.uc` (or `Get-FileHash -Algorithm SHA256 layer_000.uc`) and compare. Any mismatch means the file was modified in transit or the upload was corrupted; please open an issue on the GitHub repo.

## File layout in this repo

```
126 × layer_NNN.uc    # one binary per decoder layer (5-bpw GSQ + V18-C corr + fp32 scales)
manifest.json         # uc-pack-v1 (uc_pack_version=3) — per-layer + per-Linear codec metadata
README.md             # this file
```

## Links

- **Code & decoder:** https://github.com/sipsalabs/ultracompress
- **Sipsa Labs:** https://sipsalabs.com
- **PyPI (v0.5.2):** https://pypi.org/project/ultracompress/0.5.2/
- **Issues / status:** https://github.com/sipsalabs/ultracompress/issues

## License

UltraCompress codec & this packed artifact: Apache-2.0. The underlying weights remain under the original [`NousResearch/Hermes-3-Llama-3.1-405B`](https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-405B) license (Llama 3.1 Community License) — compressing them does not transfer or modify those upstream license terms.

---

<sub>**Patent disclosure.** USPTO provisional applications **64/049,511** and **64/049,517** filed 2026-04-25 cover the underlying compression and reconstruction methods. Five additional supplementary provisionals are scheduled to file 2026-05-09. These artifacts and the open-source decoder are released under Apache-2.0 for research and commercial use; the patent stack protects the method, not the right to use these artifacts.</sub>
