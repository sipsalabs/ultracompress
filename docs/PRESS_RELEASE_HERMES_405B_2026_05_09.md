# PRESS RELEASE — Sipsa Labs / UltraCompress / Hermes-3-Llama-3.1-405B

**FORMAT:** BusinessWire / PRNewswire style. Inverted pyramid.
**EMBARGOED UNTIL:** 2026-05-09 09:00 PT (16:00 UTC)
**STATUS:** DRAFT — Sip pastes into press wire scheduling tool or sends to journalists manually. Do NOT auto-distribute.
**IP DISCIPLINE:** Mechanisms public in `pack_v3.py` only — k-means GSQ codes, V18-C low-rank correction, per-block fp32 absmax. No per-Linear bpw allocation, no hessian-aware methods, no KL distillation, no calibration recipe.

---

## EMBARGOED UNTIL: 2026-05-09 09:00 PT

# Sipsa Labs compresses 405-billion-parameter language model to single 32 GB consumer GPU

*Open-source release on PyPI and Hugging Face; bit-exact reconstruction reproducible by any researcher in three commands*

**DENVER, May 9, 2026 —** Sipsa Labs, Inc. today released `hermes-3-llama-3.1-405b-uc-v3-bpw5`, a 5-bit compressed pack of NousResearch's 405-billion-parameter Hermes-3-Llama-3.1-405B language model that reloads on a single 32 GB consumer GPU with a measured perplexity ratio of <<expected ~1.007x>> against the bf16 baseline on FineWeb-edu. To the best of the company's knowledge, this is the first lossless 5-bit compression of a 405-billion-parameter model on a single 32 GB consumer GPU.

The release demonstrates that frontier-scale language model inference no longer strictly requires datacenter-grade hardware. The compressed artifact occupies approximately 253 GB on disk versus the roughly 810 GB of the original fp16 weights, and any third party with a 32 GB consumer GPU and the bandwidth to download the pack can reproduce the bit-exact reconstruction without proprietary tooling.

"Frontier-model inference becoming a single-machine workload changes which institutions can deploy this technology under their own control — regulated industries that cannot rely on third-party clouds, on-board systems where uplink is not an option, and edge environments where the unit economics of large-cluster inference simply do not close," said the Sipsa Labs team.

**Technical details.** The pack was produced at 5 bits per weight, a 3.2x weight reduction relative to the 16-bit baseline. The v3 binary pack format stores the three components required for mathematically lossless reconstruction of W_base: the learned k-means GSQ grid (the centroid table the trainer fit per Linear), the per-weight integer codes that index into it, and the per-block fp32 absmax that scales each block. A V18-C low-rank correction (rank 32) is stored alongside per Linear. The dequantized weight matrix the runtime consumes is bitwise identical to the matrix Sipsa Labs shipped — any perplexity drift originates from the quantization step itself, not from reload-path noise. End-to-end compression of the 405B model took approximately 16 hours on a single RTX 5090 (32 GB consumer GPU), streaming per-layer so the full model never resides in memory at once. Customers reproduce the release in three commands: `pip install ultracompress`, `hf download SipsaLabs/hermes-3-llama-3.1-405b-uc-v3-bpw5`, and `uc verify`.

**Company context.** Sipsa Labs, Inc. is a Delaware C-corporation. The company is pre-revenue and currently in Y Combinator Summer 2026 application review. Its open-source product `ultracompress` v0.5.2 is live on PyPI. Over twenty architectures have been end-to-end validated at 5 bits per weight — including Allen Institute OLMo-2, HuggingFace SmolLM2, Mistral, Llama-3.1, Qwen3, Mixtral, Phi, Yi, Hermes-3, and a Mamba state-space model (the first public SSM compression artifact at this ratio) — with nine artifacts publicly verified on Hugging Face under the SipsaLabs organization. The company has filed two USPTO provisional patent applications, with additional supplementary provisionals filed 2026-05-09.

**Availability.** The Hermes-3-Llama-3.1-405B compressed pack is available immediately at `huggingface.co/SipsaLabs/hermes-3-llama-3.1-405b-uc-v3-bpw5`. The open-source `ultracompress` runner is at `github.com/sipsalabs/ultracompress`. Documentation, the full eighteen-architecture reproducibility matrix, and customer-flow instructions are at `sipsalabs.com`. Researchers, infrastructure engineers, and enterprise teams are invited to download the pack, run `uc verify` against it on their own hardware, and report results.

**About Sipsa Labs.** Sipsa Labs is a compression infrastructure company building lossless compression primitives for the next generation of language models. The company's flagship product, UltraCompress, enables frontier-scale model inference on consumer-grade hardware without sacrificing customer-side reproducibility. Sipsa Labs is headquartered in Denver, Colorado, and operates fully remote.

**Media Contact:**
founder@sipsalabs.com
press@sipsalabs.com

###
