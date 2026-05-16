# Compression methods overview

UltraCompress ships a patent-pending lossless 5-bit pack format with bit-identical reconstruction guaranteed by a SHA-256 manifest.

This page is a high-level conceptual overview. For implementation specifics, contact `legal@sipsalabs.com` for an NDA-gated technical deep dive.

## Lossless 5-bit compression (shipping)

Quantization is the standard approach to model compression: take a 16-bit floating-point weight and store it in fewer bits. The traditional ceiling for "good quality" was **8 bits per weight** (int8). bitsandbytes pushed it to **4 bits** with NF4 in 2023, and HQQ pushed it slightly further with group-wise schemes — but those methods drift relative to the original weights.

The patent-pending UltraCompress method produces a lossless 5-bit pack: the reconstruction is a deterministic dequantization that is mathematically bit-identical to the trainer-quantized weights, verifiable against a SHA-256 manifest. Across the published architecture matrix:

- 22 architectures shipped end-to-end; 14 PPL-verified end-to-end against their bf16 baseline
- Bit-identical reconstruction — an auditor can re-derive every weight from the pack alone
- Per-model PPL ratios ship in the per-model card on each artifact's Hugging Face Hub repository

For the actual measured numbers and their scope, see [evidence/matrix.md](../evidence/matrix.md).

## Quality measure

Each published artifact reports an end-to-end perplexity ratio against its bf16 baseline (FineWeb-edu held-out tail, seq_len=1024, seed=42). Published numbers reflect what was measured; pending evals are labeled as pending. See [catastrophic-failures.md](catastrophic-failures.md).

## What we share publicly vs. under NDA

| Information | Public | NDA |
|---|---|---|
| Cohort-level compression and retention numbers | ✅ | ✅ |
| Validation cohort + benchmark methodology summary | ✅ | ✅ |
| Per-model retention envelope | ✅ (model cards) | ✅ (full breakdown) |
| Reproducibility manifest (SHA-256 file index) | reference only | full manifest |
| Operating-point parameters and codebook structure | — | ✅ |
| Codec mechanism and specification | — | ✅ |
| Patent specifications (filed April 2026) | — | ✅ when public record |

If you need NDA access to the technical deep dive, email `legal@sipsalabs.com`.

## Reproducibility

Every public number ships with:

- A deterministic seed (default seed = 42 across all runs)
- A full sample count (no cherry-picked best-of-N)
- A multi-model cohort (no single-model fluke results)
- A SHA-256-verified manifest of the input artifacts

This is increasingly a procurement gate for enterprise customers. See [reproducibility.md](reproducibility.md) for the full reproducibility commitment.

## What's next

- A patent supplement extends claim scope; details available under NDA
- `uc compress` self-compression ships in Q3 2026, gated on patent prosecution timing
- Future research is under active patent strategy; out of scope for public discussion

Codec internals and training procedure are patent-pending.
