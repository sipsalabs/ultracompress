# Compression methods overview

UltraCompress combines two complementary patent-pending methods:

- **Weight-level compression method** — a patent-pending post-training weight representation — shipping in v0.1
- **Architectural compression method** — a patent-pending architectural compression method — v0.2 (Q3 2026)

This page is a high-level conceptual overview. For implementation specifics, contact `legal@sipsalabs.com` for an NDA-gated technical deep dive.

## Weight-level compression — sub-3-bpw weight representation (v0.1, shipping)

Quantization is the standard approach to model compression: take a 16-bit floating-point weight and store it in fewer bits. The traditional ceiling for "good quality" was **8 bits per weight** (int8). bitsandbytes pushed it to **4 bits** with NF4 in 2023, and HQQ pushed it slightly further with group-wise schemes — but every public method we measured falls off a quality cliff below 4 bpw.

The patent-pending weight-level method is a novel post-training weight representation. In our 6-model benchmark cohort at 2.798 bits per weight:

- ~30% smaller than bitsandbytes NF4 at equivalent retention
- Zero catastrophic failures across the cohort — the only public method we evaluated at this compression frontier with that property in the cohort we tested
- Per-task retention curves (T1, T10, T32, T64, T128, T256) ship in the per-model card on each artifact's Hugging Face Hub repository

For the actual measured numbers and their cohort scope, see [evidence/matrix.md](../evidence/matrix.md).

## Architectural compression (v0.2, Q3 2026)

Where the weight-level method compresses *weights*, the architectural method compresses the *architecture* — restructuring the transformer block to retain expressive capacity at substantially fewer trainable parameters.

Public detail on the architectural method is intentionally limited until v0.2 ships. Public-safe evidence is at [evidence/matrix.md](../evidence/matrix.md). The Q3 2026 v0.2 release timing is gated on patent prosecution; early-access design partners can engage now via the [pilot program](../PILOT_PACKET.md).

## What "catastrophic failure" means

We use a published `T_cat` threshold: any cohort member whose perplexity ratio exceeds 10× the FP16 baseline is a catastrophic failure. HQQ at 2-bit and lower produces models that cross this threshold; the patent-pending method at 2.798 bpw does not, in the cohort we tested. See [catastrophic-failures.md](catastrophic-failures.md).

## What we share publicly vs. under NDA

| Information | Public | NDA |
|---|---|---|
| Cohort-level compression and retention numbers | ✅ | ✅ |
| Validation cohort + benchmark methodology summary | ✅ | ✅ |
| Per-model retention envelope | ✅ (model cards) | ✅ (full breakdown) |
| Reproducibility manifest (SHA-256 file index) | reference only | full manifest |
| Weight-level operating-point parameters and codebook structure | — | ✅ |
| Architectural method mechanism and specification | — | ✅ |
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
- The architectural method ships in Q3 2026, gated on patent prosecution timing
- Future research is under active patent strategy; out of scope for public discussion

Codec internals and training procedure are patent-pending.
