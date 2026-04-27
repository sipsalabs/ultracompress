# Compression methods overview

UltraCompress combines two complementary patent-pending tracks:

- **Track A** — post-training row-overlay quantization (USPTO 64/049,511)
- **Track B** — Fractal Residual Recursion (USPTO 64/049,517)

This page is a high-level conceptual overview. For implementation specifics — the *how* — please consult the published patent applications (when they become public 18 months after filing) or contact `legal@sipsalabs.com` for an NDA-gated technical deep dive.

## Track A — sub-3-bpw weight representation

Quantization is the standard approach to model compression: take a 16-bit floating-point weight and store it in fewer bits. The traditional ceiling for "good quality" was **8 bits per weight** (int8). bitsandbytes pushed it to **4 bits** with NF4 in 2023, and HQQ pushed it slightly further with group-wise schemes — but every public method falls off a quality cliff below 4 bpw.

Track A is a novel weight representation that defeats this cliff. On a 6-model × 8-method × 500-sample head-to-head benchmark, Track A at 2.798 bits per weight produces:

- 95.6% T1 retention (cohort median)
- 30% smaller than bitsandbytes NF4 at equivalent retention
- The only sub-3-bpw method on the cohort with **zero catastrophic failures**

**What "catastrophic failure" means**: HQQ at 2-bit and lower produces models whose downstream-task accuracy collapses to near-random. We measure this with a `T_cat` threshold: any cohort member whose perplexity ratio exceeds 10× the FP16 baseline is a catastrophic failure.

## Track B — architectural compression

Quantization compresses *weights*. Architectural compression compresses the *architecture itself* — by structurally factorizing the transformer block to use fewer trainable parameters with the same expressive capacity.

The closest published prior art is Google DeepMind's **Relaxed Recursive Transformer** (ICLR 2025), which achieves ~2× compression of the transformer body. Track B (Fractal Residual Recursion) achieves:

- **311× compression** of the Qwen3-1.7B body in the most published variant (1.5M trainable params instead of 467M)
- **734×** in the most aggressive variant
- Comparable downstream-task quality to RRT at the comparable compression ratio

Track B is **150× beyond the published academic frontier** in terms of compression ratio.

## Combined stack

Track A and Track B are *orthogonal* — they compose multiplicatively. Applying both:

- **26.7× end-to-end** compression of a transformer model
- 68% top-10 retention
- The smallest deployable form factor for any frontier-class capability we've measured

## What we share publicly vs. under NDA

| Information | Public | NDA |
|---|---|---|
| Compression ratios (the numbers above) | ✅ | ✅ |
| Validation cohort + benchmark methodology | ✅ | ✅ |
| Reproducibility manifest (SHA-256 file index) | reference only | full audit |
| Per-model individual quality retention | ✅ | ✅ |
| Method names and high-level conceptual claim | ✅ | ✅ |
| **The mechanism** by which Track A breaks the 4-bit floor | — | ✅ |
| **The mechanism** by which Track B achieves 150× | — | ✅ |
| Patent specifications (filed Apr 2026) | — | ✅ |
| Roadmap of future tracks (C, D, E, ...) | — | ✅ |

If you need NDA access to the technical deep dive, email `legal@sipsalabs.com`.

## Reproducibility

Every public number ships with:

- A deterministic seed (default seed = 42 across all runs)
- A full sample count (no cherry-picked best-of-N)
- A multi-model cohort (no single-model fluke results)
- A SHA-256-verified manifest of the input artifacts

This is increasingly a procurement gate for enterprise + government customers. We are the only player at our compression frontier shipping it out of the box.

## What's next

- The **Track A supplement** filing (May 2026) extends Track A's claim scope
- Future tracks (C, D, E, ...) are part of an active research roadmap; details available under NDA — contact `legal@sipsalabs.com`
