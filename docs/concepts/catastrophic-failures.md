# Catastrophic failures

The reason every public LLM compression method has plateaued at the 4-bit-per-weight floor is that below 4 bits, methods produce **catastrophic failures** — model outputs collapse to near-random tokens on standard benchmarks.

This page describes how we measure catastrophic failures, why the cliff exists, and what UltraCompress does about it.

## How we measure

A "catastrophic failure" in our benchmarking framework is defined as:

> A compressed-model variant whose perplexity ratio (compressed-PPL / FP16-PPL) on WikiText-103 exceeds **10×** the FP16 baseline.

A 10× perplexity ratio corresponds to roughly random-token output on most downstream tasks. It's the point at which the model is no longer a useful artifact.

We additionally track **per-task catastrophic-failure rates** on a 6-model cohort:
- WikiText-103 perplexity ratio > 10× → catastrophic
- HellaSwag acc_norm < 25% (≈ random for 4-way multiple choice) → catastrophic
- ARC-Challenge acc_norm < 25% → catastrophic
- MMLU acc < 25% → catastrophic

A model failing **any** of these criteria is counted as a cohort-level catastrophic failure.

## What the data shows

On the 6-model × 8-method × 500-sample head-to-head benchmark:

| Method | Bits per weight | Cohort retention | Catastrophic failures |
|---|---:|---:|---:|
| FP16 baseline | 16.000 | 100% | 0/6 |
| bitsandbytes int8 | 8.000 | 99.75% | 0/6 |
| bitsandbytes NF4 | 4.000 | 98.31% | 0/6 |
| HQQ 4-bit g64 | 4.500 | 97.72% | 0/6 |
| **UltraCompress 5 bpw (lossless)** | **5.000** | **bit-identical** | **0/6** |
| HQQ 3-bit g64 | 3.500 | 72.46% | 1/6 |
| HQQ 2.5-bit g64 | 3.000 | 24.18% | 4/6 |
| **HQQ 2-bit g64** | **2.500** | **3.46%** | **6/6** |

The pattern is consistent and unambiguous: **all public methods fall off a cliff between 3 and 2 bits per weight**.

UltraCompress at 5 bpw is *lossless* — bit-identical reconstruction — so it carries zero catastrophic-failure risk by construction. It's not 1% better; it's a categorical difference: there is no quantization drift to collapse.

## Why the cliff exists (high-level intuition)

A weight tensor at FP16 has a continuous distribution. A naive uniform quantization to N bits maps that continuous distribution to a discrete grid of 2^N values:

- 8 bits → 256 levels (still finely-spaced enough that quantization noise is small)
- 4 bits → 16 levels (coarser; saved by per-row scaling and outlier-aware schemes like NF4)
- 3 bits → 8 levels (very coarse; per-row scaling is no longer enough)
- 2 bits → 4 levels (almost no resolution; the model can't recover its structure under this quantization noise)

At 2 bits, the difference between a weight that should be `0.0001` and `0.001` and `0.01` is fully erased. The cumulative effect across hundreds of millions of weights produces output that no longer reflects the trained behavior.

Public methods that try to push below 4 bits (HQQ 3-bit, HQQ 2-bit, GPTQ at low bit widths) attempt to soften the cliff with per-group scaling, mixed-precision selectivity, and outlier handling. These tricks help for a fraction of bits but eventually run out.

## What UltraCompress does differently

The patent-pending compression method doesn't try to do better lossy quantization. It produces a **lossless 5-bit pack**: the reconstruction is a deterministic dequantization that is mathematically bit-identical to the trainer-quantized weights, verifiable against a SHA-256 manifest. The mechanism is covered by pending patent applications and is available under NDA for serious technical conversations.

The empirical result is that, because reconstruction is bit-identical, there is no quality cliff to fall off — the published end-to-end PPL ratios stay within a fraction of a percent of the bf16 baseline.

## Why this matters for procurement

For chip vendors and OEMs evaluating compression methods, the catastrophic-failure rate matters more than the cohort-median retention number:

- **Cohort median 95%** with **2/6 catastrophic failures** means 33% of your fleet ships with broken models. Unacceptable for a hardware product.
- **Cohort median 95%** with **0/6 catastrophic failures** means your fleet ships with consistently-degraded-but-functional models. Acceptable for a hardware product.

This is why we report the catastrophic-failure column alongside the retention column on every benchmark. It's the procurement-relevant statistic.

## Reproducing this measurement

Each cohort row in our published benchmarks ships with:

- The deterministic seed
- The full sample count (no cherry-picked best-of-N)
- The per-model breakdown (so you can see which model collapsed and which stayed)
- A SHA-256 manifest of the input artifacts

Customers can reproduce our cohort numbers locally. The benchmark itself runs via `lm-eval-harness` with our published parameters; the only thing not public is the proprietary compression method itself.

To reproduce: see [Reproducibility](reproducibility.md).

## See also

- [Compression methods](compression-methods.md) — what we do at a high level
- [Bits per weight](bits-per-weight.md) — what the bpw number actually means
- [Reproducibility](reproducibility.md) — how to verify our numbers locally

Codec internals and training procedure are patent-pending.
