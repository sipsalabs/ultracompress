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
| **UltraCompress 2.798 bpw** | **2.798** | **95.63%** | **0/6** |
| HQQ 3-bit g64 | 3.500 | 72.46% | 1/6 |
| HQQ 2.5-bit g64 | 3.000 | 24.18% | 4/6 |
| **HQQ 2-bit g64** | **2.500** | **3.46%** | **6/6** |

The pattern is consistent and unambiguous: **all public methods fall off a cliff between 3 and 2 bits per weight**.

UltraCompress at 2.798 bpw is the **only sub-3-bpw method on this cohort with zero catastrophic failures**. It's not 1% better; it's a categorical difference.

## Why the cliff exists (high-level intuition)

A weight tensor at FP16 has a continuous distribution. A naive uniform quantization to N bits maps that continuous distribution to a discrete grid of 2^N values:

- 8 bits → 256 levels (still finely-spaced enough that quantization noise is small)
- 4 bits → 16 levels (coarser; saved by per-row scaling and outlier-aware schemes like NF4)
- 3 bits → 8 levels (very coarse; per-row scaling is no longer enough)
- 2 bits → 4 levels (almost no resolution; the model can't recover its structure under this quantization noise)

At 2 bits, the difference between a weight that should be `0.0001` and `0.001` and `0.01` is fully erased. The cumulative effect across hundreds of millions of weights produces output that no longer reflects the trained behavior.

Public methods that try to push below 4 bits (HQQ 3-bit, HQQ 2-bit, GPTQ at low bit widths) attempt to soften the cliff with per-group scaling, mixed-precision selectivity, and outlier handling. These tricks help for a fraction of bits but eventually run out.

## What UltraCompress does differently

The Track A method doesn't try to do better quantization. It uses a **fundamentally different weight representation** that preserves more of the model's structural information at the same bit budget. The mechanism is described in the filed patent specification (USPTO 64/049,511) and is available under NDA for serious technical conversations.

The empirical result is that Track A's quality-bpw curve doesn't have a cliff at 4 bits; it's smoother all the way down to 2.798 bpw, at which point it starts degrading more steeply but stays well above the catastrophic-failure threshold.

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

Codec internals + training procedure are patent-protected (USPTO 64/049,511 + 64/049,517).
