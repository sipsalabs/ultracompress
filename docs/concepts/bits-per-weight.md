# Bits per weight (bpw)

"Bits per weight" is the standard unit of comparison for LLM compression. This page explains what it actually means, how to interpret it, and the subtleties that make naive comparisons misleading.

## Definition

> **Bits per weight (bpw)** = (total bits used to represent the compressed model on disk) / (number of trainable weights in the FP16 source model).

For a 1.7-billion-parameter model:
- FP16 source: `1.7e9 × 16 = 27.2e9 bits = 3.4 GB`
- 4-bit quantization (e.g., NF4): `1.7e9 × 4 = 6.8e9 bits = 850 MB`
- UltraCompress 5 bpw (lossless): `1.7e9 × 5 = 8.5e9 bits = 1.06 GB`

The lower the bpw, the smaller the artifact — but bpw alone says nothing about whether the model still works. UltraCompress targets a lossless 5-bit pack: bit-identical reconstruction rather than lossy drift (see [Catastrophic failures](catastrophic-failures.md)).

## What the number includes

A naive `bpw = (file size × 8) / weight count` would understate compression for methods that have meaningful overhead beyond the per-weight bits:

- **Codebooks** (lookup tables for non-uniform quantization)
- **Per-channel or per-group scales** (normalization factors)
- **Zero points** (offsets for asymmetric quantization)
- **Residual outliers** (a few weights stored at higher precision)
- **Header / metadata bytes**

We define `bpw` to include **all of these** in the numerator. The published `bpw=5` means 5 bits per weight in the **on-disk artifact**, including codebooks, scales, zero points, and metadata.

This is more honest than the alternative — some methods report `bpw` as the per-weight bits *excluding* overhead, which makes their numbers look better on paper but isn't a fair comparison to UltraCompress's reported number.

## What the number does NOT include

`bpw` is a **storage** metric. It does not directly indicate:

- **Inference speed** (depends on the runtime; some compressed formats inflate at load time, others at inference time)
- **Inference memory** (also runtime-dependent; can be larger or smaller than disk size)
- **Quality retention** (the entire question of whether the model still works after compression — this needs a separate retention metric)

When comparing methods, always look at **bpw + retention together**. A method at `bpw=2` with `30% retention` is worse than a method at `bpw=4` with `98% retention` for almost any practical purpose.

## How to interpret on a Pareto plot

The most useful visualization of compression methods is a **2D Pareto plot**: x-axis is `bpw`, y-axis is `quality retention`. Methods on the upper-left (low bpw, high retention) dominate methods on the lower-right (high bpw, low retention).

The interesting region for deployment is **bpw < 4**. Above bpw=4, every method works well; the differences are minor. Below bpw=4, the methods diverge sharply.

UltraCompress at 5 bpw is *lossless* — bit-identical reconstruction — where comparable-bpw lossy methods still drift relative to the original weights. The differentiator is not a lower bpw point; it is reconstruction fidelity at 5 bpw.

## Practical implication for hardware budgets

A 12 GB phone-class GPU can hold:

- A 7B-parameter FP16 model: **no** (needs ~14 GB)
- A 7B-parameter int8 model: **maybe** (~7 GB + activations)
- A 7B-parameter NF4 model: **yes** (~3.5 GB)
- A 7B-parameter UltraCompress 5-bpw model: **yes** (~4.4 GB on disk; lossless reconstruction)

The benefit here is not the smallest possible footprint but a *lossless* 5-bit footprint — bf16-equivalent reconstruction at sub-NF4-quality risk, which matters where "good enough on MMLU" isn't enough.

## Common pitfalls

### "We achieved 1-bit quantization!"

Some methods report aggressive numbers like 1-bit or sub-1-bit quantization. Almost always these:

1. Don't include codebook/scale overhead in the headline bpw
2. Don't measure retention on a multi-model cohort
3. Are demonstrated only on a single small model (where catastrophic failures hide)
4. Use heavy mixed-precision (some weights at 16-bit, others at 1-bit) — making the "1-bit" claim misleading

When evaluating a sub-3-bpw claim, always ask:
- Is the bpw including all overhead?
- Is the retention measured on a multi-model cohort?
- What's the catastrophic-failure rate?

### Bytes vs. bits

`bytes per weight` shows up sometimes (e.g., `0.5 BPW = 4 bpw`). The conversion is `1 byte = 8 bits`. Always check the unit.

### Total artifact size vs. weight-only size

Some artifacts include the tokenizer and config in addition to the weights. The `bpw` should be computed using **weight count and weight bits only**. We separate these in our `ultracompress.json` manifest.

## See also

- [Compression methods](compression-methods.md)
- [Catastrophic failures](catastrophic-failures.md)
- [Reproducibility](reproducibility.md)

Codec internals and training procedure are patent-pending.
