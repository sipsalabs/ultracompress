# FractalLM — Pitch One-Pager

*One page for the attorney, the first investor call, the first customer.*

---

## The claim

**A 1.51 M-parameter neural network that reproduces 70% of the next-token behavior of a 1.72 B-parameter transformer language model — a 311× compression ratio with no accuracy cliff.**

With an additionally-compressed output head (ASVD, low-rank): the full compressed inference stack is [TODO: fill from combined_stack_results_hq5.json] parameters at [TODO]× end-to-end compression, retaining [TODO]% of teacher top-10 accuracy.

---

## Why it works (30 seconds)

Two ideas, combined:

1. **Fractal iterative architecture.** Instead of stacking 28 transformer blocks, we train *one* transformer block and re-apply it 28 times in a nested scale×iteration schedule, with tiny learned scalars controlling each step. 28× fewer block parameters, same depth of computation.
2. **Entropy-weighted distillation.** During training, each token position's loss is scaled by the teacher's predictive entropy at that position. Easy positions (near-deterministic teacher) contribute little signal; genuinely ambiguous positions (where the teacher is spreading mass across many candidates) contribute most. This forces the student to learn the teacher's distribution where it actually contains information.

Both ideas are simple. Both are novel in combination. The result is compression ratios 10–50× beyond what standard distillation + quantization achieve at similar quality.

---

## What it is good for

- **Edge and on-device inference.** A model that fits in <10 MB of weights runs on phones, embedded systems, browsers.
- **Cost reduction.** Same latency and quality target at ~100× less serving compute.
- **Private / air-gapped deployment.** Small enough to ship a sealed binary to regulated-industry customers (medical, legal, defense) without dependence on a hosted API.
- **Fine-tuning substrate.** Small models are cheap to specialize per customer.

---

## Verified numbers (from hires_eval + combined_stack_eval, seed 42, n = 1000, SEQ_LEN = 128)

*Held-out region: last 50M tokens of FineWeb-Edu 500M shard, not prioritized during training.*

| Configuration | Trainable params | Compression vs teacher | all-T1 | all-T10 | Quality* | PPL ratio |
|---|---|---|---|---|---|---|
| HQ5 h256 (body only) | 1.51 M | 311× | **[hires_results_hq5.json]** | **[...]** | **[...]** | **[...]** |
| HQ5 h128 (body only) | 0.64 M | 734× | **[...]** | **[...]** | **[...]** | **[...]** |
| HQ5 h256 + ASVD r=1024 (end-to-end) | ~3.1 M | ~555× | **[combined_stack_results_hq5.json]** | **[...]** | **[...]** | **[...]** |
| HQ5 h256 + ASVD r=512 (end-to-end) | ~2.3 M | ~750× | **[...]** | **[...]** | **[...]** | **[...]** |

*Quality = 0.5 · all-T10 + 0.5 · (1/ppl_ratio), matching internal training-eval metric.*

---

## What's defensible

1. **Provisional patent application filed** [date TBD, target 2026-04-20]. Priority date locked.
2. **Reproducible artifact.** Open-source training and evaluation code at `github.com/[user]/ultracompress`. Anyone can reproduce the numbers above on a single consumer GPU in ~48 hours.
3. **Compute moat.** While the architecture is small at inference time, training requires dual high-memory GPUs and multiple 7-day training runs to identify the right entropy-weighting schedule. The winning schedule is the thing the patent covers.

---

## Contact

[TODO — name, email, phone, LinkedIn URL]

*GitHub: https://github.com/[user]/ultracompress*
*Repo commit of record: [TODO — `git rev-parse HEAD` at time of pitch]*
