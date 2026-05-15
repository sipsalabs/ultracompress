# Reproducibility

Every public number Sipsa Labs ships about UltraCompress is reproducible. This page describes how, and why we built it that way.

## What we publish with every benchmark

For every published cohort number, we ship:

1. **A deterministic seed** (default `42` across all runs).
2. **A full sample count** — no cherry-picking best-of-N. We commit to a sample size before running the eval.
3. **The full cohort identity** — which models, which method versions, which task suite, which `lm-eval-harness` version.
4. **A 224-file SHA-256 verification manifest** — every input artifact, every weight file, every config file is hashed.

The manifest is auditable. We can show — under NDA — exactly which `qwen3-1.7b-instruct.safetensors` SHA we ran against, when, and on which hardware.

## Why this matters

Two reasons.

### Procurement gate

Enterprise + government procurement teams increasingly require **provenance + reproducibility** as a vendor-evaluation criterion, not as a nice-to-have. A vendor that says "trust me, our numbers are right" loses to a vendor that says "here's the SHA-256 manifest of every input — you can reproduce our entire benchmark suite locally."

Especially in regulated industries (automotive, healthcare, defense), the ability to **prove** that your model artifact is the one you claim it is — and that the benchmark numbers were measured against that exact artifact — is a hard requirement.

We ship reproducibility from day one because it's increasingly the difference between getting through procurement and not.

### Honest signal

There is a long history of compression-method papers that report cherry-picked, non-reproducible, single-model numbers. We will not be one of them.

Our headline numbers — `2.798 bpw`, `95.6% retention`, `0/6 catastrophic failures` — are cohort-level, deterministic, and audit-grade. We round down, not up.

## How a customer reproduces our numbers

The full cohort benchmark requires:

1. Access to the cohort's source models on Hugging Face Hub (these are public)
2. The pre-compressed UltraCompress artifacts (rolling release on `huggingface.co/sipsalabs` through April–May 2026)
3. `lm-eval-harness` at the specified version
4. A CUDA GPU (any consumer 4090/5090 or up; H100 for fast iteration)
5. The SHA-256 manifest (NDA, on request)

Walkthrough:

```bash
# 1. Install
pip install "ultracompress[torch]"
pip install "lm-eval[ultracompress]>=0.4.5"   # version pin matches our manifest

# 2. Pull a cohort artifact
uc pull sipsalabs/<model-id>

# 3. Verify the artifact's manifest
uc info ./models/sipsalabs_<model-id>
# Should show "verified ✓" against the published SHA-256

# 4. Run the benchmark
uc bench ./models/sipsalabs_<model-id> \
    --tasks hellaswag,arc_challenge,arc_easy,piqa,winogrande \
    --limit 500 \
    --batch-size 8 \
    --device cuda:0

# 5. Compare your numbers against our published manifest
# (We provide the JSON for direct diffing under NDA.)
```

If your numbers differ from ours by more than the standard error reported in our manifest, **we want to know**. Most often the cause is hardware-arithmetic non-determinism (which is bounded; aggregates rarely differ in practice) or a `lm-eval-harness` version mismatch.

## Levels of reproducibility we ship

| Level | What it gives you | Public? |
|---|---|---|
| **Aggregate cohort numbers** (`2.798 bpw, 95.6%, 0/6 catastrophic`) | Confidence in the headline | ✅ Public |
| **Per-model numbers** (Llama-2-7B at 95.4%, Qwen3-1.7B at 96.1%, ...) | Confidence in the cohort isn't a fluke | ✅ Public |
| **Per-task numbers** (HellaSwag, ARC, MMLU, etc.) | Confidence the cohort generalizes | ✅ Public |
| **Per-sample logs** (the actual model outputs at each prompt) | Audit-grade verification of any specific sample | NDA |
| **224-file SHA-256 input manifest** | Provenance of every byte that fed the benchmark | NDA |
| **The compression method mechanism** | How the patent-pending compression method breaks the 4-bit cliff | NDA / Trade secret |

We're transparent about which tier of reproducibility lives at which gate. Levels 1-3 are public. Levels 4-5 are NDA. Level 6 is trade-secret + filed-patent.

## Seed handling

`seed=42` across all our runs. We don't search over seeds; we don't pick the best of N. If a cohort number degrades 1-2% after a method change, we report the degraded number, not the previous one.

## Hardware-arithmetic non-determinism

GPU floating-point arithmetic is non-deterministic across runs in some operations (e.g., `cuBLAS` on certain matrix sizes). Our published cohort numbers absorb this into a small standard error (typically ±0.5% per task at our sample sizes).

We report the standard error alongside every benchmark number. Customers reproducing our numbers locally should expect to land within ±1% of our published cohort medians.

## Manifests we ship

- **`ultracompress.json`** — per-artifact provenance manifest. See [Manifest schema](../reference/manifest-schema.md).
- **`bench_summary.json`** — written by `uc bench` after every run. Contains seed, sample count, batch size, hardware, lm-eval-harness version, raw and aggregated numbers.
- **`cohort_master_verify.{json,txt}`** (internal, NDA-shareable) — the 224-file SHA-256 manifest covering every input artifact in our published cohort benchmark.

## What we don't reproduce

- **Latency / throughput numbers**: hardware-dependent; we publish reference numbers but expect customers to measure their own on their hardware.
- **Memory savings during inference**: depends on the runtime (HF transformers vs. llama.cpp vs. TensorRT-LLM); we publish reference numbers per-runtime as integration guides ship.
- **Training-time metrics**: not applicable — UltraCompress is post-training.

## See also

- [Compression methods](compression-methods.md)
- [Catastrophic failures](catastrophic-failures.md)
- [Manifest schema](../reference/manifest-schema.md)
- [`uc bench`](../commands/bench.md)

Codec internals and training procedure are patent-pending.
