<!-- Classification: PUBLIC | Owner: Sipsa Labs | Last Reviewed: 2026-04-27 | External Sharing: allowed -->

# UltraCompress architectural-compression evidence matrix

> **This is the architectural-compression evidence matrix — architectural-compression evidence only.** It is **separate** from the weight-level method 2.798-bpw v0.1 reference-artifact benchmark in the project [README on GitHub](https://github.com/sipsalabs/ultracompress#readme). Do not compare retention numbers across methods as a single quality curve — the weight-level method is post-training quantization (shipping now); the architectural method is architectural compression (v0.2, Q3 2026).

**Method:** architectural compression method (patent pending)
**Availability:** evidence now; product availability v0.2 (Q3 2026)
**Customer ship status:** not yet downloadable; pre-compressed reference models for the architectural method release in v0.2

**Cohort size:** 6 models
**Operating point:** uniform across the cohort; method-internal parameters held constant under NDA

## Per-model results

Every row is row-level-labeled with experiment family and customer ship status so a screenshot of any single row carries its own firewall context.

| Model | Model family | Experiment family | Customer ship status | Params (B) | Method | Availability | bpw | Compression vs FP16 | T1 retention | T10 retention | T1 agreement | T10 agreement | PPL FP16 | PPL compressed |
|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TinyLlama-1.1B | Llama-2 | architectural-compression evidence | Not yet downloadable | 1.1 | shared-block | v0.2 | 2.3988 | 1.6985x | 83.61% | 91.73% | 65.01% | 94.17% | 17.0142 | 28.8989 |
| OLMo-2-1B | OLMo-2 | architectural-compression evidence | Not yet downloadable | 1.485 | shared-block | v0.2 | 2.3906 | 1.7898x | 82.75% | 90.83% | 62.76% | 93.06% | 20.1537 | 36.0711 |
| SmolLM2-1.7B | SmolLM2 | architectural-compression evidence | Not yet downloadable | 1.812 | shared-block | v0.2 | 2.3906 | 1.8988x | 80.84% | 90.18% | 62.57% | 93.20% | 18.0321 | 34.2397 |
| Qwen3-1.7B | Qwen3 | architectural-compression evidence | Not yet downloadable | 1.7 | shared-block | v0.2 | 2.4017 | 1.788x | 84.65% | 90.68% | 64.04% | 93.88% | 33.21 | 59.4 |
| Mistral-7B-v0.3 | Mistral | architectural-compression evidence | Not yet downloadable | 7.248 | shared-block | v0.2 | 2.3942 | 1.6274x | 86.21% | 93.19% | 69.69% | 95.06% | 12.3569 | 20.1093 |
| Qwen3-8B | Qwen3 | architectural-compression evidence | Not yet downloadable | 8.19 | shared-block | v0.2 | 2.3967 | 1.3859x | 91.85% | 95.83% | 73.56% | 96.98% | 20.6963 | 28.6829 |

> **bpw vs compression-ratio denominator note**: the `bpw` column measures bits-per-weight on the compressed artifact; the `Compression vs FP16` column measures total artifact size relative to the FP16 baseline. The two columns measure different denominators (per-weight cost vs total artifact accounting) and should not be reconciled with each other without consulting the manifest. The same compressed artifact can have a low bpw (per-weight) and a modest compression ratio (per-artifact) when the architectural-compression component dominates the savings.

## Cohort summary (architectural compression at the same operating point)

- **Median bpw:** 2.40
- **Median T1 retention:** 84.13%
- **Median T10 retention:** 91.28%
- **Median compression ratio vs FP16:** 1.74x

## Envelope across the cohort

- **bpw range:** 2.39 – 2.40
- **Compression ratio range:** 1.39x – 1.90x vs FP16
- **T10 retention floor:** 90.18% (worst-case across the cohort)
- **T10 agreement floor:** 93.06%

## Weight-level method (separate experiment — referenced for context only)

The README headline numbers (95.63% median T1 retention, zero catastrophic failures, 30% smaller than NF4) are the **weight-level method** benchmark at **2.798 bpw**. The weight-level and architectural methods are different methods, different operating points, different bpw targets. Customer evaluations should pick a method based on use case:

- **Weight-level method (shipping now)** — drop-in replacement for bitsandbytes / GPTQ / AWQ / HQQ. Pre-compressed reference artifacts roll out on Hugging Face Hub through April–May 2026.
- **Architectural method (v0.2)** — architectural compression layered on top of (or in place of) standard quantization. Higher absolute compression, narrower architecture support at v0.2 launch.

## Field definitions

- **`bpw`** — bits per weight; effective on-disk per-parameter cost including all overhead (codebooks, scales, zero points, metadata).
- **`Compression vs FP16`** — compressed-artifact size relative to FP16 baseline; >1.0 means smaller. Different denominator than `bpw`; see footnote above.
- **`Experiment family`** — which experimental method this row belongs to. All rows in this matrix are architectural-compression evidence.
- **`Customer ship status`** — whether artifacts from this experiment are downloadable today. All rows in this matrix are not yet downloadable; release in v0.2.
- **`PPL FP16`** — WikiText-103 perplexity of the FP16 teacher.
- **`PPL compressed`** — WikiText-103 perplexity of the compressed model at this operating point.
- **`T1 / T10 agreement`** — fraction of tokens where compressed top-k matches teacher top-k.
- **`T1 / T10 retention`** — compressed top-k accuracy / teacher top-k accuracy, expressed as a percentage.

## Provenance

- **Source:** internal Sipsa Labs benchmark archive. SHA-256-verified manifest available under NDA — email `legal@sipsalabs.com`.
- **Extracted:** 2026-04-27.
- **Method:** direct field copy of public-safe fields only; method internals (operating-point parameters, codebook sizes, calibration constants) deliberately excluded — those live with the filed patent specifications and are accessible only under NDA.
- **No hand-entered values.** Each row is a direct copy from the source archive with rounding only.

## Notes

- All numeric values are direct field copies from the source archive; no hand-entered values.
- Models without `ppl_fp16` / `ppl_compressed` ran the agreement/retention pipeline but not the perplexity pipeline.
- Cohort medians and envelope are computed by this extractor; readers can recompute from the per-model rows.
- This is the architectural-compression *evidence matrix* — architectural-compression evidence. Weight-level reference-artifact benchmarks are in the README. **Do not combine.**

---

*Both `matrix.md` (this file) and `matrix.json` are direct field copies from the source-of-truth file in the internal Sipsa Labs benchmark archive. Method-internal fields (operating-point parameters, codebook sizes, calibration constants) are deliberately excluded; the patent specifications cover those.*

Codec internals and training procedure are patent-pending.
