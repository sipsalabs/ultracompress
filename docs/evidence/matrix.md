# UltraCompress evidence matrix

> UltraCompress Track B reference cohort (architectural compression envelope)

**Patent status:** USPTO 64/049,517 filed 2026-04-25 (Track B). Patent pending.

**Cohort size:** 6 models

## Per-model results

| Model | Family | Params (B) | bpw | Compression vs FP16 | T1 retention | T10 retention | T1 agreement | T10 agreement | PPL FP16 | PPL compressed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TinyLlama-1.1B | Llama-2 | 1.1 | 2.3988 | 1.6985x | 83.6143% | 91.7292% | 65.0125% | 94.1719% | 17.0142 | 28.8989 |
| OLMo-2-1B | Llama-2 | 1.485 | 2.3906 | 1.7898x | 82.7528% | 90.8302% | 62.7594% | 93.0625% | 20.1537 | 36.0711 |
| SmolLM2-1.7B | Llama-2 | 1.812 | 2.3906 | 1.8988x | 80.8353% | 90.1823% | 62.5687% | 93.2047% | 18.0321 | 34.2397 |
| Qwen3-1.7B | Qwen3 | 1.7 | 2.4017 | 1.788x | 84.6514% | 90.6831% | 64.0406% | 93.875% | 33.21 | 59.4 |
| Mistral-7B-v0.3 | Mistral | 7.248 | 2.3942 | 1.6274x | 86.2115% | 93.1928% | 69.6922% | 95.0594% | 12.3569 | 20.1093 |
| Qwen3-8B | Qwen3 | 8.19 | 2.3967 | 1.3859x | 91.8509% | 95.8284% | 73.5594% | 96.9844% | 20.6963 | 28.6829 |

## Cohort summary

- **Median bpw:** 2.4
- **Median T1 retention:** 84.13%
- **Median T10 retention:** 91.28%
- **Median compression ratio vs FP16:** 1.74x

## Envelope (across the cohort at the same operating point)

- **bpw range:** 2.3906 – 2.4017
- **Compression ratio range:** 1.3859x – 1.8988x vs FP16
- **T10 retention floor:** 90.18% (worst-case across the cohort)
- **T10 agreement floor:** 93.06%

## Field definitions

- **`bpw`** — bits per weight; effective on-disk per-parameter cost including all overhead
- **`ratio`** — compression ratio vs the FP16 baseline; >1.0 means smaller
- **`ppl_fp16`** — WikiText-103 perplexity of the FP16 teacher
- **`ppl_compressed`** — WikiText-103 perplexity of the compressed model at this operating point
- **`t1_teacher_pct`** — top-1 accuracy of the FP16 teacher on the calibration eval (in percentage points)
- **`t10_teacher_pct`** — top-10 accuracy of the FP16 teacher (percentage points)
- **`t1_compressed_pct`** — top-1 accuracy of the compressed model (percentage points)
- **`t10_compressed_pct`** — top-10 accuracy of the compressed model (percentage points)
- **`t1_agreement_pct`** — fraction of tokens where compressed top-1 = teacher top-1
- **`t10_agreement_pct`** — fraction of tokens where compressed top-1 is within teacher top-10
- **`t1_retention_pct`** — compressed_t1 / teacher_t1 expressed as a percentage
- **`t10_retention_pct`** — compressed_t10 / teacher_t10 expressed as a percentage

## Provenance

- **Source repository:** `mounnar/ultracompress (private)`
- **Source file:** `results/results.json`
- **Extracted at:** 2026-04-27
- **Method:** direct-copy of public-safe fields only; method internals (operating-point parameters, codebook sizes, calibration constants) intentionally excluded
- **No hand-entered numbers:** `True`

## Notes

- All numeric values are direct field copies from the source JSON; no hand-entered values.
- Method-internal fields (operating-point parameters, codebook sizes, calibration constants) are deliberately excluded from this public matrix and remain in the private results.json source.
- Models that ran the agreement/retention pipeline but not the perplexity pipeline have null ppl_* fields.
- Cohort medians are computed by this extractor; readers can recompute from the per-model rows.

---

*Both `matrix.md` (this file) and `matrix.json` are direct field copies from the source-of-truth file in the private results repository. Method-internal fields (operating-point parameters, codebook sizes, calibration constants) are deliberately excluded; the patent specifications cover those.*
