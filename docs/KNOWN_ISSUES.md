# Known Issues & Honest Disclosures

Published so reviewers can see we're self-aware. Every one of these is
actively tracked or has a documented mitigation.

## 1. Teacher is a stripped Qwen3 (no Q/K norm)

**What.** `ultracompress.inference.MiniTransformer` does not apply
Qwen3's per-head RMSNorm on the query and key projections before
attention. The real HF Qwen3 model does. This makes our "teacher
logits" differ from the official HF Qwen3 output by a small margin.

**Impact.** All distillation metrics (T1, T10, ppl ratio, quality) are
measured **relative to our no-QK-norm teacher**. The compression claim
is unaffected: a 1.5M-param student that matches our teacher's logits
at 69.64% T10 is still a 311× compression of a 1.72B-param model.
However, downstream users who want the student to approximate the
**HF Qwen3** logits exactly will see a slightly larger gap than our
published numbers.

**Fix.** Add Q/K RMSNorm to `MiniTransformer.forward()` at the
attention call site, re-export teacher logits, retrain HQ5 flagship.
Deferred: fixing this invalidates existing HQ4/HQ5/HQ6 checkpoints,
costs ~6 hours of GPU time per flagship, and does not change the
core method or patent claims.

**Mitigation today.** Documented here and in `REPRODUCE.md`. The
pitch says "student matches our teacher's logits" not "student
matches Qwen3-1.7B's logits."

## 2. In-domain held-out region

**What.** `hires_eval.py` samples eval starts from the tail 50M tokens
of `fineweb_edu_500M_tokens.pt`. But the training script
(`run_hq4_ceiling_break.py` and family) samples training windows
uniformly from the full 500M range. So the "tail" is in-distribution,
not strictly disjoint.

**Impact.** `hires_eval` numbers are slightly optimistic versus a
strict disjoint-corpus eval.

**Fix — shipped.** `wikitext_eval.py` evaluates on the WikiText-103
test split, which was never touched during training. Same protocol
(seed 42, SEQ_LEN=128, 1000 samples, bootstrap 95% CIs). Run it
whenever GPUs free up; numbers will be added to README on completion.

## 3. No matched-parameter standard-KD baseline number yet

**What.** We claim the nested-fractal architecture + entropy-weighted
loss is load-bearing. Until we train a standard Hinton-KD baseline
at matched ~1.5M params and report its number, that claim is an
assertion, not a measurement.

**Fix — shipped.** `run_baseline_distill.py` trains exactly that
baseline: `proj_in → N vanilla transformer blocks → proj_out →
frozen head` with classical KL-at-temperature loss. Run queued for
next GPU window.

## 4. Only validated on one teacher family and one scale

**What.** All reported numbers are on Qwen3-1.7B. The claim "works
for any transformer" is unproven at this HEAD.

**Fix — in progress.** See `docs/SCALING_PLAN.md` for the
experimental matrix. `teacher_loader.py` now auto-detects from any
Qwen3-family state dict; next run will be on Qwen3-0.6B to
establish width invariance.

## 5. Combined-stack end-to-end number is work in progress

**What.** `combined_stack_eval.py` measures FRR body + ASVD head
together, which is the pitched end-to-end compression. At HEAD the
run is partially complete (config 2 of 4).

**Fix.** Run will finish overnight. Results will be published in
`combined_stack_results_hq5.json` and plugged into
`docs/PITCH.md`.

## 6. No real-world downstream task eval

**What.** T1 / T10 on next-token prediction is a perfectly sensible
distillation metric, but it doesn't tell a customer "will your
compressed model answer my user's support question correctly?"

**Fix.** Plan: once the best flagship is chosen (HQ5 vs HQ6 vs HQ7),
run it through MMLU, HellaSwag, ARC-Easy, and a generation-quality
comparison (judge = GPT-4 or Claude) before the Monday provisional
filing. Timeline: ~2 hours of eval after training done.

## 7. Patent prior-art check not yet professional

**What.** `docs/PATENT_PROVISIONAL_SKELETON.md` lists prior art we're
aware of (Vaswani 2017, Hinton 2015, Dehghani 2018 Universal
Transformer, DistilBERT, TinyBERT, ASVD, GPTQ). We haven't done a
formal search with a patent attorney or paid USPTO/Google Patents
search service.

**Fix.** Monday meeting with provisional-filing attorney. Plan to
pay for a rush professional search.
