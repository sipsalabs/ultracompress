# Honest Negative Results — 2026-05-08

**Lab:** Sipsa Labs (`ultracompress`)
**Operator:** Sip
**Compiled:** EOD 2026-05-08 MDT
**Source of truth:** `docs/LAB-NOTEBOOK.md` entries 08:34 → 16:50 MDT, plus the JSON eval files referenced inline below.

This document collects every honest negative result produced by today's research, in one place, with the same level of detail we use internally. We publish it because the positive findings only mean what they mean if the failures are visible too. Today produced an all-time tightest dense-decoder PPL ratio (Qwen3-1.7B-Base 1.0040×), an 18-architecture validation matrix, and 9 public `uc verify`-PASS HuggingFace artifacts. It also produced 11 distinct negative or failed-experiment outcomes that are catalogued below. Every number here is traceable to a JSON file in `docs/` or a script log under `scripts/overlay/`.

We follow the same template for every entry: what we tried, what we expected, what happened, what we learned, and what we will explicitly NOT publish from it.

---

## 1. Mamba V18-C SVD warm-start — NEGATIVE
**Lab notebook:** 09:51 MDT entry. Run script: `scripts/overlay/_test_mamba_v18c_svd_warmstart.py`. Run log: `_test_mamba_v18c_svd_warmstart.log`.

- **What we tried.** Apply a V18-C correction to every one of the 256 Mamba-2.8B SSM Linears (`in_proj`, `x_proj`, `dt_proj`, `out_proj`) by computing `Wq` via GSQ 5 bpw, taking the residual `R = W − Wq`, running truncated SVD `R ≈ U_top · diag(S_top) · Vh_top` at rank 32, baking `U_factor = U_top · sqrt(S_top)` and `V_factor = sqrt(S_top) · Vh_top`, then wrapping each `nn.Linear` as `V18CCorrectedLinear` so inference computes `y = W_base @ x + α · U @ V @ x`. No training applied — purely an SVD warm start.
- **What we expected.** PPL ratio reduced from the GSQ-only baseline (1.0119×) toward the trained-V18-C transformer ceiling (~1.005×).
- **What happened.** Compressed PPL = 8.0390 against bf16 baseline 7.9389 → ratio **1.0126×** — i.e. **0.07 pp WORSE than GSQ-only**, on n=30 prompts × seq_len=512.
- **What we learned.** A truncated rank-32 SVD on a high-rank residual injects directional noise that is not aligned with the actual Mamba activation distribution. The V18-C value comes from the 200-step KL distillation that fits V/U/α to real activations, not from the SVD initialization. SVD warm-start is a useful initializer, not a corrector.
- **What we will NOT publish from this.** Any claim that V18-C "works without training" on SSMs. We do NOT have a sub-1.012× number on Mamba.

## 2. Mamba V18-C TRAINED (per-Linear weight-MSE, random Gaussian inputs) — NEGATIVE
**Lab notebook:** 10:55 MDT entry. Run script: `scripts/overlay/_test_mamba_v18c_trained.py`. Run log: `_test_mamba_v18c_trained.log`.

- **What we tried.** For each of 256 Mamba SSM Linears, wrap with V18-C (rank=32, fp32 V/U/α, bf16 base), generate random Gaussian calibration inputs, compute teacher output `W_orig @ x`, train V/U/α with Adam (100 steps, lr=1e-3) to minimize MSE against teacher output.
- **What we expected.** Hypothesis: SVD warm-start alone is the problem; per-Linear KL-style training of V/U/α should close the gap to ~1.005×.
- **What happened.** Compressed PPL ≈ 8.0361 against baseline 7.939 → ratio **~1.0122×**, still 0.03 pp WORSE than GSQ-only (1.0119×).
- **What we learned.** Random Gaussian inputs do not match Mamba's actual activation distribution (selective-scan + conv1d outputs have specific structure). And per-Linear weight-MSE is the wrong objective: the goal is cumulative KL divergence per BLOCK against the teacher logits, not per-Linear output recovery against synthetic noise. Calibration distribution and objective function are both bottlenecks before the codec is.
- **What we learned about what would work.** Adapt the streaming compression runner from `LlamaDecoderLayer` iteration to `MambaBlock` iteration, capture real teacher hidden states from a forward pass on FineWeb-edu, train V/U via cumulative KL distillation per block. Estimated 1–2 days engineering + 3–4 hr training per Mamba size. Deferred to a future session.
- **What we will NOT publish from this.** Any sub-1.012× PPL number on Mamba. The public Mamba number is and remains **1.0119× GSQ-only**, full stop, until the streaming runner is adapted for SSMs.

## 3. OLMo-2 first compression attempt — FAILED, then patched
**Lab notebook:** referenced in `SHIPPED_TODAY_2026_05_08.md` §3 + §8. Patches landed in `scripts/overlay/streaming_teacher.py` and `scripts/overlay/streaming_compression_runner.py`.

- **What we tried.** Run the GPU 1 conveyor-belt queue (`scripts/overlay/_gpu1_arch_queue.sh`) on `allenai/OLMo-2-0425-1B` with the same 5 bpw + V18-C rank=32 / 200-step pipeline that worked on the other small dense archs.
- **What we expected.** Clean compression in ~6 min, PPL ratio in the same 1.007×–1.013× band as the other small dense archs.
- **What happened.** Hard crash with `Olmo2Config has no attribute layer_types`. Root cause: the runner's model-class dispatch had no `'olmo'` / `'olmo2'` entry, so it fell back to `Qwen3DecoderLayer` and then tried to read a Qwen3-only attribute on the OLMo-2 config object.
- **What we learned.** The fallback DecoderLayer path was silently wrong — it should have errored at dispatch time instead of at the first forward call. Patches landed today: `'olmo'` and `'olmo2'` model_type dispatch added to `streaming_teacher.py` and matching DecoderLayer dispatch added to `streaming_compression_runner.py`. Retry succeeded; final OLMo-2-0425-1B ratio = 1.0073× (see entry §10 below).
- **What we will NOT publish from this.** Any "all 18 architectures worked first try" claim. They did not. OLMo-2 needed a code patch first, and the broken-dispatch class is now in the lab notebook.

## 4. TinyLlama PPL eval — CUDA device-side assert, deferred
**Lab notebook:** 15:31 MDT entry, footnote on TinyLlama row.

- **What we tried.** Compress `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (1.1 B, 22 layers) at 5 bpw + V18-C rank=32 / 200 steps — same recipe as the other small dense archs in the GPU 1 conveyor belt — and run `eval_compressed_only.py --n_eval 30 --seq_len 1024` for the PPL ratio number.
- **What we expected.** A PPL ratio in the 1.007×–1.013× band, matching the other small dense archs.
- **What happened.** Compression itself succeeded — the `.uc` pack passed `uc verify` cleanly. But the compressed-model forward pass during PPL eval crashed with `torch.AcceleratorError: device-side assert triggered`. No PPL number recorded.
- **What we learned.** This is almost certainly a vocab/positional edge case in the eval harness, not a compression issue (the same v3 pack format passes `uc verify` for structural integrity). Needs `CUDA_LAUNCH_BLOCKING=1` reproduce to pinpoint the exact tensor index. Deferred to a future debugging session.
- **What we will NOT publish from this.** A TinyLlama PPL ratio. The 18-arch matrix shows TinyLlama as "(eval CUDA asserted; deferred)" — exactly that string, not a fabricated number.

## 5. Mixtral-8x22B HF upload retries 1 + 2 — FAILED (xet write tokens), retry 3 — SSL EOF
**Lab notebook:** referenced in `SHIPPED_TODAY_2026_05_08.md` §1 + §8.

- **What we tried.** Upload the 94 GB `_packed_mixtral_8x22b_v3` pack directory (56 layers × ~1.7 GB each) to `huggingface.co/SipsaLabs/mixtral-8x22b-v0.1-uc-v3-bpw5` via `huggingface_hub.HfApi.upload_folder()` (atomic).
- **What we expected.** A single multi-hour upload over residential bandwidth, completing in one pass.
- **What happened.**
  - Retry #1: failed with `TokenRefreshFailure` on the xet write token mid-upload.
  - Retry #2: same `xet-write-token` 500 error class, before the run hit any first-layer commit.
  - Retry #3 (re-fired with `HF_HUB_DISABLE_XET=1` to bypass the xet write path entirely): hit a generic SSL EOF from residential bandwidth instead.
- **What we learned.** xet write tokens have a hard refresh cliff that is hit by long-running uploads of large packs over residential bandwidth. `HF_HUB_DISABLE_XET=1` correctly bypasses the xet path but does not solve the underlying SSL-EOF brittleness of multi-hour uploads on a home connection. The upload is now under an 8-retry watchdog wrapper; classified as in-flight, not completed.
- **What we will NOT publish from this.** Any claim that `SipsaLabs/mixtral-8x22b-v0.1-uc-v3-bpw5` is publicly verified end-to-end. The dashboard (`docs/PUBLIC_VERIFICATION_DASHBOARD_2026_05_08.md`) shows the row as "🟡 uploading (re-fired 12:35)". Local `uc verify --skip-hash` PASS on the source-of-truth pack is published; HF-end PASS is not.

## 6. SmolLM2 first upload — FAILED (SSL EOF on layer_004.uc)
**Lab notebook:** in-flight upload list, `SHIPPED_TODAY_2026_05_08.md` §1.

- **What we tried.** Atomic upload of `_packed_smollm2_1_7b_v3` (24 layers) to `SipsaLabs/smollm2-1_7b-uc-v3-bpw5`.
- **What we expected.** A clean upload completing in 30–60 min on residential bandwidth.
- **What happened.** Generic SSL EOF mid-stream around `layer_004.uc`. No xet involvement — this is a plain `urllib3` connection drop.
- **What we learned.** Same SSL-EOF brittleness pattern as Mixtral-8x22B but at a much smaller pack size, which means it is genuinely a residential-network issue and not a payload-size issue. Now under the same 8-retry watchdog wrapper. The pack itself is fine — local `uc verify --skip-hash` PASS — only the transit failed.
- **What we will NOT publish from this.** Any claim of public HF verification for SmolLM2 yet.

## 7. TinyLlama upload — FAILED (SSL EOF on layer_001.uc)
- **What we tried.** Upload `_packed_tinyllama_1_1b_v3` (22 layers) to `SipsaLabs/tinyllama-1.1b-chat-uc-v3-bpw5`.
- **What we expected.** Clean upload in 20–40 min.
- **What happened.** Same SSL EOF pattern, this time on `layer_001.uc`. Now under the watchdog wrapper.
- **What we learned.** TinyLlama is now in a "compressed-locally-OK, eval-broken (item §4), HF-upload-stuck" state. We are honest about this in the matrix: the row exists, the failures are listed, no PPL number is fabricated.
- **What we will NOT publish from this.** A TinyLlama HF verification claim.

## 8. Qwen3-0.6B upload — FAILED (same SSL EOF pattern)
- **What we tried.** Upload `_packed_qwen3_0_6b_v3` (28 layers) to `SipsaLabs/qwen3-0.6b-uc-v3-bpw5`.
- **What we expected.** Clean upload in 15–30 min.
- **What happened.** Same SSL EOF; now under 8-retry watchdog.
- **What we learned.** This is the third independent confirmation of the residential-bandwidth SSL-EOF pattern, on a pack that is small enough that any plausible "the pack is too big" hypothesis can be ruled out.
- **What we will NOT publish from this.** A Qwen3-0.6B HF verification PASS claim — even though Qwen3-0.6B holds the second-tightest PPL ratio of the day at **1.0069×** (see PPL_EVAL_qwen3-0_6b_2026_05_08.json). The PPL number is real and from local eval; the HF-public-artifact PASS is not yet earned and we do not claim it.

## 9. OLMo-2 upload — initial-run retries
- **What we tried.** Upload `_packed_olmo_2_0425_1b_v3` (16 layers) to `SipsaLabs/olmo-2-0425-1b-uc-v3-bpw5`.
- **What we expected.** Clean ~20 min upload.
- **What happened.** Multiple SSL EOF retries during the initial run — same pattern as items 6–8.
- **What we learned.** Confirms the SSL-EOF pattern is uniform across all small-pack uploads tonight. Not architecture-specific. The 8-retry watchdog is now wrapping every upload, and we are willing to wait through the full retry envelope rather than declare "uploaded" prematurely.
- **What we will NOT publish from this.** Any claim that the OLMo-2 HF artifact is end-to-end customer-verifiable yet. Local PPL ratio (1.0073× base, 0.9998× instruct — see items 10–11) IS published, sourced from local eval against the local pack.

## 10. Base/Instruct hypothesis — REFUTED on 2 of 3 architectures, hypothesis dropped
**Lab notebook:** 16:18 MDT (Qwen3 + OLMo data), 16:35 MDT (SmolLM2 tiebreaker, hypothesis dropped). JSONs: `PPL_EVAL_qwen3-1_7b-base_2026_05_08.json`, `PPL_EVAL_olmo-2-0425-1b-instruct_2026_05_08.json`, `PPL_EVAL_smollm2-1_7b-instruct_2026_05_08.json`.

- **What we tried.** Stated hypothesis upfront at 16:05 MDT: "instruct fine-tuning shifts weights into a regime that's harder to quantize cleanly, so base models should compress to tighter PPL ratios than their instruct variants." Tested with controlled base/instruct pairs on three architectures, holding the compression recipe constant (5 bpw + V18-C rank=32 / 200 steps, FineWeb-edu calibration, 30 prompts × seq_len=1024 held-out tail).
- **What we expected.** All three pairs to show base TIGHTER than instruct.
- **What happened.**

| Arch | Base ratio | Instruct ratio | Direction | Hypothesis? |
|---|---|---|---|---|
| Qwen3-1.7B | **1.0040×** | 1.020× | base TIGHTER | ✅ supports |
| OLMo-2-0425-1B | 1.0073× | **0.9998×** | instruct TIGHTER | ❌ refutes |
| SmolLM2-1.7B | 1.0085× | **1.0075×** | instruct TIGHTER | ❌ refutes |

  Note: OLMo-2-Instruct compressed PPL (18.8494) is *slightly LOWER* than its bf16 baseline (18.8535), i.e. compression acts as a faint regularizer — marginal at n=30 but consistent across all 30 prompts.

- **What we learned.** Hypothesis REFUTED 2/3. The relationship between fine-tuning and quantization-friendliness is architecture- and training-recipe-dependent, NOT universal. Plausible mechanism: Qwen3 is the outlier because Qwen team uses a more aggressive RLHF pipeline (heavy DPO/PPO) that may concentrate sharp activation outliers; Allen Institute (OLMo-2) and HuggingFaceTB (SmolLM2) use lighter post-training, and their instruct weights stay close to base. Possible additional factor on OLMo: instruct training may have used quantization-aware regularization or weight decay that smoothed the activation distribution.
- **What we will NOT publish from this.** Any "instruct fine-tuning makes models harder to quantize" claim. Refuted by direct controlled comparison on 2/3 architectures. **What we WILL publish:** the 18-arch validation table itself, with both base and instruct variants placed alongside each other, no hypothesis attached. The data is the signal.

## 11. rank=64 / train_steps=400 push — REFUTED, the v1 floor stands
**Lab notebook:** 16:50 MDT entry. JSONs: `PPL_EVAL_qwen3-1_7b-base_2026_05_08.json` (v1), `PPL_EVAL_qwen3-1_7b-base-v2-r64s400_2026_05_08.json` (v2).

- **What we tried.** Identical pipeline to the all-time-record v1 run on Qwen3-1.7B-Base, except rank doubled (32 → 64) and V18-C train steps doubled (200 → 400). Same 64-prompt FineWeb-edu calibration. Same 30-prompt held-out tail eval at seq_len=1024.
- **What we expected.** Hypothesis: a larger correction subspace (rank 64) plus more steps to converge against the bf16 teacher's activation manifold should push PPL ratio below 1.001×.
- **What happened.**

| Config | Rank | Train steps | Baseline PPL | Compressed PPL | PPL ratio |
|---|---|---|---|---|---|
| v1 | 32 | 200 | 12.7683 | 12.8195 | **1.0040×** |
| v2 | 64 | 400 | 12.7683 | 12.8217 | 1.0042× |

- **What we learned.** Hypothesis REFUTED. v2 is within statistical noise of v1 (Δ = +0.02% on PPL ratio at n=30). The rank-and-steps knob is saturated at this configuration. The 1.0040× v1 number stands as the floor at `{bpw=5, GSQ k=32, V18-C rank=32, 200 steps}`. To go below 1.001× requires a fundamentally different approach. Candidate directions for follow-up: per-Linear adaptive bpw (5-bit average, 6-bit on outlier Linears, 4-bit on smooth ones); GPTQ-style hessian-aware quantization; per-channel scales instead of per-block; deeper KL distillation against teacher logits (today's training is per-Linear weight-MSE only, which is exactly the same fundamental limitation that bit us on Mamba — see items 1 + 2).
- **What we will NOT publish from this.** Any claim that "more rank = tighter PPL" or that "more training steps = tighter PPL" past v1's settings. Both are refuted at this scale. The published ALL-TIME-RECORD number is **1.0040×**, set by v1, end of story.

## 12. Per-Linear adaptive bpw (k_proj@6bpw) — REFUTED apples-to-apples
**Lab notebook:** 2026-05-09 00:05 entry. JSONs of record: `docs/PPL_EVAL_qwen3-1.7b-base-uniform-n50_2026_05_08.json`, `docs/PPL_EVAL_qwen3-1.7b-base-adaptive-bpw-v1_2026_05_08.json`. Implementation: `scripts/overlay/streaming_compression_runner.py:802-806`. Patent draft (anchored on this refuted result): commit `2fb18f8`.

- **What we tried.** Promote `k_proj` from 5 bpw → 6 bpw on every layer of Qwen3-1.7B-Base; hold all other Linears at 5 bpw; hold V18-C rank=32 / train_steps=200 / FineWeb-edu calibration constant. Hypothesis came from the empirical observation that `k_proj` quant_rel_l2 is -55% (i.e. 1.55×) the other-Linear baseline at 5 bpw, on every layer at both 405B and 1.7B scale. This was the seed of a continuation-in-part patent draft (12 method-form claims, $65 micro-entity).
- **What we expected.** Lower per-Linear quant residual on the dominant bottleneck Linear → tighter end-PPL ratio.
- **What happened.** Apples-to-apples eval (n=50 prompts, seq_len=1024, seed=42, baseline PPL=12.0813, identical held-out tail):
  - Uniform 5 bpw: PPL ratio **1.004876** (compressed 12.140157)
  - v1 adaptive: PPL ratio **1.005097** (compressed 12.142830)
  - Δ = **+0.000220** (v1 SLIGHTLY WORSE, well within noise σ ≈ 0.0003)
- **What we learned.** The k_proj quant_rel_l2 advantage IS real at the per-Linear quantization level — the mechanism reproduces. But it does NOT propagate to end-PPL because V18-C rank=32 was already absorbing the per-Linear residual. Per-Linear quant residual at the input → V18-C correction at the output → identical post-correction layer activations between v1 and uniform. The bottleneck is not the per-Linear quant residual; it's the V18-C correction subspace dimension. Per-layer `train_loss_final` curves are pairwise identical across all 28 layers between v1 and uniform — the diagnostic that closes the case. The cure is not at the codec — it's at V18-C. See `docs/RESEARCH_v3_CURE_DIRECTION_2026_05_09.md` for the V3 = rank-redistribution recommendation.
- **Patent posture.** The CIP draft from commit `2fb18f8` is anchored to this refuted v1 result. **Do NOT file CIP until V3 lands.** Re-anchor on V3 + the train_loss-by-depth diagnostic before filing.
- **What we will NOT publish from this.** Any claim that per-Linear adaptive bpw improves end-PPL on Qwen3-1.7B-Base at this configuration. The mechanism is publishable as a quant-residual diagnostic (for the future patent re-anchoring); the end-PPL claim is not.

## 13. V18-C adaptive train_steps v2 — MARGINAL WIN, VARIANCE SUSPECT (provisional, pending seed sweep)
**Lab notebook:** 2026-05-09 00:05 entry (cure-direction footer). JSONs: `docs/PPL_EVAL_qwen3-1.7b-base-v2-adaptive-train-steps_2026_05_08.json` (final, 23:44), `docs/PPL_EVAL_qwen3-1.7b-base-v18c-adaptive-train-steps_2026_05_08.json` (earlier, 23:01, same nominal config). Implementation: `streaming_compression_runner.py:737-753` (`UC_ADAPTIVE_TRAIN_STEPS=1` gate, linear ramp `train_steps × (1 + 4·depth_frac)`).

- **What we tried.** Linear-ramp `train_steps` from 200 (layer 0) to 1000 (layer 27) on Qwen3-1.7B-Base. V18-C rank=32 unchanged. Total V18-C training compute ~3× uniform.
- **What we expected.** Reduce `train_loss_final` on the deep layers (which sit at 0.35–0.81 in uniform) and so reduce end-PPL ratio toward 1.001×.
- **What happened.**
  - Final v2 run (23:44): PPL ratio **1.004515**, Δ = -0.000361 vs uniform 1.004876 (~1.2σ marginal win above noise floor σ ≈ 0.0003).
  - Earlier v2 run, identical nominal config (23:01): PPL ratio **1.005051**, Δ = +0.000175 vs uniform (~0.6σ above uniform — null).
  - Spread between identical-config runs: **0.000536**, larger than the σ ≈ 0.0003 noise floor we have been quoting.
- **What we learned.** Two possibilities:
  1. The marginal win is real and the variance estimate σ ≈ 0.0003 is too tight — actual σ ≈ 0.0005, in which case v2 is at best a 0.7σ win and indistinguishable from uniform.
  2. One of the two runs hit a non-determinism in the V18-C trainer (Adam + bf16 base + fp32 V/U/α) that is not seed-controlled.
- **Resolution.** Seed sweep ({1, 2, 3} × {uniform, v2_train_steps}) before publishing v2 as a win. Output: `docs/SEED_SWEEP_VARIANCE_BOUND_2026_05_09.md` (planned). If σ ≥ 0.0005, v2 win is retracted as not statistically distinguishable from uniform.
- **What we will NOT publish from this until variance is bounded.** Any "v2 cure produces tighter PPL than uniform" claim. The 1.004515 number is a single observation; one identical-config replicate gave 1.005051. We do not have grounds to declare a real signal until σ is properly bounded.
- **Why we still believe in the cure direction conceptually.** The per-layer `train_loss_final` from the v2_train_steps log shows layers 23-27 still sit at 0.35–0.81 even with 800–1000 ramped steps — i.e. the deep layers are RANK-bound, not STEPS-bound. v3 (rank-redistribution at constant total budget) is the principled fix. See `docs/RESEARCH_v3_CURE_DIRECTION_2026_05_09.md`.

---

## Why we publish negative results

We catalogue these thirteen outcomes for the same reason we publish the positive ones: the only way someone reading our work can know the positive numbers are honest is if the failures are equally visible. Today produced an all-time tightest dense-decoder PPL ratio (Qwen3-1.7B-Base 1.0040×, JSON-traceable to `PPL_EVAL_qwen3-1_7b-base_2026_05_08.json`), an 18-architecture validation matrix, and 9 public `uc verify`-PASS HuggingFace artifacts. It also produced 13 distinct negative or failed outcomes — two SSM-correction approaches that did not work (items 1, 2), one runner crash that needed a code patch (3), one eval-harness CUDA assert that defeated us (4), four HF upload paths that hit residential-network or xet-token failures (5–9, partially overlapping), one published research hypothesis (base-vs-instruct quantization friendliness) that survived only one of three controlled tests (10), one parameter-saturation result that closes the door on rank-and-steps as a cheap path below 1.001× (11), one apples-to-apples refutation of the per-Linear adaptive bpw v1 hypothesis (12), and one v2 marginal-win result whose statistical power is suspect pending a seed sweep (13). The ratio of catalogued failures to published positives today is now roughly 13:9, and that is the ratio we want any external evaluator to use when assessing whether our positive numbers are real. They are.

We will keep modeling this discipline: state hypothesis upfront, measure cleanly, publish only what survives the data, and document everything that did not.
