# Commit Knowledge Map — 2026-05-08 ship chain → 2026-05-09 overnight

Each entry: HASH | SHIP | WHY | PRODUCTION STATUS | FOLLOW-ON.

Source: `git -C C:\Users\scamd\ultracompress log --oneline -30` and per-commit grep across `docs/`, `scripts/overlay/`.

---

## Production / customer-facing

### 33fa2a3 — `release: v0.5.3 — uc verify-org + uc status subcommands`
- **Ship:** PyPI release v0.5.3. Two new subcommands: `uc verify-org <org>` (verifies ALL HF packs in an org), `uc status` (runtime/system check).
- **Why:** Customer onboarding required a one-shot "verify everything we ship" call and a system-status pre-flight before pip install.
- **Status:** SHIPPED to PyPI.
- **Follow-on:** Monitor PyPI download counts; v0.5.4 will ship V3 cure if it lands.

### 3b32fc5 — `docs: CUSTOMER_ONBOARDING_v0.5.3 — 30-min path from pip install to verified pack`
- **Ship:** First-touch customer doc.
- **Why:** Sales pipeline needed a single-doc onboarding flow.
- **Status:** SHIPPED. `docs/CUSTOMER_ONBOARDING_v0.5.3.md`.
- **Follow-on:** Link from sipsalabs.com once refreshed for the launch.

### 1ebbd14 — `audit: SHA256_MANIFEST_2026_05_08.json — ground-truth fingerprints for all 17 v3 packs`
- **Ship:** Authoritative SHA256 manifest for all 17 v3 packs.
- **Why:** Customer-side hash check needed an authoritative manifest to match against `uc verify`.
- **Status:** SHIPPED to `docs/`.
- **Follow-on:** Refresh after every public HF artifact upload (the SSL-EOF retry chain may push some packs that change the manifest).

### d7f9a3f — `research+ops: seq=2048 long-context validation + Customer POC contract template`
- **Ship:** Qwen3-Base seq=2048 PPL ratio 1.0071 + customer POC contract template.
- **Why:** Headline number needed long-context footnote; POC contract for sales pipeline.
- **Status:** SHIPPED. `docs/PPL_EVAL_qwen3-1_7b-base_seq2048_2026_05_08.json` + `docs/CUSTOMER_PHASE_0_POC_CONTRACT_TEMPLATE.md`.
- **Follow-on:** Bench tables on sipsalabs.com need seq=1024 headline + seq=2048 footnote.

---

## Research — per-Linear adaptive bpw arc (REFUTED)

### ddca8f8 — `research+ops: per-Linear adaptive bpw proposal, Phi-3-mini in MODEL_REGISTRY, evening LAB-NOTEBOOK`
- **Ship:** Hypothesis doc + Phi-3-mini in MODEL_REGISTRY + LAB-NOTEBOOK entry.
- **Why:** k_proj quant_rel_l2 -55% observation at 405B and 1.7B scale → propose +1bpw on bottleneck Linears.
- **Status:** HYPOTHESIS REFUTED at end-PPL (see 0c6a65a).
- **Follow-on:** Re-anchor proposal on V3 rank-redistribution; do NOT cite v1 result externally.

### d3c74c2 — `research: per-Linear adaptive bpw v1 IN FLIGHT + Phi-3-mini PPL eval landed`
- **Ship:** Compression kicked off; Phi-3-mini PPL ratio 1.0026 at seq=128.
- **Why:** v1 in flight; Phi-3-mini result publishable but seq=128 caveat applies.
- **Status:** Phi-3-mini result GOOD (`PPL_EVAL_phi-3-mini-4k-instruct_2026_05_08.json`); v1 result REFUTED.
- **Follow-on:** Phi-3-mini at seq=1024 retest before publishing.

### 1b378d5 — `ops: inference throughput benchmark + per-Linear v1 autopipe + Yi-9B autopipe`
- **Ship:** Throughput bench + queued autopipes.
- **Why:** Customers want tokens/sec numbers; needed automated runs for v1 + Yi-9B.
- **Status:** Yi-9B SHIPPED (1.0041, see `PPL_EVAL_yi-1.5-9b_2026_05_08.json`); per-Linear v1 REFUTED.
- **Follow-on:** Throughput numbers in `docs/BENCHMARKS_2026_05_08.json`.

### 2fb18f8 — `ip: continuation-in-part draft for per-Linear adaptive bpw`
- **Ship:** CIP draft anchored on v1 quant-residual mechanism (12 method-form claims, $65 micro-entity).
- **Why:** Patent priority window for the per-Linear bpw mechanism.
- **Status:** DRAFT ONLY — anchored to refuted result. **DO NOT FILE.**
- **Follow-on:** Re-anchor on V3 rank-redistribution result before filing. Track A supplement (drafted earlier, due 2026-05-09 for $65 micro-entity) is the prioritized pre-funding patent expense.

### b6a7334 — `research: per-Linear v1 HONEST NEGATIVE landed + apples-to-apples re-eval queued`
- **Ship:** First indication v1 wasn't beating uniform; queued the apples-to-apples re-eval.
- **Why:** A 50-prompt v1 eval against the original 30-prompt baseline was ambiguous; needed apples-to-apples.
- **Status:** Confirmation came in 0c6a65a.
- **Follow-on:** HONEST_NEGATIVE_RESULTS doc updated (see entry #12 below).

### 0c6a65a — `research: per-Linear v1 FULLY REFUTED on PPL (apples-to-apples) + Yi-1.5-9B + Phi-2 in MODEL_REGISTRY`
- **Ship:** Apples-to-apples eval (n=50, seq=1024, baseline 12.0813). Uniform 1.004876, v1 1.005097, Δ=+0.00022.
- **Why:** Definitive A/B between adaptive-bpw v1 and uniform 5bpw at SAME baseline + SAME held-out window.
- **Status:** REFUTED. JSON files of record:
  - `docs/PPL_EVAL_qwen3-1.7b-base-uniform-n50_2026_05_08.json` (uniform)
  - `docs/PPL_EVAL_qwen3-1.7b-base-adaptive-bpw-v1_2026_05_08.json` (v1)
- **Follow-on:** V3 cure direction (see `docs/RESEARCH_v3_CURE_DIRECTION_2026_05_09.md`).

---

## Research — V18-C train_steps cure arc (V2)

### 35c8667 — `research: V18-C adaptive train_steps scheduler + autopipe queued`
- **Ship:** Linear ramp `train_steps × (1 + 4·depth_frac)` gated by `UC_ADAPTIVE_TRAIN_STEPS=1`.
- **Why:** Diagnostic showed train_loss climbs 4000× layer 0 → 27. Hypothesis: more steps in deep layers fixes it.
- **Status:** SCHEDULER MERGED. Implementation: `streaming_compression_runner.py:737-753`.
- **Follow-on:** v2 result landed (4747d15).

### 4747d15 — `research: V18-C adaptive train_steps v2 cure experiment autopipe (queued behind Phi-2 retry)`
- **Ship:** v2 experiment queued and run.
- **Why:** Test whether the depth-ramped train_steps scheduler closes the deep-layer train_loss gap.
- **Status:** COMPLETED. PPL ratio **1.004515** vs uniform 1.004876, Δ=-0.00037 (~1.2σ marginal win). JSON: `docs/PPL_EVAL_qwen3-1.7b-base-v2-adaptive-train-steps_2026_05_08.json`.
- **Confounder:** earlier `v18c_adaptive_steps` autopipe (same nominal config) gave 1.005051 — a 0.0005 spread between identical-config runs. **Variance suspect.** Need seed sweep before claiming v2 win is real.
- **Follow-on:** **V3 = rank-redistribution. See `docs/RESEARCH_v3_CURE_DIRECTION_2026_05_09.md`.** v2 is at best a +1σ result; V3 should be +3σ if rank is the true bottleneck.

---

## Architecture matrix expansion

### 6f9a597 — `ops: Yi-1.5-9B 20th arch full pack+upload autopipe queued`
- **Ship:** Architecture #20.
- **Why:** Public matrix expansion.
- **Status:** COMPLETED. Yi-1.5-9B PPL ratio 1.0041 (`PPL_EVAL_yi-1.5-9b_2026_05_08.json`, 48 layers, n=50).
- **Follow-on:** HF upload subject to SSL-EOF watchdog (see HONEST_NEGATIVE_RESULTS items 5-9).

### 7ce8111 — `fix: streaming_teacher Phi-2 final_layernorm dispatcher + Phi-2 retry autopipe`
- **Ship:** Phi-2 (Microsoft) needed a model-specific final_layernorm dispatch in `streaming_teacher.py`.
- **Why:** First failed Phi-2 compression hit a missing dispatcher case; patched and queued retry.
- **Status:** PATCH MERGED. Phi-2 compression retry queued via `_phi2_retry_v2_autopipe.sh`.
- **Follow-on:** Confirm Phi-2 ratio lands and retry result documented in MODEL_REGISTRY.

### 76b3a70 — `ops: overnight chain — Phi-2 retry v2 (PYTHONDONTWRITEBYTECODE) + Gemma-2-9B (or StableLM-2-12B fallback) 22nd arch`
- **Ship:** Chained Phi-2 retry → Gemma-2-9B as #22 with StableLM-2-12B fallback.
- **Why:** Cuda:1 conveyor belt for the night.
- **Status:** IN FLIGHT (overnight on cuda:1).
- **Follow-on:** Morning briefing autopipe (f1cd4b4) collects results.

---

## Documentation / launch

### 1329d23 — `content: YouTube demo script (4:15 runtime, 6 scenes) + Phi-3-mini autopipe`
- **Ship:** YouTube demo script + Phi-3-mini #19.
- **Why:** Product launch needed video script; Phi-3-mini #19 in matrix.
- **Status:** SCRIPT SHIPPED, video unrecorded.
- **Follow-on:** Record video once V3 lands (use V3 result as the closing scene).

### c89ceca — `[GitHub direct commit via gh API] README refresh 21-arch matrix company voice`
- **Ship:** README refresh, 21-arch matrix, company voice.
- **Why:** Needed to push README update without local clone (committed via `gh api`).
- **Status:** SHIPPED to github.com/sipsalabs/ultracompress.
- **Follow-on:** Bump to 22-arch (Gemma-2-9B) when overnight chain lands.

### 50e146e — `scrub: remove personal-name leak from public blog post (2 instances)`
- **Ship:** Personal-name removal in `BLOG_POST_v3_LOSSLESS_2026_05_08.md`.
- **Why:** Per "no personal info" policy from 2026-04-26 — zero personal info on any public surface.
- **Status:** SCRUBBED.
- **Follow-on:** Full grep sweep on `docs/` for any remaining personal names before launch.

### 60b1fdf — `ops: full autonomous chain for tonight + Sip's tomorrow-morning packet`
- **Ship:** Composed all overnight automations into one chain.
- **Why:** Single-source overnight orchestration.
- **Status:** SHIPPED.
- **Follow-on:** f1cd4b4 (morning briefing) is the consumer.

### f1cd4b4 — `ops: morning briefing autopipe — auto-aggregates overnight results into one doc`
- **Ship:** Single morning artifact summarizing what landed overnight.
- **Why:** Sip reads ONE doc in the morning, not 10.
- **Status:** AUTOPIPE QUEUED. Will produce `docs/MORNING_BRIEFING_2026_05_09.md`.
- **Follow-on:** **SIP READS THIS FIRST IN THE MORNING.**

---

## Cross-cutting follow-on TODOs

1. **Variance bound on v2 win** — re-run uniform + v2 with seeds {1, 2, 3} to bound σ. Current σ≈0.0003 estimate is from a single comparison; the 0.0005 spread between v18c_adaptive (1.005051) and v2_train_steps (1.004515) suggests σ may be 2× larger than quoted.
2. **V3 implementation** — see `docs/RESEARCH_v3_CURE_DIRECTION_2026_05_09.md`. Single env flag `UC_RANK_REDISTRIBUTE=1`, ~30 lines added to `streaming_compression_runner.py`.
3. **CIP re-anchoring** — patent draft (2fb18f8) currently anchored on refuted v1 mechanism. Re-anchor on V3 + train_loss-by-depth diagnostic before filing.
4. **HF upload watchdog status** — items 5-9 of `HONEST_NEGATIVE_RESULTS_2026_05_08.md` still in-flight. Check completion in morning.
5. **Phi-2 + Gemma-2-9B / StableLM-2-12B compression status** — Phi-2 retry v2 + arch #22 still in flight.
6. **HONEST_NEGATIVE_RESULTS update** — append entry #12 (v1 per-Linear refuted apples-to-apples). #13 deferred until V3 lands or refutes.

---

## Unified PPL comparison table (apples-to-apples set, n=50, seq=1024)

| Run | bpw | rank | train_steps | Baseline PPL | Compressed PPL | PPL ratio | Δ vs uniform | JSON |
|-----|-----|------|-------------|--------------|----------------|-----------|--------------|------|
| Uniform | 5 | 32 | 200 (flat) | 12.0813 | 12.140157 | 1.004876 | — (ref) | `PPL_EVAL_qwen3-1.7b-base-uniform-n50_2026_05_08.json` |
| v1 adaptive bpw | 5 (k_proj@6) | 32 | 200 (flat) | 12.0813 | 12.142830 | 1.005097 | +0.000220 (REFUTED) | `PPL_EVAL_qwen3-1.7b-base-adaptive-bpw-v1_2026_05_08.json` |
| v2 train_steps (final) | 5 | 32 | 200→1000 ramp | 12.0813 | 12.135798 | **1.004515** | -0.000361 (~1.2σ) | `PPL_EVAL_qwen3-1.7b-base-v2-adaptive-train-steps_2026_05_08.json` |
| v2 train_steps (early) | 5 | 32 | 200→1000 ramp | 12.0813 | 12.142272 | 1.005051 | +0.000175 (~0.6σ) | `PPL_EVAL_qwen3-1.7b-base-v18c-adaptive-train-steps_2026_05_08.json` |

Cross-architecture sanity (n=50, seq varies):

| Model | n_layers | seq_len | Baseline | Compressed | Ratio | JSON |
|-------|----------|---------|----------|------------|-------|------|
| Qwen3-1.7B-Base (uniform) | 28 | 1024 | 12.0813 | 12.140 | 1.0049 | uniform-n50 |
| Yi-1.5-9B | 48 | 1024 | 8.398 | 8.433 | 1.0041 | yi-1.5-9b |
| Phi-3-mini-4k-instruct | 32 | **128** | 11.999 | 12.030 | 1.0026 | phi-3-mini-4k-instruct |

Phi-3-mini is the standout but at seq=128 — needs a seq=1024 retest before claiming the 1.0026 number publicly.
