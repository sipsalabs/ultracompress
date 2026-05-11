# Public Surface Audit — 2026-05-10

**Auditor:** Read-only sweep, no settings changed
**Scope:** github.com/sipsalabs/ultracompress (master), sipsalabs.com (local copy), HF org SipsaLabs (local cache + repo templates)
**Reference policy:** `docs/SELECTIVE_DISCLOSURE_CHARTER_2026_05_10.md` (locally staged but NOT yet on master) — explicitly classifies LAB-NOTEBOOK, HONEST_NEGATIVE_RESULTS, outreach packs, investor memos, patent provisional PDFs, and the recipe values (rank, lr, train_steps, calibration size) as **PRIVATE**. Most of those items are currently committed to master in violation of that policy.

---

## Executive summary

The single biggest leak surface is `docs/` on the public master branch, which currently exposes a complete how-to-rebuild kit for UltraCompress: the production training recipe (`UC_ADAPTIVE_TRAIN_STEPS`, `UC_RANK_REDISTRIBUTE`, exact env-flag list, cuda-device choice, calibration size = 64 prompts × seq_len 1024, rank 32, train_steps 200, K-means K=32, block_size 64, FineWeb-edu held-out tail, seed 42, lr 1e-3 / 5e-4 with documented blow-up boundaries, AdamW + bf16 base + fp32 V/U/α), the full lab notebook with hypothesis-mechanism-experiment-measurement entries, 15 named honest-negative-result entries that read as a "what we already tried, save your time" map for any competitor, the V3 cure direction including expected PPL signal and ~30-LOC implementation hint, a continuation-in-part patent draft that doubles as a method-walkthrough for the per-Linear adaptive-bpw mechanism, two unreduced full-conference paper drafts (NeurIPS 2026 v2, ICLR 2027 v3) that publish recipe + headline numbers + negative-result catalogue in one document, the Mistral-7B v6/v6b diagnosis (which is locally staged today but if it ships will reveal the streaming-logit-KL objective's failure mode and the next codec direction), the personal billion-dollar plan with cash position, the seven-figure outreach plans naming the exact hot-tier prospects with the email body and ICP scores, the YC application referencing personal background details, and a customer-onboarding doc that exposes the per-Linear classes promoted to higher precision. The repo also still ships a CLAUDE.md-flagged "no personal info" violation: `docs/PATENT_PROVISIONAL_SKELETON.md` calls out micro-entity status and income threshold, `docs/YC_APPLICATION.md` opens with "Founder: Sip, 22, Solo Technical Founder", and `docs/PATENT_CIP_DRAFT_PER_LINEAR_ADAPTIVE_BPW_2026_05_08.md` lists the legal-name spelling. The website `index.html` and the `_packed_hermes_3_405b_v3_README_TEMPLATE.md` HF model-card template are largely fine: they leak K=32, block_size=64, low-rank (production-tuned) inside HTML body and badge SVGs but those four numbers are also explicitly disclosed in PATENT_NOTICE.md, so they're not the moat anymore. Cleanup scope: roughly **65 files to MOVE_TO_PRIVATE**, **~12 files to REDACT in place**, **3 files to DELETE outright** (the personal-info-laden ones and a couple of local-path-leaking autopipe shell scripts), and **everything else KEEP_PUBLIC**. The selective-disclosure charter is already drafted and locked — this audit is what executing that charter looks like as a punch list.

---

## SURFACE 1 — github.com/sipsalabs/ultracompress (master, BUSL-1.1)

### 1A — Production-recipe leaks (CRITICAL severity)

| File | Excerpt | Severity | Action |
|---|---|---|---|
| `docs/LAB-NOTEBOOK.md` | "5 bpw + correction overlay low-rank (production-tuned), 200 train steps, 64-prompt FineWeb-edu calibration" | CRITICAL | MOVE_TO_PRIVATE |
| `docs/HONEST_NEGATIVE_RESULTS_2026_05_08.md` | "scalar quantization K=32 + per-block(64) absmax + correction overlay low-rank (production-tuned) / 200 KL-distill steps" | CRITICAL | MOVE_TO_PRIVATE |
| `docs/RESEARCH_v3_CURE_DIRECTION_2026_05_09.md` | "Linear ramp rank=round(16 + 32*i/(n_layers-1)) at constant total budget" | CRITICAL | MOVE_TO_PRIVATE |
| `docs/V4D_MULTIPASS_RESEARCH_LOG_2026_05_09.md` | "rank=16 per pass; pass 1 input = correction_0; SVD warm-start pass 0 only" | CRITICAL | MOVE_TO_PRIVATE |
| `docs/V4_CURE_RESEARCH_2026_05_09.md` | "next cure path candidates: AWQ scaling, multi-pass, rank-redist" | CRITICAL | MOVE_TO_PRIVATE |
| `docs/RESEARCH_PROPOSAL_PER_LINEAR_ADAPTIVE_BPW_2026_05_08.md` | "k_proj quant_rel_l2 -55% on every layer at 405B and 1.7B" | CRITICAL | MOVE_TO_PRIVATE |
| `docs/PATENT_CIP_DRAFT_PER_LINEAR_ADAPTIVE_BPW_2026_05_08.md` | "Per-projection promotion to bpw+1; threshold tau_high=1.30" | CRITICAL | MOVE_TO_PRIVATE |
| `docs/PATENT_FILING_PACKET_2026_05_09.md` | "DEFER CIP — refuted; re-anchor on V3 rank-redistribution" | CRITICAL | MOVE_TO_PRIVATE |
| `docs/PATENT_DRAFT.md` | Full FRR/PHM/HWI/Q2 spec; 60x/239x/959x compression numbers | CRITICAL | MOVE_TO_PRIVATE |
| `docs/PATENT_PROVISIONAL_SKELETON.md` | Filing skeleton + "micro entity if under $206K/yr gross income" | CRITICAL | DELETE (template; personal income reference) |
| `docs/PATENT_CLAIMS_SUMMARY.md` | Full claim-by-claim breakdown of inventions + headline numbers | CRITICAL | MOVE_TO_PRIVATE |
| `docs/PATENT_CIP_DRAFT_PER_LINEAR_ADAPTIVE_BPW_2026_05_08.md` | "Inventor: Missipssa Ounnar"; method-form claims 1–7 | CRITICAL | MOVE_TO_PRIVATE (also legal name leak) |
| `docs/INVENTION_SKETCH.md` | "Computational Organism… DNA-like… process not weights" | CRITICAL | MOVE_TO_PRIVATE |
| `docs/WHAT_WORKS.md` | "ENT_POW 1.5 / latent_w 1.0->0.1 / LR 2e-4 cosine / 80K steps / SEQ_LEN=128" | CRITICAL | MOVE_TO_PRIVATE |
| `docs/MISTRAL_V6_V6B_DIAGNOSIS_2026_05_10.md` (local only; not yet on master per `git ls-tree`) | "v6 lr=1e-3 r=64 catastrophic; v6b lr=5e-4 r=48 near-fail" | CRITICAL | KEEP_LOCAL_DO_NOT_PUSH |
| `docs/SELECTIVE_DISCLOSURE_CHARTER_2026_05_10.md` (local only) | The charter itself is the disclosure policy — do NOT push | CRITICAL | KEEP_LOCAL_DO_NOT_PUSH |
| `docs/NEURIPS_2026_PAPER_v2_2026_05_09.md` | Full method recipe + 13 negative-result catalogue + per-layer telemetry | CRITICAL | MOVE_TO_PRIVATE |
| `docs/ICLR_2027_PAPER_v3_2026_05_09.md` | Same recipe, different venue framing | CRITICAL | MOVE_TO_PRIVATE |
| `docs/NEURIPS_2026_PAPER_DRAFT_2026_05_08.md` | v1 of the same recipe paper | CRITICAL | MOVE_TO_PRIVATE |
| `docs/PAPER_DRAFT.md` | Reproduction-ready training & eval pipeline | CRITICAL | MOVE_TO_PRIVATE |

### 1B — Production training/compression scripts on the public tree (CRITICAL/HIGH)

| File | Excerpt | Severity | Action |
|---|---|---|---|
| (production trainer, patent-protected) | The runner. Per-Linear correction overlay trainer + UC_ADAPTIVE_BPW + UC_AWQ_SCALING + UC_ADAPTIVE_TRAIN_STEPS gates exposed | CRITICAL | MOVE_TO_PRIVATE (or strip to a stub interface) |
| `scripts/overlay/stream_compress_e2e.py` | End-to-end driver invoked by every autopipe with full flags | CRITICAL | MOVE_TO_PRIVATE |
| `scripts/overlay/streaming_compression_logit_kl_runner.py` | Alternate logit-KL objective (the v6 catastrophic path) | CRITICAL | MOVE_TO_PRIVATE |
| `scripts/overlay/streaming_compression_mistral_runner.py` | Per-arch dispatch including the Mistral diagnosis substrate | HIGH | MOVE_TO_PRIVATE |
| `scripts/overlay/streaming_compression_online_runner.py` | Online variant of the runner | HIGH | MOVE_TO_PRIVATE |
| `scripts/overlay/streaming_compression_hybrid_runner.py` | Hybrid runner | HIGH | MOVE_TO_PRIVATE |
| `scripts/overlay/streaming_teacher.py` | Teacher hidden cache builder + per-arch dispatch table | HIGH | MOVE_TO_PRIVATE |
| `scripts/overlay/streaming_teacher_ppl.py` | Streaming bf16 baseline PPL — same comparator the 405B headline uses | HIGH | MOVE_TO_PRIVATE |
| `uc verify` | Eval harness that produced every PPL_EVAL JSON cited in marketing | HIGH | MOVE_TO_PRIVATE |
| `scripts/overlay/_per_linear_v1_autopipe.sh` | Recipe + exact env flags + cuda:1 + log paths | HIGH | MOVE_TO_PRIVATE |
| `scripts/overlay/_v18c_adaptive_steps_autopipe.sh` | Same — full env-flag invocation | HIGH | MOVE_TO_PRIVATE |
| `scripts/overlay/_v4a_awq_autopipe.sh` | AWQ-scaling failure-mode reproducer | HIGH | MOVE_TO_PRIVATE |
| `scripts/overlay/_v4d_multipass_autopipe.sh` | Multi-pass failure-mode reproducer | HIGH | MOVE_TO_PRIVATE |
| `scripts/overlay/_hermes_405b_post_compression_autopipe.sh` | Reveals exact 405B post-pack pipeline + HF watchdog | HIGH | MOVE_TO_PRIVATE |
| `scripts/overlay/_v2_adaptive_train_steps_autopipe.sh` | The marginal-win cure pipeline | HIGH | MOVE_TO_PRIVATE |
| `scripts/overlay/_morning_briefing_autopipe.sh` | Founder workflow autopipe | MED | MOVE_TO_PRIVATE |
| `scripts/overlay/_apples_apples_reeval_autopipe.sh` | Re-eval methodology | MED | MOVE_TO_PRIVATE |
| `scripts/overlay/_arch20_yi9b_autopipe.sh` | Per-arch driver | MED | MOVE_TO_PRIVATE |
| `scripts/overlay/_arch21_phi2_autopipe.sh` | Per-arch driver | MED | MOVE_TO_PRIVATE |
| `scripts/overlay/_arch22_gemma2_9b_autopipe.sh` | Per-arch driver | MED | MOVE_TO_PRIVATE |
| `scripts/overlay/_eval_4_new_archs.sh` | Multi-arch eval orchestration | MED | MOVE_TO_PRIVATE |
| `scripts/overlay/_full_hash_audit.sh` | Hash audit script | MED | MOVE_TO_PRIVATE |
| `scripts/overlay/_hf_upload_watchdog.sh` | HF upload retry shape | MED | MOVE_TO_PRIVATE |
| `scripts/overlay/_olmo2_retry.sh` | Per-arch retry | MED | MOVE_TO_PRIVATE |
| `scripts/overlay/_phi2_retry_autopipe.sh` | Per-arch retry | MED | MOVE_TO_PRIVATE |
| `scripts/overlay/_phi2_retry_v2_autopipe.sh` | Per-arch retry | MED | MOVE_TO_PRIVATE |
| `scripts/overlay/_phi3_mini_after_download.sh` | Per-arch driver | MED | MOVE_TO_PRIVATE |
| `scripts/overlay/_smollm2_instruct_after_olmo.sh` | Per-arch chain driver | MED | MOVE_TO_PRIVATE |
| `scripts/overlay/_yi_1_5_9b_upload_autopipe.sh` | Per-arch upload driver | MED | MOVE_TO_PRIVATE |
| `scripts/overlay/_test_mistral_v3_repro.py` | Reproduction substrate | MED | MOVE_TO_PRIVATE |
| `scripts/overlay/_gpu1_arch_queue.sh` | The conveyor-belt queue | MED | MOVE_TO_PRIVATE |
| `scripts/overlay/_analyze_claim20.py` | Patent-claim analyzer | MED | MOVE_TO_PRIVATE |

### 1C — FRR / Track B research scripts that anchor patent 64/049,517 (HIGH)

| File | Excerpt | Severity | Action |
|---|---|---|---|
| `scripts/frr/run_hq4_ceiling_break.py` | "ENT_POW 1.5; latent_w_final 0.1; LR 2e-4; 80K steps; SEQ_LEN=128; 28 teacher layers" | HIGH | MOVE_TO_PRIVATE |
| `scripts/frr/run_hq3_breakthrough.py` | HQ3 baseline trainer | HIGH | MOVE_TO_PRIVATE |
| `scripts/frr/launch_hq4_detached.py` | Detached launcher | HIGH | MOVE_TO_PRIVATE |
| `scripts/frr/launch_hq5_detached.py` | Detached launcher | HIGH | MOVE_TO_PRIVATE |
| `scripts/frr/launch_hq6_detached.py` | Detached launcher | HIGH | MOVE_TO_PRIVATE |
| `scripts/frr/launch_hq7_longhorizon.py` | Long-horizon launcher | HIGH | MOVE_TO_PRIVATE |
| `scripts/frr/launch_hires_eval_hq5.py` | Eval launcher | HIGH | MOVE_TO_PRIVATE |
| `scripts/frr/launch_monday_eval.py` | Eval orchestrator | HIGH | MOVE_TO_PRIVATE |
| `scripts/frr/run_baseline_distill.py` | Baseline distillation | HIGH | MOVE_TO_PRIVATE |
| `scripts/frr/run_deq_frr.py` | DEQ-FRR research | HIGH | MOVE_TO_PRIVATE |
| `scripts/frr/run_frr_generic.py` | Generic FRR trainer | HIGH | MOVE_TO_PRIVATE |
| `scripts/frr/run_frr_hq8.py` | HQ8 trainer | HIGH | MOVE_TO_PRIVATE |
| `scripts/frr/run_spectral_latent.py` | Spectral-latent variant | HIGH | MOVE_TO_PRIVATE |
| `scripts/frr/scale_eval.py` | Scaling eval | HIGH | MOVE_TO_PRIVATE |
| `scripts/frr/combined_stack_eval.py` | Composed-stack eval | HIGH | MOVE_TO_PRIVATE |
| `scripts/frr/hires_eval.py` | High-res eval harness | HIGH | MOVE_TO_PRIVATE |
| `scripts/frr/run_1.7b_tinyfrr_hq2.py` | Trainer | HIGH | MOVE_TO_PRIVATE |
| `experiments/training/*.py` | Production training scripts | HIGH | MOVE_TO_PRIVATE |
| `experiments/sweeps/sweep_tinyfrr*.py` | Hyperparam sweep configs | HIGH | MOVE_TO_PRIVATE |
| `experiments/eval/*.py` | Eval harnesses | MED | MOVE_TO_PRIVATE |

### 1D — Strategy / business / personal docs (HIGH/CRITICAL)

| File | Excerpt | Severity | Action |
|---|---|---|---|
| `docs/BUSINESS_PLAN.md` | "$0 in funding; built on 2x RTX 5090; 22 years old; $500K seed" | HIGH | DELETE (personal info + outdated pitch) |
| `docs/BILLION_DOLLAR_PATH_2026_05_08.md` | "$0 revenue; $130 cash budget; June 5 YC accept; bus factor 1" | HIGH | MOVE_TO_PRIVATE |
| `docs/BILLION_DOLLAR_WEEKEND_FOCUS_2026_05_09.md` | "$7-10K/month replacement income needed to quit day job" | HIGH | MOVE_TO_PRIVATE |
| `docs/YC_APPLICATION.md` | "Founder: Sip, 22, Solo Technical Founder; The Seed scored 90.6% on AGI Gauntlet" | HIGH | DELETE (personal info; violates no-personal-info policy) |
| `docs/YC_APPLICATION_DRAFT.md` | Earlier draft, same personal info | HIGH | DELETE |
| `docs/YC_INTERVIEW_PREP_2026_05_08.md` | Interview answers + personal framing | HIGH | MOVE_TO_PRIVATE |
| `docs/YC_MAY_UPDATE_v6_2026_05_09.md` | YC update prose | HIGH | MOVE_TO_PRIVATE |
| `docs/SERIES_A_PITCH_DECK_2026_05_08.md` | Pitch deck content | HIGH | MOVE_TO_PRIVATE |
| `docs/INVESTOR_UPDATE_TEMPLATE_2026_05_08.md` | Template that may carry personal numbers | HIGH | MOVE_TO_PRIVATE |
| `docs/PITCH.md` | Pitch text | HIGH | MOVE_TO_PRIVATE |
| `docs/MORNING_CHECKLIST_2026_05_09.md` | Founder personal workflow + local paths | HIGH | MOVE_TO_PRIVATE |
| `docs/MONDAY_CHECKLIST.md` | Same shape | HIGH | MOVE_TO_PRIVATE |
| `docs/MORNING_REPORT.md` | Same shape | HIGH | MOVE_TO_PRIVATE |
| `docs/TOMORROW_MORNING_AT_A_GLANCE_2026_05_09.md` | Same shape | HIGH | MOVE_TO_PRIVATE |
| `docs/IP_AND_SPONSORSHIP_SETUP_2026_05_09.md` | Internal IP-hardening playbook | HIGH | MOVE_TO_PRIVATE |
| `docs/COMMIT_KNOWLEDGE_MAP_2026_05_09.md` | Per-commit "what shipped + what's refuted + what to file next" | CRITICAL | MOVE_TO_PRIVATE |
| `docs/RESEARCH_PHASE0_ICP_RANKING_2026_05_09.md` | Hot-tier prospect names + scores + email targets + objection-handling | CRITICAL | MOVE_TO_PRIVATE |
| `docs/ICP_EXPANSION_40_2026_05_09.md` | 40-prospect ICP expansion list | HIGH | MOVE_TO_PRIVATE |
| `docs/OUTREACH_2026_05_08/*.md` | Cold-email drafts to Tri Dao, Albert Gu, Yi Tay, Lambda, NASA | HIGH | MOVE_TO_PRIVATE |
| `docs/OUTREACH_PHASE0_POC_2026_05_09/*.md` | Cold-email drafts to Together / Fireworks / Replicate / Mistral / Modal / Glean / Harvey / Lambda / CoreWeave / Palantir / Ollama | HIGH | MOVE_TO_PRIVATE |
| `docs/CUSTOMER_PHASE_0_POC_OFFER_LETTER.md` | Offer letter template | MED | MOVE_TO_PRIVATE |
| `docs/CUSTOMER_PHASE_0_POC_CONTRACT_TEMPLATE.md` | Contract template | MED | MOVE_TO_PRIVATE |
| `docs/CUSTOMER_ONBOARDING_v0.5.3.md` | Customer flow incl. internal config notes | MED | REDACT (keep public flow; remove internal details) |
| `docs/NASA_SBIR_PHASE1_PROPOSAL_DRAFT_2026_05_08.md` | NASA SBIR draft | HIGH | MOVE_TO_PRIVATE |
| `docs/PRESS_RELEASE_DISTRIBUTION_LIST_2026_05_09.md` | 12-reporter list | HIGH | MOVE_TO_PRIVATE |
| `docs/PRESS_RELEASE_HERMES_405B_2026_05_09.md` | Press release draft | MED | KEEP_PUBLIC after publish; until then PRIVATE |
| `docs/FIVERR_GIG.md` | Fiverr gig listing | LOW | MOVE_TO_PRIVATE |
| `docs/APPS_ADMIN_SWEEP_2026_05_08.md` | Admin sweep notes | LOW | MOVE_TO_PRIVATE |

### 1E — Launch / blog / show-HN drafts that may publish recipe details (MED)

| File | Excerpt | Severity | Action |
|---|---|---|---|
| `docs/SHOW_HN.md` | Original Show HN draft | MED | REDACT (results yes, recipe no) |
| `docs/SHOW_HN_V2_2026_05_08.md` | v2 draft | MED | REDACT |
| `docs/SHOW_HN_V3_2026_05_09.md` | v3 draft (cites low-rank (production-tuned) / 200 steps) | MED | REDACT |
| `docs/LAUNCH_LINKEDIN_7_STARS_2026_05_08.md` | LinkedIn drafts | MED | KEEP_PUBLIC after publish |
| `docs/LAUNCH_LINKEDIN_HERMES_405B_2026_05_09.md` | LinkedIn drafts | MED | KEEP_PUBLIC after publish |
| `docs/LAUNCH_LINKEDIN_MULTI_ARCH_2026_05_08.md` | LinkedIn drafts | MED | KEEP_PUBLIC after publish |
| `docs/LAUNCH_THREAD_*` | Twitter thread drafts | MED | KEEP_PUBLIC after publish |
| `docs/LAUNCH_POSTS.md` | Aggregate post pool | MED | KEEP_PUBLIC after publish |
| `docs/TWITTER_THREAD.md` | Twitter thread | MED | KEEP_PUBLIC after publish |
| `docs/BLOG_POST_v3_LOSSLESS_2026_05_08.md` | Blog post (recipe-light) | MED | KEEP_PUBLIC after polish |
| `docs/PUBLIC_VERIFICATION_DASHBOARD_2026_05_08.md` | Public dashboard | LOW | KEEP_PUBLIC |
| `docs/HF_MODEL_CARD_HERMES_3_405B.md` | "Quantization: 5 bits per weight (bpw); GQA, query:kv head ratio 16:1" | MED | KEEP_PUBLIC (matches the published HF card) |
| `_packed_hermes_3_405b_v3_README_TEMPLATE.md` | "5-bit scalar quantization k-means + correction overlay low-rank correction (rank 32) + per-block fp32 absmax" | MED | KEEP_PUBLIC (already disclosed in PATENT_NOTICE) |

### 1F — Core ultracompress/* package modules (LOW; the public API surface)

| File | Excerpt | Severity | Action |
|---|---|---|---|
| `ultracompress/pack_v3.py` | The v3 pack format implementation; K=32 default | LOW | KEEP_PUBLIC (this is the consumer-side reconstruction path; required for `uc verify`) |
| `ultracompress/cli.py` | Click-style CLI surface | LOW | KEEP_PUBLIC |
| `ultracompress/codec.py` | Codec primitives | LOW | KEEP_PUBLIC |
| `ultracompress/load_uc.py` | Pack loader | LOW | KEEP_PUBLIC |
| `ultracompress/verify.py` / `verify_org.py` | Verifier (the trust asset) | LOW | KEEP_PUBLIC |
| `ultracompress/inference.py` | Reference inference path | LOW | KEEP_PUBLIC |
| `ultracompress/calibrate.py` / `calibrated_pq.py` | The per-Linear k-means calibration substrate | MED | REDACT (keep public symbols and signatures, move trainer specifics behind a private hook) |
| `ultracompress/quantize.py` / `binarize.py` / `entropy_coding.py` / `hadamard.py` / `factorize.py` | Codec building blocks | LOW | KEEP_PUBLIC |
| `ultracompress/moonshot.py` / `genome_*.py` / `compress_mind.py` / `living_frr.py` / `organism.py` / `crystal_net.py` / `hyperbolic.py` / `holographic_boundary.py` / `dendritic.py` / `thalamic.py` / `hypercomplex.py` / `paradigm_shift.py` / `protein_fold.py` / etc. | The 200+ Athena-AGI-flavored modules; each is named after a Track-D / patent-claim invention category | MED | MOVE_TO_PRIVATE (these are part of the moat; keep names off the public tree) |
| `ultracompress/multi_block_frr.py` / `spiral_frr.py` / `stateful_frr.py` / `frr_to_organism.py` | Track-B FRR subsystems | MED | MOVE_TO_PRIVATE |
| `ultracompress/cawn_resonance.py` / `wave_engine.py` / `phase_net.py` / `oscillatory.py` / `chaos_fractal.py` / `cellular_automata.py` | Bio-inspired computation paths (Track-D direction) | MED | MOVE_TO_PRIVATE |
| `archive/compress_v8.py` … `archive/compress_v13.py`, `archive/eval_v18_ppl.py`, `archive/screen_v16.py`, etc. | Older trainer / vocab / eval scripts | MED | MOVE_TO_PRIVATE |

### 1G — Patent / IP / claim leaks specific to claim 16, 20, 21 (MED/HIGH)

| File | Excerpt | Severity | Action |
|---|---|---|---|
| `docs/claim20_summary.txt` | Patent claim 20 narrative | MED | MOVE_TO_PRIVATE |
| `docs/claim16_bpw.png` / `claim16_envelope.png` | Patent claim 16 plots | MED | MOVE_TO_PRIVATE |
| `scripts/overlay/claim21_*.py` (≈ 60 files) + `results/claim21_*.json` (≈ 100 files) | Per-claim experimental sweep scripts and results | MED | MOVE_TO_PRIVATE (each one names a patent claim) |
| `scripts/overlay/claim21_finetune_delta_patent_block.py` etc. | "patent_block" naming explicitly maps script -> claim | HIGH | MOVE_TO_PRIVATE |
| `scripts/overlay/wave45_finisher.py` … `wave48_finisher.py` | Per-wave patent-claim experiments | MED | MOVE_TO_PRIVATE |
| `docs/PPL_EVAL_*.json` | Per-arch PPL JSONs (already cited in README; baselines + ratios are the public claim) | LOW | KEEP_PUBLIC |
| `docs/BENCHMARKS_2026_05_08.json` / `BENCHMARKS_2026_05_09.json` | Headline benchmark JSONs | LOW | KEEP_PUBLIC |
| `docs/SHA256_MANIFEST_2026_05_08.json` | Hash manifest for verifier flow | LOW | KEEP_PUBLIC (this IS the trust asset) |
| `docs/PATENT_FILING_CHECKLIST.md` | Filing checklist | MED | MOVE_TO_PRIVATE |
| `docs/ARXIV_CHECKLIST.md` | arXiv submission checklist | LOW | MOVE_TO_PRIVATE |

### 1H — Repo-root and CI files (LOW; mostly fine)

| File | Excerpt | Severity | Action |
|---|---|---|---|
| `README.md` | "FineWeb-edu held-out tail at seq_len=1024, seed=42" + "low-rank low-rank correction matrices, trained per-layer for production KL distillation" | MED | REDACT (replace exact 200/32 with "rank in the order of dozens, hundreds of distillation steps", or invoke patent-cover language) |
| `README.previous.md` | Earlier README; likely same leaks | MED | DELETE (old version; patent already cites "as of"-dated artifacts) |
| `LICENSE` (BUSL-1.1) | Standard text | LOW | KEEP_PUBLIC |
| `LICENSE.apache` | Standard text | LOW | KEEP_PUBLIC |
| `NOTICE.md` | Licensing rationale; references 1.0066x | LOW | KEEP_PUBLIC |
| `PATENT_NOTICE.md` | "scalar quantization at 5 bpw; correction overlay trained via KL distillation; W = scalar_dequantize(codes) * absmax + alpha * low_rank(U, V); per-layer streaming pipeline" | LOW | KEEP_PUBLIC (patent gives legal cover; this exact sentence is the cite-and-defend posture) |
| `PATENT_CLAIMS.md` | Detailed claim-by-claim breakdown referencing v17 / α=0.125 / per-role exponent (0.25, 0.125) etc. | HIGH | MOVE_TO_PRIVATE (this exposes specific operating points beyond what the provisional itself covers) |
| `REPRODUCE.md` | "FineWeb-Edu first 500 M tokens; tokenized with Qwen3 tokenizer; tail 50M tokens for eval; seed=42" + "HQ5 h256 311x / 55.40% / 69.64%" | HIGH | REDACT (drop dataset slice sizes + headline-record numbers; keep environment block) |
| `RESULTS.md` | Headline result narrative | LOW | KEEP_PUBLIC if recipe-free; REDACT otherwise |
| `CHANGELOG.md` | Per-version changes | LOW | KEEP_PUBLIC |
| `CONTRIBUTING.md` | Contributor guide | LOW | KEEP_PUBLIC |
| `SECURITY.md` | Security disclosure path | LOW | KEEP_PUBLIC |
| `.github/FUNDING.yml` | Sponsor wiring | LOW | KEEP_PUBLIC |
| `.github/workflows/test.yml` / `regression_tests.yml` | CI configs | LOW | KEEP_PUBLIC |
| `_refresh_hf_org_bio.py` / `_refresh_hf_readmes_2026_05_08.py` | Internal HF refresh helpers; reveal automation pattern | MED | MOVE_TO_PRIVATE |
| `_packed_hermes_3_405b_v3_README_TEMPLATE.md` | HF model-card template (matches the Hermes-405B card on HF) | LOW | KEEP_PUBLIC |
| `serve.py` / `demo.py` | Reference inference + demo | LOW | KEEP_PUBLIC |
| `tools/ptq_tinyfrr.py` / `tools/factor_lmhead.py` / `tools/prepare_500M_tokens.py` / `tools/wait_and_ptq.py` | Internal tooling that ties dataset prep + PTQ to specific training paths | HIGH | MOVE_TO_PRIVATE |
| `tools/compress_e2e.py` / `tools/compress_frr.py` | Wrappers around the trainers | HIGH | MOVE_TO_PRIVATE |
| `tools/create_model_card.py` | HF card generator | LOW | KEEP_PUBLIC |
| `tools/api_design.py` / `tools/app.py` / `tools/arena.py` / `tools/quickstart.py` | Customer-side helpers | LOW | KEEP_PUBLIC |

### 1I — Logs directory (HIGH severity — reveals every run that's been done)

| File | Excerpt | Severity | Action |
|---|---|---|---|
| `logs/*.log` (29 files) | per-run logs: `overlay_002.log`, `overlay_fp8_qwen8b.log`, `lambada_hifi_6m.log`, `verify_8b.log`, etc. | HIGH | MOVE_TO_PRIVATE (any one log file shows exact configs run on which arch) |

---

## SURFACE 2 — sipsalabs.com (`C:\Users\scamd\OneDrive\Desktop\Projects\sip\sipsalabs-site\index.html`)

| Excerpt | Severity | Action |
|---|---|---|
| `Mathematically lossless reconstruction of W_base for 5-bit transformer weights` | LOW | KEEP_PUBLIC (this is the headline pitch; same as PATENT_NOTICE) |
| `Hermes-3-405B eval methodology: n=50 prompts · seq_len=1024 · FineWeb-edu held-out tail · seed=42 · baseline PPL 5.0358 · compressed PPL 5.0692` (records-note line) | MED | REDACT (drop seed, drop FineWeb-edu specifically; keep n / seq / ratio) |
| `13 architectures cleared the production threshold` (production-tier line) | LOW | KEEP_PUBLIC |
| Records table — exact PPL ratios per model down to 5 decimal places | LOW | KEEP_PUBLIC (these are the customer trust assets the entire pitch rests on) |
| `Patent pending — USPTO Filed 2026-04` badge | LOW | KEEP_PUBLIC |
| `One founder. Five months. Two patents.` headline | LOW | KEEP_PUBLIC (no name, no DOB, no income) |
| `dual RTX 5090 workstation, hand-built` | LOW | KEEP_PUBLIC (color; no operational secret) |
| `Phase 0 POC slots for Q3 2026 · One week, paid, $5K` | LOW | KEEP_PUBLIC (intentional commercial signal) |
| `huggingface.co/SipsaLabs` link | LOW | KEEP_PUBLIC |
| `github.com/sipsalabs/ultracompress/blob/main/PATENT_NOTICE.md` link | LOW | KEEP_PUBLIC |
| `Athena · Internal architecture private` card | LOW | KEEP_PUBLIC (good selective-disclosure modeling) |
| `Quant Trading System · Passed live-funded $150K · TopStep Express` | LOW | KEEP_PUBLIC (consistent with brand/portfolio narrative) |
| Side dirs `blog/` (3 sub-pages) and `poc/` and `benchmarks/` | (not read in this audit; needs separate sweep at next pass) | MED | NEXT_AUDIT |
| `POST_FILING_RESTORE.md` (in site directory) | (not read this pass) | MED | NEXT_AUDIT |

The site itself is in good shape. The single line worth tightening is the records-note seed/dataset reveal in the Hermes-405B methodology footer.

---

## SURFACE 3 — HF org SipsaLabs (local cache + repo template)

The `~/.cache/huggingface/hub/spaces--SipsaLabs--README/snapshots/<sha>/README.md` is the org landing-page README, and the `_packed_hermes_3_405b_v3_README_TEMPLATE.md` in the repo root is the per-model-card template. There are 22 SipsaLabs models in the local HF cache; this audit cannot read each per-model README from the live HF surface (would require gh/hf-hub fetch), so the assessment is based on the local org README + the repo's per-model template.

| File | Excerpt | Severity | Action |
|---|---|---|---|
| HF org README (cached) | `Apache-licensed bases retained; compression metadata under the Sipsa Labs Research Evaluation License v1.0` | LOW | KEEP_PUBLIC |
| HF org README | `USPTO 64/049,511 (Track A — Activation-Aware Row-Overlay Quantization) and 64/049,517 (Track B — Fractal Residual Recursion)` | MED | REDACT (the public USPTO numbers are fine; the internal track-name decomposition "Track A/B" maps to patent-claim moats and to the Track A/B/C/D taxonomy in `MEMORY.md`. Drop the Track-A / Track-B labels; keep only the USPTO numbers.) |
| HF org README | `Track A 2.798-bit cohort design point` | HIGH | REDACT (this exposes a specific design point we don't otherwise publicly cite, and ties Track A to a sub-3-bpw operating regime) |
| HF org README | "smollm2 · mistral · olmo2 · llama variants throughout 2026-05" rolling-release language | LOW | KEEP_PUBLIC |
| `_packed_hermes_3_405b_v3_README_TEMPLATE.md` (repo) | `5 bpw + correction overlay low-rank correction (rank 32) + per-block fp32 absmax scales` | MED | KEEP_PUBLIC (matches PATENT_NOTICE.md disclosure scope; recipe is patent-cited) |
| Same template | "Patent disclosure" footer mentioning "Five additional supplementary provisionals are scheduled to file 2026-05-09" | MED | REDACT (don't disclose forward-looking patent strategy; replace with "additional continuations and supplements pending") |
| Same template | `<<TBD AT EVAL TIME>>` placeholders + `<<sha256sum layer_000.uc TBD post-pack>>` | LOW | KEEP_PUBLIC if rendered before publish (template artifact only) |
| `docs/HF_MODEL_CARD_HERMES_3_405B.md` (repo) | Detailed model-card with full numbers + licensing block + GQA 16:1 ratio | LOW | KEEP_PUBLIC (matches what's already on HF and on the website) |
| `docs/HERMES_405B_HF_README_STUB.md` | Stub for the HF card | LOW | KEEP_PUBLIC |
| `docs/HF_TIER2_GATING_2026_05_10/*` (locally staged, not on master) | Staged tier-2 gating drafts for Yi-1.5-9B / Mixtral-8x7B / Phi-3.5-MoE / Qwen3-32B | MED | KEEP_LOCAL_DO_NOT_PUSH (these include reduced-detail rewrites — that's the point) |

---

## What got skipped (caveats)

- The audit did not crawl `experiments/archive/*.py` (~150 files) one-by-one — they're sampled-named in the listing and they're all old training scripts; the recommended action class is uniformly **MOVE_TO_PRIVATE** with no per-file justification needed.
- The audit did not crawl every `scripts/overlay/claim21_*.py` file individually — there are ~60 of them, all named after patent claims and all part of the same "patent_block" pattern. Same uniform recommendation: MOVE_TO_PRIVATE.
- The audit did not fetch live HF model-card READMEs over the network. The repo-side template + the org README + the local cached space README are the substrate. Per-model READMEs may differ from the template if Sip edited them on the HF UI; recommend a separate `gh api` sweep or manual visit to confirm.
- The site `blog/`, `poc/`, and `benchmarks/` subdirectories were not opened (only `index.html` was per the prompt). Strongly recommend a follow-up audit pass on those before the BUSL-1.1 → v0.6 launch.
- The local `MISTRAL_V6_V6B_DIAGNOSIS_2026_05_10.md` and `SELECTIVE_DISCLOSURE_CHARTER_2026_05_10.md` are NOT yet on master per `git ls-tree HEAD`. They are flagged here as "do NOT push" rather than "MOVE_TO_PRIVATE". The `HF_TIER2_GATING_2026_05_10/` directory and `HF_ARTIFACT_GATING_PLAN_2026_05_10.md` are also locally staged only.

---

## One-line summary

**Worst single leak:** `docs/COMMIT_KNOWLEDGE_MAP_2026_05_09.md` + `docs/HONEST_NEGATIVE_RESULTS_2026_05_08.md` + `docs/RESEARCH_v3_CURE_DIRECTION_2026_05_09.md` together — read in sequence they tell a competitor exactly what's been tried, what works, what's about to work, and the ~30-LOC implementation hint to land the next headline. Cleanup scope: **~65 files MOVE_TO_PRIVATE, ~12 REDACT, 3 DELETE, the rest KEEP_PUBLIC**, executed in a single PR against master that strips all of `docs/` (except dashboard / benchmark JSONs / model-card / press-release-after-publish), all of `scripts/overlay/` and `scripts/frr/` (except a stripped public-facing wrapper), `experiments/`, `logs/`, and the strategy-deck files.

Codec internals + training procedure are patent-protected (USPTO 64/049,511 + 64/049,517).
