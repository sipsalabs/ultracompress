# Public Surface Audit v2 — ACTUAL public branch

**Date:** 2026-05-10
**Auditor:** Read-only sweep, no settings or files changed, nothing pushed
**Scope:** `github.com/sipsalabs/ultracompress` branch `main` (the actual public surface). `git fetch sipsalabs main` synced first; enumeration via `git ls-tree -r sipsalabs/main --name-only` (82 files total).
**Reference policy:** `docs/SELECTIVE_DISCLOSURE_CHARTER_2026_05_10.md` (locked 2026-05-10).
**Supersedes:** `docs/PUBLIC_SURFACE_AUDIT_2026_05_10.md` (v1) was scoped to local `master` (which tracks the *private* `mounnar/ultracompress`). Most v1 findings do not apply to `sipsalabs/main`. This v2 is the corrected real-world risk picture.

---

## Executive (3 lines)

The actual public branch has **9 files (~11% of 82)** that publish enough recipe to let a competitor reproduce the production result end-to-end: `block_size=64`, `low-rank (production-tuned)`, `train_steps=200`, `seed=42`, calibration size, `n_calib`, layer-by-layer streaming mechanism, and the scalar quantization + correction overlay framing — the exact list the Selective Disclosure Charter §1 forbids. Internal "Track A / Track B / Track A+B / track-a-row-overlay / track-b-frr / correction overlay / scalar quantization / Cure-A4" nomenclature is wired into a published manifest schema and a public CLI `--track` flag, which means it is now load-bearing wire protocol — every artifact on HF inherits the leak.
**Worst-case-interpretation exposure: 7/10** (high but not catastrophic — no credentials, no personal legal name, no income disclosure, no outreach packs, no patent-claim drafts, no lab notebook on the public branch; the leak is concentrated in recipe + naming + 2 file paths with the Windows username `scamd`).

---

## File-by-file table on `sipsalabs/main`

Severity scale: CRITICAL = recipe-or-IP-moat-level disclosure that a competitor can act on tomorrow. HIGH = exposes internal taxonomy that maps to patent-claim moats. MED = single recipe field or partial recipe leak. LOW = mild personal-info or stylistic concern. CLEAN = no charter violation.

### Recipe / runner / artifact leaks (CRITICAL & HIGH)

| File | Severity | What leaks | Recommended action |
|---|---|---|---|
| (production trainer, patent-protected) (1200 LOC) | CRITICAL | Docstring: "Track A (scalar quantization + correction overlay)... Patent angle for Track A v3 supplement: layer-wise streaming compression with inter-layer hidden-state distillation." Full mechanism: `scalar quantization 5bpw + B=64`, `correction overlay r=32`, `SVD warm-start`, `KL distillation pass`, calibration file paths (`fineweb_edu_500M_tokens.pt`), per-arch model registry. **Discloses the exact patent claim moat.** | MOVE_TO_PRIVATE (single biggest single-file leak) |
| `scripts/overlay/streaming_compression_online_runner.py` | CRITICAL | Same Track A nomenclature, online-distillation cure rationale, "Cure for the offline distillation distribution-shift regression" (telegraphs research direction) | MOVE_TO_PRIVATE |
| `uc verify` | HIGH | Imports `production-trainer` symbols by name (`CorrectionMatrixC`, `MODEL_REGISTRY`); inherits all Track A naming transitively | MOVE_TO_PRIVATE (or strip the import to a public stub) |
| `scripts/overlay/quantizers/trellis.py` | HIGH | "Track A" framing, "correction overlay low-rank correction overlay in `v18c_correction.py` learns to absorb (W - Wq)", QTIP integration mechanism | MOVE_TO_PRIVATE |
| `scripts/overlay/test_trellis_smoke.py` | MED | "Cure-A4" internal name leak; cross-references the scalar quantization + per-block(64) substrate | MOVE_TO_PRIVATE |
| `scripts/overlay/artifacts/streaming_compression_8b_500steps_eval.json` | MED | Local Windows path leaks `scamd` username (`C:\\Users\\scamd\\...`); seed 42 | REDACT (replace `compressed_dir` with a relative path; drop seed) |
| `scripts/overlay/artifacts/streaming_compression_32b_500steps_eval.json` | MED | Same `scamd` path leak; seed 42 | REDACT (same as above) |
| `scripts/overlay/artifacts/streaming_compression_8b_smoke.json` | MED | Full recipe block: `bpw 5, block_size 64, rank 32, train_steps 200, train_lr 0.001, train_bs 8, n_calib 100, seq_len 128, seed 42` | REDACT (drop the recipe block; keep result + provenance) |
| `scripts/overlay/artifacts/streaming_compression_14b_smoke.json` | MED | Same recipe block | REDACT |
| `scripts/overlay/artifacts/streaming_compression_32b_smoke.json` | MED | Same recipe block | REDACT |
| `scripts/overlay/artifacts/streaming_compression_72b_smoke.json` | MED | Same recipe block | REDACT |
| `scripts/overlay/artifacts/streaming_compression_online_8b.json` | MED | Same recipe block + per-layer training loss curve (reveals difficulty profile) | REDACT |
| `scripts/overlay/artifacts/streaming_compression_online_14b.json` | MED | Same | REDACT |

### Documentation leaks (CRITICAL — these are the most-read files)

| File | Severity | What leaks | Recommended action |
|---|---|---|---|
| `README.md` (root) | CRITICAL | "5-bit scalar quantization codes per weight + per-block(64) absmax scale + per-Linear correction overlay low-rank correction (low-rank (production-tuned), alpha scalar)" — full mechanism in plain English on the project landing page. References private `HONEST_NEGATIVE_RESULTS_2026_05_08.md` (broken link, but the section title alone signals lab-notebook-style content). | REDACT (replace mechanism block with results-only language; remove `## Honest negative results` section per Charter §1 explicit rule) |
| `CHANGELOG.md` (root) | CRITICAL | 0.4.0 entry: "scalar quantization scalar quantization (5 bpw, B=64), fits correction overlay low-rank correction (r=32) via KL distillation pass." Lists internal script names, online/offline runner taxonomy, trellis vs scalar comparison numbers (T1 80.80% vs 74.29%). 0.1.1 entry: "Track A (USPTO 64/049,511, shipping now) and Track B (USPTO 64/049,517, v0.2 Q3 2026) explicitly separated under their own headings." | REDACT (replace per-version recipe with results-only entries; map the structural changes to user-facing surface, not internal mechanism) |
| `docs/QUICKSTART.md` (root-level docs) | CRITICAL | "The streaming compression recipe: scalar quantization scalar 5 bpw + per-block (B=64) absmax normalization + correction overlay low-rank low-rank correction overlay + KL distillation pass per layer." Plus an exact reproduction CLI: `--bpw 5 --block_size 64 --rank 32 --train_steps 200 --n_calib 100 --n_eval 50`. | REDACT (drop the recipe paragraph; drop the reproduction CLI line; keep the install + pull + smoke-eval flow) |
| `docs/PILOT_PACKET.md` | HIGH | "Track A — post-training row-overlay quantization (USPTO 64/049,511) — shipping now." "Track B — Fractal Residual Recursion (USPTO 64/049,517) — v0.2 (Q3 2026)." Internal track-name decomposition wired to patent claim moats. | REDACT (drop "Track A / Track B" labels; keep USPTO numbers + capability framing) |
| `docs/concepts/compression-methods.md` | HIGH | "Track A — sub-3-bpw weight representation (v0.1, shipping)" / "Track B — architectural compression (v0.2, Q3 2026)" / "Fractal Residual Recursion" by name | REDACT (drop track names; keep capability framing) |
| `docs/concepts/reproducibility.md` | MED | "How Track A breaks the 4-bit cliff" + `seed=42` made an explicit headline reproducibility commitment | REDACT (drop the Track A reference from the NDA table; the seed=42 commitment is a public-good asset and can stay) |
| `docs/evidence/matrix.md` + `matrix.json` | HIGH | Per-model PPL retention table for **Track B / FRR (Fractal Residual Recursion)** with explicit operating point `2.40 bpw` + the `track: "B"` JSON field. The data itself is fine to publish (results-only), but the `track` field + "FRR" label are charter violations. | REDACT (replace `track: "B"` with `experiment_family: "architectural_compression"`; drop "Fractal Residual Recursion" label; keep the per-model results and methodology framing) |
| `docs/concepts/catastrophic-failures.md` | LOW | Per-method comparison table (HQQ at 2-bit / 3-bit / etc. + UltraCompress at 2.798 bpw). This is results-only and exactly the kind of public artifact the Charter §5 endorses — but it commits to a specific operating point (2.798 bpw) that the rest of the repo no longer matches (production is 5 bpw streaming). | KEEP_PUBLIC (results-only; no recipe). Optional: add a one-line note that 2.798 bpw is the v0.1 reference point and the production-tier artifacts are at 5 bpw streaming. |
| `docs/reference/uc-compress-spec.md` | CRITICAL | `--bpw 2.798` default exposed as a hardcoded operating point. CLI flags `--track a/b/a+b`. "track-a-row-overlay" as a method name string. "26.7× end-to-end with 68% top-10 retention" headline number outside any cohort framing. Recipe-level performance numbers + flag-level disclosure of the patent claim moat. | REDACT (drop `--track` flag; rename `--method` enum values to remove "track-a-row-overlay" / "track-b-frr"; replace headline number with "see published benchmarks page") |
| `docs/reference/uc-serve-spec.md` | LOW | OpenAI-compatible server spec; no recipe leaks | KEEP_PUBLIC |
| `docs/reference/manifest-schema.md` | CRITICAL | Officially documents the manifest enum: `track-a-row-overlay`, `track-b-frr`, `track-a+b` — these are now load-bearing wire-protocol constants. Every published artifact on HF that conforms to this schema inherits the leak. | REDACT (rename enum values: `track-a-row-overlay` → `row-overlay-quantization-v1`, `track-b-frr` → `architectural-compression-v1`, `track-a+b` → `combined-v1`. Coordinate with already-published HF artifacts; document a v1.0 → v1.1 schema deprecation path.) |
| `docs/reference/env-vars.md` | LOW | `UC_BENCH_SEED=42` documented; consistent with the seed=42 reproducibility commitment | KEEP_PUBLIC |
| `docs/release_notes/v0.1.3.md` | CLEAN | OSS compliance hardening notes; SBOM + third-party licenses | KEEP_PUBLIC |
| `docs/getting-started/quickstart.md` | LOW | Mentions "row-overlay-quantization (Track A) v1" in a sample `uc info` output | REDACT (drop "Track A" from the sample) |
| `docs/getting-started/install.md` | (not read) | Likely CLEAN — install instructions only | KEEP_PUBLIC |
| `docs/getting-started/configuration.md` | CLEAN | Standard config doc; documents `HF_TOKEN` flow correctly | KEEP_PUBLIC |
| `docs/commands/bench.md` | LOW | Documents `cuda:0` device flag + sample output. CLEAN by recipe standards. | KEEP_PUBLIC |
| `docs/commands/info.md` | LOW | Sample output shows `Method: row-overlay-quantization (Track A) v1` | REDACT (drop "Track A" from the sample) |
| `docs/commands/list.md` / `pull.md` / `demo.md` / `version.md` | (not read) | Likely CLEAN — CLI command docs | KEEP_PUBLIC |
| `docs/concepts/bits-per-weight.md` | CLEAN | Pure educational explainer of the bpw metric; no recipe | KEEP_PUBLIC |
| `docs/contributing/index.md` / `code-of-conduct.md` / `release-process.md` / `security.md` | CLEAN | Standard governance docs | KEEP_PUBLIC |
| `docs/integration/llamacpp.md` / `vllm.md` / `transformers.md` / `tensorrt-llm.md` | CLEAN | Integration walkthroughs; no recipe | KEEP_PUBLIC |
| `docs/changelog.md` (in `docs/`) | CLEAN | Stale stub of v0.1.0 only; not the leaky one (the leaky one is `CHANGELOG.md` at root) | KEEP_PUBLIC (or DELETE as redundant with root CHANGELOG once that one is redacted) |
| `docs/index.md` (mkdocs landing) | CLEAN | Marketing copy; no recipe | KEEP_PUBLIC |

### Brand / governance / metadata (CLEAN by recipe standards; HIGH severity for one Track-name leak)

| File | Severity | What leaks | Recommended action |
|---|---|---|---|
| `LICENSE` (Apache 2.0) | CLEAN | Standard text | KEEP_PUBLIC |
| `SECURITY.md` | CLEAN | Standard disclosure path; correct `security@sipsalabs.com` routing | KEEP_PUBLIC |
| `CODE_OF_CONDUCT.md` | (assumed CLEAN — standard) | n/a | KEEP_PUBLIC |
| `CONTRIBUTING.md` | CLEAN | Includes a "no proprietary compression internals" PR-checklist line; good signal | KEEP_PUBLIC |
| `CITATION.cff` | HIGH | Abstract: "post-training row-overlay quantization (Track A) and Fractal Residual Recursion (Track B)" — same Track-name + FRR leak as the docs | REDACT (drop Track A/B + FRR labels; cite USPTO numbers only) |
| `CHANGELOG.md` | (covered above — CRITICAL) | n/a | REDACT |
| `SBOM-cyclonedx.json` | CLEAN | Auto-generated runtime dependency graph | KEEP_PUBLIC |
| `THIRD_PARTY_LICENSES.txt` | CLEAN | License attribution | KEEP_PUBLIC |
| `pyproject.toml` | CLEAN | `Sipsa Labs <founder@sipsalabs.com>` author; no personal email or legacy GitHub org leakage | KEEP_PUBLIC |
| `mkdocs.yml` | (assumed CLEAN — site config) | n/a | KEEP_PUBLIC |
| `.gitignore` | CLEAN | Correctly ignores `*.safetensors`, `*.pt`, `*.bin`, model dirs | KEEP_PUBLIC |
| `.pre-commit-config.yaml` | (assumed CLEAN — hook config) | n/a | KEEP_PUBLIC |
| `.github/dependabot.yml` | (assumed CLEAN) | n/a | KEEP_PUBLIC |
| `.github/PULL_REQUEST_TEMPLATE.md` | (assumed CLEAN) | n/a | KEEP_PUBLIC |
| `.github/ISSUE_TEMPLATE/{bug_report,feature_request,config}.md/yml` | (assumed CLEAN) | n/a | KEEP_PUBLIC |
| `.github/workflows/ci.yml` | CLEAN | Uses PyPI Trusted Publishing (no secrets); standard test matrix | KEEP_PUBLIC |
| `.github/workflows/release.yml` | CLEAN | GitHub release from CHANGELOG; no secrets | KEEP_PUBLIC |
| `.github/workflows/security.yml` | CLEAN | Bandit + pip-audit + safety + gitleaks; uses `secrets.GITHUB_TOKEN` (the legitimate built-in) | KEEP_PUBLIC |

### CLI source code (CLEAN — public API surface that should be public)

| File | Severity | What leaks | Recommended action |
|---|---|---|---|
| `src/ultracompress_cli/__init__.py` | CLEAN | Version + HF_ORG / HF_COLLECTION_TAG constants | KEEP_PUBLIC |
| `src/ultracompress_cli/__main__.py` | CLEAN | Click CLI entry point | KEEP_PUBLIC |
| `src/ultracompress_cli/benchmark.py` | CLEAN | `lm-eval-harness` shell-out wrapper | KEEP_PUBLIC |
| `src/ultracompress_cli/demo.py` | CLEAN | Scripted demo with explicit "DEMO MODE — illustrative data" labeling | KEEP_PUBLIC |
| `src/ultracompress_cli/info.py` | CLEAN | Manifest reader; reads optional `track` field but doesn't write it | KEEP_PUBLIC |
| `src/ultracompress_cli/listing.py` | CLEAN | HF Hub listing wrapper | KEEP_PUBLIC |
| `src/ultracompress_cli/pull.py` | CLEAN | `snapshot_download` wrapper | KEEP_PUBLIC |
| `tests/test_*.py` (5 files) | (assumed CLEAN — tests of public API) | n/a | KEEP_PUBLIC |
| `tools/savings_calculator.py` | CLEAN | Customer ROI calculator; pricing assumptions are public estimates | KEEP_PUBLIC |
| `render_demo_video.py` | CLEAN | Terminal screencast renderer for the YC demo video | KEEP_PUBLIC |

---

## What we got right (preserve these — they are good public artifacts)

These are the files that exemplify the Charter — public assets that build trust without giving away the recipe. Keep this pattern when adding new content.

1. **`SECURITY.md`** — clean disclosure path with the right `security@sipsalabs.com` routing. Procurement-grade.
2. **`THIRD_PARTY_LICENSES.txt` + `SBOM-cyclonedx.json`** — EU CRA / EO 14028-aligned compliance hardening that costs nothing to publish.
3. **`docs/concepts/catastrophic-failures.md`** — results-only, cohort-level, methodology-clear. The kind of "what we got" doc that builds trust without leaking how.
4. **`docs/concepts/bits-per-weight.md`** — pure educational explainer; no recipe. Good public-good content.
5. **`docs/concepts/reproducibility.md`** (mostly) — the SHA-256 manifest commitment + the seed=42 honest-signal language are exactly the trust-asset framing the Charter §5 calls out.
6. **`src/ultracompress_cli/*`** — minimal Apache-2.0 CLI surface with no internals leaked. The right scope for a public package.
7. **`tools/savings_calculator.py`** — customer-facing ROI calc; quoteable at storage + egress, honest about v0.1 vs v0.2 inference-memory caveats.
8. **`.github/workflows/ci.yml`** — Trusted Publishing pattern (no PyPI tokens in secrets), correct tag-prefix guards. Production-grade release pipeline.
9. **`docs/integration/{llamacpp,vllm,tensorrt-llm,transformers}.md`** — customer-helping integration walkthroughs that don't leak the codec.
10. **`render_demo_video.py` + `src/ultracompress_cli/demo.py`** — explicitly labeled DEMO data, clean separation from live `uc list` flow. Good honest-signal posture.
11. **`pyproject.toml`** — clean author + URL metadata; no personal email, no legacy GitHub org leakage. Already-fixed in v0.1.2.
12. **CI / governance / contributing / code-of-conduct stack** — looks like a real OSS project, which is the entire point of the public surface per Charter §1.
13. **`render_demo_video.py`** — useful artifact for the YC video pipeline; no leaks.

---

## Diff vs the v1 audit (which findings are real, which were false alarms)

The v1 audit was scoped to local `master` (= `mounnar/ultracompress`, private) and flagged ~80 files. The **vast majority do not exist on `sipsalabs/main`** and were false alarms for the public-surface question.

### v1 findings that ALSO exist on `sipsalabs/main` (real concerns, confirmed):

| v1 finding | Status on `sipsalabs/main` | Notes |
|---|---|---|
| (production trainer, patent-protected) | **PRESENT** — same file, same content, same critical leak | This was the worst v1 finding and remains the worst v2 finding. |
| `scripts/overlay/streaming_compression_online_runner.py` | **PRESENT** — same critical leak | Carries over. |
| `uc verify` | **PRESENT** — same transitive Track A leak | Carries over. |
| `scripts/overlay/quantizers/trellis.py` | **PRESENT** — Track A + correction overlay framing | Carries over. |
| `README.md` recipe block (low-rank (production-tuned), train_steps=200, B=64) | **PRESENT** — different specific leak phrasing but same recipe surface | Carries over. |
| `docs/PILOT_PACKET.md` Track A/B labels | **PRESENT** | Carries over (in v1 this was less prominent because Track A/B labels were ALSO leaking via 30+ private docs; on the public branch this is one of the few sources, so it's relatively higher-leverage). |
| Per-arch result JSONs with `seed: 42, rank: 32, block_size: 64, train_steps: 200, n_calib: 100` | **PRESENT** — 8 artifact JSONs under `scripts/overlay/artifacts/` | Carries over. New finding: 2 of these also leak the local Windows path with `scamd` username. |
| Manifest enum `track-a-row-overlay` / `track-b-frr` | **PRESENT** in `docs/reference/manifest-schema.md` AND wired through `uc compress` spec + `uc info` sample output | v1 only flagged this in passing (it's `LOW` severity in v1's 1H section because the v1 narrative was dominated by lab-notebook leaks). On the actual public branch, with everything else stripped, the manifest schema is **the** structural enabler of the Track-name leak. **Re-rated CRITICAL in v2.** |
| `CITATION.cff` Track A/B + FRR labels | **PRESENT** | Carries over (HIGH on v2). |
| `docs/evidence/matrix.{md,json}` Track B / FRR labeling | **PRESENT** | Carries over (HIGH on v2). |

### v1 findings that are FALSE ALARMS for the public surface (i.e., NOT on `sipsalabs/main`):

The following CRITICAL/HIGH v1 findings do **not** exist on `sipsalabs/main`, so they pose **zero** public-surface risk and require no action against the public remote (they may still need to be cleaned out of the private master before any cross-merge):

- All lab notebooks: `LAB-NOTEBOOK.md`, `HONEST_NEGATIVE_RESULTS_2026_05_08.md`, `RESEARCH_v3_CURE_DIRECTION_2026_05_09.md`, `V4D_MULTIPASS_RESEARCH_LOG_2026_05_09.md`, `V4_CURE_RESEARCH_2026_05_09.md`, `RESEARCH_PROPOSAL_PER_LINEAR_ADAPTIVE_BPW_2026_05_08.md`
- All patent drafts: `PATENT_CIP_DRAFT_PER_LINEAR_ADAPTIVE_BPW_2026_05_08.md`, `PATENT_FILING_PACKET_2026_05_09.md`, `PATENT_DRAFT.md`, `PATENT_PROVISIONAL_SKELETON.md`, `PATENT_CLAIMS_SUMMARY.md`, `PATENT_CLAIMS.md`
- All paper drafts: `NEURIPS_2026_PAPER_v2_2026_05_09.md`, `ICLR_2027_PAPER_v3_2026_05_09.md`, `NEURIPS_2026_PAPER_DRAFT_2026_05_08.md`, `PAPER_DRAFT.md`
- All business / personal docs: `BUSINESS_PLAN.md`, `BILLION_DOLLAR_PATH_*.md`, `BILLION_DOLLAR_WEEKEND_FOCUS_2026_05_09.md`, `YC_APPLICATION*.md`, `YC_INTERVIEW_PREP*.md`, `YC_MAY_UPDATE_v6_2026_05_09.md`, `SERIES_A_PITCH_DECK_2026_05_08.md`, `INVESTOR_UPDATE_TEMPLATE_2026_05_08.md`, `PITCH.md`
- All outreach packs: `RESEARCH_PHASE0_ICP_RANKING_2026_05_09.md`, `ICP_EXPANSION_40_2026_05_09.md`, `OUTREACH_2026_05_08/*`, `OUTREACH_PHASE0_POC_2026_05_09/*`, `NASA_SBIR_PHASE1_PROPOSAL_DRAFT_2026_05_08.md`, `PRESS_RELEASE_DISTRIBUTION_LIST_2026_05_09.md`
- All checklists / morning-reports / commit-knowledge-map / IP-setup playbooks
- All FRR / Track B research scripts under `scripts/frr/`, `experiments/training/`, `experiments/sweeps/`
- All `claim20_*` / `claim21_*` / `wave4{5,6,7,8}_*` patent-claim experiment scripts and result JSONs
- All `ultracompress/` package modules with Athena-AGI naming (`moonshot.py`, `genome_*.py`, `compress_mind.py`, `living_frr.py`, `organism.py`, `crystal_net.py`, `hyperbolic.py`, `holographic_boundary.py`, `dendritic.py`, `thalamic.py`, `hypercomplex.py`, etc.)
- All `archive/`, `logs/`, `tools/ptq_tinyfrr.py`, `tools/factor_lmhead.py`, etc.
- All Show HN / launch / blog drafts
- `BUSINESS_PLAN.md`'s "$0 in funding; built on 2x RTX 5090; 22 years old; $500K seed" personal-info leak
- `YC_APPLICATION.md`'s "Founder: Sip, 22, Solo Technical Founder" personal-info leak
- `PATENT_CIP_DRAFT_*.md`'s "Inventor: Missipssa Ounnar" legal-name leak
- `PATENT_PROVISIONAL_SKELETON.md`'s "$206K/yr gross income" micro-entity disclosure

The public branch was apparently set up clean (or an earlier scrub already happened) — none of those classes of file are on `sipsalabs/main`.

### v1 findings that are PARTIAL — present on the public branch in a different form:

| v1 finding | v2 status | Notes |
|---|---|---|
| `docs/CUSTOMER_ONBOARDING_v0.5.3.md` (private internal config) | **NOT on public** — but `docs/PILOT_PACKET.md` covers similar ground in a public-clean rewrite | The PILOT_PACKET version is fine except for Track A/B labels. |
| `_packed_hermes_3_405b_v3_README_TEMPLATE.md` HF model card template | **NOT on public** — but the manifest schema + `uc info` output do leak the same `track-a-row-overlay` etc. via a different path | The public branch substitutes a wire-protocol leak for a documentation leak. Same severity, different mechanism. |

---

## Critical sanity checks (Charter §5 cross-cuts)

- **README first page exposing recipe:** YES (low-rank (production-tuned), B=64, scalar quantization + correction overlay framing). Fix in same PR as the rest.
- **Committed credentials / tokens / .env / test API keys:** NONE found. CI uses Trusted Publishing + `secrets.GITHUB_TOKEN` only. `.gitignore` correctly excludes `.env`, `.venv`, `*.safetensors`, `*.pt`, `*.bin`. CLEAN.
- **HF model cards on public branch:** the per-model HF cards live on HuggingFace, not in this repo. The template / sample output in `docs/commands/info.md` and `docs/getting-started/quickstart.md` does leak `Method: row-overlay-quantization (Track A) v1` — minor REDACT.
- **CHANGELOG / RELEASE_NOTES referencing internal tracks:** YES — `CHANGELOG.md` 0.1.1 entry explicitly separates "Track A (USPTO 64/049,511)" and "Track B (USPTO 64/049,517)". `docs/release_notes/v0.1.3.md` is CLEAN. `docs/changelog.md` is a CLEAN stale stub.
- **Personal info on public surface:** the only personal-info leak is the `scamd` Windows username embedded in 2 artifact JSONs' `compressed_dir` field. Fix via REDACT. Legal name, age, income, day-job, cash position, micro-entity status — none of these are on `sipsalabs/main`. Major win vs the v1 master state.
- **Outreach packs naming prospects:** none on `sipsalabs/main`. CLEAN.
- **Conference paper drafts with full method recipes:** none on `sipsalabs/main`. CLEAN.
- **Lab notebook / negative results:** none on `sipsalabs/main`. The README links to a `HONEST_NEGATIVE_RESULTS_2026_05_08.md` that is NOT on the branch — that is a broken link, not a leak. Fix by removing the section reference from README.

---

## Recommended single PR (read-only audit; pure recommendation, not executed)

1. **Strip `scripts/overlay/` from `sipsalabs/main` entirely.** All 14 files in that subtree are recipe-leaking and the public CLI does not depend on any of them. Move to a private repo (`sipsalabs/ultracompress-internal` per Charter §4).
2. **Redact `README.md`** — replace the `## What's lossless` recipe paragraph with results-only language; remove the `## Honest negative results` section reference (broken link + section title implies private content).
3. **Redact `CHANGELOG.md`** — for 0.4.0, replace mechanism description with a user-facing "production tier streaming compression — see docs"; for 0.1.1, drop "Track A / Track B" labels and reference USPTO numbers only.
4. **Redact `docs/QUICKSTART.md`** — drop the "streaming compression recipe" paragraph and the `production-trainer.py --bpw 5 --block_size 64 --rank 32 --train_steps 200 --n_calib 100` reproduction CLI line (which is dead anyway after step 1).
5. **Rename manifest enum** in `docs/reference/manifest-schema.md`, `docs/reference/uc-compress-spec.md`, `docs/commands/info.md`, `docs/getting-started/quickstart.md`: `track-a-row-overlay` → `row-overlay-quantization-v1`; `track-b-frr` → `architectural-compression-v1`; `track-a+b` → `combined-v1`. Coordinate with already-published HF model cards (manifest schema bump v1.0 → v1.1 with backward-compat reader path in `src/ultracompress_cli/info.py`).
6. **Drop Track A/B labels** in `CITATION.cff`, `docs/PILOT_PACKET.md`, `docs/concepts/compression-methods.md`, `docs/concepts/reproducibility.md` (one-line fix in each), `docs/evidence/matrix.{md,json}` (rename `track: "B"` → `experiment_family: "architectural_compression"`).
7. **Drop FRR / "Fractal Residual Recursion"** from `CITATION.cff`, `docs/PILOT_PACKET.md`, `docs/concepts/compression-methods.md`, `docs/evidence/matrix.{md,json}` — replace with "architectural compression".

After this PR, the public surface should be Charter-compliant. Cleanup scope: **3 files MOVE_TO_PRIVATE (scripts/overlay/ subtree counts as one move)**, **~12 files REDACT in place**, **0 files DELETE**, **the rest (≈65 files) KEEP_PUBLIC**.

Codec internals + training procedure are patent-protected (USPTO 64/049,511 + 64/049,517).
