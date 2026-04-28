# Changelog

All notable changes to the UltraCompress CLI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- (placeholder for next release)

## [0.1.2] — 2026-04-27

Foundation pass — fixes a release-pipeline bug, scrubs personal-info leaks
in the published metadata, and tightens the demo so it cannot be misread
as showing live data.

### Fixed
- **Critical**: `pyproject.toml` author and project URL fields were preserving
  pre-rebrand values when v0.1.0 was packaged, so the published metadata on
  PyPI was leaking a personal email and the old private GitHub org name.
  v0.1.2 ships with `Sipsa Labs <founder@sipsalabs.com>` and
  `github.com/sipsalabs/ultracompress` URLs throughout. Operators with
  v0.1.0 installed should `pip install --upgrade ultracompress` once
  v0.1.2 lands.
- CI workflow `ci.yml` was not firing on tag pushes, so the `publish` job
  (which has the right tag-prefix guard internally) never ran when v0.1.1
  was tagged. Added `tags: ["v*.*.*"]` to the `push` trigger so future
  tag pushes flow through the test gate and into PyPI publishing.
- `__init__.py` `__version__` was out of sync with `pyproject.toml`
  (0.1.0 vs 0.1.1). Both now read 0.1.2.

### Changed
- `uc demo` no longer renders mock download counts that could be misread
  as real popularity numbers. The demo header explicitly shows
  "DEMO MODE — illustrative data" and the catalog table includes a
  caption pointing customers to `uc list` for the live Hub state.
- `uc demo` install scene shows the correct `0.1.2` version string.

### Notes
- No CLI runtime changes beyond demo cosmetics.
- v0.1.0 should be yanked from PyPI by the project owner after v0.1.2
  publishes successfully — that prevents new installs from receiving
  the leaky metadata while preserving the version for users who pinned
  it explicitly.

## [0.1.1] — 2026-04-27

Customer-facing artifacts and trust signals. No CLI runtime changes.

### Added
- `docs/evidence/matrix.md` + `docs/evidence/matrix.json` — six-model cohort evidence with provenance, full retention curves (T1, T10, agreement + retention, perplexity, compression ratio per model). Public-safe: method internals deliberately excluded.
- `docs/PILOT_PACKET.md` — design-partner pilot packet covering Tier 1 ($5K compression assessment, 2-week turnaround) and Tier 2 ($15K-$25K production deployment pilot, 60-day window). Convertible to per-deployment / multi-deployment / OEM-royalty licenses.
- `tools/savings_calculator.py` — customer economic calculator. Input: fleet size + per-model param count. Output: storage / egress / GPU-memory savings vs FP16 / int8 / NF4 / HQQ baselines. JSON output mode for sales-sheet integration.
- `CITATION.cff` — academic citation format (CFF v1.2.0). GitHub auto-detects and renders a "Cite this repository" button.
- README — claim discipline pass: Track A (USPTO 64/049,511, shipping now) and Track B (USPTO 64/049,517, v0.2 Q3 2026) explicitly separated under their own headings; "Who this is for" buyer-archetype list added; 4-bit-per-weight cliff narrative as the customer-pain hook; top-k retention curves (T1–T256) referenced.

### Fixed
- README typo: `uc eval` → `uc bench` (now matches the actual CLI command).
- README: `uc demo` added to the "What's available today" command list.

### Notes
- No runtime / API / dependency changes. v0.1.1 is documentation, evidence, and customer-facing artifacts only.
- Patent prosecution timing for `uc compress` (Track A self-compression) and Track B variants remains v0.2 (Q3 2026).

## [0.1.0] — 2026-04-25

Initial public alpha release. Patent-pending compression methods (USPTO 64/049,511 + 64/049,517) underpinning the pre-compressed reference models on the Hugging Face Hub.

### Added
- `uc list` — list pre-compressed models from the official Hugging Face Hub collection
- `uc pull <model-id>` — download a pre-compressed model artifact
- `uc info <path>` — inspect compression metadata of a local artifact
- `uc bench <path> --tasks <list>` — run downstream benchmarks via lm-eval-harness
- `uc demo` — play a scripted demo session (no Hub access required)
- `uc version` / `uc --version` / `-V` — print version
- Apache-2.0 license for the CLI source code
- GitHub Actions CI for Python 3.10–3.12 (lint, type-check, test, build, security scans)
- PyPI Trusted Publishing on tag push

### Notes
- Self-compression (`uc compress`) is intentionally not yet shipped — it is gated on the patent-pending compression methods being formally protected.
- Pre-compressed model artifacts are licensed separately from the CLI itself.

[Unreleased]: https://github.com/sipsalabs/ultracompress/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/sipsalabs/ultracompress/releases/tag/v0.1.2
[0.1.1]: https://github.com/sipsalabs/ultracompress/releases/tag/v0.1.1
[0.1.0]: https://github.com/sipsalabs/ultracompress/releases/tag/v0.1.0
