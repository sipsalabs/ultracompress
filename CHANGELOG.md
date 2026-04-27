# Changelog

All notable changes to the UltraCompress CLI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- (placeholder for next release)

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

[Unreleased]: https://github.com/sipsalabs/ultracompress/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/sipsalabs/ultracompress/releases/tag/v0.1.1
[0.1.0]: https://github.com/sipsalabs/ultracompress/releases/tag/v0.1.0
