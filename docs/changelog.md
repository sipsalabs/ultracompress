# Changelog

All notable changes to the UltraCompress CLI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The canonical version of this file lives in the repository at [`CHANGELOG.md`](https://github.com/mounnar/ultracompress/blob/main/CHANGELOG.md).

## [Unreleased]

### Added

- (placeholder for next release)

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

- Self-compression (`uc compress`) is intentionally not yet shipped — it is gated on the patent-pending compression methods being formally protected. Target: v0.2 (Q3 2026). Design spec: [`uc compress` spec](reference/uc-compress-spec.md).
- Local inference server (`uc serve`) is on the v0.2 roadmap. Design spec: [`uc serve` spec](reference/uc-serve-spec.md).
- Pre-compressed model artifacts are licensed separately from the CLI itself. For licensing terms, contact `legal@sipsalabs.com`.

[Unreleased]: https://github.com/mounnar/ultracompress/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/mounnar/ultracompress/releases/tag/v0.1.0
