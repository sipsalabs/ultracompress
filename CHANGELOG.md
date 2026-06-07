# Changelog

All notable changes to UltraCompress are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), versioning per [SemVer](https://semver.org/).

> **Note (2026-05-25):** Historical release notes pre-0.6.7 have been condensed during a documentation refresh. Detailed per-release internal notes (including method-internal specifics) are available to partners under NDA — contact legal@sipsalabs.com.

---

## [0.6.17] — 2026-05-24

### Changed

- **`uc try` conversion footers** rewritten as a numbered "Next steps" list (free API key → `uc catalog` → Phase 0 POC). Live-mode footer adds an explicit "Want this on YOUR model in production?" CTA block linking to <https://sipsalabs.com/poc>. Same payload, sharper conversion bridge at the moment a user just saw a real API response.
- **`uc try` signup URL fixed.** Changed from `https://sipsalabs.com/access` (which 404'd) to `https://sipsalabs.com/get-access`. A backwards-compat redirect for the old URL is also live on the site for users on 0.6.16.

### Security / defense-in-depth

- **`uc serve` reconstruction module access path hardened.** The private reconstruction module is now loaded indirectly via `importlib`-based resolution so that no literal private-module import statement remains in the shipped public source. `ImportError` is caught and the user-facing message becomes: `Error: serve mode requires Sipsa enterprise reconstruction. Contact founder@sipsalabs.com for the licensed reconstruction module.` The runtime call is also wrapped in `try/except` so unexpected failures print a clean error instead of a stack trace exposing internal symbol names.
- **`uc verify` and `uc serve` no longer surface internal pack-version field** in customer-facing output. The verifier still consumes `bpw` and `n_layers` from the manifest and reports pack structure and SHA-256 fingerprint as before.

### Notes

- No payload format change. The pack format itself (`.uc` artifacts on Hugging Face) is unchanged; existing packs verify and serve identically.
- No new dependencies. The shipped wheel size is unchanged within rounding.

---

## [0.6.16] — 2026-05-23

### Added

- **vLLM Phase 2 plugin** — decode throughput parity within 0.4% of the bf16 baseline on Qwen3-8B at batch=8 (726.1 tok/s UC vs 723.2 tok/s bf16). Runtime integration details are available to partners under NDA.

---

## [0.6.15] — 2026-05-19

### Security

- **Public-surface IP-leak audit pass.** Minimal-public-surface PyPI republish; all prior PyPI release artifacts removed (yanked + deleted) to consolidate the published wheel/sdist set on the latest clean build. No on-disk pack-format change; all published Hugging Face artifacts continue to verify clean under this release.

---

## [0.6.9] — 2026-05-15

### Security (RCE-class fix — please upgrade)

- **`torch.load(weights_only=False)` removed from every customer-facing call site.** `uc verify`, `uc.load`, the inference server `.uc` loader, and the v3 packer all use the safe loader (`weights_only=True`). A tampered `.uc` artifact that previously could have executed arbitrary code at deserialization time now raises `UnpicklingError` at load. No payload format change — the `.uc` schema is fully covered by the safe-loader allowlist; no `add_safe_globals()` registration is required.
- **Legacy `.pt` fallback gated behind explicit opt-in.** The pre-v3 fallback path now refuses to deserialize unless `--allow-unsafe-load` is passed (or `UC_ALLOW_UNSAFE_LOAD=1` is set in the environment), and emits a loud stderr warning when activated. Without the opt-in, the command exits cleanly with a remediation hint. This path is not exercised by `uc verify`, `uc.load`, `uc verify`, `uc catalog`, or `hf download`, so the typical customer workflow is unaffected.

### Tests

- New `tests/test_safe_load.py` regression suite covering the static call-site audit, a functional round-trip through the public API, and a negative test that a synthetic tampered `.uc` is rejected at load time.

### Backward compatibility

- All published `SipsaLabs/*` Hugging Face artifacts continue to verify clean under v0.6.9 — the on-disk `.uc` schema is byte-for-byte unchanged.
- The legacy bench fallback now requires `--allow-unsafe-load` on a fresh install; this is the only behavior change a customer could observe, and it does not affect any of the documented `uc` subcommands (`verify`, `load`, `bench-ppl`, `list`, `pull`).

---

## [0.6.7] — 2026-05-13

### Added

- Post-install onboarding banner. First interactive `import ultracompress` after install prints a one-time stderr banner pointing to <https://sipsalabs.com/u> for prioritized support and early v0.7 access. Marker file at `~/.ultracompress/welcomed` ensures it shows exactly once per machine. Suppressed in non-tty contexts (CI, log pipelines, redirected stderr) so it never pollutes automated environments. All exceptions caught silently — banner cannot break user code.

---

## Pre-0.6.7 releases (condensed)

In the project's first two-week ramp (late April through mid-May 2026), the project delivered:

- Initial public CLI surface (`pull`, `list`, `info`, `bench`, `bench-ppl`, `demo`, `pack`, `verify`, `load`, `version`).
- Initial Hugging Face publishing of compressed reference packs across dense, Mixture-of-Experts, and state-space architectures at the 5-bit-per-weight operating point with SHA-256 manifest verification.
- Self-contained pack format with embed/lm-head bundling so customers do not need to re-download the original bf16 from Hugging Face for reconstruction.
- License migration to BUSL-1.1 with Additional Use Grant (free for sub-$1M ARR companies, research, and individuals).
- Sipsa Labs, Inc. (Delaware C-corp) corporate banner across all public surfaces and `PATENT_NOTICE.md` at the repository root.
- Customer-grade inference throughput benchmark (`uc verify`) producing JSON reports suitable for procurement / acceptance testing.

Method-internal release notes live under NDA. Contact `legal@sipsalabs.com` for details.

---

## Versioning policy

- **Major** (X.y.z): breaking changes to the CLI surface OR the compressed artifact format.
- **Minor** (x.Y.z): new commands, new artifact format extensions, new model checkpoints (compatible with existing CLI).
- **Patch** (x.y.Z): bug fixes, security patches, documentation-only changes.

The closed-source production pipeline (commercial license) versions independently from the open-source CLI. Customer engagement contracts pin specific versions; updates require contract amendment.

---

## Contact

- Bugs: <https://github.com/sipsalabs/ultracompress/issues>
- Security: security@sipsalabs.com
- Commercial licensing: legal@sipsalabs.com
- General: hello@sipsalabs.com

Codec internals are patent-protected; USPTO provisional applications on file. Application identifiers available to partners under NDA.
