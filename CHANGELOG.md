# Changelog

All notable changes to UltraCompress are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), versioning per [SemVer](https://semver.org/).

---

## [0.6.11] - 2026-05-16

### Changed
- **Public-surface version alignment.** GitHub storefront README badge, `pip install` command, and prose bumped to v0.6.11 to match the current PyPI release (`pypi.org/project/ultracompress/`). Cosmetic consistency only -- no functional changes vs 0.6.10.

### Verified
- 22 architectures shipped end-to-end (compression complete + uploaded to HuggingFace); 14 PPL-verified end-to-end against their bf16 baseline, the remaining 8 pending eval. Hermes-3-Llama-3.1-405B at 1.0066x on a single 32 GB consumer GPU. Canonical ratios in the README architecture matrix.

## [0.6.10] - 2026-05-15

### Changed
- **Public-surface version alignment.** GitHub storefront README badge, `pip install` command, and prose bumped to v0.6.10 to match the current PyPI release (`pypi.org/project/ultracompress/`). Cosmetic consistency only -- no functional changes vs 0.6.9. No separate GitHub release or git tag was cut for 0.6.10; the latest tagged GitHub release remains v0.6.9 (see [releases/latest](https://github.com/sipsalabs/ultracompress/releases/latest)).

### Security
- Ships on the MANIFEST-scrubbed sdist introduced in 0.6.9. The 0.6.7 / 0.6.8 sdists were yanked from PyPI after the 0.6.9 RCE-class fix on `torch.load()` paths; 0.6.10 carries that fix forward.

### Verified
- 22 architectures shipped end-to-end (compression complete + uploaded to HuggingFace); 14 PPL-verified end-to-end against their bf16 baseline, the remaining 8 pending eval. Hermes-3-Llama-3.1-405B at 1.0066x on a single 32 GB consumer GPU. Canonical ratios in the README architecture matrix.

## [0.6.9] - 2026-05-15

### Security
- **RCE-class fix** on `torch.load()` reconstruction paths. Prior 0.6.7 / 0.6.8 sdists shipped without a scrubbed MANIFEST and are yanked from PyPI; 0.6.9 ships a MANIFEST-scrubbed sdist. Wheel installs were unaffected.

### Verified
- Mistral-7B-v0.3 at 1.00548x — the tightest dense 7B-class lossless 5-bit ratio we currently publish.

## [0.6.6] - 2026-05-12

### Changed
- **Public-surface version alignment.** GitHub release tag `v0.6.6` published to match PyPI (`pypi.org/project/ultracompress/0.6.6/`). Repo README badge, pip-install command, and prose all bumped to v0.6.6 across `sipsalabs/ultracompress` and 35 of 40 SipsaLabs HuggingFace model cards. Cosmetic consistency only -- no functional changes.

### Verified
- 22 architectures shipped end-to-end; 14 PPL-verified end-to-end (remaining 8 pending eval). Canonical PPL ratios in the README architecture matrix. Hermes-3-Llama-3.1-405B at 1.0066x on a single 32 GB consumer GPU.

## [0.6.5] - 2026-05-09

### Added
- **`uc pack` v0.2 format** -- self-contained pack files now persist all calibration norms and codec metadata inline. Removes the v0.1 dependency on a separate sidecar JSON for reconstruction; pack is the single source of truth.

### Changed
- `uc verify` now checks the inline metadata block before reconstruction, returning a precise mismatch diagnostic if a pack was produced under the v0.1 sidecar format.

## [0.6.4] - 2026-05-08

### Changed
- Maintenance release. Charter scrubbing across runtime CLI strings + internal symbol names. No PPL or behavior changes vs v0.6.3.

## [0.6.3] — 2026-05-12

### Changed
- **Charter-clean PyPI metadata.** v0.6.2 frozen package description on PyPI was scrubbed of internal codename references that had survived earlier sweeps. v0.6.3 ships an updated `pyproject.toml` description and keyword set that are verifiably leak-free.

### Added
- **PATENT_NOTICE.md** at the repo root, summarizing the U.S. provisional patent applications. Linked from sipsalabs.com homepage.

### Backward compatibility
- All 40 published `SipsaLabs/*` HuggingFace artifacts continue to verify clean under v0.6.3. Pin `ultracompress==0.6.2` if you need the prior package metadata exactly.

---

## [0.6.2] — 2026-05-11

### Added
- **Corp identity refresh.** Sipsa Labs, Inc. (Delaware C-corp, incorporated May 2026) banner across README, HuggingFace org card, and all 40 model cards.
- **OpenAI-compatible inference API** at https://api.sipsalabs.com/v1. Drop-in `OPENAI_BASE_URL` swap; the official `openai` Python SDK works unchanged. Backed by dual RTX 5090 over Cloudflare Tunnel.
- **Three new sub-1.005× perplexity ratio records this week:** Mixtral-8x7B 1.00368×, Qwen3-14B 1.00403×, Mistral-7B-v0.3 1.00548× (9.16× tighter than prior). Phi-3-mini-4k-instruct 1.00262× (seq_len=128 — not apples-to-apples vs the seq_len=1024 rows).
- **Hermes-3-Llama-3.1-405B 1.0066×** — a 405B-class lossless 5-bit transformer compressed end-to-end on a single 32 GB consumer GPU, verified end-to-end.
- **22 architectures shipped** spanning dense (0.6B–405B), Mixture-of-Experts (47B–235B active), and state-space (Mamba-2.8B); 14 PPL-verified end-to-end, the rest pending eval. Full matrix on the README and `huggingface.co/SipsaLabs`.

### Changed
- **Package source codename strip.** All internal method nomenclature replaced with neutral public phrasing in the published package source (preserving runtime behavior). CLI flags + class names follow.
- **Documentation hygiene.** Public-bound documents and historical HuggingFace model cards reviewed; legacy preview-format references in six older model cards were updated to the current public artifact naming.
- **`uc verify` output** prints `uc_pack_version: 3 (LOSSLESS, self-contained)` consistently. Stale `v3.0` warnings removed for v3.5 packs.

### Backward compatibility
- All 40 published SipsaLabs/* HuggingFace artifacts continue to verify clean under v0.6.2.
- Pin `ultracompress==0.5.5` if you need the prior package class names; v0.6+ is rename-only at the public API boundary.

---

## [0.6.1] — 2026-05-10

### Added
- **Self-contained pack format v3.5** stabilized. `aux_weights.uc` ships embed_tokens / model.norm / lm_head bundled inside the pack, so customers no longer need to download the original bf16 from HuggingFace to reproduce. ~1-2% pack overhead on small models, negligible on 70B+.
- **`uc pack --include-aux`** flag default-on. Customers can opt back into the legacy v3.0 path with `--legacy-v3`.

### Changed
- README v3 with the full 22-architecture matrix and a negative-results summary.
- Compatibility docs updated for SHA-256 manifest verification.

---

## [0.6.0] — 2026-05-10

### Added
- **License clarified.** The CLI source ships under Apache-2.0; the Apache-2.0 patent grant covers the as-published source code. The patent-pending compression methodology that produces the artifacts is out of scope of that grant — see PATENT_NOTICE.md. The `legacy/0.5.x` branch remains Apache-2.0 in perpetuity.
- Patent posture clarified in PATENT_NOTICE.md. U.S. provisional patent applications (filed April 2026), with supplements pending. Specific claim scope is available to commercial counterparties under NDA.

### Changed
- All public package class names refreshed for clarity (no behavior changes). See README for migration notes.
- HF model cards refreshed across the org with v3.5 self-contained format guidance.

---

## [0.5.5] — 2026-05-09

### Added
- **`uc pack v0.2` self-contained pack format** (`pack_format_version: 3.5`). Previously, customers had to download the original bf16 safetensors from HuggingFace to obtain `embed_tokens.weight`, `model.norm.weight`, and `lm_head.weight` for reconstruction — defeating most of the compression benefit on first download. The new format packs these model-level non-Linear tensors into a single `aux_weights.uc` file at the pack root.
  - `aux_weights.uc` format: `UCAX` magic + version + n_tensors + per-tensor blobs reusing the existing `_serialize_extra` framing (dtype-tagged, native bf16/fp16/fp32 preserved).
  - Pack overhead: ~1-2% of total pack size on small models (Qwen3-1.7B-Base: aux ~270 MB on a 1.1 GB pack), negligible on larger models.
  - Cryptographic provenance: the aux file gets its own SHA-256 in `manifest.json` (`aux_sha256`), joining the existing `uc verify` chain.
  - Weight-tied lm_head detection: when `lm_head.weight` is aliased to `embed_tokens.weight` (Qwen3-1.7B-Base, SmolLM2, Phi-3-Mini, Llama3 small), the loader stores only `embed_tokens` and re-ties at load time — no duplicate bytes on disk.
- **`uc pack-aux <packed_dir>` CLI** — retrofit any existing v3.0 pack with self-contained aux without re-packing the layer files. Idempotent (deterministic serialization).
- **`uc pack --include-aux/--no-aux/--base-model HF_ID`** flags on the main pack command. `--include-aux` is now the default.
- **`uc pack --legacy-v3`** flag to opt back into the original lossy v3 path (kept for back-compat).
- **`ultracompress.aux_pack`** module — public API: `serialize_aux_weights`, `parse_aux_weights`, `collect_aux_tensors_from_model`, `load_aux_into_model`.

### Changed
- `uc verify` now validates the aux file SHA-256, parses it, and confirms tensor key set matches the manifest. Output explicitly labels the pack as `SELF-CONTAINED` (v3.5) vs `requires base HF download` (v3.0).
- `uc bench` automatically detects `aux_weights.uc` and skips the base-safetensors HF download — model is built from `AutoConfig` + per-layer reconstruction + injected aux tensors.
- `uc pack` now defaults to the lossless v3 path (was previously the lossy v0.2 path); use `--legacy-v3` for the old behavior.

### Backward compatibility
- Old v3.0 packs (no `aux_file` field in manifest, no `aux_weights.uc` on disk) continue to load via the existing HF-safetensors fallback. No re-pack required.
- The 22 already-uploaded HF artifacts under `huggingface.co/SipsaLabs/*-uc-v3-bpw5` are NOT re-packed — they remain v3.0. Going forward, new uploads will ship as v3.5.

---

## [0.5.4] — 2026-05-09

### Added
- **`uc bench <packed_dir>` — sales-grade inference throughput benchmark** (`ultracompress/bench.py`). Customers can now run a one-line throughput benchmark on the v3 packed model we shipped them and produce a JSON report suitable for procurement / acceptance testing.
  - Measures TTFT (time-to-first-token, mean across `--n-prompts`), TPS overall, TPS decode-only, and peak VRAM.
  - `--baseline` flag adds an apples-to-apples bf16 comparison run on the same prompts and same device, with `tps_pct_change` / `vram_pct_change` deltas in the JSON.
  - Resolves the base HF model id automatically from the packed dir's `manifest.json` or `README.md` YAML frontmatter; `--base-model` overrides.
  - Greedy decoding only (deterministic) so the reported throughput reproduces across runs.
  - CUDA OOM is caught and surfaced as a `warnings` field in the JSON; partial results still get written.
  - Public Python API: `from ultracompress import bench_packed; result = bench_packed(packed_dir, ...)` returns a `BenchResult` dataclass.
- The legacy `uc bench --model <hf_id>` compression-vs-baseline benchmark is preserved as `uc bench-compress`.

### Changed
- `uc bench` is now a positional-arg command (`uc bench <packed_dir>`) instead of `--model <hf_id>`. The legacy form is available as `uc bench-compress`.

---

## [0.5.1] — 2026-05-08

### Fixed
- **`uc verify`, `uc pack`, and any `import ultracompress` was failing on a fresh `pip install ultracompress==0.5.0` install** with a `ModuleNotFoundError` for an internal research module that is not shipped in the public package. The 0.5.0 wheel bundled `ultracompress/api_v2.py`, which top-level-imports an internal research module that is not packaged. Customer-facing CLI commands (pack / load / verify) do not need v2 at all, but the eager import in `ultracompress/__init__.py` made the whole package un-importable when v2's dependencies were missing.
- 0.5.1 wraps the v2 + legacy-api imports in `try / except` so the customer-facing v3 API + CLI keep working when internal research modules are absent. The deprecation shim is only patched onto v2 if v2 actually loaded. `_API_V2_AVAILABLE` is exposed as a module-level flag so callers can branch on availability.

### Discovery
- Bug surfaced via the end-to-end customer reproduction test: `pip install --upgrade ultracompress` → `hf download SipsaLabs/qwen3-1.7b-uc-v3-bpw5` → `uc verify` (which `__init__.py` choked at import time before the verifier ran).

### Verified working in 0.5.1
- `uc verify <packed_dir>` passes on the public `SipsaLabs/qwen3-1.7b-uc-v3-bpw5` artifact: 28 layer.uc files present, sha256 spot-check, layer 0 reconstructs 7 quantized Linears + 4 extras with correct shapes.

---

## [0.5.0] — 2026-05-08

### Added
- **State-space-model (SSM) architectural compatibility validated** on Mamba-2.8B (`state-spaces/mamba-2.8b-hf`). 256 SSM Linear modules (`in_proj`, `x_proj`, `dt_proj`, `out_proj`) compress with bit-identical reconstruction; end-to-end PPL eval pending. The same pack path handles both transformer and state-space architectures, which should extend to emerging hybrids such as AI21 Jamba.
- **`uc pack v0.3` lossless binary format**. Reconstruction is a deterministic dequantization that is mathematically lossless — bit-identical reconstruction of the original weights. Internal format details are proprietary (patent-pending).
  - Validated end-to-end: source compressed PPL 18.3748 vs v3 reload PPL 18.3748 on Qwen3-1.7B (delta 0.000003%).
  - Bit-equal state-dict round-trip across 32 keys (max_abs_diff = 0.0).
  - File header bumped to `UC_VERSION = 3`.
- **Self-contained pack format** (proprietary; patent-pending): the pack carries the state needed for bit-identical customer-side reconstruction inline alongside the weights. Internal format details are proprietary.
- **8-architecture v3 pack matrix** uploaded to HuggingFace at `SipsaLabs/<model>-uc-v3-bpw5`:
  - Dense: Qwen3-1.7B, Mistral-7B-v0.3, Llama-3.1-8B, Qwen3-8B, Qwen3-14B, Llama-3.1-70B
  - MoE: Mixtral-8x7B-v0.1, Phi-3.5-MoE-instruct
  - Mean PPL ratio for 5 dense small models: **1.0077** (sub-1% perplexity degradation).
- **Vectorized bit-level (de)serialization** in the pack writer. **~1000× speedup** (16M-weight roundtrip in ~270ms vs prior Python loop in minutes).
- **Pack format extras section** for non-quantized layer tensors (norms, layer-level routers). Pack format now self-contained — customer doesn't need source layer files for reconstruction.
- **HF upload wrapper** for publishing v3 packs with an auto-generated README.md per repo.
- **Validation gate** (pack → reconstruct → eval PPL → compare).
- **Disk-cleanup guard** gated on `uc_pack_version >= 3` (anti-mistake guard).

### Documented
- `docs/OPERATOR_PLAYBOOK_2026_05_07.md` — cardinal rules + 2026-05 cleanup-mistake postmortem.
- `docs/AUTONOMOUS_MODE_SUMMARY_2026_05_07.md` — full session writeup of the v0.3 push.
- `docs/AFWERX_SBIR_PHASE1_PROPOSAL_DRAFT_2026_05_07.md` — submit-ready SBIR Phase I proposal.
- `docs/POST_EIN_DAY_0_CHECKLIST_2026_05_07.md` — 90-min sequence for the day Atlas EIN arrives.

### Fixed
- `uc pack` v0.2 was lossy (~22% PPL regression) because it attempted to derive internal codec state from dequantized weights under an incorrect assumption. v0.3 (this release) reads the self-contained pack directly and is lossless.
- HF upload wrapper had `subprocess.run(capture_output=True)` which deadlocked the `hf` CLI's progress display. Removed `capture_output` so stdout/stderr inherit from caller — uploads stream live and complete reliably.

### Compatibility
- `uc pack v0.3` files have `UC_VERSION = 3` in binary header. Old loaders (v1/v2) cannot read them. New loader supports v1 / v2 / v3 for backward compat.
- Customers should `pip install --upgrade ultracompress` to v0.5.0 to read v3 packs.
- Existing v0.4.x bf16 streaming-compressed checkpoints on HF continue to work unchanged.

---

## [0.4.1] — 2026-05-04 (planned, post-PRELAUNCH_BUGS_v0_4_0_FIXES)

### Fixed
- **`uc list` now returns 10 published SipsaLabs models.** v0.4.0 had `HF_ORG = "sipsalabs"` (lowercase) which mismatched the actual HuggingFace org name `SipsaLabs` (CamelCase). The HF API call returned no results. Fixed by correcting the case in `src/ultracompress_cli/__init__.py`.
- **CLI version banner now reports `0.4.1` correctly.** v0.4.0 had `__version__ = "0.1.3"` hardcoded in `src/ultracompress_cli/__init__.py`. Bumping the package version in pyproject.toml did not update the in-source string. Fixed.
- **`uc pull` no longer crashes on Windows after successful download.** Rich library emits Braille pattern characters (U+2800-U+28FF) in progress bars. Windows default cp1252 console encoding cannot render these, raising `UnicodeEncodeError` after the LFS download completes. Fixed by constructing Rich Console with `legacy_windows=False` and forcing UTF-8 stdio on Windows in the CLI entry point.
- **`uc pull` accepts `--output-dir` as alias for `--output`** to match the documentation in published HF model cards.

### Notes
- All v0.4.0 layer artifacts on HuggingFace continue to work with v0.4.1 unchanged. Only the CLI surface needed patching.
- Customers running `pip install --upgrade ultracompress` get v0.4.1 automatically.

---

## [0.4.0] — 2026-05-04

### Added
- **Streaming compression pipeline** ((production compressor, patent-pending)) that processes one transformer block at a time. Peak GPU memory bounded by ~one layer regardless of total model parameter count.
- **Four production-grade compressed checkpoints** on HuggingFace under `SipsaLabs/`:
  - `qwen3-8b-streaming-bpw5` — PPL ratio 1.028× fp16, peak compression VRAM 2.26 GB.
  - `qwen3-14b-streaming-bpw5` — PPL ratio 1.011× fp16, peak compression VRAM 3.37 GB. Best quality on the curve.
  - `qwen3-32b-streaming-bpw5` — PPL ratio 1.037× fp16, peak compression VRAM 4.85 GB.
  - `qwen2.5-72b-streaming-bpw5` — PPL ratio 1.016× fp16, peak compression VRAM **8.98 GB on a single 32 GB consumer GPU**. The headline.
- **`uc pull` command** for downloading published checkpoints from HuggingFace.
- **`uc list` command** for browsing all published SipsaLabs models.
- **`uc bench` command** for benchmarking compressed artifacts on lm-eval-harness tasks.
- **`uc info` command** for inspecting compressed artifact metadata.
- **`uc demo` command** for scripted CLI demo (screen-recording-ready).
- **Reproducibility scripts** in `uc verify` for verifying published numbers.
- **Streaming compression runtime** (reference Python implementation, `huggingface_hub`-based). Production CUDA kernels in v0.5+.

### Changed
- **Production bit-rate target raised from 4 to 5 BPW** for the streaming compression tier. The 5 BPW point is the sweet spot for PPL drift across 8B-72B; 4 BPW is the "CONSERVATIVE" tier (T1 90% but PPL ratio 1.014×).
- **Default packed-format parameters updated (proprietary; patent-pending).**

### Documentation
- Internal research log (hypothesis-mechanism-experiment-measurement-conclusion entries, including negative results) is maintained for the team; selected charter-clean negative-result summaries are surfaced via blog posts and release notes.
- FNO Darcy non-transformer transfer demo at `scripts/demo/fno_compression_demo.py` (CPU-only, 33 sec end-to-end).
- Cross-architecture results documented at internal cross-architecture validation set (FNO, U-Net, PINN — including the PINN negative result; details NDA-gated).

### Patent
- U.S. provisional patent applications filed April 2026 (patent pending), with supplements pending. Specific claim scope is available to commercial counterparties under NDA — see PATENT_NOTICE.md.

### Known issues (fixed in 0.4.1)
- See `[0.4.1]` above for the three bugs patched immediately after the v0.4.0 release.

---

## [0.1.3] — 2026-04-29

### Added
- Initial production CLI surface: `pull`, `list`, `info`, `bench`, `demo`, `version`.
- Initial published checkpoints: `qwen3-1.7b`, `qwen3-8b`, `mistral-7b`, `smollm2-1.7b`, `olmo2-1b`, and an earlier preview format for `qwen3-1.7b`.

### Notes
- Pre-streaming-compression release. The earlier preview format used patent-pending 5-bit packing.

---

## [0.1.2] — 2026-04-28

### Added
- Initial PyPI release.
- `pip install ultracompress` becomes available.
- Apache 2.0 LICENSE.

### Notes
- Documentation only; functional CLI shipped in 0.1.3.

---

## [0.1.0] — 2026-04-26 (yanked, superseded by 0.1.2)

### Added
- Initial scaffolding for PyPI distribution.

### Notes
- Yanked due to packaging error. Superseded by 0.1.2.

---

## Versioning policy

- **Major** (X.y.z): breaking changes to the CLI surface OR the compressed artifact format.
- **Minor** (x.Y.z): new commands, new artifact format extensions, new model checkpoints (compatible with existing CLI).
- **Patch** (x.y.Z): bug fixes, security patches, documentation-only changes.

The closed-source production pipeline (commercial license) versions independently from the open-source CLI. Customer engagement contracts pin specific versions; updates require contract amendment.

---

## Contact

- Bugs: github.com/sipsalabs/ultracompress/issues
- Security: security@sipsalabs.com
- Commercial licensing: legal@sipsalabs.com
- General: hello@sipsalabs.com

Implementation details are proprietary and patent-pending.
