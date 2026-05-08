# Changelog

All notable changes to UltraCompress are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), versioning per [SemVer](https://semver.org/).

---

## [0.5.0] — 2026-05-08

### Added
- **State-space-model (SSM) architectural compatibility verified** on Mamba-2.8B (`state-spaces/mamba-2.8b-hf`). 256 SSM Linear modules (`in_proj`, `x_proj`, `dt_proj`, `out_proj`) compress with mean rel_l2 = 0.0458 and bit-identical reconstruction. End-to-end PPL ratio = **1.0119** with GSQ-only at 5bpw (no V18-C correction). To our knowledge, UltraCompress is the first quantization library publicly compatible with both transformer and state-space architectures, including emerging hybrids such as AI21 Jamba.
- **`uc pack v0.3` lossless binary format** (`ultracompress/pack_v3.py`). Reads the trainer's k-means LEARNED grid + per-block scales + bit-packed integer codes from `gsq_codecs` (a new state_dict key written by the streaming compression runner). Reconstruction `W_base = absmax × grid[codes]` is mathematically lossless — bit-identical reconstruction of trainer-quantized weights.
  - Validated end-to-end: source compressed PPL 18.3748 vs v3 reload PPL 18.3748 on Qwen3-1.7B (delta 0.000003%).
  - Bit-equal state-dict round-trip across 32 keys (max_abs_diff = 0.0).
  - File header bumped to `UC_VERSION = 3`.
- **Trainer-side codec persistence** in `streaming_compression_runner.py`:
  - `gsq_quantize_weight(..., return_codec=True)` returns `(Wq, grid, codes, absmax)` tuple. Default `return_codec=False` is back-compatible.
  - `compress_single_layer` saves `gsq_codecs` dict per quantized Linear into the layer.pt file.
  - K-means sub-sampling now uses a deterministic `torch.Generator().manual_seed(42)`.
- **8-architecture v3 pack matrix** uploaded to HuggingFace at `SipsaLabs/<model>-uc-v3-bpw5`:
  - Dense: Qwen3-1.7B, Mistral-7B-v0.3, Llama-3.1-8B, Qwen3-8B, Qwen3-14B, Llama-3.1-70B
  - MoE: Mixtral-8x7B-v0.1, Phi-3.5-MoE-instruct
  - Mean PPL_r for 5 dense small models: **1.0077** (sub-1% perplexity degradation).
- **Vectorized `_bitpack` / `_bitunpack`** in `ultracompress/pack.py` via `np.packbits` / `np.unpackbits` with bitorder='little'. **~1000× speedup** (16M-weight roundtrip in ~270ms vs prior Python loop in minutes).
- **Pack format extras section** for non-quantized layer tensors (norms, layer-level routers). Pack format now self-contained — customer doesn't need source layer.pt files for reconstruction.
- **`scripts/overlay/_hf_upload_v3_pack.py`** — wrapper for uploading v3 packs to HF with auto-generated README.md per repo.
- **`scripts/overlay/_validate_uc_pack_v3.py`** — validation gate (pack → reconstruct → eval PPL → compare).
- **`scripts/overlay/_cleanup_disk_post_pack_v2.py`** — disk cleanup gated on `uc_pack_version >= 3` (anti-mistake guard).

### Documented
- `docs/OPERATOR_PLAYBOOK_2026_05_07.md` — cardinal rules + 2026-05-07 cleanup-mistake postmortem.
- `docs/AUTONOMOUS_MODE_SUMMARY_2026_05_07.md` — full session writeup of the v0.3 push.
- `docs/AFWERX_SBIR_PHASE1_PROPOSAL_DRAFT_2026_05_07.md` — submit-ready SBIR Phase I proposal.
- `docs/POST_EIN_DAY_0_CHECKLIST_2026_05_07.md` — 90-min sequence for the day Atlas EIN arrives.

### Fixed
- `uc pack` v0.2 was lossy (~22% PPL regression) because it reverse-derived k-means codes from dequantized weights assuming a uniform symmetric grid `{-15..15}/15`. The trainer's actual grid is k-means LEARNED — different. v0.3 (this release) reads the trainer-persisted codec directly and is lossless.
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
- **Streaming compression pipeline** (`scripts/overlay/streaming_compression_runner.py`) that processes one transformer block at a time. Peak GPU memory bounded by ~one layer regardless of total model parameter count.
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
- **Reproducibility scripts** in `scripts/overlay/eval_compressed_only.py` for verifying published numbers.
- **Streaming compression runtime** (reference Python implementation, `huggingface_hub`-based). Production CUDA kernels in v0.5+.

### Changed
- **Production bit-rate target raised from 4 to 5 BPW** for the streaming compression tier. The 5 BPW point is the sweet spot for PPL drift across 8B-72B; 4 BPW is the "CONSERVATIVE" tier (T1 90% but PPL_r 1.014×).
- **Default per-block size for scalar quantization is 64** (was 128 in earlier internal versions).
- **Default correction overlay rank is 32** (was 16 in earlier internal versions).
- **Default distillation steps per layer is 200** (was 1500 in earlier internal versions). Documented saturation effect: 500+ steps regresses end-to-end PPL.

### Documentation
- Open-source LAB-NOTEBOOK at `docs/LAB-NOTEBOOK.md` documenting hypothesis-mechanism-experiment-measurement-conclusion entries from the research cycle. Includes negative results.
- FNO Darcy non-transformer transfer demo at `scripts/demo/fno_compression_demo.py` (CPU-only, 33 sec end-to-end).
- Cross-architecture results documented at `docs/non_transformer_v18c_results.json` (FNO, U-Net, PINN — including the PINN negative result).

### Patent
- USPTO 64/049,511 (correction overlay) — filed 2026-04-25.
- USPTO 64/049,517 (shared-block parameter dispatch) — filed 2026-04-25.
- Track A supplement filing scheduled for 2026-05-09 ($65 micro-entity fee).

### Known issues (fixed in 0.4.1)
- See `[0.4.1]` above for the three bugs patched immediately after the v0.4.0 release.

---

## [0.1.3] — 2026-04-29

### Added
- Initial production CLI surface: `pull`, `list`, `info`, `bench`, `demo`, `version`.
- Initial published checkpoints: `qwen3-1.7b-uc2p79`, `qwen3-8b-uc2p79`, `mistral-7b-uc2p79`, `smollm2-1.7b-uc2p79`, `olmo2-1b-uc2p79`, `qwen3-1.7b-trackb-preview`.

### Notes
- Pre-streaming-compression release. The `uc2p79` format used the row-overlay-rotation packing at 2.798 bpw.

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
