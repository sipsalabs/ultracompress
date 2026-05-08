# UltraCompress v0.5.2 — Release Notes

**Release date:** 2026-05-08
**PyPI:** `pip install -U ultracompress`
**Headline:** v0.5.1 + `python -m ultracompress` fallback + SSM `TARGET_SUBS` extension; production validation refreshed across the 10-pack v3 matrix.

This is a small polish / maintenance release on top of [v0.5.1](../CHANGELOG.md) (the lossless v3 launch). No new compression algorithm, no artifact-format change, no breaking changes — just two ergonomic CLI / packing-coverage improvements and a fresh round of production validation evidence.

If you are already on v0.5.1 and not running on Mamba/RWKV/Jamba SSM models or in `python -m`-only environments, this upgrade is purely defensive. There is no behavior change for the dense-transformer pack / load / verify path.

---

## Upgrade

```bash
pip install -U ultracompress
```

That is it. v0.5.2 is a drop-in for v0.5.1 — same v3 pack format (`UC_VERSION = 3`), same CLI surface, same public HuggingFace artifacts.

---

## Added

### `python -m ultracompress` works as a CLI fallback

A new `ultracompress/__main__.py` module dispatches `python -m ultracompress <subcommand>` to the same CLI handler as the `uc` console script.

This unblocks three customer environments where the `uc` console script is awkward or unavailable:

- **Jupyter notebooks** that don't put pip-installed console scripts on the kernel's `PATH`. Customers can now run `!python -m ultracompress verify ./packed_dir` from a notebook cell without restarting the kernel or fiddling with `%env`.
- **Locked-down CI runners** (some enterprise GitHub Actions, GitLab, Buildkite images) that don't put `pip install --user` console scripts on `PATH`. `python -m ultracompress …` resolves through `sys.path` instead.
- **Minimal Docker images** (`python:3.12-slim`, distroless variants) where the user has installed UltraCompress into a venv but the venv `bin/` directory isn't on `PATH` for the entrypoint.

The fully-supported entry points remain `uc <subcommand>` and `ultracompress <subcommand>` — `python -m ultracompress` is a fallback, not a replacement.

### SSM Linear naming added to `TARGET_SUBS` in `pack` / `pack_v3`

`ultracompress/pack.py` now extends the `TARGET_SUBS` substring tuple to include the four standard state-space-model (Mamba, Mamba-2, RWKV, AI21 Jamba) block Linear names: `in_proj`, `x_proj`, `dt_proj`, `out_proj`.

This is a non-breaking extension: the new substrings don't collide with any transformer Linear naming convention used by the streaming runner. Dense and MoE transformer packs are byte-identical with v0.5.1 — verified by re-running `uc verify` on `SipsaLabs/mistral-7b-v0.3-uc-v3-bpw5` after the patch (PASS, no regression).

The end-to-end SSM compatibility was already validated in v0.5.0 against `state-spaces/mamba-2.8b-hf`. v0.5.2 closes the loop by making sure the **standalone** `uc pack` / `uc pack_v3` CLI commands also pick up SSM Linears, not just the streaming runner. Net effect: customers can now run `uc pack` directly against an SSM checkpoint without the SSM Linear modules silently being skipped as "not target Linears".

---

## Fixed

No customer-facing bugs were reported against v0.5.1 between its release earlier today and the v0.5.2 cut. The only behavior changes in v0.5.2 are the two additions above.

---

## Changed

Nothing breaking. CLI surface, pack format header (`UC_VERSION = 3`), hugging-face artifact layout, and Python API signatures are all identical to v0.5.1.

---

## Validation

This release ships with the most thorough cross-architecture pack-validation evidence we have published to date.

### `uc verify` PASS across the full 10-pack v3 production matrix

All in-flight HuggingFace artifact source packs are structurally sound. Each was re-verified end-to-end (header, sha256 spot-check, layer reconstruction shape match) on the v0.5.2 codebase:

| Architecture family | Model | Type | Status |
|---|---|---|---|
| Qwen3 | Qwen3-1.7B | dense | PASS |
| Qwen3 | Qwen3-8B | dense | PASS |
| Qwen3 | Qwen3-14B | dense | PASS |
| Llama 3.1 | Llama-3.1-8B | dense | PASS |
| Llama 3.1 | Llama-3.1-70B | dense | PASS |
| Mistral v0.3 | Mistral-7B-v0.3 | dense | PASS |
| Mixtral | Mixtral-8x7B | MoE | PASS |
| Mixtral | Mixtral-8x22B | MoE | PASS |
| Phi-3.5 | Phi-3.5-MoE | MoE | PASS |
| Qwen3 | Qwen3-235B-A22B | MoE (388 quantized Linears / layer) | PASS |

When the in-flight HF uploads commit, each will reproduce these PASS results bit-for-bit on the customer's machine.

### Second public artifact reproducible end-to-end in 3 commands

`SipsaLabs/mistral-7b-v0.3-uc-v3-bpw5` joins `SipsaLabs/qwen3-1.7b-uc-v3-bpw5` as a `uc verify`-PASS public artifact reproducible by anyone with `pip` and `huggingface_hub`:

```bash
pip install -U ultracompress huggingface_hub
hf download SipsaLabs/mistral-7b-v0.3-uc-v3-bpw5 --local-dir ./mistral-7b-v3
uc verify ./mistral-7b-v3
```

Documented in `docs/PUBLIC_VERIFICATION_DASHBOARD_2026_05_08.md`.

This is a meaningful step beyond a single-architecture proof. Two distinct dense-transformer architectures (Qwen3 and Mistral), two distinct sets of HuggingFace tooling-side metadata, both reproducing structurally identical `uc verify` PASS in the customer's hands.

### Multi-architecture PPL bench refresh at 5 bpw

PPL ratios (compressed PPL / fp16 PPL on the same evaluation harness):

| Model | PPL ratio at 5 bpw | Notes |
|---|---|---|
| Mistral-7B-v0.3 | **1.0100×** | tightest dense ratio measured |
| Llama-3.1-8B | **1.0125×** | |
| Hermes-3-405B | in flight | 47 / 126 layers complete at write time, ETA tonight |

These are not CLI or API changes — they are measurement-only updates documenting where the current public production stack sits on the quality / size frontier. Per pre-filing IP discipline, the algorithm-level reasons behind these numbers are not disclosed in this release; the numbers themselves are fair game.

---

## Compatibility

- **Pack format:** `UC_VERSION = 3` (unchanged from v0.5.1). Loaders v3, v2, v1 all keep working.
- **CLI surface:** identical to v0.5.1. `uc`, `ultracompress`, and (new in v0.5.2) `python -m ultracompress` all dispatch the same handlers.
- **Python API:** identical to v0.5.1. `_API_V2_AVAILABLE` flag still exposed at module level.
- **HuggingFace artifacts:** zero re-upload required. Existing `SipsaLabs/<model>-uc-v3-bpw5` artifacts read on v0.5.2 with no changes.
- **State-space models:** dense streaming-runner SSM compression continues to work as in v0.5.0 (validated on Mamba-2.8B, mean rel_l2 0.0458, end-to-end PPL ratio 1.0119×). v0.5.2 additionally makes the standalone `uc pack` / `uc pack_v3` commands pick up SSM Linears.

---

## Pre-filing IP discipline

Per the standing pre-filing policy, this release deliberately does **not** disclose:

- New compression algorithm specifics (Track A / B / C / D method-level details).
- Bit-rate / quality breakthroughs beyond what is already public.
- Internal research module names or activation-clustering / cross-layer-merge specifics.

Everything in this release is at the CLI / API / artifact-format layer — the layer that customers and CI pipelines integrate against — which is fair game.

---

## Acknowledgments

Thanks to the early customers running `uc verify` against fresh-pulled HuggingFace artifacts in non-standard environments — Jupyter, locked-down CI, and minimal Docker images — for the friction reports that motivated the `python -m ultracompress` fallback.

---

## Contact

- Bugs: github.com/sipsalabs/ultracompress/issues
- Security: security@sipsalabs.com
- Commercial licensing: legal@sipsalabs.com
- General: hello@sipsalabs.com
