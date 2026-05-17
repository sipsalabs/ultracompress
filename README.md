# UltraCompress

Lossless 5-bit transformer compression. Bit-identical reconstruction guaranteed by a SHA-256 manifest.

[![PyPI](https://img.shields.io/badge/pypi-0.6.12-blue.svg)](https://pypi.org/project/ultracompress/)
[![License](https://img.shields.io/badge/license-BUSL--1.1-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Patent](https://img.shields.io/badge/patent-pending-orange.svg)](./PATENT_NOTICE.md)
[![Security](https://img.shields.io/badge/v0.6.12-hardened%20sdist-green.svg)](./CHANGELOG.md)

> **v0.6.12 (current PyPI release):** ships on a MANIFEST-scrubbed sdist (the prior 0.6.7 / 0.6.8 sdists were yanked from PyPI after an RCE-class fix on `torch.load()` paths in 0.6.9). Mistral-7B-v0.3 hits **1.00548×** — the tightest dense 7B-class lossless 5-bit ratio we currently publish. OpenAI-compatible inference API at [`api.sipsalabs.com/v1`](https://api.sipsalabs.com/v1) is **publicly self-serve** — a drop-in replacement for `OPENAI_BASE_URL`. Free $5 credit on signup, no card. Pro $99/mo + Team $499/mo at sipsalabs.com/pricing. The `pip install ultracompress` substrate is fully production today (no API key required for self-host).

Hermes-3-Llama-3.1-405B compressed at 5 bpw lossless: **1.0066x PPL ratio** vs streaming bf16 teacher (5.0692 / 5.0358, n=50, seq_len=1024, FineWeb-edu held-out tail, seed=42). A 405B-class transformer compressed end-to-end on a single 32 GB consumer GPU. Reproduce in 3 commands.

UltraCompress takes a transformer at fp16/bf16 and produces a 5-bit pack you can verify against the original — not "1% PPL drift on WikiText," but a deterministic reconstruction that hashes byte-for-byte to what the trainer measured. That's the honest definition of lossless we care about: an auditor can re-derive every weight from the pack alone, and the SHA-256 manifest fails loudly if anything drifted. Codec internals are patent-pending.

It exists because the bf16-equivalent quality bar matters in places where "good enough on MMLU" isn't enough — defense, FDA-regulated healthcare, SR 11-7 model validation, internal red-team eval at frontier labs. And as a side-effect of the streaming compression path, it lets us put a 405B-parameter model through a single 32 GB consumer GPU without renting an H100 cluster.

We're a small lab shipping this in public while the patents are pending. Most days the lab notebook gets longer than the marketing site does. If you want to know what works, what doesn't, and what we tried this week that failed — read on.

---

## Try it (3 commands)

```bash
pip install ultracompress huggingface_hub[cli]
hf download SipsaLabs/qwen3-1.7b-base-uc-v3-bpw5 --local-dir ./pack
uc verify ./pack
```

Expected output (real, not aspirational — this is what the v0.6.12 verifier prints on a clean pull of the 1.7B-Base artifact):

```
uc_pack_version: 3  (LOSSLESS, self-contained)
codec_source:    trainer-persisted
n_layers:        28
bpw:             5
Spot-check SHA256:
  layer_000.uc:  f87f2aeb3996ab7d…
  layer_014.uc:  …
  layer_027.uc:  …
Layer 0: 7 quantized Linears + 4 extras
All 7 Linear reconstructions have correct shapes.
Bundled scaffold: embed_tokens, model.norm, lm_head present.
→ VERIFY: PASS — bit-identical reconstruction guaranteed.
```

If you also want measured numbers on your hardware (TTFT, steady-state TPS, peak VRAM) — `uc bench ./pack`. Same JSON schema as our published numbers, runs on whatever GPU you have, no Sipsa-side claims to take on faith.

The smallest published artifact is ~1.1 GB. The qwen3-0.6b pack is ~0.4 GB if you want a faster smoke test.

### Or call the API (no install)

```bash
export OPENAI_BASE_URL=https://api.sipsalabs.com/v1
curl $OPENAI_BASE_URL/models
```

Same OpenAI client SDK works unchanged. Inference runs on dual RTX 5090 over Cloudflare Tunnel.

---

## What works today (verified, with JSON receipts)

PyPI `v0.6.12` is the current release. v0.6.12 packs are **self-contained** — they bundle LayerNorm + `embed_tokens` + `lm_head` inside the pack directory, so reproducing a published artifact no longer requires pulling the original bf16 alongside it. ~622 MB auxiliary on top of the compressed body for typical decoder vocab.

**14 architectures independently PPL-verified end-to-end at 5 bpw** against their own bf16 baseline (FineWeb-edu held-out tail, seq_len=1024, seed=42) — these are the numbers we stand behind. A further 8 are compression-complete and uploaded to HuggingFace with eval pending (22 packs total; the OpenAI-compatible API serves 19). Verified PPL ratios are tracked in an internal benchmark ledger; the public streaming-compression eval JSONs live in `scripts/overlay/artifacts/`, and full per-architecture eval provenance is available under the verification flow / on request.

The headline result and the tightest dense records currently public on HuggingFace:

| Model | Params | Class | PPL ratio | HF artifact | Status |
|---|---|---|---|---|---|
| Hermes-3-Llama-3.1-405B | 405B | 405B-class lossless on a single 32 GB consumer GPU | **1.0066** | [`SipsaLabs/hermes-3-llama-3.1-405b-uc-v3-bpw5`](https://huggingface.co/SipsaLabs/hermes-3-llama-3.1-405b-uc-v3-bpw5) | gated · access on request |
| Mistral-7B-v0.3 | 7.2B | sub-0.6% drift (new this week) | **1.00548** | [`SipsaLabs/mistral-7b-v0.3-uc-v3-bpw5`](https://huggingface.co/SipsaLabs/mistral-7b-v0.3-uc-v3-bpw5) | live |
| Qwen3-1.7B-Base | 1.7B | sub-0.5% drift | **1.00401** | `SipsaLabs/qwen3-1.7b-base-uc-v3-bpw5` | live |
| Qwen3-14B | 14.0B | sub-0.5% drift | **1.00403** | `SipsaLabs/qwen3-14b-uc-v3-bpw5` | live |
| Qwen3-8B | 8.0B | sub-0.5% drift | **1.00440** | `SipsaLabs/qwen3-8b-uc-v3-bpw5` | live |
| Mixtral-8x7B-v0.1 (MoE) | 47B (13B active) | sub-0.5% drift | **1.00368** | `SipsaLabs/mixtral-8x7b-v0.1-uc-v3-bpw5` | live |
| Phi-3-mini-4k-instruct | 3.8B | sub-0.3% drift (seq_len=128, not apples-to-apples) | **1.00262** | `SipsaLabs/phi-3-mini-4k-instruct-uc-v3-bpw5` | live |
| Phi-3.5-MoE-instruct | 42B (MoE 16-exp) | sub-0.5% drift | (eval pending this week) | `SipsaLabs/phi-3.5-moe-uc-v3-bpw5` | upload in flight |

Hermes-3-405B is the headline. The 1.0066x ratio is `5.0692 / 5.0358` — both halves of the fraction measured under the same per-layer streaming reconstruction comparator (n=50, seq_len=1024, FineWeb-edu held-out tail, seed=42). The bf16 teacher took 7.7 hours on cuda:1; the 5-bpw pack took 14.3 hours. Pack body is ~251 GB, bit-identical SHA-256 reconstruction. The Mistral-7B 1.00548× row is new this week and is the tightest dense 7B-class lossless 5-bit ratio we currently publish.

Other notable verified results (full table in [Appendix](#appendix-full-architecture-matrix) below):

- **Lossless 5-bit state-space-model compression**: Mamba-2.8B compression validated with bit-identical reconstruction (scalar-only; end-to-end PPL eval pending, and the correction-overlay path for SSMs hasn't landed yet, see "what doesn't work").
- **HuggingFace presence**: 40 repos under [`huggingface.co/SipsaLabs`](https://huggingface.co/SipsaLabs).
- **PyPI**: [pypi.org/project/ultracompress](https://pypi.org/project/ultracompress/).
- **OpenAI-compatible API**: [api.sipsalabs.com/v1](https://api.sipsalabs.com/v1) — self-serve via [sipsalabs.com/pricing](https://sipsalabs.com/pricing) (Pro $99/mo, Team $499/mo). Free $5 trial credit on signup.

The `SipsaLabs` HuggingFace org page is the live source of truth. If a repo there has files committed, `uc verify` will pass on it after `hf download`.

---

## What doesn't work yet

Things people sometimes assume work because the rest of it does. They don't, and we'd rather you know:

- **Long-context evaluation past seq_len=1024.** Every PPL number above is at seq_len=1024 on the FineWeb-edu held-out tail. We have not yet run controlled evals at 4K/8K/32K context. If your workload depends on long-context behavior, treat the published ratios as "short-context evidence, long-context unmeasured." Eval harness for that lands in v0.7.
- **`uc compress` as a one-shot CLI.** v0.6.12 still requires the production trainer (patent-protected, not part of the public package). The release path is: trainer (private) → `pack_v3.pack_e2e_dir_v3` (public packer) → published artifact + `uc verify`.
- **State-space models past scalar-only.** Mamba-2.8B compression is validated (bit-identical reconstruction) but its end-to-end PPL eval is still pending; scalar-only is as far as the SSM path goes today. We tried two correction-overlay paths on top — both made it worse. The streaming compression runner has to be adapted for `MambaBlock` iteration with real activations to break this; deferred.
- **TinyLlama-1.1B-Chat PPL eval.** The pack itself verifies clean (`uc verify` PASS) and the HF artifact uploaded. But the PPL eval forward pass throws a CUDA device-side assert that we haven't traced yet. The matrix shows it as `(deferred)`, not a fabricated number.
- **Qwen3-32B and Llama-3.1-70B PPL ratios.** Both have local `uc verify` PASS; both have stale or suspect baseline PPL numbers we won't republish. Apples-to-apples re-evals at the standard methodology are queued.
- **Below 1.0040× on Qwen3-1.7B-Base.** This is our tightest dense floor and we tried 5 different paths to break it this week. Three were within noise; two were catastrophic regressions (1.0682× and 1.1306×). 1.0040× stands as the empirical floor at the current configuration.
- **Cross-architecture generalization of corrective methods.** Methods that meaningfully tighten Mistral-7B and Phi-3-mini are a wash on Llama-3.1-8B and Qwen3-0.6B at our current configuration. Generalization is per-architecture, not universal — published numbers reflect what generalized.
- **HF uploads on residential bandwidth.** Several large-pack uploads (Mixtral-8x22B at 100GB, SmolLM2, Qwen3-0.6B) hit SSL EOF mid-stream. Our 8-attempt watchdog wrapper catches it but multi-hour residential uploads remain brittle.

---

## Why this isn't AWQ / GPTQ / EXL3

Every other 4–5 bit compression library targets a quality threshold ("sub-1% PPL on WikiText"). UltraCompress targets a **reconstruction contract**: the customer artifact contains everything needed to deterministically reconstruct — bit-identically — the exact dequantized weights the trainer used. Internal codec details are patent-pending and deliberately not described here. A SHA-256 manifest covers the pack end-to-end. If anything drifts, `uc verify` fails loudly; you don't have to take "it should be close" on faith. Codec internals are patent-pending.

This matters when "the model picks a slightly-wrong variable name" is a regulatory finding rather than a cosmetic complaint. Defense / aerospace deploy-bit-exactness is a compliance requirement. FDA-regulated healthcare AI requires model equivalence between dev and deploy. SR 11-7 (Federal Reserve model validation) requires reproducible audit recovery. A frontier lab's red-team eval is only valid against the same inference path the team will actually deploy.

For pure-throughput inference on a fixed prompt distribution that matches your AWQ calibration set, with no downstream fine-tuning, AWQ at 4 bpw on vLLM is genuinely fine and we'll say so on a sales call. The Phase 0 POC is structured to find out: bring a model, we deliver a UC pack, you `uc bench` it on your hardware against your existing AWQ/GPTQ build. If we don't materially help, you keep the diagnostic and we don't push Phase 1.

The short version: as of mid-2026 we are not aware of another published library targeting a bit-identical reconstruction contract (as opposed to a PPL-threshold) for 5-bit transformer compression on the public HuggingFace Hub. If you find one, tell us — we'd rather benchmark against it than claim a gap that isn't there.

---

## Honest negative results

Most projects hide their failures. We catalogue them at the same level of detail as the wins. Across the 2026-05-08 → 2026-05-10 research arc the ratio of catalogued failures to published wins was roughly 15:9, and that's the ratio we'd want any external evaluator to use when assessing whether the positive numbers are real. They are.

A taste of what's in there:

- **An initialization shortcut we tried** — made PPL 0.07 pp WORSE on Mamba and was discarded. Documented as a negative result; method specifics withheld (patent-pending).
- **A multi-pass variant we hypothesized would help** — produced a catastrophic 13.7× regression vs. the single-pass baseline. CLOSED.
- **Importing an AWQ-style pre-scaling step** — produced a catastrophic +13% regression and was ruled out. CLOSED.
- **Pushing the training schedule past the current configuration** — gained nothing (1.0042× vs 1.0040×, within noise). The 1.0040× floor stands.
- **"Base models compress tighter than instruct" hypothesis** — refuted 2/3 of architectures. Instruct-fine-tuning effects on quantization-friendliness are architecture-dependent, not universal. Hypothesis dropped.
- **A universal cure for the dense PPL floor** — methods that tighten Mistral and Phi-3 are a wash on Llama-3.1-8B and Qwen3-0.6B at our current configuration. Per-architecture, not universal. Documented this week.

Researchers comparing 5-bit codecs can treat this list as a partial audit trail — it will save you from re-running experiments we already ran. Detailed methodology for any specific failure is available to design partners under NDA.

---

## Who this is for

Direct, not aspirational:

- **If you serve LLMs in production and your VRAM bill is the constraint**, this might help. The streaming compression path bounds peak compression-time VRAM to roughly one transformer layer regardless of total depth (8.98 GB for Qwen2.5-72B; same recipe scales to 405B), and the v3 pack format is bit-exact-reproducible at inference time. Email `founder@sipsalabs.com` with your stack and a target latency/quality bar; we'll tell you honestly whether UC fits.
- **If you're a researcher comparing 5-bit codecs**, the public streaming-compression eval JSONs in `scripts/overlay/artifacts/` plus the fixed methodology (FineWeb-edu held-out tail, seq_len=1024, seed=42, per-model n noted in the matrix) are the audit trail; full per-architecture eval provenance is available on request, and the negative-results section above tells you what we already tried that didn't work. The BUSL-1.1 license covers research, reproduction, and citation freely.
- **If you're in a regulated domain** (defense, FDA-regulated healthcare, SR 11-7 model validation, frontier lab red-team), the bit-identical reconstruction contract is the actual reason to talk to us. Phase 0 POC ($5K, 5 business days, customer-picked model) gets you a pack you can audit yourself. Email `founder@sipsalabs.com` for the POC scope.
- **If you're at a frontier lab** distributing internal model artifacts and want red-team eval fidelity preserved across deploy environments, the SHA-256 manifest exists for exactly that.

If your workload is "MMLU has to stay above X" and you're not pushing the model into long-tail or downstream-fine-tuning territory, AWQ at 4 bpw is probably a better answer than this. We'll say so.

---

## We're a small company looking for design partners

Sipsa Labs is a small lab shipping in public. Our compression methods are patent-pending; details are in [`PATENT_NOTICE.md`](./PATENT_NOTICE.md). Short version: the CLI/verifier source is Business Source License 1.1 — free for individuals, research, and organizations under $1M revenue; any other commercial or production use needs a commercial license — and the patent-pending method and the compressed artifacts are licensed separately. Email `founder@sipsalabs.com`.

We ship in public and work with a small number of design partners. The most useful ways to engage:

- **Paid Phase 0 POC** — `founder@sipsalabs.com`, $5K / 5 business days / customer-picked model. The Day 7 deliverable is a pack you can self-verify with `uc verify` + benchmark with `uc bench`. Acceptance gate is `uc verify` PASS + PPL ratio within 1.5% on your eval set.
- **GitHub Sponsors** — [github.com/sponsors/sipsalabs](https://github.com/sponsors/sipsalabs). Supports continued open releases.
- **Press / commentary** — `press@sipsalabs.com`. The framing we can stand behind: a 405B-class model compressed end-to-end on a single 32 GB consumer GPU, and a 5-bit codec built around a bit-identical reconstruction contract — both verifiable via the artifacts above.
- **Twitter** — `@SipsaLabs`. New account; if you found this repo first that's because we ship faster than we tweet.
- **CHANGELOG** — [v0.6.12 release notes](./CHANGELOG.md) for the hardened-sdist + MANIFEST-scrub narrative.

If you're tracking the project: release notes in `CHANGELOG.md` and the `/blog` posts on sipsalabs.com are the canonical "what shipped" surfaces.

---

## How v3 lossless actually works

The pack persists trainer-side codec state alongside the quantized weights so customer-side reconstruction is a deterministic function of bytes-on-disk that runs the same arithmetic the trainer ran. Bit-identity is verified by SHA-256 manifest at customer load. Internal codec specifics are NDA-gated — contact founder@sipsalabs.com for technical due diligence.

The streaming path is what lets UltraCompress scale to a 405B-class model on a single 32 GB consumer GPU instead of a multi-GPU server. How that bound is achieved is patent-pending and deliberately not described here.

The PPL evaluator + verifier ship public in this package; the production trainer is patent-protected. Released artifacts include the SHA-256 manifest needed to reproduce every published number via `uc verify`.

---

## Repository layout

```
ultracompress/
├── src/ultracompress_cli/        CLI entry points (verify / pull / list / info / bench / demo)
├── scripts/overlay/              Streaming-compression evaluators + ground-truth JSON artifacts
├── tests/                        Regression tests
├── examples/                     Minimal runnable usage
├── docs/                         Quickstart, concepts, command + reference docs (mkdocs site)
├── PATENT_NOTICE.md              Patent posture + commercial-license contact
├── CHANGELOG.md                  Per-release narrative (canonical "what shipped")
└── LICENSE                       BUSL-1.1 (CLI source) + patent-scope notice
```

The production trainer that produces packs is patent-protected and not part of this public package; this repo ships the verifier, benchmarker, and CLI that operate on published artifacts.

---

## Appendix: full architecture matrix

14 architectures independently PPL-verified end-to-end; a further 8 are compression-complete and uploaded to HuggingFace with eval pending (22 packs total, 19 served on the OpenAI-compatible API) as of 2026-05-16. PPL = FineWeb-edu held-out tail, seq_len=1024 (Phi-3-mini noted at seq_len=128 — not apples-to-apples), seed=42, against the model's own bf16 baseline on a single RTX 5090. Most rows use n=30 prompts; the 405B row uses n=50 with per-layer streaming reconstruction on both halves of the fraction (apples-to-apples comparator). Sub-baseline OLMo-2-Instruct (0.9998×) is a real measurement — compression appears to act as a faint regularizer at n=30 — not a typo.

| Model | HF artifact | Params | Layers | PPL ratio |
|---|---|---|---|---|
| OLMo-2-0425-1B-Instruct | `olmo-2-0425-1b-instruct-uc-v3-bpw5` | 1.0B | 16 | **0.9998** |
| Phi-3-mini-4k-instruct | `phi-3-mini-4k-instruct-uc-v3-bpw5` | 3.8B | 32 | **1.00262** (seq_len=128 caveat) |
| Mixtral-8x7B-v0.1 (MoE) | `mixtral-8x7b-v0.1-uc-v3-bpw5` | 47B | 32 | **1.00368** |
| Qwen3-1.7B-Base | `qwen3-1.7b-base-uc-v3-bpw5` | 1.7B | 28 | **1.00401** |
| Qwen3-14B | `qwen3-14b-uc-v3-bpw5` | 14.0B | 40 | **1.00403** |
| Yi-1.5-9B | `yi-1.5-9b-uc-v3-bpw5` | 8.8B | — | 1.00414 |
| Qwen3-8B | `qwen3-8b-uc-v3-bpw5` | 8.0B | 36 | **1.00440** |
| Mistral-7B-v0.3 | `mistral-7b-v0.3-uc-v3-bpw5` | 7.2B | 32 | **1.00548** |
| Qwen3-0.6B | `qwen3-0.6b-uc-v3-bpw5` | 0.6B | 28 | 1.0069 |
| OLMo-2-0425-1B | `olmo-2-0425-1b-uc-v3-bpw5` | 1.0B | 16 | 1.0073 |
| SmolLM2-1.7B-Instruct | `smollm2-1.7b-instruct-uc-v3-bpw5` | 1.7B | 24 | 1.0075 |
| SmolLM2-1.7B | `smollm2-1.7b-uc-v3-bpw5` | 1.7B | 24 | 1.0085 |
| Mamba-2.8B (SSM) | `mamba-2.8b-hf-uc-v3-bpw5` | 2.8B | 64 | (compression validated; PPL eval pending) |
| Llama-3.1-8B | `llama-3.1-8b-uc-v3-bpw5` | 8.0B | 32 | 1.0125 |
| Qwen3-1.7B (Instruct) | `qwen3-1.7b-uc-v3-bpw5` | 1.7B | 28 | 1.0200 *(eval pending reconciliation)* |
| Hermes-3-Llama-3.1-405B | `hermes-3-llama-3.1-405b-uc-v3-bpw5` | 405B | 126 | **1.0066** (5.0692 / 5.0358, n=50, per-layer streaming) |
| Qwen3-32B | `qwen3-32b-streaming-bpw5` | 32B | 64 | (re-eval pending) |
| Llama-3.1-70B | `llama-3.1-70b-uc-v3-bpw5` | 70B | 80 | (re-eval pending) |
| Qwen3-235B-A22B (MoE) | `qwen3-235b-a22b-uc-v3-bpw5` | 235B | 94 | (eval pending) |
| Mixtral-8x22B-v0.1 (MoE) | `mixtral-8x22b-v0.1-uc-v3-bpw5` | 141B | 56 | (eval pending) |
| Phi-3.5-MoE-instruct (MoE) | `phi-3.5-moe-uc-v3-bpw5` | 42B | 32 | (eval pending this week) |
| TinyLlama-1.1B-Chat | `tinyllama-1.1b-chat-v1.0-uc-v3-bpw5` | 1.1B | 22 | (CUDA assert in eval harness; pack verifies clean) |

---

## License

- The CLI/verifier source in this repository is published under the [Business Source License 1.1](./LICENSE). It is **free of charge** for (a) individuals, (b) non-commercial research/academic use, and (c) organizations with annual revenue under US $1,000,000. On the Change Date (2030-05-16) it converts to Apache License 2.0.
- **Any other commercial or production use** — including using it in or for a paid product, a hosted/managed service, or any revenue-generating activity — requires a commercial license. Contact `legal@sipsalabs.com`.
- BUSL-1.1 conveys **no** license to the patent-pending compression/reconstruction methods or the production trainer (not in this repo) — see [`PATENT_NOTICE.md`](./PATENT_NOTICE.md).
- Pre-compressed model artifacts on HuggingFace are licensed **separately** (Business Source License 1.1, per the license on each artifact) and additionally carry the upstream teacher model's license. They are not covered by this repository's grant.
- Building a product whose core value depends on the method, or want technical due diligence under NDA? `legal@sipsalabs.com`.

## Citation

```bibtex
@software{sipsa_ultracompress_2026,
  author = {{Sipsa Labs, Inc.}},
  title  = {UltraCompress: Lossless 5-bit Transformer Compression},
  year   = {2026},
  url    = {https://github.com/sipsalabs/ultracompress}
}
```

## Contact

- Commercial / Phase 0 POC: `founder@sipsalabs.com`
- Security: `security@sipsalabs.com`
- Press: `press@sipsalabs.com`
- HuggingFace: [`huggingface.co/SipsaLabs`](https://huggingface.co/SipsaLabs)
- PyPI: [`pypi.org/project/ultracompress`](https://pypi.org/project/ultracompress/)
- API: [`api.sipsalabs.com/v1`](https://api.sipsalabs.com/v1)
- GitHub release: [latest release](https://github.com/sipsalabs/ultracompress/releases/latest)
- Sponsors: [`github.com/sponsors/sipsalabs`](https://github.com/sponsors/sipsalabs)
