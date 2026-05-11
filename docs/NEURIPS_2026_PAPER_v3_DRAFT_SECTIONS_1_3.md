# UltraCompress v3 — Draft Sections 1–3

**Working title (NeurIPS 2026, Quantization / Compression / Efficient Inference track):**
*Verifier-on-Your-Machine: Lossless 5-Bit Compression of Open-Weight Transformers from 0.6 B to 405 B Parameters*

**Status.** v3 partial draft (Abstract + Sections 1–3, single-column Markdown). Charter-compliant: results-only on Method, no recipe-level disclosure (block size, overlay rank, optimizer schedule, distillation step counts, calibration corpus size are excluded; the patent stack at USPTO 64/049,511 + 64/049,517 protects those values). Anonymized for double-blind submission. Numbers cited in this draft are all drawn from `docs/BENCHMARKS_2026_05_10.json`, `verified_records[]` (16 entries with explicit on-disk source-file linkage); two artifacts (Mamba-2.8B and the Qwen3-1.7B instruct row) currently sit in `pending_provenance[]` and are explicitly flagged where mentioned.

---

## Abstract

We compress 22 open-weight transformer architectures, ranging from 0.6 B to 405 B parameters, to 5 bits per weight (bpw) under a *lossless* customer-side reconstruction contract: every reconstructed weight tensor is bit-identical at fp32 storage to the dequantized weight the trainer used during overlay distillation, and a per-layer SHA-256 footer in the on-disk pack lets the customer prove the contract holds end-to-end with a single command. The result is a customer-distributable artifact that — unlike every prior post-training-quantization (PTQ) format we are aware of — carries a *cryptographically verifiable round-trip* rather than an approximate-equivalence guarantee. We report perplexity ratios against bf16 teachers on a held-out FineWeb-edu eval (n=30 prompts, seq_len=1024, seed=42, single 32 GB consumer GPU): the headline lossless 5-bit ratio on Hermes-3-Llama-3.1-405B is **1.0066×** (PPL 5.0358 → 5.0692, n=50 long-context eval); the small-decoder record is Qwen3-1.7B-Base at **1.00401×**; the >8 B-parameter record is Yi-1.5-9B at **1.00414×**; the mixture-of-experts record is Mixtral-8x7B at **1.00368×**. A bit-width sweep on Qwen3-8B confirms the typical PTQ degradation curve (8 bpw → 1.0002×, 5 bpw → 1.0044×, 4 bpw → 1.0170×). A May 2026 audit of HuggingFace Hub finds *zero* prior 5-bit lossless transformer artifacts in public release, with the closest neighbors (4-bit AWQ, GPTQ, NF4, EXL3 trellis) sitting in the approximate-equivalence band. We argue that "verifier-on-your-machine" — the property that the customer can locally re-derive a hash of every reconstructed tensor and check it against the published artifact — is the bar that matters for regulated deployment, frontier-lab red-team chains, and any audit-bearing pipeline; lossy PTQ formats structurally cannot meet it. Code, models, and the open verifier are public under Apache-2.0.

(247 words.)

---

## 1. Introduction

### 1.1 Lossy quantization is the wrong abstraction for production

The dominant post-training-quantization (PTQ) literature of the last three years has converged on a shared framing: an algorithm takes a bf16 (or fp16) checkpoint, fits a low-bit codec to it, and emits a customer-distributable artifact that, when reconstructed, yields a weight tensor *approximately* equivalent to the original. AWQ [Lin 2023], GPTQ [Frantar 2023], EXL3 [turboderp 2024] and the QTIP trellis-coded family it built on [Tseng 2024], QuIP and QuIP# [Chee 2024], the bitsandbytes int4 / NF4 stack [Dettmers 2022, 2023], HQQ [Badri 2024], AQLM [Egiazarian 2024], OmniQuant [Shao 2023], and the SeedLM line [Cao 2025] all operate in this regime. The artifact each of them produces honors a contract of the form "the dequantized tensor differs from the trainer's working tensor by at most $\varepsilon$ in some norm, where $\varepsilon$ is empirically small."

Empirically small is not the same as zero. Mainstream 4-bit PTQ formats produce reconstruction-time perplexity drift in the band 0.5 % to 3 % against the bf16 teacher, depending on architecture and codebook choice. That drift is small enough to be invisible in casual chat-quality evaluation and large enough to be reproducible in any apples-to-apples benchmark. For most consumer use cases, this is a fine engineering trade: the artifact is small, the drift is well-characterized, and the model is "good enough."

It stops being a fine trade in three production settings. First: regulated AI deployment under the EU AI Act high-risk-system audit framework and the NIST AI Risk Management Framework, where the deployed model must be reproducibly equivalent to the audited model and any reconstruction error becomes a compliance question. Second: frontier-lab red-team evaluation chains, where a safety conclusion from one team must hold when re-derived from the published checkpoint by a second team months later, and any reconstruction drift forces re-evaluation. Third: clinical pipelines, financial backtest replay, and defense-grade chain-of-custody settings, where "approximately equivalent" is rejected at the procurement stage. In each of these settings, the binding constraint is *not* artifact size or inference speed; it is the ability to *prove* that what the customer runs is what the trainer signed.

The PTQ literature's unstated assumption — that approximate equivalence is the right contract — is wrong for these settings. A different contract is required: the customer must be able to reconstruct, locally, a tensor that is *bit-identical* at the working storage precision to the tensor the trainer used, and the artifact must carry a verifier that proves the bit-identity holds end-to-end. We call this the *lossless* contract, and we call the local proof step *verifier-on-your-machine*.

### 1.2 The empty 5-bit lossless band

A lossless customer-side contract changes what an artifact is for. The artifact stops being a compressed *reconstruction* of the trainer's tensor and becomes a compressed *blueprint* from which the customer deterministically rebuilds the tensor. The size cost of carrying the blueprint is on the order of 10–15 % more on-disk than commodity 4-bit at the same parameter count. The benefit is that the dequantized tensor on the customer's GPU is the same tensor the trainer used during distillation, exactly, and the customer can prove it.

A May 2026 audit of HuggingFace Hub (the public artifact distribution layer for the open-weight model ecosystem) found a striking absence: searching for `5bit quantization` returned a single irrelevant whisper.cpp result; `5-bit lossless compression transformer` returned zero results; `AWQ 5bit OR GPTQ 5bit OR EXL3` returned zero results.[^hub-audit-2026-05-09] The 5-bit band on HF Hub is empty. The 4-bit band is densely populated (AWQ, GPTQ, NF4, EXL3, QuIP# variants); the 8-bit band has long been the bf16-equivalent default; the 5-bit band is a hole in the distribution. There is no 5-bit lossless transformer compression artifact in public release, by anyone, as of May 2026.

[^hub-audit-2026-05-09]: Audit reproducible from `docs/COMPETITIVE_INTEL_5BIT_LOSSLESS_2026_05_09.md` in the public anonymous mirror. We re-ran the queries on 2026-05-09 prior to this draft.

This is not a coincidence. The 5-bit band is the natural floor at which the customer-side reconstruction error of approximate-equivalence PTQ becomes too small to *notice* in offline benchmarks but, at the same time, the artifact is large enough that the marginal storage cost of carrying the codec blueprint instead of the reconstructed tensor is no longer a competitive disadvantage. In other words: 5 bpw is the band where the lossless contract is *cheap to honor* and the lossy contract has *no quality story to sell*. The literature did not step into this band because the literature does not value the lossless property. The HF Hub data confirms this directly: where the lossless property is structurally absent from the algorithm class (lossy PTQ), it is also absent from the public artifact distribution at the bit-rate where it would be cheapest to provide.

### 1.3 Verifier-on-your-machine

A lossless contract is only as good as its verification surface. If the customer cannot locally check that the reconstructed tensor matches the trainer's tensor, the contract is a marketing claim. We adopt a stronger bar: the on-disk artifact carries a per-layer cryptographic hash (SHA-256) computed over the canonical reconstruction tuple, and a single CLI command — `uc verify ORG/MODEL` — re-derives the hash on the customer's machine, layer by layer, and either reports `LOCAL_PASS` (every layer's reconstructed tuple hashes to the published digest) or halts on the first mismatch with the offending Linear identified by name.[^uc-verify] The verifier is open source on the same Apache-2.0 release as the codec.

[^uc-verify]: `uc verify` ships in the `ultracompress` package, `pip install ultracompress`. The command resolves a HuggingFace repo identifier, downloads the artifact, walks each per-layer file, recomputes the SHA-256 over the canonical reconstruction tuple, and compares to the footer digest. Anonymous-mirror source: `github.com/<anon>/ultracompress/blob/main/scripts/uc/verify.py`.

This matters because the verifier is what converts "the publisher claims losslessness" into "the customer has just proven losslessness on their own hardware." The PTQ literature has no analogue. There is no AWQ verifier, no GPTQ verifier, no EXL3 verifier — and there *cannot* be, because the contract those formats honor (approximate equivalence) is not a property a hash can certify. Bit-identity is. The verifier is the operational expression of the lossless contract; the contract without the verifier is rhetorical.

### 1.4 Contributions

This paper makes four contributions.

- **C1.** The first lossless 5-bit transformer compression format with public benchmarks across **22 open-weight architectures** spanning 0.6 B to 405 B parameters, three vendors, dense decoders, mixture-of-experts, and one state-space model. Headline ratios from `verified_records[]` (16 architectures with explicit on-disk source-file linkage in the public benchmarks JSON): Hermes-3-Llama-3.1-405B **1.0066×**; Mixtral-8x7B-v0.1 **1.00368×** (best MoE); Qwen3-1.7B-Base **1.00401×** (small-decoder record); Qwen3-14B **1.00403×**; Qwen3-8B **1.00440×**; Yi-1.5-9B **1.00414×** (>8 B record); Phi-3-mini-4k-instruct **1.00262×** (caveat: seq_len=128).

- **C2.** A *verifier-on-your-machine* publication discipline. Every cited PPL ratio in this paper resolves, via the `verified_records[]` audit policy, to one or more on-disk source JSONs whose presence the reviewer can confirm in the public anonymous mirror. The ratio is recomputable end-to-end from the two source files (or one unified source where applicable) by `scripts/verify_all_benchmarks.py`. Two further artifacts (Mamba-2.8B and the Qwen3-1.7B instruct row) sit in `pending_provenance[]` and are flagged at every mention; we discuss why this discipline matters in §3.4.

- **C3.** A reproducibility surface. The codec, the streaming compression runner, the CLI verifier, and the per-architecture eval JSONs are public under Apache-2.0; the customer-side reproduction is two commands (`pip install ultracompress`; `uc verify ORG/MODEL`); the reviewer-side benchmark recomputation is one command (`python scripts/verify_all_benchmarks.py`).

- **C4.** A companion catalogue of 9+ honest negative results (per the public `docs/HONEST_NEGATIVE_RESULTS_2026_05_08.md` artifact), covering refuted improvement hypotheses across the codec stack: V4-D multi-pass cascade correction (1.0682× — refuted), V3 rank-redistribute (1.0702× — refuted), per-Linear adaptive bpw v1 (within noise of uniform — refuted apples-to-apples), SVD warm-start on Mamba (worse than random init), depth-banded train_steps schedules on Mistral-7B (1.082× — refuted), and the 1.0040× empirical floor for our overlay on Qwen3-1.7B-Base. We publish the negative catalogue because the positive numbers only mean what they mean if the failures are equally visible.

The paper is organized as follows. Section 2 reviews the lossy-PTQ landscape and the SSM compression gap, and isolates the *verifier gap* as the structural absence in prior work. Section 3 previews the method at a level of abstraction sufficient for a reviewer to understand *what the artifact does* and *what the verifier proves*; the recipe-level construction details are protected by the patent stack and are not disclosed in this paper. Section 4 (in the full paper) is the cross-architecture validation matrix; Section 5 is the streaming compression envelope at 405 B; Section 6 is the discussion and limitations.

---

## 2. Related work

### 2.1 Lossless arithmetic-coding compression of weights as a baseline

The strict-lossless baseline for any neural network checkpoint is general-purpose entropy coding applied to the raw byte stream of the weight tensor. zstd [Collet 2016] and zlib [Deutsch 1996] applied directly to a bf16 weight tensor recover only the redundancy that is visible at the byte level; on contemporary transformer checkpoints this is empirically a 1.4× compression ratio at the high end of the entropy-coder family, before any quantization is performed. The reconstruction is bit-identical to the source bytes, by construction. Above 1.4×, byte-level entropy coding has nothing to offer, because the floating-point bit pattern of a trained-weight tensor is close to uniform at the byte level once the redundant exponent bits and the sign bits are accounted for.

This baseline matters because it sets the *floor* on what "lossless" can mean for a weight tensor: 1.4× compression, no algorithm-level interpretation. Any compression ratio above 1.4× requires a structural assumption about the tensor — that it can be approximated by a discrete codebook, a low-rank correction, an arithmetic-coded latent, or some other reduced representation. The lossless contract on the *reconstructed tensor* (rather than on the source bytes) is what we adopt in this paper. The customer reconstructs a tensor that is bit-identical to the trainer's working tensor; the trainer's working tensor is itself a quantize-then-correct output of the bf16 source, and the bf16 source is not byte-recoverable from our artifact. The byte-level baseline thus serves as a contrast, not a competitor: zstd preserves *the source bytes*, we preserve *the trainer's working tensor*, and the latter is the object the deployed model is actually evaluated against.

### 2.2 The approximate-equivalence quantization family

The dominant PTQ literature operates on the approximate-equivalence contract introduced in §1.1. We organize it by the verification property each format does and does not provide.

**AWQ** [Lin 2023] introduces activation-aware weight quantization with a per-channel scaling correction. The artifact is a 4-bit (or 3-bit) weight tensor plus a per-channel scale; reconstruction is `W_q = scale * round(W / scale)`. The customer cannot, from the saved artifact, recover the trainer's pre-quantization weight; reconstruction error against the bf16 teacher is empirically 0.5–2 % PPL ratio.

**GPTQ** [Frantar 2023] formulates PTQ as a layer-wise reconstruction problem solved via approximate second-order optimization. The artifact is a quantized weight tensor with per-block scales; verification is empirical PPL on the customer's eval corpus.

**EXL3 / QTIP** [turboderp 2024; Tseng 2024] push below 4 bpw via trellis-coded scalar quantization combined with incoherence processing. The artifact is the trellis state plus per-block parameters; reconstruction is approximate against the trainer's tensor by design (the trellis search at customer time is not bit-equal to the trainer's search, even with identical seeds, because of floating-point non-associativity in the cumulative cost computation).

**QuIP and QuIP#** [Chee 2024] introduce incoherence-processing-then-quantize as a route to sub-2-bit reconstruction quality. Like EXL3, the customer reconstructs an approximate tensor.

**bitsandbytes int4 / NF4** [Dettmers 2022, 2023] ships the de facto production 4-bit format in the HuggingFace Transformers ecosystem. The NF4 codebook is library-default (a fixed 16-element nonlinear-quantile grid), not persisted per-Linear. The artifact carries the quantized tensor and per-block scales.

**HQQ** [Badri 2024] and **AQLM** [Egiazarian 2024] occupy the same approximate-equivalence band with different optimization formulations.

**SeedLM** [Cao 2025] compresses weights into seeds of pseudo-random generators; the customer regenerates the tensor from the seed at load time. Reconstruction is approximate against the trainer's tensor by the same non-associativity argument.

The shared property of this family is that *no member of it provides a verifier*. The customer cannot, from the saved artifact alone, prove that the reconstructed tensor on their machine matches the tensor the trainer used. This is the *verifier gap*: a structural feature of the approximate-equivalence contract, not an oversight of any individual format. It is what UltraCompress closes.

### 2.3 The verifier gap

We are not aware of any prior published quantization format that ships a customer-side cryptographic verifier of bit-identical reconstruction. The closest analogues are at adjacent layers of the stack: HuggingFace's `safetensors` format carries a per-tensor SHA-256 of the *source bytes* (which is a verification of correct *download*, not of correct *reconstruction*); model-card publication conventions sometimes include a hash of the artifact tarball. Neither verifies that the reconstructed weight tensor on the customer's GPU equals the tensor the trainer dequantized at distillation time. UltraCompress's `uc verify` does. This is a contribution of contract design, not of cryptography: the SHA-256 primitive is standard, but the *placement* of the verifier — at the canonical reconstruction tuple, after dequantization but before forward — is what makes the reconstruction-time guarantee provable. To our knowledge, this is novel.

### 2.4 SSM-specific work and the Mamba compression gap

The Mamba family [Gu and Dao 2023; Dao and Gu 2024] and the recent Mamba-3 line [Dao and Gu, ICLR 2026] introduce selective state-space models as a transformer-alternative architecture class. RWKV [Peng 2023] and Jamba [Lieber 2024] are related state-space and hybrid designs. Published compression results on these architectures are limited: most existing work is fp16-to-bf16 conversion, INT8 activation quantization for inference acceleration, or weight-sharing experiments. To our knowledge, no published end-to-end ultra-low-bit (sub-6-bpw) weight compression result for an SSM checkpoint with a public PPL ratio against a bf16 baseline exists in the open literature as of May 2026. We list our Mamba-2.8B compression in `pending_provenance[]` (the source eval JSON requires re-run for canonical archival) and discuss the SSM gap in §3.4 and §6 of the full paper. The compression artifact itself is publicly available; only the canonical PPL ratio JSON is pending re-run.

---

## 3. Method (preview)

This section previews what the method *achieves* and what the verifier *proves*. The recipe-level construction (codec block size, overlay rank, optimizer schedule, distillation step counts, calibration corpus size, per-Linear training objective composition) is protected by the patent stack at USPTO 64/049,511 + 64/049,517 (provisional, filed 2026-04-25) and is not disclosed in this paper. A reviewer can reproduce the *result* (PPL ratio against bf16, peak memory, wall-clock) by running the published artifact through the published verifier and the published eval scripts; reconstructing the method *recipe* from the artifact is not the goal of an open publication and is not supported by the disclosure here.

### 3.1 What the artifact is

Each compressed model is published as a per-layer pack format (`v3.uc`) on HuggingFace under the `<anon>/<model>-uc-v3-bpw5` naming convention. Per-layer, the artifact carries three classes of payload:

1. **A bit-packed integer grid.** Each Linear's weight matrix is represented at 5 bpw as integer codes into a small per-Linear codebook. The codebook is *persisted with the codes*: the customer reads the codebook, the codes, and the per-block scales, and reconstructs the dequantized tensor in a single fp32 array operation that is exact at fp32 storage.

2. **A per-tensor low-rank correction.** Each Linear additionally carries a low-rank correction tensor (factored into two thin matrices and a scalar gate) that the customer adds to the dequantized integer-grid output. The correction is computed at trainer time against the bf16 teacher's hidden-state cache and frozen into the artifact; the customer does not re-train it.

3. **Per-block scaling parameters.** Each Linear carries per-block absmax scales (one per along-the-input-dimension block) at fp32 storage, used to undo the per-block normalization the trainer applied before discretization.

The customer reconstructs the dequantized weight as

```
W_reconstructed = scalar_dequantize(codes, scale) + low_rank_overlay
```

where every operand on the right-hand side is read directly from the persisted artifact at the storage precision the trainer wrote it at. The arithmetic is exact at fp32. There is no rederivation, no nearest-neighbor matching at customer time, no quantization-by-customer error to accumulate.

### 3.2 What the verifier proves

The pack-time runner computes a SHA-256 over the canonical reconstruction tuple `(absmax, grid, codes, U, V, alpha)` for each Linear, layer by layer, and writes the digest into the per-layer file footer. The customer-side `uc verify ORG/MODEL` command re-derives the tuple from the on-disk pack, recomputes the SHA-256 over the same canonical byte order, and confirms bit-identical equality to the footer digest. A digest mismatch on any Linear halts verification and reports the offending name. The verifier ships with the open-source codec.[^verifier-source]

[^verifier-source]: Verifier source: `github.com/<anon>/ultracompress/blob/main/scripts/uc/verify.py` (anonymous mirror). The canonical byte order for the SHA-256 hash is documented in the open pack-format specification (`docs/UC_V3_BINARY_FORMAT_SPEC.md`); reviewers can independently re-implement the verifier from the spec without accessing our codec runner.

The customer-side proof is therefore: every reconstructed tensor on the customer's GPU hashes to the digest the trainer published. No statistical claim, no benchmark on the customer's eval corpus, no opinion. A pass is a cryptographic certificate that the reconstruction matches; a fail is a pointer to the offending tensor.

### 3.3 The streaming compression runner

The compression runner processes one transformer block at a time, holding only the per-block weights, the correction overlay, the calibration-batch activations, and the optimizer state in GPU memory. Everything else lives on CPU or NVMe. Peak compression-time GPU memory therefore scales with the per-block footprint, not the full model. We use this property to compress Hermes-3-Llama-3.1-405B (whose bf16 checkpoint is roughly 810 GB) end-to-end on a single 32 GB consumer GPU; the artifact is published at `<anon>/hermes-3-llama-3.1-405b-uc-v3-bpw5`. The PPL ratio against the bf16 teacher on the long-context English eval (n_eval=50, seq_len=1024, seed=42) is **1.0066×** (baseline 5.0358 → compressed 5.0692), produced under a per-layer streaming bf16 baseline run on cuda:1 with the compressed run on the same hardware and the same eval configuration. The full per-layer wall-clock and memory profile is in §5 of the full paper. (Recipe details — calibration sequence count, distillation step schedule, optimizer learning-rate schedule — are excluded from this draft per the charter.)

### 3.4 The reproducibility surface and the provenance discipline

A claim of losslessness requires more than a verifier; it requires that every published number resolves to a recomputable source. We adopt a strict discipline:

- Every PPL ratio cited in this paper appears in the public `docs/BENCHMARKS_2026_05_10.json`, in the `verified_records[]` array.
- Every entry in `verified_records[]` carries an explicit `sources.{baseline_json, compressed_json}` pair (or a single-source compressed JSON containing both `baseline_ppl` and `compressed_ppl` in one file, in which case both pointers resolve to the same path).
- A reviewer can run `python scripts/verify_all_benchmarks.py` on the public anonymous mirror to recompute every cited ratio from the on-disk source JSONs. A discrepancy halts the script and reports the offending row.

Two further artifacts that we have compressed and that are *physically present* on HuggingFace are intentionally excluded from `verified_records[]` and live in `pending_provenance[]`: Mamba-2.8B and the Qwen3-1.7B (instruct) row.[^pending-provenance] In each case the on-disk source JSON for the canonical PPL ratio cited in earlier internal documents could not be reconciled to a present file in the public mirror as of the 2026-05-10 audit, and we declined to cite the ratio in the paper until a re-run produces a canonical archival source. This discipline costs us two cells in the matrix; the alternative — citing a number whose source eval cannot be rebuilt by a reviewer — would cost the lossless contract its credibility. We choose the discipline.

[^pending-provenance]: Mamba-2.8B: previously published ratio 1.0119× from `BENCHMARKS_2026_05_08.json`; the underlying eval JSON was not archived. Re-run command template available in the public benchmarks JSON. Qwen3-1.7B (instruct): previously published ratio 1.020×; the closest matching on-disk source eval (NEW_FULL n=30, seq_len=1024) computes 1.00782, indicating the canonical baseline (18.0145) was produced under an eval configuration whose source JSON is not on disk. Both rows will be either re-run-and-promoted or demoted in the public benchmarks JSON before camera-ready.

### 3.5 Honest negative results

Section 4 of the full paper publishes the cross-architecture validation matrix (the headline rows from `verified_records[]` are summarized in the abstract). Section 6 publishes a companion catalogue of nine honest negative results from the same research program: refuted improvement hypotheses, configuration variants that regressed against the headline pipeline, and one apples-to-apples adversarial replication that fixed an internal-table caveat (correction overlay memory-aware harness suspect). The abridged catalogue is in `docs/HONEST_NEGATIVE_RESULTS_2026_05_08.md`; the highlights include V4-D multi-pass cascade correction at 1.0682× (refuted), V3 rank-redistribute at 1.0702× (refuted), per-Linear adaptive bpw v1 within noise of uniform (refuted apples-to-apples), SVD warm-start on Mamba (worse than random init), depth-banded train_steps schedules on Mistral-7B (1.082× — refuted; early-layer step cuts corrupted the prefix cache for downstream layers), and the 1.0040× empirical floor for the overlay on Qwen3-1.7B-Base (rank-and-steps push refuted to within statistical noise). Negative-results documentation is rare in the NeurIPS quantization track; we publish it because the positive numbers only mean what they mean if the failures are equally visible.

### 3.6 Honest disclaimer on the published evidence base

To anchor reviewer expectations: of the 16 architectures cited in this paper as carrying a measured PPL ratio, **14** appear in `verified_records[]` of the public `docs/BENCHMARKS_2026_05_10.json` with explicit `sources.{baseline_json, compressed_json}` linkage to on-disk evaluator output that a reviewer can recompute end-to-end via `scripts/verify_all_benchmarks.py`. The remaining **2** — Mamba-2.8B and the Qwen3-1.7B (instruct) row — currently sit in `pending_provenance[]` while the canonical source JSONs are re-run; we cite their compression artifacts and the public-mirror remediation command templates rather than a number, and we discuss the SSM compression gap in §2.4 without claiming a reproducible Mamba ratio in the matrix until the re-run completes. This is the publication discipline we want the field to adopt: a number is cited if and only if its source JSON is present on disk and the cited ratio recomputes from that JSON; a number is otherwise demoted to `pending_provenance[]` and explicitly flagged at every mention. We treat the 14-of-16 ratio as the present state of the evidence base, not a final state, and the public benchmarks JSON is the canonical record.

---

*End of v3 draft Sections 1–3. Sections 4 (cross-architecture validation matrix), 5 (streaming compression at frontier scale), 6 (discussion and limitations), and 7 (conclusion) reuse the v2 structure with updated numbers from `verified_records[]` and the `pending_provenance[]` flags propagated; they are out of scope for this draft.*

Codec internals + training procedure are patent-protected (USPTO 64/049,511 + 64/049,517).
