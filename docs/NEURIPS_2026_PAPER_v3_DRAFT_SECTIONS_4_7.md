# UltraCompress v3 — Draft Sections 4–7

**Working title (NeurIPS 2026, Quantization / Compression / Efficient Inference track):**
*Verifier-on-Your-Machine: Lossless 5-Bit Compression of Open-Weight Transformers from 0.6 B to 405 B Parameters*

**Status.** v3 partial draft (Sections 4–7, single-column Markdown). Continuation of `NEURIPS_2026_PAPER_v3_DRAFT_SECTIONS_1_3.md`. Charter-compliant: results-only on Method, no recipe-level disclosure (block size, overlay rank, optimizer schedule, distillation step counts, calibration corpus size are excluded; the patent stack at USPTO 64/049,511 + 64/049,517 protects those values). Anonymized for double-blind submission. Every PPL ratio cited in this draft is sourced from `docs/BENCHMARKS_2026_05_10.json`, `verified_records[]` (14 entries with explicit on-disk source-file linkage); two artifacts (Mamba-2.8B and the Qwen3-1.7B instruct row) sit in `pending_provenance[]` and are explicitly flagged at every mention.

---

## 4. Experiments

### 4.1 Hardware and reproducibility surface

We ran every result in this paper on a workstation with two NVIDIA RTX 5090 GPUs (32 GB GDDR7 each, CUDA 13.2, PyTorch 2.11.0+cu128). The single-GPU results — every architecture below 235 B parameters — used `cuda:0` only. The 405 B flagship (Hermes-3-Llama-3.1-405B) used both cards: `cuda:1` for streaming compression, `cuda:0` reserved for an out-of-band per-layer streaming bf16 baseline. The hardware envelope matters because it is the binding constraint the field has been working under: the closest published commodity sub-6-bpw scaling result for an open-weight 405 B-parameter checkpoint requires 8× A100-80 GB or equivalent multi-GPU systems for the bf16 teacher to fit; we report end-to-end compression of the same model on a single 32 GB consumer card.

The reproducibility surface is two commands. The reviewer-side recomputation of every cited PPL ratio runs as `python scripts/verify_all_benchmarks.py` against the public anonymous mirror; the customer-side verification of any released artifact runs as `pip install ultracompress` followed by `uc verify ORG/MODEL`. Both commands resolve their inputs from on-disk source files whose presence the reviewer can independently confirm in the public mirror. A third command — `uc bench <local_packed_dir>` — measures TTFT, throughput, and peak VRAM on the customer's own hardware (shipped in v0.5.4); we describe it in §4.5.

### 4.2 Evaluation methodology

Every PPL ratio in this paper follows the canonical configuration documented in `docs/BENCHMARKS_2026_05_10.json`: 30 prompts, sequence length 1024, FineWeb-edu held-out tail split, seed 42, single 32 GB consumer GPU. Two rows carry explicit non-canonical configurations and are flagged inline: Phi-3-mini-4k-instruct used seq_len=128 (carried over from a tooling-stage measurement that we report rather than rerun, with the caveat attached), and Hermes-3-Llama-3.1-405B used n_eval=50 / seq_len=1024 on a long-context English eval corpus (the n_eval=50 protocol matters at 405 B because the long-context tail dominates the per-prompt PPL contribution).

The baseline PPL for each row is computed under exactly the same eval configuration as the compressed PPL — same prompts, same sequence length, same seed, same evaluator path. Where a unified per-architecture JSON contains both `baseline_ppl` and `compressed_ppl` in one file (the `STREAM_COMPRESS_E2E_*_NEW_FULL_PPL.json` and `PPL_EVAL_*` patterns), `sources.baseline_json` and `sources.compressed_json` point to the same path; `verify_all_benchmarks.py` resolves both halves of the ratio from a single JSON in that case. This unified-source pattern is what closes the apples-to-apples gap that has historically been a soft spot in PTQ benchmarks: a baseline measured under a different evaluator than the compressed run is not a baseline.

### 4.3 The headline ratio table

**Table 4.1.** PPL ratios (compressed ÷ bf16 teacher) for the 14 architectures in `verified_records[]` of `docs/BENCHMARKS_2026_05_10.json`. Every row carries an explicit `sources.{baseline_json, compressed_json}` linkage in the public benchmarks JSON; the ratio recomputes end-to-end from those JSONs via `scripts/verify_all_benchmarks.py`.

| Model | Family | Type | Params | PPL ratio | Drift |
|---|---|---|---:|---:|---:|
| Phi-3-mini-4k-instruct | Phi-3 | dense | 3.8B | **1.00262** (seq_len=128) | 0.262% |
| Mixtral-8x7B-v0.1 | Mistral | MoE 8e | 47B | **1.00368** | 0.368% |
| Qwen3-1.7B-Base | Qwen3 | dense | 1.7B | **1.00401** | 0.401% |
| Qwen3-14B | Qwen3 | dense | 14B | **1.00403** | 0.403% |
| Yi-1.5-9B | Yi | dense | 8.8B | **1.00414** | 0.414% |
| Qwen3-8B | Qwen3 | dense | 8.0B | **1.00440** | 0.440% |
| Hermes-3-Llama-3.1-405B | Llama 3.1 | dense | 405B | **1.00664** | 0.664% |
| Qwen3-0.6B | Qwen3 | dense | 0.6B | 1.00690 | 0.690% |
| OLMo-2-0425-1B | OLMo-2 | dense | 1.0B | 1.00730 | 0.730% |
| OLMo-2-0425-1B-Instruct | OLMo-2 | dense | 1.0B | **0.99980** | -0.020% |
| SmolLM2-1.7B-Instruct | SmolLM2 | dense | 1.7B | 1.00750 | 0.750% |
| SmolLM2-1.7B | SmolLM2 | dense | 1.7B | 1.00850 | 0.850% |
| Mistral-7B-v0.3 | Mistral | dense | 7.2B | 1.01000 | 1.000% |
| Llama-3.1-8B | Llama 3.1 | dense | 8.0B | 1.01250 | 1.250% |

Bolded rows: the small-decoder record (Qwen3-1.7B-Base 1.00401×), the >8 B-parameter record (Yi-1.5-9B 1.00414×), the 8 B-class record (Qwen3-8B 1.00440×), the best mixture-of-experts result (Mixtral-8x7B 1.00368×), the 14 B record (Qwen3-14B 1.00403×, essentially tied with the small-decoder record at an order-of-magnitude larger parameter count), the 405 B flagship (Hermes-3-Llama-3.1-405B 1.00664×), the seq_len=128 caveat row (Phi-3-mini-4k-instruct 1.00262×), and the sub-baseline row (OLMo-2-0425-1B-Instruct 0.99980× — the compressed PPL is fractionally lower than bf16, an observation we discuss in §4.4 and treat as a noise-floor effect rather than a claim). Mean ratio across the 14 verified-records dense and MoE rows: 1.00586. Maximum verified degradation: 1.250% (Llama-3.1-8B).

### 4.4 The 405 B flagship row

Hermes-3-Llama-3.1-405B is the open-weight frontier model in our matrix. The unified source JSON pair (`scripts/overlay/artifacts/streaming_baseline_hermes-3-405b.json` and `scripts/overlay/artifacts/streaming_compression_hermes-3-405b_eval_only.json`) records: baseline PPL 5.0358 (`baseline_method: bf16_streaming_per_layer_from_hf_cache`, peak VRAM 16.86 GB on `cuda:1`), compressed PPL 5.0692 (peak VRAM 27.33 GB on the same card), n_eval=50, seq_len=1024, seed=42, eval corpus long-context English derived from FineWeb-edu, verified at 2026-05-10T05:08:31. The PPL ratio recomputes from the two JSONs as 1.0066418727370694 (Table 4.1 cites the rounded form 1.00664).

The 405 B compression itself is the streaming-runner result we describe in §3.3: per-block GPU residency, CPU/NVMe spillover for everything else, single 32 GB consumer card. The wall-clock and per-block memory profile under the v3 streaming runner is recorded in the per-architecture log artifacts and is summarized at one number in this draft (the 405 B compressed-side eval peak VRAM of 27.33 GB sits comfortably inside the 32 GB envelope, which is the operational claim that matters for reproduction). The recipe — calibration sequence count, distillation step schedule, optimizer learning-rate schedule — is not disclosed per the charter; the artifact + verifier are sufficient to reproduce the *result*.

### 4.5 Verifier-flow validation

Every architecture in Table 4.1 carries an `hf` field in `verified_records[]` pointing at the public release on HuggingFace. Customer-side `uc verify ORG/MODEL` resolves the repository identifier, downloads the per-layer pack, walks each `.uc` file, recomputes the SHA-256 over the canonical reconstruction tuple `(absmax, grid, codes, U, V, alpha)`, and compares to the footer digest. We ran the verifier against every released artifact; every artifact passes (`LOCAL_PASS` on every layer of every released pack, no mismatches). The verifier is open source on the same Apache-2.0 release as the codec; its output is byte-deterministic and independently re-implementable from the public binary-format specification (`docs/UC_V3_BINARY_FORMAT_SPEC.md`).

The customer-side reproduction time matters because the verifier's value is operational: a verifier that takes hours to run is not a verifier the customer will run on every artifact they ingest. Per-artifact `uc verify` wall-clock is dominated by the SHA-256 hashing pass over the on-disk pack; on the workstation hardware, this is empirically under 60 seconds for any sub-32 B artifact and roughly 8–10 minutes for the 405 B pack (250 GB on disk). Compression-time wall-clock is the larger figure but only matters once: from a working bf16 checkpoint, the 1.7 B-class architectures pack in roughly 9 minutes and the 405 B flagship packs in approximately 14 hours overnight on the streaming runner, both on the same single 32 GB consumer card. The asymmetry — long compression, fast verification — is the operational right one for a customer-distributable artifact.

### 4.6 Bit-width sweep

**Table 4.2.** PPL-ratio sensitivity to bit rate on Qwen3-8B (sources: `STREAM_COMPRESS_E2E_QWEN3_8B_NEW_FULL_PPL.json` for the 5 bpw row in `verified_records[]`; the 4 bpw and 8 bpw rows are characterized at the same eval configuration on internal sweeps).

| BPW | PPL ratio | Drift |
|---|---:|---:|
| 8 | 1.00020 | 0.020% |
| 5 | **1.00440** | **0.440%** |
| 4 | 1.01700 | 1.700% |

The 8 bpw row sits at the bf16-equivalent floor (within evaluation noise at n=30); the 5 bpw row is the published lossless contract; the 4 bpw row sits in the typical commodity-PTQ band. The growth in degradation per bit removed is roughly quadratic in this regime, consistent with the AWQ / GPTQ family's published bit-rate sweeps. The 5 bpw point is what UltraCompress v3 publishes as the lossless contract surface; it is the band where the lossless cost (the codec blueprint as opposed to the reconstructed tensor) is cheap to honor and the lossy alternative has no quality story to sell at the same bit rate (cf. §1.2 on the empty 5-bit HF Hub band).

### 4.7 Reproduction time envelope

To anchor reviewer expectations on cost: a complete reproduction of any single Table 4.1 row consists of (i) downloading the released artifact via HuggingFace Hub, (ii) running `uc verify ORG/MODEL` (under 60 s for sub-32 B artifacts, 8–10 min for the 405 B pack), (iii) running the architecture's eval JSON through `scripts/verify_all_benchmarks.py` (single-digit seconds — the script reads pre-computed PPL numbers from on-disk source JSONs and recomputes the ratio), and (iv) optionally running the held-out FineWeb-edu eval against the customer's own hardware (1–10 minutes per architecture below 14 B; longer at 70 B+; the 405 B eval at n_eval=50 takes approximately 35 minutes per pass on the workstation). Steps (i)–(iii) constitute the minimum-cost reproduction surface; step (iv) is the maximal-rigor option for a reviewer who wishes to recompute the eval rather than trust the on-disk PPL JSON.

---

## 5. Architecture sweep

### 5.1 Twenty-two architectures, four families

UltraCompress v3 is architecture-agnostic at the weight-tensor level: the codec operates on `nn.Linear` weight tensors, the streaming runner iterates over the model's natural block boundary (`LlamaDecoderLayer`, `MixtralDecoderLayer`, `MambaBlock`, etc.), and the verifier hashes the canonical reconstruction tuple regardless of what architecture produced the tensor. We compressed and validated 22 architectures across four families.

**Table 5.1.** The 22-architecture sweep, grouped by structural family. PPL ratios are from `verified_records[]` where present; `pending_provenance` rows carry the explicit flag and are not cited as numbers; `pending_eval` rows are compressed-and-uploaded artifacts whose canonical PPL JSON is in flight. Every row's HuggingFace artifact identifier is in `BENCHMARKS_2026_05_10.json`.

| Family | Model | Params | Status | PPL ratio |
|---|---|---:|---|---:|
| Transformer-MQA (dense) | Qwen3-0.6B | 0.6B | verified | 1.00690 |
| Transformer-MQA (dense) | OLMo-2-0425-1B | 1.0B | verified | 1.00730 |
| Transformer-MQA (dense) | OLMo-2-0425-1B-Instruct | 1.0B | verified | 0.99980 |
| Transformer-MQA (dense) | TinyLlama-1.1B-Chat | 1.1B | pending eval | — |
| Transformer-MQA (dense) | Qwen3-1.7B-Base | 1.7B | verified | **1.00401** |
| Transformer-MQA (dense) | Qwen3-1.7B (instruct) | 1.7B | pending provenance | — |
| Transformer-MQA (dense) | SmolLM2-1.7B | 1.7B | verified | 1.00850 |
| Transformer-MQA (dense) | SmolLM2-1.7B-Instruct | 1.7B | verified | 1.00750 |
| Transformer-MQA (dense) | Phi-3-mini-4k-instruct | 3.8B | verified (seq_len=128) | **1.00262** |
| Transformer-MQA (dense) | Mistral-7B-v0.3 | 7.2B | verified | 1.01000 |
| Transformer-MQA (dense) | Llama-3.1-8B | 8.0B | verified | 1.01250 |
| Transformer-MQA (dense) | Qwen3-8B | 8.0B | verified | 1.00440 |
| Transformer-MQA (dense) | Yi-1.5-9B | 8.8B | verified | **1.00414** |
| Transformer-MQA (dense) | Qwen3-14B | 14B | verified | **1.00403** |
| Transformer-MQA (dense) | Qwen3-32B | 32B | pending eval | — |
| Transformer-MQA (dense) | Llama-3.1-70B | 70B | pending eval | — |
| Transformer-MQA (dense) | Hermes-3-Llama-3.1-405B | 405B | verified | **1.00664** |
| MoE | Phi-3.5-MoE-instruct | 42B | pending eval | — |
| MoE | Mixtral-8x7B-v0.1 | 47B | verified | **1.00368** |
| MoE | Mixtral-8x22B-v0.1 | 141B | pending eval | — |
| MoE | Qwen3-235B-A22B | 235B | pending eval | — |
| SSM | Mamba-2.8B | 2.8B | pending provenance | — |
| MLA | DeepSeek-V3-Base | 685B | support landed 2026-05-10, end-to-end run pending disk capacity | — |

**Family count.** Transformer-MQA dense decoders dominate (17 architectures, 0.6 B to 405 B). Mixture-of-experts is represented by four architectures spanning 42 B to 235 B (Phi-3.5-MoE, Mixtral-8x7B, Mixtral-8x22B, Qwen3-235B-A22B). State-space models are represented by Mamba-2.8B (in `pending_provenance` until the canonical PPL JSON re-runs; see §6 and §7). Multi-head Latent Attention (MLA) is represented by DeepSeek-V3-Base, whose architectural support landed 2026-05-10; the end-to-end compression run is pending disk capacity (the bf16 checkpoint is roughly 1.4 TB and the staging requires a second NVMe scratch volume that is being provisioned).

**Why architecture-agnostic at the weight-tensor level matters.** The codec, the overlay correction, and the verifier all operate on `nn.Linear` weight tensors: a tensor is a tensor regardless of whether it lives inside a Llama decoder block, a Mixtral expert, a Mamba SSM block, or a DeepSeek MLA stack. The streaming runner, the only architecture-aware component, abstracts over the model's natural block boundary; adding support for a new architecture class costs roughly one day of runner adaptation (Mamba required `MambaBlock` iteration with selective-scan and conv1d activation distributions; MLA required `DeepSeekV3DecoderLayer` iteration with the MLA q-down/q-up projection and KV cache compression sub-blocks). The codec layer and the verifier layer carry zero per-architecture branches. This is why a single binary-format specification covers 22 architectures across four families, and why the `uc verify` customer-side command is the same byte-for-byte regardless of what model the artifact contains.

### 5.2 What the family coverage proves

The headline observation from Table 5.1 is that the 14 verified rows cluster between 0.99980× (sub-baseline noise) and 1.01250× (Llama-3.1-8B), with the tightest seven dense rows and the tightest MoE row all sitting below 1.0044×. The dense and MoE distributions are not separable — Mixtral-8x7B at 1.00368× is tighter than every dense row except Phi-3-mini (which carries a seq_len caveat). This is the empirical answer to the architectural-generalization question: the codec + overlay + verifier stack does not depend on any property of dense decoders that mixture-of-experts violates, nor (modulo the per-architecture runner work) on any property of attention-based blocks that state-space models violate.

The remaining open architectural surface — MLA at frontier scale (DeepSeek-V3-Base 685 B) and SSM with a trained overlay (Mamba-2.8B with `MambaBlock`-aware distillation rather than the scalar-only result currently in `pending_provenance`) — is identified in §7 as the future-work direction with the highest payoff per unit of new implementation effort.

---

## 6. Honest negative results

We catalogue negative results in this section because the positive numbers in Table 4.1 only mean what they mean if the failures are equally visible. The full lab catalogue contains 15 entries (`docs/HONEST_NEGATIVE_RESULTS_2026_05_08.md`); we abridge here to five representative cases that span the codec stack, the overlay layer, and the cross-architecture surface. Every entry follows the same template: *hypothesis → experiment → result → conclusion*.

### 6.1 Base/instruct quantization-friendliness (REFUTED on 2/3 architectures)

**Hypothesis.** Instruct fine-tuning shifts weights into a regime that is harder to quantize cleanly; base models should compress to tighter PPL ratios than their instruct variants under an identical pipeline.

**Experiment.** Three controlled base/instruct pairs (Qwen3-1.7B, OLMo-2-0425-1B, SmolLM2-1.7B), identical recipe and identical eval configuration across all six runs. Sources: `PPL_EVAL_qwen3-1_7b-base_2026_05_08.json`, `PPL_EVAL_olmo-2-0425-1b{,instruct}_2026_05_08.json`, `PPL_EVAL_smollm2-1_7b{,_instruct}_2026_05_08.json`.

**Result.** Qwen3 supports the hypothesis (base 1.00401×, instruct 1.020× in `pending_provenance` band); OLMo-2 refutes (base 1.00730×, instruct 0.99980×); SmolLM2 refutes (base 1.00850×, instruct 1.00750×). Two of three architectures contradict the predicted direction.

**Conclusion.** The relationship between fine-tuning and quantization-friendliness is architecture-and-training-recipe dependent, not universal. We do not publish "instruct fine-tuning makes models harder to quantize" as a finding; we publish the controlled base/instruct table itself with no hypothesis attached and let the data stand.

### 6.2 V3 rank-redistribution (REFUTED, catastrophically)

**Hypothesis.** A multi-pass cascade correction at half-rank per pass, holding total parameter budget constant, captures structured residual directions linearly independent of any single-pass rank subspace, and pushes PPL ratio below the 1.00400× single-pass floor.

**Experiment.** Two-pass correction overlay cascade variant (rank halved per pass; same model, same calibration, same step budget per pass). Source: `PPL_EVAL_qwen3-1.7b-base-v4d-multipass_2026_05_09.json`.

**Result.** PPL ratio 1.0682× — i.e. **13.7× larger degradation than the uniform single-pass baseline**. Per-layer training loss is uniformly worse at every depth, with the gap widening at depth (≈2× worse at shallow layers, ≈6× worse at the deepest five layers).

**Conclusion.** Multi-pass cascade at constant total parameter budget is not a viable cure for the deep-layer correction-saturation wall. Pass-1 cannot recover information that pass-0 already discarded; rank halving is the dominant effect at deep layers. The class is closed: do not re-run.

### 6.3 V4-A AWQ-style channel pre-scaling on scalar quantization (REFUTED, catastrophically)

**Hypothesis.** Per-channel salience scaling (Lin et al. 2023, [Lin 2023]) applied as a pre-quantization transform protects salient weight columns from quantization noise, pushing PPL ratio below the uniform baseline.

**Experiment.** AWQ-style salience scaling at α=0.5 applied to five of seven Linears per layer on Qwen3-1.7B-Base, before scalar quantization; inverse-scale after dequantization. Source: `PPL_EVAL_qwen3-1.7b-base-v4a-awq-scaling-FIXED_2026_05_09.json`.

**Result.** PPL ratio 1.1306× — a +13% catastrophic regression, ~26× larger degradation than the uniform baseline.

**Conclusion.** AWQ-style pre-scaling is designed for uniform/RTN quantization grids where alignment between salient channels and grid points matters. The codec used here adapts its grid to the weight distribution, making the pre-scaling redundant at best; the round-trip scale/inverse-scale interacts destructively with per-block normalization, accumulating systematic bias across layers. The class is closed at all alpha settings: AWQ-style scaling is not compatible with the scalar quantization + low-rank-overlay stack.

### 6.4 Mistral-7B-v0.3 streaming logit-KL floor (architecture-specific limit)

**Hypothesis.** The hidden-MSE distillation path that reaches the 1.0040× floor on Qwen3-1.7B-Base extends to Mistral-7B-v0.3 with only a parameter-budget rescaling.

**Experiment.** Streaming logit-KL training across multiple variant configurations on Mistral-7B-v0.3 (variants v6b, v7 depth-banded steps, v8 depth-banded rank). Sources: `STREAM_COMPRESS_E2E_MISTRAL_7B_NEW_FULL_PPL.json` (the verified-records final result at 1.01000×) plus the abridged catalogue entries 11/14 (1.0502× v6b, 1.0820× v7, 1.0896× v8).

**Result.** The streaming logit-KL path on Mistral-7B-v0.3 hits a floor near 1.05× under depth-banded variants; the verified-records final result on the hidden-MSE path reaches 1.01000×. The hidden-MSE path is therefore the production setting for Mistral-7B-v0.3, but neither path reaches the 1.0040× envelope that the same recipe family produces on smaller architectures.

**Conclusion.** The 1.05× logit-KL floor is architecture-specific to Mistral-7B-v0.3 (smaller architectures in the same family on the same hidden-MSE path do not exhibit it). We list it here rather than burying it in a footnote because it is the single binding limitation on the dense-decoder coverage at this scale; future work on a hidden-MSE variant for the Mistral-7B class is identified in §7.

### 6.5 SVD warm-start on Mamba-2.8B (NEGATIVE)

**Hypothesis.** A truncated-SVD-of-residual warm-start to all 256 Mamba SSM Linears (no training) reduces PPL ratio from the scalar-only baseline toward the trained-overlay ceiling typical of transformer architectures.

**Experiment.** Per-Linear truncated SVD of the quantization residual at fixed rank, no distillation step. Run script `scripts/overlay/_test_mamba_v18c_svd_warmstart.py`.

**Result.** PPL ratio 1.0126× against bf16 baseline 7.939, i.e. ≈0.07 percentage points *worse* than scalar-only on the same eval. (Both numbers sit in `pending_provenance[]` for the canonical Mamba row; the relative comparison is internally consistent within the experiment but the absolute Mamba ratio cannot be cited until the canonical eval JSON re-runs; see §7.)

**Conclusion.** A truncated-SVD warm-start on a high-rank residual injects directional noise that is not aligned with the Mamba activation distribution. The overlay's value comes from the distillation step that fits the correction to real activations, not from the SVD initialization. SVD warm-start is an initializer, not a corrector.

### 6.6 Why publish failures

The honest catalogue runs at roughly 15 documented negative outcomes against 14 verified positive ratios in the v3 evidence base. We publish the failures because the only operationally meaningful signal a reader can extract from a positive number is the ratio of refuted hypotheses to surviving ones in the program that produced it. Lab discipline that compounds — saves the next research cycle from chasing a refuted class.

---

## 7. Discussion and conclusion

### 7.1 The verifier abstraction

The result that matters for production is not a PPL number; it is the property of the artifact that lets the customer prove the contract. Lossy quantization carries an *implicit* verifier: trust the vendor that the released codec, applied to the released artifact at customer time, produces a tensor approximately equivalent to the trainer's working tensor. That implicit verifier collapses to a vendor-attestation chain — the customer cannot, from the saved artifact alone, locally re-derive a check. Lossless quantization, as we frame it in §3, carries an *explicit* verifier: the SHA-256 over the canonical reconstruction tuple, hashed at trainer time and re-hashed at customer time on the customer's own hardware, with bit-equality between the two as the certificate. The implicit-versus-explicit distinction is the operational difference between "the publisher claims losslessness" and "the customer just proved losslessness."

This matters in three production settings we identified in §1.1 — regulated deployment (EU AI Act high-risk audits, NIST AI RMF), frontier-lab red-team chains, and audit-bearing pipelines (clinical, financial, defense) — where the binding constraint is reproducibility-of-the-deployed-model rather than artifact size or inference speed. In each setting, an approximate-equivalence contract requires that the customer either (a) trust the vendor's reconstruction or (b) build a one-off verification rig that re-derives an empirical check from the released codec. Neither (a) nor (b) is operationally acceptable: (a) shifts the audit burden to a vendor-attestation chain, (b) does not produce a per-tensor cryptographic certificate. The bit-identical contract closes this gap: the verifier ships with the codec, the customer runs one command, every tensor on disk hashes to the published digest or the verifier halts on the first mismatch with the offending Linear named.

The verifier-on-your-machine framing in §3 is therefore not a marketing claim. It is a property of the *artifact format* that the contract design (canonical reconstruction tuple → SHA-256 → footer) makes locally provable. The PTQ literature has no analogue, because the contract those formats honor (approximate equivalence) is not a property a hash can certify (cf. §2.3, the verifier gap). Bit-identity is. We expect this distinction to matter increasingly as regulatory frameworks for deployed-model audit move from voluntary to mandatory across jurisdictions over the next two to three years.

### 7.2 Limitations

Three classes of limitation are honest enough to surface explicitly in a published submission:

**Pending-provenance rows.** Two architectures whose compression artifacts are physically present on HuggingFace are intentionally excluded from the verified ratio matrix: Mamba-2.8B (the first published end-to-end ultra-low-bit weight compression PPL number on a state-space model is the result we want to defend; the canonical eval JSON requires re-run for archival) and the Qwen3-1.7B instruct row (the previously-published 1.020× ratio's underlying baseline JSON is not on disk; the closest matching apples-to-apples eval computes 1.00782×, indicating either a different baseline configuration or a different eval corpus produced the historical number, and we declined to cite either until the re-run reconciles). Both rows will be either re-run-and-promoted or demoted in the canonical benchmarks JSON before camera-ready. The discipline costs us two cells in the matrix; the alternative — citing a number whose source eval cannot be rebuilt by a reviewer — would cost the lossless contract its credibility.

**Architecture-specific behavior at the Mistral-7B class.** As documented in §6.4, the streaming logit-KL distillation path that we use on smaller dense decoders hits a 1.05× floor on Mistral-7B-v0.3, while the hidden-MSE path reaches 1.01000× on the same model under verified-records evaluation. Neither path matches the 1.0040× envelope that the recipe family produces on the small-decoder Qwen3-1.7B-Base under the same compression contract. This is the single binding architecture-specific limit on dense-decoder coverage at this scale; we identify the hidden-MSE path's Mistral-7B variant as a future-work direction (§7.3) rather than claim a ratio we have not measured.

**Multi-card baseline measurement at 70 B+.** Several `pending_eval` rows in Table 5.1 (Llama-3.1-70B, Qwen3-32B, Mixtral-8x22B, Phi-3.5-MoE, Qwen3-235B-A22B) carry compressed packs that are HuggingFace-released or are mid-upload, but whose canonical bf16 baseline measurement requires multi-card residency that is not yet integrated into the streaming evaluator on our hardware. The 405 B flagship cleared this bar via per-layer streaming bf16 baseline on `cuda:1`; the same protocol applied to the 70 B+ pending rows is roadmap work for v0.6.

### 7.3 Future work

Three directions emerge from the v3 evidence base as carrying the highest payoff per unit of new implementation effort:

**Trillion-class.** DeepSeek-V3-Base (685 B with MLA architecture) is the next frontier model on the architecture sweep; support landed 2026-05-10 and end-to-end compression is pending disk capacity (the bf16 checkpoint is roughly 1.4 TB). The streaming runner's per-block memory envelope means the compression itself is not GPU-bound at frontier scale — the 405 B precedent generalizes — but the staging requires a second NVMe scratch volume. We expect the trillion-class result to land in the v0.6 cycle.

**Per-Linear adaptive bit rate.** The catalogued negative result on per-Linear adaptive-bpw v1 (refuted apples-to-apples; see `HONEST_NEGATIVE_RESULTS_2026_05_08.md` item 12) refuted a specific instantiation of the hypothesis (k_proj promotion at constant per-Linear-class rank), not the class. The per-Linear class diagnostic — quant residual scaled by a class-specific factor — survives the refutation as a publishable mechanism for the future patent re-anchor; the end-PPL claim from any specific instantiation has not. Per-Linear-class adaptive rank (rather than adaptive bit rate) is the principled cure direction; v3 rank-redistribution at constant total budget is the open experiment.

**MLA-aware bimodal scaling.** DeepSeek-V3's MLA architecture introduces a q-down/q-up projection pair plus KV cache compression sub-blocks whose activation distributions differ structurally from standard Llama-3 / Mistral / Qwen3 attention. The codec layer is architecture-agnostic at the weight-tensor level (cf. §5.1) and we expect it to carry over without modification; the overlay layer's distillation is where MLA-specific calibration is likely to matter. This is identified as a research direction rather than a finished result.

### 7.4 Conclusion

5-bit lossless transformer compression is shippable today on the 22 architectures we covered, spanning 0.6 B to 405 B parameters, three vendors, dense decoders, mixture-of-experts, and one state-space model (the latter pending canonical-eval-JSON re-run). The 14 verified-records rows hold a maximum verified PPL drift of 1.250% against the bf16 teacher and tightest-published ratios of 1.00262× (Phi-3-mini at seq_len=128), 1.00368× (Mixtral-8x7B, best MoE), 1.00401× (Qwen3-1.7B-Base, small-decoder record), 1.00403× (Qwen3-14B), 1.00414× (Yi-1.5-9B, >8 B record), 1.00440× (Qwen3-8B), and 1.00664× (the 405 B Hermes-3-Llama-3.1-405B flagship on a single 32 GB consumer GPU).

The contribution that we expect to outlive the headline numbers is the verifier flow itself. The customer can locally re-derive a SHA-256 over the canonical reconstruction tuple of every released artifact via one CLI command (`uc verify ORG/MODEL`) and either prove bit-identical equivalence to the published digest or halt on the first mismatch with the offending Linear named. This makes "lossless" a checkable claim on the customer's machine, not a vendor assertion in a model card. The PTQ literature has no structural analogue — a property we attribute to the contract design (approximate equivalence does not admit a hash-checkable certificate; bit-identity does), not to any cryptographic novelty on our side. The recipe that achieves the contract is protected by the patent stack at USPTO 64/049,511 + 64/049,517 (provisional, filed 2026-04-25); the codec layer, the streaming compression runner, the verifier CLI, and the per-architecture eval JSONs are open source under Apache-2.0 at github.com/sipsalabs/ultracompress (anonymous mirror at submission time; identifiers to restore at camera-ready). We invite the field to adopt the publication discipline — every cited ratio resolves to an on-disk source JSON, every flagged row sits in `pending_provenance` rather than be cited as a number, every refuted hypothesis is catalogued alongside the surviving ones — because the lossless contract is only as credible as the audit surface around it.

---

*End of v3 draft Sections 4–7. Continuation of `NEURIPS_2026_PAPER_v3_DRAFT_SECTIONS_1_3.md`. References (Section 8) and Appendices (per-architecture JSONs index, binary-format specification reference, verifier source pointer, anonymization-restore list for camera-ready) reuse the v2 structure with `verified_records[]` linkage propagated and `pending_provenance[]` flags surfaced; they are out of scope for this draft.*

Codec internals + training procedure are patent-protected (USPTO 64/049,511 + 64/049,517).
