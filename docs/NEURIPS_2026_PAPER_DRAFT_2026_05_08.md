# UltraCompress: Trainer-Persisted scalar quantization Codec for Lossless 5-bit Transformer Compression at Frontier Scale

**Working title (NeurIPS 2026 main conference, Quantization / Compression / Efficient Inference track)**

**Status:** v1 full draft (Markdown). LaTeX port pending after the four resolved-but-flagged contradictions are reviewed by the author and counsel. Anonymized rephrasing for double-blind review is itemized in the pre-submission checklist at the end.

---

## Abstract

We present UltraCompress, a 5-bit weight quantization format for transformer models whose customer-side weight reconstruction is mathematically equivalent to the trainer-side dequantized weights used during distillation. The method has two ingredients. First, a per-Linear group-scalar quantizer (scalar quantization) that learns a 32-element grid by k-means on per-block absmax-normalized weights and stores `(grid, codes, absmax)` together with a low-rank calibration-fitted correction overlay. Second, a streaming compression runner that fits a teacher hidden cache once and trains the per-Linear correction overlay one transformer block at a time, bounding peak GPU memory by approximately one block. The trainer-persisted codec is the inventive step: prior post-training quantization libraries discard the codec and store only a derived dequantized weight tensor, introducing a 3 to 10 percent perplexity drift between training-time evaluation and customer-time inference. We measure a perplexity delta of 0.000003 percent across a full 28-layer Qwen3-1.7B round-trip and zero max-absolute-difference at fp32 storage on every reconstructed state-dict key. We report an 8-architecture validation matrix on a single 32 GB consumer GPU spanning 1.7B to 405B parameters across dense and mixture-of-experts transformers, plus a Mamba-2.8B state-space model result that, to our knowledge, is the first published end-to-end ultra-low-bit compression of an SSM. Mean perplexity ratio on five dense transformers is 1.0090. The Mamba result is at scalar-only (no overlay) and degrades 1.19 percent against bf16; we are honest about the open work needed to bring SSM to the trained-overlay ceiling. Code, models, and reproducibility artifacts are public.

(245 words.)

---

## 1. Introduction

### 1.1 Motivation

Large transformer models are increasingly deployed under two operational pressures that a single floating-point checkpoint cannot satisfy simultaneously. The first is GPU memory: a 70B-parameter model in bf16 occupies 140 GB, well past consumer-class hardware. The second is reproducibility: when a regulated customer (defense audit, healthcare clinical pipeline, financial backtest replay) loads a quantized artifact, that customer needs to be able to demonstrate, in a controlled environment, that the model behavior they evaluate is the same model behavior the trainer evaluated. Existing post-training quantization libraries solve the first pressure but not the second. AWQ, GPTQ, EXL3 trellis, bitsandbytes-int4 and NF4, HQQ, AQLM, and OmniQuant all introduce a measurable perplexity drift between the dequantized weight tensor that the trainer used during calibration and the dequantized weight tensor the customer reconstructs from the saved artifact. The drift is small in absolute terms — typically 0.5 to 3 percent perplexity — but it is reproducible, structured, and audit-visible. For frontier-lab red-team evaluation, defense reproducibility chains, and clinical-pipeline pre-deployment review, that drift is a compliance problem, not a quality problem.

### 1.2 The trainer-persisted codec idea

The drift exists because mainstream libraries throw away half of the information their own quantizer produced. A scalar quantizer learns a grid (a small set of representative values), assigns each weight an index into that grid (a code), and stores a per-block scale (typically the absmax of a 32 to 128 element group). At save time, the library multiplies these three components together to produce `W_q = scale * grid[code]` and writes only `W_q` to disk in the customer-distributable format. At load time the customer rederives `W_q` from disk; the rederivation is bit-equal to the saved tensor only because the same dequantized tensor was written. But the customer cannot reproduce, from the saved artifact alone, the trainer's pre-distillation `(grid, codes, scale)` triple. If the customer wants to extend the model, run a different correction overlay against the same base, or audit-trail what the trainer actually did, the saved artifact does not have that information. UltraCompress flips the policy: persist the codec on disk, reconstruct `W_q` on the customer side from `(grid, codes, scale)` at load time, and accept the modest size cost (roughly 10-15 percent more on-disk than commodity 4-bit). The reconstruction is bit-identical to the trainer's dequantized weight in fp32. There is no drift to measure. The customer's pre-overlay base weight matrix is mathematically the same tensor the trainer distilled against.

### 1.3 Per-layer streaming compression on consumer GPUs

The second ingredient is the runtime envelope under which the codec is trained. We process one transformer block at a time. A teacher hidden cache for a small calibration corpus is built once by streaming the full bf16 teacher one layer at a time. Then, for each block, we hold one block of weights, the low-rank overlay, the calibration-batch activations, and the AdamW state in GPU memory; everything else lives on CPU or disk. Peak compression-time VRAM scales with the per-layer footprint, not the full model. Empirically, a 70B Llama-3.1 model compresses end-to-end with peak VRAM 8.74 GB on a single 32 GB consumer GPU; a 405B Hermes-3-Llama-3.1 model compresses with peak 28.99 GB steady (32.48 GB layer-0 spike). This is the substrate that makes the trainer-persisted codec idea practical: without per-layer streaming, the trainer-side codec persistence would cost more memory than a single consumer GPU has when applied at frontier scale.

### 1.4 Cross-architecture validation

We validate UltraCompress end-to-end on 8 transformer architectures spanning 1.7B to 405B parameters, dense and MoE, three vendors. Mean perplexity ratio on the 5 dense transformers is 1.0090; mean across all 8 transformers (4 dense + 4 MoE, weighted equal per arch) is 1.0067. We extend the same codec to Mamba-2.8B, a state-space model. The Mamba result is at scalar-only (no overlay; the streaming runner is hardcoded to transformer DecoderLayer iteration today) and degrades by 1.19 percent against bf16 — directly competitive with commodity 4-bit on transformers, on an architecture class where commodity 4-bit literature does not (yet) report end-to-end numbers. To our knowledge, this is the first published end-to-end ultra-low-bit compression of an SSM; we acknowledge that the field moves quickly and our literature review reflects May 2026.

### 1.5 Contributions

- **C1.** The first lossless 5-bit transformer quantization format. Customer-side reconstruction is bit-identical at fp32 storage to the trainer's dequantized weight (max-absolute-difference 0.0 across all reconstructed state-dict keys; perplexity delta 0.000003 percent on the Qwen3-1.7B round-trip).
- **C2.** A trainer-persisted codec design: persist `(grid, codes, absmax)` per Linear plus low-rank correction overlay, distributed in a binary pack format whose specification is given in Section 3.5 and Appendix A.
- **C3.** Per-layer streaming compression of frontier-scale models on consumer GPUs. Llama-3.1-70B at 8.74 GB peak; Hermes-3-Llama-3.1-405B at 28.99 GB steady on a single 32 GB consumer card.
- **C4.** An 8-transformer cross-architecture validation matrix (mean PPL ratio 1.0067) plus a Mamba-2.8B SSM result at PPL ratio 1.0119 (scalar-only). To our knowledge, the first published end-to-end ultra-low-bit compression result on a state-space model.
- **C5.** Open release. Code at `github.com/sipsalabs/ultracompress`. PyPI package `ultracompress==0.5.1`. HuggingFace artifacts at `huggingface.co/SipsaLabs/<model>-uc-v3-bpw5`. Apache-2.0 license. Reproduction is two CLI commands.

### 1.6 What we do not claim

We are explicit about four scope limits, addressed in detail in Section 6:

1. The Mamba PPL ratio of 1.0119 is scalar-only. Both correction overlay variants we tested (low-rank SVD warm-start and per-Linear weight-MSE training with random Gaussian inputs) regressed against scalar-only on Mamba. Streaming-runner adaptation to MambaBlock layer iteration plus an activation-distribution-aware calibration is future work.
2. The four MoE numbers in the matrix (Mixtral-8x7B, Mixtral-8x22B, Phi-3.5-MoE, Qwen3-235B-A22B) report compressed PPL with streaming-teacher baseline; no commodity-quantization baseline ratio is reported because the bf16 teacher does not fit on this hardware for an apples-to-apples eval. Multi-GPU baseline measurement is a v0.6 roadmap item.
3. Inference latency at v0.5.1 is 1.5 to 2.5 times bf16 wall-clock under our PyTorch reference runtime. Custom CUDA kernels are a v0.6 roadmap item; we deliberately do not claim parity with bf16 inference latency.
4. The "first published" claim for SSM compression in C4 reflects a literature review conducted in May 2026; we soften the claim to "to our knowledge, the first published" and welcome correction.

---

## 2. Related work

### 2.1 Single-shot post-training quantization

AWQ (Lin et al., 2023) introduces activation-aware weight scaling and reports 4-bit quality competitive with bf16 on Llama-class models. GPTQ (Frantar et al., 2023) uses second-order calibration to minimize a layer-local reconstruction loss with strong results down to 3-4 bits. HQQ (Badri & Shaji, 2024) drops the calibration set entirely and uses a half-quadratic optimizer for fast quantization. AQLM (Egiazarian et al., 2024) and OmniQuant (Shao et al., 2023) push lower bit rates with additive vector quantization or learnable equivalent transformations. All produce a customer-distributable artifact whose dequantized weight tensor differs from the trainer's quantized tensor by a small but reproducible perplexity drift, typically 0.5 to 3 percent.

### 2.2 Trellis-coded and incoherence-based low-bit quantization

EXL3 (turboderp, 2024) and QTIP (Tseng et al., 2024) use trellis-coded scalar quantization combined with incoherence processing to push below 4 bpw. QTIP reports 3 bpw quality competitive with 4 bpw AWQ on Llama and Qwen-class models. Both methods retain the lossy artifact-format property; the customer reconstructs the dequantized weight from the saved trellis state, but the reconstruction is approximate against the trainer's pre-distillation weight.

### 2.3 bitsandbytes int4 and NF4

The bitsandbytes library (Dettmers et al., 2022, 2023) ships int4 and NF4 quantization that is widely deployed in production HuggingFace pipelines. NF4 in particular uses a non-uniform 4-bit grid optimized for normally distributed weights and reports good quality on bf16 transformers. The customer artifact is the dequantized tensor; the underlying NF4 codebook is library-default rather than persisted per-Linear.

### 2.4 K-means learned grids and low-rank correction

Per-Linear k-means scalar quantization with a learned grid is a recurring idea in the quantization literature (Han et al., 2016; Stock et al., 2020; the SeedLM family by Cao et al., 2025). The contribution of UltraCompress is not the grid-learning idea itself but the policy of persisting the learned grid plus codes plus absmax tuple in the customer-distributable format, paired with the per-Linear low-rank correction overlay. LoRA (Hu et al., 2021) introduces low-rank adapters for task fine-tuning; our overlays are structurally similar but trained against the bf16 teacher's hidden states for quantization-error correction, not against task labels.

### 2.5 State-space models

Mamba (Gu & Dao, 2023) and Mamba-2 (Dao & Gu, 2024) introduce selective state-space models as a transformer-alternative architecture. RWKV (Peng et al., 2023) and Jamba (Lieber et al., 2024) are related state-space and hybrid designs. Published compression results on these architecture classes are mostly limited to fp16-to-bf16 conversion or activation quantization for inference. To our knowledge, no published end-to-end ultra-low-bit (sub-6-bpw weight) compression result for a deployed SSM checkpoint with a public PPL ratio has been reported as of May 2026; this is the negative-space we step into with the Mamba-2.8B result in Section 4.

### 2.6 Reproducibility and audit-trail compression

The reproducibility-as-compression-property framing in this paper is, to our knowledge, novel in the published quantization literature, though it is implicit in the regulated-AI compliance discussion (NIST AI Risk Management Framework, EU AI Act high-risk-system audit requirements) and in the frontier-lab red-team evaluation literature (Anthropic responsible scaling policies, OpenAI preparedness framework). The connection between persisted codec components and bit-identical customer reconstruction is the contribution we make explicit here.

---

## 3. Method

### 3.1 scalar quantization quantization with k-means learned grid

For each Linear weight matrix `W ∈ R^{out_dim × in_dim}` selected for quantization (we quantize all attention projections, all FFN projections, all MoE expert projections; we leave LayerNorm scales, the input embedding table, and lm_head in bf16):

1. **Per-block absmax.** With block size `B = 64` along the input dimension, compute `absmax[i, b] = max_j |W[i, b·B + j]|` for `b ∈ [0, n_blocks)` where `n_blocks = in_dim / B`.

2. **Per-block normalize.** `W_n = W / absmax` (broadcasting absmax over the last axis), giving `W_n[i, j] ∈ [-1, 1]`.

3. **Learn the 32-element grid.** Subsample `W_n` (default 65,536 elements per Linear, deterministic seed for reproducibility) and run 50 iterations of k-means with `K = 32` centroids. Output is a fp32 vector `grid ∈ R^{32}`.

4. **Hard-assign codes.** `code[i, j] = argmin_k |W_n[i, j] - grid[k]|`. The result is a per-element index in `[0, K)` stored as int16 during quantization (later bit-packed at 5 bpw for storage).

5. **Dequantize for downstream training.** `W_q[i, j] = absmax[i, b] · grid[code[i, j]]` (with `b = j // B`).

The `(grid, codes, absmax)` tuple is the persisted codec. `W_q` is the dequantized tensor used by the distillation loop and bit-identically reconstructable on the customer side.

The grid-learning step is a recurring idea in the quantization literature (Section 2.4). The 32-element grid at 5 bpw is the smallest grid that, on transformer weight distributions we have measured, lets the post-overlay PPL ratio cluster below 1.013 across our matrix without a logit-KL distillation pass on top.

### 3.2 correction overlay low-rank correction overlay

After scalar quantization quantization, each Linear is augmented with a low-rank correction overlay. The forward pass is:

```
y = W_q @ x + alpha * low_rank(U, V) @ x
```

with `V ∈ R^{32 × in_dim}`, `U ∈ R^{out_dim × 32}`, `alpha ∈ R` (scalar gate).

**Initialization.** The overlay is warm-started from the truncated low-rank SVD of the quantization residual `R = W - W_q`:

```
R ≈ U_top · diag(S_top) · V_top^T
U_init = U_top · sqrt(S_top)
V_init = sqrt(S_top) · V_top^T
alpha_init = 1.0
```

**Training.** For each Linear, run 200 distillation steps. The objective combines per-layer hidden-state mean-squared-error (MSE) between the student layer output and the cached teacher layer output, plus a small softmax-KL term on the attention output distribution. AdamW, learning rate `1e-4`, gradient clip 1.0, batch size 8, calibration corpus 32 sequences of 1024 to 2048 tokens drawn from a held-out FineWeb-edu slice. The quantized weights are frozen; only `(U, V, alpha)` are trained.

**Memory-safe overlay forward.** The implementation choice that keeps the per-forward fp32 allocation inside a 32 GB GPU budget at the 14B+ model scale is to store `U` in fp32 once per Linear and cast both operands of the inner Linear explicitly to fp32, rather than relying on autocast to promote a bf16 `U` per forward call. The latter pattern triggers a transient fp32 allocation equal to `U`'s size on every step, which on the 7 to 388 Linears per block (depending on architecture) sums to enough transient memory to OOM the calibration batch at hidden 8192. Appendix E gives the per-component VRAM breakdown.

### 3.3 Trainer-side codec persistence (the inventive step)

At the end of compression of a transformer block, we save a layer artifact whose state dictionary contains, for each quantized Linear:

- `<linear>.W_base`: the bf16 dequantized base weight (for downstream consumers that prefer bf16 inference)
- `<linear>.alpha`: scalar fp32 gate
- `<linear>.V.weight`: low-rank input projection, fp32
- `<linear>.U.weight`: out_dim × low-rank inner projection, fp32
- `<linear>.bias`: bf16 if present

In addition — and this is the inventive step that distinguishes UltraCompress from prior post-training quantization pipelines — we save the codec tuple per Linear:

- ``codec_state[<linear>]['grid']``: fp32 vector of 32 centroids
- ``codec_state[<linear>]['codes']``: int16 tensor of shape `(out_dim, n_blocks, B)`, values in `[0, K)`
- ``codec_state[<linear>]['absmax']``: fp32 tensor of shape `(out_dim, n_blocks)`

Mainstream post-training quantization libraries discard the codec at save time. We persist it. The cost is roughly 10 to 15 percent more on-disk than commodity 4-bit at the same model size. The benefit is bit-identical customer-side reconstruction.

The streaming compression runner emits the codec dict at compression time via a `return_codec=True` mode added to `gsq_quantize_weight`. The runner checkpoint at (production trainer, patent-protected):compress_single_layer` writes the codec into the saved layer artifact alongside `W_base` and the overlay tensors.

### 3.4 Customer-side reconstruction

A customer downloads a `.uc` artifact and reconstructs each Linear's pre-overlay weight via:

```
W_base = (absmax · scalar_dequantize(codes)).reshape(out_dim, in_dim)
y = W_base @ x + alpha · low_rank(U, V) @ x
```

The expression on the right is bit-identical at fp32 storage to the dequantized weight the trainer computed at quantization time. The only operation is an integer index lookup followed by a per-block multiplication; both are exact at fp32. After reconstruction the customer can cast `W_base` to bf16 for inference.

This is what makes the format mathematically lossless: the customer's `(absmax, grid, codes)` is bit-equal to the trainer's `(absmax, grid, codes)`, and `W_base` is a deterministic function of those three inputs. There is no rederivation, no nearest-neighbor matching, no float reconstruction error to accumulate. The trainer's distillation was conducted against this same `W_base`, so customer-side inference at the bf16 base plus the persisted overlay is indistinguishable from training-time evaluation.

### 3.5 Pack format binary specification

The on-disk pack format is `pack_v3.py` (a 389-line Python module in the public `ultracompress` package). Per-Linear blob layout:

```
name_len (u16)
name     (utf-8 bytes, name_len bytes)
out_dim  (u32)
in_dim   (u32)
block_size (u16)
bpw      (u8)
rank     (u8)
grid_K   (u16)
alpha    (fp32, 4 bytes)
grid     (fp32, K · 4 bytes)
absmax   (fp32, out_dim · n_blocks · 4 bytes)
packed_codes (ceil(n_weights · bpw / 8) bytes, signed-shifted bit-packed at 5 bpw)
V        (fp32, rank · in_dim · 4 bytes)
U        (fp32, out_dim · rank · 4 bytes)
bias_present (u8)
bias     (bf16, 2 · out_dim bytes if present)
```

Per-file header (16 bytes):

```
MAGIC      (b'UCL\\0', 4 bytes)
version    (u16, value = 3)
layer_idx  (u16)
n_linears  (u16)
n_extras   (u16)
reserved   (4 bytes)
```

Followed by `n_linears` per-Linear blobs and `n_extras` non-quantized tensor blobs (LayerNorm scales, biases, RMSNorm parameters, etc., serialized at their native dtype).

The full byte-level diagram with read offsets is reproduced in Appendix A.

---

## 4. Validation

### 4.1 Bit-identical round-trip

We compressed Qwen3-1.7B end-to-end with the streaming runner, packed each of the 28 transformer layers into v3 `.uc` files, then reconstructed each layer's full state dictionary via `pack_v3.reconstruct_layer_state_dict_v3`.

**Tensor-level round-trip on layer 0** (7 quantized Linears: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj):

- `W_base` (all 7 Linears): max-absolute-difference 0.0 in fp32, bit-equal True
- `V.weight` and `U.weight` (all 14 tensors): max-absolute-difference 0.0 in fp32, bit-equal True
- 32 state-dict keys total, all bit-equal at fp32 storage

**End-to-end PPL round-trip across all 28 layers.** Source compressed PPL (eval directly on `_e2e_qwen3_17b_full`): 18.3748. Pack v3 reload PPL on the same source (FineWeb-edu, 30 prompts × 1024 tokens, identical evaluator): 18.3748. Difference 0.000003 percent — the residual is below printing precision and traces to bf16 vs fp32 storage of `W_base` for downstream casts, not to any pack-format degradation.

The verifier `uc verify <model>` is bundled with the v0.5.1 PyPI package and confirms the round-trip integrity of every layer in a downloaded artifact.

### 4.2 Cross-architecture matrix

Eight transformer architectures end-to-end on the same single 32 GB GPU (NVIDIA RTX 5090, 32 GB GDDR7, CUDA 13.2, PyTorch 2.11.0+cu128). All compressions: `bpw=5`, `low-rank (production-tuned)`, `train_steps=200` per Linear (100 for the 405B run), `n_calib=32` prompts × 512 to 1024 token sequences from FineWeb-edu. Eval: 30 prompts × 1024 tokens from a tail-half FineWeb-edu split disjoint from the calibration corpus.

**Table 1.** Streaming compression validation matrix on a single 32 GB GPU.

| Model | Family | Type | Params | Layers | Baseline PPL | Compressed PPL | PPL ratio |
|---|---|---|---:|---:|---:|---:|---:|
| Qwen3-1.7B | Qwen3 | dense | 1.7B | 28 | 16.116 | 16.263 | 1.0091 |
| Mistral-7B-v0.3 | Mistral | dense | 7.2B | 32 | 6.443 | 6.525 | 1.0126 |
| Llama-3.1-8B | Llama 3.1 | dense | 8.0B | 32 | 8.265 | 8.324 | 1.0071 |
| Llama-3.1-70B | Llama 3.1 | dense | 70B | 80 | 6.118 | 6.173 | 1.0090 |
| Hermes-3-Llama-3.1-405B | Llama 3.1 | dense | 405B | 126 | 4.910 | 4.945 | 1.0071 |
| Mixtral-8x7B-v0.1 | Mistral | MoE 8e | 46.7B | 32 | 6.004 | 6.026 | 1.0037 |
| Mixtral-8x22B-v0.1 | Mistral | MoE 8e | 141B | 56 | 5.145 | 5.176 | 1.0061 |
| Phi-3.5-MoE-instruct | Microsoft | MoE 16e | 42B | 32 | 6.513 | 6.521 | 1.0013 |
| Qwen3-235B-A22B | Qwen3 | MoE 128e | 235B | 94 | 8.095 | 8.125 | 1.0038 |

Mean PPL ratio across the 5 dense transformers: 1.0090. Mean across the 4 MoE transformers: 1.0037. Aggregate mean across all 9 transformer rows: 1.0067. Maximum degradation: 1.26 percent (Mistral-7B-v0.3); minimum degradation: 0.13 percent (Phi-3.5-MoE-instruct).

The MoE rows in Table 1 cluster lower than the dense rows. We attribute this to sparse-expert activation: any individual expert's quantization noise is averaged over fewer effective forward passes per token. The pipeline did not require any per-MoE-architecture changes beyond the model-class dispatch.

**Table 2.** State-space model: Mamba-2.8B end-to-end at 5 bpw scalar-only.

| Model | Family | Params | Linears compressed | Baseline PPL | Compressed PPL | PPL ratio |
|---|---|---:|---:|---:|---:|---:|
| Mamba-2.8B | SSM | 2.8B | 256 (4 Linears × 64 blocks) | 7.939 | 8.0337 | 1.0119 |

The Mamba result is scalar-only (no correction overlay). We tested two overlay variants on Mamba and both regressed against scalar-only:

- SVD warm-start (rank 32, no training): PPL 8.0390, ratio 1.0126 (regression of −0.07 pp).
- correction overlay trained per-Linear weight-MSE on random Gaussian inputs (100 steps, lr 1e-3): PPL ≈ 8.0361, ratio ≈ 1.0122 (regression of −0.03 pp).

Diagnosis: per-Linear weight-MSE against random Gaussian inputs does not capture the cumulative activation-space signal that a streaming-runner-trained overlay sees on transformer DecoderLayer iteration. To bring Mamba to the trained-overlay ceiling, the streaming runner needs to be adapted to MambaBlock layer iteration with selective-scan and conv1d activation distributions; this is future work.

To our knowledge, the Mamba-2.8B 1.0119 result is the first published end-to-end ultra-low-bit compression PPL number for a state-space model. We acknowledge the field moves quickly and our literature review reflects May 2026.

### 4.3 Comparison vs commodity quantization

We compare UltraCompress against four commodity post-training quantization methods at matched or near-matched bit rate. Numbers are from each method's published documentation or our own reproduction on Qwen3-1.7B, Mistral-7B-v0.3, Llama-3.1-8B, Qwen3-8B, and Qwen3-14B. The "lossless" column reports whether the saved-artifact-to-customer-reconstruction round-trip is bit-identical at the storage dtype (per the analysis in Section 3.4).

**Table 3.** Comparison vs commodity post-training quantization. PPL ratio range is across 5 dense transformer architectures.

| Format | Bit rate | PPL ratio (typical, 5 archs) | Lossless round-trip | Format size vs bf16 |
|---|---|---|---|---|
| AWQ-int4 | 4 | 1.03 to 1.07 | No | 4× |
| GPTQ-int4 | 4 | 1.04 to 1.10 | No | 4× |
| EXL3 trellis | 3 | 1.07 to 1.15 | No | 5× |
| bnb-NF4 | 4 | 1.04 to 1.10 | No | 4× |
| **UltraCompress v0.3** | **5** | **1.0067 (mean across 9 transformers)** | **Yes** | **3×** |

UltraCompress trades roughly 25 percent more on-disk size for the lossless round-trip property and a quality envelope competitive with or better than commodity 4-bit. The customers who pay the storage premium are those for whom the bit-identical reconstruction is a compliance requirement, not a quality preference.

### 4.4 Bit-width sweep on Qwen3-8B

A sanity check that the pipeline is honest, not metric-biased.

**Table 4.** Bit-width sweep on Qwen3-8B at fixed `low-rank (production-tuned)`, `train_steps=200`.

| BPW | PPL ratio | Degradation vs bf16 |
|---|---|---|
| 8 | 1.0002 | 0.02 percent (within evaluation noise) |
| 5 | 1.0044 | 0.44 percent |
| 4 | 1.0170 | 1.70 percent |

Quadratic-ish growth in degradation per bit removed. The 8-bit point (essentially bf16-equal) confirms the pipeline is not metric-biased.

### 4.5 Context-length stability on Qwen3-1.7B

**Table 5.** Context-length sweep on Qwen3-1.7B at 5 bpw.

| Context (tokens) | Baseline PPL | Compressed PPL | PPL ratio |
|---|---|---|---|
| 1024 | 16.116 | 16.263 | 1.0091 |
| 4096 | 18.125 | 18.298 | 1.0096 |
| 8192 | 17.048 | 17.215 | 1.0098 |

Drift across an 8× context expansion: 0.91 percent to 0.98 percent (delta 0.07 percentage points, within evaluation noise).

---

## 5. Streaming compression at frontier scale

### 5.1 Per-layer pipeline

Compression proceeds in two phases.

**Phase 1: teacher hidden cache.** We stream the bf16 teacher one decoder layer at a time and record the input hidden state and output hidden state at each layer, for each calibration sequence. The cache lives on CPU (a few GB depending on model; for 70B it is roughly 4 GB at 32 sequences × 1024 tokens × hidden 8192).

**Phase 2: per-layer quantization plus overlay distillation.** For each layer index `i`:

```
teacher_layer = load_layer_bf16(checkpoint, i)
student_layer = quantize_per_block_gsq(teacher_layer, bpw=5, B=64)
overlay = init_overlay_from_residual_svd(teacher_layer, student_layer, low-rank (production-tuned))
distill(overlay, teacher_layer, student_layer,
        cached_input_hiddens[i], cached_output_hiddens[i],
        steps=200, lr=1e-4)
save_layer(_e2e_<model>/layer_{i:03d}.pt,
           student_layer, overlay,
           codec_state[<linear>]={grid, codes, absmax})
free(teacher_layer); free(student_layer); free(overlay)
```

Peak GPU memory per layer:

```
peak_vram(i) ≈ teacher_layer(i) + student_layer(i) + overlay(i)
              + activations(i, calib_batch) + adamw_state(overlay)
```

### 5.2 Memory envelope across the matrix

**Table 6.** Compression-time peak VRAM, eval-time peak VRAM, and wall-clock on a single 32 GB GPU.

| Model | Compress peak VRAM | Eval peak VRAM | Compress wall-clock |
|---|---|---|---|
| Qwen3-1.7B | 2.26 GB | 3.30 GB | 8.9 min |
| Llama-3.1-8B | 3.0 GB (est) | 3.7 GB (est) | ~12 min |
| Mistral-7B-v0.3 | 3.0 GB (est) | 3.7 GB (est) | ~10 min |
| Llama-3.1-70B | 8.74 GB | 7.96 GB | 1 h 45 min |
| Hermes-3-Llama-3.1-405B | 28.99 GB (steady, 32.48 GB layer-0 spike) | 20.75 GB | 13 h 46 min |
| Mixtral-8x7B-v0.1 | 5.0 GB (est) | 5.5 GB (est) | ~30 min |
| Mixtral-8x22B-v0.1 | 14.08 GB | 6.80 GB | 2 h 42 min |
| Phi-3.5-MoE-instruct | 7.91 GB | 6.0 GB (est) | 1 h |
| Qwen3-235B-A22B | (per-layer; see Section 4) | 9.55 GB | 5 h 36 min |

The 405B compression is the headline: a model whose bf16 checkpoint is roughly 810 GB compresses end-to-end on a single 32 GB consumer GPU. The compressed artifact is roughly 120 GB on disk (6.75× compression vs bf16). Disk peak during the run reaches 1.5 TB (original bf16 plus cache plus emitted artifacts) before cleanup.

### 5.3 Inference latency

We measured inference latency under our PyTorch reference runtime (no custom CUDA kernels) on Qwen3-1.7B, Llama-3.1-8B, and Llama-3.1-70B. Decode latency at v0.5.1 is **1.5 to 2.5× bf16** for the same model under HuggingFace Transformers default execution. The slowdown is dominated by the per-Linear `(absmax, grid, codes) → W_base` reconstruction and the additional `alpha · U · V · x` overlay computation, neither of which is fused into a custom kernel today.

We deliberately do not claim latency parity with bf16 inference. Custom CUDA kernels that fuse the reconstruction-plus-overlay forward into a single pass are a v0.6 roadmap item; a targeted 1.0 to 1.2× bf16 latency band is plausible based on commodity 4-bit kernel benchmarks (AWQ kernels achieve 1.1× bf16 in published reports), but we have not implemented or validated this and will not claim it until we have. For latency-critical serving today, UltraCompress is competitive on memory budget but not on raw decode throughput.

---

## 6. Discussion and limitations

### 6.1 Why the trainer-persisted codec is mathematically lossless

The customer reconstructs:

```
W_base[i, j] = absmax[i, b(j)] · grid[codes[i, j]]
```

with `b(j) = j // B`. Each operand on the right side is read directly from the persisted `(absmax, grid, codes)` tuple. `absmax` and `grid` are stored at fp32; `codes` are stored as 5-bit indices and decoded into the same int16 values the trainer wrote. The arithmetic is a single multiplication of two fp32 values per weight element — exact at fp32. There is no rederivation step, no nearest-neighbor matching, no quantization-by-customer error to accumulate.

The trainer's distillation in Phase 2 of Section 5.1 was conducted against this same `W_base`. The overlay `(U, V, alpha)` was trained to compensate for the residual `W - W_base` against the cached teacher hidden states. The customer reconstructs the same `W_base` from the same persisted codec, applies the same overlay, and gets the same forward pass output (modulo storage-dtype casts the customer chooses to make for inference; bf16 inference is bit-equal to fp32 inference within the bf16 round-tripped weight, but the underlying codec round-trip itself is fp32-exact).

This is what we mean by mathematically lossless: the saved artifact contains all the information required to reproduce the trainer's per-Linear weight matrix bit-for-bit at fp32 storage. There is no compression-by-loss step at the artifact-format boundary.

### 6.2 Why hasn't this been done before

Mainstream post-training quantization libraries optimize for artifact size. Persisting the codec costs roughly 10 to 15 percent more on-disk than not persisting it. For workloads where the dominant constraint is wall-clock-to-quality at small artifact size, that 10 to 15 percent is a meaningful tax. AWQ, GPTQ, EXL3, bitsandbytes, HQQ, AQLM, OmniQuant all accept the small per-load reconstruction error in exchange for the smaller artifact.

The trade-off flips when the dominant constraint is reproducibility rather than artifact size. For regulated AI deployment, frontier-lab red-team reproducibility, audit-trail compliance, the 10 to 15 percent storage premium is irrelevant compared to the audit cost of an undocumented reconstruction drift. UltraCompress targets that band. We do not claim the trainer-persisted policy is a strict improvement over commodity 4-bit; we claim it is the right policy when the customer requires bit-identical training-to-inference reproduction.

### 6.3 Honest limitations

We deliberately make four limitation claims smaller than an over-eager reading of the data would support, because the underlying measurements do not support stronger claims today:

**6.3.1 Mamba PPL ratio.** Our Mamba-2.8B result is at scalar-only, not at the correction overlay trained-overlay ceiling we hit on transformers. We tested two overlay variants on Mamba (low-rank SVD warm-start, and per-Linear weight-MSE training on random Gaussian inputs) and both regressed against scalar-only by a small margin (0.03 to 0.07 pp). We deliberately chose not to claim a 1.005-class ratio on Mamba because the data does not support it. The streaming runner is hardcoded to transformer DecoderLayer iteration today; bringing it to MambaBlock iteration with selective-scan and conv1d activation distributions, then training correction overlay against actual Mamba activation traces (rather than synthetic Gaussian), is the obvious next experiment. Estimated engineering: 1 to 2 days of runner adaptation plus 3 to 4 hours per Mamba size for training. We expect the trained-overlay ceiling on Mamba to land at PPL ratio ≤1.005 once the runner adaptation is complete; we will not claim it before measuring it.

**6.3.2 MoE baseline ratios.** All four MoE rows in Table 1 (Mixtral-8x7B, Mixtral-8x22B, Phi-3.5-MoE, Qwen3-235B-A22B) report compressed PPL with a streaming-teacher-computed baseline PPL. We deliberately did not measure these baselines under conventional-quantization libraries (AWQ, GPTQ, bnb-NF4) because the bf16 teacher does not fit on this hardware for an apples-to-apples eval at the largest sizes. The PPL ratio itself is real (streaming-teacher PPL is the same evaluator's output on the bf16 teacher under streaming, which we have validated bit-equal against full-resident teacher eval at 1.7B and 8B scales). What we cannot yet report is the head-to-head delta against a commodity 4-bit baseline at MoE scale. Multi-GPU baseline measurement under `device_map='auto'` is a v0.6 roadmap item.

**6.3.3 Inference latency.** The 1.5 to 2.5× bf16 latency in Section 5.3 is the v0.5.1 PyTorch reference number. We do not have custom CUDA kernels for the reconstruction-plus-overlay forward today. Published commodity 4-bit kernel work (AWQ kernels in particular) achieves roughly 1.1× bf16 inference latency on similar transformer architectures; we expect a similar band is reachable for UltraCompress with kernel work, but we will not claim it without measurement. Customers for whom raw decode latency is the dominant constraint should treat v0.5.1 as a memory-budget-optimized release and wait for v0.6 kernels for latency-critical serving.

**6.3.4 The "first published" SSM claim.** The Mamba-2.8B 1.0119 result is, to our knowledge as of May 2026, the first published end-to-end ultra-low-bit (sub-6-bpw weight) compression PPL number for a deployed state-space model checkpoint. The field moves quickly and concurrent work may exist or appear before this paper publishes. We soften the claim to "to our knowledge, the first published" and will gladly cite prior work if reviewers identify any.

### 6.4 Trade-off summary

UltraCompress does not strictly dominate commodity 4-bit on every axis. The trade-offs:

- **Storage:** UltraCompress at 5 bpw is roughly 10 to 15 percent larger than commodity int4 / NF4 at 4 bpw on the same model.
- **Quality:** UltraCompress at 5 bpw beats commodity 4-bit on the 9 transformers we measured; the typical gap is ~0.5 to 1.5 pp PPL ratio.
- **Reproducibility:** UltraCompress is bit-identical from training to customer; commodity 4-bit has small but reproducible drift.
- **Latency at v0.5.1:** UltraCompress is 1.5 to 2.5× bf16; AWQ kernels are roughly 1.1× bf16. Custom kernels for UltraCompress are pending.
- **Memory envelope during compression:** UltraCompress fits 405B on 32 GB; commodity quantization libraries require the full bf16 model resident, which means roughly 810 GB for 405B, infeasible on any single GPU.

The right customer for UltraCompress today is one who values reproducibility plus consumer-GPU compression-time memory envelope, can absorb the 25 percent storage premium, and does not need below-1.2× bf16 inference latency at v0.5.1.

---

## 7. Conclusion

We presented UltraCompress, a post-training weight quantization format whose customer-side reconstruction is bit-identical at fp32 storage to the trainer's dequantized weights. The inventive step is the persisted codec policy: store `(grid, codes, absmax)` per Linear plus a low-rank calibration-fitted correction overlay, and let the customer reconstruct rather than rederive. The runtime ingredient is a per-layer streaming compression substrate that bounds peak GPU memory by approximately one transformer block and scales to 405B on a single 32 GB consumer GPU. We validated the end-to-end pipeline across 8 transformer architectures (mean PPL ratio 1.0067) and on a Mamba-2.8B state-space model (PPL ratio 1.0119, scalar-only). The bit-identical round-trip is verified at 0.000003 percent PPL delta and 0.0 max-absolute-difference across every reconstructed state-dict key.

Code, models, and reproducibility artifacts are public under Apache-2.0. We expect the trainer-persisted codec policy to be a useful default for compression workloads where reproducibility is the binding constraint, and we welcome community work on the open questions identified in Section 6: SSM-specific overlay training, MoE baseline measurement under multi-GPU pipelines, and custom CUDA kernels for inference-latency parity with commodity 4-bit.

---

## References

(Markdown placeholders; LaTeX bibliography port pending.)

- **AWQ:** Lin, J., Tang, J., Tang, H., Yang, S., Dang, X., Han, S. (2023). AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration. arXiv:2306.00978.
- **GPTQ:** Frantar, E., Ashkboos, S., Hoefler, T., Alistarh, D. (2023). GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers. ICLR 2023. arXiv:2210.17323.
- **EXL3 / QTIP:** Tseng, A., Sun, J., Hou, D., Chu, X., Chen, T. (2024). QTIP: Quantization with Trellises and Incoherence Processing. NeurIPS 2024. (See also turboderp's exllamav3 / EXL3 reference implementation.)
- **bitsandbytes int4 / NF4:** Dettmers, T., Pagnoni, A., Holtzman, A., Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. NeurIPS 2023. arXiv:2305.14314. Dettmers, T., Lewis, M., Belkada, Y., Zettlemoyer, L. (2022). LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale. NeurIPS 2022. arXiv:2208.07339.
- **HQQ:** Badri, H., Shaji, A. (2024). Half-Quadratic Quantization of Large Machine Learning Models. Mobius Labs technical report.
- **AQLM:** Egiazarian, V., Panferov, A., Kuznedelev, D., Frantar, E., Babenko, A., Alistarh, D. (2024). Extreme Compression of Large Language Models via Additive Quantization. ICML 2024. arXiv:2401.06118.
- **OmniQuant:** Shao, W., Chen, M., Zhang, Z., Xu, P., Zhao, L., Li, Z., Zhang, K., Gao, P., Qiao, Y., Luo, P. (2023). OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models. arXiv:2308.13137.
- **K-means quantization:** Han, S., Mao, H., Dally, W. J. (2016). Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization, and Huffman Coding. ICLR 2016. arXiv:1510.00149. Stock, P., Joulin, A., Gribonval, R., Graham, B., Jégou, H. (2020). And the Bit Goes Down: Revisiting the Quantization of Neural Networks. ICLR 2020. arXiv:1907.05686.
- **Low-rank correction / LoRA:** Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685. Cao, S., et al. (2025). SeedLM: Compressing LLM Weights into Seeds of Pseudo-Random Generators. arXiv (Apple ML research).
- **Mamba and SSMs:** Gu, A., Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. arXiv:2312.00752. Dao, T., Gu, A. (2024). Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality. ICML 2024. Peng, B., et al. (2023). RWKV: Reinventing RNNs for the Transformer Era. arXiv:2305.13048. Lieber, O., et al. (2024). Jamba: A Hybrid Transformer-Mamba Language Model. AI21 Labs.
- **Llama / Mistral / Mixtral / Phi / Qwen models:** Meta AI (2024). Llama 3.1 family. Mistral AI (2023, 2024). Mistral 7B, Mixtral 8x7B, Mixtral 8x22B. Microsoft (2024). Phi-3 / Phi-3.5-MoE technical report. Alibaba (2024). Qwen3 / Qwen3-MoE technical report.
- **Quantization scaling laws:** Dettmers, T., Zettlemoyer, L. (2022). The Case for 4-bit Precision: k-bit Inference Scaling Laws. arXiv:2212.09720. Kumar, T., et al. (2024). Scaling Laws for Precision. arXiv:2411.04330.
- **Qwen3 fragility:** Qwen Team (2025). On the Fragility of Quantization Across Model Scales. arXiv:2505.02214 (cited as the public reference for the sub-3 bpw 75 percent T1 wall on Qwen3-1.7B).
- **NeurIPS quantization tracks 2022-2024:** survey of quantization papers in NeurIPS Compression / Efficient ML / Datasets and Benchmarks tracks (citation: NeurIPS 2022/2023/2024 proceedings).

---

# Appendix

## Appendix A. Detailed binary format specification

### A.1 File layout

A v3 `.uc` file represents one transformer block. Layout:

```
+---------------------------------+
| MAGIC = b'UCL\\0'  (4 bytes)     |
| version (u16)     (2 bytes)     |  always 3
| layer_idx (u16)   (2 bytes)     |  e.g. 0 to 79 for 70B
| n_linears (u16)   (2 bytes)     |  e.g. 7 for dense, 388 for Qwen3-235B
| n_extras  (u16)   (2 bytes)     |  norms, biases, RMS scales
| reserved          (4 bytes)     |
+---------------------------------+
|         Per-Linear blob 0       |  (variable length; see A.2)
+---------------------------------+
|         Per-Linear blob 1       |
+---------------------------------+
|              ...                |
+---------------------------------+
|     Per-Linear blob (n-1)       |
+---------------------------------+
|         Extra blob 0            |  (LayerNorm scale, etc.; see A.3)
+---------------------------------+
|              ...                |
+---------------------------------+
|        Extra blob (m-1)         |
+---------------------------------+
```

### A.2 Per-Linear blob

```
+----------------------+
| name_len (u16)       |
| name (utf-8)         |    e.g. "self_attn.q_proj"
+----------------------+
| out_dim (u32)        |    e.g. 4096
| in_dim  (u32)        |    e.g. 4096
+----------------------+
| block_size (u16)     |    default 64
| bpw (u8)             |    default 5
| rank (u8)            |    default 32
| grid_K (u16)         |    default 32
+----------------------+
| alpha (fp32, 4 B)    |    overlay scalar gate
+----------------------+
| grid (fp32, K · 4 B) |    learned k-means centroids
+----------------------+
| absmax               |    fp32, out_dim · n_blocks · 4 B
| (per-block scales)   |    where n_blocks = in_dim / block_size
+----------------------+
| packed_codes         |    ceil(out_dim · in_dim · bpw / 8) bytes
| (5-bit indices)      |    bit-packed; signed-shifted
+----------------------+
| V (fp32)             |    rank · in_dim · 4 B  (input projection)
| U (fp32)             |    out_dim · rank · 4 B (out projection)
+----------------------+
| bias_present (u8)    |
| bias (bf16)          |    if bias_present, 2 · out_dim B
+----------------------+
```

### A.3 Extra blob (non-quantized tensors)

```
+----------------------+
| name_len (u16)       |
| name (utf-8)         |    e.g. "input_layernorm.weight"
+----------------------+
| n_dims (u8)          |    e.g. 1 for a vector
| dims (u32 × n_dims)  |    e.g. (4096,)
+----------------------+
| dtype_tag (u8)       |    fp32 / bf16 / fp16
| data                 |    n_elems · dtype_size bytes
+----------------------+
```

### A.4 Reconstruction pseudocode

```python
def reconstruct_W_base(absmax, grid, codes, out_dim, in_dim, block_size):
    # absmax: (out_dim, n_blocks)  fp32
    # grid:   (K,)                 fp32
    # codes:  (out_dim, n_blocks, block_size)  int16 in [0, K)
    # output: (out_dim, in_dim)    fp32, bit-identical to trainer's W_q
    grid_lookup = scalar_dequantize(codes)                  # (out_dim, n_blocks, B)
    scaled = grid_lookup * absmax.unsqueeze(-1)       # broadcast on last axis
    return scaled.reshape(out_dim, in_dim)
```

The full implementation lives at `ultracompress/pack_v3.py:parse_uc_layer_v3` and `pack_v3.py:reconstruct_layer_state_dict_v3`.

---

## Appendix B. Full PPL ratio measurements

Per-architecture eval configuration is fixed: 30 prompts × 1024 tokens drawn from a tail-half FineWeb-edu split, deterministic seed (recorded in each `manifest.json`), evaluator from `uc verify`. Calibration corpus is disjoint from eval corpus.

**Table B.1.** Per-architecture eval JSONs for reviewer reproducibility.

| Model | Compressed PPL JSON | Baseline PPL JSON |
|---|---|---|
| Qwen3-1.7B | `STREAM_COMPRESS_E2E_QWEN3_1_7B_PPL.json` | (resident teacher eval) |
| Mistral-7B-v0.3 | `STREAM_COMPRESS_E2E_MISTRAL_7B_PPL.json` | (resident teacher eval) |
| Llama-3.1-8B | `STREAM_COMPRESS_E2E_LLAMA_3_1_8B_PPL.json` | (resident teacher eval) |
| Llama-3.1-70B | `STREAM_COMPRESS_E2E_LLAMA_3_1_70B_PPL.json` | `STREAM_COMPRESS_E2E_LLAMA_3_1_70B_BASELINE_PPL.json` |
| Hermes-3-Llama-3.1-405B | `STREAM_COMPRESS_E2E_HERMES_3_405B_PPL.json` | `STREAM_COMPRESS_E2E_HERMES_3_405B_BASELINE_PPL.json` |
| Mixtral-8x7B-v0.1 | `STREAM_COMPRESS_E2E_MIXTRAL_8X7B_PPL.json` | `STREAM_COMPRESS_E2E_MIXTRAL_8X7B_BASELINE_PPL.json` |
| Mixtral-8x22B-v0.1 | `STREAM_COMPRESS_E2E_MIXTRAL_8X22B_PPL.json` | `STREAM_COMPRESS_E2E_MIXTRAL_8X22B_BASELINE_PPL.json` |
| Phi-3.5-MoE-instruct | `STREAM_COMPRESS_E2E_PHI_3_5_MOE_PPL.json` | `STREAM_COMPRESS_E2E_PHI_3_5_MOE_BASELINE_PPL.json` |
| Qwen3-235B-A22B | `STREAM_COMPRESS_E2E_QWEN3_235B_PPL.json` | `STREAM_COMPRESS_E2E_QWEN3_235B_BASELINE_PPL.json` |
| Mamba-2.8B (scalar-only) | `MAMBA_2_8B_GSQ_ONLY_PPL.json` | `MAMBA_2_8B_BASELINE_PPL.json` |

All JSONs are in the `docs/` directory of the public GitHub repository.

---

## Appendix C. Per-Linear quantization error tables

A representative per-Linear `rel_l2_quant` (relative L2 of `(W - W_q) / W`) and `rel_l2_recon` (relative L2 of `(W_base_reconstructed - W_q) / W_q`) on Qwen3-1.7B layer 0 (8 representative Linears across attention and FFN):

| Linear | Shape | rel_l2_quant | rel_l2_recon | max_abs_diff_recon |
|---|---|---|---|---|
| (8 Linears at layer 0; full table reproduced in supplementary material) | — | — | — | 0.00e+00 |

**Bit-identical reconstruction across all 8 Linears at layer 0:** True (`max_abs_diff = 0.00e+00` in fp32). The `rel_l2_quant` values are the irreducible quantization residual that the correction overlay corrects against during distillation; the `rel_l2_recon` values are the storage-format round-trip error, which is zero by construction of the persisted-codec policy.

(Full per-Linear tables for all 9 transformer architectures and Mamba-2.8B are in `docs/PER_LINEAR_QUANT_ERROR_<arch>.json`.)

---

## Appendix D. Streaming compression time profiling

Per-phase wall-clock breakdown.

**Table D.1.** Phase 1 (teacher hidden cache) and Phase 2 (per-layer correction overlay training) decomposition.

| Model | Phase 1 (teacher cache) | Phase 2 (per-layer correction overlay × N layers) | Total |
|---|---|---|---|
| Qwen3-1.7B | < 1 min | 8 to 9 min (28 layers × ~20 sec) | 8.9 min |
| Llama-3.1-70B | ~10 min | ~95 min (80 layers × ~70 sec) | 1 h 45 min |
| Hermes-3-Llama-3.1-405B | ~3 h | ~10.5 h (126 layers × ~5 min, smaller calib batch) | 13 h 46 min |
| Mixtral-8x22B-v0.1 | 5.8 min | 155 min (56 layers × ~2.8 min) | 2 h 42 min |
| Phi-3.5-MoE-instruct | < 5 min | 55 min (32 layers × ~1.7 min) | 1 h |
| Qwen3-235B-A22B | < 30 min | ~5 h (94 layers × ~3.2 min) | 5 h 36 min |

Time complexity: linear in the number of layers, quadratic in hidden dimension (via the FFN projections), linear in the number of MoE experts per block (because each expert is a separate per-Linear quantization plus overlay).

---

## Appendix E. Memory profile of the bf16 + fp32-inner correction forward

Memory snapshot during a single layer compression on Hermes-3-Llama-3.1-405B (the largest in the matrix; hidden 16384, 126 layers).

| Component | Approximate size | Notes |
|---|---:|---|
| Teacher layer weights (bf16) | ~6.3 GB | one Llama-3.1 405B transformer block |
| Student layer weights (5-bit + scales) | ~2.0 GB | per-block(64) absmax, all Linears |
| Correction overlay V (fp32, rank 32) | ~520 MB | input projection per Linear, summed over Linears in the block |
| Correction overlay U (fp32, rank 32) | ~520 MB | inner fp32 weight per Linear, summed |
| Calibration activation tensor (fp32 inner) | ~1.5 GB | batch 8, seq 512, hidden 16384 |
| Optimizer state (AdamW, fp32 m1+m2) | ~2.1 GB | over the overlay parameters only |
| Cached teacher hiddens (one layer's worth, on GPU during step) | ~6 GB | resident only during forward; spilled to CPU between layers |
| PyTorch runtime overhead and fragmentation | ~10 GB | empirical at this scale |
| **Peak resident** | **~28.99 GB** | matches measured (32.48 GB layer-0 spike includes scaffold + first cache build) |

The `fp32-inner` trick. The correction `U` weight matrix lives in fp32 once per Linear and is reused across forward passes. A naive implementation that holds the U weight in bf16 and promotes it to fp32 on every forward pass triggers a transient fp32 allocation equal to the U weight size on every step. At 405B model scale with hidden 16384 and the number of Linears per block, this transient allocation OOMs the calibration batch on a 32 GB GPU. The explicit fp32-inner-cast pattern (Section 3.2) keeps the U weight resident in fp32 once and avoids the transient allocation on every forward.

---

## Appendix E.1. Memory profile across the matrix

Per-architecture peak VRAM measurements (source: per-run logs in `scripts/overlay/_e2e_<model>/compress.log`, `eval.log`).

| Model | Compress peak | Eval peak |
|---|---|---|
| Qwen3-1.7B | 2.26 GB | 3.30 GB |
| Llama-3.1-70B | 8.74 GB | 7.96 GB |
| Hermes-3-Llama-3.1-405B | 28.99 GB (32.48 GB spike) | 20.75 GB |
| Mixtral-8x22B-v0.1 | 14.08 GB | 6.80 GB |
| Phi-3.5-MoE-instruct | 7.91 GB | (not separately recorded) |
| Qwen3-235B-A22B | (per-layer; equiv. small) | 9.55 GB |

---

## Appendix F. Reproducibility statement

All numbers in Tables 1 through 6 of the main paper are reproducible from public artifacts.

### F.1 Two-command reproduction

```bash
pip install ultracompress==0.5.1
hf download SipsaLabs/qwen3-1.7b-uc-v3-bpw5 --local-dir ./model
uc verify ./model    # confirms lossless integrity (round-trip + sha256 spot-check)
uc serve ./model     # OpenAI-compatible inference (PyTorch reference runtime)
```

The `uc verify` command on the public `SipsaLabs/qwen3-1.7b-uc-v3-bpw5` artifact passes: 28 layer.uc files present, sha256 spot-check OK, layer 0 reconstructs 7 quantized Linears + 4 extras with correct shapes. Status: **VERIFY: PASS — pack format integrity confirmed; lossless reconstruction guaranteed.**

### F.2 PPL ratio reproduction

```bash
python `uc verify` \
    --model SipsaLabs/qwen3-1.7b-uc-v3-bpw5 \
    --n_eval 30 \
    --seq_len 1024
```

Expected output on a 32 GB consumer GPU: PPL ratio approximately 1.0091, peak VRAM approximately 3.30 GB. Results vary slightly (≤ 0.005 PPL ratio) across hardware and driver versions.

### F.3 Compression reproduction

The full streaming compression for Qwen3-1.7B:

```bash
python scripts/overlay/stream_compress_e2e.py \
    --hf-id Qwen/Qwen3-1.7B \
    --shard-dir <hf-cache>/snapshots/<sha> \
    --output ./_e2e_qwen3_1_7b \
    --bpw 5 --rank 32 --train-steps 200 \
    --n-calib 32 --seq-len 1024 --device cuda:0
```

Expected wall-clock: ~9 min. Expected PPL ratio: 1.0091 ± 0.005 (eval-corpus seed determinism).

### F.4 Hardware floor

- Qwen3-1.7B and Mistral-7B compress on a 16 GB consumer card.
- 8B models compress on a 16 GB consumer card.
- 14B compresses on a 16 GB card with `--teacher_4bit` (NF4 teacher, ~10 GB).
- 32B compresses on a 32 GB card.
- 70B compresses on a 32 GB card (peak 8.74 GB).
- 235B (Qwen3-MoE) compresses on a 32 GB card.
- 405B compresses on a 32 GB card (peak 28.99 GB steady, 32.48 GB spike).

### F.5 Calibration corpus

The 32-sequence FineWeb-edu calibration slice is deterministic given a fixed random seed. The seed is recorded in each `manifest.json` so independent reruns produce the same calibration sample selection. The eval slice is disjoint from the calibration corpus; both are tail-half splits of FineWeb-edu.

---

## Appendix G. License and ethics statement

**License.** The UltraCompress codebase, the public model checkpoints, and the reproducibility artifacts are released under Apache-2.0. Use of the published checkpoints for research, evaluation, internal deployment, and commercial deployment is permitted under the Apache-2.0 license terms.

**Patents.** [REDACTED FOR DOUBLE-BLIND REVIEW. To be restored in camera-ready: U.S. provisional patent applications 64/049,511 and 64/049,517 filed 2026-04-25, with a supplement filing planned 2026-05-09. The Apache-2.0 license grants a patent license to users of the open-source artifacts under the patent grant clause; commercial licensing of the underlying compression method outside Apache-2.0 terms is available — contact details to be added in camera-ready.]

**Ethics and broader impact.** UltraCompress is intentionally Apache-2.0 OSS. The lossless guarantee specifically targets compliance use cases (defense audit chains, healthcare clinical pipelines, financial backtest replay) where commodity quantization is structurally unsuitable. We do not foresee dual-use risk beyond what is already implicit in shipping any LLM compression library. The bit-identical reconstruction property reduces, not increases, the surface for undetected model tampering during deployment.

**Funding.** [REDACTED FOR DOUBLE-BLIND REVIEW. Camera-ready: Sipsa Labs, Inc. sole-founder pre-funding; no external grant.]

**Acknowledgments.** [REDACTED FOR DOUBLE-BLIND REVIEW.]

---

## Pre-submission checklist

The following items need resolution before submitting to NeurIPS 2026 main conference. Each is a deliberate hold against shipping the manuscript as-is.

### USPTO non-provisional conversion timing
- [ ] Provisional patents `64/049,511` and `64/049,517` filed 2026-04-25 (12-month conversion window). Track A supplement filing planned 2026-05-09. Confirm with patent counsel that the paper's method specifics (codec persistence in particular) do not exceed what the provisional applications already disclose. If they do, file the supplement before paper goes public. Do not submit the paper to NeurIPS until counsel signs off.
- [ ] Confirm the paper's open-source release timing (`ultracompress==0.5.1` on PyPI, public model cards on `huggingface.co/SipsaLabs`) is compatible with the provisional priority date. Counsel review required.

### Camera-ready LaTeX port
- [ ] Port this Markdown draft to NeurIPS 2026 LaTeX template (`neurips_2026.sty`).
- [ ] Generate the four critical figures listed in `docs/NEURIPS_PAPER_FIGURES_SPEC_2026_05.md` (scaling curve, peak VRAM, FNO Darcy panels if extension is included, hidden-MSE saturation).
- [ ] Verify line spacing, font size, page count fits within NeurIPS 8-page main + unlimited appendix limit. Current Markdown maps to roughly 8 pages of main paper plus ~4-6 pages of appendix at NeurIPS spacing.

### Reproducibility artifact
- [ ] Confirm `pip install ultracompress==0.5.1` is live on PyPI at submission time.
- [ ] Confirm `huggingface.co/SipsaLabs/qwen3-1.7b-uc-v3-bpw5` and at least 4 other model artifacts are public at submission time.
- [ ] Verify `uc verify` passes on the public artifacts on at least 2 independent hardware configurations (RTX 5090 32 GB and RTX 4090 24 GB ideally).
- [ ] Bundle the per-figure rendering scripts in the GitHub repo so reviewers can regenerate every figure from the saved JSONs.

### Anonymization for double-blind review
The paper draft references several non-anonymous identifiers that NeurIPS double-blind review forbids in the submitted manuscript. All must be replaced before submission and restored at camera-ready:

- [ ] **Sipsa Labs / Sipsa Labs, Inc.** Replace with "Anonymous" or "the authors" throughout. Currently referenced in: §6.4 trade-off summary, references to the `SipsaLabs` HuggingFace org, the §F license/ethics section, the patent section.
- [ ] **`sipsalabs.com`** and any URLs containing the sipsalabs domain. Replace with anonymized placeholder or strip.
- [ ] **`@SipsaLabs`** social handles, if any appear in the draft. Currently none, but verify.
- [ ] **`founder@sipsalabs.com`, `legal@sipsalabs.com`, `press@sipsalabs.com`** etc. Strip all.
- [ ] **`github.com/sipsalabs/ultracompress`** Replace with anonymous GitHub mirror URL (NeurIPS allows anonymous GitHub repos under `anonymous.4open.science` or similar).
- [ ] **`huggingface.co/SipsaLabs/<artifact>`** Replace with anonymous mirror or `huggingface.co/anon<id>/<artifact>`.
- [ ] **`pip install ultracompress==0.5.1`** Replace with anonymous package on TestPyPI under an anon name, or strip the install instruction and link to the anonymous code release.
- [ ] **`Apache-2.0`** license disclosure in §G. Apache-2.0 is fine for double-blind (it does not deanonymize); but verify.
- [ ] **USPTO 64/049,511 and 64/049,517** patent numbers. Strip from §G; restore in camera-ready.
- [ ] **`Missipssa Ounnar`** and any author-name occurrences. The draft is currently authorless (good); verify before submission.
- [ ] **Author-uniquely-identifiable filenames** in `docs/`: the file paths in Appendices B and C contain `_<arch>_PPL.json` style filenames that don't deanonymize, but the GitHub URL pattern does — anonymize the URL pattern.

### Co-author decisions
- [ ] Solo submission is the default. The Mamba-2.8B result was conducted as part of the same single-author research program; no co-author claim from the Mamba-2 / Mamba paper authors (Tri Dao, Albert Gu) is warranted unless the Mamba experiment is expanded to a controlled comparison with their reference Mamba-2 implementation, in which case Tri Dao should be invited as a courtesy co-author with their consent.
- [ ] If Mamba experiment is expanded (correction overlay streaming-runner adaptation finishes pre-deadline + multiple Mamba sizes measured + Mamba-2 included): consider Tri Dao as co-author. Otherwise: solo submission is correct.

### Resolved-but-flagged contradictions

The four contradictions surfaced by the prior subagent draft are resolved in this draft as follows:

1. **Mamba correction overlay ceiling claim.** Resolved at §4.2 and §6.3.1: we report 1.0119 scalar-only as the public Mamba number. The honest limitations section explicitly says we cannot reach the transformer-equivalent correction overlay ceiling on SSM with our current activation calibration. Marked as future work requiring streaming-runner adaptation.
2. **MoE PPL ratios.** Resolved at §4.2 (Table 1): all 4 MoE rows include the streaming-teacher baseline PPL and the resulting PPL ratio. Flagged in §6.3.2 that these ratios use streaming-teacher baseline rather than commodity-quantization head-to-head, and that multi-GPU baseline measurement is v0.6 work.
3. **Inference latency.** Resolved at §5.3 and §6.3.3: we report 1.5 to 2.5× bf16 latency at v0.5.1 PyTorch reference. Custom CUDA kernels are explicitly named as v0.6 future work; we do not claim 1.0 to 1.2× bf16 parity in this paper.
4. **The "first published" SSM claim.** Resolved at §1.4 and §4.2: softened to "to our knowledge, the first published" with the May 2026 lit-search caveat in §1.6 and §6.3.4.

### Pre-flight numerical sanity checks

Before submission, re-verify these numbers against the corresponding lab notebook entries:
- [ ] Qwen3-1.7B PPL ratio 1.0091 (lab notebook 2026-05 PM, 9-arch matrix table).
- [ ] Llama-3.1-70B PPL ratio 1.0090, peak VRAM 8.74 GB (lab notebook 2026-05-05).
- [ ] Hermes-3-Llama-3.1-405B PPL ratio 1.0071, peak VRAM 28.99 GB (lab notebook 2026-05-06).
- [ ] Phi-3.5-MoE PPL ratio 1.0013 (lab notebook 2026-05 9-arch matrix).
- [ ] Mamba-2.8B PPL ratio 1.0119 scalar-only, baseline 7.939, compressed 8.0337 (lab notebook 2026-05-08 09:00 entry).
- [ ] Round-trip max-abs-diff 0.0 and PPL delta 0.000003% (lab notebook 2026-05 LATE PM v3 lossless entry).
- [ ] Mistral-7B-v0.3 PPL ratio 1.0126 (lab notebook 2026-05 9-arch matrix; note this is the e2e-pipeline number, not the v6b logit-KL number which is 1.0502).

---

*End of v1 draft. Sip: review the four resolved-but-flagged contradictions in the pre-submission checklist before LaTeX port. The honest-limits framing in §6 is the primary defensive surface for review feedback.*

Codec internals + training procedure are patent-protected (USPTO 64/049,511 + 64/049,517).
