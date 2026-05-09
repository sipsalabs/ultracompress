# Continuation-in-Part — Per-Linear Adaptive Bits-Per-Weight Allocation

**Parent application:** USPTO Provisional Application No. 64/049,511 (filed 2026-04-25 by Sipsa Labs, Inc.) — "Group Symmetric Quantization with Low-Rank Residual Correction for Transformer Weight Compression."

**Filer:** Sipsa Labs, Inc. (Missipssa Ounnar, founder)
**Date drafted:** 2026-05-08
**Filing target:** 2026-05-09 or 2026-05-10 (within $130 / 2-week window from parent)
**Filing fee:** $65 USPTO micro-entity continuation-in-part

---

## ABSTRACT (≤150 words)

A method for compressing transformer-based neural network weights at non-uniform bits-per-weight (bpw) within a single layer, in which an importance signal computed from the residual quantization error of a uniform-bpw calibration pass identifies one or more "bottleneck" linear projections (preferentially the key projection in grouped-query attention layers, where the projection's narrower output dimension concentrates uniform-bpw quantization error). Bottleneck projections are reallocated to a higher bpw than other projections in the same layer while maintaining a target average bpw budget across all projections. The compressed model maintains the same low-rank residual correction structure (V·U decomposition with learned alpha scalar) on each projection regardless of its assigned bpw. Storage cost over a uniform-bpw scheme is bounded by the bpw promotion delta times the fraction of bottleneck projections (typically <1% on a representative grouped-query-attention transformer).

---

## BACKGROUND

The parent application discloses Group Symmetric Quantization (GSQ) at uniform bits-per-weight, with a learned low-rank residual correction (V18-C: V ∈ ℝ^(r×d_in), U ∈ ℝ^(d_out×r), α ∈ ℝ) per linear layer to recover quality lost in quantization. The reconstructed weight is W_full = (absmax · grid[codes]).reshape(d_out, d_in) + α · U @ V.

Empirical measurements on transformer architectures employing **grouped-query attention** (GQA — Llama-3, Qwen3, Mistral, Hermes-3, etc.) reveal that the **key projection** (k_proj.weight) consistently exhibits substantially higher residual quantization error under uniform bpw compared to other projections in the same transformer block. For example, on a representative 405-billion-parameter GQA model with 16:1 query-to-key-value head ratio, k_proj quantization error averaged 9.9% above the mean of the six other projections in the same block, with stable replication across 9 sampled blocks (standard deviation tightly bounded). The structural cause is that the k_proj output dimension is narrower than other projections by the GQA factor (16x in this example), reducing the number of rows over which per-row quantization scales amortize and concentrating quantization noise in the score-function-relevant subspace Q·K^T/√d.

Existing weight-compression schemes (AWQ, GPTQ, HQQ, SpQR, NF4, AQLM) treat all linear projections within a layer uniformly with respect to bpw, even when their per-projection compression error differs systematically. The trellis-based codebooks of QTIP and EXL3 vary effective bits per weight per projection but allocate based on Hessian-derived activation importance, not on the post-quantization residual signal.

---

## SUMMARY OF THE INVENTION

The invention provides a method that:

1. Performs an initial uniform-bpw calibration quantization pass on each linear projection within a transformer layer.
2. Measures, per projection, the relative L2 norm of the quantization residual: ε(L_i) = ‖W_i − Q(W_i, bpw)‖_2 / ‖W_i‖_2.
3. Identifies one or more "bottleneck" projections satisfying ε(L_i) > τ_high · mean_j(ε(L_j)) for a layer-mean-relative threshold τ_high (preferably between 1.20 and 1.40).
4. Optionally identifies "headroom" projections satisfying ε(L_i) < τ_low · mean_j(ε(L_j)) (preferably between 0.80 and 0.90), but excluding any projection whose residual feeds directly into a softmax-normalized attention path.
5. Re-allocates bpw such that bottleneck projections are quantized at bpw + Δ_high (typically Δ_high = 1) and headroom projections at bpw − Δ_low (typically Δ_low = 1), while non-classified projections remain at the base bpw.
6. Re-runs quantization at the per-projection bpw assignment.
7. Trains the low-rank residual correction (V18-C) on each projection independently using the per-projection-bpw quantized weight as the base.
8. Stores in the per-layer manifest the per-projection bpw assignment, enabling lossless reconstruction at inference time using the GSQ + V18-C inverse formula.

The method preserves the per-layer average bpw within a small tolerance of the base bpw (typically less than 0.10 bpw average drift) by balancing Δ_high promotions against Δ_low demotions when feasible.

---

## DETAILED DESCRIPTION

### Bottleneck identification

In a preferred embodiment for grouped-query attention transformers:
- The bottleneck projection is determined deterministically as the key projection (k_proj) in any attention block where the key/value head count is strictly less than the query head count.
- The promotion is Δ_high = 1 bpw.
- No demotion is performed.

In a generalized embodiment:
- After uniform-bpw calibration quantization of all projections in layer L, the per-projection ε is computed.
- Threshold τ_high is selected adaptively as max(1.30, mean_j(ε_j) + 1.5 · std_j(ε_j)) so that promotion occurs only on statistically distinguishable outliers.
- For each promoted projection, bpw is increased to the smallest bpw value in {6, 7, 8} that reduces the projection's residual to within τ_high · mean_j(ε_j).

### Storage and inference path

Per-projection bpw is stored as an unsigned 8-bit integer in the per-projection manifest entry of the v3 binary pack format described in the parent application. The packed weight bytes use the existing bit-packed representation at the assigned per-projection bpw. Inference-time reconstruction calls the existing per-projection unpacking primitive with the manifest-stored bpw value; no inference-side software change is required to support the per-projection bpw allocation.

### Empirical evidence

In a representative experiment:
- Subject model: Qwen3-1.7B-Base (28 layers, 7 quantized projections per layer, GQA with 8 K/V heads vs 16 query heads).
- Base bpw: 5; promotion: k_proj to 6 bpw; no demotion.
- Per-layer quant_rel_l2 measured at uniform 5 bpw: mean = 0.0457 ± 0.0025.
- Per-layer quant_rel_l2 with k_proj promoted to 6 bpw: mean = 0.0409 ± 0.0007 (-10.5% layer aggregate, -55% on the promoted projection itself).
- Storage cost: +0.16% (28 layers × 64 KB per promoted projection on a 1.1 GB packed model).

### Composition with low-rank residual correction

The V18-C correction (V ∈ ℝ^(r×d_in), U ∈ ℝ^(d_out×r), α ∈ ℝ) is trained independently on each projection using the per-projection-bpw quantized base weight. Because the base bpw assignment changes only the magnitude of the quantization residual (not its shape), the V18-C training procedure is unchanged: gradient descent on a knowledge-distillation loss against the original bf16 layer's hidden-state output, with the same hyperparameters (rank, learning rate, training step count).

---

## CLAIMS

**Claim 1 (independent).** A computer-implemented method for compressing weights of a transformer neural network, comprising:

- (a) for each transformer layer of the network, performing a uniform-bits-per-weight calibration quantization pass on each linear projection within the layer;
- (b) computing, per projection, a quantization-residual metric;
- (c) identifying at least one projection within the layer for which the residual metric exceeds a threshold proportional to the mean residual metric of all projections in the same layer;
- (d) re-quantizing the identified projection at a higher bits-per-weight than the unidentified projections within the same layer;
- (e) storing per-projection bits-per-weight assignments in a per-layer manifest;
- (f) training a low-rank residual correction module on each projection using the per-projection-bpw quantized base weight as the input.

**Claim 2.** The method of claim 1 wherein the identified projection is the key projection (k_proj) of an attention block whose key/value head count is strictly less than its query head count.

**Claim 3.** The method of claim 1 wherein the higher bits-per-weight assignment in step (d) is the base bits-per-weight value plus 1.

**Claim 4.** The method of claim 1 wherein step (c) further comprises identifying one or more projections for which the residual metric is below a second threshold proportional to the mean residual metric, and step (d) further comprises re-quantizing the identified low-residual projections at a lower bits-per-weight than the base bpw.

**Claim 5.** The method of claim 4 wherein the second threshold is between 0.80 and 0.90 of the mean residual metric, and wherein no projection feeding a softmax-normalized attention path is selected for low-bpw assignment.

**Claim 6.** The method of claim 1 wherein the per-layer average bits-per-weight after the per-projection assignment differs from the base bits-per-weight by less than 0.10 bits-per-weight.

**Claim 7.** The method of claim 1 wherein the residual metric is the relative L2 norm of the difference between the original bf16 weight matrix and the dequantized weight matrix.

**Claim 8.** The method of claim 1 wherein the low-rank residual correction module of step (f) is the V·U·α decomposition disclosed in the parent application No. 64/049,511, with rank parameter r ∈ {16, 32, 48, 64}.

**Claim 9.** The method of claim 1 wherein the per-projection bits-per-weight value is stored as an unsigned 8-bit integer in the projection's manifest entry of a binary pack file format.

**Claim 10.** The method of claim 1 wherein no inference-time software modification beyond passing the per-projection bits-per-weight value to the existing unpacking primitive is required to support the per-projection bits-per-weight allocation.

**Claim 11 (independent).** A computer-readable medium storing instructions that, when executed by a processor, cause the processor to perform the method of claim 1.

**Claim 12 (independent).** A compressed transformer-model artifact comprising:
- a plurality of per-layer files;
- each per-layer file comprising a manifest specifying, for each linear projection in the layer, a quantization bits-per-weight value;
- the bits-per-weight values being non-uniform within at least one layer;
- the bits-per-weight assignment to the projections in the layer being deterministically derived from a residual quantization error signal measured during compression as recited in claim 1.

---

## STRATEGIC RATIONALE FOR FILING

1. **Empirical anchoring (today).** The k_proj bottleneck signal is replicable from public model weights (Hermes-3-405B, Qwen3-1.7B-Base) and we have direct measurements in production trainer logs.

2. **Defensibility.** The claim language is method-form (not result-form), covering the residual-driven allocation procedure rather than a specific PPL number. This makes the claim hard to design around — any system that promotes specific projections to higher bpw based on post-quantization residual error implements the method.

3. **Inference compatibility.** Step (e)+(j) ensures the claim covers the pack-format aspect: any v3-style binary pack with per-projection bpw stored in the manifest practices the invention. This pulls model artifacts (not just compressors) into the claim scope, useful for licensing.

4. **Distinguished from prior art.** AWQ (activation-magnitude scaling, uniform bpw), GPTQ (Hessian-based ordering, uniform bpw), HQQ (median-thresholded scales, uniform bpw), SpQR (outlier rows in fp16, uniform on the rest), QTIP/EXL3 (trellis codebooks, importance-weighted but not residual-driven). No prior art uses the post-quantization residual L2 signal to drive per-projection bpw allocation.

5. **Cost.** $65 micro-entity continuation-in-part fee. Same fast-track filing path as the parent (no formal claims required for provisional; this draft is for archival and Sip's review before filing).

---

## FILING CHECKLIST FOR SIP

- [ ] Pay $65 USPTO micro-entity CIP fee via Patent Center
- [ ] Upload this document as the specification (filed as PDF — convert from markdown via pandoc)
- [ ] Reference parent application 64/049,511 in the cover sheet
- [ ] Specify "Continuation-in-Part" as the application type
- [ ] Same inventor: Missipssa Ounnar
- [ ] Same assignee: Sipsa Labs, Inc.
- [ ] Filing target: 2026-05-09 (Saturday) or 2026-05-10 (Sunday) — within 2-week window
- [ ] After filing: Workbench reflects the new application within 24-72 hours

---

## NEXT-WEEK STEPS

1. **Once v1 PPL eval lands tonight** (autopipe firing): if PPL ratio <1.0030, the document goes from "drafted" to "filed in 24 hours."
2. **Honest negative result fallback:** if v1 doesn't break the PPL floor but the bottleneck cure on quant_rel_l2 still replicates, file the claim anyway — the claim is on the *method*, not on the quality outcome. The mechanism is empirically novel.
3. **v2 (data-driven):** if v1 confirms, the v2 implementation (full residual-driven policy) becomes Claim 13+ in a future continuation.
4. **Parallel patent activity:** the existing 5-provisional batch ($325 USPTO) for filing tomorrow is independent — this CIP is additional, not a substitute.
