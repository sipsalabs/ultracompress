# Provisional Patent Application — Drafting Skeleton

**Working title:** *Fractal Iterative Knowledge Distillation with Entropy-Weighted Teacher Supervision for Extreme Language Model Compression*

**Short working title (marketing):** FractalLM

**Status:** Draft skeleton for attorney meeting. Fill in **[TODO]** sections with attorney or immediately before filing.

> **Important.** This document is a drafting aid only. It is **not** legal advice. A registered patent agent or patent attorney must review, revise, and file the final provisional application.

---

## 1. Bibliographic data (attorney fills)

- **Applicant / inventor legal name:** [TODO]
- **Residence address:** [TODO]
- **Correspondence address:** [TODO]
- **Citizenship:** [TODO]
- **Small / micro entity status:** [TODO — likely "micro" if under $206K/yr gross income]
- **Filing date:** [TODO — target 2026-04-20 Monday]
- **Docket number (attorney-assigned):** [TODO]

---

## 2. Title of the invention

**Fractal Iterative Knowledge Distillation with Entropy-Weighted Teacher Supervision for Compression of Autoregressive Language Models.**

---

## 3. Cross-references to related applications

None. This is a first filing.

---

## 4. Field of the invention

The present invention relates generally to compression of neural-network language models. More specifically, it relates to methods for training a small student neural network to reproduce the next-token prediction distribution of a large pre-trained transformer language model (a "teacher"), using a combination of (a) a parameter-shared fractal-iterative student architecture and (b) an entropy-aware weighting of the teacher's per-position supervisory signal.

---

## 5. Background of the invention

### 5.1 State of the art

Pre-trained autoregressive transformer language models (e.g. GPT-style, Qwen, Llama) contain hundreds of millions to hundreds of billions of parameters. Deploying such models on edge devices or in bandwidth-limited settings requires compression. Known approaches include:

- **Weight quantization** — reducing numerical precision (e.g. INT8, INT4). Typical compression: 2×–8×, small quality loss.
- **Low-rank factorization** — replacing weight matrices W with A·B where rank(A·B) ≪ min(dim). Typical compression: 2×–10×.
- **Structured pruning and distillation into shallower models** — e.g. DistilBERT, TinyBERT. Typical compression: 2×–6×.
- **Universal Transformer** (Dehghani et al., 2018) — iteratively applies a *single* shared transformer block with an adaptive-computation-time halting mechanism. Reduces parameter count via weight sharing but uses a single iteration scale and a learned halting rule, trained from scratch on end-to-end language modelling loss.
- **Knowledge distillation** (Hinton et al., 2015) — a student model is trained to match the soft output distribution of a teacher. Conventional distillation applies uniform weighting across positions and uses forward KL divergence.

### 5.2 Problems in the prior art

- Achieving compression ratios substantially beyond ~10× while retaining a usable fraction of teacher quality has proven difficult. Standard small students typically lose the majority of teacher top-10 agreement at compression ratios above ~50×.
- Conventional distillation weights every token position equally, even though the teacher's distribution is near-deterministic at low-entropy positions (where the student is learning almost nothing new) and richly informative at high-entropy positions (where the student most needs signal).
- Universal Transformer–style weight sharing is trained from scratch on end-to-end cross-entropy and does not leverage a fixed high-quality teacher, and does not modulate the training loss by the teacher's per-position entropy.

### 5.3 Object of the invention

The invention seeks to achieve compression ratios on the order of 150× – 750× (measured as teacher parameter count divided by student parameter count) while retaining teacher top-10 token-agreement rates of 60% or greater on a held-out natural-language corpus.

---

## 6. Summary of the invention

The invention combines two complementary mechanisms:

**Mechanism A — Fractal iterative shared-block student (FRR).**
A student neural network whose core consists of a single transformer block whose weights are repeatedly applied in a nested schedule of S *scales* × K *iterations-per-scale*. Each (scale, iteration) pair has its own small learned scalar gating coefficients (γ, β, iteration-step-size), so the same parameter tensor is re-used depth × S × K times while behavior across depth is modulated by O(S·K) learned scalars.

**Mechanism B — Entropy-weighted teacher distillation with per-token margin ranking.**
The distillation loss applied during training weights each token position by an increasing function of the teacher's predictive entropy at that position, so that high-entropy positions (where the teacher itself is uncertain and therefore carries informative distributional signal) contribute proportionally more to the gradient than low-entropy positions (where matching argmax alone suffices). A top-K margin-ranking term is added to preserve the relative ordering of the teacher's top candidate tokens.

Together, mechanisms A and B enable a student with approximately 0.6 M – 2.6 M trainable parameters to reproduce 60% – 70% of the teacher's top-10 next-token behavior at compression ratios of 180× – 740×.

---

## 7. Detailed description

### 7.1 Architectural details of the fractal student

- **Input pipeline.** Input token ids → frozen teacher embedding table → linear projection `proj_in : R^{H_outer} → R^{H_inner}` (trainable, bias-less).
- **Core (fractal body).** A single transformer block B(·, γ, β) with standard multi-head attention + gated-MLP, parameterized by weight tensor θ_B. The block is applied in a nested schedule of S = [TODO, e.g. 4] scales, each comprising K = [TODO, e.g. 7] iterations-per-scale, giving S · K = [TODO, e.g. 28] total applications of θ_B per forward pass. Each (s, k) pair has learned scalar parameters:
  - γ_s (scale gain), β_s (scale bias), α_{s,k} (iteration step-size)
  - Update rule: x ← x + α_{s,k} · (B(x, γ_s, β_s) − x).
- **Output pipeline.** `proj_out : R^{H_inner} → R^{H_outer}` (trainable, bias-less) → RMSNorm initialised from teacher's final norm (frozen or free) → frozen teacher lm_head (tied or separate).
- **Parameter budget.** For inner widths 128, 256, 384 the trainable parameter counts are approximately 0.64 M, 1.5 M, and 2.6 M respectively, corresponding to compression ratios of 740×, 311× and 180× relative to a 1.72 B-parameter teacher.

### 7.2 Training objective

For each training position i with teacher next-token logits z_T^{(i)} and student logits z_S^{(i)}:

- **Teacher entropy:**  H_i = −Σ_v softmax(z_T^{(i)})_v · log softmax(z_T^{(i)})_v
- **Per-position weight:**  w_i = (1 + H_i)^p, where p ∈ {1.0, 1.5, 2.0, ...} is an entropy-power hyperparameter.
- **Forward KL:**  L_fkl = w_i · KL( softmax(z_T^{(i)}/T) || softmax(z_S^{(i)}/T) )  with temperature schedule T(t).
- **Reverse KL:**  L_rkl = KL( softmax(z_S^{(i)}/T) || softmax(z_T^{(i)}/T) )
- **Latent-matching:**  L_lat = ||normalize(h_student^{(i)}) − normalize(h_teacher^{(i)})||²
- **Ground-truth cross-entropy:**  L_ce = CE(z_S^{(i)}, y_i)
- **Top-K margin:**  L_mrg = Σ_{k=2..K} max(0, z_S^{(i)}[r_T^{(i)}[k]] − z_S^{(i)}[r_T^{(i)}[1]] + δ), where r_T^{(i)} is the teacher's top-K index ranking.

**Total:** L = w · L_fkl + λ_rkl · L_rkl + λ_lat(t) · L_lat + λ_ce(t) · L_ce + λ_mrg(t) · w · L_mrg.

The coefficients λ_lat(t), λ_ce(t), λ_mrg(t) and temperature T(t) are scheduled functions of training step t.

### 7.3 Training schedule (non-limiting example)

- Optimiser: AdamW, weight decay 0.01, bfloat16 mixed precision.
- Learning rate: 2e-4 → 1e-5 cosine, 500-step warmup.
- Sequence length: 128 tokens; effective batch 8 sequences; 80,000 – 160,000 total steps.
- Data: FineWeb-Edu (~500 M tokens of English educational web text).

### 7.4 Inference

At inference the trained body can be combined with the full teacher lm_head, or with a separately-compressed activation-aware SVD (ASVD) head for further parameter reduction at the output.

### 7.5 Measured results (to be updated with hires-eval numbers)

| Inner width | Trainable params | Compression | all-T1 | all-T10 | quality | PPL ratio |
|---|---|---|---|---|---|---|
| 256 | 1.51 M | 311× | **[hires_results_hq5.json]** | **[...]** | **[...]** | **[...]** |
| 128 | 0.64 M | 734× | **[...]** | **[...]** | **[...]** | **[...]** |

With ASVD r=1024 compressed output head, combined end-to-end:

| Stack | Total params | Compression | all-T10 | quality |
|---|---|---|---|---|
| HQ5 h256 + ASVD r=1024 | ~3.1 M | ~555× | **[combined_stack_results_hq5.json]** | **[...]** |

---

## 8. Claims (draft — attorney to refine)

### Independent claims

**Claim 1.** A method of training a student neural network to approximate the next-token predictive distribution of a pre-trained autoregressive transformer teacher model, the method comprising:

(a) providing the student with a single transformer block B parameterised by a weight tensor θ_B;

(b) configuring a forward pass of the student to apply B to an input hidden state in a nested schedule comprising S scales and K iterations per scale, wherein for each scale s and iteration k the update is x ← x + α_{s,k} · (B(x; γ_s, β_s) − x), where γ_s, β_s, α_{s,k} are learned scalar parameters distinct for each (s, k);

(c) for each token position i of a training example, computing the teacher's predictive entropy H_i over the teacher's next-token distribution;

(d) applying to the distillation loss at position i a weighting w_i that is a monotonically increasing function of H_i; and

(e) updating θ_B and said scalar parameters α_{s,k}, γ_s, β_s by gradient descent on a combined loss comprising at least the said weighted distillation loss.

**Claim 2.** A system comprising a student neural network configured according to Claim 1, wherein the student has fewer than 1/100 the trainable parameter count of the teacher and achieves at least 60% top-10 agreement with the teacher's next-token predictions on a held-out corpus of natural-language text.

### Dependent claims (non-exhaustive)

**Claim 3.** The method of Claim 1, wherein the weighting function is w_i = (1 + H_i)^p with p ≥ 1.

**Claim 4.** The method of Claim 1, wherein the combined loss additionally includes a top-K margin-ranking term penalising the student for violating the ranking of the teacher's top-K candidate tokens.

**Claim 5.** The method of Claim 1, wherein the combined loss additionally includes a latent-matching term between a normalised intermediate hidden state of the student and a corresponding normalised hidden state of the teacher.

**Claim 6.** The method of Claim 1, wherein S ∈ {2, 3, 4, 5, 6} and K ∈ {3, 4, 5, 6, 7, 8, 9, 10}.

**Claim 7.** The method of Claim 1, wherein the student's output representation is consumed by a separately-trained low-rank factorisation of the teacher's output-projection (lm_head) weight matrix.

**Claim 8.** The method of Claim 1, wherein the coefficient λ_lat(t) of the latent-matching loss is annealed from an initial value toward zero over the course of training according to a decreasing schedule.

**Claim 9.** The method of Claim 1, wherein the distillation temperature T(t) is annealed from a value greater than 1 toward 1 over the course of training.

**Claim 10.** A non-transitory computer-readable medium storing the trained weight tensor θ_B and associated scalar parameters produced by the method of Claim 1.

---

## 9. Abstract (attorney — 150 words max)

Disclosed is a method and system for training a small student neural network to approximate the next-token predictive distribution of a large pre-trained autoregressive transformer language model. The student comprises a single shared transformer block that is applied iteratively in a nested schedule of S scales and K iterations per scale, modulated by per-(scale, iteration) learned scalar parameters. During training each token position's contribution to the distillation loss is weighted by a monotonically increasing function of the teacher's predictive entropy at that position, so that informative high-entropy positions contribute proportionally more gradient than near-deterministic low-entropy positions. Combined with latent-matching, top-K margin-ranking, and ground-truth cross-entropy auxiliary losses, the method produces student models achieving compression ratios in excess of 300× while retaining at least 60% of the teacher's top-10 next-token agreement on held-out text.

---

## 10. Drawings (list — attorney / draftsperson to produce)

1. **Fig. 1.** Overall system block diagram: input tokens → frozen teacher embed → proj_in → fractal body (S × K block applications) → proj_out → RMSNorm → lm_head → logits.
2. **Fig. 2.** Detail of the fractal body showing the update rule x ← x + α_{s,k} (B(x, γ_s, β_s) − x) at each of S × K steps.
3. **Fig. 3.** Training loop schematic showing teacher and student forward passes, per-position entropy computation, weighting, and combined loss.
4. **Fig. 4.** Schematic of the combined inference stack (compressed body + ASVD head).
5. **Fig. 5.** Measured compression-vs-quality trade-off graph across inner-width configurations.

---

## 11. Prior art to cite

- Vaswani et al., "Attention Is All You Need," 2017.
- Hinton, Vinyals, Dean, "Distilling the Knowledge in a Neural Network," 2015.
- Dehghani et al., "Universal Transformers," ICLR 2019.
- Sanh et al., "DistilBERT, a distilled version of BERT," NeurIPS Workshop 2019.
- Jiao et al., "TinyBERT: Distilling BERT for Natural Language Understanding," Findings of EMNLP 2020.
- Yuan et al., "ASVD: Activation-aware Singular Value Decomposition for Compressing Large Language Models," 2023.
- Frantar & Alistarh, "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers," ICLR 2023.

**Attorney to run a proper novelty search before filing non-provisional.**

---

## 12. Filing checklist

- [ ] Provisional specification (this document, attorney-revised) — PDF
- [ ] Claims set — separate PDF
- [ ] Abstract — one paragraph
- [ ] Drawings — 5 figures, black-and-white line art, USPTO-compliant format
- [ ] Cover sheet (SB/16 or equivalent)
- [ ] Application data sheet (ADS) — attorney prepares
- [ ] Micro entity certification — attorney prepares
- [ ] USPTO filing fee (micro entity, provisional, as of 2026: approximately $[TODO — confirm at time of filing])
- [ ] Pay and file via USPTO Patent Center
- [ ] Record application number and filing date
- [ ] Calendar 12-month deadline to file non-provisional or PCT

---

*End of skeleton.*
