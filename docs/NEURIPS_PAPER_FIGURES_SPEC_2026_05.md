# NeurIPS 2026 Paper — Figures + Tables Spec

**Use:** Sip's checklist for generating the figures the paper draft references. Each entry: figure name, what it shows, source data, rendering tool/script, target filename, location in paper.
**Submission deadline:** 2026-05-15
**Paper draft:** `docs/PAPER_DRAFT_NEURIPS_2026.md`

---

## Critical figures (must-have for submission)

### Figure 1 — Streaming compression scaling curve

**What it shows:** PPL ratio vs model parameter count, log-log scale, four data points (Qwen3-8B, Qwen3-14B, Qwen3-32B, Qwen2.5-72B) plus the 1.7B baseline reference. Y-axis: PPL ratio (1.011 to 1.037 range plus 1.0018 reference). X-axis: parameter count (1.7B to 72B, log scale).

**Data source:**
- 4 streaming compression result JSONs at `scripts/overlay/artifacts/streaming_compression_{8b,14b,32b,72b}_smoke.json`
- 1.7B baseline from `scripts/overlay/quality_at_scale_qwen3-1.7b.json` or `docs/v18_richer_correction_results.json`

**Rendering tool:** matplotlib via Python script. Save script as `scripts/research/figures/figure_1_scaling_curve.py`.

**Target filename:** `figures/fig1_scaling_curve.pdf` (vector graphic for camera-ready) AND `figures/fig1_scaling_curve.png` (raster for previews).

**Location in paper:** Section 4.3 (Empirical scaling laws → Scaling-law fit). Caption: "Streaming compression PPL ratio vs model size on a single RTX 5090. Four-point fit shows non-monotone shape; 14B is the cleanest point at 1.011× and 32B is the worst at 1.037×."

**Annotations on the figure:**
- Marker for each model with model name above and PPL ratio next to.
- Dashed reference line at PPL ratio = 1.0 (lossless).
- Quadratic-in-log fit overlay (coefficients from §4.3).
- 95% confidence band shading.

**Dimensions:** 6 inches wide × 4 inches tall (NeurIPS single-column standard).

---

### Figure 2 — Peak VRAM during compression vs model size

**What it shows:** Peak GPU memory (GB) during compression as a function of model parameter count. Demonstrates the streaming substrate's bounded-by-one-layer property.

**Data source:** Same as Figure 1.

**Rendering tool:** matplotlib via `scripts/research/figures/figure_2_peak_vram.py`.

**Target filename:** `figures/fig2_peak_vram.pdf` + `.png`.

**Location in paper:** Section 4.3 (Empirical scaling laws), accompanies Figure 1.

**Annotations:**
- Reference line at 32 GB (RTX 5090 capacity) — shows we have headroom even at 72B.
- Reference line at 80 GB (H100 80 GB) — shows competitor hardware.
- Marker for each model with VRAM value labeled.

**Dimensions:** 6 × 4 inches.

---

### Figure 3 — FNO Darcy non-transformer transfer

**What it shows:** Output of the FNO solving 2D Darcy flow, three panels side-by-side: (a) reference fp16 output, (b) compressed BPW 6 + low-rank correction output, (c) per-pixel error (compressed - reference). Demonstrates that compression preserves PDE solution structure exactly.

**Data source:** Run `python scripts/demo/fno_compression_demo.py` with `--save_figure` flag (may need to add this flag to the script).

**Rendering tool:** Existing matplotlib code in `scripts/demo/fno_compression_demo.py`. Add `--save_figure figures/fig3_fno_darcy.pdf` argument support.

**Target filename:** `figures/fig3_fno_darcy.pdf` + `.png`.

**Location in paper:** Section 5.1 (Non-transformer extension → FNO Darcy 2D). Caption: "FNO solving 2D Darcy flow: (left) fp16 reference output, (middle) compressed BPW 6 + low-rank correction output (cosine 0.999998), (right) per-pixel error (1e-6 scale)."

**Annotations:** Colorbar shared across left and middle panels; right panel uses different colorbar with smaller scale. Panel labels (a), (b), (c).

**Dimensions:** 6 × 2 inches (single column, three panels stacked horizontally).

---

### Figure 4 — Hidden-MSE saturation effect

**What it shows:** End-to-end PPL ratio vs distillation steps on Qwen3-8B and Qwen3-32B. Shows the regression at 500 steps relative to 200 steps, demonstrating the saturation effect.

**Data source:**
- 200-step results from streaming compression smoke runs
- 500-step results from `scripts/overlay/streaming_compression_500step_*` results (per the LAB-NOTEBOOK 2026-05-04 entry)
- May need to re-instrument with intermediate 100, 300, 400 step measurements if the smoke runs only saved 200 and 500.

**Rendering tool:** matplotlib via `scripts/research/figures/figure_4_saturation.py`.

**Target filename:** `figures/fig4_hidden_mse_saturation.pdf` + `.png`.

**Location in paper:** Section 4.5 (The 32B regression and the next research push).

**Annotations:** Two curves (8B and 32B), x-axis log scale on steps, y-axis PPL ratio. Annotate the minimum point on each curve. Shaded region from 200 to 500 steps showing the regression.

**Dimensions:** 6 × 4 inches.

---

## Helpful figures (include if time)

### Figure 5 — Per-layer correction-rank ablation

**What it shows:** PPL ratio vs correction overlay rank (8, 16, 32, 64, 128) on Qwen3-8B at BPW 5.

**Data source:** Need to generate. Run streaming compression at varying ranks. Each run ~30 min on a 5090.

**Rendering tool:** matplotlib via `scripts/research/figures/figure_5_rank_ablation.py`.

**Target filename:** `figures/fig5_rank_ablation.pdf` + `.png`.

**Location in paper:** Appendix A (Per-layer correction-rank ablation).

**If data not generated by submission:** drop figure, leave Appendix A as "in progress" with planned methodology.

---

### Figure 6 — Per-bpw sweep

**What it shows:** PPL ratio vs BPW (3, 4, 5, 6, 8) on Qwen3-8B at fixed low-rank (production-tuned).

**Data source:** Need to generate. Run streaming compression at varying BPW.

**Rendering tool:** matplotlib via `scripts/research/figures/figure_6_bpw_sweep.py`.

**Target filename:** `figures/fig6_bpw_sweep.pdf` + `.png`.

**Location in paper:** Appendix B (Per-bpw sweep).

**If data not generated by submission:** drop figure, leave Appendix B as "in progress."

---

### Figure 7 — Comparison vs AWQ + HQQ at matched bit-rate

**What it shows:** PPL ratio comparison at the same model (Qwen3-8B) and roughly matched bit-rate. Bar chart with three bars: Sipsa BPW 5 vs AWQ 4 bpw vs HQQ 4 bpw.

**Data source:** Head-to-head benchmark subagent in flight (task #79). Once completes: `docs/HEAD_TO_HEAD_BENCHMARK_RESULTS_2026_05_04.md` should have the data.

**Rendering tool:** matplotlib via `scripts/research/figures/figure_7_competitive.py`.

**Target filename:** `figures/fig7_competitive.pdf` + `.png`.

**Location in paper:** Section 2 (Related work) OR Discussion §6 — frontier-comparison context.

**If data not generated by submission:** include it as a paragraph in Related Work without a figure.

---

## Tables (already in paper draft — verify before submission)

### Table 1 — Streaming compression scaling curve

Already in paper draft §4.2. Verify numbers match:
- Qwen3-8B: PPL ratio 1.028× / peak VRAM 2.26 GB
- Qwen3-14B: PPL ratio 1.011× / peak VRAM 3.37 GB
- Qwen3-32B: PPL ratio 1.037× / peak VRAM 4.85 GB
- Qwen2.5-72B: PPL ratio 1.016× / peak VRAM 8.98 GB

### Table 2 — Quadratic-in-log fit + extrapolations

Already in paper draft §4.4. Verify the extrapolated values at 100B / 200B / 1T match a fresh quadratic fit on the actual 4 streaming points + 1.7B reference.

### Table 3 — FNO Darcy results

Already in paper draft §5.1. Verify against `docs/fno_demo_results.json`:
- Cosine: 0.999998
- MSE reduction: 86.3%
- Wall-clock: 32.8 sec (NOT 29.4 — the LAB-NOTEBOOK has the canonical 32.8s)
- Reference L2: 0.498302
- Compressed L2: 0.498348

(Per the subagent's contradiction list: this is one of the 4 numbers Sip needs to reconcile across docs.)

### Table 4 — PINN failure

Already in paper draft §5.3. Verify against the cross-architecture sweep JSON (if exists at `docs/non_transformer_v18c_results.json`).

### Table 5 — Comparison vs AWQ / HQQ

Pending head-to-head benchmark subagent results.

---

## Rendering pipeline

1. Make a `figures/` subdirectory at the repo root.
2. Each figure script in `scripts/research/figures/` is independently runnable: `python scripts/research/figures/figure_N_xxx.py` produces both the PDF and PNG in `figures/`.
3. Use matplotlib defaults but override:
   - Font: serif (matches NeurIPS template). `plt.rcParams['font.family'] = 'serif'`.
   - Font size: 9 pt for body, 8 pt for tick labels, 10 pt for axis labels.
   - Line width: 1.0 default, 1.5 for main curves.
   - DPI: 300 for PNG, vector for PDF.
   - Colors: use the matplotlib `tab:blue`, `tab:orange`, `tab:green`, `tab:red` palette consistently.
4. After rendering, verify all figures pass:
   - Render under 1 sec each (no slow operations).
   - Reproducible from seed (no random elements).
   - Legible at 50% zoom on a printed page (test by printing at 2-up).
5. Bundle all figure scripts in the GitHub repo so reviewers can reproduce.

---

## Appendix tables that need data generation

These are placeholder sections in the paper draft that should be filled in before submission:

### Appendix A — Per-layer correction-rank ablation
Need: PPL ratio at rank ∈ {8, 16, 32, 64, 128} on Qwen3-8B at BPW 5 streaming. 5 runs × ~30 min = ~2.5 hours of compute.

### Appendix B — Per-bpw sweep
Need: PPL ratio at BPW ∈ {3, 4, 5, 6, 8} on Qwen3-8B at fixed low-rank (production-tuned). 5 runs × ~30 min = ~2.5 hours.

### Appendix C — Calibration-set size sweep
Need: PPL ratio at calibration size ∈ {16, 32, 64, 128, 256} samples × 2048 tokens on Qwen3-8B. 5 runs × ~30 min = ~2.5 hours.

### Appendix D — Training-step convergence
Already partial from the saturation experiment. Need: 100, 300, 400 step measurements to fill in the curve between 200 and 500.

### Appendix E — Memory profile
Already illustrative. Re-generate exact numbers via `torch.cuda.memory_summary()` during a streaming compression run. ~1 hour of work (instrumentation + run + write-up).

### Appendix F — Composition tables
Counsel-review pending: how much disclosure of composition mechanism specifics is allowable post-supplement filing on May 9. Resolve with patent counsel before filling in numbers.

### Appendix G — PINN failure analysis
Already partial from the per-Linear MSE breakdown. Need: full instrumentation of which operators in SIREN MLP fail and why. ~2 hours of work.

**Total appendix-data work: ~10-15 hours of compute + writing.** Fit into 2026-05-12 to 2026-05-14 if Sip has bandwidth.

---

## Submission package contents

Final NeurIPS submission should include:
1. `paper.pdf` (9 pages main + appendix, NeurIPS template).
2. All figures in PDF (vector) and PNG (raster).
3. Reproducibility statement files: `uc verify`, (production trainer, patent-protected), `requirements.txt`, `pyproject.toml`.
4. Pointer to public GitHub repo.
5. Pointer to public HuggingFace artifacts.
6. Funding statement (per the submission boilerplate at the bottom of the paper draft).

---

## Pre-submission checklist (from paper outline, replicated here)

- [ ] Patent counsel review of every claim in the paper.
- [ ] No internal codenames anywhere in submitted manuscript (correction overlay / scalar quantization / FRR / CHBR / DSR-Q / SP-band / PCR).
- [ ] All four streaming compression checkpoints public on HuggingFace before submission (so reviewers can reproduce).
- [ ] PyPI install verified clean (`pip install ultracompress==0.4.1` followed by smoke).
- [ ] Reproducibility check: numbers reproduce within 0.005 PPL ratio.
- [ ] Founder name correct ("Missipssa Ounnar, Sipsa Labs").
- [ ] Funding statement: solo-founder pre-funding, USPTO numbers cited.
- [ ] No personal email in submission metadata. Use founder@sipsalabs.com only.

---

*This figures spec exists so Sip doesn't have to re-derive the figure plan from the paper draft text. Each figure is independently scoped and rendered.*

Codec internals + training procedure are patent-protected (USPTO 64/049,511 + 64/049,517).
