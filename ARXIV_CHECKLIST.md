# ArXiv Submission Checklist

## Paper Content
- [x] Abstract with proven results (63% T10 at 60x, 959x E2E, 1.7B scaling)
- [x] Introduction with motivation and positioning
- [x] Method section (FRR architecture, distillation, modulation)
- [x] Results table with all experiments
- [x] Scaling section with 0.6B + 1.7B data
- [x] Related work (ALBERT, Universal Transformers, Relaxed RT, SpiralFormer, Ouroboros V2)
- [x] Future work section
- [ ] Figures/diagrams (FRR architecture, scaling curve, compression comparison)
- [ ] Ablation tables (from MEGA test data)
- [ ] Training curve plots
- [ ] Conclusion updated with scaling data

## Experiments Needed
- [x] 0.6B FRR at 10K, 50K steps
- [x] 1.7B FRR at 15K steps (61% T10 at 48x)
- [x] E2E proof (FRR + Q2 = 959x)
- [x] PHM variant (53% at 239x)
- [x] MEGA test all 15 modules
- [x] Multi-block comparison (doesn't help)
- [ ] 1.7B at 50K steps (RUNNING on GPU 1)
- [ ] 0.6B at 100K steps (RUNNING on GPU 0)
- [ ] 8B scaling test (script ready, needs GPU time)
- [ ] Standard eval (WikiText-2 perplexity, HellaSwag)
- [ ] Speed benchmark (FRR vs teacher latency)

## Submission Requirements
- [ ] Convert PAPER_DRAFT.md to LaTeX
- [ ] Generate plots from log files
- [ ] Add author info (Mounir)
- [ ] Choose arXiv categories (cs.LG, cs.CL)
- [ ] File patent BEFORE submission (Monday April 13)

## Timeline
1. Monday: File patent ($80)
2. Monday/Tuesday: Finalize paper with 100K + 1.7B-50K results
3. Tuesday: Submit to arXiv
4. Wednesday: Post Show HN (paper + GitHub link)
