# UltraCompress Roadmap (Updated April 12, 2026)

## PROVEN RESULTS
- [x] FRR 0.6B: 63% T10 at 60x (50K steps)
- [x] FRR 1.7B: **66% T10 at 48x** (50K steps) — ALL-TIME BEST
- [x] FRR + Q2 E2E: 53% T10 at 959x (proven end-to-end)
- [x] PHM variant: 53% T10 at 239x (4x fewer params)
- [x] Scaling confirmed: 1.7B > 0.6B quality
- [x] MEGA test: 15 modules tested, PredCoding (+7%), PHM (4x efficiency)
- [x] Multi-block: does NOT help (same quality, 11x more params)
- [x] Entropy coding: 6x lossless on Q2 weights
- [x] L2 cache inference advantage: 60x fewer VRAM reads

## RUNNING NOW
- [ ] 0.6B 100K training (step 70K/100K, 64% T10, finishing ~12:45 PM)
- [ ] **1.7B 100K training** (step 10K/100K, targeting **70%+ T10**)
- [ ] 8B model cached and ready (16 GB FP16)

## READY TO RUN (scripts built)
- [ ] 8B dual-GPU FRR (run_8b_dual_gpu.py — teacher GPU 0, student GPU 1)
- [ ] Controller hypernetwork test (input-dependent modulation)
- [ ] MoL test (Mixture of LoRA experts, token-conditional routing)
- [ ] Optimized training (2x batch, 1.5x LR, T warmup)
- [ ] Born-again distillation (3 generations, +2-4% quality)
- [ ] Speed benchmark (FRR vs teacher inference latency)
- [ ] Speculative decoding benchmark (measure 2x speedup)
- [ ] Standard eval (WikiText-2 perplexity, HellaSwag accuracy)
- [ ] FRR from scratch v2 (real LM training on FineWeb-Edu)

## NEAR-TERM (this week)
- [ ] File provisional patent Monday ($80) — 25 claims
- [ ] 8B scaling test (projected 70%+ T10)
- [ ] Submit arxiv paper (after 8B results)
- [ ] Post Show HN (after patent + paper)
- [ ] Upload compressed models to HuggingFace
- [ ] Deploy Gradio demo to HF Spaces
- [ ] Run lm-eval benchmarks (MMLU, HellaSwag, ARC)

## MID-TERM (this month)
- [ ] 70B scaling test (NF4 teacher + CPU offload)
- [ ] llama.cpp fork with FRR architecture support
- [ ] Speculative decoding product (2x inference speedup)
- [ ] First paying customer (Fiverr/direct)
- [ ] GitHub public launch + GitHub Sponsors
- [ ] pip install ultracompress on PyPI

## LONG-TERM (Q2-Q3 2026)
- [ ] YC S26 application (deadline TBD)
- [ ] From-scratch FRR training at 1B+ scale
- [ ] 100T model compression demonstration
- [ ] Enterprise API (compression-as-a-service)
- [ ] Series A fundraise ($20-25M at $60-120M)
