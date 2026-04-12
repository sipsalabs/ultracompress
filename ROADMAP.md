# UltraCompress Roadmap

## PROVEN (tested, results in hand)
- [x] FRR V1: 62% top-10 at 42x (one shared block)
- [x] FRR V2: 62% top-10 at 42x (7-scale + hidden supervision matches V1)
- [x] Genome: 63% top-10 at 37x (baseline)
- [x] Product CLI with .ucz format (working)
- [x] 8B analysis (zero cross-layer redundancy confirmed)
- [x] Grad-level LLM research (info-theoretic floor = 1.5 bit/weight)

## RUNNING NOW
- [ ] Ablation study: PHM vs Dendritic vs LoRA vs HiddenSup vs TempAnneal (on FRR)
- [ ] HWI moonshot (holographic weight interference)
- [ ] FRR text generation demo
- [ ] Breakers (Swarm + Program, fixed)

## BUILT, READY TO TEST
- [ ] FRR V3 with LoRA adapters (run_frr_v3.py)
- [ ] 8B FRR streaming distillation (run_8b_frr.py, confirmed fits in 11GB)
- [ ] Tensor Train embedding compression (tensor_train.py)

## RESEARCHED, NEED TO BUILD
- [ ] Thalamic query-biasing (from TRC2 paper — modulate Q stream)
- [ ] TRN divisive competition (sparse competitive modulation)
- [ ] Surprise-gated pathway (dynamic modulation based on input novelty)
- [ ] Activation sparsity integration (ProSparse — 89% sparse, 4.5x speedup)
- [ ] Predictive coding training (MDL objective — inherently compressed)
- [ ] PRISM phase-based networks (complex-valued, fewer params)
- [ ] FRR with DendriticFractalBlock replacing standard block
- [ ] FRR with PHMFractalBlock replacing standard block
- [ ] FRR with HyperbolicFractalBlock for Q/K projections
- [ ] Combined: FRR + PHM + LoRA + activation sparsity (the mega-stack)

## PRODUCT TRACK
- [ ] Test product CLI on 8B model
- [ ] Add Tensor Train to embedding compression stage
- [ ] Add TurboQuant-style sparse+VQ to product pipeline
- [ ] Build pre-compressed model zoo
- [ ] Launch open-source CLI on GitHub
- [ ] Write product landing page

## MOONSHOT TRACK
- [ ] 8B FRR distillation (the scaling proof)
- [ ] FRR + PHM (168x theoretical compression)
- [ ] FRR + PHM + activation sparsity (756x theoretical)
- [ ] FRR + PHM + sparsity + 2-bit quant (6000x theoretical)
- [ ] New architecture trained from scratch (not distilled)
- [ ] Publish FRR paper on arxiv

## NEUROSCIENCE TRACK
- [ ] Dendritic FRR block (more compute per param)
- [ ] Cortical column routing (thalamic from TRC2)
- [ ] Predictive coding training paradigm
- [ ] Oscillatory binding (phase-based information)
- [ ] Astrocyte modulation network (meta-network controlling main)

## BUSINESS TRACK  
- [ ] Market: $3-5B by 2027, our moat = architectural compression
- [ ] Open-source CLI + paper for credibility
- [ ] Paid API: compression-as-a-service
- [ ] Enterprise: $50-200K/yr contracts
- [ ] Pre-compressed model marketplace
