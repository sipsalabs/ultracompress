# Roadmap

## Shipped

- **HQ5 h256** — 311× body compression, 70.0% teacher-quality on Qwen3-1.7B. Current flagship.
- **HQ5 h128** — 734× body compression, 68.4% quality. Strongest small variant.
- **HQ4** — broke the HQ3 ceiling via inverted entropy weighting + latent decay.
- **HQ3** — 5-loss objective with confidence-weighted CE + margin.
- **Detached training launcher** — survives VS Code / terminal close, dual-GPU.

## In progress

- **HQ6 h256** — entropy_power 2.0 extension (GPU 0, ~6 h).
- **HQ6 h384** — capacity-headroom test, 180× body compression (GPU 1, ~6 h).

## Next

1. **Hires eval** (1000 stratified samples, seed 42) on HQ5 h256 for publication CIs.
2. **Combined stack**: plug HQ5 body into ASVD r=1024 head — end-to-end joint quality benchmark.
3. **Q2 weight quantization** on HQ5 body for another +16× compression.
4. **Entropy coding** on Q2 weights for an additional ~6×.
5. **Extended token horizon** — 100T projection requires ~12 GB with FRR + Q2 + entropy coding.

## Longer horizon

- Port recipe to Qwen3-7B and Llama-3.1-8B teachers.
- Structured sparsity on the single shared block for on-device inference.
- Joint body+head HQ objective (currently head and body trained separately).
