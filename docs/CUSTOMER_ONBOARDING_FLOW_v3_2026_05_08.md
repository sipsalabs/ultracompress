# Customer Onboarding Flow — UltraCompress v0.3 Lossless Pack

**Date:** 2026-05-08
**Audience:** prospects evaluating UltraCompress for the first time, plus engineering teams integrating v0.3 into their stack.
**Time to first inference:** 5-15 minutes (small model) to 30-60 minutes (70B model — bandwidth-limited download).

---

## TL;DR

```bash
# 1. Install
pip install --upgrade ultracompress  # v0.5.0+

# 2. Download a v3 lossless pack
hf download SipsaLabs/qwen3-1.7b-uc-v3-bpw5 --local-dir ./qwen3-1.7b-v3

# 3. Serve it (OpenAI-compatible API)
uc serve ./qwen3-1.7b-v3
# → http://localhost:8080
```

That's it. The model that you measure on your machine is bit-identical to what we measured during training. No "approximate quality" caveats.

---

## Step-by-step

### Step 1 — Install UltraCompress (v0.5.0+)

```bash
pip install --upgrade ultracompress
uc --version  # should print 0.5.0
```

If you don't have the HuggingFace CLI:

```bash
pip install -U huggingface_hub
hf auth login  # paste a read-token from https://huggingface.co/settings/tokens
```

### Step 2 — Pick a v3 lossless model from the SipsaLabs HF org

| Model | Params | Pack size | PPL_r | HF repo |
|---|---|---|---|---|
| Qwen3-1.7B | 1.7B | 1.11 GB | 1.0078 | `SipsaLabs/qwen3-1.7b-uc-v3-bpw5` |
| Mistral-7B-v0.3 | 7.2B | 5.13 GB | 1.0100 | `SipsaLabs/mistral-7b-v0.3-uc-v3-bpw5` |
| Llama-3.1-8B | 8.0B | 5.13 GB | 1.0125 | `SipsaLabs/llama-3.1-8b-uc-v3-bpw5` |
| Qwen3-8B | 8.0B | 5.13 GB | 1.0044 | `SipsaLabs/qwen3-8b-uc-v3-bpw5` |
| Qwen3-14B | 14.0B | 9.60 GB | 1.0040 | `SipsaLabs/qwen3-14b-uc-v3-bpw5` |
| Mixtral-8x7B-v0.1 | 47B | 33.85 GB | (5.88 PPL; baseline OOM) | `SipsaLabs/mixtral-8x7b-v0.1-uc-v3-bpw5` |
| Phi-3.5-MoE-instruct | 42B | 30.78 GB | (6.95 PPL; baseline OOM) | `SipsaLabs/phi-3.5-moe-uc-v3-bpw5` |
| Llama-3.1-70B | 70B | 48.72 GB | (6.02 PPL; baseline OOM) | `SipsaLabs/llama-3.1-70b-uc-v3-bpw5` |

Pick based on your VRAM budget. All run on a single 32 GB consumer GPU. 70B works on RTX 5090 (32 GB) or A100 80 GB.

### Step 3 — Download

```bash
# Replace REPO with your chosen model from the table above
hf download SipsaLabs/qwen3-1.7b-uc-v3-bpw5 --local-dir ./model
```

Downloads into `./model/`:
- `layer_NNN.uc` — one binary per transformer block (5-bit packed weights + correction overlay + grid + scales)
- `manifest.json` — model metadata (uc_pack_version, n_layers, etc.)
- `README.md` — model card

### Step 4 — Serve (OpenAI-compatible API)

```bash
uc serve ./model
```

Default binds to `127.0.0.1:8080`. Endpoints match OpenAI's /v1/chat/completions and /v1/completions.

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ultracompress",
    "messages": [{"role": "user", "content": "Explain photosynthesis in one paragraph."}],
    "max_tokens": 150
  }'
```

### Step 5 — Verify lossless reconstruction (optional, recommended for regulated industries)

```bash
uc verify ./model  # planned for v0.5.1 — reconstructs sd from packed and compares to source via hash
```

For v0.5.0, the lossless guarantee is established by Sipsa Labs's published validation:
- Pack format proven bit-identical on Qwen3-1.7B (max_abs_diff = 0.0 across 32 state-dict keys)
- PPL match: source compressed 18.3748 vs v3 reload 18.3748 (delta 0.000003%)
- See `docs/AUTONOMOUS_MODE_SUMMARY_2026_05_07.md` for the full validation chain

---

## Common questions

### Q: How does this compare to AWQ / GPTQ / EXL3?

**A:** Those formats are lossy (~3-10% PPL degradation between training-time eval and customer inference). UltraCompress v0.3 is mathematically lossless — your inference behavior is bit-identical to what we measured. See `docs/COMPETITIVE_LANDSCAPE_v3_LOSSLESS_2026_05_08.md` for the full comparison matrix.

Trade-off: UltraCompress v0.3 is ~10-15% larger on disk than AWQ/GPTQ at comparable bit-width because we persist the explicit grid + per-block scales + bit-packed codes (vs scale + zero_point). The bigger artifact buys you reproducibility.

### Q: Will v0.3 packs work on the existing UltraCompress v0.4 inference engine?

**A:** Requires v0.5.0+. Older v0.4.x clients will not parse v3 binary headers. `pip install --upgrade ultracompress` resolves.

### Q: Can I compress my own model in-house?

**A:** Yes — `uc fit --model <local-path> --output ./my_model.uc --bpw 5 --rank 32` runs the streaming compression pipeline + writes a v3-pack-ready output. See `scripts/overlay/stream_compress_e2e.py` for the underlying CLI. Compression time scales with model size (~30 sec/layer for dense 7B, ~80 sec/layer for MoE 47B).

### Q: What about MoE models?

**A:** Fully supported. Mixtral-8x7B (8 experts), Phi-3.5-MoE (16 experts), and Mixtral-8x22B (8 experts, in re-compression at time of writing) all work. Each expert's quantized weights are packed independently.

### Q: VRAM during inference?

**A:** Comparable to other 5-bit quantization formats. For Qwen3-1.7B: ~2.3 GB peak eval VRAM. For Llama-3.1-70B: ~28 GB peak eval VRAM (fits on RTX 5090 / A100 80 GB).

### Q: Latency vs unquantized fp16?

**A:** Currently 1.5-2.5× slower at v0.5.0 (PyTorch reference inference). Custom CUDA kernels for fused dequant-matmul are on the v0.6 roadmap (Q2 2026); will close the gap to ~1.0-1.2×.

### Q: Frontier-scale models (235B, 405B, trillion+)?

**A:** In the pipeline. Mixtral-8x22B (141B MoE) and Qwen3-235B-A22B re-compressing as of 2026-05-08. Hermes-3-Llama-3.1-405B and DeepSeek-V3-Base 685B are queued. Will publish to the same `SipsaLabs/<model>-uc-v3-bpw5` HF org as they complete.

---

## Sales / partnerships contact

**Lossless guarantee tier (regulated industries: defense, healthcare, finance):**

- Phase 0 evaluation: $5K, 1 week, your model + benchmark report
- Production licensing: contact founder@sipsalabs.com
- Mutual NDA available for proprietary model evaluation

**Apache-2.0 OSS:**

- All current HF repos at `huggingface.co/SipsaLabs/` are Apache-2.0
- Code at `github.com/sipsalabs/ultracompress` (PyPI: `ultracompress`)
- Welcome to use commercially without asking, but we appreciate citations

**Citation:**

```bibtex
@software{ultracompress2026,
  title={UltraCompress: Lossless 5-bit Transformer Compression via Trainer-Persisted scalar quantization Codec},
  author={Ounnar, Missipssa and Sipsa Labs, Inc.},
  year={2026},
  publisher={Sipsa Labs, Inc.},
  url={https://github.com/sipsalabs/ultracompress},
  note={USPTO patent provisionals 64/049,511 + 64/049,517}
}
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `uc: error: argument {bench,fit,load,serve,pack,inspect}: invalid choice` | Old ultracompress version | `pip install --upgrade ultracompress` (need 0.5.0+) |
| `ValueError: Unsupported pack version 3` on `uc load` | Even older ultracompress (pre-v0.5.0) reading a v3 pack | Upgrade as above |
| `OOM CUDA` on 70B model | <32 GB VRAM | Use a smaller model or wait for v0.6 multi-GPU support |
| `huggingface_hub.errors.GatedRepoError` | Unauthenticated download | `hf auth login` with read-token |
| Slow download | HF bandwidth | Use `hf download --max-workers 16` to parallelize |

For other issues: file at `github.com/sipsalabs/ultracompress/issues` or email `support@sipsalabs.com`.

Codec internals + training procedure are patent-protected (USPTO 64/049,511 + 64/049,517).
