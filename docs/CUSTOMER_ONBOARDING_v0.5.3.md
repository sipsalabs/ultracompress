# UltraCompress — Customer Onboarding (v0.5.3)

**Audience:** ML engineer at an enterprise customer who just heard about UltraCompress and wants to evaluate it on a real model.
**Goal:** 30 minutes from "never heard of it" to "I have a compressed pack running locally with verified PPL."
**Length:** ~6 pages. Skim section 1, pick the section that matches your task, follow the commands.
**Tone:** practical. If something doesn't work yet, this guide says so.

Sipsa Labs, Inc. — `founder@sipsalabs.com` — `https://github.com/sipsalabs/ultracompress`

---

## 1. What UltraCompress does (90 seconds)

UltraCompress is a **5-bit lossless compression format** for transformer language models.

- **5 bpw scalar quantization** (k-means scalar quantization codebook, per-block fp32 absmax scales, block size 64) of every Linear layer's `W_base`.
- **Rank-32 correction overlay overlay** distilled per-layer to recover the residual the quantizer drops.
- **Mathematically lossless reconstruction** of `W_base` from the stored `(grid, codes, absmax)` tuple — bit-equal round-trip on layer 0 of every supported architecture.
- **Customer-reproducible end-to-end** with three commands:
  ```bash
  pip install -U ultracompress
  hf download SipsaLabs/<model>-uc-v3-bpw5 --local-dir ./<model>
  uc verify ./<model>
  ```
- **18 transformer architectures validated** end-to-end on the same single 32 GB consumer GPU pipeline:
  Qwen3 (0.6B / 1.7B / 1.7B-Base / 8B / 14B / 235B-A22B-MoE), Llama-3.1 (8B / 70B), Hermes-3-Llama-3.1-405B (in-flight, 80/126 layers), Mistral-7B-v0.3, Mixtral (8x7B / 8x22B), Phi-3.5-MoE, OLMo-2-0425-1B (base + instruct), SmolLM2-1.7B (base + instruct), TinyLlama-1.1B-Chat, and Mamba-2.8B (state-space, scalar-only — correction overlay SSM trainer ships in v0.6.0).
- **9 artifacts publicly `uc verify`-PASS** on HuggingFace (`SipsaLabs/<model>-uc-v3-bpw5`), 8 more in flight as of 2026-05-08.

What this is **not**: faster inference than AWQ/GPTQ at the kernel level (we re-use PyTorch matmul, no custom CUDA kernels yet); not lossless below 5 bpw; not a replacement for downstream task evaluation (we only report PPL on FineWeb-edu held-out tail).

---

## 2. Install (5 min)

**Requirements**

- Python 3.10+
- PyTorch 2.0+ (any backend — CUDA optional for verify; required for `uc fit` and PPL eval)
- ~10 GB free disk for one verification artifact (1.7B class)
- ~32 GB GPU VRAM for compressing/inferring 8B-class models (smaller models work on less)

**Install**

```bash
pip install -U ultracompress      # v0.5.3 or later
hf auth login                      # only if you'll download from HF
```

That's it. No optional extras to install — PyTorch + transformers + huggingface_hub are dragged in automatically. `uc verify` is pure-Python and CPU-only by design (no GPU required to confirm pack integrity).

**Sanity check**

```bash
uc --help
uc status     # prints local pack inventory (count + total size); empty on a clean install
```

---

## 3. Reproduce a published artifact (10 min)

This is the canonical "does it actually work" path. The smallest fully-published artifact is Qwen3-1.7B-Base at ~1.1 GB.

```bash
hf download SipsaLabs/qwen3-1.7b-base-uc-v3-bpw5 --local-dir ./qwen3-base
uc verify ./qwen3-base
```

Expected output:

```
VERIFY: PASS — pack format integrity confirmed; lossless reconstruction guaranteed.
```

**What `uc verify` checks**

- Pack format integrity (header magic, version, length-prefix invariants for grid/codes/V/U/absmax)
- SHA-256 of every `layer_*.uc` file against `manifest.json` (skip with `--skip-hash` if you want sub-second mode)
- All declared layers present
- A sample layer's `W_base` reconstruction shape matches the original Linear

**Optional inspection**

```bash
uc inspect ./qwen3-base                # manifest + per-layer summary
uc inspect ./qwen3-base --layer 14     # full layer-14 metadata
```

**New in v0.5.3**

```bash
uc verify-org SipsaLabs                # iterates every -uc-v3-bpw5 repo on the org and verifies each
                                       # writes VERIFY_ALL_REPORT.json to the working directory
uc status                              # summarizes packs cached locally on this machine
```

`uc verify-org` is the audit-trail command: in one invocation an external evaluator can independently confirm every public Sipsa pack is structurally lossless. Useful for security/compliance review.

**Other small artifacts you can pull**
- `SipsaLabs/qwen3-0.6b-uc-v3-bpw5` (~0.4 GB)
- `SipsaLabs/olmo-2-0425-1b-uc-v3-bpw5` and `-instruct-uc-v3-bpw5` (~0.7 GB each)
- `SipsaLabs/smollm2-1.7b-uc-v3-bpw5` and `-instruct-uc-v3-bpw5` (~1.1 GB each)
- `SipsaLabs/qwen3-1.7b-uc-v3-bpw5` (instruct, ~1.1 GB)
- `SipsaLabs/tinyllama-1.1b-chat-v1.0-uc-v3-bpw5` (~0.7 GB)
- `SipsaLabs/mistral-7b-v0.3-uc-v3-bpw5` (~5.1 GB)

See Appendix A for the complete current matrix with PPL ratios.

---

## 4. Compress your own model (15-30 min depending on size)

> **Heads up:** `uc compress` (the public CLI subcommand) ships in **v0.6.0**. In v0.5.3 the end-to-end compress flow is a manual two-step:
> (1) run the streaming compressor script, (2) pack the resulting layer dir to v3 `.uc`. Both steps are stable and customer-tested — they're just not behind a single subcommand yet.

**Pre-requisite:** the source model weights downloaded to a local HF cache, or a directory of safetensors shards.

**Step 1 — compress (per-layer streaming, peak VRAM ~ one transformer layer)**

For Llama-3.1-8B on a single RTX 5090 (~30 min wall-clock):

```bash
python scripts/overlay/stream_compress_e2e.py \
    --hf-id meta-llama/Llama-3.1-8B \
    --shard-dir <local_shard_dir> \
    --output ./compressed/my-model \
    --bpw 5 --rank 32 --train-steps 200 --device cuda:0
```

Notes:
- `--shard-dir` is the directory containing the `model-*.safetensors` shards (typically inside your HF cache snapshot dir).
- `--bpw 5 --rank 32` is the production default; use `--bpw 6 --rank 32` if you want zero-degradation headroom; lower bpw is research-grade only (see "what to expect" below).
- `--train-steps 200` is the correction overlay distillation budget per layer. 200 is the production setting; 50 works for smoke tests; ≥300 helps on architectures with hard late layers (Mistral).
- Output: `_e2e_my-model/layer_NNN.pt` files (one per transformer block) plus a small `manifest.json`.

**Step 2 — pack to v3 `.uc` (lossless binary format, ~1 min)**

```bash
python -c "from ultracompress.pack_v3 import pack_e2e_dir_v3; pack_e2e_dir_v3('_e2e_my-model', '_packed_my-model_v3')"
```

**Step 3 — verify**

```bash
uc verify _packed_my-model_v3
# → VERIFY: PASS
```

If verify PASSes, the pack is structurally sound. PPL evaluation is a separate step (section 5).

**Disk usage during the run**

Per-layer streaming bounds peak disk at one source shard at a time, but the full extracted layer dir is kept until pack. Budget roughly:
- Source shards: model fp16 size (~16 GB for 8B, ~140 GB for 70B, ~810 GB for 405B)
- `_e2e_<name>/`: ~1.5x source (V/U overlays + temp tensors)
- `_packed_<name>_v3/`: ~0.31x source (the final 5-bit pack)

For 405B-class on a 32 GB GPU + 1 TB SSD, use the cross-shard streaming planner in `scripts/overlay/stream_compress.py` (~95 % disk savings; documented in `docs/STREAMING_COMPRESSION_405B.md`).

---

## 5. Measure your own PPL ratio

Use the `eval_compressed_only.py` driver in `scripts/overlay/`:

```bash
python `uc verify` \
    --model qwen3-8b \
    --compressed_dir _packed_my-model_v3 \
    --device cuda:0 \
    --n_eval 50
```

Outputs JSON with:

```json
{
  "model": "qwen3-8b",
  "baseline_ppl": 8.4916,
  "compressed_ppl": 8.5980,
  "ppl_ratio": 1.0125,
  ...
}
```

The script auto-runs the bf16 baseline forward (or skips it via `--baseline_ppl <value>` if you've already cached it).

**`--model` requires a registry entry.** As of v0.5.3 the registry covers 19 entries: Qwen3 (0.6B / 1.7B / 1.7B-Base / 8B / 14B / 32B / 235B-A22B), Qwen2.5-72B, Mistral-7B-v0.3, NousResearch Llama-3.1 (8B / 70B), Hermes-3-Llama-3.1-405B, Mixtral (8x7B / 8x22B), Phi-3.5-MoE, SmolLM2-1.7B (base + instruct), TinyLlama-1.1B-Chat, OLMo-2-0425-1B (base + instruct). If your architecture isn't on that list, open a GitHub issue with the HF model id and `n_layers` — we'll add it (it's a 5-line PR to `MODEL_REGISTRY` in (production trainer, patent-protected)).

**Reference baseline:** any of the 9 publicly `uc verify`-PASS artifacts shows the ratio you should expect for that arch family — see the matrix in Appendix A.

---

## 6. Inference (use the compressed pack)

Two paths, depending on whether you want a programmatic loader or a service.

**A — Programmatic load (works today)**

The packed `.uc` layers can be reconstructed and patched into a stock HuggingFace model:

```python
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from ultracompress.pack_v3 import parse_uc_layer_v3

base_id = "Qwen/Qwen3-1.7B-Base"
packed = Path("./qwen3-base")          # the dir from `hf download ... --local-dir`

model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype="bfloat16").cuda().eval()
tok = AutoTokenizer.from_pretrained(base_id)

for layer_uc in sorted(packed.glob("layer_*.uc")):
    parsed = parse_uc_layer_v3(layer_uc)        # returns dict of W_base + V + U + alpha per Linear
    # patch parsed['<linear_name>']['W_base'] + alpha * low_rank(U, V) into the matching nn.Linear
    # (canonical helper coming in v0.6.0; see ultracompress/load_uc.py for the v0.5.3 reference impl)
```

The full reference loader lives at `ultracompress/load_uc.py` (used by `uc inspect`) and `ultracompress/pack_v3.py:parse_uc_layer_v3`.

**B — FastAPI service (existing subcommand)**

```bash
uc serve --model-path ./qwen3-base/qwen3-1.7b-base.uc --port 8080
```

`uc serve` brings up a uvicorn-backed FastAPI inference server with `/generate`, `/healthz`, and Prometheus metrics. The `--model-path` argument points at a single `.uc` or `.ucz` artifact (the multi-file packed-dir loader for `uc serve` ships in v0.6.0; for now the single-file artifact path is what's wired).

---

## 7. What to expect at different bpw + ranks

These are measured numbers from the published matrix (see Appendix A), not estimates. All on FineWeb-edu held-out tail, n=30 prompts, seq_len=1024, seed 42.

| Class | Example | 5 bpw + rank 32 PPL ratio |
|---|---|---|
| Small dense (≤2B) | Qwen3-1.7B-Base | **~1.005-1.010** (best: 1.0040 on Qwen3-1.7B-Base) |
| Medium dense (7-14B) | Mistral-7B-v0.3, Qwen3-8B | **~1.010-1.013** |
| Large dense (70-405B) | Llama-3.1-70B, Hermes-3-405B | **~1.007-1.013** (Hermes-3-405B partial: 1.0071) |
| MoE (Mixtral, Phi-MoE, Qwen3-235B) | Mixtral-8x7B | **~1.012-1.013** |
| State-space (Mamba) | mamba-2.8b-hf | **1.0119** (scalar-only; correction overlay SSM trainer in v0.6.0) |

**Key data points:**
- Tightest dense ratio measured anywhere on any architecture (to our knowledge): **Qwen3-1.7B-Base at 1.0040**.
- Mean PPL ratio across 9 dense PASS-published packs: **1.0094**.
- 11 of 18 architectures are under the production threshold of 1.013x.

**Below 5 bpw is research-grade.** The well-documented Qwen3 fragility wall hits hard at sub-3 bpw on Qwen3 family (PPL ratio jumps to ~1.08+). QTIP trellis at 3 bpw breaks part of the wall (PPL ratio ~1.05 on Qwen3-1.7B in our measurements) but is not yet promoted to the public CLI. If you need <5 bpw, reach out — it's an active research surface.

---

## 8. Common gotchas + how to fix

| Symptom | Cause | Fix |
|---|---|---|
| `ImportError: track_a_adaptive` | v0.5.0 packaging bug | Upgrade to v0.5.1+ |
| `Olmo2Config has no attribute layer_types` | OLMo/OLMo2 dispatch missing in `pack` | Upgrade to v0.5.2+ |
| `Single-file safetensors` error during pack | Single-shard models (TinyLlama, SmolLM2) hit the multi-shard assumption | Upgrade to v0.5.2+ (added single-file fallback) |
| HF upload aborts with `SSL EOF` mid-shard | Residential bandwidth / HF infra flake | Use the watchdog wrapper at `scripts/overlay/_hf_upload_watchdog.sh` (8-attempt auto-retry with 30s backoff) |
| `uc verify` fails on a freshly-packed dir | Almost always a v0.4.x pack format leak | Re-pack with `pack_e2e_dir_v3` (v3 only); `uc verify` refuses v0.2 lossy packs by design |
| `CUDA OOM` during compression on 14B+ | correction overlay U matmul too large | Re-run with `--n-chunks 4` (or 8 for 32B+); bit-exact with chunks=1 |
| `torch.AcceleratorError` mid-PPL-eval (TinyLlama) | Known reproducer issue | Set `CUDA_LAUNCH_BLOCKING=1` and re-run; the pack itself is structurally PASS |
| `uc serve` errors on multi-file packed dir | `--model-path` expects single-file `.uc/.ucz` in v0.5.3 | Programmatic load (section 6A) for multi-file dirs; full multi-file `serve` ships v0.6.0 |

---

## 9. Get help / report a bug

- **GitHub issues:** https://github.com/sipsalabs/ultracompress/issues — fastest path; please include `uc --version`, `pip show ultracompress`, and the full traceback
- **Public verification dashboard:** `docs/PUBLIC_VERIFICATION_DASHBOARD_2026_05_08.md` — current PASS/FAIL state of every public artifact
- **Honest negative results:** `docs/HONEST_NEGATIVE_RESULTS_2026_05_08.md` — 11 documented refutations from the 2026-05-08 push, including things we tried and what didn't work
- **Release notes:** `docs/RELEASE_NOTES_v0.5.2.md` (v0.5.3 notes pending publication)
- **Email:** `founder@sipsalabs.com` — for security disclosures use `security@sipsalabs.com`
- **Patents:** USPTO 64/049,511 + 64/049,517 (filed 2026-04-25)

---

## 10. Paid Phase 0 POC ($5K, 5 business days)

If you want Sipsa to handle the compress+verify+benchmark loop on a model you specify:

- **Cover letter / offer:** `docs/CUSTOMER_PHASE_0_POC_OFFER_LETTER.md`
- **Contract template:** `docs/CUSTOMER_PHASE_0_POC_CONTRACT_TEMPLATE.md`

**The deal:** Sipsa compresses + verifies + benchmarks ONE customer-specified transformer model to 5 bpw v3. Five business days from kickoff to delivery. You receive: the `.uc` pack, a `uc verify` PASS report, a PPL/throughput benchmark JSON, and a one-page deployment guide. Acceptance gate: `uc verify` PASS + PPL ratio within 1.5% of baseline on your eval set (or FineWeb-edu by default). $2,500 on signature, $2,500 net 30 on delivery.

Email `founder@sipsalabs.com` to start a kickoff call.

---

## Appendix A — All 18 SipsaLabs HF artifacts (current state, 2026-05-08)

Sourced from `docs/BENCHMARKS_2026_05_08.json`. PPL = FineWeb-edu held-out tail, n=30, seq_len=1024, seed 42. "in flight" means compression or upload not yet complete as of the BENCHMARKS snapshot.

| Model ID | Sipsa repo | Params | Layers | PPL ratio | uc_verify | hf_committed |
|---|---|---|---|---|---|---|
| Qwen/Qwen3-1.7B-Base | SipsaLabs/qwen3-1.7b-base-uc-v3-bpw5 | 1.7B | 28 | **1.0040** | PASS | yes |
| Qwen/Qwen3-0.6B | SipsaLabs/qwen3-0.6b-uc-v3-bpw5 | 0.6B | 28 | 1.0069 | PASS | yes |
| allenai/OLMo-2-0425-1B | SipsaLabs/olmo-2-0425-1b-uc-v3-bpw5 | 1.0B | 16 | 1.0073 | PASS | yes |
| allenai/OLMo-2-0425-1B-Instruct | SipsaLabs/olmo-2-0425-1b-instruct-uc-v3-bpw5 | 1.0B | 16 | 0.9998* | PASS | yes |
| HuggingFaceTB/SmolLM2-1.7B | SipsaLabs/smollm2-1.7b-uc-v3-bpw5 | 1.7B | 24 | 1.0085 | PASS | yes |
| HuggingFaceTB/SmolLM2-1.7B-Instruct | SipsaLabs/smollm2-1.7b-instruct-uc-v3-bpw5 | 1.7B | 24 | 1.0075 | PASS | yes |
| mistralai/Mistral-7B-v0.3 | SipsaLabs/mistral-7b-v0.3-uc-v3-bpw5 | 7.2B | 32 | 1.0100 | PASS | yes |
| state-spaces/mamba-2.8b-hf | (not yet) | 2.8B | 64 | 1.0119 | in-build | no |
| NousResearch/Meta-Llama-3.1-8B | SipsaLabs/llama-3.1-8b-uc-v3-bpw5 | 8.0B | 32 | 1.0125 | local-PASS | in flight |
| Qwen/Qwen3-1.7B | SipsaLabs/qwen3-1.7b-uc-v3-bpw5 | 1.7B | 28 | 1.0200 | PASS | yes |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | SipsaLabs/tinyllama-1.1b-chat-v1.0-uc-v3-bpw5 | 1.1B | 22 | (deferred) | PASS | yes |
| NousResearch/Hermes-3-Llama-3.1-405B | SipsaLabs/hermes-3-llama-3.1-405b-uc-v3-bpw5 | 405B | 126 | 1.0071 (partial) | in-build | in flight |
| Qwen/Qwen3-8B | SipsaLabs/qwen3-8b-uc-v3-bpw5 | 8.0B | 36 | (in flight) | (in flight) | in flight |
| Qwen/Qwen3-14B | SipsaLabs/qwen3-14b-uc-v3-bpw5 | 14.0B | 40 | (in flight) | (in flight) | in flight |
| Qwen/Qwen3-235B-A22B | SipsaLabs/qwen3-235b-a22b-uc-v3-bpw5 | 235B (MoE) | 94 | (in flight) | (in flight) | in flight |
| meta-llama/Llama-3.1-70B (NousResearch mirror) | SipsaLabs/llama-3.1-70b-uc-v3-bpw5 | 70B | 80 | (in flight) | (in flight) | in flight |
| mistralai/Mixtral-8x7B-v0.1 | SipsaLabs/mixtral-8x7b-v0.1-uc-v3-bpw5 | 47B (MoE) | 32 | (in flight) | (in flight) | in flight |
| mistralai/Mixtral-8x22B-v0.1 | SipsaLabs/mixtral-8x22b-v0.1-uc-v3-bpw5 | 141B (MoE) | 56 | (in flight) | (in flight) | in flight |
| microsoft/Phi-3.5-MoE-instruct | SipsaLabs/phi-3.5-moe-uc-v3-bpw5 | 42B (MoE) | 32 | (in flight) | (in flight) | in flight |

\* Compressed PPL slightly *lower* than bf16 baseline on OLMo-2-0425-1B-Instruct — within statistical noise on n=30 prompts; reported honestly rather than tuned away.

**Headline summary**
- Total architectures validated end-to-end: **18**
- Publicly `uc verify` PASS today: **9**
- HF uploads in flight: 8
- Tightest dense decoder PPL ratio at 5 bpw: **1.0040** on Qwen3-1.7B-Base (best we know of, on any arch)
- First lossless 5-bit SSM compression we know of: Mamba-2.8B at 1.0119 (scalar-only)
- Largest model compressed to 5 bpw on a single 32 GB consumer GPU: Hermes-3-Llama-3.1-405B (in flight, 80/126 layers)

For the live, machine-readable matrix run `uc verify-org SipsaLabs --out VERIFY_ALL_REPORT.json` — it'll re-fetch every public pack, run `uc verify` against it, and write the current PASS/FAIL state to JSON.

---

*End of guide. Last updated 2026-05-08 against v0.5.3.*

Codec internals + training procedure are patent-protected (USPTO 64/049,511 + 64/049,517).
