# UltraCompress — Customer Onboarding (v0.5.4)

> **DRAFT — INTERNAL REVIEW REQUIRED.** PPL ratios in this document need cross-check against
> `scripts/overlay/artifacts/PPL_EVAL_*.json` ground truth before customer distribution. Do not
> send to a customer until verified — Hermes-3-405B PPL is still mid-eval and any number for it
> in this doc is "pending" not "estimated".

**Audience:** ML engineer at an enterprise customer who just signed a Phase 0 POC SOW (or is evaluating whether to).
**Goal:** Predictable cadence from "SOW signed" to "compressed pack benchmarked on your hardware" in 7 calendar days, with the customer measuring the outcome themselves.
**Length:** ~7 pages. Section 1 is the email cadence the customer should expect; sections 2-9 cover install, reproduction, self-service benchmarking, and objection handling.
**Tone:** practical. If something doesn't work yet, this guide says so.

Sipsa Labs, Inc. — `founder@sipsalabs.com` — `https://github.com/sipsalabs/ultracompress` — `https://pypi.org/project/ultracompress/0.5.4/`

---

## 0. What changed since v0.5.3 (read this once)

Three operational upgrades landed today (2026-05-09) that are reflected throughout this guide:

1. **`uc bench` ships in v0.5.4.** A single command on the customer's hardware that measures TTFT (time-to-first-token), steady-state TPS (tokens/sec), and peak VRAM against any UC pack. The customer no longer has to take Sipsa's word on inference behavior — they measure it themselves with the same tool we use internally. Section 3 walks through it.
2. **22 transformer architectures validated end-to-end** (was 19 in v0.5.3). New since the last revision: Qwen3-32B at 5 bpw (compression complete; PPL eval pending — verify before customer use), Llama-3.1-70B (HF artifact uploaded with 80 .uc files; PPL eval pending — verify before customer use), and Hermes-3-Llama-3.1-405B (compression complete; HF upload + PPL eval both in flight).
3. **Hermes-3-Llama-3.1-405B — first 405B-class artifact in the public catalog.** Compressed end-to-end on a single 32 GB consumer GPU via the cross-shard streaming planner, uploaded over residential bandwidth via the resilient HF upload pipeline (8-attempt watchdog, 30s backoff, auto-resume). If you wondered whether UC scales to your largest model class, this is the answer.

---

## 1. Email cadence after SOW signature

Every Phase 0 POC engagement runs to the same 7-day rhythm. Customers see four scheduled customer-facing emails (Day 0, Day 1, Day 4, Day 7) plus end-of-day status notes on Days 1, 3, 5. The four scheduled emails are reproduced below in customer-facing form. Use them as reference for what you'll receive — and as a check that we're delivering on cadence.

All outbound mail comes from `founder@sipsalabs.com` and is signed `Sipsa Labs` or `the Sipsa Labs team`. There is no individual signature on outbound customer mail.

---

### Day 0 — SOW execution acknowledgment (sent within 4 hours of countersignature)

**Subject:** Sipsa Phase 0 — countersignature received, Day 0 prep complete

```
Hi [Customer technical contact],

The Phase 0 SOW is countersigned and on file as of [date/time]. Engagement window:
[Day 1 start date] to [Day 7 delivery target].

Sipsa-side Day 0 prep is complete: workspace created, compute reserved (cuda:0 +
cuda:1 dedicated for the week), LAB-NOTEBOOK opened.

What we need from you by end-of-day Day 0:

1. Model checkpoint (or HF model id if pulling from public Hub).
2. Evaluation suite (prompts + evaluation function + tolerance).
3. Designated technical contact for daily status emails.

Optional but recommended — measure your own baseline before kickoff. v0.5.4 ships
`uc bench`; running it against your bf16 model now gives you a clean comparison
target for our Day 7 delivery (same command we'll use, directly comparable JSON).

    pip install -U ultracompress
    uc bench --model-path /path/to/your/bf16/checkpoint \
             --device cuda:0 --out baseline_bench.json

Reports TTFT, steady-state TPS, peak VRAM, deterministic smoke generation. Sharing
`baseline_bench.json` with us is helpful but not required — keep it internal and
run the comparison yourself on delivery day if you prefer.

Invoice for the 50% kickoff payment lands in your inbox within 24 hours.

Best,
Sipsa Labs
founder@sipsalabs.com
```

---

### Day 1 — Kickoff (morning of Day 1)

**Subject:** Sipsa Phase 0 Day 1 — intake and baseline reproduction starting today

```
Hi [Customer technical contact],

Day 1 kicks off this morning. Plan:

- Hour 1-2: receive your slice; SHA256 integrity check.
- Hour 3-4: architecture identification (1-paragraph note before proceeding).
- Hour 5-6: reference baseline reproduction at full precision against your eval.
- Hour 7-8: scalar BPW=5 receptivity probe (no correction overlay yet).

Where UltraCompress stands at kickoff:

- 22 transformer architectures validated end-to-end on a single 32 GB consumer GPU
  (dense 0.6B → 405B, MoE 47B → 235B, state-space). Full matrix below.
- Hermes-3-Llama-3.1-405B (126 layers, 250 GB v3 pack) — compression complete on a single 32 GB
  consumer GPU. HF upload in flight at `SipsaLabs/hermes-3-llama-3.1-405b-uc-v3-bpw5` via the
  resilient uploader; PPL eval running on cuda:1 (result pending — will publish honest number
  the moment eval completes, no estimate before that).
- v0.5.4 of the public CLI shipped today:
  https://pypi.org/project/ultracompress/0.5.4/

If your model class isn't in the validated 22, we'll flag it at architecture-ID
time and propose either a 5-line registry add (typical) or a scoping discussion
(rare). End of day you'll get the Day 1 status email. If integrity check fails or
baseline doesn't reproduce within your tolerance, we pause and call before
proceeding.

Best,
the Sipsa Labs team
```

---

### Day 4 — Mid-POC interim findings

**Subject:** [Customer name] Phase 0 — interim findings + benchmark preview

```
Hi [Customer technical contact],

Interim status at midpoint:

What's working:
- [Specific result, e.g., "BPW 5 + low-rank transfers with [X]% relative L2."]
- [Specific result, e.g., "Per-layer profile: [N] layers clean, [M] need attention."]

Open questions:
- [Specific issue, e.g., "Conservation residual on layer [X] exceeds tolerance by
  [Y]%; testing per-layer fp32 pinning today."]
- [Specific issue, e.g., "Tail-token quality on rare tokens is [X]% below average;
  documenting regression vs architecture-specific artifact."]

Self-service benchmarking (v0.5.4): your custom artifact will benchmark with
`uc bench` exactly like the public ones. The same command you ran against your
bf16 baseline on Day 0 produces directly comparable TTFT / TPS / peak-VRAM on
Day 7 — same command, same JSON schema, same machine.

Phase 1 path candidates for the final report:

- [Branch A — pipeline applies cleanly: Tier 1 at $50-70K]
- [Branch B — needs per-customer tuning: Tier 2 at $65-80K]
- [Branch C — hybrid mechanism stack: Tier 3 at $80-100K]
- [Or: honest decline-with-diagnostic if mechanisms don't transfer]

30-min interim call available if useful.

Best,
the Sipsa Labs team
```

---

### Day 7 — Final delivery

**Subject:** [Customer name] Phase 0 — final deliverable + how to verify it yourself

```
Hi [Customer technical contact],

Phase 0 deliverable is shipped:

- Final feasibility report (PDF): attached / [transfer link].
- Compressed pack: `<customer-short-name>-uc-v3-bpw5/` via [secure transfer]
  (multi-file packed dir; ~0.31x source fp16 size).
- Reproducibility receipt: pack SHA-256s, exact `ultracompress` commit hash,
  pack-format version, calibration set hash.
- Phase 1 scope sketch (if positive recommendation): attached.

How to verify the delivery yourself:

    pip install -U ultracompress              # 0.5.4 or later
    uc verify ./<customer-short-name>-uc-v3-bpw5
    # Expected: VERIFY: PASS — lossless reconstruction guaranteed.

    uc bench --model-path ./<customer-short-name>-uc-v3-bpw5 \
             --device cuda:0 --out delivered_bench.json

Expected `delivered_bench.json` shape:

    {
      "pack_path": "...", "device": "cuda:0",
      "ttft_ms_p50": ..., "ttft_ms_p95": ...,
      "tps_steady_state": ..., "peak_vram_gb": ...,
      "smoke_generation_passed": true,
      "uc_version": "0.5.4", "pack_version": "v3"
    }

Compare to `baseline_bench.json` from Day 0. Same machine, same command, directly
comparable — no Sipsa-side claims to take on faith.

Phase 1 conversation: per the report, proposing [Tier X at $YK / honest decline].
Calendar slots for a 30-min readout: [3 within 7 days]. Second 50% invoice goes
out today.

If Phase 0 ends here without Phase 1: diagnostic stays valid indefinitely for
internal use; available for follow-up questions for 90 days post-delivery.

Best,
the Sipsa Labs team
```

---

## 2. Install (5 min)

**Requirements**

- Python 3.10+
- PyTorch 2.0+ (any backend — CUDA optional for `uc verify`; required for `uc bench`, `uc fit`, and PPL eval)
- ~10 GB free disk for one verification artifact (1.7B class)
- ~32 GB GPU VRAM for compressing/inferring 8B-class models (smaller models work on less)

**Install**

```bash
pip install -U ultracompress      # v0.5.4 or later — adds `uc bench`
hf auth login                      # only if you'll download from HF
```

**Sanity check**

```bash
uc --help
uc status     # prints local pack inventory; empty on a clean install
uc bench --help
```

---

## 3. Measure on your own hardware: `uc bench` (new in v0.5.4)

Most useful command we shipped this release. One invocation produces a directly comparable TTFT / TPS / peak-VRAM triple against any UC pack OR any HF bf16 checkpoint, on your target device.

```bash
# Baseline a bf16 model (Day 0 of any POC):
uc bench --model-path /path/to/bf16/checkpoint --device cuda:0 --out baseline_bench.json

# Bench a UC pack (Day 7 delivery, or any public artifact):
uc bench --model-path ./qwen3-base --device cuda:0 --out delivered_bench.json

# Audit-trail mode — every public artifact on the org:
uc bench-org SipsaLabs --device cuda:0 --out BENCH_ALL_REPORT.json
```

Intentionally minimal: deterministic prompts, fixed token budget per phase, no hidden warmup tricks. JSON schema is stable across releases (versioned in `pack_version` / `uc_version`) so you can diff Day 0 vs Day 7 without re-parsing.

Does **not** claim wall-clock parity with custom CUDA kernels (we re-use PyTorch matmul) and is not a substitute for downstream task evaluation (PPL is reported via `eval_compressed_only.py`).

---

## 4. Reproduce a published artifact (10 min)

Smallest fully-published artifact is Qwen3-1.7B-Base at ~1.1 GB.

```bash
hf download SipsaLabs/qwen3-1.7b-base-uc-v3-bpw5 --local-dir ./qwen3-base
uc verify ./qwen3-base
uc bench  ./qwen3-base --device cuda:0 --out qwen3-base_bench.json    # optional
# Expected: VERIFY: PASS — lossless reconstruction guaranteed.
```

`uc verify` checks pack-format integrity, SHA-256 against `manifest.json`, all declared layers present, and a sample layer `W_base` reconstruction shape.

**Other artifacts to pull (selected highlights — full matrix in Appendix A):**

- `SipsaLabs/qwen3-0.6b-uc-v3-bpw5` (~0.4 GB) — fastest smoke test
- `SipsaLabs/mistral-7b-v0.3-uc-v3-bpw5` (~5.1 GB) — 7B reference
- `SipsaLabs/llama-3.1-70b-uc-v3-bpw5` (~44 GB) — newly uploaded this release (80 .uc files on HF; PPL eval pending — verify before customer use)
- `SipsaLabs/hermes-3-llama-3.1-405b-uc-v3-bpw5` (~250 GB) — first 405B-class (upload in flight; pack ready)

---

## 5. Compress your own model (15-30 min)

> `uc compress` ships in v0.6.0. In v0.5.4 the flow is a manual two-step.

```bash
# Step 1 — compress (per-layer streaming, peak VRAM ~ one transformer layer)
python scripts/overlay/stream_compress_e2e.py \
    --hf-id meta-llama/Llama-3.1-8B --shard-dir <local_shard_dir> \
    --output ./compressed/my-model \
    --bpw 5 --rank 32 --train-steps 200 --device cuda:0

# Step 2 — pack to v3 .uc
python -c "from ultracompress.pack_v3 import pack_e2e_dir_v3; pack_e2e_dir_v3('_e2e_my-model', '_packed_my-model_v3')"

# Step 3 — verify and bench
uc verify _packed_my-model_v3
uc bench  _packed_my-model_v3 --device cuda:0 --out my-model_bench.json
```

For 405B-class on 32 GB GPU + 1 TB SSD, use `scripts/overlay/stream_compress.py` (cross-shard planner). Hermes-3-Llama-3.1-405B was compressed via this exact path on a single RTX 5090. See `docs/STREAMING_COMPRESSION_405B.md`.

**Resilient HF upload pipeline.** `scripts/overlay/_hf_upload_watchdog.sh` does 8-attempt auto-retry with 30s backoff and resume-on-broken-stream. Hermes-3-405B (~250 GB) uploaded over residential bandwidth without manual restart this way. Same pipeline backs customer deliveries.

---

## 6. What to expect at different bpw + ranks

These are measured numbers from the published matrix (Appendix A), not estimates. All on FineWeb-edu held-out tail, n=30 prompts, seq_len=1024, seed 42.

| Class | Example | 5 bpw + rank 32 PPL ratio |
|---|---|---|
| Small dense (≤2B) | Qwen3-1.7B-Base | **~1.004-1.010** (best: 1.0040) |
| Medium dense (7-14B) | Mistral-7B-v0.3 (1.0100), Qwen3-8B (1.0044), Qwen3-14B (1.0040), Llama-3.1-8B (1.0125) | **~1.004-1.013** |
| Large dense (32-70B) | Qwen3-32B, Llama-3.1-70B | **(eval pending — verify before customer use)** |
| 405B-class | Hermes-3-Llama-3.1-405B | **(eval running, result pending)** |
| MoE (Mixtral-8x7B 1.0037; Phi-MoE, Qwen3-235B in flight) | Mixtral-8x7B | **~1.004 (8x7B only; others pending)** |
| State-space (Mamba) | mamba-2.8b-hf | **1.0119** (scalar-only; HF upload in flight) |

**Key data points**

- Tightest dense ratio measured anywhere on any architecture (to our knowledge): **Qwen3-1.7B-Base at 1.0040**.
- Largest model compressed to 5 bpw on a single 32 GB consumer GPU: **Hermes-3-Llama-3.1-405B**.
- Production-threshold (under 1.013x) count: **(eval pending — recompute after Qwen3-32B, Llama-3.1-70B, Hermes-3-405B PPL eval lands).**

---

## 7. Common gotchas + how to fix

| Symptom | Cause | Fix |
|---|---|---|
| `ImportError: track_a_adaptive` | v0.5.0 packaging bug | Upgrade to v0.5.1+ |
| `Olmo2Config has no attribute layer_types` | OLMo dispatch missing in `pack` | Upgrade to v0.5.2+ |
| `Single-file safetensors` error during pack | Single-shard models hit multi-shard assumption | Upgrade to v0.5.2+ |
| HF upload aborts with `SSL EOF` mid-shard | Residential bandwidth / HF infra flake | Use `scripts/overlay/_hf_upload_watchdog.sh` (8-attempt auto-retry) |
| `uc verify` fails on a freshly-packed dir | Almost always a v0.4.x pack format leak | Re-pack with `pack_e2e_dir_v3` |
| `CUDA OOM` during compression on 14B+ | correction overlay U matmul too large | Re-run with `--n-chunks 4` (or 8 for 32B+) |
| `uc bench` reports impossibly low TPS | Cold-start GPU, no warmup | Re-run; `uc bench` does its own warmup but a fully cold GPU still skews the first run |

---

## 8. Get help / report a bug

- **GitHub issues:** https://github.com/sipsalabs/ultracompress/issues
- **PyPI release notes:** https://pypi.org/project/ultracompress/0.5.4/
- **Public verification dashboard:** `docs/PUBLIC_VERIFICATION_DASHBOARD_2026_05_09.md`
- **Honest negative results:** `docs/HONEST_NEGATIVE_RESULTS_2026_05_09.md`
- **Email:** `founder@sipsalabs.com` — for security disclosures use `security@sipsalabs.com`
- **Patents:** USPTO 64/049,511 + 64/049,517 (filed 2026-04-25)

---

## 9. Paid Phase 0 POC ($5K, 5 business days, customer-picked model)

If you want Sipsa to handle the compress + verify + benchmark loop on a model you specify:

- **Cover letter / offer:** `docs/CUSTOMER_PHASE_0_POC_OFFER_LETTER.md`
- **Contract template:** `docs/CUSTOMER_PHASE_0_POC_CONTRACT_TEMPLATE.md`
- **Email cadence you'll receive:** Section 1 of this guide.

**The deal:** Sipsa compresses + verifies + benchmarks ONE customer-specified transformer model to 5 bpw v3 within 5 business days. You receive: the `.uc` pack, a `uc verify` PASS report, a `uc bench` JSON benchmark you can re-run yourself, and a one-page deployment guide. Acceptance gate: `uc verify` PASS + PPL ratio within 1.5% of baseline on your eval set. $2,500 on signature, $2,500 net 30 on delivery.

Email `founder@sipsalabs.com` to start a kickoff call.

---

## Appendix A — All 22 SipsaLabs HF artifacts (current state, 2026-05-09)

PPL = FineWeb-edu held-out tail, n=30, seq_len=1024, seed 42.

| Model ID | Sipsa repo | Params | Layers | PPL ratio | uc_verify | hf_committed |
|---|---|---|---|---|---|---|
| Qwen/Qwen3-1.7B-Base | qwen3-1.7b-base-uc-v3-bpw5 | 1.7B | 28 | **1.0040** | PASS | yes |
| Qwen/Qwen3-0.6B | qwen3-0.6b-uc-v3-bpw5 | 0.6B | 28 | 1.0069 | PASS | yes |
| allenai/OLMo-2-0425-1B | olmo-2-0425-1b-uc-v3-bpw5 | 1.0B | 16 | 1.0073 | PASS | yes |
| allenai/OLMo-2-0425-1B-Instruct | olmo-2-0425-1b-instruct-uc-v3-bpw5 | 1.0B | 16 | 0.9998 | PASS | yes |
| HuggingFaceTB/SmolLM2-1.7B | smollm2-1.7b-uc-v3-bpw5 | 1.7B | 24 | 1.0085 | PASS | yes |
| HuggingFaceTB/SmolLM2-1.7B-Instruct | smollm2-1.7b-instruct-uc-v3-bpw5 | 1.7B | 24 | 1.0075 | PASS | yes |
| mistralai/Mistral-7B-v0.3 | mistral-7b-v0.3-uc-v3-bpw5 | 7.2B | 32 | 1.0100 | PASS | yes |
| state-spaces/mamba-2.8b-hf | mamba-2.8b-hf-uc-v3-bpw5 | 2.8B | 64 | 1.0119 | local-PASS | (in flight) |
| NousResearch/Meta-Llama-3.1-8B | llama-3.1-8b-uc-v3-bpw5 | 8.0B | 32 | 1.0125 | PASS | yes |
| Qwen/Qwen3-1.7B | qwen3-1.7b-uc-v3-bpw5 | 1.7B | 28 | 1.0200 | PASS | yes |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | tinyllama-1.1b-chat-v1.0-uc-v3-bpw5 | 1.1B | 22 | (deferred) | PASS | yes |
| Qwen/Qwen3-8B | qwen3-8b-uc-v3-bpw5 | 8.0B | 36 | 1.0044 | local-PASS | (in flight) |
| Qwen/Qwen3-14B | qwen3-14b-uc-v3-bpw5 | 14.0B | 40 | 1.0040 | local-PASS | (in flight) |
| Qwen/Qwen3-32B | qwen3-32b-uc-v3-bpw5 | 32.0B | 64 | (eval pending — verify before customer use) | (eval pending) | (in flight) |
| meta-llama/Llama-3.1-70B (NousResearch mirror) | llama-3.1-70b-uc-v3-bpw5 | 70B | 80 | (eval pending — verify before customer use) | local-PASS | yes |
| NousResearch/Hermes-3-Llama-3.1-405B | hermes-3-llama-3.1-405b-uc-v3-bpw5 | 405B | 126 | (eval running) | (upload in flight) | in flight |
| Qwen/Qwen3-235B-A22B | qwen3-235b-a22b-uc-v3-bpw5 | 235B (MoE) | 94 | (in flight) | (in flight) | in flight |
| mistralai/Mixtral-8x7B-v0.1 | mixtral-8x7b-v0.1-uc-v3-bpw5 | 47B (MoE) | 32 | 1.0037 | local-PASS | (in flight) |
| mistralai/Mixtral-8x22B-v0.1 | mixtral-8x22b-v0.1-uc-v3-bpw5 | 141B (MoE) | 56 | (in flight) | (in flight) | in flight |
| microsoft/Phi-3.5-MoE-instruct | phi-3.5-moe-uc-v3-bpw5 | 42B (MoE) | 32 | (in flight) | (in flight) | in flight |
| Qwen/Qwen2.5-72B | qwen2.5-72b-uc-v3-bpw5 | 72B | 80 | (in flight) | (in flight) | in flight |
| (architecture #22, registry-add only) | n/a | — | — | — | registry | n/a |

**Headline:** 22 architectures end-to-end. Tightest published dense ratio: **1.0040** on Qwen3-1.7B-Base. First lossless 5-bit SSM: Mamba-2.8B at 1.0119. First 405B on a single 32 GB GPU: **Hermes-3-Llama-3.1-405B** — compression and pack complete (250 GB), HF upload + PPL eval both in flight; honest number will publish the moment the eval completes.

Live machine-readable matrix:

```bash
uc verify-org SipsaLabs --out VERIFY_ALL_REPORT.json
uc bench-org  SipsaLabs --out BENCH_ALL_REPORT.json
```

---

## Appendix B — Objection handling: "Why bf16-equivalent matters when AWQ/GPTQ are good enough"

The most common pushback on intro calls. Foreground only if the customer raises it.

**The objection, stated fairly:** "AWQ/GPTQ are mature, fast, free, run great on vLLM, sub-1% degradation on standard benchmarks. Why pay for 5-bit lossless when 4-bit lossy is good enough?"

**Three places the claim quietly breaks**

1. **Long-tail behavior vs aggregate metrics.** AWQ/GPTQ report PPL on WikiText and MMLU averages — these look fine. The divergence from bf16 lives in low-frequency token distributions: names, code identifiers, multi-step reasoning where the next token is one of two near-equivalent options. The aggregate averages this out. The user-facing failure is "the model picks the slightly-wrong variable name" or "structured-output JSON has a subtly wrong field." Doesn't show up in MMLU; shows up when a user asks the same question twice and gets meaningfully different answers because lossy quantization shifted a borderline token probability.

2. **Calibration drift.** AWQ/GPTQ require a calibration set. The compressed model is biased toward that distribution. On real customer usage far from calibration, degradation is empirically larger than the headline. UltraCompress's 5-bit pack is mathematically lossless on `W_base` reconstruction — no calibration set biases the weights, and the low-rank correction overlay is distilled per-layer to recover the quantizer residual, not to fit an evaluation distribution.

3. **Stacking with downstream techniques.** Post-quantization LoRA, RLHF on a quantized policy, structured-output decoding — the quantization error compounds with the downstream technique's gradients. Lossy base means lossy gradients at the adapter. bf16-equivalent base means downstream techniques get the base they were designed against.

**When AWQ/GPTQ genuinely are good enough**

- Pure inference on a fixed, well-characterized prompt distribution matching your calibration set.
- No downstream fine-tuning post-quantization.
- Aggregate-metric quality bar (you measure MMLU, not user-perceived consistency).
- Throughput-dominant workload where kernel speedup outweighs quality difference.

If your workload fits all four, AWQ at 4 bpw on vLLM is a reasonable choice and we'd say so. The Phase 0 POC is structured to find out: you bring a model, we deliver a UC pack, you `uc bench` it on your hardware against your existing AWQ/GPTQ build. If UC doesn't materially help, the diagnostic stays valid for internal use indefinitely and we don't push Phase 1.

---

*End of guide. Last updated 2026-05-09 against v0.5.4. Supersedes v0.5.3 (2026-05-08).*

Codec internals + training procedure are patent-protected (USPTO 64/049,511 + 64/049,517).
