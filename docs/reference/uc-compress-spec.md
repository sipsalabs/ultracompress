# `uc compress` design spec (v0.2 — target Q3 2026)

**Status**: 🔵 PLANNED for v0.2.

This page describes the planned `uc compress` command — self-compression of a customer-provided source model. **It is NOT in v0.1**; this page is the design spec we'll implement against.

`uc compress` is the most-requested feature in pre-launch design-partner conversations. It's also the feature most-tightly gated on the patent prosecution timeline (we hold the implementation closed until the non-provisional applications are filed in April 2027).

This feature is planned for v0.2 (Q3 2026). To register interest, email `founder@sipsalabs.com`.

---

## Goals

1. **One-command compression** of any HF Hub or local-disk transformer model
2. **Reproducibility** — every output artifact ships with a complete provenance manifest
3. **Sensible defaults** — `uc compress <model>` Just Works for typical models without per-model tuning
4. **Pluggable compression methods** — both Row-Overlay Quantization (RoQ) and shared-block parameter dispatch (shared-block) accessible via flags
5. **Resumable** — long-running compression jobs can resume after interruption

## Synopsis (planned)

```
uc compress <source-model> [--bpw FLOAT] [--method STR] [--track STR]
                            [--output-dir PATH] [--device STR]
                            [--max-batch-size INT] [--seed INT]
                            [--license STR] [--api-key STR]
```

## Options

| Option | Default | Description |
|---|---|---|
| `<source-model>` | required | HF Hub model ID (e.g., `Qwen/Qwen3-1.7B`) or local-disk path |
| `--bpw FLOAT` | `2.798` | Target bits per weight |
| `--method STR` | `track-a-row-overlay` | Compression method to apply |
| `--track STR` | `a` | Which patent-pending track: `a`, `b`, `a+b` |
| `--output-dir PATH` | `./models/<source-name>-uc<bpw>` | Where to save the compressed artifact |
| `--device STR` | `cuda:0` | PyTorch device for compression |
| `--max-batch-size INT` | `8` | Batch size for the calibration data pass |
| `--seed INT` | `42` | Deterministic seed for reproducibility |
| `--license STR` | `sipsalabs-research-eval-1.0` | License identifier to write into the manifest |
| `--api-key STR` | (env: `UC_COMPRESS_API_KEY`) | API key for the remote compression service |

## What it does (high-level flow)

```
1. Parse the source model: download from HF Hub if needed, or load from local disk
2. Validate the model is a transformer-class architecture (raise otherwise)
3. Compute the FP16 baseline metadata: parameter count, layer breakdown, etc.
4. Submit the compression job to Sipsa Labs' compression service
   (compute-intensive; not feasible on a customer's local laptop in v0.2)
5. Stream the compressed artifact back as it's produced
6. Write the artifact to disk:
     - `model.safetensors` — compressed weights
     - `tokenizer/` — pre-loaded tokenizer (copy from source)
     - `config.json` — pre-loaded config (copy from source)
     - `ultracompress.json` — provenance manifest
     - `LICENSE` — per the --license flag
7. Verify SHA-256 of the produced artifact against the manifest
8. Print a summary
```

## Example

```bash
# Default settings — compress Qwen3-1.7B to 2.798 bpw via Row-Overlay Quantization (RoQ)
uc compress Qwen/Qwen3-1.7B

# Output:
# ./models/<source-model-name>-uc<bpw>/
#   ├── model.safetensors           (1.04 GB)
#   ├── tokenizer/                  (2.7 MB, copied from source)
#   ├── config.json                 (4 KB)
#   ├── ultracompress.json          (provenance manifest)
#   └── LICENSE                     (Sipsa Labs Research and Evaluation License)
```

## shared-block parameter dispatch (shared-block) (architectural compression)

Architectural compression is the most aggressive variant; it produces a model with substantially fewer trainable parameters but requires a calibration pass on representative training data.

```bash
uc compress Qwen/Qwen3-1.7B --method shared-block --output-dir ./models/qwen3-shared-block-311x \
    --calibration-data ./calibration.jsonl
```

The calibration data is a JSONL file with prompt/response pairs representative of the customer's deployment workload. The compression service uses this to validate that the architectural-compression preserves quality on the customer's distribution.

## Combined Row-Overlay Quantization (RoQ) + shared-block parameter dispatch (shared-block)

```bash
uc compress Qwen/Qwen3-1.7B --method roq+shared-block --output-dir ./models/qwen3-uc-combo
```

Stacks shared-block parameter dispatch (shared-block)'s architectural compression with Row-Overlay Quantization (RoQ)'s quantization. ~26.7× end-to-end with 68% top-10 retention (cohort median).

## Why a remote service vs. local

Compression at our quality bar requires:

- ~10-100 GPU-hours per model (depending on size + cohort verification)
- Specialized kernels not yet pip-installable
- Calibration data from a curated cohort to maintain quality

Running this on a customer's laptop / desktop is not practical in v0.2. We'll lift the local-only requirement as the methods mature, but the v0.2 release uses a remote compression service.

The customer's source model is uploaded to the service over TLS, processed, and the result returned. Customer data never persists on Sipsa Labs's infrastructure beyond the duration of the compression job (typically minutes).

## Authentication

- Sign up at sipsalabs.com to get a `UC_COMPRESS_API_KEY`
- Free tier: 1 compression per month on models < 7B parameters
- Paid tiers: per `PRICING_CALCULATOR.md`

## Reproducibility

Every compression run is deterministic given the same seed + same source model. The `ultracompress.json` manifest captures:

- Source model SHA-256
- Compression method version
- Seed
- Calibration data SHA-256 (for shared-block parameter dispatch (shared-block) + combined)
- Compute environment fingerprint

A second `uc compress` run with the same inputs and same seed produces a byte-identical output (within GPU-arithmetic non-determinism bounds; we publish the bound).

## Resumability

For long-running compression jobs (shared-block parameter dispatch (shared-block) at 70B+ parameters, expected to take several hours), `uc compress` supports resume:

```bash
# Submit the job
uc compress Qwen/Qwen3-32B --method shared-block --bpw 2.5 --output-dir ./models/qwen3-32b-shared-block-2p5

# Job interrupted (say, network drop)
# Resume with the same command:
uc compress Qwen/Qwen3-32B --method shared-block --bpw 2.5 --output-dir ./models/qwen3-32b-shared-block-2p5
# CLI detects the partial state in the output dir, resumes from the last checkpoint
```

## Privacy + IP

- **Customer's source model**: stays customer's. We do not retain customer source models beyond the compression job duration. The current subprocessor list is available on request — email `legal@sipsalabs.com`.
- **Compressed output artifact**: customer owns it; subject to the license written into `LICENSE` by `uc compress` (defaults to Research and Evaluation License; commercial customers use a different license value).
- **Compression methods**: Sipsa Labs's IP, patent-pending. Use of `uc compress` requires acceptance of a Subscription Agreement provided by `legal@sipsalabs.com` at onboarding.
- **Calibration data**: stays customer's. Used only for the compression job; not retained.

## What this command will NOT do (v0.2)

- Compress arbitrary architectures outside the transformer family (CNN, GNN, etc.)
- Compress encoder-only models (T5, BERT) — defer to v0.3
- Compress quantization-aware-training models (already-aware models) — defer to v0.3
- Run entirely locally without internet — defer; some path forward via `--offline` flag with pre-shipped kernels

## Pricing for `uc compress`

Per `PRICING_CALCULATOR.md`:

| Plan | Cost | What you get |
|---|---|---|
| Free | $0 | 1 compression / month, models ≤ 7B params, research license only |
| Solo | $99/mo | 5 compressions / month |
| Team | $499/mo | 50 compressions / month, 5 users |
| Business | $1,999/mo | 200 compressions / month, 15 users, SLA |
| Enterprise | $5K-$50K/mo | Custom volume, custom users, SLA, audit logs, custom calibration cohorts |

For chip vendors and OEMs: separate per-device royalty model in `OEM_LICENSING_TERMS.md`.

## Migration path from v0.1

v0.1 doesn't have `uc compress`. The migration path:

- v0.1 → v0.2: `uc compress` becomes available for customers on a paid tier
- Existing v0.1 reference-model artifacts (downloaded via `uc pull`) keep working in v0.2

## Roadmap

| Feature | Target |
|---|---|
| Basic `uc compress` with Row-Overlay Quantization (RoQ) | v0.2 (Q3 2026) |
| shared-block parameter dispatch (shared-block) support (architectural compression) | v0.2 |
| Combined Row-Overlay Quantization (RoQ) + shared-block parameter dispatch (shared-block) | v0.2 |
| Resumable jobs | v0.2.1 |
| Custom calibration cohorts (enterprise tier) | v0.3 |
| Encoder-only model support (T5, BERT) | v0.3 |
| `--offline` mode (local-only compression) | v1.0+ |

---

## Open questions

1. **API surface for programmatic access**: `from ultracompress_cli import compress(model_id, bpw, ...)` or stay CLI-only? Lean toward CLI-only for v0.2; add Python API in v0.3 if customer demand justifies.

2. **Custom calibration cohorts**: how do we expose them safely? Tentatively: customers upload calibration JSONL with stricter NDA/contractual terms, similar to enterprise data-handling.

3. **Performance benchmarks per output**: should `uc compress` automatically run `uc bench` on the output before declaring success? Yes; small `--bench-on-finish` flag (default on, suppressible).

4. **Per-customer compression-method versioning**: should customers be able to pin to a specific method version (e.g., `--method-version 1.2.0`)? Yes; expose via flag.

These are open. Customer feedback welcome — file an issue at [github.com/sipsalabs/ultracompress](https://github.com/sipsalabs/ultracompress) or email `founder@sipsalabs.com`.

---

*Last updated: 2026-04-25 evening. This is a design spec for v0.2; revise as implementation progresses.*
