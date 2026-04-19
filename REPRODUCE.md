# Reproducibility Guide

All verified numbers in this repo can be re-derived from scratch on a
single modern GPU (24 GB+ VRAM). This document lists the exact steps,
seeds, and commands used to produce every figure we publish.

## 1. Environment

| Component | Version used |
|-----------|--------------|
| OS        | Windows 11 Pro (also tested on Ubuntu 22.04) |
| Python    | 3.12.10 |
| CUDA      | 13.2 |
| PyTorch   | 2.11.0+cu128 |
| transformers | 4.57.2 |
| datasets  | 4.8.4 |
| numpy     | 2.2.6 |

Reproduce:

```bash
python -m venv .venv && .venv/Scripts/activate     # Windows
pip install -r requirements.txt
```

## 2. Teacher cache

```bash
python -c "from transformers import AutoModelForCausalLM; \
           m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-1.7B'); \
           import torch; torch.save(m.state_dict(), 'qwen3_1.7b_cache.pt')"
```

Total cache size: ~6.8 GB (fp32). Fetched once.

## 3. Training data

FineWeb-Edu, first 500 M tokens, tokenized with the Qwen3 tokenizer:

```bash
python prepare_500M_tokens.py      # produces fineweb_edu_500M_tokens.pt
```

Training windows are drawn with `torch.randint(0, 500M - SEQ_LEN)` at
each step. **No explicit train/eval split inside this file** — see
section 5 for how evaluation is handled.

## 4. Reproducing a flagship training run

HQ5 h256 (the headline 311x, 55.40% / 69.64% configuration):

```bash
python run_hq4_ceiling_break.py --h 256 --steps 80000 --device cuda:0 \
    --tag hq5_h256                   # HQ5 is produced by this script
                                     # with the HQ5-tuned hyperparams
                                     # in the file header
```

Runtime on an RTX 5090: ~6 hours. The checkpoint lands in
`checkpoints_1.7b_tinyfrr_hq5_h256/best.pt`.

## 5. Evaluations

We ship **two** eval harnesses with different rigor guarantees. Report
both numbers; they should agree within ~1 point.

### 5a. In-domain eval (hires_eval.py)

`hires_eval.py` samples 1000 starts from the tail 50 M tokens of
`fineweb_edu_500M_tokens.pt` with seed 42. The tail region is drawn
from during training, so this is **in-distribution** held-out, not
strictly disjoint. Use it for relative comparison between training
runs.

```bash
python hires_eval.py --tags hq5_h256 hq5_h128 --n 1000 --seed 42 --device cuda:1
```

### 5b. Fully-disjoint eval (wikitext_eval.py)

`wikitext_eval.py` uses the WikiText-103 test split (~245 K tokens),
which was not used in any training and is a public standard benchmark.
Seed and protocol otherwise identical.

```bash
python wikitext_eval.py --tags hq5_h256 hq5_h128 --n 1000 --seed 42 --device cuda:1
```

First run tokenizes the WikiText test split with the Qwen3 tokenizer
and caches it to `wikitext103_test_qwen3.pt` (~1 MB).

Both scripts produce bootstrap 95 % confidence intervals and
entropy-stratified buckets. Report `all_T10` with its CI as the
primary number.

## 6. Baseline comparison (standard Hinton KD)

To show that the nested-fractal + entropy-weighted loss is load-bearing,
we include a matched-parameter standard-KD baseline:

```bash
python run_baseline_distill.py --h 256 --n_layers 2 --steps 80000 \
    --device cuda:0 --tag baseline_h256_L2
python hires_eval.py --tags baseline_h256_L2 --n 1000 --device cuda:1
```

Expected delta (HQ5 vs standard KD at matched ~1.5 M params): we expect
HQ5 to outperform on both all-T10 and ppl_ratio. The baseline number is
what competitive prior work reports; beating it by the reported margin
is the contribution.

> Status as of commit HEAD: the baseline run has not yet been executed
> (both GPUs occupied by HQ6/HQ7). It will be run and its numbers added
> here on the next training-window opening.

## 7. Determinism caveats

- `torch.manual_seed`, `numpy.random.seed`, and every `torch.randint`
  call use explicit `Generator` objects with documented seeds. Given
  the same software versions and GPU driver, the training curve and
  eval numbers should match bit-for-bit.
- cuDNN non-determinism on attention kernels can cause small (< 0.1 %)
  drift in T10 between runs on different hardware. This is normal.
- The combined-stack eval (`combined_stack_eval.py`) uses seed 42 for
  its 1000-sample draw.

## 8. Integrity hashes

Once verified training runs ship, we will publish SHA-256 hashes of
the checkpoints in `CHECKPOINT_HASHES.txt` so downloaded weights can
be confirmed.
