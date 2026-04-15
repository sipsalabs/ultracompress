"""
Standalone checkpoint evaluator — HellaSwag, WikiText-2 PPL, T1/T10.

Works with both 0.6B and 1.7B teacher checkpoints. Low memory mode
available for running alongside training.

Usage:
  python eval_checkpoint.py checkpoints_1.7b_real_text/frr_1.7b_step10000.pt --teacher 1.7b
  python eval_checkpoint.py checkpoints_1.7b_real_text/frr_1.7b_best.pt --teacher 1.7b --device cuda:0
  python eval_checkpoint.py some_0.6b_checkpoint.pt --teacher 0.6b --hellaswag-samples 100

Options:
  --teacher 0.6b|1.7b   Which teacher to compare against (default: 1.7b)
  --device cuda:0|cuda:1 GPU to use (default: cuda:0)
  --hellaswag-samples N  Number of HellaSwag samples (default: 300)
  --skip-hellaswag       Skip HellaSwag (saves time/VRAM)
  --skip-wikitext        Skip WikiText-2 PPL
  --t10-samples N        Number of T1/T10 eval samples (default: 200)
  --low-memory           Use smaller batch sizes, fewer samples
"""
import lib.unbuffered
import argparse
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from ultracompress.inference import ModelConfig, MiniTransformer
from ultracompress.moonshot import FractalModel
from transformers import AutoTokenizer
from datasets import load_dataset


# ── Teacher configs ───────────────────────────────────────────────────
TEACHER_CONFIGS = {
    '0.6b': {
        'cache': 'qwen3_0.6b_cache.pt',
        'hidden': 1024,
        'n_heads': 16,
        'n_kv_heads': 8,
        'n_layers': 28,
        'intermediate_size': 3072,
        'vocab_size': 151936,
        'head_dim': 128,
        'hf_name': 'Qwen/Qwen3-0.6B',
    },
    '1.7b': {
        'cache': 'qwen3_1.7b_cache.pt',
        'hidden': 2048,
        'n_heads': 16,
        'n_kv_heads': 8,
        'n_layers': 28,
        'intermediate_size': 6144,
        'vocab_size': 151936,
        'head_dim': 128,
        'hf_name': 'Qwen/Qwen3-0.6B',  # tokenizer only
    },
}

HF_TO_GGUF = {
    'self_attn.q_proj.weight': 'attn_q.weight',
    'self_attn.k_proj.weight': 'attn_k.weight',
    'self_attn.v_proj.weight': 'attn_v.weight',
    'self_attn.o_proj.weight': 'attn_output.weight',
    'self_attn.q_norm.weight': 'attn_q_norm.weight',
    'self_attn.k_norm.weight': 'attn_k_norm.weight',
    'input_layernorm.weight': 'attn_norm.weight',
    'post_attention_layernorm.weight': 'ffn_norm.weight',
    'mlp.gate_proj.weight': 'ffn_gate.weight',
    'mlp.up_proj.weight': 'ffn_up.weight',
    'mlp.down_proj.weight': 'ffn_down.weight',
}


def load_teacher(teacher_name: str, device: str):
    """Load teacher model and return (teacher, embed_w, norm_w, lm_head_w, cfg)."""
    cfg = TEACHER_CONFIGS[teacher_name]
    print(f"Loading teacher ({teacher_name})...")
    wd = torch.load(cfg['cache'], weights_only=True)

    gd = {}
    gd['token_embd.weight'] = wd['model.embed_tokens.weight'].float()
    gd['output_norm.weight'] = wd.get(
        'model.norm.weight', torch.ones(cfg['hidden'])
    ).float()
    gd['output.weight'] = wd.get(
        'lm_head.weight', gd['token_embd.weight']
    ).float()
    for li in range(cfg['n_layers']):
        for h, g in HF_TO_GGUF.items():
            k = f'model.layers.{li}.{h}'
            if k in wd:
                gd[f'blk.{li}.{g}'] = wd[k].float()
    del wd

    model_cfg = ModelConfig(
        n_layers=cfg['n_layers'],
        n_heads=cfg['n_heads'],
        n_kv_heads=cfg['n_kv_heads'],
        hidden_size=cfg['hidden'],
        intermediate_size=cfg['intermediate_size'],
        vocab_size=cfg['vocab_size'],
        head_dim=cfg['head_dim'],
    )
    teacher = MiniTransformer(model_cfg, device)
    teacher.load_weights(gd)
    teacher.embed_weight = teacher.embed_weight.to(device)
    if teacher.lm_head is not None:
        teacher.lm_head = teacher.lm_head.to(device)

    embed_w = gd['token_embd.weight'].to(device)
    norm_w = gd['output_norm.weight'].to(device)
    lm_head_w = gd['output.weight'].to(device)
    del gd

    return teacher, embed_w, norm_w, lm_head_w, cfg


def load_frr(checkpoint_path: str, cfg: dict, embed_w, norm_w, lm_head_w, device: str):
    """Load FRR model from checkpoint."""
    frr = FractalModel(
        cfg['hidden'], cfg['n_heads'], 4, 7, cfg['vocab_size'], 1,
        embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w,
    ).to(device)
    state = torch.load(checkpoint_path, weights_only=True, map_location=device)
    frr.load_state_dict(state, strict=False)
    frr.eval()
    trainable = sum(p.numel() for p in frr.parameters() if p.requires_grad)
    print(f"FRR loaded: {trainable:,} trainable params")
    return frr


# ── Evaluation functions ──────────────────────────────────────────────
def eval_t1_t10(
    model, teacher, tokenizer, device: str, cfg: dict, n_samples: int = 200
) -> tuple[float, float]:
    """T1/T10 agreement with teacher on FineWeb-Edu real text."""
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu", name="sample-10BT",
        split="train", streaming=True,
    )
    ds_iter = iter(ds)

    t1_correct, t10_scores = 0, []
    model.eval()

    for i in range(n_samples):
        # Get a real text sample
        while True:
            sample = next(ds_iter)
            text = sample.get('text', '')
            if len(text) < 200:
                continue
            toks = tokenizer.encode(
                text, max_length=32, truncation=True, return_tensors='pt',
            )[0]
            if len(toks) >= 32:
                batch = toks[:32].unsqueeze(0).to(device)
                break

        with torch.no_grad():
            tl = teacher.forward(batch, max_layers=cfg['n_layers'])
            sl = model(batch)
            t_top = tl[0, -1].argmax().item()
            s_top = sl[0, -1].argmax().item()
            t_top10 = set(tl[0, -1].topk(10).indices.tolist())
            s_top10 = set(sl[0, -1].topk(10).indices.tolist())
            if t_top == s_top:
                t1_correct += 1
            t10_scores.append(len(t_top10 & s_top10) / 10)

        if (i + 1) % 50 == 0:
            t1_so_far = t1_correct / (i + 1)
            t10_so_far = sum(t10_scores) / len(t10_scores)
            print(f"    T1/T10 progress: {i + 1}/{n_samples}  "
                  f"T1={t1_so_far * 100:.1f}%  T10={t10_so_far * 100:.1f}%")

    t1 = t1_correct / n_samples
    t10 = sum(t10_scores) / len(t10_scores)
    return t1, t10


def eval_hellaswag(
    model_fn, tokenizer, device: str, n_samples: int = 300
) -> float | None:
    """HellaSwag accuracy via log-probability scoring."""
    try:
        ds_hs = load_dataset("Rowan/hellaswag", split="validation")
    except Exception as e:
        print(f"  Could not load HellaSwag: {e}")
        return None

    correct = 0
    total = 0
    t0 = time.time()

    with torch.no_grad():
        for i, sample in enumerate(ds_hs):
            if i >= n_samples:
                break
            ctx = sample['ctx']
            endings = sample['endings']
            label = int(sample['label'])

            best_score = float('-inf')
            best_idx = 0
            for j, ending in enumerate(endings):
                text = ctx + " " + ending
                tokens = tokenizer.encode(
                    text, max_length=128, truncation=True, return_tensors='pt',
                ).to(device)
                if tokens.shape[1] < 2:
                    continue
                ctx_len = len(
                    tokenizer.encode(ctx, max_length=128, truncation=True)
                )
                if ctx_len >= tokens.shape[1] - 1:
                    continue

                logits = model_fn(tokens)
                log_probs = F.log_softmax(logits[0, ctx_len - 1:-1], dim=-1)
                targets = tokens[0, ctx_len:]
                score = log_probs.gather(
                    1, targets.unsqueeze(1)
                ).mean().item()

                if score > best_score:
                    best_score = score
                    best_idx = j

            if best_idx == label:
                correct += 1
            total += 1

            if (i + 1) % 50 == 0:
                acc = correct / total
                elapsed = time.time() - t0
                print(f"    HellaSwag progress: {i + 1}/{n_samples}  "
                      f"acc={acc * 100:.1f}%  ({elapsed:.0f}s)")

    return correct / total if total > 0 else None


def eval_wikitext_ppl(
    model_fn, tokenizer, device: str, n_samples: int = 100, seq_len: int = 128
) -> float | None:
    """WikiText-2 perplexity."""
    try:
        ds_wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [t for t in ds_wt['text'] if len(t) > 100][:n_samples]
    except Exception as e:
        print(f"  Could not load WikiText-2: {e}")
        return None

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            tokens = tokenizer.encode(
                text, max_length=seq_len + 1, truncation=True, return_tensors='pt',
            )
            if tokens.shape[1] < 10:
                continue
            tokens = tokens.to(device)
            inputs = tokens[:, :-1]
            targets = tokens[:, 1:]
            logits = model_fn(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction='sum',
            )
            total_loss += loss.item()
            total_tokens += targets.numel()

    if total_tokens == 0:
        return None
    avg_loss = total_loss / total_tokens
    return math.exp(min(avg_loss, 100))


# ── Main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate FRR checkpoint")
    parser.add_argument("checkpoint", help="Path to .pt checkpoint file")
    parser.add_argument("--teacher", default="1.7b", choices=["0.6b", "1.7b"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hellaswag-samples", type=int, default=300)
    parser.add_argument("--t10-samples", type=int, default=200)
    parser.add_argument("--skip-hellaswag", action="store_true")
    parser.add_argument("--skip-wikitext", action="store_true")
    parser.add_argument("--low-memory", action="store_true",
                        help="Use fewer samples and smaller batches")
    args = parser.parse_args()

    if args.low_memory:
        args.hellaswag_samples = min(args.hellaswag_samples, 100)
        args.t10_samples = min(args.t10_samples, 50)

    if not os.path.exists(args.checkpoint):
        print(f"Error: checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    print("=" * 70)
    print(f"CHECKPOINT EVALUATION")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Teacher: {args.teacher}")
    print(f"  Device: {args.device}")
    print("=" * 70)

    # Load teacher
    teacher, embed_w, norm_w, lm_head_w, cfg = load_teacher(
        args.teacher, args.device
    )
    tokenizer = AutoTokenizer.from_pretrained(
        cfg['hf_name'], trust_remote_code=True
    )

    # Load FRR
    frr = load_frr(
        args.checkpoint, cfg, embed_w, norm_w, lm_head_w, args.device
    )

    results = {}
    t_total = time.time()

    # ── T1/T10 ────────────────────────────────────────────────────
    print(f"\n--- T1/T10 Agreement ({args.t10_samples} samples) ---")
    t1, t10 = eval_t1_t10(
        frr, teacher, tokenizer, args.device, cfg, n_samples=args.t10_samples
    )
    results['t1'] = t1
    results['t10'] = t10
    print(f"  T1: {t1 * 100:.1f}%  T10: {t10 * 100:.1f}%")

    # ── HellaSwag ─────────────────────────────────────────────────
    if not args.skip_hellaswag:
        print(f"\n--- HellaSwag ({args.hellaswag_samples} samples) ---")

        # Teacher baseline
        print("  Teacher:")
        teacher_hs = eval_hellaswag(
            lambda t: teacher.forward(t, max_layers=cfg['n_layers']),
            tokenizer, args.device, n_samples=args.hellaswag_samples,
        )
        if teacher_hs is not None:
            results['teacher_hellaswag'] = teacher_hs
            print(f"  Teacher HellaSwag: {teacher_hs * 100:.1f}%")

        # FRR
        print("  FRR:")
        frr_hs = eval_hellaswag(
            lambda t: frr(t),
            tokenizer, args.device, n_samples=args.hellaswag_samples,
        )
        if frr_hs is not None:
            results['frr_hellaswag'] = frr_hs
            print(f"  FRR HellaSwag: {frr_hs * 100:.1f}%")
            if teacher_hs is not None and teacher_hs > 0:
                retention = frr_hs / teacher_hs * 100
                results['hellaswag_retention'] = retention
                print(f"  Retention: {retention:.1f}%")

    # ── WikiText-2 PPL ────────────────────────────────────────────
    if not args.skip_wikitext:
        print(f"\n--- WikiText-2 Perplexity ---")

        teacher_ppl = eval_wikitext_ppl(
            lambda t: teacher.forward(t, max_layers=cfg['n_layers']),
            tokenizer, args.device,
        )
        if teacher_ppl is not None:
            results['teacher_ppl'] = teacher_ppl
            print(f"  Teacher PPL: {teacher_ppl:.1f}")

        frr_ppl = eval_wikitext_ppl(
            lambda t: frr(t), tokenizer, args.device,
        )
        if frr_ppl is not None:
            results['frr_ppl'] = frr_ppl
            print(f"  FRR PPL: {frr_ppl:.1f}")
            if teacher_ppl is not None:
                ratio = frr_ppl / teacher_ppl
                results['ppl_ratio'] = ratio
                print(f"  PPL ratio (lower=better): {ratio:.2f}x")

    # ── Summary ───────────────────────────────────────────────────
    total_time = time.time() - t_total
    print(f"\n{'=' * 70}")
    print(f"EVALUATION SUMMARY — {os.path.basename(args.checkpoint)}")
    print(f"{'=' * 70}")
    print(f"  Teacher: Qwen3-{args.teacher}")
    print(f"  T1: {results.get('t1', 0) * 100:.1f}%  "
          f"T10: {results.get('t10', 0) * 100:.1f}%")
    if 'frr_hellaswag' in results:
        print(f"  HellaSwag: {results['frr_hellaswag'] * 100:.1f}%  "
              f"(teacher: {results.get('teacher_hellaswag', 0) * 100:.1f}%, "
              f"retention: {results.get('hellaswag_retention', 0):.1f}%)")
    if 'frr_ppl' in results:
        print(f"  WikiText-2 PPL: {results['frr_ppl']:.1f}  "
              f"(teacher: {results.get('teacher_ppl', 0):.1f})")
    print(f"  Eval time: {total_time:.0f}s")
    print(f"\n  Records to beat:")
    print(f"    T1=46% (1.7B 50K), T10=67% (1.7B random 100K)")
    print(f"    HellaSwag retention: 83.3% (0.6B 100K)")
    print(f"    FRR BEATS teacher PPL: 1614 < 2404 (0.6B 100K)")

    # Save results
    results_file = args.checkpoint.replace('.pt', '_eval.json')
    import json
    with open(results_file, 'w') as f:
        json.dump({k: float(v) for k, v in results.items()}, f, indent=2)
    print(f"\n  Results saved: {results_file}")


if __name__ == "__main__":
    main()
