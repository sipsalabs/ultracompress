"""
Diagnostic evaluation — deeper analysis of checkpoint quality.

Runs on CPU or specified GPU. Designed to run alongside training.
Tests:
  1. T1/T10 with high sample count (stable readings)
  2. Per-position T10 (does agreement vary by position?)
  3. Temperature sensitivity (how does T affect agreement?)
  4. Top-K agreement for K=1,3,5,10,20,50
  5. Token entropy analysis (what does the student struggle with?)

Usage:
  python run_diagnostic_eval.py checkpoints_1.7b_real_text/frr_1.7b_best.pt --teacher 1.7b
  python run_diagnostic_eval.py checkpoints_1.7b_real_text/frr_1.7b_step10000.pt --teacher 1.7b --device cuda:0
"""
import lib.unbuffered
import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict

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
        'hidden': 1024, 'n_heads': 16, 'n_kv_heads': 8, 'n_layers': 28,
        'intermediate_size': 3072, 'vocab_size': 151936, 'head_dim': 128,
        'ff_mult': 1,
    },
    '1.7b': {
        'cache': 'qwen3_1.7b_cache.pt',
        'hidden': 2048, 'n_heads': 16, 'n_kv_heads': 8, 'n_layers': 28,
        'intermediate_size': 6144, 'vocab_size': 151936, 'head_dim': 128,
        'ff_mult': 1,
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
    """Load teacher model."""
    cfg = TEACHER_CONFIGS[teacher_name]
    print(f"Loading teacher ({teacher_name})...")
    wd = torch.load(cfg['cache'], weights_only=True)

    gd = {}
    gd['token_embd.weight'] = wd['model.embed_tokens.weight'].float()
    gd['output_norm.weight'] = wd.get('model.norm.weight', torch.ones(cfg['hidden'])).float()
    gd['output.weight'] = wd.get('lm_head.weight', gd['token_embd.weight']).float()
    for li in range(cfg['n_layers']):
        for h, g in HF_TO_GGUF.items():
            k = f'model.layers.{li}.{h}'
            if k in wd:
                gd[f'blk.{li}.{g}'] = wd[k].float()
    del wd

    model_cfg = ModelConfig(
        n_layers=cfg['n_layers'], n_heads=cfg['n_heads'], n_kv_heads=cfg['n_kv_heads'],
        hidden_size=cfg['hidden'], intermediate_size=cfg['intermediate_size'],
        vocab_size=cfg['vocab_size'], head_dim=cfg['head_dim'],
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


def load_student(checkpoint_path: str, cfg: dict, embed_w, norm_w, lm_head_w, device: str):
    """Load FRR student from checkpoint."""
    model = FractalModel(
        cfg['hidden'], cfg['n_heads'], 4, 7, cfg['vocab_size'], cfg['ff_mult'],
        embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w,
    ).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Student: {trainable:,} trainable params")
    return model


def get_real_batch(tokenizer, ds_iter, ds, batch_size: int, seq_len: int, device: str):
    """Get a batch of real text tokens."""
    tokens_list = []
    for _ in range(batch_size):
        while True:
            try:
                sample = next(ds_iter)
                text = sample.get('text', '')
                if len(text) < 200:
                    continue
                toks = tokenizer.encode(text, max_length=seq_len, truncation=True, return_tensors='pt')[0]
                if len(toks) >= seq_len:
                    tokens_list.append(toks[:seq_len])
                    break
            except StopIteration:
                ds_iter = iter(ds)
    return torch.stack(tokens_list).to(device), ds_iter


def main():
    parser = argparse.ArgumentParser(description="Diagnostic checkpoint evaluation")
    parser.add_argument('checkpoint', help="Path to FRR checkpoint .pt file")
    parser.add_argument('--teacher', default='1.7b', choices=['0.6b', '1.7b'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--samples', type=int, default=500, help="Number of eval samples")
    parser.add_argument('--seq-len', type=int, default=64, help="Sequence length for eval")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    print("=" * 70)
    print("DIAGNOSTIC EVALUATION")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Teacher: {args.teacher}, Device: {args.device}")
    print(f"Samples: {args.samples}, Seq len: {args.seq_len}")
    print("=" * 70)

    # Load models
    teacher, embed_w, norm_w, lm_head_w, cfg = load_teacher(args.teacher, args.device)
    model = load_student(args.checkpoint, cfg, embed_w, norm_w, lm_head_w, args.device)
    n_layers = cfg['n_layers']

    # Data
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    ds_iter = iter(ds)
    print("Data loaded!\n")

    results = {}
    t0 = time.time()

    # ══════════════════════════════════════════════════════════════
    # TEST 1: High-sample T1/T10 (stable reading)
    # ══════════════════════════════════════════════════════════════
    print(f"[1/5] T1/T10 agreement ({args.samples} samples, all positions)...")
    t1_correct = 0
    t10_total = 0.0
    t1_per_position = defaultdict(int)
    t10_per_position = defaultdict(float)
    position_counts = defaultdict(int)

    for i in range(args.samples):
        batch, ds_iter = get_real_batch(tokenizer, ds_iter, ds, 1, args.seq_len, args.device)
        with torch.no_grad():
            tl = teacher.forward(batch, max_layers=n_layers)
            sl = model(batch)

        # Evaluate ALL positions (not just last)
        for pos in range(tl.shape[1]):
            t_top = tl[0, pos].argmax().item()
            s_top = sl[0, pos].argmax().item()
            t_top10 = set(tl[0, pos].topk(10).indices.tolist())
            s_top10 = set(sl[0, pos].topk(10).indices.tolist())

            overlap = len(t_top10 & s_top10) / 10
            if t_top == s_top:
                t1_correct += 1
                t1_per_position[pos] += 1
            t10_total += overlap
            t10_per_position[pos] += overlap
            position_counts[pos] += 1

        if (i + 1) % 100 == 0:
            n_total = sum(position_counts.values())
            print(f"  {i+1}/{args.samples}: T1={t1_correct/n_total*100:.1f}% T10={t10_total/n_total*100:.1f}%")

    n_total_positions = sum(position_counts.values())
    overall_t1 = t1_correct / n_total_positions
    overall_t10 = t10_total / n_total_positions

    # Last-position only (for comparison with training eval)
    last_pos = args.seq_len - 1
    last_t1 = t1_per_position[last_pos] / position_counts[last_pos] if position_counts[last_pos] > 0 else 0
    last_t10 = t10_per_position[last_pos] / position_counts[last_pos] if position_counts[last_pos] > 0 else 0

    print(f"\n  ALL POSITIONS: T1={overall_t1*100:.1f}% T10={overall_t10*100:.1f}%")
    print(f"  LAST POS ONLY: T1={last_t1*100:.1f}% T10={last_t10*100:.1f}%")
    print(f"  ({n_total_positions} total predictions, {args.samples} sequences)\n")

    results['t1_all'] = overall_t1
    results['t10_all'] = overall_t10
    results['t1_last'] = last_t1
    results['t10_last'] = last_t10

    # ══════════════════════════════════════════════════════════════
    # TEST 2: Per-position agreement curve
    # ══════════════════════════════════════════════════════════════
    print("[2/5] Per-position T10 agreement curve...")
    pos_t10 = {}
    for pos in sorted(t10_per_position.keys()):
        if position_counts[pos] > 0:
            pos_t10[pos] = t10_per_position[pos] / position_counts[pos]
    results['per_position_t10'] = pos_t10

    early = sum(pos_t10[p] for p in range(min(10, args.seq_len))) / min(10, args.seq_len)
    late = sum(pos_t10[p] for p in range(max(0, args.seq_len - 10), args.seq_len)) / min(10, args.seq_len)
    print(f"  Positions 0-9: T10={early*100:.1f}%")
    print(f"  Positions {args.seq_len-10}-{args.seq_len-1}: T10={late*100:.1f}%")
    print(f"  Early/Late ratio: {early/late:.2f}x\n")
    results['early_t10'] = early
    results['late_t10'] = late

    # ══════════════════════════════════════════════════════════════
    # TEST 3: Top-K agreement for various K
    # ══════════════════════════════════════════════════════════════
    k_values = [1, 3, 5, 10, 20, 50]
    print(f"[3/5] Top-K agreement for K={k_values} ({min(args.samples, 200)} samples)...")
    topk_results = {k: 0.0 for k in k_values}
    topk_count = 0
    n_topk_samples = min(args.samples, 200)

    ds_iter2 = iter(ds)  # Fresh iterator
    for i in range(n_topk_samples):
        batch, ds_iter2 = get_real_batch(tokenizer, ds_iter2, ds, 1, args.seq_len, args.device)
        with torch.no_grad():
            tl = teacher.forward(batch, max_layers=n_layers)
            sl = model(batch)

        # Use last position
        for k in k_values:
            t_topk = set(tl[0, -1].topk(k).indices.tolist())
            s_topk = set(sl[0, -1].topk(k).indices.tolist())
            topk_results[k] += len(t_topk & s_topk) / k
        topk_count += 1

    print("  K  | Agreement")
    print("  ---|----------")
    for k in k_values:
        val = topk_results[k] / topk_count
        topk_results[k] = val
        print(f"  {k:>2} | {val*100:.1f}%")
    print()
    results['topk'] = {str(k): topk_results[k] for k in k_values}

    # ══════════════════════════════════════════════════════════════
    # TEST 4: KL divergence at different temperatures
    # ══════════════════════════════════════════════════════════════
    temperatures = [1.0, 2.0, 3.0, 5.0, 10.0]
    print(f"[4/5] KL divergence at temperatures {temperatures} ({min(args.samples, 100)} samples)...")
    n_kl_samples = min(args.samples, 100)
    kl_per_temp = {t: 0.0 for t in temperatures}

    ds_iter3 = iter(ds)
    for i in range(n_kl_samples):
        batch, ds_iter3 = get_real_batch(tokenizer, ds_iter3, ds, 1, args.seq_len, args.device)
        with torch.no_grad():
            tl = teacher.forward(batch, max_layers=n_layers)
            sl = model(batch)

        for temp in temperatures:
            kl = F.kl_div(
                F.log_softmax(sl / temp, dim=-1),
                F.softmax(tl / temp, dim=-1),
                reduction='batchmean',
            ).item()
            kl_per_temp[temp] += kl

    print("  T   | KL div")
    print("  ----|-------")
    for temp in temperatures:
        val = kl_per_temp[temp] / n_kl_samples
        kl_per_temp[temp] = val
        print(f"  {temp:>4.1f} | {val:.4f}")
    print()
    results['kl_by_temp'] = {str(t): kl_per_temp[t] for t in temperatures}

    # ══════════════════════════════════════════════════════════════
    # TEST 5: Entropy analysis — where does student struggle?
    # ══════════════════════════════════════════════════════════════
    print(f"[5/5] Entropy analysis ({min(args.samples, 100)} samples)...")
    teacher_entropies = []
    student_entropies = []
    agree_when_confident = 0
    agree_when_uncertain = 0
    n_confident = 0
    n_uncertain = 0

    ds_iter4 = iter(ds)
    for i in range(min(args.samples, 100)):
        batch, ds_iter4 = get_real_batch(tokenizer, ds_iter4, ds, 1, args.seq_len, args.device)
        with torch.no_grad():
            tl = teacher.forward(batch, max_layers=n_layers)
            sl = model(batch)

        for pos in range(tl.shape[1]):
            t_probs = F.softmax(tl[0, pos], dim=-1)
            s_probs = F.softmax(sl[0, pos], dim=-1)

            t_ent = -torch.sum(t_probs * torch.log(t_probs + 1e-10)).item()
            s_ent = -torch.sum(s_probs * torch.log(s_probs + 1e-10)).item()
            teacher_entropies.append(t_ent)
            student_entropies.append(s_ent)

            t_top = tl[0, pos].argmax().item()
            s_top = sl[0, pos].argmax().item()
            agree = (t_top == s_top)

            # Split by teacher confidence
            if t_ent < 2.0:  # Confident teacher
                n_confident += 1
                if agree:
                    agree_when_confident += 1
            else:  # Uncertain teacher
                n_uncertain += 1
                if agree:
                    agree_when_uncertain += 1

    t1_confident = agree_when_confident / n_confident if n_confident > 0 else 0
    t1_uncertain = agree_when_uncertain / n_uncertain if n_uncertain > 0 else 0
    t_ent_mean = sum(teacher_entropies) / len(teacher_entropies)
    s_ent_mean = sum(student_entropies) / len(student_entropies)

    print(f"  Teacher mean entropy: {t_ent_mean:.2f}")
    print(f"  Student mean entropy: {s_ent_mean:.2f}")
    print(f"  Student/Teacher entropy ratio: {s_ent_mean/t_ent_mean:.2f}x")
    print(f"  T1 when teacher confident (entropy<2): {t1_confident*100:.1f}% ({n_confident} tokens)")
    print(f"  T1 when teacher uncertain (entropy≥2): {t1_uncertain*100:.1f}% ({n_uncertain} tokens)")
    print()

    results['teacher_entropy'] = t_ent_mean
    results['student_entropy'] = s_ent_mean
    results['t1_confident'] = t1_confident
    results['t1_uncertain'] = t1_uncertain
    results['n_confident'] = n_confident
    results['n_uncertain'] = n_uncertain

    # ── Summary ───────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Teacher: {args.teacher}")
    print(f"  T1 (all pos, {args.samples} seqs): {overall_t1*100:.1f}%")
    print(f"  T10 (all pos, {args.samples} seqs): {overall_t10*100:.1f}%")
    print(f"  T1 (last pos only): {last_t1*100:.1f}%")
    print(f"  T10 (last pos only): {last_t10*100:.1f}%")
    print(f"  Best K=10 agreement: {topk_results[10]*100:.1f}%")
    print(f"  Entropy ratio (S/T): {s_ent_mean/t_ent_mean:.2f}x")
    print(f"  Student does {t1_confident/t1_uncertain:.1f}x better on confident tokens")
    print(f"  Elapsed: {elapsed:.0f}s")
    print("=" * 70)

    # Save results
    results['checkpoint'] = args.checkpoint
    results['teacher'] = args.teacher
    results['samples'] = args.samples
    results['elapsed_s'] = elapsed

    out_path = args.checkpoint.replace('.pt', '_diagnostic.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
