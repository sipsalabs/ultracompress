"""Multi-Block FRR: Quality breakthrough experiment.

HYPOTHESIS: Single-block FRR caps at ~62% top-10 because one shared block
cannot specialize for early(syntax), mid(semantics), and late(prediction).
Multi-block FRR uses 2-3 specialized shared blocks, each owning a range of
virtual layers. This should break through 62% -> 80%+ while staying at ~14x
compression (3 blocks) or ~20x (2 blocks).

Configs tested:
  - 3-block (early/mid/late): the full architecture, best quality expected
  - 2-block (early/late): faster, still better than single-block baseline
"""
import torch, sys, os, time, json, math, gc, traceback
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultracompress.inference import ModelConfig, MiniTransformer
from ultracompress.multi_block_frr import MultiBlockFRR

device = 'cuda'

# ── Load teacher (Qwen3-0.6B) ──────────────────────────────────────────────
print("Loading teacher (Qwen3-0.6B)...")
wd = torch.load('qwen3_0.6b_cache.pt', weights_only=True)

hf_to_gguf = {
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
gd = {}
gd['token_embd.weight'] = wd['model.embed_tokens.weight'].float()
gd['output_norm.weight'] = wd.get('model.norm.weight', torch.ones(1024)).float()
gd['output.weight'] = wd.get('lm_head.weight', gd['token_embd.weight']).float()
for li in range(28):
    for h, g in hf_to_gguf.items():
        k = f'model.layers.{li}.{h}'
        if k in wd:
            gd[f'blk.{li}.{g}'] = wd[k].float()

config = ModelConfig(n_layers=28, n_heads=16, n_kv_heads=8, hidden_size=1024,
                     intermediate_size=3072, vocab_size=151936, head_dim=128)
teacher = MiniTransformer(config, device)
teacher.load_weights(gd)
teacher.embed_weight = teacher.embed_weight.to(device)
if teacher.lm_head is not None:
    teacher.lm_head = teacher.lm_head.to(device)

embed_w = gd['token_embd.weight'].to(device)
norm_w = gd['output_norm.weight'].to(device)
lm_head_w = gd['output.weight'].to(device)

teacher_layer_params = sum(v.numel() for k, v in gd.items() if k.startswith('blk.'))

all_results = {}
pipeline_start = time.time()


# ── Eval helper ─────────────────────────────────────────────────────────────
def eval_model(forward_fn, n=100):
    t1, t10s = 0, []
    for trial in range(n):
        torch.manual_seed(trial * 13 + 9999)
        t = torch.randint(100, 50000, (1, 16), device=device)
        with torch.no_grad():
            tl = teacher.forward(t, max_layers=28)
            tp = tl[0, -1].argmax().item()
            tt10 = set(tl[0, -1].topk(10).indices.tolist())
            gl = forward_fn(t)
            gp = gl[0, -1].argmax().item()
            gt10 = set(gl[0, -1].topk(10).indices.tolist())
            if tp == gp:
                t1 += 1
            t10s.append(len(tt10 & gt10) / 10)
    return t1 / n, sum(t10s) / len(t10s)


# ── Configs ─────────────────────────────────────────────────────────────────
# (n_blocks, n_heads, ff_mult, lora_rank, n_steps, name)
configs = [
    (3, 16, 2, 32, 15000, "MB-FRR-3blk-r32"),   # Full 3-block: early/mid/late
    (2, 16, 2, 32, 15000, "MB-FRR-2blk-r32"),   # 2-block: early/late
]

print("=" * 70)
print("MULTI-BLOCK FRR: QUALITY BREAKTHROUGH EXPERIMENT")
print("Specialized shared blocks for depth ranges + per-layer LoRA adapters")
print(f"Baseline to beat: single-block FRR ~62% top-10")
print("=" * 70)
sys.stdout.flush()

for n_blocks, n_heads, ff_mult, lora_rank, n_steps, name in configs:
    print(f"\n{'─'*70}")
    print(f"  {name}: {n_blocks} blocks x 28 layers, rank={lora_rank}")
    print(f"{'─'*70}")
    sys.stdout.flush()
    t0 = time.time()

    try:
        model = MultiBlockFRR(
            hidden_dim=1024,
            n_heads=n_heads,
            total_layers=28,
            n_blocks=n_blocks,
            vocab_size=151936,
            ff_mult=ff_mult,
            lora_rank=lora_rank,
            embed_weight=embed_w,
            lm_head_weight=lm_head_w,
            norm_weight=norm_w,
        ).to(device)

        trainable_p = model.trainable_params()
        total_p = sum(p.numel() for p in model.parameters())
        compression = teacher_layer_params / trainable_p

        print(f"  Trainable params: {trainable_p:,} ({trainable_p * 2 / 1e6:.1f} MB)")
        print(f"  Total params:     {total_p:,} ({total_p * 2 / 1e6:.1f} MB)")
        print(f"  Teacher layers:   {teacher_layer_params:,} ({teacher_layer_params * 2 / 1e6:.1f} MB)")
        print(f"  Compression:      {compression:.1f}x")
        model.block_summary()
        sys.stdout.flush()

        # Collect trainable parameters (everything except frozen embed/head/norm)
        trainable = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=5e-4, weight_decay=0.01)
        warmup = 1500

        best_t10 = 0
        best_state = None

        for step in range(n_steps):
            # Cosine schedule with warmup
            if step < warmup:
                lr = 5e-4 * step / warmup
            else:
                lr = 5e-4 * 0.5 * (1 + math.cos((step - warmup) / (n_steps - warmup) * math.pi))
            for pg in opt.param_groups:
                pg['lr'] = lr

            tokens = torch.randint(100, 100000, (8, 32), device=device)
            with torch.no_grad():
                teacher_logits = teacher.forward(tokens, max_layers=28)

            student_logits = model(tokens)

            # All-position KL distillation (temperature=2)
            B, T, V = student_logits.shape
            loss = F.kl_div(
                F.log_softmax(student_logits.reshape(-1, V) / 2, -1),
                F.softmax(teacher_logits.reshape(-1, V) / 2, -1),
                reduction='batchmean') * 4

            if torch.isnan(loss):
                print(f"    NaN at step {step}, reducing LR and continuing...")
                for pg in opt.param_groups:
                    pg['lr'] *= 0.1
                continue

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 0.5)
            opt.step()

            # Periodic eval + checkpointing
            if step % 3000 == 0 or step == n_steps - 1:
                t1_e, t10_e = eval_model(lambda t, _m=model: _m(t))
                elapsed = time.time() - t0
                print(f"    Step {step:>5d}: loss={loss.item():.4f}  "
                      f"Top1={t1_e*100:.0f}%  Top10={t10_e*100:.0f}%  "
                      f"lr={lr:.6f}  ({elapsed:.0f}s)")
                if t10_e > best_t10:
                    best_t10 = t10_e
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()
                                  if v.requires_grad or 'blocks' in k or 'gamma' in k
                                  or 'beta' in k or 'gate' in k or 'adapter' in k}
                sys.stdout.flush()

        # Final eval with best checkpoint
        if best_state is not None:
            model.load_state_dict(best_state, strict=False)
            model.to(device)

        t1, t10 = eval_model(lambda t, _m=model: _m(t), n=200)
        elapsed = time.time() - t0
        size_mb = trainable_p * 2 / 1e6

        improvement = (t10 - 0.62) * 100  # vs single-block baseline
        print(f"\n  RESULT {name}:")
        print(f"    Top-1:       {t1*100:.1f}%")
        print(f"    Top-10:      {t10*100:.1f}%")
        print(f"    Params:      {trainable_p:,}")
        print(f"    Size:        {size_mb:.1f} MB")
        print(f"    Compression: {compression:.1f}x")
        print(f"    vs Baseline: {'+' if improvement >= 0 else ''}{improvement:.1f}% (baseline=62% top-10)")
        print(f"    Time:        {elapsed:.0f}s")

        all_results[name] = {
            'top1': t1, 'top10': t10, 'size_mb': size_mb,
            'compression': compression, 'params': trainable_p,
            'n_blocks': n_blocks, 'lora_rank': lora_rank,
            'time': elapsed, 'improvement_over_baseline': improvement,
        }

    except Exception as e:
        traceback.print_exc()
        print(f"  FAILED: {e}")
    finally:
        if 'model' in dir():
            del model
        if 'best_state' in dir():
            del best_state
        torch.cuda.empty_cache()
        gc.collect()
    sys.stdout.flush()


# ── Leaderboard ─────────────────────────────────────────────────────────────
total_time = time.time() - pipeline_start
print(f"\n{'='*70}")
print(f"MULTI-BLOCK FRR RESULTS (Total: {total_time/60:.0f} min)")
print(f"{'='*70}")
print(f"  Baseline: Single-block FRR = ~62% top-10 at ~42x compression")
print(f"  Target:   80%+ top-10 (quality breakthrough)")
print()

sorted_all = sorted(all_results.items(), key=lambda x: x[1]['top10'], reverse=True)
for i, (n, r) in enumerate(sorted_all):
    delta = r['improvement_over_baseline']
    marker = "***" if r['top10'] >= 0.80 else "**" if r['top10'] >= 0.70 else "*" if r['top10'] >= 0.62 else ""
    print(f"  {i+1}. {n:<25} Top1={r['top1']*100:>5.1f}%  Top10={r['top10']*100:>5.1f}%  "
          f"{r['size_mb']:>6.1f}MB  {r['compression']:>5.1f}x  "
          f"({'+' if delta >= 0 else ''}{delta:.1f}%) {marker}")

print()
print(f"  Legend: *** = 80%+ breakthrough  ** = 70%+ strong  * = beats baseline")

with open('multi_block_frr_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

if sorted_all:
    best_n, best = sorted_all[0]
    print(f"\n  Best: {best_n}")
    print(f"    Top-10: {best['top10']*100:.1f}%  ({best['n_blocks']} blocks, rank {best['lora_rank']})")
    print(f"    Size:   {best['size_mb']:.1f} MB at {best['compression']:.1f}x compression")

    if best['top10'] >= 0.80:
        print("\n  >>> BREAKTHROUGH! 80%+ top-10 achieved!")
        print("  >>> Multi-block specialization is the key insight.")
        print("  >>> Next: scale to larger models, optimize block boundaries.")
    elif best['top10'] >= 0.70:
        print("\n  >>> Strong improvement over single-block FRR.")
        print("  >>> Try: more steps, rank=64, 4 blocks, or progressive training.")
    elif best['top10'] >= 0.62:
        print("\n  >>> Beats single-block baseline. Multi-block helps but needs more.")
        print("  >>> Try: deeper training (25K steps), curriculum, or attention distillation.")
    else:
        print("\n  >>> Below baseline. Check: block ranges, gradient flow, modulation.")

print(f"{'='*70}")
