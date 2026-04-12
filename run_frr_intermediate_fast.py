"""
FRR WITH INTERMEDIATE HIDDEN STATE MATCHING
============================================
The #1 technique for distillation quality improvement.
Instead of only matching final logits, match hidden states at each
FRR recurrence step to the corresponding teacher layer.

Expected: 62% -> 72-78% top-10 from this alone.

Uses cosine similarity loss (not MSE) for intermediate matching,
since FRR's representations live at different scales per recurrence step.
"""
import lib.unbuffered  # Fix Windows buffering
import torch, sys, os, time, math
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultracompress.inference import ModelConfig, MiniTransformer
from ultracompress.moonshot import FractalModel

device = 'cuda'
STEPS = 10000

print("=" * 70)
print("FRR WITH INTERMEDIATE HIDDEN STATE MATCHING")
print("Expected: 62% -> 72-78% top-10")
print("=" * 70)

# Load teacher
print("Loading teacher...")
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
        if k in wd: gd[f'blk.{li}.{g}'] = wd[k].float()

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


def eval_model(model, n=100):
    t1, t10s = 0, []
    for trial in range(n):
        torch.manual_seed(trial * 13 + 9999)
        t = torch.randint(100, 50000, (1, 16), device=device)
        with torch.no_grad():
            tl = teacher.forward(t, max_layers=28)
            tp = tl[0, -1].argmax().item()
            tt10 = set(tl[0, -1].topk(10).indices.tolist())
            gl = model(t)
            gp = gl[0, -1].argmax().item()
            gt10 = set(gl[0, -1].topk(10).indices.tolist())
            if tp == gp: t1 += 1
            t10s.append(len(tt10 & gt10) / 10)
    return t1 / n, sum(t10s) / len(t10s)


def cosine_loss(a, b):
    """Cosine similarity loss between hidden states. Scale-invariant."""
    # a, b: (batch, seq, hidden)
    a_flat = a.reshape(-1, a.shape[-1])
    b_flat = b.reshape(-1, b.shape[-1])
    cos = F.cosine_similarity(a_flat, b_flat, dim=-1)
    return (1 - cos).mean()


# ════════════════════════════════════════════════════════════════════
# TEST CONFIGS
# ════════════════════════════════════════════════════════════════════
configs = [
    # FAST version: 3 most important configs for overnight run
    # (name, n_scales, iters, ff_mult, intermediate_weight, match_every_n)
    ("Baseline (logits only)", 4, 7, 1, 0.0, 1),
    ("Intermediate 1.0x", 4, 7, 1, 1.0, 1),
    ("Intermediate 1.0x + cosine anneal", 4, 7, 1, -1.0, 1),  # -1 = annealing
]

results = {}

for name, n_scales, iters, ff_mult, inter_weight, match_every in configs:
    print(f"\n{'='*60}")
    print(f"  CONFIG: {name}")
    print(f"{'='*60}")

    model = FractalModel(
        hidden_dim=1024, n_heads=16, n_scales=n_scales, iters_per_scale=iters,
        vocab_size=151936, ff_mult=ff_mult,
        embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    compression = teacher_layer_params / trainable
    print(f"  Trainable: {trainable:,} ({compression:.1f}x compression)")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=5e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)

    t0 = time.time()
    for step in range(STEPS):
        torch.manual_seed(step * 7)
        tokens = torch.randint(100, 50000, (4, 32), device=device)

        # Teacher forward with hidden states
        with torch.no_grad():
            teacher_logits, teacher_hidden = teacher.forward(
                tokens, max_layers=28, return_hidden=True)

        # Student forward with hidden states
        student_logits, student_hidden = model(tokens, return_hidden=True)

        # KL divergence on logits (with temperature annealing)
        T = max(2.0, 5.0 * (1 - step / STEPS))
        kl_loss = F.kl_div(
            F.log_softmax(student_logits / T, dim=-1),
            F.softmax(teacher_logits / T, dim=-1),
            reduction='batchmean'
        ) * (T * T)

        # Intermediate hidden state matching
        hidden_loss = torch.tensor(0.0, device=device)
        if inter_weight != 0:
            # Determine weight for this step
            if inter_weight == -1.0:  # annealing: start high, decay
                w = max(0.1, 2.0 * (1 - step / STEPS))
            else:
                w = inter_weight

            # Match student hidden[i] to teacher hidden[i]
            n_match = min(len(student_hidden), len(teacher_hidden))
            count = 0
            for i in range(n_match):
                if i % match_every != 0:
                    continue
                hidden_loss = hidden_loss + cosine_loss(
                    student_hidden[i], teacher_hidden[i].detach())
                count += 1
            if count > 0:
                hidden_loss = hidden_loss / count

            loss = kl_loss + w * hidden_loss
        else:
            loss = kl_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()

        if step % 2000 == 0 or step == STEPS - 1:
            t1, t10 = eval_model(model)
            elapsed = time.time() - t0
            print(f"    Step {step}: kl={kl_loss.item():.4f} hidden={hidden_loss.item():.4f} "
                  f"T1={t1*100:.0f}% T10={t10*100:.0f}% ({elapsed:.0f}s)")

    t1, t10 = eval_model(model, n=200)
    print(f"  FINAL (200 samples): T1={t1*100:.0f}% T10={t10*100:.0f}%")
    results[name] = {'top1': t1, 'top10': t10, 'compression': compression}

    del model
    torch.cuda.empty_cache()
    import gc; gc.collect()


# SUMMARY
print(f"\n{'='*70}")
print("SUMMARY: Intermediate Matching Experiment")
print("=" * 70)
for name, res in results.items():
    t10 = res['top10'] * 100
    comp = res['compression']
    print(f"  {name:45s}: T10={t10:.0f}% @ {comp:.0f}x")

import json
with open('intermediate_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print("\nSaved to intermediate_results.json")
