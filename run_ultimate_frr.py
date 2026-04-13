"""
ULTIMATE FRR — Stack EVERY proven technique into one run.

PHM block (4x fewer params) + mode routing (72B configs) +
dual-objective loss (T10 + T1) + long training (50K steps).

If this doesn't beat 67%, nothing in the current paradigm will.
That would confirm: we need a fundamentally new invention.
"""
import lib.unbuffered
import torch, sys, os, time, gc
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultracompress.inference import ModelConfig, MiniTransformer
from ultracompress.moonshot import FractalBlock
from ultracompress.hypercomplex import PHMFractalBlock
from ultracompress.mode_routing import ModeRoutedFRR
from ultracompress.dual_objective import dual_phase_loss

device = 'cuda'
STEPS = 50000

print("=" * 60)
print("ULTIMATE FRR: Every proven technique stacked")
print("PHM + Mode Routing + Dual Loss + 50K steps")
print("=" * 60)

print("Loading teacher...")
wd = torch.load('qwen3_0.6b_cache.pt', weights_only=True)
hf_to_gguf = {
    'self_attn.q_proj.weight': 'attn_q.weight', 'self_attn.k_proj.weight': 'attn_k.weight',
    'self_attn.v_proj.weight': 'attn_v.weight', 'self_attn.o_proj.weight': 'attn_output.weight',
    'self_attn.q_norm.weight': 'attn_q_norm.weight', 'self_attn.k_norm.weight': 'attn_k_norm.weight',
    'input_layernorm.weight': 'attn_norm.weight', 'post_attention_layernorm.weight': 'ffn_norm.weight',
    'mlp.gate_proj.weight': 'ffn_gate.weight', 'mlp.up_proj.weight': 'ffn_up.weight',
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
if teacher.lm_head is not None: teacher.lm_head = teacher.lm_head.to(device)
embed_w = gd['token_embd.weight'].to(device)
norm_w = gd['output_norm.weight'].to(device)
lm_head_w = gd['output.weight'].to(device)
teacher_layers = sum(v.numel() for k, v in gd.items() if k.startswith('blk.'))


def eval_model(model, n=200):
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


# Build Mode-Routed FRR with PHM block
print("Building Ultimate FRR (Mode Routing + PHM core)...")
model = ModeRoutedFRR(
    hidden_dim=1024, n_heads=16, n_scales=4, iters_per_scale=7,
    vocab_size=151936, ff_mult=1, n_modes=4,
    embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w
).to(device)

# Replace the shared block with PHM variant
phm_block = PHMFractalBlock(1024, 16, ff_mult=1, n=4).to(device)
model.block = phm_block

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
compression = teacher_layers / trainable
print(f"  Trainable: {trainable:,} ({compression:.1f}x)")
print(f"  PHM block: {sum(p.numel() for p in phm_block.parameters()):,}")
print(f"  Mode routing: 4 modes * 28 layers = 4^28 configs")

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                        lr=5e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)

print(f"Training {STEPS} steps with dual-objective loss...")
t0 = time.time()
best_t10 = 0
for step in range(STEPS):
    torch.manual_seed(step * 7)
    tokens = torch.randint(100, 50000, (4, 32), device=device)
    with torch.no_grad():
        teacher_logits = teacher.forward(tokens, max_layers=28)
    student_logits = model(tokens)

    # Dual-objective loss: KL + ramping CE for T1
    loss = dual_phase_loss(student_logits, teacher_logits, step, STEPS)

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    scheduler.step()

    if step % 5000 == 0 or step == STEPS - 1:
        t1, t10 = eval_model(model, n=100)
        marker = " <-- NEW BEST" if t10 > best_t10 else ""
        if t10 > best_t10: best_t10 = t10
        print(f"  Step {step}: loss={loss.item():.4f} T1={t1*100:.0f}% T10={t10*100:.0f}% ({time.time()-t0:.0f}s){marker}")

t1, t10 = eval_model(model, n=200)
print(f"\nFINAL: T1={t1*100:.0f}% T10={t10*100:.0f}%")
print(f"Compression: {compression:.0f}x")
print(f"Previous bests: FRR 67% at 48x, PHM 53% at 239x")
print(f"Ultimate FRR: {t10*100:.0f}% at {compression:.0f}x")
if t10 > 0.67:
    print("NEW RECORD! Stacking techniques works!")
elif t10 > 0.60:
    print("Good but not breakthrough. Current paradigm has limits.")
else:
    print("Stacking didn't help. Need fundamentally new invention.")
print("Done!")
