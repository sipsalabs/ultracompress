"""
TEST: Sip's 4D Dimensional Block — cross-depth attention.
Compare: standard FRR vs 4D FRR (with cross-depth connections).

The hypothesis: letting later passes attend to earlier passes'
outputs creates a richer computation space that improves quality.
"""
import lib.unbuffered
import torch, sys, os, time, gc
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultracompress.inference import ModelConfig, MiniTransformer
from ultracompress.moonshot import FractalModel
from ultracompress.dimensional_block import DimensionalFRR

device = 'cuda'
STEPS = 10000

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


def train_and_eval(model, name, steps=STEPS):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    compression = teacher_layers / trainable
    print(f"  Trainable: {trainable:,} ({compression:.1f}x)")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=5e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    t0 = time.time()
    for step in range(steps):
        torch.manual_seed(step * 7)
        tokens = torch.randint(100, 50000, (4, 32), device=device)
        with torch.no_grad():
            teacher_logits = teacher.forward(tokens, max_layers=28)
        student_logits = model(tokens)
        T = max(2.0, 5.0 * (1 - step / steps))
        loss = F.kl_div(
            F.log_softmax(student_logits / T, dim=-1),
            F.softmax(teacher_logits / T, dim=-1),
            reduction='batchmean') * (T * T)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()
        if step % 2000 == 0 or step == steps - 1:
            t1, t10 = eval_model(model, n=100)
            print(f"    Step {step}: loss={loss.item():.4f} T1={t1*100:.0f}% T10={t10*100:.0f}% ({time.time()-t0:.0f}s)")

    t1, t10 = eval_model(model, n=200)
    print(f"  FINAL: T1={t1*100:.0f}% T10={t10*100:.0f}% @ {compression:.0f}x")
    return {'top1': t1, 'top10': t10, 'compression': compression, 'params': trainable}


results = {}

# 1. Standard FRR baseline
model = FractalModel(1024, 16, 4, 7, 151936, 1,
                     embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w).to(device)
results['Standard'] = train_and_eval(model, "Standard FRR (baseline)")
del model; gc.collect(); torch.cuda.empty_cache()

# 2. 4D Dimensional FRR (Sip's idea)
model = DimensionalFRR(1024, 16, 4, 7, 151936, 1, depth_heads=4,
                       embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w).to(device)
results['4D-Block'] = train_and_eval(model, "4D Dimensional FRR (cross-depth attention)")
del model; gc.collect(); torch.cuda.empty_cache()

print(f"\n{'='*60}")
print("4D BLOCK RESULTS")
print("=" * 60)
for name, r in results.items():
    print(f"  {name:<30}: T10={r['top10']*100:.0f}% @ {r['compression']:.0f}x ({r['params']:,} params)")
print("Done!")
