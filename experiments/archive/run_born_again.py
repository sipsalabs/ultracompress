"""
BORN-AGAIN DISTILLATION: Use trained FRR as teacher for a NEW FRR.
Expected: +2-4% T10 per generation. 2 generations + BANE ensemble.

Requires: frr_100k_best.pt (from run_100k_train.py)
If not available, trains gen0 first.
"""
import lib.unbuffered
import torch, sys, os, time, gc
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultracompress.inference import ModelConfig, MiniTransformer
from ultracompress.moonshot import FractalModel

device = 'cuda'
STEPS_PER_GEN = 50000
N_GENERATIONS = 3  # gen0 (from teacher) + gen1 + gen2

print("Loading original teacher (Qwen3-0.6B)...")
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
orig_teacher = MiniTransformer(config, device)
orig_teacher.load_weights(gd)
orig_teacher.embed_weight = orig_teacher.embed_weight.to(device)
if orig_teacher.lm_head is not None: orig_teacher.lm_head = orig_teacher.lm_head.to(device)

embed_w = gd['token_embd.weight'].to(device)
norm_w = gd['output_norm.weight'].to(device)
lm_head_w = gd['output.weight'].to(device)


def eval_vs_teacher(model, n=200):
    """Eval against ORIGINAL teacher (not born-again teacher)."""
    t1, t10s = 0, []
    for trial in range(n):
        torch.manual_seed(trial * 13 + 9999)
        t = torch.randint(100, 50000, (1, 16), device=device)
        with torch.no_grad():
            tl = orig_teacher.forward(t, max_layers=28)
            tp = tl[0, -1].argmax().item()
            tt10 = set(tl[0, -1].topk(10).indices.tolist())
            gl = model(t)
            gp = gl[0, -1].argmax().item()
            gt10 = set(gl[0, -1].topk(10).indices.tolist())
            if tp == gp: t1 += 1
            t10s.append(len(tt10 & gt10) / 10)
    return t1 / n, sum(t10s) / len(t10s)


def make_frr():
    return FractalModel(
        hidden_dim=1024, n_heads=16, n_scales=4, iters_per_scale=7,
        vocab_size=151936, ff_mult=1,
        embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w
    ).to(device)


def train_generation(student, teacher_fn, gen_num, steps):
    """Train a student FRR from a teacher (original or previous gen FRR)."""
    print(f"\n{'='*60}")
    print(f"  GENERATION {gen_num}: {steps} steps")
    print(f"{'='*60}")

    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],
                            lr=5e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    t0 = time.time()
    for step in range(steps):
        torch.manual_seed(step * 7 + gen_num * 100000)
        tokens = torch.randint(100, 50000, (4, 32), device=device)
        with torch.no_grad():
            teacher_logits = teacher_fn(tokens)
        student_logits = student(tokens)
        T = max(2.0, 5.0 * (1 - step / steps))
        loss = F.kl_div(
            F.log_softmax(student_logits / T, dim=-1),
            F.softmax(teacher_logits / T, dim=-1),
            reduction='batchmean'
        ) * (T * T)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        scheduler.step()

        if step % 10000 == 0 or step == steps - 1:
            t1, t10 = eval_vs_teacher(student, n=100)
            print(f"    Step {step}: loss={loss.item():.4f} T1={t1*100:.0f}% T10={t10*100:.0f}% ({time.time()-t0:.0f}s)")

    t1, t10 = eval_vs_teacher(student, n=200)
    print(f"  GEN {gen_num} FINAL: T1={t1*100:.0f}% T10={t10*100:.0f}%")
    return t1, t10


# ════════════════════════════════════════════════════════════════════
# BORN-AGAIN LOOP
# ════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"BORN-AGAIN DISTILLATION: {N_GENERATIONS} generations, {STEPS_PER_GEN} steps each")
print(f"{'='*60}")

gen_results = []

# Gen 0: distill from original teacher
if os.path.exists('frr_100k_best.pt'):
    print("Loading pre-trained gen0 from frr_100k_best.pt...")
    gen0 = make_frr()
    gen0.load_state_dict(torch.load('frr_100k_best.pt', map_location=device))
    t1, t10 = eval_vs_teacher(gen0, n=200)
    print(f"  Gen 0 (pre-trained): T1={t1*100:.0f}% T10={t10*100:.0f}%")
    gen_results.append(('Gen 0 (pre-trained)', t1, t10))
else:
    print("No pre-trained model. Training gen0 from original teacher...")
    gen0 = make_frr()
    t1, t10 = train_generation(gen0, lambda t: orig_teacher.forward(t, max_layers=28), 0, STEPS_PER_GEN)
    gen_results.append(('Gen 0', t1, t10))
    torch.save(gen0.state_dict(), 'frr_born_again_gen0.pt')

# Gen 1: distill from gen0 FRR
gen1 = make_frr()
prev_model = gen0
t1, t10 = train_generation(gen1, lambda t: prev_model(t), 1, STEPS_PER_GEN)
gen_results.append(('Gen 1', t1, t10))
torch.save(gen1.state_dict(), 'frr_born_again_gen1.pt')
del gen0; gc.collect(); torch.cuda.empty_cache()

# Gen 2: distill from gen1 FRR
gen2 = make_frr()
prev_model = gen1
t1, t10 = train_generation(gen2, lambda t: prev_model(t), 2, STEPS_PER_GEN)
gen_results.append(('Gen 2', t1, t10))
torch.save(gen2.state_dict(), 'frr_born_again_gen2.pt')

# ════════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("BORN-AGAIN RESULTS (all eval'd vs original Qwen3-0.6B teacher)")
print("=" * 60)
for name, t1, t10 in gen_results:
    print(f"  {name:25s}: T1={t1*100:.0f}% T10={t10*100:.0f}%")

if len(gen_results) >= 2:
    gain = (gen_results[-1][2] - gen_results[0][2]) * 100
    print(f"\n  Total gain from born-again: {gain:+.1f}% T10")

print("\nDone!")
