"""Run remaining experiments: MoE Genome + Quantized+Correction.
Hybrid and Neural DNA results already collected.
"""
import torch, sys, os, time, json
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultracompress.inference import ModelConfig, MiniTransformer, TransformerLayer
from ultracompress.genome_compressor import GenomeModel
from ultracompress.genome_v2 import LoRAGenomeLayer

device = 'cuda'
wd = torch.load('qwen3_0.6b_cache.pt', weights_only=True)

hf_to_gguf = {'self_attn.q_proj.weight': 'attn_q.weight', 'self_attn.k_proj.weight': 'attn_k.weight', 'self_attn.v_proj.weight': 'attn_v.weight', 'self_attn.o_proj.weight': 'attn_output.weight', 'self_attn.q_norm.weight': 'attn_q_norm.weight', 'self_attn.k_norm.weight': 'attn_k_norm.weight', 'input_layernorm.weight': 'attn_norm.weight', 'post_attention_layernorm.weight': 'ffn_norm.weight', 'mlp.gate_proj.weight': 'ffn_gate.weight', 'mlp.up_proj.weight': 'ffn_up.weight', 'mlp.down_proj.weight': 'ffn_down.weight'}
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

embed = gd['token_embd.weight'].to(device)
norm_w = gd['output_norm.weight'].to(device)
lm_head_w = gd['output.weight'].to(device)

all_results = {}


def quantize_weight(w, bits):
    if bits >= 16: return w
    n_levels = 2 ** bits
    wmin, wmax = w.min(), w.max()
    scale = (wmax - wmin) / (n_levels - 1)
    if scale == 0: return w
    return torch.round((w - wmin) / scale) * scale + wmin


def eval_forward(forward_fn, n=100):
    t1, t10s = 0, []
    for trial in range(n):
        torch.manual_seed(trial * 13 + 9999)
        t = torch.randint(100, 50000, (1, 16), device=device)
        with torch.no_grad():
            tl = teacher.forward(t, max_layers=28)
            tp = tl[0,-1].argmax().item()
            tt10 = set(tl[0,-1].topk(10).indices.tolist())
            gl = forward_fn(t)
            gp = gl[0,-1].argmax().item()
            gt10 = set(gl[0,-1].topk(10).indices.tolist())
            if tp == gp: t1 += 1
            t10s.append(len(tt10 & gt10) / 10)
    return t1/n, sum(t10s)/len(t10s)


# ============================================================
# EXPERIMENT A: MoE GENOME
# ============================================================
print("=" * 70)
print("EXPERIMENT A: MoE GENOME")
print("=" * 70)
sys.stdout.flush()

from ultracompress.genome_moe import MoEGenomeModel

for n_experts, expert_dim, name in [(8, 32, "MoE-8x32"), (8, 64, "MoE-8x64"), (16, 32, "MoE-16x32")]:
    print(f"\n--- {name} ---")
    model = MoEGenomeModel(
        vocab_size=151936, big_dim=1024,
        expert_dim=expert_dim, n_experts=n_experts, top_k=2,
        n_layers=28,
        embed_weight=embed, lm_head_weight=lm_head_w, norm_weight=norm_w,
    ).to(device)
    m_params = model.genome_param_count()
    print(f"  Params: {m_params:,} ({m_params*2/1e6:.1f} MB)")
    sys.stdout.flush()

    opt = torch.optim.AdamW(model.genome_layers.parameters(), lr=0.001, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 10000)
    t0 = time.time()
    for step in range(10000):
        tokens = torch.randint(100, 100000, (8, 32), device=device)
        with torch.no_grad():
            teacher_logits = teacher.forward(tokens, max_layers=28)[:, -1, :]
        student_logits = model(tokens, max_layers=28)[:, -1, :]
        loss = F.kl_div(F.log_softmax(student_logits/2, -1), F.softmax(teacher_logits/2, -1), reduction='batchmean') * 4
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.genome_layers.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 2000 == 0:
            print(f"    Step {step}: loss={loss.item():.4f} ({time.time()-t0:.0f}s)")
            sys.stdout.flush()

    def moe_fwd(tokens, _m=model): return _m(tokens, max_layers=28)
    t1, t10 = eval_forward(moe_fwd)
    print(f"  RESULT: {name} Top1={t1*100:.0f}% Top10={t10*100:.0f}% Size={m_params*2/1e6:.1f}MB Time={time.time()-t0:.0f}s")
    all_results[name] = {'top1': t1, 'top10': t10, 'size_mb': m_params*2/1e6}
    sys.stdout.flush()
    del model; torch.cuda.empty_cache()


# ============================================================
# EXPERIMENT B: GENOME V1 ONLINE (fresh data baseline)
# ============================================================
print(f"\n{'='*70}")
print("EXPERIMENT B: GENOME V1 ONLINE (no cache overfitting)")
print(f"{'='*70}")
sys.stdout.flush()

genome = GenomeModel(
    vocab_size=151936, big_dim=1024, small_dim=128, n_heads=4, n_layers=28,
    embed_weight=embed, lm_head_weight=lm_head_w, norm_weight=norm_w,
).to(device)
g_params = genome.genome_param_count()
print(f"Genome: {g_params:,} ({g_params*2/1e6:.1f} MB)")
sys.stdout.flush()

opt = torch.optim.AdamW(genome.genome_layers.parameters(), lr=0.001, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 10000)
t0 = time.time()
for step in range(10000):
    tokens = torch.randint(100, 100000, (8, 32), device=device)
    with torch.no_grad():
        teacher_logits = teacher.forward(tokens, max_layers=28)[:, -1, :]
    student_logits = genome(tokens, max_layers=28)[:, -1, :]
    loss = F.kl_div(F.log_softmax(student_logits/2, -1), F.softmax(teacher_logits/2, -1), reduction='batchmean') * 4
    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(genome.genome_layers.parameters(), 1.0)
    opt.step(); sched.step()
    if step % 2000 == 0:
        print(f"  Step {step}: loss={loss.item():.4f} ({time.time()-t0:.0f}s)")
        sys.stdout.flush()

def g_fwd(tokens): return genome(tokens, max_layers=28)
t1, t10 = eval_forward(g_fwd)
print(f"  RESULT: V1-online Top1={t1*100:.0f}% Top10={t10*100:.0f}% Size={g_params*2/1e6:.1f}MB Time={time.time()-t0:.0f}s")
all_results['V1-online'] = {'top1': t1, 'top10': t10, 'size_mb': g_params*2/1e6}
sys.stdout.flush()
del genome; torch.cuda.empty_cache()


# ============================================================
# EXPERIMENT C: QUANTIZED + GENOME CORRECTION
# ============================================================
print(f"\n{'='*70}")
print("EXPERIMENT C: QUANTIZED + GENOME CORRECTION")
print(f"{'='*70}")
sys.stdout.flush()

for quant_bits in [4, 2]:
    print(f"\n--- Q{quant_bits} + LoRA correction ---")
    comp = dict(wd)
    for key in list(comp.keys()):
        if any(p in key for p in ['q_proj.weight', 'k_proj.weight', 'v_proj.weight',
                                   'o_proj.weight', 'gate_proj.weight', 'up_proj.weight', 'down_proj.weight']):
            try:
                li = int(key.split('model.layers.')[1].split('.')[0])
                if li >= 28: continue
            except: continue
            comp[key] = quantize_weight(comp[key].float(), quant_bits)

    qgd = dict(gd)
    for li in range(28):
        for h, g in hf_to_gguf.items():
            k = f'model.layers.{li}.{h}'
            if k in comp: qgd[f'blk.{li}.{g}'] = comp[k].float()

    q_model = MiniTransformer(config, device)
    q_model.load_weights(qgd)
    q_model.embed_weight = q_model.embed_weight.to(device)
    if q_model.lm_head is not None:
        q_model.lm_head = q_model.lm_head.to(device)

    def q_fwd(tokens): return q_model.forward(tokens, max_layers=28)
    t1_q, t10_q = eval_forward(q_fwd, n=50)
    print(f"  Q{quant_bits} alone: Top1={t1_q*100:.0f}% Top10={t10_q*100:.0f}%")
    sys.stdout.flush()

    correction = nn.ModuleList([LoRAGenomeLayer(1024, rank=32) for _ in range(28)]).to(device)
    corr_params = sum(p.numel() for p in correction.parameters())

    opt = torch.optim.AdamW(correction.parameters(), lr=0.001, weight_decay=0.005)
    pos_train = torch.arange(32, device=device)
    t0 = time.time()
    for step in range(5000):
        torch.manual_seed(step)
        tokens = torch.randint(100, 100000, (4, 32), device=device)
        with torch.no_grad():
            teacher_logits = teacher.forward(tokens, max_layers=28)[:, -1, :]
        x = F.embedding(tokens, embed).float()
        for li in range(28):
            x_q = q_model.layers[li](x, pos_train)
            x = x_q + correction[li](x) * 0.1
        var = x.float().pow(2).mean(-1, keepdim=True)
        xn = x.float() * torch.rsqrt(var + 1e-6) * norm_w
        student_logits = F.linear(xn, lm_head_w)[:, -1, :]
        loss = F.kl_div(F.log_softmax(student_logits/2, -1), F.softmax(teacher_logits/2, -1), reduction='batchmean') * 4
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(correction.parameters(), 1.0)
        opt.step()
        if step % 1000 == 0:
            print(f"    Step {step}: loss={loss.item():.3f} ({time.time()-t0:.0f}s)")
            sys.stdout.flush()

    def qc_fwd(tokens, _qm=q_model, _corr=correction):
        pos = torch.arange(tokens.shape[1], device=tokens.device)
        x = F.embedding(tokens, embed).float()
        for li in range(28):
            x_q = _qm.layers[li](x, pos)
            x = x_q + _corr[li](x) * 0.1
        var = x.float().pow(2).mean(-1, keepdim=True)
        xn = x.float() * torch.rsqrt(var + 1e-6) * norm_w
        return F.linear(xn, lm_head_w)

    t1_qc, t10_qc = eval_forward(qc_fwd)
    orig_p = sum(wd[k].numel() for k in wd if any(f'layers.{i}' in k for i in range(28)) and 'weight' in k and wd[k].ndim >= 2)
    total_mb = orig_p * quant_bits / 8 / 1e6 + corr_params * 2 / 1e6

    print(f"  Q{quant_bits}+correction: Top1={t1_qc*100:.0f}% Top10={t10_qc*100:.0f}% Size={total_mb:.1f}MB")
    print(f"  Improvement: Top1 +{(t1_qc-t1_q)*100:.0f}%, Top10 +{(t10_qc-t10_q)*100:.0f}%")
    all_results[f'Q{quant_bits}+corr'] = {'top1': t1_qc, 'top10': t10_qc, 'size_mb': total_mb,
                                           'q_alone': {'top1': t1_q, 'top10': t10_q}}
    sys.stdout.flush()
    del q_model, correction, comp, qgd; torch.cuda.empty_cache()


# ============================================================
# COMBINED LEADERBOARD (with previous results)
# ============================================================
# Add results from previous runs
all_results['V1-progressive'] = {'top1': 0.20, 'top10': 0.53, 'size_mb': 23.9}
all_results['V1-hybrid-50K'] = {'top1': 0.27, 'top10': 0.46, 'size_mb': 23.9}
all_results['V2a-progressive'] = {'top1': 0.26, 'top10': 0.48, 'size_mb': 16.5}
all_results['Hybrid-INT4-r32'] = {'top1': 0.17, 'top10': 0.44, 'size_mb': 95.5}
all_results['Hybrid-INT4-r128'] = {'top1': 0.15, 'top10': 0.45, 'size_mb': 117.5}
all_results['NeuralDNA-512g'] = {'top1': 0.04, 'top10': 0.30, 'size_mb': 44.4}

print(f"\n{'='*70}")
print("FULL LEADERBOARD — ALL APPROACHES")
print(f"{'='*70}")

sorted_r = sorted(all_results.items(), key=lambda x: x[1].get('top10', 0), reverse=True)
for i, (name, r) in enumerate(sorted_r):
    t1 = r.get('top1', 0)
    t10 = r.get('top10', 0)
    sz = r.get('size_mb', 0)
    marker = " <<<" if i == 0 else ""
    print(f"  {i+1:>2}. {name:<25} Top1={t1*100:>4.0f}% Top10={t10*100:>4.0f}% Size={sz:>7.1f}MB{marker}")

with open('full_results.json', 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\nSaved to full_results.json")
print(f"{'='*70}")
