"""OVERNIGHT PIPELINE — Run all experiments sequentially, log everything.

Chain: Hybrid → Neural DNA → Arena → Analysis → Update STATUS
"""
import torch, sys, os, time, json
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultracompress.inference import ModelConfig, MiniTransformer, TransformerLayer
from ultracompress.genome_compressor import GenomeModel, MicroTransformerLayer
from ultracompress.genome_v2 import MultiViewGenomeLayer, LoRAGenomeLayer

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
# CRITICAL: pre-cache embed/lm_head on GPU (avoids 1.2GB CPU→GPU transfer per step!)
teacher.embed_weight = teacher.embed_weight.to(device)
if teacher.lm_head is not None:
    teacher.lm_head = teacher.lm_head.to(device)

embed = gd['token_embd.weight'].to(device)
norm_w = gd['output_norm.weight'].to(device)
lm_head = gd['output.weight'].to(device)

all_results = {}
pipeline_start = time.time()


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
# EXPERIMENT 1: HYBRID (INT4 attention + genome FFN)
# ============================================================
print("=" * 70)
print("EXPERIMENT 1: HYBRID — INT4 Attention + Genome FFN")
print("=" * 70)
sys.stdout.flush()

attn_weights = {}
for li in range(28):
    for wtype in ['attn_q', 'attn_k', 'attn_v', 'attn_output']:
        key = f'blk.{li}.{wtype}.weight'
        if key in gd:
            attn_weights[key] = quantize_weight(gd[key].float(), 4).to(device)
    for wtype in ['attn_norm', 'ffn_norm', 'attn_q_norm', 'attn_k_norm']:
        key = f'blk.{li}.{wtype}.weight'
        if key in gd:
            attn_weights[key] = gd[key].float().to(device)

attn_params = sum(v.numel() for v in attn_weights.values())
print(f"Attention at INT4: {attn_params:,} params ({attn_params*4/8/1e6:.1f} MB)")
sys.stdout.flush()

# Pre-build TransformerLayer objects ONCE (huge speedup vs recreating each step)
attn_layers = []
for li in range(28):
    tw = {}
    for wtype in ['attn_q', 'attn_k', 'attn_v', 'attn_output', 'attn_norm', 'ffn_norm', 'attn_q_norm', 'attn_k_norm']:
        key = f'blk.{li}.{wtype}.weight'
        if key in attn_weights:
            tw[wtype] = attn_weights[key]
    attn_layers.append(TransformerLayer(tw, config))

for rank in [32, 64, 128]:
    print(f"\n--- Hybrid rank={rank} ---")
    ffn_genome = nn.ModuleList([LoRAGenomeLayer(1024, rank=rank) for _ in range(28)]).to(device)
    ffn_params = sum(p.numel() for p in ffn_genome.parameters())
    total_mb = attn_params * 4 / 8 / 1e6 + ffn_params * 2 / 1e6
    print(f"  Genome FFN: {ffn_params:,} ({ffn_params*2/1e6:.1f} MB), Total: {total_mb:.1f} MB")
    sys.stdout.flush()

    opt = torch.optim.AdamW(ffn_genome.parameters(), lr=0.001, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 5000)

    t0 = time.time()
    for step in range(5000):
        torch.manual_seed(step + rank * 10000)
        tokens = torch.randint(100, 100000, (4, 32), device=device)
        with torch.no_grad():
            teacher_logits = teacher.forward(tokens, max_layers=28)[:, -1, :]

        pos = torch.arange(32, device=device)
        x = F.embedding(tokens, embed).float()
        for li in range(28):
            layer = attn_layers[li]
            attn_out = layer.attention(layer.attn_norm(x), pos)
            if attn_out.shape == x.shape:
                h = x + attn_out
            else:
                h = x.clone()
                min_d = min(attn_out.shape[-1], x.shape[-1])
                h[..., :min_d] = h[..., :min_d] + attn_out[..., :min_d]
            h_norm = layer.ffn_norm(h) if hasattr(layer, 'ffn_norm') else h
            x = h + ffn_genome[li](h_norm)

        var = x.float().pow(2).mean(-1, keepdim=True)
        xn = x.float() * torch.rsqrt(var + 1e-6) * norm_w
        student_logits = F.linear(xn, lm_head)[:, -1, :]
        loss = F.kl_div(F.log_softmax(student_logits/2, -1), F.softmax(teacher_logits/2, -1), reduction='batchmean') * 4
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(ffn_genome.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 1000 == 0:
            print(f"    Step {step}: loss={loss.item():.3f} ({time.time()-t0:.0f}s)")
            sys.stdout.flush()

    def hybrid_forward(tokens, _fg=ffn_genome, _al=attn_layers):
        pos = torch.arange(tokens.shape[1], device=tokens.device)
        x = F.embedding(tokens, embed).float()
        for li in range(28):
            layer = _al[li]
            attn_out = layer.attention(layer.attn_norm(x), pos)
            if attn_out.shape == x.shape:
                h = x + attn_out
            else:
                h = x.clone()
                min_d = min(attn_out.shape[-1], x.shape[-1])
                h[..., :min_d] = h[..., :min_d] + attn_out[..., :min_d]
            h_norm = layer.ffn_norm(h) if hasattr(layer, 'ffn_norm') else h
            x = h + _fg[li](h_norm)
        var = x.float().pow(2).mean(-1, keepdim=True)
        xn = x.float() * torch.rsqrt(var + 1e-6) * norm_w
        return F.linear(xn, lm_head)

    t1, t10 = eval_forward(hybrid_forward)
    elapsed = time.time() - t0
    print(f"  RESULT: rank={rank} Top1={t1*100:.0f}% Top10={t10*100:.0f}% Size={total_mb:.1f}MB Time={elapsed:.0f}s")
    all_results[f'hybrid_rank{rank}'] = {'top1': t1, 'top10': t10, 'size_mb': total_mb, 'params': ffn_params}
    sys.stdout.flush()

    del ffn_genome
    torch.cuda.empty_cache()

del attn_weights
torch.cuda.empty_cache()


# ============================================================
# EXPERIMENT 2: NEURAL DNA EXPRESSION
# ============================================================
print(f"\n{'='*70}")
print("EXPERIMENT 2: NEURAL DNA EXPRESSION")
print(f"{'='*70}")
sys.stdout.flush()

from sandbox2_neuraldna import NeuralDNAModel

for n_genes, gene_dim, n_active, name in [
    (256, 64, 32, "DNA-256g"),
    (512, 128, 64, "DNA-512g"),
    (1024, 256, 128, "DNA-1024g"),
]:
    print(f"\n--- {name} ---")
    model = NeuralDNAModel(
        vocab_size=151936, big_dim=1024,
        n_genes=n_genes, gene_dim=gene_dim, n_active=n_active,
        n_layers=28,
        embed_weight=embed, lm_head_weight=lm_head, norm_weight=norm_w,
    ).to(device)

    total_params = model.genome_param_count()
    dna_params = sum(p.numel() for p in model.dna_bank.parameters())
    print(f"  DNA: {dna_params:,} (shared) | Total: {total_params:,} ({total_params*2/1e6:.1f} MB)")
    sys.stdout.flush()

    # Deduplicate params
    seen = set()
    trainable = []
    for p in list(model.dna_bank.parameters()) + list(model.genome_layers.parameters()):
        if id(p) not in seen:
            seen.add(id(p))
            trainable.append(p)

    opt = torch.optim.AdamW(trainable, lr=0.001, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 10000)

    t0 = time.time()
    for step in range(10000):
        tokens = torch.randint(100, 100000, (8, 32), device=device)
        with torch.no_grad():
            teacher_logits = teacher.forward(tokens, max_layers=28)[:, -1, :]
        student_logits = model(tokens, max_layers=28)[:, -1, :]
        loss = F.kl_div(F.log_softmax(student_logits/2, -1), F.softmax(teacher_logits/2, -1), reduction='batchmean') * 4
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step(); sched.step()
        if step % 2000 == 0:
            print(f"    Step {step}: loss={loss.item():.4f} ({time.time()-t0:.0f}s)")
            sys.stdout.flush()

    def dna_forward(tokens, _m=model):
        return _m(tokens, max_layers=28)

    t1, t10 = eval_forward(dna_forward)
    elapsed = time.time() - t0
    print(f"  RESULT: {name} Top1={t1*100:.0f}% Top10={t10*100:.0f}% Size={total_params*2/1e6:.1f}MB Time={elapsed:.0f}s")
    all_results[name] = {'top1': t1, 'top10': t10, 'size_mb': total_params*2/1e6, 'params': total_params}
    sys.stdout.flush()

    del model
    torch.cuda.empty_cache()


# ============================================================
# EXPERIMENT 3: GENOME V1 ONLINE (fresh data, no overfitting)
# ============================================================
print(f"\n{'='*70}")
print("EXPERIMENT 3: GENOME V1 ONLINE DISTILLATION")
print(f"{'='*70}")
sys.stdout.flush()

genome = GenomeModel(
    vocab_size=151936, big_dim=1024, small_dim=128, n_heads=4,
    n_layers=28,
    embed_weight=embed, lm_head_weight=lm_head, norm_weight=norm_w,
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

def genome_forward(tokens):
    return genome(tokens, max_layers=28)

t1, t10 = eval_forward(genome_forward)
elapsed = time.time() - t0
print(f"  RESULT: V1-online Top1={t1*100:.0f}% Top10={t10*100:.0f}% Size={g_params*2/1e6:.1f}MB Time={elapsed:.0f}s")
all_results['genome_v1_online'] = {'top1': t1, 'top10': t10, 'size_mb': g_params*2/1e6, 'params': g_params}
sys.stdout.flush()

del genome
torch.cuda.empty_cache()


# ============================================================
# EXPERIMENT 4: MoE GENOME
# ============================================================
print(f"\n{'='*70}")
print("EXPERIMENT 4: MoE GENOME")
print(f"{'='*70}")
sys.stdout.flush()

from ultracompress.genome_moe import MoEGenomeModel

for n_experts, expert_dim in [(8, 32), (8, 64), (16, 32)]:
    name = f"MoE-{n_experts}x{expert_dim}"
    print(f"\n--- {name} ---")
    model = MoEGenomeModel(
        vocab_size=151936, big_dim=1024,
        expert_dim=expert_dim, n_experts=n_experts, top_k=2,
        n_layers=28,
        embed_weight=embed, lm_head_weight=lm_head, norm_weight=norm_w,
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

    def moe_forward(tokens, _m=model):
        return _m(tokens, max_layers=28)

    t1, t10 = eval_forward(moe_forward)
    elapsed = time.time() - t0
    print(f"  RESULT: {name} Top1={t1*100:.0f}% Top10={t10*100:.0f}% Size={m_params*2/1e6:.1f}MB Time={elapsed:.0f}s")
    all_results[name] = {'top1': t1, 'top10': t10, 'size_mb': m_params*2/1e6, 'params': m_params}
    sys.stdout.flush()

    del model
    torch.cuda.empty_cache()


# ============================================================
# EXPERIMENT 5: QUANTIZED + GENOME CORRECTION (the sweep)
# ============================================================
print(f"\n{'='*70}")
print("EXPERIMENT 5: QUANTIZED + GENOME CORRECTION")
print(f"{'='*70}")
sys.stdout.flush()

for quant_bits in [2, 4]:
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

    # Eval quantized alone
    def q_forward(tokens): return q_model.forward(tokens, max_layers=28)
    t1_q, t10_q = eval_forward(q_forward)
    print(f"  Q{quant_bits} alone: Top1={t1_q*100:.0f}% Top10={t10_q*100:.0f}%")
    sys.stdout.flush()

    # Train correction
    correction = nn.ModuleList([LoRAGenomeLayer(1024, rank=32) for _ in range(28)]).to(device)
    corr_params = sum(p.numel() for p in correction.parameters())

    opt = torch.optim.AdamW(correction.parameters(), lr=0.001, weight_decay=0.005)
    positions_train = torch.arange(32, device=device)
    for step in range(3000):
        torch.manual_seed(step)
        tokens = torch.randint(100, 100000, (4, 32), device=device)
        with torch.no_grad():
            teacher_logits = teacher.forward(tokens, max_layers=28)[:, -1, :]
        x = F.embedding(tokens, embed).float()
        for li in range(28):
            x_q = q_model.layers[li](x, positions_train)
            x = x_q + correction[li](x) * 0.1
        var = x.float().pow(2).mean(-1, keepdim=True)
        xn = x.float() * torch.rsqrt(var + 1e-6) * norm_w
        student_logits = F.linear(xn, lm_head)[:, -1, :]
        loss = F.kl_div(F.log_softmax(student_logits/2, -1), F.softmax(teacher_logits/2, -1), reduction='batchmean') * 4
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(correction.parameters(), 1.0)
        opt.step()
        if step % 1000 == 0:
            print(f"    Step {step}: loss={loss.item():.3f}")
            sys.stdout.flush()

    def qc_forward(tokens, _qm=q_model, _corr=correction):
        pos = torch.arange(tokens.shape[1], device=tokens.device)
        x = F.embedding(tokens, embed).float()
        for li in range(28):
            x_q = _qm.layers[li](x, pos)
            x = x_q + _corr[li](x) * 0.1
        var = x.float().pow(2).mean(-1, keepdim=True)
        xn = x.float() * torch.rsqrt(var + 1e-6) * norm_w
        return F.linear(xn, lm_head)

    t1_qc, t10_qc = eval_forward(qc_forward)

    orig_params = sum(wd[k].numel() for k in wd if any(f'layers.{i}' in k for i in range(28)) and 'weight' in k and wd[k].ndim >= 2)
    q_size_mb = orig_params * quant_bits / 8 / 1e6
    c_size_mb = corr_params * 2 / 1e6
    total_mb = q_size_mb + c_size_mb

    print(f"  Q{quant_bits}+correction: Top1={t1_qc*100:.0f}% Top10={t10_qc*100:.0f}% Size={total_mb:.1f}MB")
    print(f"  Improvement over Q{quant_bits}: Top1 +{(t1_qc-t1_q)*100:.0f}%, Top10 +{(t10_qc-t10_q)*100:.0f}%")
    all_results[f'Q{quant_bits}_correction'] = {
        'top1': t1_qc, 'top10': t10_qc, 'size_mb': total_mb,
        'q_alone_top1': t1_q, 'q_alone_top10': t10_q,
    }
    sys.stdout.flush()

    del q_model, correction, comp, qgd
    torch.cuda.empty_cache()


# ============================================================
# FINAL LEADERBOARD
# ============================================================
total_time = time.time() - pipeline_start
print(f"\n{'='*70}")
print(f"OVERNIGHT LEADERBOARD (Total time: {total_time/60:.0f} min)")
print(f"{'='*70}")

# Sort by top10
sorted_results = sorted(all_results.items(), key=lambda x: x[1].get('top10', 0), reverse=True)
for i, (name, r) in enumerate(sorted_results):
    medal = ["1st", "2nd", "3rd"][i] if i < 3 else f"{i+1}th"
    t1 = r.get('top1', 0)
    t10 = r.get('top10', 0)
    size = r.get('size_mb', 0)
    print(f"  {medal:>3}: {name:<25} Top1={t1*100:>4.0f}% Top10={t10*100:>4.0f}% Size={size:>7.1f}MB")

# Save
with open('overnight_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\nResults saved to overnight_results.json")
print(f"Total pipeline time: {total_time/3600:.1f} hours")

# Scaling projections for top approach
best_name, best = sorted_results[0]
print(f"\n{'='*70}")
print(f"WINNER: {best_name}")
print(f"  Top1={best['top1']*100:.0f}% Top10={best['top10']*100:.0f}% at {best.get('size_mb', 0):.1f} MB")
if 'hybrid' in best_name:
    print(f"  8B projection: ~{best.get('size_mb', 0) * 8 / 0.6:.0f} MB")
    print(f"  70B projection: ~{best.get('size_mb', 0) * 70 / 0.6:.0f} MB")
    print(f"  1000T projection: NOT FEASIBLE (attention too large)")
    print(f"  But: attention sharing + INT2 could help")
elif 'DNA' in best_name or 'genome' in best_name.lower() or 'MoE' in best_name:
    print(f"  8B projection: ~{best.get('size_mb', 0) * (36/28) * (4096/1024)**0.5:.0f} MB")
    print(f"  1000T projection: ~{best.get('size_mb', 0) * (200/28) * (30000/1024)**0.5:.0f} MB")
print(f"{'='*70}")
