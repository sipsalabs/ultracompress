"""SANDBOX 1: Hybrid Attention+Genome — Real INT4 attention + genome FFN.

The theory: attention is what makes LLMs work (token routing).
FFN is 2/3 of parameters but more compressible.
Keep real attention at INT4, replace FFN with tiny genome.

Runs on GPU 1 (cuda:1).
"""
import torch, sys, os, time
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultracompress.inference import ModelConfig, MiniTransformer, TransformerLayer
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

embed = gd['token_embd.weight'].to(device)
norm_w = gd['output_norm.weight'].to(device)
lm_head = gd['output.weight'].to(device)


def quantize_weight(w, bits):
    if bits >= 16: return w
    n_levels = 2 ** bits
    wmin, wmax = w.min(), w.max()
    scale = (wmax - wmin) / (n_levels - 1)
    if scale == 0: return w
    return torch.round((w - wmin) / scale) * scale + wmin


print("=" * 60)
print("SANDBOX 1: HYBRID ATTENTION + GENOME FFN")
print("=" * 60)

# Prepare attention weights at INT4
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

# Test different genome FFN sizes
for rank in [32, 64, 128]:
    print(f"\n--- LoRA rank={rank} genome FFN ---")
    ffn_genome = nn.ModuleList([LoRAGenomeLayer(1024, rank=rank) for _ in range(28)]).to(device)
    ffn_params = sum(p.numel() for p in ffn_genome.parameters())
    print(f"Genome FFN: {ffn_params:,} params ({ffn_params*2/1e6:.1f} MB)")
    total_mb = attn_params * 4 / 8 / 1e6 + ffn_params * 2 / 1e6
    print(f"Total: {total_mb:.1f} MB")

    # Online training — fresh data every batch
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
            tw = {}
            for wtype in ['attn_q', 'attn_k', 'attn_v', 'attn_output', 'attn_norm', 'ffn_norm', 'attn_q_norm', 'attn_k_norm']:
                key = f'blk.{li}.{wtype}.weight'
                if key in attn_weights:
                    tw[wtype] = attn_weights[key]
            layer = TransformerLayer(tw, config)
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
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(ffn_genome.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 1000 == 0:
            print(f"    Step {step}: loss={loss.item():.3f} ({time.time()-t0:.0f}s)")
            sys.stdout.flush()

    # Eval
    def hybrid_forward(tokens):
        pos = torch.arange(tokens.shape[1], device=tokens.device)
        x = F.embedding(tokens, embed).float()
        for li in range(28):
            tw = {}
            for wtype in ['attn_q', 'attn_k', 'attn_v', 'attn_output', 'attn_norm', 'ffn_norm', 'attn_q_norm', 'attn_k_norm']:
                key = f'blk.{li}.{wtype}.weight'
                if key in attn_weights:
                    tw[wtype] = attn_weights[key]
            layer = TransformerLayer(tw, config)
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
        return F.linear(xn, lm_head)

    t1, t10s = 0, []
    for trial in range(100):
        torch.manual_seed(trial * 13 + 9999)
        t = torch.randint(100, 50000, (1, 16), device=device)
        with torch.no_grad():
            tl = teacher.forward(t, max_layers=28)
            tp = tl[0,-1].argmax().item()
            tt10 = set(tl[0,-1].topk(10).indices.tolist())
            gl = hybrid_forward(t)
            gp = gl[0,-1].argmax().item()
            gt10 = set(gl[0,-1].topk(10).indices.tolist())
            if tp == gp: t1 += 1
            t10s.append(len(tt10 & gt10) / 10)

    t10_avg = sum(t10s)/len(t10s)
    print(f"  RESULT rank={rank}: Top1={t1}% Top10={t10_avg*100:.0f}% Size={total_mb:.1f}MB")
    print(f"  Time: {time.time()-t0:.0f}s")
    sys.stdout.flush()

    del ffn_genome
    torch.cuda.empty_cache()

print(f"\n{'='*60}")
print("SANDBOX 1 COMPLETE")
print(f"{'='*60}")
