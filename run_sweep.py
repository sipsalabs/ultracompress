"""Monte Carlo Architecture Sweep — Test all approaches systematically.

Runs multiple compression strategies in rapid succession,
evaluates each, finds the optimal combination.

Approaches:
  A: Quantized weights + genome correction
  B: Keep attention, genome FFN only
  C: Full genome at different scales
  + Combinations of the above
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

# Build teacher
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
positions = torch.arange(32, device=device)


def quick_eval(forward_fn, n=50):
    """Quick eval — returns top1, top10."""
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


def quantize_weight(w, bits):
    """Quick uniform quantization."""
    if bits >= 16: return w
    n_levels = 2 ** bits
    wmin, wmax = w.min(), w.max()
    scale = (wmax - wmin) / (n_levels - 1)
    if scale == 0: return w
    return torch.round((w - wmin) / scale) * scale + wmin


# ============================================================
# APPROACH A: Quantized weights + genome CORRECTION
# ============================================================
print("=" * 60)
print("APPROACH A: Quantized weights + genome correction")
print("Keep original weights at low precision, genome fixes errors")
print("=" * 60)

for quant_bits in [2, 4]:
    # Build quantized model
    comp = dict(wd)
    for key in list(comp.keys()):
        if any(p in key for p in ['q_proj.weight', 'k_proj.weight', 'v_proj.weight',
                                   'o_proj.weight', 'gate_proj.weight', 'up_proj.weight', 'down_proj.weight']):
            try:
                li = int(key.split('model.layers.')[1].split('.')[0])
                if li >= 28: continue
            except: continue
            comp[key] = quantize_weight(comp[key].float(), quant_bits)

    # Build quantized teacher for this config
    qgd = dict(gd)
    for li in range(28):
        for h, g in hf_to_gguf.items():
            k = f'model.layers.{li}.{h}'
            if k in comp: qgd[f'blk.{li}.{g}'] = comp[k].float()

    q_model = MiniTransformer(config, device)
    q_model.load_weights(qgd)

    # Eval quantized model alone
    def q_forward(tokens):
        return q_model.forward(tokens, max_layers=28)
    t1_q, t10_q = quick_eval(q_forward)
    print(f"\n  Q{quant_bits} alone: Top1={t1_q*100:.0f}% Top10={t10_q*100:.0f}%")

    # Now train a genome CORRECTION on top of quantized model
    # correction(x) = genome(x) trained to predict (teacher_output - quantized_output)
    correction = nn.ModuleList([LoRAGenomeLayer(1024, rank=32) for _ in range(28)]).to(device)
    corr_params = sum(p.numel() for p in correction.parameters())

    opt = torch.optim.AdamW(correction.parameters(), lr=0.001, weight_decay=0.005)
    for step in range(3000):
        torch.manual_seed(step)
        tokens = torch.randint(100, 100000, (4, 32), device=device)
        with torch.no_grad():
            teacher_logits = teacher.forward(tokens, max_layers=28)[:, -1, :]
        # Forward through quantized + correction
        x = F.embedding(tokens, embed).float()
        for li in range(28):
            # Quantized layer forward
            x_q = q_model.layers[li](x, positions)
            # Genome correction
            x = x_q + correction[li](x) * 0.1  # scaled correction

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

    # Eval quantized + correction
    def qc_forward(tokens):
        pos = torch.arange(tokens.shape[1], device=tokens.device)
        x = F.embedding(tokens, embed).float()
        for li in range(28):
            x_q = q_model.layers[li](x, pos)
            x = x_q + correction[li](x) * 0.1
        var = x.float().pow(2).mean(-1, keepdim=True)
        xn = x.float() * torch.rsqrt(var + 1e-6) * norm_w
        return F.linear(xn, lm_head)

    t1_qc, t10_qc = quick_eval(qc_forward)

    # Size: quantized weights + correction params
    orig_params = sum(wd[k].numel() for k in wd if any(f'layers.{i}' in k for i in range(28)) and 'weight' in k and wd[k].ndim >= 2)
    q_size = orig_params * quant_bits / 8  # quantized weight size in bytes
    c_size = corr_params * 2  # correction size in bytes (FP16)
    total_bpw = (q_size * 8 + c_size * 8) / orig_params

    print(f"  Q{quant_bits}+correction: Top1={t1_qc*100:.0f}% Top10={t10_qc*100:.0f}% BPW={total_bpw:.2f} correction={corr_params:,}")
    print(f"  Improvement: Top1 +{(t1_qc-t1_q)*100:.0f}%, Top10 +{(t10_qc-t10_q)*100:.0f}%")

    del q_model, correction, comp, qgd
    torch.cuda.empty_cache()

# ============================================================
# APPROACH B: Keep attention, genome FFN only
# ============================================================
print("\n" + "=" * 60)
print("APPROACH B: Keep attention weights, genome only FFN")
print("Attention at full/quantized precision, FFN replaced by genome")
print("=" * 60)

for attn_bits in [16, 4]:
    # Build model: attention at attn_bits, FFN replaced by genome
    ffn_genome = nn.ModuleList([LoRAGenomeLayer(1024, rank=64) for _ in range(28)]).to(device)
    ffn_params = sum(p.numel() for p in ffn_genome.parameters())

    # Prepare attention weights (quantized or full)
    attn_weights = {}
    for li in range(28):
        for wtype in ['attn_q', 'attn_k', 'attn_v', 'attn_output', 'attn_norm', 'ffn_norm']:
            key = f'blk.{li}.{wtype}.weight'
            if key in gd:
                if attn_bits < 16 and 'norm' not in wtype:
                    attn_weights[key] = quantize_weight(gd[key].float(), attn_bits).to(device)
                else:
                    attn_weights[key] = gd[key].float().to(device)
        # Also keep q_norm, k_norm
        for wtype in ['attn_q_norm', 'attn_k_norm']:
            key = f'blk.{li}.{wtype}.weight'
            if key in gd:
                attn_weights[key] = gd[key].float().to(device)

    opt = torch.optim.AdamW(ffn_genome.parameters(), lr=0.001, weight_decay=0.005)
    for step in range(3000):
        torch.manual_seed(step)
        tokens = torch.randint(100, 100000, (4, 32), device=device)
        with torch.no_grad():
            teacher_logits = teacher.forward(tokens, max_layers=28)[:, -1, :]

        x = F.embedding(tokens, embed).float()
        for li in range(28):
            # Real attention (with original/quantized weights)
            tw = {}
            for wtype in ['attn_q', 'attn_k', 'attn_v', 'attn_output', 'attn_norm', 'ffn_norm', 'attn_q_norm', 'attn_k_norm']:
                key = f'blk.{li}.{wtype}.weight'
                if key in attn_weights:
                    tw[wtype] = attn_weights[key]
            layer = TransformerLayer(tw, config)
            # Run attention only (not FFN)
            from ultracompress.inference import RMSNorm, linear_forward, rope_embed
            import numpy as np
            B, T, C = x.shape
            attn_out = layer.attention(layer.attn_norm(x), positions)
            if attn_out.shape == x.shape:
                h = x + attn_out
            else:
                h = x.clone()
                min_d = min(attn_out.shape[-1], x.shape[-1])
                h[..., :min_d] = h[..., :min_d] + attn_out[..., :min_d]

            # Genome FFN instead of real FFN
            h_norm = layer.ffn_norm(h) if hasattr(layer, 'ffn_norm') else h
            ffn_out = ffn_genome[li](h_norm)
            x = h + ffn_out

        var = x.float().pow(2).mean(-1, keepdim=True)
        xn = x.float() * torch.rsqrt(var + 1e-6) * norm_w
        student_logits = F.linear(xn, lm_head)[:, -1, :]
        loss = F.kl_div(F.log_softmax(student_logits/2, -1), F.softmax(teacher_logits/2, -1), reduction='batchmean') * 4
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(ffn_genome.parameters(), 1.0)
        opt.step()
        if step % 1000 == 0:
            print(f"  Attn@{attn_bits}bit step {step}: loss={loss.item():.3f}")
            sys.stdout.flush()

    def b_forward(tokens):
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

    t1_b, t10_b = quick_eval(b_forward)
    attn_params = sum(v.numel() for v in attn_weights.values())
    total_size = attn_params * attn_bits / 8 + ffn_params * 2
    print(f"  Attn@{attn_bits}bit + genome FFN: Top1={t1_b*100:.0f}% Top10={t10_b*100:.0f}%")
    print(f"  Size: {total_size/1e6:.1f} MB (attn={attn_params*attn_bits/8/1e6:.1f}MB + genome={ffn_params*2/1e6:.1f}MB)")

    del ffn_genome, attn_weights
    torch.cuda.empty_cache()

print("\n" + "=" * 60)
print("SWEEP COMPLETE — check results above")
print("=" * 60)
