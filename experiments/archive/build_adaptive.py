"""BUILD: Adaptive Element-Level Precision — keep important elements exact, crush the rest."""
import torch, sys, math
import torch.nn.functional as F
sys.path.insert(0, '.')
from ultracompress.inference import ModelConfig, MiniTransformer, compare_layer_outputs
from ultracompress.product_quantize import product_quantize

device = 'cuda'
wd = torch.load('qwen3_0.6b_cache.pt', weights_only=True)

hf_to_gguf = {
    'self_attn.q_proj.weight': 'attn_q.weight', 'self_attn.k_proj.weight': 'attn_k.weight',
    'self_attn.v_proj.weight': 'attn_v.weight', 'self_attn.o_proj.weight': 'attn_output.weight',
    'self_attn.q_norm.weight': 'attn_q_norm.weight', 'self_attn.k_norm.weight': 'attn_k_norm.weight',
    'input_layernorm.weight': 'attn_norm.weight', 'post_attention_layernorm.weight': 'ffn_norm.weight',
    'mlp.gate_proj.weight': 'ffn_gate.weight', 'mlp.up_proj.weight': 'ffn_up.weight',
    'mlp.down_proj.weight': 'ffn_down.weight',
}
def build_gguf(wdict, n):
    gd = {}
    gd['token_embd.weight'] = wdict['model.embed_tokens.weight'].float()
    gd['output_norm.weight'] = wdict.get('model.norm.weight', torch.ones(1024)).float()
    gd['output.weight'] = wdict.get('lm_head.weight', gd['token_embd.weight']).float()
    for li in range(n):
        for h, g in hf_to_gguf.items():
            k = f'model.layers.{li}.{h}'
            if k in wdict: gd[f'blk.{li}.{g}'] = wdict[k].float()
    return gd

config = ModelConfig(n_layers=4, n_heads=16, n_kv_heads=8, hidden_size=1024,
                     intermediate_size=3072, vocab_size=151936, head_dim=128)
model = MiniTransformer(config, device)
orig_gguf = build_gguf(wd, 4)
model.load_weights(orig_gguf)

# Diverse activations for importance computation
print("Generating diverse activations...")
all_acts = []
for seed in range(50):
    torch.manual_seed(seed * 137)
    tokens = torch.randint(50, 50000, (4, 32), device=device)
    with torch.no_grad():
        x = F.embedding(tokens, model.embed_weight.to(device)).float()
        norm_w = orig_gguf.get('blk.0.attn_norm.weight', torch.ones(1024)).to(device)
        var = x.float().pow(2).mean(-1, keepdim=True)
        x_normed = x.float() * torch.rsqrt(var + 1e-6) * norm_w
        all_acts.append(x_normed.reshape(-1, 1024))

weight_keys = [k for k in wd.keys() if any(p in k for p in
    ['q_proj.weight', 'k_proj.weight', 'v_proj.weight', 'o_proj.weight',
     'gate_proj.weight', 'up_proj.weight', 'down_proj.weight'])
    and int(k.split('model.layers.')[1].split('.')[0]) < 4]

# Importance distribution
W = wd['model.layers.0.self_attn.q_proj.weight'].float().to(device)
avg_x = torch.zeros(W.shape[1], device=device)
for acts in all_acts:
    if acts.shape[1] == W.shape[1]:
        avg_x += acts.abs().mean(dim=0)
avg_x /= sum(1 for a in all_acts if a.shape[1] == W.shape[1])
imp = W.abs() * avg_x.unsqueeze(0)
flat_imp = imp.reshape(-1)
sorted_imp = flat_imp.sort(descending=True).values
total_imp = flat_imp.sum()

print("\nElement importance distribution (q_proj layer 0):")
for pct in [1, 2, 5, 10, 20, 50]:
    k = int(W.numel() * pct / 100)
    print(f"  Top {pct:>2}%: {sorted_imp[:k].sum()/total_imp*100:.1f}% of total importance")

# Text quality test
print("\n=== MIXED ELEMENT PRECISION: TEXT QUALITY ===")
test_tokens = torch.randint(100, 5000, (1, 16), device=device)
total_elements = sum(wd[k].numel() for k in weight_keys)

for keep_pct in [5, 10, 20, 30, 50, 75]:
    comp = dict(wd)
    total_bits = 0

    for key in weight_keys:
        w = comp[key].float().to(device)
        # Per-element importance
        avg_act = torch.zeros(w.shape[1], device=device)
        count = 0
        for acts in all_acts[:20]:
            if acts.shape[1] == w.shape[1]:
                avg_act += acts.abs().mean(dim=0)
                count += 1
        if count > 0:
            avg_act /= count
        else:
            avg_act = torch.ones(w.shape[1], device=device)

        element_imp = w.abs() * avg_act.unsqueeze(0)
        threshold = torch.quantile(element_imp.reshape(-1), 1 - keep_pct / 100)
        important = element_imp >= threshold

        # Important: keep FP16. Unimportant: 1-bit (sign * avg magnitude)
        w_mixed = w.clone()
        not_imp = ~important
        if not_imp.any():
            avg_mag = w[not_imp].abs().mean()
            w_mixed[not_imp] = w[not_imp].sign() * avg_mag

        comp[key] = w_mixed.cpu()
        total_bits += important.sum().item() * 16 + not_imp.sum().item() * 1

    bpw = total_bits / total_elements
    comp_gguf = build_gguf(comp, 4)
    res = compare_layer_outputs(orig_gguf, comp_gguf, config, test_tokens, device='cuda', max_layers=4)

    l0 = res.get('layer_0', {}).get('cosine_sim', 0)
    l3 = res.get('layer_3', {}).get('cosine_sim', 0)
    logit = res.get('logits', {}).get('cosine_sim', 0)
    top10 = res.get('top10_agreement', 0)
    top1 = 'Y' if res.get('top1_match', False) else 'N'

    print(f"  {keep_pct:>2}% FP16 + rest 1-bit: BPW={bpw:.2f} L0={l0:.4f} L3={l3:.4f} logit={logit:.4f} T10={top10*100:.0f}% T1={top1}")

# Baselines
print("\nBaselines:")
for M, K, G in [(8, 256, 32), (4, 64, 32), (8, 16, 64)]:
    comp2 = dict(wd)
    tb = 0
    for key in weight_keys:
        w = comp2[key].float().to(device)
        if w.numel() >= G * 2:
            pq = product_quantize(w, n_subvectors=M, codebook_size=K, group_size=G, n_iter=20)
            comp2[key] = pq.decompress().reshape(w.shape).cpu()
            tb += pq.storage_bytes() * 8
        else:
            tb += w.numel() * 16
    bpw2 = tb / total_elements
    c2g = build_gguf(comp2, 4)
    r2 = compare_layer_outputs(orig_gguf, c2g, config, test_tokens, device='cuda', max_layers=4)
    l0 = r2.get('layer_0', {}).get('cosine_sim', 0)
    logit = r2.get('logits', {}).get('cosine_sim', 0)
    top10 = r2.get('top10_agreement', 0)
    top1 = 'Y' if r2.get('top1_match', False) else 'N'
    print(f"  PQ M={M}K={K}G={G}: BPW={bpw2:.2f} L0={l0:.4f} logit={logit:.4f} T10={top10*100:.0f}% T1={top1}")
