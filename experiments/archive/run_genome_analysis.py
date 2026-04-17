"""Analyze WHERE the genome fails — what patterns does it miss?

This tells us what to improve:
- If it misses rare tokens → needs more diverse training
- If it misses specific layers → those layers need bigger genomes
- If it's always "close" → just needs more capacity
- If it's randomly wrong → fundamental limitation
"""
import torch, sys, os
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultracompress.inference import ModelConfig, MiniTransformer
from ultracompress.genome_compressor import GenomeModel

device = 'cuda'
wd = torch.load('qwen3_0.6b_cache.pt', weights_only=True)
config = ModelConfig(
    n_layers=28, n_heads=16, n_kv_heads=8, hidden_size=1024,
    intermediate_size=3072, vocab_size=151936, head_dim=128,
)

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
teacher = MiniTransformer(config, device)
teacher.load_weights(gd)

# Load best genome (progressive sd=128)
genome = GenomeModel(
    vocab_size=151936, big_dim=1024, small_dim=128, n_heads=4,
    n_layers=28,
    embed_weight=gd['token_embd.weight'].to(device),
    lm_head_weight=gd['output.weight'].to(device),
    norm_weight=gd['output_norm.weight'].to(device),
).to(device)

# Try loading saved genome
saved = None
for path in ['genome_hybrid_sd128_28L.pt', 'genome_online_best.pt']:
    if os.path.exists(path):
        saved = torch.load(path, weights_only=True)
        print(f"Loaded genome from {path}")
        break

if saved and 'genome_state' in saved:
    genome.load_state_dict(saved['genome_state'], strict=False)
elif saved:
    # Try direct state dict
    try:
        genome.genome_layers.load_state_dict(
            {k.replace('genome_layers.', ''): v for k, v in saved.items()
             if 'genome_layers' in k}, strict=False)
    except:
        pass

print("=== GENOME FAILURE ANALYSIS ===\n")

# Analysis 1: Distribution of rank of correct answer
print("--- Where does the correct token rank in genome's predictions? ---")
ranks = []
top10_overlaps = []
kl_divs = []
teacher_entropies = []

for trial in range(500):
    torch.manual_seed(trial * 7 + 1234)
    t = torch.randint(100, 50000, (1, 16), device=device)
    with torch.no_grad():
        tl = teacher.forward(t, max_layers=28)[0, -1]
        gl = genome(t, max_layers=28)[0, -1]

        # Teacher's top-1 token
        teacher_token = tl.argmax().item()

        # Where does teacher's top-1 rank in genome's sorted list?
        genome_sorted = gl.argsort(descending=True)
        rank = (genome_sorted == teacher_token).nonzero(as_tuple=True)[0]
        if len(rank) > 0:
            ranks.append(rank[0].item())
        else:
            ranks.append(151936)  # worst case

        # Top-10 overlap
        tt10 = set(tl.topk(10).indices.tolist())
        gt10 = set(gl.topk(10).indices.tolist())
        top10_overlaps.append(len(tt10 & gt10))

        # KL divergence
        tp = F.softmax(tl, dim=-1)
        gp = F.softmax(gl, dim=-1)
        kl = F.kl_div(gp.log().clamp(min=-100), tp, reduction='sum').item()
        kl_divs.append(kl)

        # Teacher entropy (how confident is teacher?)
        entropy = -(tp * tp.log().clamp(min=-100)).sum().item()
        teacher_entropies.append(entropy)

import numpy as np
ranks = np.array(ranks)
print(f"  Rank 1 (exact match): {(ranks == 0).sum()}/{len(ranks)} = {(ranks == 0).mean()*100:.1f}%")
print(f"  Rank 1-5: {(ranks < 5).sum()}/{len(ranks)} = {(ranks < 5).mean()*100:.1f}%")
print(f"  Rank 1-10: {(ranks < 10).sum()}/{len(ranks)} = {(ranks < 10).mean()*100:.1f}%")
print(f"  Rank 1-50: {(ranks < 50).sum()}/{len(ranks)} = {(ranks < 50).mean()*100:.1f}%")
print(f"  Rank 1-100: {(ranks < 100).sum()}/{len(ranks)} = {(ranks < 100).mean()*100:.1f}%")
print(f"  Rank 100+: {(ranks >= 100).sum()}/{len(ranks)} = {(ranks >= 100).mean()*100:.1f}%")
print(f"  Median rank: {np.median(ranks):.0f}")
print(f"  Mean rank: {np.mean(ranks):.0f}")

# Analysis 2: Does teacher confidence predict genome accuracy?
print(f"\n--- Does teacher confidence predict genome accuracy? ---")
low_entropy = np.array(teacher_entropies) < np.median(teacher_entropies)
high_entropy = ~low_entropy
print(f"  When teacher is confident (low entropy):")
print(f"    Top-1 match: {(ranks[low_entropy] == 0).mean()*100:.1f}%")
print(f"    Top-10 overlap: {np.array(top10_overlaps)[low_entropy].mean():.1f}/10")
print(f"  When teacher is uncertain (high entropy):")
print(f"    Top-1 match: {(ranks[high_entropy] == 0).mean()*100:.1f}%")
print(f"    Top-10 overlap: {np.array(top10_overlaps)[high_entropy].mean():.1f}/10")

# Analysis 3: Per-layer activation divergence
print(f"\n--- Per-layer activation divergence ---")
torch.manual_seed(42)
tokens = torch.randint(100, 50000, (4, 32), device=device)
with torch.no_grad():
    # Teacher activations
    t_x = F.embedding(tokens, gd['token_embd.weight'].to(device)).float()
    positions = torch.arange(32, device=device)
    teacher_acts = [t_x.clone()]
    for li in range(28):
        t_x = teacher.layers[li](t_x, positions)
        teacher_acts.append(t_x.clone())

    # Genome activations
    g_x = genome.embed(tokens).float()
    genome_acts = [g_x.clone()]
    for li in range(28):
        g_x = g_x + genome.genome_layers[li](g_x)
        genome_acts.append(g_x.clone())

    # Compare
    for li in range(0, 29, 4):
        cos = F.cosine_similarity(teacher_acts[li].flatten(), genome_acts[li].flatten(), dim=0)
        rel_err = (teacher_acts[li] - genome_acts[li]).norm() / teacher_acts[li].norm()
        print(f"  After layer {li:>2}: cosine={cos.item():.4f} rel_err={rel_err.item():.4f}")

# Analysis 4: KL divergence distribution
print(f"\n--- KL divergence distribution ---")
kl_arr = np.array(kl_divs)
print(f"  Min KL: {kl_arr.min():.4f}")
print(f"  Median KL: {np.median(kl_arr):.4f}")
print(f"  Mean KL: {kl_arr.mean():.4f}")
print(f"  Max KL: {kl_arr.max():.4f}")
print(f"  Std KL: {kl_arr.std():.4f}")

print(f"\n{'='*60}")
print(f"SUMMARY: These patterns tell us WHERE to invest to push quality higher")
print(f"{'='*60}")
