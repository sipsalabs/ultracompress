"""Test V2 genome architectures: MultiView + LoRA-style.

Compares V1 (MicroTransformerLayer) vs V2 (MultiView, LoRA)
at the same parameter count to see which architecture gives
best quality per parameter.
"""
import torch, sys, time
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, '.')
from ultracompress.inference import ModelConfig, MiniTransformer
from ultracompress.genome_compressor import GenomeModel, MicroTransformerLayer, GenomeCompressor
from ultracompress.genome_v2 import MultiViewGenomeLayer, LoRAGenomeLayer

device = 'cuda'
wd = torch.load('qwen3_0.6b_cache.pt', weights_only=True)
config = ModelConfig(n_layers=28, n_heads=16, n_kv_heads=8, hidden_size=1024,
                     intermediate_size=3072, vocab_size=151936, head_dim=128)

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
embed = gd['token_embd.weight'].to(device)
norm_w = gd['output_norm.weight'].to(device)
lm_head = gd['output.weight'].to(device)

# Build cache
print("Building cache...")
compressor = GenomeCompressor(model_weights=wd, model_config=config, device=device)
cache = compressor.build_cache(n_samples=5000, batch_size=16, seq_len=32, n_layers=28)

def train_and_eval(layers, name, n_steps=20000):
    genome = GenomeModel(
        vocab_size=151936, big_dim=1024, small_dim=128, n_heads=4,
        n_layers=28, embed_weight=embed, lm_head_weight=lm_head, norm_weight=norm_w,
    ).to(device)
    # Replace genome layers with custom ones
    genome.genome_layers = layers.to(device)

    params = sum(p.numel() for p in layers.parameters())
    opt = torch.optim.AdamW(layers.parameters(), lr=0.001, weight_decay=0.005)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_steps)

    t0 = time.time()
    for step in range(n_steps):
        idx = torch.randint(0, cache['tokens'].shape[0], (64,))
        tokens = cache['tokens'][idx].to(device)
        target = cache['logits'][idx].to(device)
        student = genome(tokens, max_layers=28)[:, -1, :]
        loss = F.kl_div(F.log_softmax(student/2, -1), F.softmax(target/2, -1), reduction='batchmean') * 4
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(layers.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 5000 == 0:
            print(f"  {name} step {step}: loss={loss.item():.3f}")
            sys.stdout.flush()

    # Eval
    t1, t10s = 0, []
    for trial in range(50):
        torch.manual_seed(trial * 13 + 9999)
        t = torch.randint(100, 50000, (1, 16), device=device)
        with torch.no_grad():
            tl = teacher.forward(t, max_layers=28)
            tp = tl[0,-1].argmax().item()
            tt10 = set(tl[0,-1].topk(10).indices.tolist())
            gl = genome(t, max_layers=28)
            gp = gl[0,-1].argmax().item()
            gt10 = set(gl[0,-1].topk(10).indices.tolist())
            if tp == gp: t1 += 1
            t10s.append(len(tt10 & gt10) / 10)

    elapsed = time.time() - t0
    print(f"  {name}: Top1={t1*2}% Top10={sum(t10s)/len(t10s)*100:.0f}% params={params:,} time={elapsed:.0f}s")
    return t1/50, sum(t10s)/len(t10s)

print("\n=== ARCHITECTURE COMPARISON ===\n")

# V1: MicroTransformerLayer (baseline)
v1_layers = nn.ModuleList([MicroTransformerLayer(1024, 128, 4) for _ in range(28)])
print(f"V1 MicroTransformer: {sum(p.numel() for p in v1_layers.parameters()):,} params")
train_and_eval(v1_layers, "V1-MicroTF")
del v1_layers; torch.cuda.empty_cache()

# V2a: MultiView
v2a_layers = nn.ModuleList([MultiViewGenomeLayer(1024, 128, n_views=4) for _ in range(28)])
print(f"\nV2a MultiView: {sum(p.numel() for p in v2a_layers.parameters()):,} params")
train_and_eval(v2a_layers, "V2a-MultiView")
del v2a_layers; torch.cuda.empty_cache()

# V2b: LoRA-style
v2b_layers = nn.ModuleList([LoRAGenomeLayer(1024, rank=64) for _ in range(28)])
print(f"\nV2b LoRA: {sum(p.numel() for p in v2b_layers.parameters()):,} params")
train_and_eval(v2b_layers, "V2b-LoRA")
del v2b_layers; torch.cuda.empty_cache()
