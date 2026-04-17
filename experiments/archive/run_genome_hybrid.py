"""Hybrid: Progressive init + long cached fine-tuning.
Best of both: good per-layer init + fast scaling via cache."""
import torch, sys
sys.path.insert(0, '.')
from ultracompress.inference import ModelConfig
from ultracompress.genome_compressor import GenomeCompressor

device = 'cuda'
wd = torch.load('qwen3_0.6b_cache.pt', weights_only=True)
config = ModelConfig(
    n_layers=28, n_heads=16, n_kv_heads=8, hidden_size=1024,
    intermediate_size=3072, vocab_size=151936, head_dim=128,
)

print("=== HYBRID: Progressive init + long cached training ===")
print("Phase 1: Progressive per-layer init (1000 steps/layer)")
print("Phase 2: Cache teacher outputs")
print("Phase 3: Long cached fine-tuning (50K steps)")
print()

compressor = GenomeCompressor(model_weights=wd, model_config=config, device=device)

# Phase 1: Progressive init
result = compressor.compress_progressive(
    small_dim=128, n_heads=4, n_layers=28,
    steps_per_layer=1000, batch_size=4, seq_len=32,
    lr=0.001, eval_samples=50, verbose=True,
)
print(f"\nAfter progressive: Top1={result.top1_accuracy*100:.0f}% Top10={result.top10_overlap*100:.0f}%")

# Save the progressive-initialized genome
genome = result.genome

# Phase 2: Build cache
cache = compressor.build_cache(n_samples=10000, batch_size=16, seq_len=32, n_layers=28)

# Phase 3: Long cached fine-tuning using the progressive-initialized genome
import torch.nn.functional as F
import torch.nn as nn
import time

embed = cache['embed'].to(device)
norm_w = cache['norm'].to(device)
lm_head = cache['head'].to(device)
cached_tokens = cache['tokens']
cached_logits = cache['logits']

opt = torch.optim.AdamW(genome.genome_layers.parameters(), lr=0.0003, weight_decay=0.005)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 50000)

print(f"\nPhase 3: Cached fine-tuning (50K steps from progressive init)...")
t0 = time.time()
for step in range(50000):
    idx = torch.randint(0, cached_tokens.shape[0], (64,))
    tokens = cached_tokens[idx].to(device)
    target = cached_logits[idx].to(device)

    student = genome(tokens, max_layers=28)[:, -1, :]
    loss = F.kl_div(
        F.log_softmax(student / 2, dim=-1),
        F.softmax(target / 2, dim=-1),
        reduction='batchmean',
    ) * 4

    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(genome.genome_layers.parameters(), 1.0)
    opt.step()
    sched.step()

    if step % 10000 == 0:
        print(f"  Step {step:>5}: loss={loss.item():.3f} [{(step+1)/(time.time()-t0):.1f} steps/s]")
        sys.stdout.flush()

# Final eval
from ultracompress.inference import MiniTransformer
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

t1, t10s = 0, []
for trial in range(100):
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
print(f"\n{'='*60}")
print(f"  HYBRID RESULT: Top1={t1}% Top10={sum(t10s)/len(t10s)*100:.0f}%")
print(f"  Genome: 23.9 MB (37x compression)")
print(f"  Total time: {elapsed:.0f}s ({elapsed/60:.0f}min)")
print(f"{'='*60}")

genome.save_genome("genome_hybrid_sd128_28L.pt")
