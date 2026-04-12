"""Test MoE genome — mixture of expert layers vs single bottleneck.

Compare at similar parameter counts:
  - V1 MicroTransformer (sd=128): 11.9M params
  - MoE (8 experts, expert_dim=32, top_k=2): ~12M params
  - MoE (16 experts, expert_dim=32, top_k=2): ~20M params

MoE should be more expressive because different tokens
activate different experts → more effective capacity.
"""
import torch, sys, os, time
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultracompress.inference import ModelConfig, MiniTransformer
from ultracompress.genome_moe import MoEGenomeModel
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

embed = gd['token_embd.weight']
norm_w = gd['output_norm.weight']
head_w = gd['output.weight']


def eval_model(model, n=100):
    t1, t10s = 0, []
    for trial in range(n):
        torch.manual_seed(trial * 13 + 9999)
        t = torch.randint(100, 50000, (1, 16), device=device)
        with torch.no_grad():
            tl = teacher.forward(t, max_layers=28)
            tp = tl[0, -1].argmax().item()
            tt10 = set(tl[0, -1].topk(10).indices.tolist())
            gl = model(t, max_layers=28)
            gp = gl[0, -1].argmax().item()
            gt10 = set(gl[0, -1].topk(10).indices.tolist())
            if tp == gp: t1 += 1
            t10s.append(len(tt10 & gt10) / 10)
    return t1 / n, sum(t10s) / len(t10s)


def train_online(model, name, n_steps=10000, batch_size=8, lr=0.001):
    """Online distillation — fresh data each batch."""
    params = [p for p in model.genome_layers.parameters()]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_steps)

    t0 = time.time()
    best_t10 = 0
    for step in range(n_steps):
        tokens = torch.randint(100, 100000, (batch_size, 32), device=device)
        with torch.no_grad():
            teacher_logits = teacher.forward(tokens, max_layers=28)[:, -1, :]

        student_logits = model(tokens, max_layers=28)[:, -1, :]
        loss = F.kl_div(
            F.log_softmax(student_logits / 2, -1),
            F.softmax(teacher_logits / 2, -1),
            reduction='batchmean',
        ) * 4

        # Add MoE load balancing loss if applicable
        if hasattr(model.genome_layers[0], 'aux_loss'):
            for layer in model.genome_layers:
                loss = loss + layer.aux_loss(
                    model.embed(tokens).float().detach()
                ) * 0.1

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()

        if step % 2000 == 0:
            t1, t10 = eval_model(model, n=50)
            elapsed = time.time() - t0
            speed = (step + 1) / elapsed if elapsed > 0 else 0
            print(f"  {name} step {step:>5}: loss={loss.item():.4f} Top1={t1*100:.0f}% Top10={t10*100:.0f}% [{speed:.1f} s/s]")
            sys.stdout.flush()
            if t10 > best_t10:
                best_t10 = t10
    return best_t10


print("=== MOE GENOME COMPARISON ===\n")

# Config 1: Standard V1 baseline (sd=128)
print("--- V1 MicroTransformer (sd=128) ---")
v1 = GenomeModel(
    vocab_size=151936, big_dim=1024, small_dim=128, n_heads=4,
    n_layers=28,
    embed_weight=embed.to(device),
    lm_head_weight=head_w.to(device),
    norm_weight=norm_w.to(device),
).to(device)
v1_params = v1.genome_param_count()
print(f"Params: {v1_params:,} ({v1_params*2/1e6:.1f} MB)")
best_v1 = train_online(v1, "V1-sd128", n_steps=10000)

# Config 2: MoE (8 experts, expert_dim=32, top_k=2) — similar param count
print("\n--- MoE (8 experts, dim=32, top_k=2) ---")
moe8 = MoEGenomeModel(
    vocab_size=151936, big_dim=1024, expert_dim=32, n_experts=8, top_k=2,
    n_layers=28,
    embed_weight=embed.to(device),
    lm_head_weight=head_w.to(device),
    norm_weight=norm_w.to(device),
).to(device)
moe8_params = moe8.genome_param_count()
print(f"Params: {moe8_params:,} ({moe8_params*2/1e6:.1f} MB)")
best_moe8 = train_online(moe8, "MoE-8x32", n_steps=10000)

# Config 3: MoE (16 experts, expert_dim=32, top_k=2) — more experts
print("\n--- MoE (16 experts, dim=32, top_k=2) ---")
moe16 = MoEGenomeModel(
    vocab_size=151936, big_dim=1024, expert_dim=32, n_experts=16, top_k=2,
    n_layers=28,
    embed_weight=embed.to(device),
    lm_head_weight=head_w.to(device),
    norm_weight=norm_w.to(device),
).to(device)
moe16_params = moe16.genome_param_count()
print(f"Params: {moe16_params:,} ({moe16_params*2/1e6:.1f} MB)")
best_moe16 = train_online(moe16, "MoE-16x32", n_steps=10000)

# Config 4: MoE (8 experts, expert_dim=64, top_k=2) — bigger experts
print("\n--- MoE (8 experts, dim=64, top_k=2) ---")
moe_big = MoEGenomeModel(
    vocab_size=151936, big_dim=1024, expert_dim=64, n_experts=8, top_k=2,
    n_layers=28,
    embed_weight=embed.to(device),
    lm_head_weight=head_w.to(device),
    norm_weight=norm_w.to(device),
).to(device)
moe_big_params = moe_big.genome_param_count()
print(f"Params: {moe_big_params:,} ({moe_big_params*2/1e6:.1f} MB)")
best_moe_big = train_online(moe_big, "MoE-8x64", n_steps=10000)

# Final comparison
print(f"\n{'='*60}")
print(f"RESULTS:")
print(f"  V1 sd=128:     {v1_params:>12,} params ({v1_params*2/1e6:>6.1f} MB) best_top10={best_v1*100:.0f}%")
print(f"  MoE 8x32:      {moe8_params:>12,} params ({moe8_params*2/1e6:>6.1f} MB) best_top10={best_moe8*100:.0f}%")
print(f"  MoE 16x32:     {moe16_params:>12,} params ({moe16_params*2/1e6:>6.1f} MB) best_top10={best_moe16*100:.0f}%")
print(f"  MoE 8x64:      {moe_big_params:>12,} params ({moe_big_params*2/1e6:>6.1f} MB) best_top10={best_moe_big*100:.0f}%")
print(f"{'='*60}")
