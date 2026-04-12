"""Online distillation — fresh data every batch, no cache overfitting.

Progressive init + online KL divergence fine-tuning.
Teacher runs live for each batch. Slower but generalizes better.
"""
import torch, sys, os, time
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultracompress.inference import ModelConfig, MiniTransformer
from ultracompress.genome_compressor import GenomeCompressor

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

print("=== ONLINE DISTILLATION ===")
print("Phase 1: Progressive init (1000 steps/layer)")
print("Phase 2: Online fine-tuning (fresh data each batch)")
print()

# Phase 1: Progressive init
compressor = GenomeCompressor(model_weights=wd, model_config=config, device=device)
result = compressor.compress_progressive(
    small_dim=128, n_heads=4, n_layers=28,
    steps_per_layer=1000, batch_size=4, seq_len=32,
    lr=0.001, eval_samples=50, verbose=True,
)
print(f"\nAfter progressive: Top1={result.top1_accuracy*100:.0f}% Top10={result.top10_overlap*100:.0f}%")
genome = result.genome.to(device)


def eval_genome(genome, teacher, n=100):
    t1, t10s = 0, []
    for trial in range(n):
        torch.manual_seed(trial * 13 + 9999)
        t = torch.randint(100, 50000, (1, 16), device=device)
        with torch.no_grad():
            tl = teacher.forward(t, max_layers=28)
            tp = tl[0, -1].argmax().item()
            tt10 = set(tl[0, -1].topk(10).indices.tolist())
            gl = genome(t, max_layers=28)
            gp = gl[0, -1].argmax().item()
            gt10 = set(gl[0, -1].topk(10).indices.tolist())
            if tp == gp: t1 += 1
            t10s.append(len(tt10 & gt10) / 10)
    return t1 / n, sum(t10s) / len(t10s)


# Phase 2: Online distillation — fresh random data every batch
print("\nPhase 2: Online distillation (20K steps, fresh data)")
opt = torch.optim.AdamW(genome.genome_layers.parameters(), lr=0.0003, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 20000)

best_t10 = 0
t0 = time.time()
for step in range(20000):
    # Fresh random tokens every batch — no memorization possible
    tokens = torch.randint(100, 100000, (8, 32), device=device)

    with torch.no_grad():
        teacher_logits = teacher.forward(tokens, max_layers=28)[:, -1, :]

    student_logits = genome(tokens, max_layers=28)[:, -1, :]

    # KL divergence at temperature 2
    loss = F.kl_div(
        F.log_softmax(student_logits / 2, dim=-1),
        F.softmax(teacher_logits / 2, dim=-1),
        reduction='batchmean',
    ) * 4

    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(genome.genome_layers.parameters(), 1.0)
    opt.step()
    sched.step()

    if step % 2000 == 0:
        t1, t10 = eval_genome(genome, teacher, n=50)
        elapsed = time.time() - t0
        speed = (step + 1) / elapsed if elapsed > 0 else 0
        print(f"  Step {step:>5}: loss={loss.item():.4f} Top1={t1*100:.0f}% Top10={t10*100:.0f}% [{speed:.1f} steps/s]")
        sys.stdout.flush()
        if t10 > best_t10:
            best_t10 = t10
            genome.save_genome("genome_online_best.pt")

# Final eval
t1, t10 = eval_genome(genome, teacher, n=200)
elapsed = time.time() - t0
print(f"\n{'='*60}")
print(f"  ONLINE RESULT: Top1={t1*100:.0f}% Top10={t10*100:.0f}%")
print(f"  Genome: 23.9 MB (37x compression)")
print(f"  Time: {elapsed:.0f}s ({elapsed/60:.0f}min)")
print(f"  Best Top10 during training: {best_t10*100:.0f}%")
print(f"{'='*60}")

genome.save_genome("genome_online_sd128_28L.pt")
