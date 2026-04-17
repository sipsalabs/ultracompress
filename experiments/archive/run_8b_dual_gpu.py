"""
8B FRR TRAINING — Dual GPU setup.
Teacher (Qwen3-8B, ~16GB) on GPU 0, FRR student on GPU 1.
This is the next scaling milestone after 1.7B.

8B has 32 layers, hidden=4096, n_heads=32, n_kv_heads=8.
FRR block: ~200M params. Compression: 32x (32 layers shared).

Memory estimate:
  GPU 0: Teacher 8B FP32 ~32GB (tight!) or FP16 ~16GB
  GPU 1: FRR student ~800MB + optimizer ~2.4GB + activations ~2GB = ~5GB

Strategy: Load teacher in FP16 on GPU 0, student on GPU 1.
Forward teacher on GPU 0, transfer logits to GPU 1, train student.
"""
import lib.unbuffered
import torch, sys, os, time
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

STEPS = 15000
TEACHER_DEVICE = 'cuda:0'
STUDENT_DEVICE = 'cuda:1'

print("=" * 70)
print("8B FRR TRAINING — Dual GPU")
print("=" * 70)

# Check if 8B cache exists
cache_path = 'qwen3_8b_cache.pt'
if not os.path.exists(cache_path):
    print(f"{cache_path} not found. Downloading Qwen3-8B...")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", torch_dtype=torch.float16, device_map="cpu"
    )
    sd = model.state_dict()
    torch.save(sd, cache_path)
    print(f"Saved {cache_path} ({os.path.getsize(cache_path)/1e9:.1f} GB)")
    del model, sd
    import gc; gc.collect()

print(f"Loading {cache_path}...")
wd = torch.load(cache_path, weights_only=True, map_location='cpu')

# Get 8B architecture info
sample_q = wd['model.layers.0.self_attn.q_proj.weight']
hidden_size = sample_q.shape[1]
sample_k = wd['model.layers.0.self_attn.k_proj.weight']
n_kv_heads = sample_k.shape[0] // 128  # head_dim=128
n_heads = sample_q.shape[0] // 128
sample_gate = wd['model.layers.0.mlp.gate_proj.weight']
intermediate_size = sample_gate.shape[0]
n_layers = max(int(k.split('.')[2]) for k in wd.keys() if k.startswith('model.layers.')) + 1

print(f"Architecture: {n_layers} layers, hidden={hidden_size}, "
      f"heads={n_heads}, kv_heads={n_kv_heads}, intermediate={intermediate_size}")

# Build teacher on GPU 0
from ultracompress.inference import ModelConfig, MiniTransformer

hf_to_gguf = {
    'self_attn.q_proj.weight': 'attn_q.weight', 'self_attn.k_proj.weight': 'attn_k.weight',
    'self_attn.v_proj.weight': 'attn_v.weight', 'self_attn.o_proj.weight': 'attn_output.weight',
    'self_attn.q_norm.weight': 'attn_q_norm.weight', 'self_attn.k_norm.weight': 'attn_k_norm.weight',
    'input_layernorm.weight': 'attn_norm.weight', 'post_attention_layernorm.weight': 'ffn_norm.weight',
    'mlp.gate_proj.weight': 'ffn_gate.weight', 'mlp.up_proj.weight': 'ffn_up.weight',
    'mlp.down_proj.weight': 'ffn_down.weight',
}
# Build gd dict — FP16 on teacher GPU (16GB fits alongside Ollama on GPU 1)
# MiniTransformer's linear_forward casts to float32 during forward pass
print(f"Building teacher on {TEACHER_DEVICE} (FP16 — 16GB, forward casts to FP32)...")
gd = {}
gd['token_embd.weight'] = wd['model.embed_tokens.weight'].half().to(TEACHER_DEVICE)
gd['output_norm.weight'] = wd.get('model.norm.weight', torch.ones(hidden_size)).half().to(TEACHER_DEVICE)
gd['output.weight'] = wd.get('lm_head.weight', wd['model.embed_tokens.weight']).half().to(TEACHER_DEVICE)
for li in range(n_layers):
    for h, g in hf_to_gguf.items():
        k = f'model.layers.{li}.{h}'
        if k in wd: gd[f'blk.{li}.{g}'] = wd[k].half().to(TEACHER_DEVICE)
    # Free each layer from CPU after moving to GPU
    for h in hf_to_gguf:
        k = f'model.layers.{li}.{h}'
        if k in wd: del wd[k]
    if li % 4 == 0:
        import gc; gc.collect()
        print(f"  Loaded layer {li}/{n_layers}")

del wd  # Free remaining CPU memory
import gc; gc.collect()

config = ModelConfig(n_layers=n_layers, n_heads=n_heads, n_kv_heads=n_kv_heads,
                     hidden_size=hidden_size, intermediate_size=intermediate_size,
                     vocab_size=151936, head_dim=128)

print(f"Loading teacher on {TEACHER_DEVICE}...")
teacher = MiniTransformer(config, TEACHER_DEVICE)
teacher.load_weights(gd)
teacher.embed_weight = teacher.embed_weight.to(TEACHER_DEVICE)
if teacher.lm_head is not None: teacher.lm_head = teacher.lm_head.to(TEACHER_DEVICE)

teacher_layer_params = sum(v.numel() for k, v in gd.items() if k.startswith('blk.'))

# Build FRR student on GPU 1
from ultracompress.moonshot import FractalModel

embed_w = gd['token_embd.weight'].float().to(STUDENT_DEVICE)
norm_w = gd['output_norm.weight'].float().to(STUDENT_DEVICE)
lm_head_w = gd['output.weight'].float().to(STUDENT_DEVICE)

print(f"Building FRR student on {STUDENT_DEVICE}...")
model = FractalModel(
    hidden_dim=hidden_size, n_heads=n_heads, n_scales=4,
    iters_per_scale=n_layers // 4,
    vocab_size=151936, ff_mult=1,
    embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w
).to(STUDENT_DEVICE)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
compression = teacher_layer_params / trainable
print(f"FRR trainable: {trainable:,} ({compression:.1f}x)")
print(f"Teacher layer params: {teacher_layer_params:,}")


def eval_model(model, n=50):
    t1, t10s = 0, []
    for trial in range(n):
        torch.manual_seed(trial * 13 + 9999)
        t = torch.randint(100, 50000, (1, 16))
        with torch.no_grad():
            tl = teacher.forward(t.to(TEACHER_DEVICE), max_layers=n_layers)
            tp = tl[0, -1].argmax().item()
            tt10 = set(tl[0, -1].topk(10).indices.tolist())
            gl = model(t.to(STUDENT_DEVICE))
            gp = gl[0, -1].argmax().item()
            gt10 = set(gl[0, -1].topk(10).indices.tolist())
            if tp == gp: t1 += 1
            t10s.append(len(tt10 & gt10) / 10)
    return t1 / n, sum(t10s) / len(t10s)


print(f"\nTraining {STEPS} steps (dual GPU)...")
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                        lr=2e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)

t0 = time.time()
for step in range(STEPS):
    torch.manual_seed(step * 7)
    tokens = torch.randint(100, 50000, (1, 32))  # Small batch for memory

    # Teacher forward on GPU 0
    with torch.no_grad():
        teacher_logits = teacher.forward(tokens.to(TEACHER_DEVICE), max_layers=n_layers)
        teacher_logits = teacher_logits.to(STUDENT_DEVICE)  # Transfer to GPU 1

    # Student forward on GPU 1
    student_logits = model(tokens.to(STUDENT_DEVICE))

    T = max(2.0, 5.0 * (1 - step / STEPS))
    loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=-1),
        F.softmax(teacher_logits / T, dim=-1),
        reduction='batchmean'
    ) * (T * T)

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    scheduler.step()

    if step % 3000 == 0 or step == STEPS - 1:
        t1, t10 = eval_model(model, n=50)
        elapsed = time.time() - t0
        print(f"  Step {step}/{STEPS}: loss={loss.item():.4f} T1={t1*100:.0f}% T10={t10*100:.0f}% ({elapsed:.0f}s)")

t1, t10 = eval_model(model, n=200)
print(f"\nFINAL (200 samples): T1={t1*100:.0f}% T10={t10*100:.0f}%")
print(f"Compression: {compression:.1f}x")
print(f"\nSCALING COMPARISON:")
print(f"  0.6B (15K): 56% T10 at 60x")
print(f"  1.7B (15K): 61% T10 at 48x")
print(f"  8B (15K):   {t10*100:.0f}% T10 at {compression:.0f}x")
torch.save(model.state_dict(), 'frr_8b_best.pt')
print("Done!")
