#!/usr/bin/env python3
"""
QUICKSTART: Try UltraCompress in 5 minutes.

This script:
1. Downloads Qwen3-0.6B (~1.5 GB)
2. Trains FRR compression (5K steps, ~5 min on GPU)
3. Shows before/after comparison
4. Reports compression ratio and quality

Works on any GPU with 8+ GB VRAM. No setup required beyond pip install.

Usage:
  pip install torch transformers
  python quickstart.py
"""
import torch
import torch.nn.functional as F
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
if device == 'cpu':
    print("WARNING: CPU training will be very slow. Use a GPU for best results.")

# Step 1: Download model
print("\n[1/4] Downloading Qwen3-0.6B...")
from transformers import AutoModelForCausalLM, AutoTokenizer

if not os.path.exists('qwen3_0.6b_cache.pt'):
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B",
            torch_dtype=torch.float32, device_map='cpu')
    torch.save(model.state_dict(), 'qwen3_0.6b_cache.pt')
    del model
    import gc; gc.collect()
    print("  Downloaded and cached.")
else:
    print("  Already cached.")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)

# Step 2: Load teacher + build FRR
print("\n[2/4] Building FRR compressed model...")
from ultracompress.inference import ModelConfig, MiniTransformer
from ultracompress.moonshot import FractalModel

wd = torch.load('qwen3_0.6b_cache.pt', weights_only=True)
hf_to_gguf = {
    'self_attn.q_proj.weight': 'attn_q.weight', 'self_attn.k_proj.weight': 'attn_k.weight',
    'self_attn.v_proj.weight': 'attn_v.weight', 'self_attn.o_proj.weight': 'attn_output.weight',
    'self_attn.q_norm.weight': 'attn_q_norm.weight', 'self_attn.k_norm.weight': 'attn_k_norm.weight',
    'input_layernorm.weight': 'attn_norm.weight', 'post_attention_layernorm.weight': 'ffn_norm.weight',
    'mlp.gate_proj.weight': 'ffn_gate.weight', 'mlp.up_proj.weight': 'ffn_up.weight',
    'mlp.down_proj.weight': 'ffn_down.weight',
}
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
teacher.embed_weight = teacher.embed_weight.to(device)
if teacher.lm_head is not None: teacher.lm_head = teacher.lm_head.to(device)

embed_w = gd['token_embd.weight'].to(device)
norm_w = gd['output_norm.weight'].to(device)
lm_head_w = gd['output.weight'].to(device)

frr = FractalModel(1024, 16, 4, 7, 151936, 1,
                   embed_weight=embed_w, lm_head_weight=lm_head_w,
                   norm_weight=norm_w).to(device)

teacher_params = sum(v.numel() for v in gd.values())
frr_params = sum(p.numel() for p in frr.parameters())
frr_trainable = sum(p.numel() for p in frr.parameters() if p.requires_grad)
print(f"  Teacher: {teacher_params:,} params ({teacher_params*2/1e6:.0f} MB FP16)")
print(f"  FRR:     {frr_trainable:,} trainable ({frr_trainable*2/1e6:.1f} MB FP16)")
print(f"  Compression: {teacher_params/frr_params:.0f}x smaller")

# Step 3: Quick distillation
STEPS = 5000
print(f"\n[3/4] Distilling ({STEPS} steps, ~5 min on GPU)...")
opt = torch.optim.AdamW([p for p in frr.parameters() if p.requires_grad],
                        lr=5e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)

t0 = time.time()
for step in range(STEPS):
    torch.manual_seed(step * 7)
    tokens = torch.randint(100, 50000, (4, 32), device=device)
    with torch.no_grad():
        teacher_logits = teacher.forward(tokens, max_layers=28)
    student_logits = frr(tokens)
    T = max(2.0, 5.0 * (1 - step / STEPS))
    loss = F.kl_div(F.log_softmax(student_logits / T, dim=-1),
                    F.softmax(teacher_logits / T, dim=-1),
                    reduction='batchmean') * (T * T)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(frr.parameters(), 1.0)
    opt.step(); scheduler.step()
    if step % 1000 == 0:
        print(f"  Step {step}/{STEPS}: loss={loss.item():.4f} ({time.time()-t0:.0f}s)")

# Step 4: Evaluate
print(f"\n[4/4] Evaluating...")
t1, t10s = 0, []
for trial in range(100):
    torch.manual_seed(trial * 13 + 9999)
    t = torch.randint(100, 50000, (1, 16), device=device)
    with torch.no_grad():
        tl = teacher.forward(t, max_layers=28)
        gl = frr(t)
        if tl[0,-1].argmax() == gl[0,-1].argmax(): t1 += 1
        tt10 = set(tl[0,-1].topk(10).indices.tolist())
        gt10 = set(gl[0,-1].topk(10).indices.tolist())
        t10s.append(len(tt10 & gt10) / 10)

t10 = sum(t10s) / len(t10s)
print(f"\n{'='*50}")
print(f"  RESULTS")
print(f"{'='*50}")
print(f"  Top-1 agreement:  {t1}%")
print(f"  Top-10 agreement: {t10*100:.0f}%")
print(f"  Compression:      {teacher_params/frr_params:.0f}x")
print(f"  Original size:    {teacher_params*2/1e6:.0f} MB")
print(f"  FRR size:         {frr_trainable*2/1e6:.1f} MB")
print(f"  Training time:    {time.time()-t0:.0f}s")
print(f"\n  For better quality, train longer:")
print(f"    python compress_frr.py --model Qwen/Qwen3-0.6B --steps 50000")
print(f"    (63% T10 at 50K steps, 64% at 1.7B scale)")
