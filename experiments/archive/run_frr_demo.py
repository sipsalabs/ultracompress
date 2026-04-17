"""FRR TEXT GENERATION DEMO — Make the fractal model actually speak.

Train FRR on Qwen3-0.6B, then generate text from it.
This is the demo that proves the concept works.
"""
import torch, sys, os, time, math
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultracompress.inference import ModelConfig, MiniTransformer
from ultracompress.moonshot import FractalModel

device = 'cuda'
print("Loading teacher + tokenizer...")
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
teacher.embed_weight = teacher.embed_weight.to(device)
if teacher.lm_head is not None:
    teacher.lm_head = teacher.lm_head.to(device)

embed = gd['token_embd.weight'].to(device)
norm_w = gd['output_norm.weight'].to(device)
lm_head_w = gd['output.weight'].to(device)

# Load tokenizer
try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
    print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")
except:
    tokenizer = None
    print("No tokenizer available, will use token IDs only")

# Build FRR model (best config from V1: 4 scales x 7 iters)
print("\nBuilding FRR model (4s7i, 21MB, 42x compression)...")
model = FractalModel(
    hidden_dim=1024, n_heads=8, n_scales=4, iters_per_scale=7,
    vocab_size=151936, ff_mult=2,
    embed_weight=embed, lm_head_weight=lm_head_w, norm_weight=norm_w,
).to(device)
print(f"Fractal params: {model.fractal_params():,} ({model.fractal_params()*2/1e6:.1f} MB)")

# Train with all-position KL (same as V1 best recipe)
print("\nTraining FRR (15K steps, all-position KL)...")
trainable = list(model.block.parameters()) + [model.scale_gamma, model.scale_beta, model.iter_scale]
opt = torch.optim.AdamW(trainable, lr=0.0005, weight_decay=0.01)

t0 = time.time()
for step in range(15000):
    if step < 1000:
        lr = 0.0005 * step / 1000
    else:
        lr = 0.0005 * 0.5 * (1 + math.cos((step - 1000) / 14000 * math.pi))
    for pg in opt.param_groups: pg['lr'] = lr

    tokens = torch.randint(100, 100000, (8, 32), device=device)
    with torch.no_grad():
        teacher_logits = teacher.forward(tokens, max_layers=28)
    student_logits = model(tokens)

    B, T, V = student_logits.shape
    loss = F.kl_div(
        F.log_softmax(student_logits.reshape(-1, V) / 2, -1),
        F.softmax(teacher_logits.reshape(-1, V) / 2, -1),
        reduction='batchmean') * 4

    if torch.isnan(loss):
        for pg in opt.param_groups: pg['lr'] *= 0.1
        continue

    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(trainable, 0.5)
    opt.step()

    if step % 3000 == 0:
        print(f"  Step {step}: loss={loss.item():.4f} ({time.time()-t0:.0f}s)")
        sys.stdout.flush()

print(f"\nTraining done ({time.time()-t0:.0f}s)")
sys.stdout.flush()


# ============================================================
# GENERATE TEXT
# ============================================================
def generate(model, prompt_tokens, max_new=100, temperature=0.8, top_k=50):
    """Autoregressive generation from FRR model."""
    tokens = prompt_tokens.clone()
    model.eval()
    with torch.no_grad():
        for _ in range(max_new):
            logits = model(tokens)
            next_logits = logits[0, -1, :] / temperature
            # Top-k filtering
            if top_k > 0:
                topk_vals, topk_idx = next_logits.topk(top_k)
                next_logits = torch.full_like(next_logits, float('-inf'))
                next_logits[topk_idx] = topk_vals
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
            # Stop on EOS
            if next_token.item() in [151643, 151644, 151645]:  # Qwen EOS tokens
                break
    model.train()
    return tokens


def generate_teacher(prompt_tokens, max_new=100, temperature=0.8, top_k=50):
    """Generate from teacher for comparison."""
    tokens = prompt_tokens.clone()
    with torch.no_grad():
        for _ in range(max_new):
            logits = teacher.forward(tokens, max_layers=28)
            next_logits = logits[0, -1, :] / temperature
            if top_k > 0:
                topk_vals, topk_idx = next_logits.topk(top_k)
                next_logits = torch.full_like(next_logits, float('-inf'))
                next_logits[topk_idx] = topk_vals
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
            if next_token.item() in [151643, 151644, 151645]:
                break
    return tokens


print("\n" + "=" * 70)
print("TEXT GENERATION COMPARISON: Teacher vs FRR (42x compression)")
print("=" * 70)

prompts = [
    "The future of artificial intelligence is",
    "Once upon a time in a land far away",
    "The most important thing about compression is",
    "In the year 2030, technology will",
    "To build something truly revolutionary, you need",
]

for prompt_text in prompts:
    print(f"\n--- Prompt: '{prompt_text}' ---")

    if tokenizer:
        prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
    else:
        # Fallback: encode as bytes
        prompt_ids = torch.tensor([[ord(c) for c in prompt_text]], device=device)

    # Teacher output
    torch.manual_seed(42)
    teacher_tokens = generate_teacher(prompt_ids, max_new=60, temperature=0.7, top_k=40)
    if tokenizer:
        teacher_text = tokenizer.decode(teacher_tokens[0], skip_special_tokens=True)
    else:
        teacher_text = str(teacher_tokens[0].tolist())
    print(f"  TEACHER: {teacher_text}")

    # FRR output
    torch.manual_seed(42)
    frr_tokens = generate(model, prompt_ids, max_new=60, temperature=0.7, top_k=40)
    if tokenizer:
        frr_text = tokenizer.decode(frr_tokens[0], skip_special_tokens=True)
    else:
        frr_text = str(frr_tokens[0].tolist())
    print(f"  FRR:     {frr_text}")
    sys.stdout.flush()

print(f"\n{'='*70}")
print(f"FRR model: {model.fractal_params():,} params ({model.fractal_params()*2/1e6:.1f} MB)")
print(f"Teacher:   {sum(v.numel() for v in gd.values()):,} params")
print(f"Compression: 42x (ONE shared block, recursive)")
print(f"{'='*70}")

# Save the trained FRR model
torch.save({
    'block': model.block.state_dict(),
    'scale_gamma': model.scale_gamma,
    'scale_beta': model.scale_beta,
    'iter_scale': model.iter_scale,
    'config': {'n_scales': 4, 'iters_per_scale': 7, 'n_heads': 8, 'ff_mult': 2},
}, 'frr_demo_model.pt')
print("Saved frr_demo_model.pt")
