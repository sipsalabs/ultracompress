"""FRR WITH REAL TEXT — The quality fix.

Previous demo produced "the the the" because we trained on random tokens.
This trains FRR on REAL text from FineWeb-Edu (same as Ouroboros uses).

Expected improvement: +15-25% top-10 from real text patterns alone.
"""
import torch, sys, os, time, math
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultracompress.inference import ModelConfig, MiniTransformer
from ultracompress.moonshot import FractalModel, GatedRecurrence

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
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
print(f"Tokenizer: {tokenizer.__class__.__name__}")

# Load real text dataset (streaming — no big download)
print("Loading FineWeb-Edu (streaming)...")
try:
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)
    ds_iter = iter(ds)
    USE_REAL_TEXT = True
    print("FineWeb-Edu loaded! Using REAL text.")
except Exception as e:
    print(f"Could not load FineWeb-Edu: {e}")
    print("Falling back to random tokens (install datasets: pip install datasets)")
    USE_REAL_TEXT = False


def get_real_batch(batch_size=8, seq_len=64):
    """Get a batch of real tokenized text."""
    global ds_iter
    if not USE_REAL_TEXT:
        return torch.randint(100, 50000, (batch_size, seq_len), device=device)

    tokens_list = []
    for _ in range(batch_size):
        while True:
            try:
                sample = next(ds_iter)
                text = sample.get('text', '')
                if len(text) < 50:
                    continue
                toks = tokenizer.encode(text, max_length=seq_len + 1,
                                       truncation=True, return_tensors='pt')[0]
                if len(toks) >= seq_len:
                    tokens_list.append(toks[:seq_len])
                    break
            except StopIteration:
                ds_iter = iter(ds)
    return torch.stack(tokens_list).to(device)


# Build FRR model
print("\nBuilding FRR model...")
model = FractalModel(
    hidden_dim=1024, n_heads=8, n_scales=4, iters_per_scale=7,
    vocab_size=151936, ff_mult=2,
    embed_weight=embed, lm_head_weight=lm_head_w, norm_weight=norm_w,
).to(device)
model.enable_adapters(rank=16)
print(f"Params: {model.fractal_params():,} ({model.fractal_params()*2/1e6:.1f} MB)")


def eval_model(n=100):
    t1, t10s = 0, []
    for trial in range(n):
        torch.manual_seed(trial * 13 + 9999)
        t = torch.randint(100, 50000, (1, 16), device=device)
        with torch.no_grad():
            tl = teacher.forward(t, max_layers=28)
            tp = tl[0, -1].argmax().item()
            tt10 = set(tl[0, -1].topk(10).indices.tolist())
            gl = model(t)
            gp = gl[0, -1].argmax().item()
            gt10 = set(gl[0, -1].topk(10).indices.tolist())
            if tp == gp: t1 += 1
            t10s.append(len(tt10 & gt10) / 10)
    return t1/n, sum(t10s)/len(t10s)


# Train with REAL TEXT
STEPS = 20000
print(f"\nTraining FRR on {'REAL TEXT (FineWeb-Edu)' if USE_REAL_TEXT else 'random tokens'}")
print(f"Steps: {STEPS}")
sys.stdout.flush()

trainable = [p for p in model.parameters() if p.requires_grad
             and id(p) not in {id(ep) for ep in model.embed.parameters()}
             and id(p) not in {id(lp) for lp in model.lm_head.parameters()}
             and id(p) not in {id(np_) for np_ in model.norm.parameters()}]

opt = torch.optim.AdamW(trainable, lr=0.0005, weight_decay=0.01)
t0 = time.time()

for step in range(STEPS):
    if step < 1000:
        lr = 0.0005 * step / 1000
    else:
        lr = 0.0005 * 0.5 * (1 + math.cos((step - 1000) / (STEPS - 1000) * math.pi))
    for pg in opt.param_groups: pg['lr'] = lr

    tokens = get_real_batch(batch_size=8, seq_len=64)

    with torch.no_grad():
        teacher_logits = teacher.forward(tokens, max_layers=28)

    student_logits = model(tokens)
    B, T, V = student_logits.shape
    loss = F.kl_div(F.log_softmax(student_logits.reshape(-1, V)/2, -1),
                   F.softmax(teacher_logits.reshape(-1, V)/2, -1),
                   reduction='batchmean') * 4

    if torch.isnan(loss):
        for pg in opt.param_groups: pg['lr'] *= 0.1
        continue

    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(trainable, 0.5)
    opt.step()

    if step % 4000 == 0:
        t1, t10 = eval_model()
        print(f"  Step {step}: loss={loss.item():.4f} Top1={t1*100:.0f}% Top10={t10*100:.0f}% "
              f"({'REAL' if USE_REAL_TEXT else 'RANDOM'} text) ({time.time()-t0:.0f}s)")
        sys.stdout.flush()

# Final eval
t1, t10 = eval_model()
print(f"\nFINAL: Top1={t1*100:.0f}% Top10={t10*100:.0f}%")
print(f"Training data: {'FineWeb-Edu (REAL)' if USE_REAL_TEXT else 'Random tokens'}")
print(f"Time: {time.time()-t0:.0f}s")

# Generate text
print("\n" + "=" * 70)
print("TEXT GENERATION from FRR trained on REAL TEXT")
print("=" * 70)

prompts = [
    "The future of artificial intelligence is",
    "Once upon a time in a land far away",
    "The most important thing about compression is",
]

model.eval()
for prompt in prompts:
    print(f"\n--- Prompt: '{prompt}' ---")
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Teacher
    torch.manual_seed(42)
    tokens = input_ids.clone()
    with torch.no_grad():
        for _ in range(40):
            logits = teacher.forward(tokens, max_layers=28)
            next_logits = logits[0, -1] / 0.7
            topk = next_logits.topk(40)
            next_logits = torch.full_like(next_logits, float('-inf'))
            next_logits[topk.indices] = topk.values
            next_token = torch.multinomial(F.softmax(next_logits, -1), 1)
            if next_token.item() in [151643, 151644, 151645]: break
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], 1)
    print(f"  TEACHER: {tokenizer.decode(tokens[0], skip_special_tokens=True)}")

    # FRR
    torch.manual_seed(42)
    tokens = input_ids.clone()
    with torch.no_grad():
        for _ in range(40):
            logits = model(tokens)
            next_logits = logits[0, -1] / 0.7
            topk = next_logits.topk(40)
            next_logits = torch.full_like(next_logits, float('-inf'))
            next_logits[topk.indices] = topk.values
            next_token = torch.multinomial(F.softmax(next_logits, -1), 1)
            if next_token.item() in [151643, 151644, 151645]: break
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], 1)
    print(f"  FRR:     {tokenizer.decode(tokens[0], skip_special_tokens=True)}")
    sys.stdout.flush()

torch.save(model.state_dict(), 'frr_real_text_best.pt')
print(f"\nSaved frr_real_text_best.pt")
print(f"{'='*70}")
