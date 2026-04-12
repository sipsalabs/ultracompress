"""
SPECULATIVE DECODING BENCHMARK: Measure actual wall-clock speedup.

Compares:
1. Standard generation (full model only)
2. Speculative decoding (FRR draft + full model verification)

Requires a trained FRR model (frr_100k_best.pt or similar).
"""
import lib.unbuffered
import torch, sys, os, time
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultracompress.inference import ModelConfig, MiniTransformer
from ultracompress.moonshot import FractalModel
from ultracompress.speculative import SpeculativeDecoder

device = 'cuda'
N_TOKENS = 100
N_TRIALS = 20

print("=" * 60)
print("SPECULATIVE DECODING BENCHMARK")
print("=" * 60)

# Load teacher
print("Loading teacher...")
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

# Load or build FRR
print("Loading FRR draft model...")
frr = FractalModel(1024, 16, 4, 7, 151936, 1,
                   embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w).to(device)

for ckpt in ['frr_100k_best.pt', 'frr_optimized_50k.pt']:
    if os.path.exists(ckpt):
        frr.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"Loaded: {ckpt}")
        break
else:
    print("No trained FRR found. Using random weights (results won't be meaningful).")

frr.eval()

frr_params = sum(p.numel() for p in frr.parameters())
frr_block = sum(p.numel() for p in frr.block.parameters())
print(f"FRR: {frr_params/1e6:.1f}M params, block: {frr_block*2/1e6:.1f} MB FP16")


def standard_generate(input_ids, max_tokens):
    """Standard autoregressive generation."""
    tokens = input_ids.clone()
    for _ in range(max_tokens):
        with torch.no_grad():
            logits = teacher.forward(tokens, max_layers=28)
        next_tok = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        tokens = torch.cat([tokens, next_tok], dim=1)
    return tokens


# Warmup
print("\nWarming up...")
prompt = torch.randint(100, 50000, (1, 16), device=device)
for _ in range(3):
    standard_generate(prompt, 10)
    spec = SpeculativeDecoder(
        lambda t: frr(t), lambda t: teacher.forward(t, max_layers=28), n_draft=4)
    spec.generate(prompt, 10)

# Benchmark standard generation
print(f"\nBenchmarking standard generation ({N_TRIALS} trials, {N_TOKENS} tokens each)...")
std_times = []
for i in range(N_TRIALS):
    torch.manual_seed(i * 42)
    prompt = torch.randint(100, 50000, (1, 16), device=device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    standard_generate(prompt, N_TOKENS)
    torch.cuda.synchronize()
    std_times.append(time.perf_counter() - t0)

std_avg = sum(std_times) / len(std_times)
std_tps = N_TOKENS / std_avg
print(f"  Standard: {std_avg*1000:.1f} ms avg, {std_tps:.0f} tokens/sec")

# Benchmark speculative decoding at different n_draft
for n_draft in [2, 3, 4, 5]:
    print(f"\nBenchmarking speculative decoding (n_draft={n_draft})...")
    spec_times = []
    spec_decoder = SpeculativeDecoder(
        lambda t: frr(t), lambda t: teacher.forward(t, max_layers=28),
        n_draft=n_draft, temperature=1.0)

    for i in range(N_TRIALS):
        torch.manual_seed(i * 42)
        prompt = torch.randint(100, 50000, (1, 16), device=device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        spec_decoder.generate(prompt, N_TOKENS)
        torch.cuda.synchronize()
        spec_times.append(time.perf_counter() - t0)

    spec_avg = sum(spec_times) / len(spec_times)
    spec_tps = N_TOKENS / spec_avg
    speedup = std_avg / spec_avg
    stats = spec_decoder.stats()

    print(f"  Speculative (n={n_draft}): {spec_avg*1000:.1f} ms avg, {spec_tps:.0f} tokens/sec")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Acceptance rate: {stats['acceptance_rate']*100:.1f}%")

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"  Standard: {std_tps:.0f} tokens/sec")
print(f"  FRR block size: {frr_block*2/1e6:.1f} MB (fits in L2 cache: {'YES' if frr_block*2 < 96e6 else 'NO'})")
print("Done!")
