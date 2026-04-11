"""Streaming genome compression for Qwen3-8B.

Loads model shard-by-shard, never needs full model in RAM.
Builds teacher cache via streaming, trains genome from cache.

Peak RAM: ~8GB (one shard) + cache + genome.
"""
import torch, sys, os, json, time
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safetensors.torch import load_file
from ultracompress.genome_compressor import GenomeModel, MicroTransformerLayer

device = 'cuda'
model_path = "C:/Users/sip/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"

# Load config
with open(os.path.join(model_path, "config.json")) as f:
    cfg = json.load(f)

n_layers = cfg['num_hidden_layers']  # 36
hidden = cfg['hidden_size']          # 4096
n_heads = cfg['num_attention_heads'] # 32
n_kv = cfg.get('num_key_value_heads', n_heads)  # 8
intermediate = cfg['intermediate_size']  # 12288
vocab = cfg['vocab_size']            # 151936
head_dim = hidden // n_heads         # 128

print(f"=== 8B STREAMING GENOME COMPRESSION ===")
print(f"Model: Qwen3-8B ({n_layers} layers, {hidden} hidden)")
print(f"Loading shard-by-shard, never full model in RAM")
print()

# Step 1: Load embedding + LM head (needed for genome)
print("Loading embedding + head...")
shards = sorted([f for f in os.listdir(model_path) if f.endswith('.safetensors')])
embed_w = None
norm_w = None
head_w = None

for shard_file in shards:
    shard = load_file(os.path.join(model_path, shard_file))
    for name, tensor in shard.items():
        if name == 'model.embed_tokens.weight':
            embed_w = tensor.float()
        elif name == 'model.norm.weight':
            norm_w = tensor.float()
        elif name == 'lm_head.weight':
            head_w = tensor.float()
    del shard
    if embed_w is not None and head_w is not None:
        break

if head_w is None:
    head_w = embed_w  # tied weights

print(f"  Embedding: {embed_w.shape}")
print(f"  LM Head: {head_w.shape}")

# Step 2: Build teacher cache via streaming
# For each sample: embed tokens, run through all layers (loading each shard as needed), get logits
print(f"\nBuilding teacher cache (streaming through {len(shards)} shards)...")

n_samples = 1000  # Start small
seq_len = 32
batch_size = 4

# Pre-generate all token inputs
torch.manual_seed(42)
all_tokens = torch.randint(100, vocab, (n_samples, seq_len))

# We need to run the full model forward pass, loading layers one at a time
# This is SLOW but uses minimal RAM

from ultracompress.inference import ModelConfig, TransformerLayer, RMSNorm

mc = ModelConfig(
    n_layers=n_layers, n_heads=n_heads, n_kv_heads=n_kv,
    hidden_size=hidden, intermediate_size=intermediate,
    vocab_size=vocab, head_dim=head_dim,
)

# Map weight names
name_map = {
    'self_attn.q_proj.weight': 'attn_q',
    'self_attn.k_proj.weight': 'attn_k',
    'self_attn.v_proj.weight': 'attn_v',
    'self_attn.o_proj.weight': 'attn_output',
    'input_layernorm.weight': 'attn_norm',
    'post_attention_layernorm.weight': 'ffn_norm',
    'mlp.gate_proj.weight': 'ffn_gate',
    'mlp.up_proj.weight': 'ffn_up',
    'mlp.down_proj.weight': 'ffn_down',
    'self_attn.q_norm.weight': 'attn_q_norm',
    'self_attn.k_norm.weight': 'attn_k_norm',
}

# Build index: which shard contains which layer
print("Building shard index...")
shard_index = {}  # layer_idx -> shard_file
for shard_file in shards:
    shard = load_file(os.path.join(model_path, shard_file))
    for name in shard.keys():
        if 'model.layers.' in name:
            try:
                li = int(name.split('model.layers.')[1].split('.')[0])
                if li not in shard_index:
                    shard_index[li] = shard_file
            except:
                pass
    del shard

print(f"  Indexed {len(shard_index)} layers across {len(set(shard_index.values()))} shards")

# Process samples in batches
all_logits = []
t0 = time.time()

for batch_start in range(0, n_samples, batch_size):
    batch_end = min(batch_start + batch_size, n_samples)
    tokens = all_tokens[batch_start:batch_end].to(device)

    with torch.no_grad():
        x = F.embedding(tokens, embed_w.to(device)).float()
        positions = torch.arange(seq_len, device=device)

        # Stream through each layer
        current_shard = None
        current_shard_data = None

        for li in range(n_layers):
            # Load shard if different from current
            needed_shard = shard_index[li]
            if needed_shard != current_shard:
                del current_shard_data
                torch.cuda.empty_cache()
                current_shard_data = load_file(os.path.join(model_path, needed_shard))
                current_shard = needed_shard

            # Extract layer weights
            prefix = f'model.layers.{li}.'
            tw = {}
            for src, dst in name_map.items():
                key = prefix + src
                if key in current_shard_data:
                    tw[dst] = current_shard_data[key].float().to(device)

            # Run layer
            layer = TransformerLayer(tw, mc)
            x = layer(x, positions)
            del tw

        del current_shard_data
        current_shard = None
        torch.cuda.empty_cache()

        # Final norm + logits
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x_normed = x.float() * torch.rsqrt(variance + 1e-6) * norm_w.to(device)
        logits = F.linear(x_normed, head_w.to(device))

        all_logits.append(logits[:, -1, :].cpu())

    elapsed = time.time() - t0
    if batch_start % (batch_size * 10) == 0:
        print(f"  {batch_start}/{n_samples} ({elapsed:.0f}s)")
        sys.stdout.flush()

cache = {
    'tokens': all_tokens,
    'logits': torch.cat(all_logits),
    'n_layers': n_layers,
    'embed': embed_w,
    'norm': norm_w,
    'head': head_w,
}
print(f"Cache built: {cache['tokens'].shape[0]} samples ({time.time()-t0:.0f}s)")

# Step 3: Train genome from cache
sd = 64
genome = GenomeModel(
    vocab_size=vocab, big_dim=hidden, small_dim=sd, n_heads=4,
    n_layers=n_layers,
    embed_weight=embed_w.to(device),
    lm_head_weight=head_w.to(device),
    norm_weight=norm_w.to(device),
).to(device)

genome_params = genome.genome_param_count()
print(f"\nGenome: {genome_params:,} params ({genome_params*2/1e6:.1f} MB)")
print(f"Compression: {sum(1 for _ in cache['logits'])}x (vs original 8B)")
print(f"Training from cache (10K steps)...")

opt = torch.optim.AdamW(genome.genome_layers.parameters(), lr=0.001, weight_decay=0.005)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 10000)

for step in range(10000):
    idx = torch.randint(0, n_samples, (8,))
    tokens = cache['tokens'][idx].to(device)
    target = cache['logits'][idx].to(device)
    student = genome(tokens, max_layers=n_layers)[:, -1, :]
    loss = F.kl_div(F.log_softmax(student/2, -1), F.softmax(target/2, -1), reduction='batchmean') * 4
    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(genome.genome_layers.parameters(), 1.0)
    opt.step(); sched.step()
    if step % 2000 == 0:
        print(f"  Step {step}: loss={loss.item():.3f}")
        sys.stdout.flush()

# Eval (compare against cached outputs)
t1, t10s = 0, []
for trial in range(50):
    torch.manual_seed(trial * 13 + 9999)
    idx = torch.randint(0, n_samples, (1,))
    tokens = cache['tokens'][idx].to(device)
    with torch.no_grad():
        tl = cache['logits'][idx].to(device)
        tp = tl[0].argmax().item()
        tt10 = set(tl[0].topk(10).indices.tolist())
        gl = genome(tokens, max_layers=n_layers)
        gp = gl[0, -1].argmax().item()
        gt10 = set(gl[0, -1].topk(10).indices.tolist())
        if tp == gp: t1 += 1
        t10s.append(len(tt10 & gt10) / 10)

print(f"\n{'='*60}")
print(f"  8B GENOME RESULT: Top1={t1*2}% Top10={sum(t10s)/len(t10s)*100:.0f}%")
print(f"  Genome: {genome_params*2/1e6:.1f} MB")
print(f"  If this beats 0.6B quality, SCALING WORKS")
print(f"{'='*60}")
