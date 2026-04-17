"""MOONSHOT: Genomic Weight Expression — A tiny MLP generates ALL weights.

Like DNA encoding an organism. The genome network takes coordinates
(layer, function_type) and generates the transformation matrices on the fly.

Simplified for speed: instead of block-by-block generation, use
a single forward pass per layer that generates a low-rank transform.
The genome outputs (down_proj, up_proj) pair for each layer+function.
"""
import torch, sys, os, time, json, math, gc, traceback
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultracompress.inference import ModelConfig, MiniTransformer

device = 'cuda'
print("Loading teacher...")
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

all_results = {}


def eval_model(forward_fn, n=100):
    t1, t10s = 0, []
    for trial in range(n):
        torch.manual_seed(trial * 13 + 9999)
        t = torch.randint(100, 50000, (1, 16), device=device)
        with torch.no_grad():
            tl = teacher.forward(t, max_layers=28)
            tp = tl[0, -1].argmax().item()
            tt10 = set(tl[0, -1].topk(10).indices.tolist())
            gl = forward_fn(t)
            gp = gl[0, -1].argmax().item()
            gt10 = set(gl[0, -1].topk(10).indices.tolist())
            if tp == gp: t1 += 1
            t10s.append(len(tt10 & gt10) / 10)
    return t1/n, sum(t10s)/len(t10s)


class GenomeNet(nn.Module):
    """The DNA — generates low-rank transforms from continuous coordinates."""
    def __init__(self, hidden_dim=1024, genome_hidden=256, rank=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rank = rank

        # Positional encoding
        self.n_freqs = 6
        input_dim = 2 * (2 * self.n_freqs + 1)  # 2 coords (layer_frac, func_type)

        self.net = nn.Sequential(
            nn.Linear(input_dim, genome_hidden),
            nn.SiLU(),
            nn.Linear(genome_hidden, genome_hidden),
            nn.SiLU(),
            nn.Linear(genome_hidden, genome_hidden),
            nn.SiLU(),
            # Output: down_proj (hidden_dim * rank) + up_proj (rank * hidden_dim)
            nn.Linear(genome_hidden, hidden_dim * rank + rank * hidden_dim),
        )
        # Scale output to be small initially
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _encode(self, x):
        enc = [x]
        for freq in range(self.n_freqs):
            f = 2.0 ** freq
            enc.append(torch.sin(f * math.pi * x))
            enc.append(torch.cos(f * math.pi * x))
        return torch.cat(enc, dim=-1)

    def generate_transform(self, layer_frac, func_type):
        """Generate a low-rank (down, up) pair for one layer+function."""
        coord = torch.tensor([[layer_frac, func_type]], device=next(self.parameters()).device)
        encoded = self._encode(coord)
        raw = self.net(encoded).squeeze(0)
        # Split into down and up
        split = self.hidden_dim * self.rank
        down = raw[:split].reshape(self.hidden_dim, self.rank)
        up = raw[split:].reshape(self.rank, self.hidden_dim)
        return down * 0.01, up * 0.01  # Scale for stability


class GWEModel(nn.Module):
    """Genomic Weight Expression model — genome generates all transforms."""
    def __init__(self, hidden_dim, n_layers, vocab_size, genome_hidden=256, rank=64,
                 embed_weight=None, lm_head_weight=None, norm_weight=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.genome = GenomeNet(hidden_dim, genome_hidden, rank)

        # Norms per layer (cheap, stored directly)
        self.norms = nn.ModuleList([
            nn.ModuleList([nn.RMSNorm(hidden_dim), nn.RMSNorm(hidden_dim)])
            for _ in range(n_layers)
        ])

        if embed_weight is not None:
            self.embed = nn.Embedding.from_pretrained(embed_weight, freeze=True)
        else:
            self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        if lm_head_weight is not None:
            self.lm_head.weight = nn.Parameter(lm_head_weight, requires_grad=False)
        self.final_norm = nn.RMSNorm(hidden_dim)
        if norm_weight is not None:
            self.final_norm.weight = nn.Parameter(norm_weight, requires_grad=False)

    def forward(self, tokens, max_layers=None):
        x = self.embed(tokens).float()
        n = max_layers or self.n_layers
        for li in range(min(n, self.n_layers)):
            layer_frac = li / max(self.n_layers - 1, 1)

            # Attention-like transform (func_type=0.0)
            h = self.norms[li][0](x)
            down, up = self.genome.generate_transform(layer_frac, 0.0)
            x = x + F.linear(F.silu(F.linear(h, down.T)), up.T)

            # FFN-like transform (func_type=1.0)
            h = self.norms[li][1](x)
            down, up = self.genome.generate_transform(layer_frac, 1.0)
            x = x + F.linear(F.silu(F.linear(h, down.T)), up.T)

        x = self.final_norm(x)
        return self.lm_head(x)

    def genome_params(self):
        return sum(p.numel() for p in self.genome.parameters()) + \
               sum(p.numel() for p in self.norms.parameters())


print("=" * 70)
print("MOONSHOT: GENOMIC WEIGHT EXPRESSION")
print("One tiny genome generates all layer transforms")
print("=" * 70)
sys.stdout.flush()

for genome_hidden, rank, n_steps, name in [
    (256, 64, 15000, "GWE-h256-r64"),
    (512, 64, 15000, "GWE-h512-r64"),
    (256, 128, 15000, "GWE-h256-r128"),
]:
    print(f"\n--- {name} ---")
    sys.stdout.flush()
    t0 = time.time()

    try:
        model = GWEModel(
            1024, 28, 151936, genome_hidden=genome_hidden, rank=rank,
            embed_weight=embed, lm_head_weight=lm_head_w, norm_weight=norm_w,
        ).to(device)

        g_params = model.genome_params()
        teacher_params = sum(v.numel() for k, v in gd.items() if k.startswith('blk.'))
        compression = teacher_params / g_params
        print(f"  Genome params: {g_params:,} ({g_params*2/1e6:.1f} MB) = {compression:.0f}x compression")
        sys.stdout.flush()

        trainable = list(model.genome.parameters()) + list(model.norms.parameters())
        opt = torch.optim.AdamW(trainable, lr=0.0003, weight_decay=0.01)
        warmup = 500

        for step in range(n_steps):
            if step < warmup:
                lr = 0.0003 * step / warmup
            else:
                lr = 0.0003 * 0.5 * (1 + math.cos((step - warmup) / (n_steps - warmup) * math.pi))
            for pg in opt.param_groups: pg['lr'] = lr

            tokens = torch.randint(100, 100000, (8, 32), device=device)
            with torch.no_grad():
                teacher_logits = teacher.forward(tokens, max_layers=28)
            student_logits = model(tokens, max_layers=28)

            B, T, V = student_logits.shape
            loss = F.kl_div(
                F.log_softmax(student_logits.reshape(-1, V) / 2, -1),
                F.softmax(teacher_logits.reshape(-1, V) / 2, -1),
                reduction='batchmean') * 4

            if torch.isnan(loss):
                for pg in opt.param_groups: pg['lr'] *= 0.1
                continue

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()

            if step % 3000 == 0:
                t1_e, t10_e = eval_model(lambda t, _m=model: _m(t, max_layers=28))
                print(f"    Step {step}: loss={loss.item():.4f} Top1={t1_e*100:.0f}% Top10={t10_e*100:.0f}% ({time.time()-t0:.0f}s)")
                sys.stdout.flush()

        t1, t10 = eval_model(lambda t, _m=model: _m(t, max_layers=28))
        elapsed = time.time() - t0
        size_mb = g_params * 2 / 1e6
        print(f"  RESULT {name}: Top1={t1*100:.0f}% Top10={t10*100:.0f}% "
              f"Size={size_mb:.1f}MB {compression:.0f}x Time={elapsed:.0f}s")
        all_results[name] = {'top1': t1, 'top10': t10, 'size_mb': size_mb,
                            'compression': compression, 'time': elapsed}
        sys.stdout.flush()

    except Exception as e:
        traceback.print_exc()
        print(f"  FAILED: {e}")
    finally:
        if 'model' in dir(): del model
        torch.cuda.empty_cache(); gc.collect()

print(f"\n{'='*70}")
print("GWE RESULTS")
print(f"{'='*70}")
for n, r in sorted(all_results.items(), key=lambda x: x[1]['top10'], reverse=True):
    print(f"  {n:<20} Top1={r['top1']*100:.0f}% Top10={r['top10']*100:.0f}% {r['size_mb']:.1f}MB {r['compression']:.0f}x")
with open('gwe_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"{'='*70}")
