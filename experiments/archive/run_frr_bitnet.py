"""FRR + BitNet: Ternary (1.58-bit) Fractal Residual Recursion from scratch.

THE IDEA: Combine FRR's weight sharing (42x compression from shared block)
with BitNet's ternary quantization (16/1.58 = 10.1x from weight precision).
Theoretical max: 42x * 10.1x = ~425x compression.

BitNet approach (Microsoft):
  - Weights are ternary {-1, 0, 1}, stored at 1.58 bits each
  - Forward: quantize latent weights to ternary via sign_ternary()
  - Backward: straight-through estimator (STE) — gradients flow through as if continuous
  - Update: optimizer updates continuous "latent" weights, re-quantize on next forward

sign_ternary(w):
  -1 if w < -0.5
   0 if |w| < 0.5
   1 if w > 0.5

We train from scratch via KL distillation from Qwen3-0.6B teacher.
NOT compressing existing weights — learning a ternary FRR from the ground up.
"""
import torch, sys, os, time, json, math, gc, traceback
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultracompress.inference import ModelConfig, MiniTransformer

device = 'cuda'
print("Loading teacher (Qwen3-0.6B)...")
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


# ================================================================
# TERNARY QUANTIZATION WITH STE
# ================================================================

class TernaryQuantize(torch.autograd.Function):
    """Ternary quantization with straight-through estimator.

    Forward: w -> {-1, 0, 1} based on thresholds
    Backward: gradient passes through unchanged (STE)
    """
    @staticmethod
    def forward(ctx, w):
        # sign_ternary: -1 if w<-0.5, 0 if |w|<0.5, 1 if w>0.5
        out = torch.zeros_like(w)
        out[w > 0.5] = 1.0
        out[w < -0.5] = -1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # STE: pass gradient through unchanged
        return grad_output


def ternary_quantize(w):
    return TernaryQuantize.apply(w)


class TernaryLinear(nn.Module):
    """Linear layer with ternary weights.

    Stores continuous "latent" weights that get quantized to {-1, 0, 1}
    on each forward pass. Includes a learned per-output-channel scale factor
    (as in BitNet 1.58b) to recover expressiveness.
    """
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Latent (continuous) weights — these are what the optimizer updates
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        # Per-output-channel scale: compensates for ternary's limited range
        self.scale = nn.Parameter(torch.ones(out_features) * 0.1)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    def forward(self, x):
        # Quantize to ternary (STE for backward)
        w_ternary = ternary_quantize(self.weight)
        # Scale per output channel
        w_scaled = w_ternary * self.scale.unsqueeze(1)
        return F.linear(x, w_scaled, self.bias)


class TernaryFractalBlock(nn.Module):
    """FractalBlock with ternary weights throughout.

    Same structure as FractalBlock (attention + SwiGLU FFN) but every
    weight matrix uses TernaryLinear for 1.58-bit storage.
    """
    def __init__(self, hidden_dim, n_heads=8, ff_mult=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        # Shared attention — all ternary
        self.qkv = TernaryLinear(hidden_dim, 3 * hidden_dim)
        self.o_proj = TernaryLinear(hidden_dim, hidden_dim)

        # Shared FFN (SwiGLU) — all ternary
        ff_dim = hidden_dim * ff_mult
        self.gate = TernaryLinear(hidden_dim, ff_dim)
        self.up = TernaryLinear(hidden_dim, ff_dim)
        self.down = TernaryLinear(ff_dim, hidden_dim)

        # Norms (kept full precision — tiny and critical for stability)
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.norm2 = nn.RMSNorm(hidden_dim)

    def forward(self, x, scale_gamma=None, scale_beta=None):
        B, T, D = x.shape

        # Scale-conditional modulation
        h = self.norm1(x)
        if scale_gamma is not None:
            h = h * scale_gamma + (scale_beta if scale_beta is not None else 0)

        # Attention
        qkv = self.qkv(h).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if T > 1:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.o_proj(out)

        # FFN with modulation
        h = self.norm2(x)
        if scale_gamma is not None:
            h = h * scale_gamma + (scale_beta if scale_beta is not None else 0)
        x = x + self.down(F.silu(self.gate(h)) * self.up(h))

        return x


class TernaryFractalModel(nn.Module):
    """Fractal Residual Recursion with ternary (1.58-bit) weights.

    Architecture: one shared TernaryFractalBlock applied recursively.
    Weights: ternary {-1, 0, 1} with per-channel scales.
    Training: STE allows gradient-based optimization of latent weights.

    Storage cost: fractal_params * 1.58 bits (ternary encoding)
    + per-channel scales in fp16 (negligible overhead)
    """
    def __init__(self, hidden_dim, n_heads, n_scales=4, iters_per_scale=7,
                 vocab_size=151936, ff_mult=2,
                 embed_weight=None, lm_head_weight=None, norm_weight=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_scales = n_scales
        self.iters_per_scale = iters_per_scale
        self.total_layers = n_scales * iters_per_scale

        # Shared TERNARY block (reused everywhere)
        self.block = TernaryFractalBlock(hidden_dim, n_heads, ff_mult)

        # Per-scale modulation (full precision — tiny)
        self.scale_gamma = nn.Parameter(torch.ones(n_scales, hidden_dim))
        self.scale_beta = nn.Parameter(torch.zeros(n_scales, hidden_dim))

        # Per-iteration scaling within each scale
        self.iter_scale = nn.Parameter(torch.ones(n_scales, iters_per_scale))

        # Embedding and head (shared from teacher, frozen)
        if embed_weight is not None:
            self.embed = nn.Embedding.from_pretrained(embed_weight, freeze=True)
        else:
            self.embed = nn.Embedding(vocab_size, hidden_dim)

        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        if lm_head_weight is not None:
            self.lm_head.weight = nn.Parameter(lm_head_weight, requires_grad=False)

        self.norm = nn.RMSNorm(hidden_dim)
        if norm_weight is not None:
            self.norm.weight = nn.Parameter(norm_weight, requires_grad=False)

    def forward(self, tokens, max_layers=None):
        x = self.embed(tokens).float()
        total = max_layers or self.total_layers

        layer_count = 0
        for scale in range(self.n_scales):
            gamma = self.scale_gamma[scale]
            beta = self.scale_beta[scale]
            for it in range(self.iters_per_scale):
                if layer_count >= total:
                    break
                iter_s = self.iter_scale[scale, it]
                x = x + (self.block(x, gamma, beta) - x) * iter_s
                layer_count += 1

        x = self.norm(x)
        return self.lm_head(x)

    def ternary_params(self):
        """Count ternary block params (what gets stored at 1.58 bits)."""
        # Count weight elements in TernaryLinear layers (the ternary part)
        ternary_weight_count = 0
        scale_count = 0
        for m in self.block.modules():
            if isinstance(m, TernaryLinear):
                ternary_weight_count += m.weight.numel()
                scale_count += m.scale.numel()
        return ternary_weight_count, scale_count

    def fractal_params(self):
        """Total trainable fractal params (block + modulation)."""
        total = sum(p.numel() for p in self.block.parameters())
        total += self.scale_gamma.numel() + self.scale_beta.numel()
        total += self.iter_scale.numel()
        return total

    def effective_size_bits(self):
        """Effective storage in bits.

        Ternary weights: count * 1.58 bits
        Scales (fp16): count * 16 bits
        Modulation (fp16): count * 16 bits
        Norms (fp16): count * 16 bits
        """
        ternary_count, scale_count = self.ternary_params()
        # Norm params in the block
        norm_count = sum(p.numel() for m in self.block.modules()
                         if isinstance(m, nn.RMSNorm) for p in m.parameters())
        # Modulation params
        mod_count = (self.scale_gamma.numel() + self.scale_beta.numel() +
                     self.iter_scale.numel())

        bits = (ternary_count * 1.58 +    # ternary weights
                scale_count * 16 +          # per-channel scales in fp16
                norm_count * 16 +           # norms in fp16
                mod_count * 16)             # modulation in fp16
        return bits

    def ternary_sparsity(self):
        """What fraction of quantized weights are zero."""
        total, zeros = 0, 0
        for m in self.block.modules():
            if isinstance(m, TernaryLinear):
                w_q = ternary_quantize(m.weight)
                total += w_q.numel()
                zeros += (w_q == 0).sum().item()
        return zeros / total if total > 0 else 0


# ================================================================
# EVAL
# ================================================================

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


# ================================================================
# MAIN
# ================================================================

all_results = {}
pipeline_start = time.time()
teacher_layer_params = sum(v.numel() for k, v in gd.items() if k.startswith('blk.'))

print("=" * 70)
print("FRR + BitNet: TERNARY FRACTAL RESIDUAL RECURSION")
print("Shared block (42x) + ternary weights (10.1x) = theoretical 425x")
print("Training from scratch via KL distillation from Qwen3-0.6B")
print("=" * 70)
sys.stdout.flush()

# Two configs:
#   hidden=1024: match Qwen3-0.6B hidden dim, maximize quality
#   hidden=512:  smaller model, test if ternary FRR scales down
configs = [
    # (hidden_dim, n_heads, n_scales, iters_per_scale, ff_mult, n_steps, name)
    (1024, 8, 4, 7, 2, 10000, "BitFRR-1024"),
    (512,  8, 4, 7, 2, 10000, "BitFRR-512"),
]

for hidden_dim, n_heads, n_scales, iters, ff_mult, n_steps, name in configs:
    eff_layers = n_scales * iters
    print(f"\n--- {name} (hidden={hidden_dim}, {n_scales}x{iters}={eff_layers} layers) ---")
    sys.stdout.flush()
    t0 = time.time()

    try:
        # For hidden=512, we need a projection from embed (1024) to hidden (512)
        needs_proj = (hidden_dim != 1024)

        model = TernaryFractalModel(
            hidden_dim=hidden_dim, n_heads=n_heads,
            n_scales=n_scales, iters_per_scale=iters,
            vocab_size=151936, ff_mult=ff_mult,
            embed_weight=embed if not needs_proj else None,
            lm_head_weight=lm_head_w if not needs_proj else None,
            norm_weight=norm_w if not needs_proj else None,
        ).to(device)

        # For smaller hidden, add trainable projections
        proj_in = None
        proj_out = None
        if needs_proj:
            proj_in = nn.Linear(1024, hidden_dim, bias=False).to(device)
            proj_out = nn.Linear(hidden_dim, 151936, bias=False).to(device)
            # Initialize proj_out from teacher lm_head via truncation
            with torch.no_grad():
                # SVD-based init: approximate lm_head_w (151936x1024) projected down
                # Simple approach: just use random init, let training handle it
                nn.init.xavier_normal_(proj_in.weight)
                nn.init.xavier_normal_(proj_out.weight)

        def forward_fn_train(tokens):
            if needs_proj:
                x = F.embedding(tokens, embed).float()
                x = proj_in(x)
                # Run through fractal block manually
                total = model.total_layers
                layer_count = 0
                for scale in range(model.n_scales):
                    gamma = model.scale_gamma[scale]
                    beta = model.scale_beta[scale]
                    for it in range(model.iters_per_scale):
                        if layer_count >= total:
                            break
                        iter_s = model.iter_scale[scale, it]
                        x = x + (model.block(x, gamma, beta) - x) * iter_s
                        layer_count += 1
                x = nn.RMSNorm(hidden_dim, device=device)(x) if not hasattr(model, '_proj_norm') else model._proj_norm(x)
                return proj_out(x)
            else:
                return model(tokens)

        # For the projection case, add a norm
        if needs_proj:
            model._proj_norm = nn.RMSNorm(hidden_dim).to(device)

        # Count params and compute effective size
        fractal_p = model.fractal_params()
        ternary_count, scale_count = model.ternary_params()
        effective_bits = model.effective_size_bits()
        effective_mb = effective_bits / 8 / 1e6

        if needs_proj:
            proj_params = proj_in.weight.numel() + proj_out.weight.numel()
            # Projections stored in fp16
            effective_bits += proj_params * 16
            effective_mb = effective_bits / 8 / 1e6
            fractal_p += proj_params
            print(f"  Projection params: {proj_params:,}")

        print(f"  Fractal params (latent fp32): {fractal_p:,}")
        print(f"  Ternary weights: {ternary_count:,} @ 1.58 bits = {ternary_count*1.58/8/1e6:.2f} MB")
        print(f"  Per-channel scales: {scale_count:,} @ 16 bits = {scale_count*16/8/1e6:.4f} MB")
        print(f"  Effective storage: {effective_mb:.2f} MB ({effective_bits/1e6:.1f} Mbits)")
        print(f"  Teacher layers: {teacher_layer_params:,} ({teacher_layer_params*2/1e6:.1f} MB fp16)")
        teacher_bits = teacher_layer_params * 16
        compression = teacher_bits / effective_bits
        print(f"  Compression vs teacher: {compression:.0f}x")
        sys.stdout.flush()

        # Collect trainable params
        trainable = list(model.block.parameters()) + [
            model.scale_gamma, model.scale_beta, model.iter_scale
        ]
        if needs_proj:
            trainable += list(proj_in.parameters()) + list(proj_out.parameters())
            trainable += list(model._proj_norm.parameters())

        opt = torch.optim.AdamW(trainable, lr=0.001, weight_decay=0.01)
        warmup = 1000

        for step in range(n_steps):
            # Cosine schedule with warmup
            if step < warmup:
                lr = 0.001 * step / warmup
            else:
                lr = 0.001 * 0.5 * (1 + math.cos((step - warmup) / (n_steps - warmup) * math.pi))
            for pg in opt.param_groups: pg['lr'] = lr

            tokens = torch.randint(100, 100000, (8, 32), device=device)
            with torch.no_grad():
                teacher_logits = teacher.forward(tokens, max_layers=28)

            student_logits = forward_fn_train(tokens)

            # KL divergence with temperature
            B, T, V = student_logits.shape
            temp = 2.0
            loss = F.kl_div(
                F.log_softmax(student_logits.reshape(-1, V) / temp, -1),
                F.softmax(teacher_logits.reshape(-1, V) / temp, -1),
                reduction='batchmean') * (temp ** 2)

            if torch.isnan(loss):
                print(f"    NaN at step {step}, reducing LR...")
                for pg in opt.param_groups: pg['lr'] *= 0.1
                continue

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()

            if step % 2000 == 0 or step == n_steps - 1:
                sparsity = model.ternary_sparsity()
                if needs_proj:
                    t1_e, t10_e = eval_model(forward_fn_train)
                else:
                    t1_e, t10_e = eval_model(lambda t, _m=model: _m(t))
                print(f"    Step {step:>5d}: loss={loss.item():.4f} "
                      f"Top1={t1_e*100:.0f}% Top10={t10_e*100:.0f}% "
                      f"sparsity={sparsity*100:.1f}% lr={lr:.6f} ({time.time()-t0:.0f}s)")
                sys.stdout.flush()

        # Final eval
        if needs_proj:
            t1, t10 = eval_model(forward_fn_train)
        else:
            t1, t10 = eval_model(lambda t, _m=model: _m(t))
        elapsed = time.time() - t0
        sparsity = model.ternary_sparsity()

        print(f"\n  RESULT {name}:")
        print(f"    Quality: Top1={t1*100:.0f}% Top10={t10*100:.0f}%")
        print(f"    Ternary sparsity: {sparsity*100:.1f}% zeros")
        print(f"    Effective size: {effective_mb:.2f} MB")
        print(f"    Compression: {compression:.0f}x vs teacher (fp16)")
        print(f"    Time: {elapsed:.0f}s")

        all_results[name] = {
            'top1': t1, 'top10': t10,
            'effective_mb': effective_mb,
            'compression': compression,
            'ternary_weights': ternary_count,
            'sparsity': sparsity,
            'fractal_params': fractal_p,
            'hidden_dim': hidden_dim,
            'eff_layers': eff_layers,
            'time': elapsed,
        }

    except Exception as e:
        traceback.print_exc()
        print(f"  FAILED: {e}")
    finally:
        for v in ['model', 'proj_in', 'proj_out']:
            if v in dir(): exec(f'del {v}', {}, locals())
        torch.cuda.empty_cache(); gc.collect()
    sys.stdout.flush()


# ================================================================
# LEADERBOARD
# ================================================================

total_time = time.time() - pipeline_start
print(f"\n{'='*70}")
print(f"FRR + BitNet TERNARY RESULTS (Total: {total_time/60:.0f} min)")
print(f"{'='*70}")
print(f"Comparison baselines:")
print(f"  FRR fp32:  ~50-60% top-10 at ~23 MB (42x)")
print(f"  Genome:    63% top-10 at 23.9 MB (37x)")
print(f"  Theory:    FRR 42x * ternary 10.1x = 425x")
print()

sorted_all = sorted(all_results.items(), key=lambda x: x[1]['top10'], reverse=True)
for i, (n, r) in enumerate(sorted_all):
    print(f"  {n:<20} Top1={r['top1']*100:>4.0f}% Top10={r['top10']*100:>4.0f}% "
          f"Size={r['effective_mb']:>6.2f}MB {r['compression']:>5.0f}x "
          f"sparsity={r['sparsity']*100:.0f}%")

with open('frr_bitnet_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

if sorted_all:
    best_n, best = sorted_all[0]
    print(f"\nBest: {best_n} at {best['top10']*100:.0f}% top-10, "
          f"{best['effective_mb']:.2f} MB ({best['compression']:.0f}x)")
    if best['compression'] > 200 and best['top10'] > 0.30:
        print(">>> TERNARY FRR WORKS! 200x+ compression with meaningful quality.")
        print(">>> This is a new compression frontier. Scale up training.")
    elif best['top10'] > 0.20:
        print(">>> Promising. Ternary FRR learns something. Needs more steps/tuning.")
    else:
        print(">>> Ternary weights too restrictive for FRR shared block.")
        print(">>> Try: mixed precision (ternary FFN + 4-bit attention), more scales.")
print(f"{'='*70}")
