"""PARADIGM BREAKERS — Three approaches that reframe compression entirely.

Not "make weights smaller." Make weights UNNECESSARY.

1. PROGRAM SYNTHESIS: Extract model behavior as compact executable rules
2. THE SEED: Input-conditional computation generator (different input = different virtual model)
3. MICRO-SWARM: Domain-specialist micro-models + router

All work on existing models. No retraining the original.
All tested on Qwen3-0.6B with standard eval.
"""
import torch, sys, os, time, json, gc, math, traceback
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultracompress.inference import ModelConfig, MiniTransformer

device = 'cuda'
print("Loading Qwen3-0.6B teacher...")
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
positions = torch.arange(32, device=device)

all_results = {}
pipeline_start = time.time()


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
# PARADIGM 1: THE SEED
# Input-conditional computation generator.
# One tiny network that generates DIFFERENT transformations per input.
# Key insight: each input only uses a fraction of the model's knowledge.
# The seed generates only what's needed, on the fly.
# ================================================================
print("=" * 70)
print("PARADIGM 1: THE SEED")
print("Input-conditional computation — different input = different virtual model")
print("=" * 70)
sys.stdout.flush()


class SeedLayer(nn.Module):
    """A seed-based layer that generates its own transformation per input.

    Instead of fixed weights W, the transformation is:
      output = input + Generated_Transform(input)

    Where Generated_Transform is created on-the-fly by a tiny "genome"
    conditioned on the input content.

    Key difference from genome V1: the transformation is INPUT-DEPENDENT.
    V1 used the same fixed micro-transformer for all inputs.
    The seed adapts its computation to what the input needs.
    """
    def __init__(self, hidden_dim, seed_dim=64, n_programs=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.seed_dim = seed_dim
        self.n_programs = n_programs

        # Input analyzer: reads the input and decides what computation to generate
        self.analyzer = nn.Sequential(
            nn.Linear(hidden_dim, seed_dim),
            nn.SiLU(),
            nn.Linear(seed_dim, n_programs),  # soft routing over programs
        )

        # Program bank: each "program" is a compact transformation
        # Stored as low-rank: down (hidden->seed) + up (seed->hidden)
        self.program_down = nn.Parameter(torch.randn(n_programs, hidden_dim, seed_dim) * 0.01)
        self.program_up = nn.Parameter(torch.zeros(n_programs, seed_dim, hidden_dim))

        # Nonlinear core per program
        self.program_gate = nn.Parameter(torch.randn(n_programs, hidden_dim, seed_dim) * 0.01)

        self.scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, x):
        """x: (B, T, hidden_dim) -> (B, T, hidden_dim)"""
        B, T, D = x.shape

        # Analyze input to get program routing weights
        # Use mean-pooled representation for efficiency
        x_mean = x.mean(dim=1)  # (B, D)
        # Use float32 for routing to avoid numerical instability
        routing = F.softmax(self.analyzer(x_mean.float()), dim=-1)  # (B, n_programs)
        routing = torch.clamp(routing, min=1e-6, max=1.0)

        # Generate transformation by mixing programs
        # Weighted sum of program weights
        # down: (B, D, seed_dim) = sum over programs of routing * program_down
        down_w = torch.einsum('bp,pds->bds', routing, self.program_down)  # (B, D, S)
        up_w = torch.einsum('bp,psd->bsd', routing, self.program_up)  # (B, S, D)
        gate_w = torch.einsum('bp,pds->bds', routing, self.program_gate)  # (B, D, S)

        # Normalize generated weight matrices to prevent explosion
        down_w = F.normalize(down_w, dim=1) * (D ** 0.5)
        up_w = F.normalize(up_w, dim=1) * (self.seed_dim ** 0.5)
        gate_w = F.normalize(gate_w, dim=1) * (D ** 0.5)

        # Apply generated transformation
        # x: (B, T, D), down_w: (B, D, S) -> projected: (B, T, S)
        projected = torch.bmm(x, down_w)  # (B, T, S)
        gated = torch.bmm(x, gate_w)  # (B, T, S)
        hidden = F.silu(gated) * projected  # (B, T, S)
        output = torch.bmm(hidden, up_w)  # (B, T, D)

        return x + output * self.scale


class SeedModel(nn.Module):
    """Complete model using seed-based layers."""
    def __init__(self, n_layers, hidden_dim, vocab_size, seed_dim=64, n_programs=16,
                 embed_weight=None, lm_head_weight=None, norm_weight=None):
        super().__init__()
        self.n_layers = n_layers

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

        self.layers = nn.ModuleList([
            SeedLayer(hidden_dim, seed_dim, n_programs)
            for _ in range(n_layers)
        ])

    def forward(self, tokens, max_layers=None):
        x = self.embed(tokens).float()
        n = max_layers or self.n_layers
        for i in range(min(n, len(self.layers))):
            x = self.layers[i](x)
        x = self.norm(x)
        return self.lm_head(x)

    def seed_params(self):
        return sum(p.numel() for p in self.layers.parameters())


for seed_dim, n_programs, n_steps, name in [
    (32, 8, 10000, "Seed-s32-p8"),
    (64, 16, 10000, "Seed-s64-p16"),
    (128, 32, 15000, "Seed-s128-p32"),
]:
    print(f"\n--- {name} ---")
    sys.stdout.flush()
    t0 = time.time()

    try:
        model = SeedModel(
            28, 1024, 151936, seed_dim=seed_dim, n_programs=n_programs,
            embed_weight=embed, lm_head_weight=lm_head_w, norm_weight=norm_w,
        ).to(device)

        params = model.seed_params()
        print(f"  Seed params: {params:,} ({params*2/1e6:.1f} MB)")

        opt = torch.optim.AdamW(model.layers.parameters(), lr=0.0003, weight_decay=0.01)
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

            # All-position KL loss
            B, T, V = student_logits.shape
            loss = F.kl_div(
                F.log_softmax(student_logits.reshape(-1, V) / 2, -1),
                F.softmax(teacher_logits.reshape(-1, V) / 2, -1),
                reduction='batchmean') * 4

            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.layers.parameters(), 1.0)
            opt.step()

            if step % 2000 == 0:
                t1_e, t10_e = eval_model(lambda t, _m=model: _m(t, max_layers=28))
                print(f"    Step {step}: loss={loss.item():.4f} Top1={t1_e*100:.0f}% Top10={t10_e*100:.0f}% ({time.time()-t0:.0f}s)")
                sys.stdout.flush()

        t1, t10 = eval_model(lambda t, _m=model: _m(t, max_layers=28))
        elapsed = time.time() - t0
        size_mb = params * 2 / 1e6
        print(f"  RESULT {name}: Top1={t1*100:.0f}% Top10={t10*100:.0f}% Size={size_mb:.1f}MB Time={elapsed:.0f}s")
        all_results[name] = {'top1': t1, 'top10': t10, 'size_mb': size_mb, 'params': params, 'time': elapsed, 'approach': 'seed'}

    except Exception as e:
        traceback.print_exc()
        print(f"  FAILED: {e}")
    finally:
        if 'model' in dir(): del model
        torch.cuda.empty_cache(); gc.collect()
    sys.stdout.flush()


# ================================================================
# PARADIGM 2: MICRO-INTELLIGENCE SWARM
# Multiple tiny specialists + router. Each expert handles a domain.
# Collectively they match the big model.
# ================================================================
print(f"\n{'='*70}")
print("PARADIGM 2: MICRO-INTELLIGENCE SWARM")
print("Domain specialists + router = collective intelligence")
print(f"{'='*70}")
sys.stdout.flush()


class SwarmExpert(nn.Module):
    """A tiny specialist model. Handles one domain of inputs."""
    def __init__(self, hidden_dim, small_dim, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, small_dim, bias=False),
                nn.SiLU(),
                nn.Linear(small_dim, hidden_dim, bias=False),
            )
            for _ in range(n_layers)
        ])
        # Init near identity
        for layer in self.layers:
            nn.init.zeros_(layer[2].weight)

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x) * 0.1
        return x


class SwarmModel(nn.Module):
    """Multiple specialists + a router."""
    def __init__(self, n_experts, hidden_dim, small_dim, n_expert_layers, n_model_layers,
                 embed_weight=None, lm_head_weight=None, norm_weight=None, vocab_size=151936):
        super().__init__()
        self.n_experts = n_experts
        self.n_model_layers = n_model_layers

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

        # Router: assigns input to experts
        self.router = nn.Linear(hidden_dim, n_experts, bias=False)

        # Experts: each is a tiny model
        self.experts = nn.ModuleList([
            SwarmExpert(hidden_dim, small_dim, n_expert_layers)
            for _ in range(n_experts)
        ])

        # Shared backbone (cheap, handles universal patterns)
        self.backbone = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, small_dim, bias=False),
                nn.SiLU(),
                nn.Linear(small_dim, hidden_dim, bias=False),
            )
            for _ in range(n_model_layers)
        ])
        for layer in self.backbone:
            nn.init.zeros_(layer[2].weight)

    def forward(self, tokens, max_layers=None):
        x = self.embed(tokens).float()

        # Route based on input content
        x_mean = x.mean(dim=1)  # (B, D)
        routing = F.softmax(self.router(x_mean) * 2, dim=-1)  # (B, n_experts)
        top2 = routing.topk(2, dim=-1)
        top2_weights = F.softmax(top2.values, dim=-1)  # (B, 2)
        top2_indices = top2.indices  # (B, 2)

        # Apply backbone (universal computation)
        n = max_layers or self.n_model_layers
        for i in range(min(n, len(self.backbone))):
            x = x + self.backbone[i](x) * 0.1

        # Apply top-2 experts (specialized computation)
        expert_output = torch.zeros_like(x)
        for k in range(2):
            for e in range(self.n_experts):
                mask = (top2_indices[:, k] == e)
                if mask.any():
                    x_e = x[mask]
                    out_e = self.experts[e](x_e)
                    weight = top2_weights[mask, k:k+1].unsqueeze(-1)  # (n, 1, 1)
                    expert_output[mask] += out_e * weight

        x = x + expert_output

        x = self.norm(x)
        return self.lm_head(x)

    def swarm_params(self):
        return sum(p.numel() for p in self.backbone.parameters()) + \
               sum(p.numel() for p in self.experts.parameters()) + \
               sum(p.numel() for p in self.router.parameters())


for n_experts, small_dim, n_expert_layers, n_backbone_layers, n_steps, name in [
    (8, 64, 4, 28, 10000, "Swarm-8x-s64"),
    (16, 64, 4, 28, 10000, "Swarm-16x-s64"),
    (32, 32, 4, 28, 15000, "Swarm-32x-s32"),
]:
    print(f"\n--- {name} ---")
    sys.stdout.flush()
    t0 = time.time()

    try:
        model = SwarmModel(
            n_experts=n_experts, hidden_dim=1024, small_dim=small_dim,
            n_expert_layers=n_expert_layers, n_model_layers=n_backbone_layers,
            embed_weight=embed, lm_head_weight=lm_head_w, norm_weight=norm_w,
        ).to(device)

        params = model.swarm_params()
        print(f"  Swarm params: {params:,} ({params*2/1e6:.1f} MB) [{n_experts} experts]")

        trainable = list(model.backbone.parameters()) + list(model.experts.parameters()) + list(model.router.parameters())
        opt = torch.optim.AdamW(trainable, lr=0.001, weight_decay=0.01)
        warmup = 500

        for step in range(n_steps):
            if step < warmup:
                lr = 0.001 * step / warmup
            else:
                lr = 0.001 * 0.5 * (1 + math.cos((step - warmup) / (n_steps - warmup) * math.pi))
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

            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()

            if step % 2000 == 0:
                t1_e, t10_e = eval_model(lambda t, _m=model: _m(t, max_layers=28))
                print(f"    Step {step}: loss={loss.item():.4f} Top1={t1_e*100:.0f}% Top10={t10_e*100:.0f}% ({time.time()-t0:.0f}s)")
                sys.stdout.flush()

        t1, t10 = eval_model(lambda t, _m=model: _m(t, max_layers=28))
        elapsed = time.time() - t0
        size_mb = params * 2 / 1e6
        print(f"  RESULT {name}: Top1={t1*100:.0f}% Top10={t10*100:.0f}% Size={size_mb:.1f}MB Time={elapsed:.0f}s")
        all_results[name] = {'top1': t1, 'top10': t10, 'size_mb': size_mb, 'params': params, 'time': elapsed, 'approach': 'swarm'}

    except Exception as e:
        traceback.print_exc()
        print(f"  FAILED: {e}")
    finally:
        if 'model' in dir(): del model
        torch.cuda.empty_cache(); gc.collect()
    sys.stdout.flush()


# ================================================================
# PARADIGM 3: PROGRAM SYNTHESIS
# Extract behavior as a structured lookup + interpolation system.
# The "program" is a hash table of learned embeddings that map
# input patterns to output transformations.
# ================================================================
print(f"\n{'='*70}")
print("PARADIGM 3: PROGRAM SYNTHESIS")
print("Extract behavior as learned pattern lookup + interpolation")
print(f"{'='*70}")
sys.stdout.flush()


class ProgramLayer(nn.Module):
    """A layer that works by pattern matching and interpolation.

    Instead of matrix multiply (W @ x), this does:
    1. Hash the input into a set of learned "pattern keys"
    2. Look up the closest patterns in a codebook
    3. Interpolate the corresponding "program outputs"

    Like a neural hash table: input -> nearest patterns -> weighted output.
    The codebook IS the compressed program.
    """
    def __init__(self, hidden_dim, n_patterns=256, pattern_dim=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_patterns = n_patterns
        self.pattern_dim = pattern_dim

        # Pattern keys: what inputs look like
        self.keys = nn.Parameter(torch.randn(n_patterns, pattern_dim) * 0.1)

        # Pattern values: what transformation to apply
        self.values_down = nn.Parameter(torch.randn(n_patterns, hidden_dim, pattern_dim) * 0.01)
        self.values_up = nn.Parameter(torch.zeros(n_patterns, pattern_dim, hidden_dim))

        # Input hasher: projects input to pattern space
        self.hasher = nn.Linear(hidden_dim, pattern_dim, bias=False)

        self.scale = nn.Parameter(torch.tensor(0.1))
        self.top_k = 4  # Number of patterns to match

    def forward(self, x):
        B, T, D = x.shape

        # Hash input
        h = self.hasher(x)  # (B, T, pattern_dim)

        # Find nearest patterns
        # Similarity: h @ keys^T
        sim = torch.matmul(h, self.keys.T)  # (B, T, n_patterns)
        topk = sim.topk(self.top_k, dim=-1)
        weights = F.softmax(topk.values, dim=-1)  # (B, T, top_k)
        indices = topk.indices  # (B, T, top_k)

        # Look up and interpolate transformations
        output = torch.zeros_like(x)
        for k in range(self.top_k):
            idx = indices[:, :, k]  # (B, T)
            w = weights[:, :, k:k+1]  # (B, T, 1)

            # Gather the value matrices for selected patterns
            # For each token, get its pattern's down and up matrices
            flat_idx = idx.reshape(-1)  # (B*T,)
            v_down = self.values_down[flat_idx]  # (B*T, D, pd)
            v_up = self.values_up[flat_idx]  # (B*T, pd, D)

            x_flat = x.reshape(-1, D).unsqueeze(1)  # (B*T, 1, D)
            proj = torch.bmm(x_flat, v_down)  # (B*T, 1, pd)
            proj = F.silu(proj)
            out = torch.bmm(proj, v_up).squeeze(1)  # (B*T, D)
            out = out.reshape(B, T, D)

            output += out * w

        return x + output * self.scale


class ProgramModel(nn.Module):
    def __init__(self, n_layers, hidden_dim, vocab_size, n_patterns=256, pattern_dim=32,
                 embed_weight=None, lm_head_weight=None, norm_weight=None):
        super().__init__()
        self.n_layers = n_layers

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

        self.layers = nn.ModuleList([
            ProgramLayer(hidden_dim, n_patterns, pattern_dim)
            for _ in range(n_layers)
        ])

    def forward(self, tokens, max_layers=None):
        x = self.embed(tokens).float()
        n = max_layers or self.n_layers
        for i in range(min(n, len(self.layers))):
            x = self.layers[i](x)
        x = self.norm(x)
        return self.lm_head(x)

    def program_params(self):
        return sum(p.numel() for p in self.layers.parameters())


for n_patterns, pattern_dim, n_steps, name in [
    (128, 32, 10000, "Prog-128p"),
    (256, 32, 10000, "Prog-256p"),
    (512, 64, 15000, "Prog-512p"),
]:
    print(f"\n--- {name} ---")
    sys.stdout.flush()
    t0 = time.time()

    try:
        model = ProgramModel(
            28, 1024, 151936, n_patterns=n_patterns, pattern_dim=pattern_dim,
            embed_weight=embed, lm_head_weight=lm_head_w, norm_weight=norm_w,
        ).to(device)

        params = model.program_params()
        print(f"  Program params: {params:,} ({params*2/1e6:.1f} MB)")

        opt = torch.optim.AdamW(model.layers.parameters(), lr=0.001, weight_decay=0.01)
        warmup = 500

        for step in range(n_steps):
            if step < warmup:
                lr = 0.001 * step / warmup
            else:
                lr = 0.001 * 0.5 * (1 + math.cos((step - warmup) / (n_steps - warmup) * math.pi))
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

            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.layers.parameters(), 1.0)
            opt.step()

            if step % 2000 == 0:
                t1_e, t10_e = eval_model(lambda t, _m=model: _m(t, max_layers=28))
                print(f"    Step {step}: loss={loss.item():.4f} Top1={t1_e*100:.0f}% Top10={t10_e*100:.0f}% ({time.time()-t0:.0f}s)")
                sys.stdout.flush()

        t1, t10 = eval_model(lambda t, _m=model: _m(t, max_layers=28))
        elapsed = time.time() - t0
        size_mb = params * 2 / 1e6
        print(f"  RESULT {name}: Top1={t1*100:.0f}% Top10={t10*100:.0f}% Size={size_mb:.1f}MB Time={elapsed:.0f}s")
        all_results[name] = {'top1': t1, 'top10': t10, 'size_mb': size_mb, 'params': params, 'time': elapsed, 'approach': 'program'}

    except Exception as e:
        traceback.print_exc()
        print(f"  FAILED: {e}")
    finally:
        if 'model' in dir(): del model
        torch.cuda.empty_cache(); gc.collect()
    sys.stdout.flush()


# ================================================================
# LEADERBOARD
# ================================================================
total_time = time.time() - pipeline_start
print(f"\n{'='*70}")
print(f"PARADIGM BREAKERS RESULTS (Total: {total_time/60:.0f} min)")
print(f"{'='*70}")
print(f"Previous bests: Genome=63% top-10 (23.9MB), Supertrain-D=63% (23.9MB)")
print()

for approach in ['seed', 'swarm', 'program']:
    results = [(n, r) for n, r in all_results.items() if r.get('approach') == approach]
    if results:
        aname = {'seed': 'THE SEED', 'swarm': 'MICRO-SWARM', 'program': 'PROGRAM SYNTHESIS'}[approach]
        print(f"  {aname}:")
        for n, r in sorted(results, key=lambda x: x[1]['top10'], reverse=True):
            print(f"    {n:<20} Top1={r['top1']*100:>4.0f}% Top10={r['top10']*100:>4.0f}% Size={r['size_mb']:>6.1f}MB")
        print()

sorted_all = sorted(all_results.items(), key=lambda x: x[1]['top10'], reverse=True)
print("  OVERALL CHAMPION:")
for i, (n, r) in enumerate(sorted_all[:5]):
    medal = [">>>1st", "   2nd", "   3rd", "   4th", "   5th"][i] if i < 5 else f"   {i+1}th"
    print(f"  {medal}: {n:<20} Top1={r['top1']*100:>4.0f}% Top10={r['top10']*100:>4.0f}% {r['size_mb']:>6.1f}MB [{r['approach']}]")

with open('paradigm_breakers_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

best_name = max(all_results, key=lambda k: all_results[k]['top10']) if all_results else "none"
if all_results:
    best = all_results[best_name]
    print(f"\nWinner: {best_name} at {best['top10']*100:.0f}% top-10")
    if best['top10'] > 0.63:
        print(">>> BEATS GENOME BASELINE (63%). New approach validated!")
    elif best['top10'] > 0.50:
        print(">>> Competitive. With hidden supervision could push higher.")
    else:
        print(">>> Below genome. But these are V1 — architecture refinement needed.")
print(f"{'='*70}")
