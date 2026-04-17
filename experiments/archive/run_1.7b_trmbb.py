"""
1.7B TOKEN-ROUTED MULTI-BASIS BLOCK (TRMBB) — NOVEL ARCHITECTURE

OBSERVATION: Three hires confirmations all cluster at last-tok T10 = 68%:
  - pure_kl_step5000: 68.23% (CI 67.36-69.11)
  - lora_best:        67.67% (CI 66.82-68.56)
  - lora_final:       68.15% (CI 67.30-69.10)

Every PER-LAYER variant (LoRA, dual-block, frozen, MBBR K=2) collapses to
the same 68% ceiling. They ALL share one assumption: every token at layer
ℓ gets the SAME weights.

NOVEL BREAK: Token-conditioned routing. At each virtual layer, each token
chooses its OWN soft mixture of K basis matrices via a tiny learned gate.
Different tokens = different functions at the same depth.

ROUTER:
  α_ℓ(h) = softmax(W_route_ℓ · h / √D)              [B, T, K]
  W_qkv_effective(h) = Σ_k α_ℓ(h)[k] · W_qkv_base_k

Per-token, per-layer, per-role gate. Implementation trick: since
  α · (W · h) = Σ_k α_k · (W_k · h),
we compute K parallel projections and mix the OUTPUTS (not reconstruct W).
This keeps the matmul cost 2K× but is memory-efficient and differentiable.

ROUTER PARAM COST:
  Per role (qkv, o, gate, up, down), per layer: D × K = 2048 × 2 = 4096
  × 5 roles × 28 layers = 573K params for routing
  + K=2 bases (same as MBBR): 58.7M
  + per-layer γ/β: 115K
  TOTAL: ~59.3M

COMPRESSION: ~25.8× (same as MBBR)

INIT: W_route_ℓ ≈ 0 + small noise, so α ≈ [1, 0, ..., 0] initially
  (uniform at (1/K, 1/K, ...)). Adjustment: bias toward basis-0 at init via
  bias vector in router, so α ≈ [1, 0, ...] exactly at start.

Loss: PURE KL (confirmed winning recipe).
Device: cuda:1.
"""
import lib.unbuffered
import torch
import sys
import os
import time
import math

import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from ultracompress.inference import ModelConfig, MiniTransformer

# ── Configuration ────────────────────────────────────────────────────
DEVICE = 'cuda:1'
TOTAL_STEPS = 15_000
K_BASES = 2
LR_BASE = 3e-5
LR_ALPHA = 1e-3
LR_MOD = 3e-4
LR_ROUTE = 5e-4         # LR for router
WD_BASE = 0.05
WD_ROUTE = 0.0
WD_MOD = 0.01
BATCH_SIZE = 4
SEQ_LEN = 64
TEMP = 2.0
BASE2_NOISE = 0.02
ROUTE_INIT_SCALE = 0.02  # Small router init → α ≈ [1, 0, ...] via bias
EVAL_INTERVAL = 2_500
CHECKPOINT_INTERVAL = 5_000
N_SCALES = 4
ITERS_PER_SCALE = 7
N_TEACHER_LAYERS = 28
RESUME_FROM = 'checkpoints_1.7b_real_text/frr_1.7b_100k_final.pt'
CHECKPOINT_DIR = 'checkpoints_1.7b_trmbb'

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("=" * 70)
print(f"1.7B TOKEN-ROUTED MULTI-BASIS BLOCK  (K={K_BASES}, per-token α)")
print(f"Device: {DEVICE}  |  Temp: {TEMP}  |  Steps: {TOTAL_STEPS:,}")
print("NOVEL: input-conditioned routing — different tokens see different weights")
print("=" * 70)


class TokenRoutedBlock(nn.Module):
    """K parallel weight bases per role, with per-TOKEN routing gates.

    At layer ℓ, for input x (B, T, D):
        α_ℓ(x) = softmax(W_route_ℓ · x + b_ℓ)    shape (B, T, K)
        out_ℓ(x)[b,t,:] = Σ_k α[b,t,k] · (W_k · x[b,t,:])

    Bias b_ℓ is initialized to [log(1), log(0), ...] = [0, -∞, ...]
    so α starts as one-hot on basis-0 → forward pass == baseline at init.
    """
    def __init__(self, hidden_dim, n_heads, ff_mult, K, total_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.K = K
        self.total_layers = total_layers
        ff_dim = hidden_dim * ff_mult
        self.ff_dim = ff_dim

        # K parallel weight bases per role
        self.qkv_bases = nn.Parameter(torch.zeros(K, 3 * hidden_dim, hidden_dim))
        self.o_bases = nn.Parameter(torch.zeros(K, hidden_dim, hidden_dim))
        self.gate_bases = nn.Parameter(torch.zeros(K, ff_dim, hidden_dim))
        self.up_bases = nn.Parameter(torch.zeros(K, ff_dim, hidden_dim))
        self.down_bases = nn.Parameter(torch.zeros(K, hidden_dim, ff_dim))

        # Router per layer per role: W_route [L, K, D] and bias b [L, K]
        # At init, W_route ≈ 0 and b = [large, 0, 0, ...] → α ≈ [1, 0, ...]
        self.route_qkv = nn.Parameter(torch.zeros(total_layers, K, hidden_dim))
        self.route_o = nn.Parameter(torch.zeros(total_layers, K, hidden_dim))
        self.route_gate = nn.Parameter(torch.zeros(total_layers, K, hidden_dim))
        self.route_up = nn.Parameter(torch.zeros(total_layers, K, hidden_dim))
        self.route_down = nn.Parameter(torch.zeros(total_layers, K, hidden_dim))

        # Bias: start with α[:, :, 0] = 1 (force basis-0 at init)
        bias_init = torch.zeros(total_layers, K)
        bias_init[:, 0] = 5.0   # softmax([5, 0, ...]) ≈ [0.993, 0.007, ...]
        self.bias_qkv = nn.Parameter(bias_init.clone())
        self.bias_o = nn.Parameter(bias_init.clone())
        self.bias_gate = nn.Parameter(bias_init.clone())
        self.bias_up = nn.Parameter(bias_init.clone())
        self.bias_down = nn.Parameter(bias_init.clone())

        self.norm1 = nn.RMSNorm(hidden_dim)
        self.norm2 = nn.RMSNorm(hidden_dim)

    def load_from_pretrained_block(self, block_sd, noise_std=0.02):
        """W_1 = pretrained, W_{k>1} = pretrained + noise."""
        mapping = [
            (self.qkv_bases, 'qkv.weight'),
            (self.o_bases, 'o_proj.weight'),
            (self.gate_bases, 'gate.weight'),
            (self.up_bases, 'up.weight'),
            (self.down_bases, 'down.weight'),
        ]
        with torch.no_grad():
            for param, key in mapping:
                w = block_sd[key]
                for k in range(self.K):
                    param[k].copy_(w)
                    if k > 0:
                        param[k].add_(torch.randn_like(w) * noise_std * w.std())
            self.norm1.weight.copy_(block_sd['norm1.weight'])
            self.norm2.weight.copy_(block_sd['norm2.weight'])

    def _route(self, x, W_route, bias, layer_idx):
        """Compute routing weights α[B, T, K] for this layer."""
        # W_route: [L, K, D], bias: [L, K]
        # x: [B, T, D] → logits [B, T, K]
        logits = torch.einsum('btd,kd->btk', x, W_route[layer_idx])
        logits = logits / math.sqrt(self.hidden_dim)
        logits = logits + bias[layer_idx]
        return F.softmax(logits, dim=-1)

    def _mixed_linear(self, x, bases, route_w, route_b, layer_idx):
        """Compute y[B,T,out] = Σ_k α[b,t,k] · (bases[k] · x[b,t,:])."""
        # x: (B, T, D_in), bases: (K, D_out, D_in)
        # Per-k projection: y_k = x @ bases[k].T  → (B, T, D_out)
        # Stack: (K, B, T, D_out) then weighted sum by α
        alpha = self._route(x, route_w, route_b, layer_idx)   # (B, T, K)
        # Efficient: y_k = F.linear(x, bases[k])  — loop over K is fine at K=2..4
        y_parts = [F.linear(x, bases[k]) for k in range(self.K)]
        y_stack = torch.stack(y_parts, dim=-1)   # (B, T, D_out, K)
        # Weighted sum over K
        y = (y_stack * alpha.unsqueeze(-2)).sum(dim=-1)   # (B, T, D_out)
        return y, alpha

    def layer_forward(self, x, layer_idx, gamma, beta, iter_scale):
        B, T, D = x.shape
        Hd = self.head_dim

        # ── Attention ──
        h = self.norm1(x)
        h = h * gamma + beta

        qkv, a_qkv = self._mixed_linear(h, self.qkv_bases, self.route_qkv, self.bias_qkv, layer_idx)
        qkv = qkv.reshape(B, T, 3, self.n_heads, Hd).permute(2, 0, 3, 1, 4)
        q, k_v, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k_v.transpose(-2, -1)) / math.sqrt(Hd)
        if T > 1:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)

        attn_out, a_o = self._mixed_linear(out, self.o_bases, self.route_o, self.bias_o, layer_idx)
        x_attn = x + attn_out

        # ── FFN ──
        h = self.norm2(x_attn)
        h = h * gamma + beta
        gate_out, a_g = self._mixed_linear(h, self.gate_bases, self.route_gate, self.bias_gate, layer_idx)
        up_out, a_u = self._mixed_linear(h, self.up_bases, self.route_up, self.bias_up, layer_idx)
        ffn_in = F.silu(gate_out) * up_out
        down_out, a_d = self._mixed_linear(ffn_in, self.down_bases, self.route_down, self.bias_down, layer_idx)
        x_ffn = x_attn + down_out

        return x + (x_ffn - x) * iter_scale, a_qkv


class TRMBBModel(nn.Module):
    def __init__(self, hidden_dim, n_heads, n_scales, iters_per_scale,
                 vocab_size, ff_mult, K,
                 embed_weight=None, lm_head_weight=None, norm_weight=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_scales = n_scales
        self.iters_per_scale = iters_per_scale
        self.total_layers = n_scales * iters_per_scale

        self.block = TokenRoutedBlock(
            hidden_dim, n_heads, ff_mult, K, self.total_layers
        )
        self.layer_gamma = nn.Parameter(torch.ones(self.total_layers, hidden_dim))
        self.layer_beta = nn.Parameter(torch.zeros(self.total_layers, hidden_dim))
        self.iter_scale = nn.Parameter(torch.ones(n_scales, iters_per_scale))

        self.embed = nn.Embedding.from_pretrained(embed_weight, freeze=True)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.lm_head.weight = nn.Parameter(lm_head_weight, requires_grad=False)
        self.norm = nn.RMSNorm(hidden_dim)
        self.norm.weight = nn.Parameter(norm_weight, requires_grad=False)

    def forward(self, tokens, return_alpha_stats=False):
        x = self.embed(tokens).float()
        layer_idx = 0
        alpha_stats = []
        for s in range(self.n_scales):
            for it in range(self.iters_per_scale):
                gamma = self.layer_gamma[layer_idx]
                beta = self.layer_beta[layer_idx]
                it_s = self.iter_scale[s, it]
                x, a_qkv = self.block.layer_forward(x, layer_idx, gamma, beta, it_s)
                if return_alpha_stats:
                    alpha_stats.append(a_qkv.mean(dim=(0, 1)))   # (K,)
                layer_idx += 1
        x = self.norm(x)
        logits = self.lm_head(x)
        if return_alpha_stats:
            return logits, torch.stack(alpha_stats, dim=0)   # (L, K)
        return logits

    def base_params(self):
        b = self.block
        return sum(p.numel() for p in [
            b.qkv_bases, b.o_bases, b.gate_bases, b.up_bases, b.down_bases,
            b.norm1.weight, b.norm2.weight,
        ])

    def route_params(self):
        b = self.block
        return sum(p.numel() for p in [
            b.route_qkv, b.route_o, b.route_gate, b.route_up, b.route_down,
            b.bias_qkv, b.bias_o, b.bias_gate, b.bias_up, b.bias_down,
        ])

    def mod_params(self):
        return self.layer_gamma.numel() + self.layer_beta.numel() + self.iter_scale.numel()


# ── Load teacher ──────────────────────────────────────────────────────
print("\nLoading teacher...")
wd = torch.load('qwen3_1.7b_cache.pt', weights_only=True)
hf_to_gguf = {
    'self_attn.q_proj.weight': 'attn_q.weight',
    'self_attn.k_proj.weight': 'attn_k.weight',
    'self_attn.v_proj.weight': 'attn_v.weight',
    'self_attn.o_proj.weight': 'attn_output.weight',
    'self_attn.q_norm.weight': 'attn_q_norm.weight',
    'self_attn.k_norm.weight': 'attn_k_norm.weight',
    'input_layernorm.weight': 'attn_norm.weight',
    'post_attention_layernorm.weight': 'ffn_norm.weight',
    'mlp.gate_proj.weight': 'ffn_gate.weight',
    'mlp.up_proj.weight': 'ffn_up.weight',
    'mlp.down_proj.weight': 'ffn_down.weight',
}
gd = {}
gd['token_embd.weight'] = wd['model.embed_tokens.weight'].float()
gd['output_norm.weight'] = wd.get('model.norm.weight', torch.ones(2048)).float()
gd['output.weight'] = wd.get('lm_head.weight', gd['token_embd.weight']).float()
for li in range(N_TEACHER_LAYERS):
    for h, g in hf_to_gguf.items():
        k = f'model.layers.{li}.{h}'
        if k in wd:
            gd[f'blk.{li}.{g}'] = wd[k].float()
del wd

hidden = gd['token_embd.weight'].shape[1]
n_heads = 16
vocab_size = gd['token_embd.weight'].shape[0]

cfg = ModelConfig(
    n_layers=N_TEACHER_LAYERS, n_heads=n_heads, n_kv_heads=8,
    hidden_size=hidden, intermediate_size=hidden * 3,
    vocab_size=vocab_size, head_dim=hidden // n_heads,
)
teacher = MiniTransformer(cfg, DEVICE)
teacher.load_weights(gd)
teacher.embed_weight = teacher.embed_weight.to(DEVICE)
if teacher.lm_head is not None:
    teacher.lm_head = teacher.lm_head.to(DEVICE)
embed_w = gd['token_embd.weight'].to(DEVICE)
norm_w = gd['output_norm.weight'].to(DEVICE)
lm_head_w = gd['output.weight'].to(DEVICE)
del gd

# ── Data ──────────────────────────────────────────────────────────────
print("Loading data...")
all_tokens = torch.load('fineweb_edu_100M_tokens.pt', weights_only=True)
N_TOKENS = all_tokens.shape[0]
print(f"  {N_TOKENS:,} tokens")

data_offset = 0
def get_batch():
    global data_offset
    end = data_offset + BATCH_SIZE * SEQ_LEN
    if end > N_TOKENS:
        data_offset = 0
        end = BATCH_SIZE * SEQ_LEN
    chunk = all_tokens[data_offset:end].long().reshape(BATCH_SIZE, SEQ_LEN)
    data_offset = end
    return chunk.to(DEVICE)


# ── Build model ───────────────────────────────────────────────────────
print(f"\nBuilding TRMBB (K={K_BASES})...")
model = TRMBBModel(
    hidden_dim=hidden, n_heads=n_heads,
    n_scales=N_SCALES, iters_per_scale=ITERS_PER_SCALE,
    vocab_size=vocab_size, ff_mult=1, K=K_BASES,
    embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w,
).to(DEVICE)

print(f"Loading base: {RESUME_FROM}")
ckpt = torch.load(RESUME_FROM, map_location=DEVICE, weights_only=True)
block_sd = {k.replace('block.', ''): v for k, v in ckpt.items() if k.startswith('block.')}
torch.manual_seed(1337)
model.block.load_from_pretrained_block(block_sd, noise_std=BASE2_NOISE)

# γ/β from scale → per-layer
scale_gamma = ckpt['scale_gamma']
scale_beta = ckpt['scale_beta']
with torch.no_grad():
    for s in range(N_SCALES):
        for i in range(ITERS_PER_SCALE):
            li = s * ITERS_PER_SCALE + i
            model.layer_gamma.data[li] = scale_gamma[s]
            model.layer_beta.data[li] = scale_beta[s]
    model.iter_scale.data.copy_(ckpt['iter_scale'])
print("  Loaded pretrained block as basis-0 (others = +noise)")

bp = model.base_params()
rp = model.route_params()
mp = model.mod_params()
total = bp + rp + mp
teacher_layer_total = N_TEACHER_LAYERS * (4 * hidden * hidden + 3 * hidden * hidden * 3)
print(f"\n  Bases:     {bp:,}  ({bp/1e6:.1f}M)")
print(f"  Routers:   {rp:,}  ({rp/1e6:.2f}M)")
print(f"  Mod:       {mp:,}")
print(f"  TOTAL:     {total:,}  ({total/1e6:.1f}M)")
print(f"  Compression: {teacher_layer_total/total:.1f}x")


# ── Eval ──────────────────────────────────────────────────────────────
@torch.no_grad()
def eval_all_pos(n=50):
    model.eval()
    t1, t10, nt = 0, 0, 0
    for _ in range(n):
        s = torch.randint(0, all_tokens.numel() - SEQ_LEN, (1,)).item()
        tokens = all_tokens[s:s + SEQ_LEN].unsqueeze(0).long().to(DEVICE)
        tl = teacher.forward(tokens, max_layers=N_TEACHER_LAYERS)
        sl = model(tokens)
        for pos in range(SEQ_LEN):
            t_top = tl[0, pos].topk(10).indices
            s_top = sl[0, pos].topk(10).indices
            t1 += int(s_top[0] == t_top[0])
            t10 += len(set(t_top.tolist()) & set(s_top.tolist())) / 10
            nt += 1
    model.train()
    return t1 / nt, t10 / nt


@torch.no_grad()
def eval_last_tok(n=200):
    model.eval()
    t1, t10 = 0, 0
    for _ in range(n):
        s = torch.randint(0, all_tokens.numel() - SEQ_LEN, (1,)).item()
        tokens = all_tokens[s:s + SEQ_LEN].unsqueeze(0).long().to(DEVICE)
        tl = teacher.forward(tokens, max_layers=N_TEACHER_LAYERS)
        sl = model(tokens)
        t_top = tl[0, -1].topk(10).indices
        s_top = sl[0, -1].topk(10).indices
        t1 += int(s_top[0] == t_top[0])
        t10 += len(set(t_top.tolist()) & set(s_top.tolist())) / 10
    model.train()
    return t1 / n, t10 / n


print("\nBaseline (should match pretrained, α ≈ [1, 0]):")
t1_b, t10_b = eval_all_pos(50)
t1_l, t10_l = eval_last_tok(200)
print(f"  all-pos:  T1={t1_b*100:.1f}%  T10={t10_b*100:.1f}%")
print(f"  last-tok: T1={t1_l*100:.1f}%  T10={t10_l*100:.1f}%")


# ── Training ──────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"TRAINING: TRMBB + PURE KL ({TOTAL_STEPS:,} steps)")
print(f"{'='*70}")

base_list = [
    model.block.qkv_bases, model.block.o_bases,
    model.block.gate_bases, model.block.up_bases, model.block.down_bases,
    model.block.norm1.weight, model.block.norm2.weight,
]
route_list = [
    model.block.route_qkv, model.block.route_o,
    model.block.route_gate, model.block.route_up, model.block.route_down,
    model.block.bias_qkv, model.block.bias_o,
    model.block.bias_gate, model.block.bias_up, model.block.bias_down,
]
mod_list = [model.layer_gamma, model.layer_beta, model.iter_scale]

opt = torch.optim.AdamW([
    {'params': base_list, 'lr': LR_BASE, 'weight_decay': WD_BASE},
    {'params': route_list, 'lr': LR_ROUTE, 'weight_decay': WD_ROUTE},
    {'params': mod_list, 'lr': LR_MOD, 'weight_decay': WD_MOD},
])
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, TOTAL_STEPS)

loss_hist = []
best_last = 0.0
best_step = -1
t0 = time.time()

for step in range(TOTAL_STEPS):
    tokens = get_batch()
    with torch.no_grad():
        tl = teacher.forward(tokens, max_layers=N_TEACHER_LAYERS)

    if step % EVAL_INTERVAL == 0:
        sl, alpha_by_layer = model(tokens, return_alpha_stats=True)
    else:
        sl = model(tokens)

    loss = F.kl_div(
        F.log_softmax(sl / TEMP, dim=-1),
        F.softmax(tl / TEMP, dim=-1),
        reduction='batchmean',
    ) * TEMP * TEMP

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    loss_hist.append(loss.item())

    if step % EVAL_INTERVAL == 0 or step == TOTAL_STEPS - 1:
        t1, t10 = eval_all_pos(50)
        t1_l, t10_l = eval_last_tok(200)
        elapsed = time.time() - t0
        mark = ""
        if t10_l > best_last:
            best_last = t10_l
            best_step = step
            mark = " *** NEW BEST (last-tok) ***"
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'trmbb_best.pt'))

        avg_loss = sum(loss_hist[-500:]) / min(len(loss_hist), 500)
        # Alpha diagnostics
        alpha_min = alpha_by_layer.min().item()
        alpha_max = alpha_by_layer.max().item()
        alpha_entropy = -(alpha_by_layer * alpha_by_layer.clamp(min=1e-8).log()).sum(dim=-1).mean().item()
        max_entropy = math.log(K_BASES)

        print(
            f"  Step {step:>6d}/{TOTAL_STEPS}: KL={avg_loss:.4f}  "
            f"T1={t1*100:.1f}% T10={t10*100:.1f}%  ({elapsed:.0f}s){mark}"
        )
        print(
            f"    [last-tok n=200] T1={t1_l*100:.1f}% T10={t10_l*100:.1f}% "
            f"best={best_last*100:.1f}% @step{best_step}"
        )
        print(
            f"    α_range=[{alpha_min:.3f},{alpha_max:.3f}]  "
            f"α_entropy={alpha_entropy:.3f} / max={max_entropy:.3f}  "
            f"(higher entropy = more routing diversity)"
        )

    if step > 0 and step % CHECKPOINT_INTERVAL == 0:
        path = os.path.join(CHECKPOINT_DIR, f'trmbb_step{step}.pt')
        torch.save(model.state_dict(), path)
        print(f"  >> Saved: {path}")

torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'trmbb_final.pt'))
print(f"\n{'='*70}")
print(f"TRMBB COMPLETE | Best last-tok T10: {best_last*100:.1f}% @ step {best_step}")
print(f"  Elapsed: {time.time()-t0:.0f}s  |  Record to beat: 68.23% (pure_kl)")
print(f"{'='*70}")
