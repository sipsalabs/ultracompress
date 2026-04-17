"""
1.7B MULTI-BASIS BLOCK ROUTING (MBBR) — NOVEL ARCHITECTURE

CORE IDEA:
  Every existing FRR variant we tested (dual-block, LoRA per-layer, frozen+mod,
  deep conditioning) tops out at ~68.23% last-tok T10. They all share a fatal
  assumption: ONE shared weight W per role, with small per-layer perturbations.

  Evidence it's fundamental:
    - +29M params (dual-block) → +0.03pp (noise)
    - +0.5M LoRA → regressed to 67.9%
    - Freezing block but training γ/β → same ceiling

  Hypothesis: 28 teacher layers perform QUALITATIVELY different computations
  (syntax → semantics → task). A single W plus γ/β scaling cannot represent
  that. LoRA fails because rank-r ADDITIVE deltas cannot change the function
  class — they just warp one fixed W.

NOVEL SOLUTION:
  For each weight role r ∈ {qkv, o_proj, gate, up, down}, store K parallel
  basis matrices W_1^(r), ..., W_K^(r). At virtual layer ℓ, the effective
  projection is a per-layer mixture:
    out_ℓ^(r) = Σ_k α_{ℓ,k}^(r) · (W_k^(r) · h)

  This is a CP-style tensor decomposition of the teacher's stack of 28
  layer-weight-tensors. K basis matrices span a K-dim subspace of functions;
  per-layer α's pick a unique operator for each virtual layer.

  Unlike LoRA: bases are FULL-RANK, not low-rank.
  Unlike dual-block: mixing is PER-LAYER and SMOOTH, not discrete hard-switch.
  Unlike MoE: all K paths contribute (no routing), so no load-balancing pain.

INIT:
  W_1 = pretrained block weights (from frr_1.7b_100k_final.pt)
  W_2..K = pretrained + small Gaussian noise (breaks symmetry)
  α_{ℓ,:} = [1, 0, 0, ...] (so initial forward pass == baseline)
  → first step T10 should match record baseline, then diverges

PARAMS (K=2, hidden=2048, ff_mult=1):
  5 bases × (qkv: 3·2048²/K | o: 2048²/K | gate+up+down: 3·2048²/K) ≈ 58.7M
  + per-layer α: 28 · 5 · 2 = 280 scalars (negligible)
  + per-layer γ/β: 28 · 2 · 2048 = 115K
  + per-layer iter_scale: 28 scalars
  TOTAL: ~58.9M trainable

COMPRESSION:
  Teacher layer params: 28 · (4·2048² + 3·2048·6144) ≈ 840M
  Student: 58.9M → ~14.3× compression (vs 27.9× for single-block record)

  Trade-off: halved compression for architectural diversity. If we beat 68.23%
  by >1pp, we justify the trade. If not, we confirm architecture isn't the
  wall and need to rethink (input-cond weights, INR, continuous-depth).

PURE KL loss (confirmed winning recipe from step-5000 record).
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
K_BASES = 2                   # number of weight bases per role
LR_BASE = 3e-5                # LR for the K weight matrices
LR_ALPHA = 1e-3               # higher LR for mixing coefs (fresh params)
LR_MOD = 3e-4                 # γ/β
WD_BASE = 0.05
WD_ALPHA = 0.0                # no decay on mixing (small init)
WD_MOD = 0.01
BATCH_SIZE = 4
SEQ_LEN = 64
TEMP = 2.0
BASE2_NOISE = 0.01            # small Gaussian perturbation for basis 2..K init
EVAL_INTERVAL = 2_500
CHECKPOINT_INTERVAL = 5_000
N_SCALES = 4
ITERS_PER_SCALE = 7
N_TEACHER_LAYERS = 28
RESUME_FROM = 'checkpoints_1.7b_real_text/frr_1.7b_100k_final.pt'
CHECKPOINT_DIR = 'checkpoints_1.7b_multibasis'

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("=" * 70)
print(f"1.7B MULTI-BASIS BLOCK ROUTING  (K={K_BASES} bases)")
print(f"Device: {DEVICE}  |  Temp: {TEMP}  |  Steps: {TOTAL_STEPS:,}")
print(f"LR_base={LR_BASE}  LR_alpha={LR_ALPHA}  LR_mod={LR_MOD}")
print(f"Loss: PURE KL")
print("=" * 70)


# ── Model ─────────────────────────────────────────────────────────────
class MultiBasisBlock(nn.Module):
    """K parallel weight bases per role, mixed per virtual layer.

    Output at layer ℓ:
        qkv_ℓ(h) = Σ_k α_qkv[ℓ,k] · (qkv_base_k · h)
    Similarly for o, gate, up, down.
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

        # K parallel weight matrices per role
        # Shape: (K, out_features, in_features)
        self.qkv_bases = nn.Parameter(torch.zeros(K, 3 * hidden_dim, hidden_dim))
        self.o_bases = nn.Parameter(torch.zeros(K, hidden_dim, hidden_dim))
        self.gate_bases = nn.Parameter(torch.zeros(K, ff_dim, hidden_dim))
        self.up_bases = nn.Parameter(torch.zeros(K, ff_dim, hidden_dim))
        self.down_bases = nn.Parameter(torch.zeros(K, hidden_dim, ff_dim))

        # Per-layer mixing coefficients — init [1, 0, 0, ...] so layer ℓ starts
        # as basis-0 (the pretrained weights)
        alpha_init = torch.zeros(total_layers, K)
        alpha_init[:, 0] = 1.0
        self.alpha_qkv = nn.Parameter(alpha_init.clone())
        self.alpha_o = nn.Parameter(alpha_init.clone())
        self.alpha_gate = nn.Parameter(alpha_init.clone())
        self.alpha_up = nn.Parameter(alpha_init.clone())
        self.alpha_down = nn.Parameter(alpha_init.clone())

        # Shared norms (they're tiny — 4K params, layer-specific γ/β handles diff)
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.norm2 = nn.RMSNorm(hidden_dim)

    def load_from_pretrained_block(self, block_state_dict, noise_std=0.01):
        """Initialize: W_1 = pretrained, W_{2..K} = pretrained + noise."""
        with torch.no_grad():
            # qkv
            w = block_state_dict['qkv.weight']  # (3*D, D)
            for k in range(self.K):
                self.qkv_bases[k].copy_(w)
                if k > 0:
                    self.qkv_bases[k].add_(torch.randn_like(w) * noise_std * w.std())
            # o_proj
            w = block_state_dict['o_proj.weight']
            for k in range(self.K):
                self.o_bases[k].copy_(w)
                if k > 0:
                    self.o_bases[k].add_(torch.randn_like(w) * noise_std * w.std())
            # gate
            w = block_state_dict['gate.weight']
            for k in range(self.K):
                self.gate_bases[k].copy_(w)
                if k > 0:
                    self.gate_bases[k].add_(torch.randn_like(w) * noise_std * w.std())
            # up
            w = block_state_dict['up.weight']
            for k in range(self.K):
                self.up_bases[k].copy_(w)
                if k > 0:
                    self.up_bases[k].add_(torch.randn_like(w) * noise_std * w.std())
            # down
            w = block_state_dict['down.weight']
            for k in range(self.K):
                self.down_bases[k].copy_(w)
                if k > 0:
                    self.down_bases[k].add_(torch.randn_like(w) * noise_std * w.std())
            # Norms
            self.norm1.weight.copy_(block_state_dict['norm1.weight'])
            self.norm2.weight.copy_(block_state_dict['norm2.weight'])

    def layer_forward(self, x, layer_idx, gamma, beta, iter_scale):
        """One virtual layer of block at index layer_idx."""
        B, T, D = x.shape
        Hd = self.head_dim

        # ── Attention ──
        h = self.norm1(x)
        h = h * gamma + beta

        # QKV: compute K parallel projections, mix by α
        # qkv_bases: (K, 3D, D) ; h: (B, T, D) → (K, B, T, 3D) via einsum
        # More memory-efficient: fold α into basis first, then single matmul
        a_qkv = self.alpha_qkv[layer_idx]           # (K,)
        W_qkv = torch.einsum('k,kij->ij', a_qkv, self.qkv_bases)   # (3D, D)
        qkv = F.linear(h, W_qkv)                    # (B, T, 3D)
        qkv = qkv.reshape(B, T, 3, self.n_heads, Hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(Hd)
        if T > 1:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)

        a_o = self.alpha_o[layer_idx]
        W_o = torch.einsum('k,kij->ij', a_o, self.o_bases)
        attn_out = F.linear(out, W_o)
        x_attn = x + attn_out

        # ── FFN (SwiGLU) ──
        h = self.norm2(x_attn)
        h = h * gamma + beta

        a_g = self.alpha_gate[layer_idx]
        a_u = self.alpha_up[layer_idx]
        a_d = self.alpha_down[layer_idx]
        W_g = torch.einsum('k,kij->ij', a_g, self.gate_bases)
        W_u = torch.einsum('k,kij->ij', a_u, self.up_bases)
        W_d = torch.einsum('k,kij->ij', a_d, self.down_bases)

        gate = F.linear(h, W_g)
        up = F.linear(h, W_u)
        ffn_in = F.silu(gate) * up
        ffn_out = F.linear(ffn_in, W_d)

        block_out = x_attn + ffn_out
        # Residual scaling (iter_scale preserves FRR convention)
        return x + (block_out - x) * iter_scale


class MultiBasisFRR(nn.Module):
    def __init__(self, hidden_dim, n_heads, n_scales, iters_per_scale,
                 vocab_size, ff_mult, K,
                 embed_weight=None, lm_head_weight=None, norm_weight=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_scales = n_scales
        self.iters_per_scale = iters_per_scale
        self.total_layers = n_scales * iters_per_scale

        self.block = MultiBasisBlock(
            hidden_dim, n_heads, ff_mult, K, self.total_layers
        )
        # Per-layer γ/β (on top of mixing — still useful for fine mod)
        self.layer_gamma = nn.Parameter(torch.ones(self.total_layers, hidden_dim))
        self.layer_beta = nn.Parameter(torch.zeros(self.total_layers, hidden_dim))
        self.iter_scale = nn.Parameter(torch.ones(n_scales, iters_per_scale))

        # Frozen embed + head + final norm
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

    def forward(self, tokens):
        x = self.embed(tokens).float()
        layer_idx = 0
        for scale in range(self.n_scales):
            for it in range(self.iters_per_scale):
                gamma = self.layer_gamma[layer_idx]
                beta = self.layer_beta[layer_idx]
                iter_s = self.iter_scale[scale, it]
                x = self.block.layer_forward(x, layer_idx, gamma, beta, iter_s)
                layer_idx += 1
        x = self.norm(x)
        return self.lm_head(x)

    def base_params(self):
        b = self.block
        return sum(p.numel() for p in [
            b.qkv_bases, b.o_bases, b.gate_bases, b.up_bases, b.down_bases,
            b.norm1.weight, b.norm2.weight
        ])

    def alpha_params(self):
        b = self.block
        return sum(p.numel() for p in [
            b.alpha_qkv, b.alpha_o, b.alpha_gate, b.alpha_up, b.alpha_down
        ])

    def mod_params(self):
        return self.layer_gamma.numel() + self.layer_beta.numel() + self.iter_scale.numel()

    def trainable_params(self):
        return self.base_params() + self.alpha_params() + self.mod_params()


# ── Load teacher ──────────────────────────────────────────────────────
print("\nLoading Qwen3-1.7B teacher...")
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
print(f"  Hidden: {hidden}, Heads: {n_heads}, Vocab: {vocab_size}")

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
print("\nLoading data...")
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
print(f"\nBuilding MultiBasis FRR (K={K_BASES})...")
model = MultiBasisFRR(
    hidden_dim=hidden, n_heads=n_heads,
    n_scales=N_SCALES, iters_per_scale=ITERS_PER_SCALE,
    vocab_size=vocab_size, ff_mult=1, K=K_BASES,
    embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w,
).to(DEVICE)

# Load pretrained block + γ/β
print(f"Loading base checkpoint: {RESUME_FROM}")
ckpt = torch.load(RESUME_FROM, map_location=DEVICE, weights_only=True)
block_state = {k.replace('block.', ''): v for k, v in ckpt.items() if k.startswith('block.')}
torch.manual_seed(1337)
model.block.load_from_pretrained_block(block_state, noise_std=BASE2_NOISE)

# Copy scale-gamma/beta → per-layer
scale_gamma = ckpt['scale_gamma']
scale_beta = ckpt['scale_beta']
with torch.no_grad():
    for scale in range(N_SCALES):
        for it in range(ITERS_PER_SCALE):
            layer_idx = scale * ITERS_PER_SCALE + it
            model.layer_gamma.data[layer_idx] = scale_gamma[scale]
            model.layer_beta.data[layer_idx] = scale_beta[scale]
    model.iter_scale.data.copy_(ckpt['iter_scale'])
print("  Loaded pretrained block as basis-0 (others = +noise)")

bp = model.base_params()
ap = model.alpha_params()
mp = model.mod_params()
total = bp + ap + mp
teacher_layer_params = 4 * hidden * hidden + 3 * hidden * (hidden * 3)
teacher_total = N_TEACHER_LAYERS * teacher_layer_params
print(f"\n  Base matrices: {bp:,}  ({bp/1e6:.1f}M)")
print(f"  Alpha (mixing): {ap:,}")
print(f"  Mod (γ/β/iter): {mp:,}")
print(f"  TOTAL trainable: {total:,}  ({total/1e6:.1f}M)")
print(f"  Teacher layer params: ~{teacher_total/1e6:.0f}M")
print(f"  Compression: {teacher_total/total:.1f}x")


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
def eval_last_tok(n=100):
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


print("\nBaseline (should match single-block pretrained):")
t1_b, t10_b = eval_all_pos(50)
t1_l, t10_l = eval_last_tok(200)
print(f"  all-pos:  T1={t1_b*100:.1f}%  T10={t10_b*100:.1f}%")
print(f"  last-tok: T1={t1_l*100:.1f}%  T10={t10_l*100:.1f}%")


# ── Training ──────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"TRAINING: MULTI-BASIS + PURE KL ({TOTAL_STEPS:,} steps)")
print(f"{'='*70}")

base_list = [
    model.block.qkv_bases, model.block.o_bases,
    model.block.gate_bases, model.block.up_bases, model.block.down_bases,
    model.block.norm1.weight, model.block.norm2.weight,
]
alpha_list = [
    model.block.alpha_qkv, model.block.alpha_o,
    model.block.alpha_gate, model.block.alpha_up, model.block.alpha_down,
]
mod_list = [model.layer_gamma, model.layer_beta, model.iter_scale]

opt = torch.optim.AdamW([
    {'params': base_list, 'lr': LR_BASE, 'weight_decay': WD_BASE},
    {'params': alpha_list, 'lr': LR_ALPHA, 'weight_decay': WD_ALPHA},
    {'params': mod_list, 'lr': LR_MOD, 'weight_decay': WD_MOD},
])
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, TOTAL_STEPS)

loss_hist = []
best_last_t10 = 0.0
best_step = -1
t0 = time.time()

for step in range(TOTAL_STEPS):
    tokens = get_batch()
    with torch.no_grad():
        tl = teacher.forward(tokens, max_layers=N_TEACHER_LAYERS)

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
        if t10_l > best_last_t10:
            best_last_t10 = t10_l
            best_step = step
            mark = " *** NEW BEST (last-tok) ***"
            torch.save(model.state_dict(),
                       os.path.join(CHECKPOINT_DIR, 'mbbr_best.pt'))
        avg_loss = sum(loss_hist[-500:]) / min(len(loss_hist), 500)

        # Alpha divergence diagnostic: std across layers for each role
        with torch.no_grad():
            a_qkv_div = model.block.alpha_qkv.std(dim=0).mean().item()
            a_ffn_div = (model.block.alpha_gate.std(dim=0).mean().item() +
                         model.block.alpha_up.std(dim=0).mean().item() +
                         model.block.alpha_down.std(dim=0).mean().item()) / 3
            # Basis divergence: ||B_1 - B_0|| / ||B_0||
            b0, b1 = model.block.qkv_bases[0], model.block.qkv_bases[1]
            basis_div = (b1 - b0).norm().item() / b0.norm().item()

        print(
            f"  Step {step:>6d}/{TOTAL_STEPS}: KL={avg_loss:.4f}  "
            f"T1={t1*100:.1f}% T10={t10*100:.1f}%  ({elapsed:.0f}s){mark}"
        )
        print(
            f"    [last-tok n=200] T1={t1_l*100:.1f}% T10={t10_l*100:.1f}% "
            f"best={best_last_t10*100:.1f}% @step{best_step}"
        )
        print(
            f"    α_qkv_std={a_qkv_div:.4f}  α_ffn_std={a_ffn_div:.4f}  "
            f"‖B1-B0‖/‖B0‖={basis_div:.4f}"
        )

    if step > 0 and step % CHECKPOINT_INTERVAL == 0:
        path = os.path.join(CHECKPOINT_DIR, f'mbbr_step{step}.pt')
        torch.save(model.state_dict(), path)
        print(f"  >> Saved: {path}")

# Final
torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'mbbr_final.pt'))
print(f"\n{'='*70}")
print(f"MBBR COMPLETE  |  Best last-tok T10: {best_last_t10*100:.1f}% @ step {best_step}")
print(f"  Elapsed: {(time.time()-t0):.0f}s ({(time.time()-t0)/60:.1f}m)")
print(f"  Record to beat: 68.23% (pure_kl_step5000, hires)")
print(f"{'='*70}")
