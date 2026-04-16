"""
FlowODE v2 — fixes from v1 (ceiling at 59.9% last-T10 @ 970x compression):

v1 problems observed:
  - Peak at step 2000, then flat 55-59% for 6k steps. Plateau = saturated capacity.
  - f_out_norm kept growing (48 -> 89). Vector field overshooting.
  - Same vector field every step = effectively a shared block in 256-D space.
    Hit same "shared block" ceiling, just lower.

v2 FIXES (each genuinely adds expressiveness, not a repackage):

(1) TIME-CONDITIONED VECTOR FIELD
    f(z, t) where t is concatenated 16-D sinusoidal step embedding.
    Now f is NOT the same at every step — one MLP, but different behavior
    per step. 27 distinct dynamics from one shared MLP, driven by t.
    Cost: +16*4r (first layer input bump) = +16KB params.

(2) WEIGHT DECAY ON f to cap overshoot (v1 f_out grew unbounded)

(3) ADAPTIVE LAYER WEIGHTING IN LOSS
    Late layers have 150x residual norm. Plain MSE → gradient dominated by
    final layers. Scale-normalized MSE (v1) → loses magnitude info entirely.
    v2: w_ℓ = 1 / ||t_ℓ||, so each layer contributes ≈ same gradient.

(4) PATIENCE-BASED EARLY STOP
    If no new best-last-T10 in 1500 steps → stop. Saves time for next iterate.

(5) SAVE BEST AGGRESSIVELY (every 250 steps)

Expected: v1 plateau'd because a static vector field can only describe
a 1-D trajectory through manifold space. Time-conditioned f can trace
arbitrary curves.
"""
import lib.unbuffered
import sys, os, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultracompress.inference import ModelConfig, MiniTransformer

DEVICE = 'cuda:0'
TOTAL_STEPS = 8_000
LR = 3e-4
WD_F = 0.1         # strong decay on vector field
WD_PQ = 0.0        # projections should refine freely
BATCH_SIZE = 4
SEQ_LEN = 64
MANIFOLD_DIM = 256
TIME_DIM = 16
EVAL_INTERVAL = 500
PATIENCE = 1500
N_TEACHER_LAYERS = 28

os.makedirs('checkpoints_1.7b_flowode_v2', exist_ok=True)

print("=" * 72)
print("FlowODE v2 — time-conditioned vector field on 256-D measured manifold")
print(f"  steps={TOTAL_STEPS}  manifold={MANIFOLD_DIM}  time_dim={TIME_DIM}")
print(f"  patience={PATIENCE}  LR={LR}  WD_f={WD_F}")
print("=" * 72)

# ── Load teacher (same as v1) ──
print("\nLoading teacher...")
wd = torch.load('qwen3_1.7b_cache.pt', weights_only=True)
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
gd['output_norm.weight'] = wd.get('model.norm.weight', torch.ones(2048)).float()
gd['output.weight'] = wd.get('lm_head.weight', gd['token_embd.weight']).float()
for li in range(N_TEACHER_LAYERS):
    for h, g in hf_to_gguf.items():
        k = f'model.layers.{li}.{h}'
        if k in wd:
            gd[f'blk.{li}.{g}'] = wd[k].float()
del wd

hidden = gd['token_embd.weight'].shape[1]
vocab_size = gd['token_embd.weight'].shape[0]
cfg = ModelConfig(n_layers=N_TEACHER_LAYERS, n_heads=16, n_kv_heads=8,
                  hidden_size=hidden, intermediate_size=hidden*3,
                  vocab_size=vocab_size, head_dim=hidden//16)
teacher = MiniTransformer(cfg, DEVICE)
teacher.load_weights(gd)
teacher.embed_weight = teacher.embed_weight.to(DEVICE)
if teacher.lm_head is not None:
    teacher.lm_head = teacher.lm_head.to(DEVICE)
embed_w = gd['token_embd.weight'].to(DEVICE)
norm_w = gd['output_norm.weight'].to(DEVICE)
lm_head_w = gd['output.weight'].to(DEVICE)
del gd

# ── Measured basis ──
print("\nLoading teacher intrinsic structure...")
analysis = torch.load('teacher_intrinsic_analysis.pt', weights_only=False)
basis_128 = analysis['basis']

with torch.no_grad():
    extra = torch.randn(MANIFOLD_DIM - 128, hidden)
    B = basis_128.clone()
    for i in range(extra.shape[0]):
        v = extra[i]
        for b in B:
            v = v - (v @ b) * b
        for p in extra[:i]:
            v = v - (v @ p) * p
        v = v / v.norm().clamp_min(1e-8)
        extra[i] = v
    basis_256 = torch.cat([basis_128, extra], 0)

# Log-interpolated measured norms (27 layers)
tau_init = torch.tensor([21, 21, 17.7, 18, 23.5, 23, 23.4, 25, 37.2, 38, 48, 45, 41.2, 45, 62.5, 80,
                         137, 180, 260, 320, 392, 400, 416, 500, 635, 1200, 3186], dtype=torch.float32)
tau_init = tau_init / tau_init.mean()

# Build sinusoidal time embedding (cached)
def sinusoidal_time_emb(step_idx, n_steps, dim):
    """step_idx: int in [0, n_steps). returns (dim,) tensor."""
    t = step_idx / max(n_steps - 1, 1)       # normalize to [0, 1]
    half = dim // 2
    freqs = torch.arange(half).float() * (2 * math.pi)    # 1x, 2x, 3x, ...
    a = torch.sin(t * freqs)
    b = torch.cos(t * freqs)
    return torch.cat([a, b], 0)


class FlowODEv2(nn.Module):
    def __init__(self, hidden_dim, manifold_dim, n_steps, time_dim, vocab_size,
                 basis, tau_init, embed_w, lm_head_w, norm_w):
        super().__init__()
        self.D = hidden_dim
        self.r = manifold_dim
        self.n_steps = n_steps
        self.time_dim = time_dim

        # P: D -> r, Q: r -> D
        self.P = nn.Parameter(basis.clone())
        self.Q = nn.Parameter(basis.T.clone().contiguous())

        # Time-conditioned vector field: f(z, t) = MLP(concat(z, t))
        self.f = nn.Sequential(
            nn.Linear(manifold_dim + time_dim, manifold_dim * 4),
            nn.GELU(),
            nn.Linear(manifold_dim * 4, manifold_dim),
        )
        with torch.no_grad():
            self.f[-1].weight.mul_(0.01)
            self.f[-1].bias.zero_()

        # Precompute time embeddings (frozen)
        time_embs = torch.stack([sinusoidal_time_emb(i, n_steps, time_dim)
                                 for i in range(n_steps)], 0)     # (n_steps, time_dim)
        self.register_buffer('time_embs', time_embs)

        # log-tau (27 scalars)
        self.log_tau = nn.Parameter(tau_init.log())

        # Pass-through stack
        self.embed = nn.Embedding.from_pretrained(embed_w, freeze=True)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.lm_head.weight = nn.Parameter(lm_head_w, requires_grad=False)
        self.norm = nn.RMSNorm(hidden_dim)
        self.norm.weight = nn.Parameter(norm_w, requires_grad=False)

    def forward(self, tokens, return_trajectory=False):
        B, T = tokens.shape
        x = self.embed(tokens).float()
        z = x @ self.P.T                                      # (B, T, r)

        tau = self.log_tau.exp()
        traj = []
        for step in range(self.n_steps):
            t_emb = self.time_embs[step].unsqueeze(0).unsqueeze(0).expand(B, T, -1)  # (B, T, time_dim)
            zt = torch.cat([z, t_emb], -1)                     # (B, T, r+time_dim)
            dz = self.f(zt)                                    # (B, T, r)
            z = z + tau[step] * dz * 0.01
            if return_trajectory:
                x_step = x + z @ self.Q.T
                traj.append(x_step)

        x_out = x + z @ self.Q.T
        logits = self.lm_head(self.norm(x_out))

        if return_trajectory:
            return logits, traj
        return logits

    def param_groups(self, lr):
        """Return optimizer param groups with different WD."""
        f_params = list(self.f.parameters())
        pq_params = [self.P, self.Q]
        tau_params = [self.log_tau]
        return [
            {'params': f_params, 'lr': lr, 'weight_decay': WD_F},
            {'params': pq_params, 'lr': lr, 'weight_decay': WD_PQ},
            {'params': tau_params, 'lr': lr * 0.3, 'weight_decay': 0.0},
        ]

    def param_count(self):
        return {
            'P': self.P.numel(),
            'Q': self.Q.numel(),
            'f': sum(p.numel() for p in self.f.parameters()),
            'tau': self.log_tau.numel(),
        }

model = FlowODEv2(hidden, MANIFOLD_DIM, N_TEACHER_LAYERS - 1, TIME_DIM, vocab_size,
                  basis_256, tau_init, embed_w, lm_head_w, norm_w).to(DEVICE)
pc = model.param_count()
total = sum(pc.values())
teacher_layer_p = N_TEACHER_LAYERS * (4 * hidden * hidden + 3 * hidden * hidden * 3)
print(f"\nParams:")
for k, v in pc.items():
    print(f"  {k:5s}: {v:>10,}  ({v/1e6:.3f}M)")
print(f"  TOTAL:  {total:>10,}  ({total/1e6:.3f}M)")
print(f"  Compression: {teacher_layer_p/total:.0f}x")

# ── Data ──
print("\nLoading data...")
all_tokens = torch.load('fineweb_edu_100M_tokens.pt', weights_only=True)
data_offset = 0
def get_batch():
    global data_offset
    end = data_offset + BATCH_SIZE * SEQ_LEN
    if end > all_tokens.numel() - 1:
        data_offset = 0; end = BATCH_SIZE * SEQ_LEN
    return all_tokens[data_offset:end].long().reshape(BATCH_SIZE, SEQ_LEN).to(DEVICE), (
        data_offset := end
    )[0]  # shim; globals work

# ── Eval ──
@torch.no_grad()
def eval_all(n=50):
    model.eval(); t1=t10=0; nt=0
    for _ in range(n):
        s = torch.randint(0, all_tokens.numel() - SEQ_LEN, (1,)).item()
        toks = all_tokens[s:s+SEQ_LEN].unsqueeze(0).long().to(DEVICE)
        tl = teacher.forward(toks, max_layers=N_TEACHER_LAYERS)
        sl = model(toks)
        for pos in range(SEQ_LEN):
            t_top = tl[0, pos].topk(10).indices
            s_top = sl[0, pos].topk(10).indices
            t1 += int(s_top[0] == t_top[0])
            t10 += len(set(t_top.tolist()) & set(s_top.tolist())) / 10
            nt += 1
    model.train()
    return t1/nt, t10/nt

@torch.no_grad()
def eval_last(n=200):
    model.eval(); t1=t10=0
    for _ in range(n):
        s = torch.randint(0, all_tokens.numel() - SEQ_LEN, (1,)).item()
        toks = all_tokens[s:s+SEQ_LEN].unsqueeze(0).long().to(DEVICE)
        tl = teacher.forward(toks, max_layers=N_TEACHER_LAYERS)
        sl = model(toks)
        t_top = tl[0, -1].topk(10).indices
        s_top = sl[0, -1].topk(10).indices
        t1 += int(s_top[0] == t_top[0])
        t10 += len(set(t_top.tolist()) & set(s_top.tolist())) / 10
    model.train()
    return t1/n, t10/n

print("\nBaseline:")
t1, t10 = eval_all(50); t1l, t10l = eval_last(100)
print(f"  all-pos T1={t1*100:.1f}% T10={t10*100:.1f}%  last-tok T10={t10l*100:.1f}%")

# ── Training ──
print(f"\n{'='*72}")
print(f"TRAINING: per-layer-weighted hidden MSE, time-conditioned vector field")
print(f"{'='*72}")

opt = torch.optim.AdamW(model.param_groups(LR))
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, TOTAL_STEPS)

# Get batch implementation that actually works
data_ptr = [0]
def batch():
    e = data_ptr[0] + BATCH_SIZE * SEQ_LEN
    if e > all_tokens.numel() - 1:
        data_ptr[0] = 0; e = BATCH_SIZE * SEQ_LEN
    toks = all_tokens[data_ptr[0]:e].long().reshape(BATCH_SIZE, SEQ_LEN).to(DEVICE)
    data_ptr[0] = e
    return toks

best = 0.0; best_step = -1; steps_since_best = 0
t0 = time.time()
loss_hist = []

for step in range(TOTAL_STEPS):
    tokens = batch()
    with torch.no_grad():
        _, t_h = teacher.forward(tokens, max_layers=N_TEACHER_LAYERS, return_hidden=True)
        t_stack = torch.stack(t_h, 0).detach()     # (28, B, T, D)

    _, s_traj = model(tokens, return_trajectory=True)
    s_stack = torch.stack(s_traj, 0)                # (27, B, T, D)
    t_tgt = t_stack[1:1+len(s_traj)]                # (27, B, T, D)

    # Per-layer weighted MSE: w_ℓ = 1 / mean||t_ℓ||  → equalize gradient contribution
    loss = 0.0
    for l in range(t_tgt.shape[0]):
        num = F.mse_loss(s_stack[l], t_tgt[l])
        w_l = 1.0 / (t_tgt[l].detach().pow(2).mean().sqrt().clamp_min(1e-3))
        loss = loss + num * w_l
    loss = loss / t_tgt.shape[0]

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    loss_hist.append(loss.item())

    if step % EVAL_INTERVAL == 0 or step == TOTAL_STEPS - 1:
        t1, t10 = eval_all(50); t1l, t10l = eval_last(200)
        elapsed = time.time() - t0
        avg = sum(loss_hist[-100:]) / min(len(loss_hist), 100)
        mark = ""
        if t10l > best:
            best = t10l; best_step = step; steps_since_best = 0
            mark = " *** NEW BEST ***"
            torch.save(model.state_dict(), 'checkpoints_1.7b_flowode_v2/flowode_v2_best.pt')
        else:
            steps_since_best += EVAL_INTERVAL
        tau_now = model.log_tau.exp().detach().cpu()
        f_norm = model.f[-1].weight.norm().item()
        print(f"  Step {step:>5d}/{TOTAL_STEPS}  loss={avg:.4f}  all-T10={t10*100:.1f}%  "
              f"last-T10={t10l*100:.1f}%  best={best*100:.1f}%@{best_step}  ({elapsed:.0f}s){mark}")
        print(f"    tau=[{tau_now.min():.2f},{tau_now.max():.2f}]  f_out_norm={f_norm:.2f}  "
              f"patience={steps_since_best}/{PATIENCE}")
        if steps_since_best >= PATIENCE:
            print(f"\n  Early stop: no improvement in {PATIENCE} steps")
            break

torch.save(model.state_dict(), 'checkpoints_1.7b_flowode_v2/flowode_v2_final.pt')
print(f"\n{'='*72}")
print(f"v2 DONE  best last-T10 = {best*100:.2f}% @ step {best_step}")
print(f"  v1 peak: 59.9%   record(pure_kl 28x): 68.23%")
print(f"  {total/1e6:.2f}M params ({teacher_layer_p/total:.0f}x compression)")
print(f"  Elapsed: {(time.time()-t0)/60:.1f} min")
print(f"{'='*72}")
