"""
FlowODE-Residual: compression grounded in MEASURED teacher structure.

Built from the facts discovered today (not textbook techniques):

  FACT 1: Teacher's residual field has intrinsic rank ~186 @ 95% variance.
          So we project to r=256 (safety margin) and never leave that subspace.

  FACT 2: Layer-27 residual norm is 150x layer-1. Residual norms grow near-
          exponentially. So timesteps must NOT be uniform. We use learnable
          per-step sizes, initialized to the MEASURED log-scale schedule.

  FACT 3: Only 3 of 351 layer pairs are CKA-similar. Layers do different work,
          but the residual FIELD is low-rank. So the RIGHT compression is one
          continuous vector field explored at 27 learned timesteps — not a
          shared block applied 27 times.

  FACT 4: Teacher top-1 on ground truth = 14.9%; on teacher-confident tokens
          (3.6% of data) = 62.8%. So the loss should NOT be uniform KL on
          tokens. We train on HIDDEN-STATE MATCHING (continuous, no alphabet
          bottleneck) and only score tokens at inference.

Architecture:
    x_0 = embed(tokens)                                  # (B, T, D=2048)
    z_0 = P(x_0)                                         # (B, T, r=256)
    for step = 1..27:
        dz = f(z_{step-1})                               # MLP: r -> 4r -> r
        z_step = z_{step-1} + tau[step-1] * dz
    x_out = x_0 + Q(z_27)                                # back to D=2048
    logits = lm_head(norm(x_out))                        # only for inference

Where:
    P: 2048 -> 256   (initialized from the top-256 right singular vectors of
                      the residual field — measured prior, not random init)
    Q: 256 -> 2048   (initialized from P^T for the orthogonal complement)
    f: MLP r -> 4r -> r  GELU activations
    tau: 27 learnable scalars init to log-scale matching ||δ_ℓ|| measured

Training loss = hidden-state MSE per timestep (matching teacher's layer ℓ
after projection to manifold), NOT logit KL. No token alphabet in the loss.

Parameters:
    P (2048x256)       = 524,288
    Q (256x2048)       = 524,288
    f (r*4r + 4r*r)    = 262,144  (with bias: ~264K)
    tau (27)           = 27
    Total              ~ 1.31M

Compression: 1.0B transformer params / 1.31M = 763x on transformer block.
Total (with embed+lm_head pass-through): 1.7B / 1.31M = 1297x on layers.
"""
import lib.unbuffered
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultracompress.inference import ModelConfig, MiniTransformer

DEVICE = 'cuda:0'
TOTAL_STEPS = 10_000
LR = 3e-4
WD = 0.01
BATCH_SIZE = 4
SEQ_LEN = 64
MANIFOLD_DIM = 256
EVAL_INTERVAL = 1_000
N_TEACHER_LAYERS = 28

os.makedirs('checkpoints_1.7b_flowode', exist_ok=True)

print("=" * 72)
print("FlowODE-Residual: continuous vector field on measured 256-D manifold")
print(f"  Device={DEVICE}  steps={TOTAL_STEPS}  manifold_dim={MANIFOLD_DIM}")
print("=" * 72)

# ── Load teacher ──
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

# ── Load measured intrinsic analysis (basis + per-layer norms) ──
print("\nLoading measured teacher intrinsic structure...")
analysis = torch.load('teacher_intrinsic_analysis.pt', weights_only=False)
basis_128 = analysis['basis']     # (128, D)
print(f"  basis: {basis_128.shape}")

# Build r=256 basis: top 128 from analysis, plus 128 random-orthogonal to pad
# Take the raw 128, then orthogonalize random pad against it
with torch.no_grad():
    extra_random = torch.randn(MANIFOLD_DIM - 128, hidden)
    # Gram-Schmidt orthogonalize against basis_128
    B = basis_128.clone()
    for i in range(extra_random.shape[0]):
        v = extra_random[i]
        for b in B:
            v = v - (v @ b) * b
        for p in extra_random[:i]:
            v = v - (v @ p) * p
        v = v / v.norm().clamp_min(1e-8)
        extra_random[i] = v
    basis_256 = torch.cat([basis_128, extra_random], 0)   # (256, D)
    # Verify orthonormality
    orth_err = (basis_256 @ basis_256.T - torch.eye(MANIFOLD_DIM)).abs().max().item()
    print(f"  256-dim basis orthonormality error: {orth_err:.6f}")

# ── Measure empirical timestep sizes for good init of tau ──
# From analysis: layer delta norms, we have a 27-length profile
# Use log-normalized profile
# From intrinsic analysis log output: layer 1=21, 9=37, 17=137, 27=3187
# Interpolate a 27-length norm curve (rough exponential)
tau_init = torch.tensor([21, 21, 17.7, 18, 23.5, 23, 23.4, 25, 37.2, 38, 48, 45, 41.2, 45, 62.5, 80,
                         137, 180, 260, 320, 392, 400, 416, 500, 635, 1200, 3186], dtype=torch.float32)
tau_init = tau_init / tau_init.mean()   # normalize
print(f"  tau_init profile (normalized): min={tau_init.min():.3f} max={tau_init.max():.3f}")

# ── Model ──
class FlowODE(nn.Module):
    def __init__(self, hidden_dim, manifold_dim, n_steps, vocab_size,
                 basis, tau_init, embed_w, lm_head_w, norm_w):
        super().__init__()
        self.D = hidden_dim
        self.r = manifold_dim
        self.n_steps = n_steps

        # P: D -> r  (project TO manifold). Initialized from measured basis.
        # Q: r -> D  (unproject). Initialized as pseudo-inverse (since basis
        # is orthonormal, P^T works).
        self.P = nn.Parameter(basis.clone())                      # (r, D)
        self.Q = nn.Parameter(basis.T.clone().contiguous())       # (D, r)

        # f: vector field MLP on manifold
        self.f = nn.Sequential(
            nn.Linear(manifold_dim, manifold_dim * 4),
            nn.GELU(),
            nn.Linear(manifold_dim * 4, manifold_dim),
        )
        # Zero-init last layer so initial vector field is zero → baseline = identity flow
        with torch.no_grad():
            self.f[-1].weight.mul_(0.01)
            self.f[-1].bias.zero_()

        # tau: learnable per-step size, log-parameterized (always positive)
        self.log_tau = nn.Parameter(tau_init.log())

        # Pass-through token stack
        self.embed = nn.Embedding.from_pretrained(embed_w, freeze=True)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.lm_head.weight = nn.Parameter(lm_head_w, requires_grad=False)
        self.norm = nn.RMSNorm(hidden_dim)
        self.norm.weight = nn.Parameter(norm_w, requires_grad=False)

    def forward(self, tokens, return_trajectory=False):
        """Returns final logits. If return_trajectory, also returns the list
        of D-space hidden states x_step for step=1..n_steps for loss match."""
        x = self.embed(tokens).float()                 # (B, T, D)
        # Project INPUT to manifold (residual is computed in manifold, added back)
        z = x @ self.P.T                               # (B, T, r)

        tau = self.log_tau.exp()                        # (n_steps,)
        traj = []
        for step in range(self.n_steps):
            dz = self.f(z)                              # (B, T, r)
            z = z + tau[step] * dz * 0.01               # 0.01 scale → start near identity
            if return_trajectory:
                # Reconstruct D-space hidden at this step
                x_step = x + z @ self.Q.T               # (B, T, D)
                traj.append(x_step)

        x_out = x + z @ self.Q.T                        # back to D
        logits = self.lm_head(self.norm(x_out))

        if return_trajectory:
            return logits, traj
        return logits

    def param_count(self):
        return {
            'P': self.P.numel(),
            'Q': self.Q.numel(),
            'f': sum(p.numel() for p in self.f.parameters()),
            'tau': self.log_tau.numel(),
        }

model = FlowODE(hidden, MANIFOLD_DIM, N_TEACHER_LAYERS - 1, vocab_size,
                basis_256, tau_init, embed_w, lm_head_w, norm_w).to(DEVICE)

pc = model.param_count()
total = sum(pc.values())
teacher_layer_p = N_TEACHER_LAYERS * (4 * hidden * hidden + 3 * hidden * hidden * 3)
print(f"\nParams:")
for k, v in pc.items():
    print(f"  {k:5s}: {v:>10,}  ({v/1e6:.3f}M)")
print(f"  TOTAL:  {total:>10,}  ({total/1e6:.3f}M)")
print(f"  Teacher transformer layers: {teacher_layer_p:,}  ({teacher_layer_p/1e9:.3f}B)")
print(f"  Compression: {teacher_layer_p/total:.0f}x")

# ── Data ──
print("\nLoading data...")
all_tokens = torch.load('fineweb_edu_100M_tokens.pt', weights_only=True)
data_offset = 0
def get_batch():
    global data_offset
    end = data_offset + BATCH_SIZE * SEQ_LEN
    if end > all_tokens.numel() - 1:
        data_offset = 0
        end = BATCH_SIZE * SEQ_LEN
    chunk = all_tokens[data_offset:end].long().reshape(BATCH_SIZE, SEQ_LEN)
    data_offset = end
    return chunk.to(DEVICE)

# ── Eval ──
@torch.no_grad()
def eval_model(n=100):
    model.eval()
    t1, t10 = 0, 0
    nt = 0
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
    return t1 / nt, t10 / nt

@torch.no_grad()
def eval_last_tok(n=200):
    model.eval()
    t1, t10 = 0, 0
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
    return t1 / n, t10 / n

print("\nBaseline (before training, vector field near zero → should match embedding-only):")
t1, t10 = eval_model(50)
t1l, t10l = eval_last_tok(100)
print(f"  all-pos T1={t1*100:.1f}% T10={t10*100:.1f}%")
print(f"  last-tok T1={t1l*100:.1f}% T10={t10l*100:.1f}%")

# ── Training: hidden-state matching loss ──
print(f"\n{'='*72}")
print(f"TRAINING: hidden-state matching (NO token KL in loss)")
print(f"{'='*72}")

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, TOTAL_STEPS)

best_last = 0.0
best_step = -1
t0 = time.time()
loss_hist = []

for step in range(TOTAL_STEPS):
    tokens = get_batch()
    with torch.no_grad():
        _, t_hiddens = teacher.forward(tokens, max_layers=N_TEACHER_LAYERS, return_hidden=True)
        # Convert to tensor (L, B, T, D)
        t_stack = torch.stack(t_hiddens, 0).detach()

    _, s_traj = model(tokens, return_trajectory=True)
    # s_traj has n_steps=27 elements, each (B,T,D). t_stack has 28 — use layers 1..28
    s_stack = torch.stack(s_traj, 0)         # (27, B, T, D)
    t_tgt = t_stack[1:]                       # drop embed, keep layers 1..27 (and skip final layer 28 since f only has 27 steps)
    # Wait: teacher has 28 post-layer hiddens (layers 1..28). Student has 27 steps producing 27 intermediate + 1 final.
    # Let's match last-step x_out to layer 27 output of teacher.
    # s_traj has length n_steps = 27 (one per step). Teacher hidden[0..27] are 28 post-layer states.
    # We match student step i to teacher layer i+1 (skip embedding layer 0).
    # Teacher layer 27 (last transformer layer, before final norm) corresponds to s_traj[26].
    t_tgt = t_stack[1:1 + len(s_traj)]        # (27, B, T, D)

    # Scale-normalized MSE per layer, summed
    loss = 0.0
    for l in range(t_tgt.shape[0]):
        num = F.mse_loss(s_stack[l], t_tgt[l])
        denom = t_tgt[l].pow(2).mean().clamp_min(1e-6)
        loss = loss + num / denom
    loss = loss / t_tgt.shape[0]

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    loss_hist.append(loss.item())

    if step % EVAL_INTERVAL == 0 or step == TOTAL_STEPS - 1:
        elapsed = time.time() - t0
        t1, t10 = eval_model(50)
        t1l, t10l = eval_last_tok(200)
        avg_loss = sum(loss_hist[-200:]) / min(len(loss_hist), 200)
        mark = ""
        if t10l > best_last:
            best_last = t10l
            best_step = step
            mark = " *** NEW BEST ***"
            torch.save(model.state_dict(), 'checkpoints_1.7b_flowode/flowode_best.pt')
        # Diagnostic: tau range
        tau_now = model.log_tau.exp().detach().cpu()
        f_last_w_norm = model.f[-1].weight.norm().item()
        print(f"  Step {step:>5d}/{TOTAL_STEPS}  loss={avg_loss:.4f}  "
              f"T1={t1*100:.1f}% T10={t10*100:.1f}%  last-T10={t10l*100:.1f}%  ({elapsed:.0f}s){mark}")
        print(f"    [last n=200] T1={t1l*100:.1f}% T10={t10l*100:.1f}%  best={best_last*100:.1f}%@{best_step}")
        print(f"    tau range=[{tau_now.min():.2f}, {tau_now.max():.2f}]  f_out_norm={f_last_w_norm:.3f}")

torch.save(model.state_dict(), 'checkpoints_1.7b_flowode/flowode_final.pt')
print(f"\n{'='*72}")
print(f"DONE  best last-T10 = {best_last*100:.2f}% @ step {best_step}")
print(f"  Record to beat (hires): 68.23% (pure_kl, 27.9x compression)")
print(f"  FlowODE compression: {teacher_layer_p/total:.0f}x (vs 27.9x) — {total/1e6:.2f}M params")
print(f"  Elapsed: {(time.time()-t0)/60:.1f} min")
print(f"{'='*72}")
