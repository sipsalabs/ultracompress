"""
FlowODE v3 — CAPACITY test. v1/v2/v2+KL all plateau at 59-60% last-T10.
Consistent across different losses → ceiling is capacity, not objective.

v3 scales:
  manifold:  256 -> 512
  f:         1-block MLP (4r) -> 2-block residual MLP (3r each)
  params:    1.6M -> ~6.7M
  compression: 960x -> ~228x  (still 8x more aggressive than pure_kl@28x=68.23%)

Loss: KL directly from start (v2 showed KL works fine cold on a warm init; MSE
adds no value over KL). Temperature 2.0, same as pure_kl.

Question: if 228x gets to ~65%, compression/quality curve is gentler than
expected and ultra-aggressive is feasible. If it caps at 60-61%, FlowODE's
structural prior (residual on fixed basis) is the ceiling.
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
KL_TEMP = 2.0
BATCH_SIZE = 4
SEQ_LEN = 64
MANIFOLD_DIM = 512
TIME_DIM = 32
EVAL_INTERVAL = 500
PATIENCE = 2000
N_TEACHER_LAYERS = 28

os.makedirs('checkpoints_1.7b_flowode_v3', exist_ok=True)

print("=" * 72)
print("FlowODE v3 — CAPACITY test: manifold=512, depth=2, KL from scratch")
print(f"  steps={TOTAL_STEPS}  LR={LR}  T={KL_TEMP}  patience={PATIENCE}")
print("=" * 72)

# Teacher
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

# Basis: 128 measured + 384 orthogonal padding
print("\nLoading basis...")
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
    basis_full = torch.cat([basis_128, extra], 0)    # (512, 2048)

tau_init = torch.tensor([21, 21, 17.7, 18, 23.5, 23, 23.4, 25, 37.2, 38, 48, 45, 41.2, 45, 62.5, 80,
                         137, 180, 260, 320, 392, 400, 416, 500, 635, 1200, 3186], dtype=torch.float32)
tau_init = tau_init / tau_init.mean()

def sinusoidal_time_emb(step_idx, n_steps, dim):
    t = step_idx / max(n_steps - 1, 1)
    half = dim // 2
    freqs = torch.arange(half).float() * (2 * math.pi)
    return torch.cat([torch.sin(t*freqs), torch.cos(t*freqs)], 0)


class ResMLP(nn.Module):
    """2-block residual MLP: z -> z + block1 -> z + block2. Each block is Linear-GELU-Linear."""
    def __init__(self, dim, time_dim, hidden_mult=3):
        super().__init__()
        self.b1_1 = nn.Linear(dim + time_dim, dim * hidden_mult)
        self.b1_2 = nn.Linear(dim * hidden_mult, dim)
        self.b2_1 = nn.Linear(dim + time_dim, dim * hidden_mult)
        self.b2_2 = nn.Linear(dim * hidden_mult, dim)
        with torch.no_grad():
            self.b1_2.weight.mul_(0.01); self.b1_2.bias.zero_()
            self.b2_2.weight.mul_(0.01); self.b2_2.bias.zero_()

    def forward(self, z, t_emb):
        zt = torch.cat([z, t_emb], -1)
        z = z + self.b1_2(F.gelu(self.b1_1(zt)))
        zt = torch.cat([z, t_emb], -1)
        z = z + self.b2_2(F.gelu(self.b2_1(zt)))
        return z


class FlowODEv3(nn.Module):
    def __init__(self, hidden_dim, manifold_dim, n_steps, time_dim, vocab_size,
                 basis, tau_init, embed_w, lm_head_w, norm_w):
        super().__init__()
        self.D = hidden_dim; self.r = manifold_dim; self.n_steps = n_steps
        self.P = nn.Parameter(basis.clone())
        self.Q = nn.Parameter(basis.T.clone().contiguous())
        self.f = ResMLP(manifold_dim, time_dim)
        time_embs = torch.stack([sinusoidal_time_emb(i, n_steps, time_dim)
                                 for i in range(n_steps)], 0)
        self.register_buffer('time_embs', time_embs)
        self.log_tau = nn.Parameter(tau_init.log())
        self.embed = nn.Embedding.from_pretrained(embed_w, freeze=True)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.lm_head.weight = nn.Parameter(lm_head_w, requires_grad=False)
        self.norm = nn.RMSNorm(hidden_dim)
        self.norm.weight = nn.Parameter(norm_w, requires_grad=False)

    def forward(self, tokens):
        B, T = tokens.shape
        x = self.embed(tokens).float()
        z = x @ self.P.T
        tau = self.log_tau.exp()
        for step in range(self.n_steps):
            t_emb = self.time_embs[step].unsqueeze(0).unsqueeze(0).expand(B, T, -1)
            zt = torch.cat([z, t_emb], -1)
            d1 = self.f.b1_2(F.gelu(self.f.b1_1(zt)))
            z1 = z + d1
            zt = torch.cat([z1, t_emb], -1)
            d2 = self.f.b2_2(F.gelu(self.f.b2_1(zt)))
            delta = d1 + d2                              # pure vector field
            z = z + tau[step] * delta * 0.01
        x_out = x + z @ self.Q.T
        return self.lm_head(self.norm(x_out))


model = FlowODEv3(hidden, MANIFOLD_DIM, N_TEACHER_LAYERS - 1, TIME_DIM, vocab_size,
                  basis_full, tau_init, embed_w, lm_head_w, norm_w).to(DEVICE)
pc = {
    'P': model.P.numel(), 'Q': model.Q.numel(),
    'f': sum(p.numel() for p in model.f.parameters()),
    'tau': model.log_tau.numel(),
}
total = sum(pc.values())
teacher_layer_p = N_TEACHER_LAYERS * (4*hidden*hidden + 3*hidden*hidden*3)
print(f"\nParams:")
for k, v in pc.items():
    print(f"  {k:5s}: {v:>12,}  ({v/1e6:.3f}M)")
print(f"  TOTAL:  {total:>12,}  ({total/1e6:.3f}M)")
print(f"  Compression: {teacher_layer_p/total:.0f}x")

# Data
print("\nLoading data...")
all_tokens = torch.load('fineweb_edu_100M_tokens.pt', weights_only=True)
data_ptr = [0]
def batch():
    e = data_ptr[0] + BATCH_SIZE * SEQ_LEN
    if e > all_tokens.numel() - 1:
        data_ptr[0] = 0; e = BATCH_SIZE * SEQ_LEN
    toks = all_tokens[data_ptr[0]:e].long().reshape(BATCH_SIZE, SEQ_LEN).to(DEVICE)
    data_ptr[0] = e
    return toks

@torch.no_grad()
def eval_all(n=50):
    model.eval(); t1=t10=0; nt=0
    for _ in range(n):
        s = torch.randint(0, all_tokens.numel() - SEQ_LEN, (1,)).item()
        toks = all_tokens[s:s+SEQ_LEN].unsqueeze(0).long().to(DEVICE)
        tl = teacher.forward(toks, max_layers=N_TEACHER_LAYERS)
        sl = model(toks)
        for pos in range(SEQ_LEN):
            tt = tl[0, pos].topk(10).indices
            st = sl[0, pos].topk(10).indices
            t1 += int(st[0] == tt[0])
            t10 += len(set(tt.tolist()) & set(st.tolist())) / 10
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
        tt = tl[0, -1].topk(10).indices
        st = sl[0, -1].topk(10).indices
        t1 += int(st[0] == tt[0])
        t10 += len(set(tt.tolist()) & set(st.tolist())) / 10
    model.train()
    return t1/n, t10/n

print("\nBaseline:")
t1, t10 = eval_all(50); t1l, t10l = eval_last(200)
print(f"  all-pos T10={t10*100:.1f}%  last-tok T10={t10l*100:.1f}%")

# Optimizer
f_params = list(model.f.parameters())
pq_params = [model.P, model.Q]
tau_params = [model.log_tau]
opt = torch.optim.AdamW([
    {'params': f_params, 'lr': LR, 'weight_decay': 0.1},
    {'params': pq_params, 'lr': LR, 'weight_decay': 0.0},
    {'params': tau_params, 'lr': LR * 0.3, 'weight_decay': 0.0},
])
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, TOTAL_STEPS, eta_min=LR*0.1)

print(f"\n{'='*72}\nTRAINING: token-KL from scratch\n{'='*72}")
best = 0.0; best_step = -1; since_best = 0
t0 = time.time()
losses = []

for step in range(TOTAL_STEPS):
    toks = batch()
    with torch.no_grad():
        t_logits = teacher.forward(toks, max_layers=N_TEACHER_LAYERS)
        t_soft = F.log_softmax(t_logits / KL_TEMP, -1)
    s_logits = model(toks)
    s_logsoft = F.log_softmax(s_logits / KL_TEMP, -1)
    loss = F.kl_div(s_logsoft, t_soft, reduction='batchmean', log_target=True) * (KL_TEMP ** 2)

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    losses.append(loss.item())

    if step % EVAL_INTERVAL == 0 or step == TOTAL_STEPS - 1:
        t1, t10 = eval_all(50); t1l, t10l = eval_last(200)
        elapsed = time.time() - t0
        avg = sum(losses[-100:]) / min(len(losses), 100)
        mark = ""
        if t10l > best:
            best = t10l; best_step = step; since_best = 0
            mark = " *** NEW BEST ***"
            torch.save(model.state_dict(), 'checkpoints_1.7b_flowode_v3/flowode_v3_best.pt')
        else:
            since_best += EVAL_INTERVAL
        print(f"  Step {step:>5d}/{TOTAL_STEPS}  KL={avg:.4f}  all-T10={t10*100:.1f}%  "
              f"last-T10={t10l*100:.1f}%  best={best*100:.1f}%@{best_step}  ({elapsed:.0f}s){mark}")
        if since_best >= PATIENCE:
            print(f"  Early stop @ step {step}")
            break

print(f"\n{'='*72}")
print(f"v3 DONE  best last-T10 = {best*100:.2f}% @ step {best_step}")
print(f"  v1=59.9%  v2-MSE=59.05%  v2+KL=59.85%  pure_kl@28x=68.23%")
print(f"  {total/1e6:.2f}M params ({teacher_layer_p/total:.0f}x compression)")
print(f"  Elapsed: {(time.time()-t0)/60:.1f} min")
print(f"{'='*72}")
