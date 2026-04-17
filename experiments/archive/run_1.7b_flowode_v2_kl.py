"""
FlowODE v2 Phase 2 — fine-tune best MSE checkpoint with token KL.

Hypothesis: MSE-trained FlowODE plateau'd at ~59% in both v1 and v2 despite
very different tricks. But pure_kl got 68% at 28x compression. The difference
may just be the LOSS: hidden MSE cares about reconstructing every feature
equally, token KL cares only about what matters for the final distribution.

Load flowode_v2_best.pt (59.05% last-T10, MSE-trained), switch to token KL,
see if we can break 60%.

If FlowODE + KL -> 60-65%: proves the 960x architecture has more room; MSE
was just the wrong objective. That's a real compression breakthrough.

If it stays flat or regresses: 59% is the intrinsic capacity of 1.57M
params at 960x and we need more params.
"""
import lib.unbuffered
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ultracompress.inference import ModelConfig, MiniTransformer

DEVICE = 'cuda:0'
TOTAL_STEPS = 5_000
LR = 5e-5            # lower — fine-tuning, not fresh
KL_TEMP = 2.0        # soft target
BATCH_SIZE = 4
SEQ_LEN = 64
MANIFOLD_DIM = 256
TIME_DIM = 16
EVAL_INTERVAL = 500
PATIENCE = 1500
N_TEACHER_LAYERS = 28

os.makedirs('checkpoints_1.7b_flowode_v2', exist_ok=True)

print("=" * 72)
print("FlowODE v2 PHASE 2 — token-KL fine-tune on best MSE checkpoint")
print(f"  loaded: flowode_v2_best.pt (MSE phase: 59.05%)")
print(f"  steps={TOTAL_STEPS}  LR={LR}  T={KL_TEMP}")
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

# Model (same as v2)
def sinusoidal_time_emb(step_idx, n_steps, dim):
    t = step_idx / max(n_steps - 1, 1)
    half = dim // 2
    freqs = torch.arange(half).float() * (2 * math.pi)
    return torch.cat([torch.sin(t*freqs), torch.cos(t*freqs)], 0)


class FlowODEv2(nn.Module):
    def __init__(self, hidden_dim, manifold_dim, n_steps, time_dim, vocab_size,
                 embed_w, lm_head_w, norm_w):
        super().__init__()
        self.D = hidden_dim; self.r = manifold_dim; self.n_steps = n_steps
        self.P = nn.Parameter(torch.zeros(manifold_dim, hidden_dim))
        self.Q = nn.Parameter(torch.zeros(hidden_dim, manifold_dim))
        self.f = nn.Sequential(
            nn.Linear(manifold_dim + time_dim, manifold_dim * 4),
            nn.GELU(),
            nn.Linear(manifold_dim * 4, manifold_dim),
        )
        time_embs = torch.stack([sinusoidal_time_emb(i, n_steps, time_dim)
                                 for i in range(n_steps)], 0)
        self.register_buffer('time_embs', time_embs)
        self.log_tau = nn.Parameter(torch.zeros(n_steps))
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
            z = z + tau[step] * self.f(zt) * 0.01
        x_out = x + z @ self.Q.T
        return self.lm_head(self.norm(x_out))


model = FlowODEv2(hidden, MANIFOLD_DIM, N_TEACHER_LAYERS - 1, TIME_DIM, vocab_size,
                  embed_w, lm_head_w, norm_w).to(DEVICE)

print("\nLoading MSE-trained best checkpoint...")
sd = torch.load('checkpoints_1.7b_flowode_v2/flowode_v2_best.pt', weights_only=True, map_location=DEVICE)
model.load_state_dict(sd, strict=True)
print("  loaded.")

total = sum(p.numel() for p in [model.P, model.Q, *model.f.parameters(), model.log_tau])
teacher_layer_p = N_TEACHER_LAYERS * (4 * hidden * hidden + 3 * hidden * hidden * 3)
print(f"  trainable = {total:,} ({total/1e6:.3f}M)  compression={teacher_layer_p/total:.0f}x")

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

print("\nBaseline (MSE checkpoint):")
t1, t10 = eval_all(50); t1l, t10l = eval_last(200)
print(f"  all-pos T10={t10*100:.1f}%  last-tok T10={t10l*100:.1f}%")
mse_baseline = t10l

# Trainable groups
f_params = list(model.f.parameters())
pq_params = [model.P, model.Q]
tau_params = [model.log_tau]
opt = torch.optim.AdamW([
    {'params': f_params, 'lr': LR, 'weight_decay': 0.01},
    {'params': pq_params, 'lr': LR, 'weight_decay': 0.0},
    {'params': tau_params, 'lr': LR * 0.3, 'weight_decay': 0.0},
])
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, TOTAL_STEPS, eta_min=LR*0.1)

print(f"\n{'='*72}\nPHASE 2 — KL fine-tune\n{'='*72}")
best = mse_baseline; best_step = -1; since_best = 0
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
            torch.save(model.state_dict(), 'checkpoints_1.7b_flowode_v2/flowode_v2_kl_best.pt')
        else:
            since_best += EVAL_INTERVAL
        print(f"  Step {step:>5d}/{TOTAL_STEPS}  KL={avg:.4f}  all-T10={t10*100:.1f}%  "
              f"last-T10={t10l*100:.1f}%  best={best*100:.1f}%@{best_step}  ({elapsed:.0f}s){mark}")
        if since_best >= PATIENCE:
            print(f"  Early stop @ step {step}")
            break

print(f"\n{'='*72}")
print(f"PHASE 2 DONE  MSE baseline={mse_baseline*100:.2f}%  ->  KL best={best*100:.2f}%  (+{(best-mse_baseline)*100:+.2f}pp)")
print(f"  records: FlowODE v1=59.9%  v2-MSE=59.05%  pure_kl@28x=68.23%")
print(f"  {total/1e6:.2f}M params ({teacher_layer_p/total:.0f}x compression)")
print(f"{'='*72}")
