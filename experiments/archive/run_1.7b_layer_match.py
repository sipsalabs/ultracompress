"""
1.7B LAYER-MATCHING DISTILLATION (Pure KL + intermediate MSE)

HYPOTHESIS: Previous experiments only supervise FINAL logits (KL on output).
The FRR has 28 iterations that SHOULD match 28 teacher layer outputs, but
we've never directly supervised that. Adding per-layer hidden-state matching
provides 28x more supervision signal and should force the block to learn
layer-general computation.

LOSS: alpha * KL(output) + (1-alpha) * mean(MSE(frr_iter_i, teacher_layer_i))

Teacher returns hidden_states list of 28 tensors from return_hidden=True.
FRR captures x after each of 28 block invocations.

NEW TERM: layer-matching MSE normalized by activation scale (cosine-like).
"""
import lib.unbuffered
import torch
import sys
import os
import time

import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from ultracompress.inference import ModelConfig, MiniTransformer
from ultracompress.moonshot import FractalBlock

# ── Configuration ─────────────────────────────────────────────────────
DEVICE = 'cuda:1'
TOTAL_STEPS = 15_000
LR_BLOCK = 3e-5
LR_MOD = 3e-4
WD_BLOCK = 0.05
WD_MOD = 0.01
BATCH_SIZE = 4
SEQ_LEN = 64
TEMP = 2.0
ALPHA_KL = 0.5  # weight on output KL vs layer matching MSE
EVAL_INTERVAL = 2_500
CHECKPOINT_INTERVAL = 5_000
RESUME_FROM = 'checkpoints_1.7b_real_text/frr_1.7b_100k_final.pt'
CHECKPOINT_DIR = 'checkpoints_1.7b_layer_match'
N_TEACHER_LAYERS = 28

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("=" * 70)
print("1.7B LAYER-MATCHING DISTILLATION")
print(f"Device: {DEVICE}  |  Alpha_KL: {ALPHA_KL}  |  Steps: {TOTAL_STEPS:,}")
print(f"Loss: {ALPHA_KL}*KL(output) + {1-ALPHA_KL}*MSE(hidden states per layer)")
print("=" * 70)


class LayerMatchFRR(nn.Module):
    """FRR with per-iteration mod + hidden state capture."""
    def __init__(self, hidden_dim, n_heads, n_scales, iters_per_scale,
                 vocab_size, ff_mult,
                 embed_weight=None, lm_head_weight=None, norm_weight=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_scales = n_scales
        self.iters_per_scale = iters_per_scale
        self.total_layers = n_scales * iters_per_scale

        self.block = FractalBlock(hidden_dim, n_heads, ff_mult)
        self.layer_gamma = nn.Parameter(torch.ones(self.total_layers, hidden_dim))
        self.layer_beta = nn.Parameter(torch.zeros(self.total_layers, hidden_dim))
        self.iter_scale = nn.Parameter(torch.ones(n_scales, iters_per_scale))

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

    def forward(self, tokens, return_hidden=False):
        x = self.embed(tokens).float()
        hiddens = [] if return_hidden else None
        layer_idx = 0
        for scale in range(self.n_scales):
            for it in range(self.iters_per_scale):
                gamma = self.layer_gamma[layer_idx]
                beta = self.layer_beta[layer_idx]
                iter_s = self.iter_scale[scale, it]
                block_out = self.block(x, gamma, beta)
                x = x + (block_out - x) * iter_s
                if return_hidden:
                    hiddens.append(x)
                layer_idx += 1
        x_norm = self.norm(x)
        logits = self.lm_head(x_norm)
        if return_hidden:
            return logits, hiddens
        return logits


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
def get_real_batch():
    global data_offset
    end = data_offset + BATCH_SIZE * SEQ_LEN
    if end > N_TOKENS:
        data_offset = 0
        end = BATCH_SIZE * SEQ_LEN
    chunk = all_tokens[data_offset:end].long().reshape(BATCH_SIZE, SEQ_LEN)
    data_offset = end
    return chunk.to(DEVICE)


# ── Model ─────────────────────────────────────────────────────────────
print(f"\nBuilding FRR...")
model = LayerMatchFRR(
    hidden_dim=hidden, n_heads=n_heads, n_scales=4, iters_per_scale=7,
    vocab_size=vocab_size, ff_mult=1,
    embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w,
).to(DEVICE)

print(f"Loading base: {RESUME_FROM}")
ckpt = torch.load(RESUME_FROM, map_location=DEVICE, weights_only=True)
block_state = {k.replace('block.', ''): v for k, v in ckpt.items() if k.startswith('block.')}
model.block.load_state_dict(block_state)
scale_gamma = ckpt['scale_gamma']
scale_beta = ckpt['scale_beta']
for scale in range(4):
    for it in range(7):
        layer_idx = scale * 7 + it
        model.layer_gamma.data[layer_idx] = scale_gamma[scale]
        model.layer_beta.data[layer_idx] = scale_beta[scale]
model.iter_scale.data.copy_(ckpt['iter_scale'])

n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Trainable: {n_trainable:,}, Compression: {N_TEACHER_LAYERS * 7 * hidden * hidden / n_trainable:.1f}x")


# ── Eval ──────────────────────────────────────────────────────────────
@torch.no_grad()
def eval_vs_teacher(mdl, n=50):
    mdl.eval()
    t1_total, t10_total, n_tokens = 0, 0, 0
    for _ in range(n):
        starts = torch.randint(0, all_tokens.numel() - SEQ_LEN, (1,))
        tokens = all_tokens[starts[0]:starts[0] + SEQ_LEN].unsqueeze(0).long().to(DEVICE)
        tl = teacher.forward(tokens, max_layers=N_TEACHER_LAYERS)
        sl = mdl(tokens)
        for pos in range(SEQ_LEN):
            t_top = tl[0, pos].topk(10).indices
            s_top = sl[0, pos].topk(10).indices
            t1_total += int(s_top[0] == t_top[0])
            t10_total += len(set(t_top.tolist()) & set(s_top.tolist())) / 10
            n_tokens += 1
    mdl.train()
    return t1_total / n_tokens, t10_total / n_tokens


@torch.no_grad()
def eval_vs_teacher_last(mdl, n=200):
    mdl.eval()
    t1_hits, t10_hits = 0, 0
    for _ in range(n):
        starts = torch.randint(0, all_tokens.numel() - SEQ_LEN, (1,))
        tokens = all_tokens[starts[0]:starts[0] + SEQ_LEN].unsqueeze(0).long().to(DEVICE)
        tl = teacher.forward(tokens, max_layers=N_TEACHER_LAYERS)
        sl = mdl(tokens)
        t_top = tl[0, -1].topk(10).indices
        s_top = sl[0, -1].topk(10).indices
        t1_hits += int(s_top[0] == t_top[0])
        t10_hits += len(set(t_top.tolist()) & set(s_top.tolist())) / 10
    mdl.train()
    return t1_hits / n, t10_hits / n


print("\nBaseline:")
t1_base, t10_base = eval_vs_teacher(model, n=50)
t1_last, t10_last = eval_vs_teacher_last(model, n=200)
print(f"  all-pos:  T1={t1_base*100:.1f}%  T10={t10_base*100:.1f}%")
print(f"  last-tok: T1={t1_last*100:.1f}%  T10={t10_last*100:.1f}% (n=200)")


# ── Training ──────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"TRAINING: LAYER-MATCH ({TOTAL_STEPS:,} steps)")
print(f"{'='*70}")

block_params_list = list(model.block.parameters())
mod_params_list = [model.layer_gamma, model.layer_beta, model.iter_scale]
opt = torch.optim.AdamW([
    {'params': block_params_list, 'lr': LR_BLOCK, 'weight_decay': WD_BLOCK},
    {'params': mod_params_list, 'lr': LR_MOD, 'weight_decay': WD_MOD},
])
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, TOTAL_STEPS)

loss_history = []
kl_history = []
mse_history = []
best_t10 = 0.0
best_t10_last = 0.0
best_step = 0
t0 = time.time()

for step in range(TOTAL_STEPS):
    tokens = get_real_batch()
    with torch.no_grad():
        # Teacher with hidden states
        t_logits, t_hidden = teacher.forward(
            tokens, max_layers=N_TEACHER_LAYERS, return_hidden=True
        )
        # Stack hidden states (28, B, T, H)
        t_hidden_stack = torch.stack(t_hidden, dim=0)  # (28, B, T, H)

    s_logits, s_hidden = model(tokens, return_hidden=True)
    s_hidden_stack = torch.stack(s_hidden, dim=0)  # (28, B, T, H)

    # Output KL loss
    kl_loss = F.kl_div(
        F.log_softmax(s_logits / TEMP, dim=-1),
        F.softmax(t_logits / TEMP, dim=-1),
        reduction='batchmean',
    ) * TEMP * TEMP

    # Layer-match MSE (normalized by teacher norm for scale invariance)
    # t_hidden_stack: (28, B, T, H)
    # Per-layer MSE / norm²
    diff = (s_hidden_stack - t_hidden_stack).pow(2).mean(dim=(1, 2, 3))  # (28,)
    t_norm_sq = t_hidden_stack.pow(2).mean(dim=(1, 2, 3)) + 1e-8  # (28,)
    norm_mse = (diff / t_norm_sq).mean()  # scalar, scale-invariant

    # Combined loss
    loss = ALPHA_KL * kl_loss + (1 - ALPHA_KL) * 100.0 * norm_mse
    # The 100.0 rescales MSE to match KL magnitude (tunable)

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    loss_history.append(loss.item())
    kl_history.append(kl_loss.item())
    mse_history.append(norm_mse.item())

    if step % EVAL_INTERVAL == 0 or step == TOTAL_STEPS - 1:
        t1, t10 = eval_vs_teacher(model, n=50)
        t1_l, t10_l = eval_vs_teacher_last(model, n=200)
        elapsed = time.time() - t0
        new_best = ""
        if t10 > best_t10:
            best_t10 = t10
            best_step = step
            new_best = " *** NEW BEST (all-pos) ***"
            torch.save(model.state_dict(),
                       os.path.join(CHECKPOINT_DIR, 'layer_match_best.pt'))
        if t10_l > best_t10_last:
            best_t10_last = t10_l

        avg_loss = sum(loss_history[-500:]) / min(len(loss_history), 500)
        avg_kl = sum(kl_history[-500:]) / min(len(kl_history), 500)
        avg_mse = sum(mse_history[-500:]) / min(len(mse_history), 500)

        with torch.no_grad():
            gamma_std = model.layer_gamma.std(dim=0).mean().item()
            beta_std = model.layer_beta.std(dim=0).mean().item()
            # Per-layer MSE diagnostic (pick 4 layers)
            per_layer = (s_hidden_stack.detach() - t_hidden_stack).pow(2).mean(dim=(1,2,3))
            per_layer_norm = t_hidden_stack.pow(2).mean(dim=(1,2,3)) + 1e-8
            rel = (per_layer / per_layer_norm).cpu().tolist()

        print(
            f"  Step {step:>6d}/{TOTAL_STEPS}: loss={avg_loss:.4f} "
            f"(KL={avg_kl:.2f} MSE={avg_mse:.4f}) "
            f"T1={t1*100:.1f}% T10={t10*100:.1f}% ({elapsed:.0f}s){new_best}"
        )
        print(
            f"    [last-tok n=200] T1={t1_l*100:.1f}% T10={t10_l*100:.1f}% "
            f"best_last={best_t10_last*100:.1f}%"
        )
        print(
            f"    per-layer rel MSE (L0,L7,L14,L21,L27): "
            f"{rel[0]:.3f} {rel[7]:.3f} {rel[14]:.3f} {rel[21]:.3f} {rel[27]:.3f}"
        )

    if step > 0 and step % CHECKPOINT_INTERVAL == 0:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f'layer_match_step{step}.pt')
        torch.save(model.state_dict(), ckpt_path)
        print(f"  >> Saved: {ckpt_path}")

final_path = os.path.join(CHECKPOINT_DIR, 'layer_match_final.pt')
torch.save(model.state_dict(), final_path)
total_time = time.time() - t0
print(f"\n{'='*70}")
print(f"LAYER-MATCH COMPLETE")
print(f"  Best T10 (all-pos): {best_t10*100:.1f}% at step {best_step}")
print(f"  Best T10 (last-tok, n=200): {best_t10_last*100:.1f}%")
print(f"  Total time: {total_time:.0f}s ({total_time/3600:.1f}h)")
print(f"{'='*70}")
