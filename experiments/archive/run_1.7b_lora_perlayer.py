"""
1.7B PER-LAYER LoRA ADAPTERS (Pure KL)

HYPOTHESIS: 1 block × 28 iterations hits ceiling at T10=62.9% hires. Dual-block
adds 29M params, gets +0.1pp (noise). This proves the problem isn't RAW capacity
but WEIGHT SHARING rigidity. Block parameters must simultaneously serve all 28
teacher layers' semantics.

SOLUTION: Add tiny rank-4 LoRA adapters per iteration on top of the shared block.
28 layers × rank-4 × 2048 × 2 (down+up) = 458,752 adapter params.
Each iteration: block_out = block(x, γ_i, β_i) + α_i * LoRA_i(x)
where LoRA_i = W_up_i @ W_down_i, W_up initialized to zero (identity init).

Compression: 28 × 2048² × 3 / (29.4M + 458K + 114K) ≈ 12.2x
Trainable: 30.0M (vs 29.5M single-block) — only 1.5% more params

This is the minimal change that:
  1. Breaks weight-sharing cleanly (each iter gets unique transform)
  2. Starts at identity (W_up=0) so init matches pre-trained
  3. Forces the adapter to learn ONLY what the block can't share
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
from ultracompress.moonshot import FractalBlock

# ── Config ────────────────────────────────────────────────────────────
DEVICE = 'cuda:1'
TOTAL_STEPS = 15_000
LR_BLOCK = 3e-5
LR_MOD = 3e-4
LR_LORA = 5e-4          # higher LR for fresh LoRAs
WD_BLOCK = 0.05
WD_MOD = 0.01
WD_LORA = 0.0           # no decay on new adapters
BATCH_SIZE = 4
SEQ_LEN = 64
TEMP = 2.0
LORA_RANK = 4
EVAL_INTERVAL = 2_500
CHECKPOINT_INTERVAL = 5_000
RESUME_FROM = 'checkpoints_1.7b_real_text/frr_1.7b_100k_final.pt'
CHECKPOINT_DIR = 'checkpoints_1.7b_lora_perlayer'
N_TEACHER_LAYERS = 28

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("=" * 70)
print("1.7B PER-LAYER LoRA (rank-{}) + Pure KL".format(LORA_RANK))
print(f"Device: {DEVICE}  |  Steps: {TOTAL_STEPS:,}")
print("=" * 70)


class PerLayerLoRAFRR(nn.Module):
    """FRR with per-iteration rank-r LoRA adapters."""
    def __init__(self, hidden_dim, n_heads, n_scales, iters_per_scale,
                 vocab_size, ff_mult, lora_rank=4,
                 embed_weight=None, lm_head_weight=None, norm_weight=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_scales = n_scales
        self.iters_per_scale = iters_per_scale
        self.total_layers = n_scales * iters_per_scale
        self.lora_rank = lora_rank

        self.block = FractalBlock(hidden_dim, n_heads, ff_mult)

        # Per-layer gamma/beta
        self.layer_gamma = nn.Parameter(torch.ones(self.total_layers, hidden_dim))
        self.layer_beta = nn.Parameter(torch.zeros(self.total_layers, hidden_dim))
        self.iter_scale = nn.Parameter(torch.ones(n_scales, iters_per_scale))

        # Per-layer LoRA: [L, H, r] and [L, r, H]
        # Standard LoRA init: down ~ Kaiming small, up = 0
        # W_up=0 means forward output=0 but gradient for W_up = (x @ W_down).T @ d_out
        # which is NONZERO since W_down is random. W_up learns freely.
        self.lora_down = nn.Parameter(
            torch.randn(self.total_layers, hidden_dim, lora_rank) / math.sqrt(hidden_dim)
        )
        self.lora_up = nn.Parameter(torch.zeros(self.total_layers, lora_rank, hidden_dim))

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

                # Shared block
                block_out = self.block(x, gamma, beta)

                # Per-layer LoRA contribution: x @ W_down @ W_up
                Wd = self.lora_down[layer_idx]        # (H, r)
                Wu = self.lora_up[layer_idx]          # (r, H)
                lora_out = (x @ Wd) @ Wu              # (B, T, H), zero at init

                # Combine: residual of (block + LoRA)
                x = x + (block_out - x) * iter_s + lora_out
                layer_idx += 1

        x = self.norm(x)
        return self.lm_head(x)

    def n_lora_params(self):
        return self.lora_down.numel() + self.lora_up.numel()


# ── Teacher load ──────────────────────────────────────────────────────
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
print(f"\nBuilding Per-Layer LoRA FRR (rank={LORA_RANK})...")
model = PerLayerLoRAFRR(
    hidden_dim=hidden, n_heads=n_heads, n_scales=4, iters_per_scale=7,
    vocab_size=vocab_size, ff_mult=1, lora_rank=LORA_RANK,
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

n_block = sum(p.numel() for p in model.block.parameters())
n_mod = model.layer_gamma.numel() + model.layer_beta.numel() + model.iter_scale.numel()
n_lora = model.n_lora_params()
n_total = n_block + n_mod + n_lora
print(f"  Block: {n_block:,}  Mod: {n_mod:,}  LoRA: {n_lora:,}")
print(f"  Total trainable: {n_total:,}")
print(f"  Compression: {N_TEACHER_LAYERS * 7 * hidden * hidden / n_total:.1f}x")


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


print("\nBaseline (should match pretrained, LoRA=0 init):")
t1, t10 = eval_vs_teacher(model, n=50)
t1_l, t10_l = eval_vs_teacher_last(model, n=200)
print(f"  all-pos:  T1={t1*100:.1f}%  T10={t10*100:.1f}%")
print(f"  last-tok: T1={t1_l*100:.1f}%  T10={t10_l*100:.1f}% (n=200)")


# ── Training ──────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"TRAINING: Per-Layer LoRA + Pure KL")
print(f"{'='*70}")

block_params = list(model.block.parameters())
mod_params = [model.layer_gamma, model.layer_beta, model.iter_scale]
lora_params = [model.lora_down, model.lora_up]
opt = torch.optim.AdamW([
    {'params': block_params, 'lr': LR_BLOCK, 'weight_decay': WD_BLOCK},
    {'params': mod_params, 'lr': LR_MOD, 'weight_decay': WD_MOD},
    {'params': lora_params, 'lr': LR_LORA, 'weight_decay': WD_LORA},
])
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, TOTAL_STEPS)

loss_history = []
best_t10 = 0.0
best_t10_last = 0.0
best_step = 0
t0 = time.time()

for step in range(TOTAL_STEPS):
    tokens = get_real_batch()
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
    loss_history.append(loss.item())

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
                       os.path.join(CHECKPOINT_DIR, 'lora_best.pt'))
        if t10_l > best_t10_last:
            best_t10_last = t10_l

        avg_loss = sum(loss_history[-500:]) / min(len(loss_history), 500)

        with torch.no_grad():
            up_norms = model.lora_up.norm(dim=(1, 2))  # (L,)
            down_norms = model.lora_down.norm(dim=(1, 2))
            up_norm_mean = up_norms.mean().item()
            up_norm_max = up_norms.max().item()
            # Effective LoRA impact: ||W_down @ W_up||_F approximates influence
            eff = torch.einsum('lhr,lrk->lhk', model.lora_down, model.lora_up).norm(dim=(1,2))
            eff_mean = eff.mean().item()
            eff_max = eff.max().item()

        print(
            f"  Step {step:>6d}/{TOTAL_STEPS}: KL={avg_loss:.4f} "
            f"T1={t1*100:.1f}% T10={t10*100:.1f}% ({elapsed:.0f}s){new_best}"
        )
        print(
            f"    [last-tok n=200] T1={t1_l*100:.1f}% T10={t10_l*100:.1f}% "
            f"best_last={best_t10_last*100:.1f}%"
        )
        print(
            f"    LoRA: up_norm mean={up_norm_mean:.4f} max={up_norm_max:.4f}  "
            f"eff_delta mean={eff_mean:.4f} max={eff_max:.4f}"
        )

    if step > 0 and step % CHECKPOINT_INTERVAL == 0:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f'lora_step{step}.pt')
        torch.save(model.state_dict(), ckpt_path)
        print(f"  >> Saved: {ckpt_path}")

final_path = os.path.join(CHECKPOINT_DIR, 'lora_final.pt')
torch.save(model.state_dict(), final_path)
total_time = time.time() - t0
print(f"\n{'='*70}")
print(f"PER-LAYER LoRA COMPLETE")
print(f"  Best T10 (all-pos): {best_t10*100:.1f}% at step {best_step}")
print(f"  Best T10 (last-tok, n=200): {best_t10_last*100:.1f}%")
print(f"  Total time: {total_time:.0f}s ({total_time/3600:.1f}h)")
print(f"{'='*70}")
