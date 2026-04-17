"""
Fresh CKA + SVD analysis of teacher's per-layer hidden states.

Question: what is the ACTUAL intrinsic dimension of the teacher's
layer-to-layer computation on real data?

Prior claim (commit ddc4008): 4 modes capture 99.7%. Verify on current data.

Collects hidden states h_1..h_28 from N samples, then:
 (a) Stack into H of shape (N*T, 28, D) — one tensor per layer
 (b) Compute per-layer delta: δ_ℓ = h_ℓ - h_{ℓ-1}  (residual step)
 (c) SVD on stacked deltas → how many singular vectors capture 99%?
 (d) Linear CKA between all pairs of layers → identify functional duplicates
 (e) SVD on the 28-layer-mean residual field to find a common basis

Outputs decisive numbers + saves top-K basis vectors to disk for use
by the subsequent ODE-manifold compressor.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from ultracompress.inference import ModelConfig, MiniTransformer

DEVICE = 'cuda:0'
SEQ_LEN = 64
N_SAMPLES = 200
N_TEACHER_LAYERS = 28

print("=" * 72)
print("FRESH CKA + SVD INTRINSIC-DIMENSION ANALYSIS")
print(f"  {N_SAMPLES} samples x {SEQ_LEN} positions, 28 layers")
print("=" * 72)

# Load teacher
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
del gd

# Collect hidden states
print("\nCollecting hidden states...")
tokens_all = torch.load('fineweb_edu_100M_tokens.pt', weights_only=True)
torch.manual_seed(0)
starts = torch.randint(0, tokens_all.numel() - SEQ_LEN, (N_SAMPLES,)).tolist()

# Shape: (N*T, L, D). Keep on CPU — 200*64*28*2048*4 = 2.7GB.
H_all = torch.zeros(N_SAMPLES * SEQ_LEN, N_TEACHER_LAYERS, hidden, dtype=torch.float32)

with torch.no_grad():
    for i, s in enumerate(starts):
        inp = tokens_all[s:s+SEQ_LEN].long().unsqueeze(0).to(DEVICE)
        _, hiddens = teacher.forward(inp, max_layers=N_TEACHER_LAYERS, return_hidden=True)
        for li, h in enumerate(hiddens):
            H_all[i*SEQ_LEN:(i+1)*SEQ_LEN, li] = h[0].cpu()
        if (i+1) % 50 == 0:
            print(f"  {i+1}/{N_SAMPLES}")

N, L, D = H_all.shape
print(f"\nHidden-state tensor: {H_all.shape}  ({H_all.numel()*4/1e9:.2f} GB)")

# ── (a) Per-layer residual deltas ──
print("\n" + "-"*72)
print("(a) Residual deltas δ_ℓ = h_ℓ - h_{ℓ-1}")
# Layers are POST each block, so δ_ℓ = H[:,ℓ] - H[:,ℓ-1]  for ℓ=1..27 (27 deltas)
deltas = H_all[:, 1:] - H_all[:, :-1]    # (N, 27, D)
print(f"  Mean ||δ|| per layer (1-27):")
dnorms = deltas.pow(2).mean(0).sum(-1).sqrt()     # (27,)
for l in range(0, 27, 2):
    print(f"    layer {l+1:2d}: {dnorms[l].item():8.4f}")

# ── (b) SVD of stacked deltas — what's the intrinsic dim? ──
print("\n" + "-"*72)
print("(b) SVD of all deltas stacked: (N*27, D)")
deltas_flat = deltas.reshape(-1, D).float()
# Subsample if huge
max_rows = 30_000
if deltas_flat.shape[0] > max_rows:
    idx = torch.randperm(deltas_flat.shape[0])[:max_rows]
    deltas_flat = deltas_flat[idx]
print(f"  subsampled to {deltas_flat.shape}")
deltas_flat = deltas_flat - deltas_flat.mean(0, keepdim=True)
# Use torch.linalg.svd on GPU for speed
deltas_gpu = deltas_flat.to(DEVICE)
print("  running SVD on GPU...")
U, S, V = torch.linalg.svd(deltas_gpu, full_matrices=False)
S = S.cpu()
cum_var = (S.pow(2).cumsum(0) / S.pow(2).sum()).numpy()
print(f"  Singular values: top 20 = {S[:20].tolist()}")
print(f"  Cumulative variance explained:")
for k in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
    if k <= len(cum_var):
        print(f"    top {k:4d} modes: {cum_var[k-1]*100:6.2f}%")

# Dim needed for 90/95/99/99.5/99.9%
thresholds = [0.90, 0.95, 0.99, 0.995, 0.999]
print(f"  Modes needed for threshold variance:")
for t in thresholds:
    k = int((cum_var < t).sum()) + 1
    print(f"    {t*100:5.1f}%: {k} modes")

# ── (c) Per-layer SVD: does each layer live in a low-D subspace? ──
print("\n" + "-"*72)
print("(c) Per-layer SVD: intrinsic dim of each δ_ℓ separately")
per_layer_dims = []
for l in range(27):
    dl = deltas[:, l, :].to(DEVICE) - deltas[:, l, :].to(DEVICE).mean(0, keepdim=True)
    Sl = torch.linalg.svdvals(dl.to(torch.float32))
    cv = (Sl.pow(2).cumsum(0) / Sl.pow(2).sum()).cpu().numpy()
    k95 = int((cv < 0.95).sum()) + 1
    k99 = int((cv < 0.99).sum()) + 1
    per_layer_dims.append((l+1, k95, k99))
print(f"  layer  dim@95%  dim@99%")
for l, k95, k99 in per_layer_dims[::3]:
    print(f"   {l:3d}    {k95:5d}    {k99:5d}")

# ── (d) Linear CKA between layer deltas ──
print("\n" + "-"*72)
print("(d) Linear CKA between layer deltas (are any layers functionally duplicates?)")

def linear_cka(X, Y):
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    xy = (X.T @ Y).pow(2).sum()
    xx = (X.T @ X).pow(2).sum()
    yy = (Y.T @ Y).pow(2).sum()
    return (xy / (xx.sqrt() * yy.sqrt() + 1e-12)).item()

deltas_sub = deltas[:5000].to(DEVICE).float()
cka = torch.zeros(27, 27)
for i in range(27):
    for j in range(i, 27):
        v = linear_cka(deltas_sub[:, i, :], deltas_sub[:, j, :])
        cka[i, j] = v
        cka[j, i] = v

print(f"  CKA matrix (showing key pairs):")
print(f"    layer  1 <-> layer  2:  {cka[0,1]:.3f}")
print(f"    layer  1 <-> layer 14:  {cka[0,13]:.3f}")
print(f"    layer  1 <-> layer 27:  {cka[0,26]:.3f}")
print(f"    layer 14 <-> layer 15:  {cka[13,14]:.3f}")
print(f"    layer 14 <-> layer 20:  {cka[13,19]:.3f}")
print(f"    layer 20 <-> layer 27:  {cka[19,26]:.3f}")

# Find groups of similar layers (CKA > 0.8)
print(f"\n  Layer pairs with CKA > 0.8 (functionally similar):")
similar = []
for i in range(27):
    for j in range(i+1, 27):
        if cka[i,j] > 0.8:
            similar.append((i+1, j+1, cka[i,j].item()))
print(f"    {len(similar)} such pairs (of {27*26//2} total)")
for i, j, v in similar[:15]:
    print(f"      layer {i:2d} <-> layer {j:2d}: {v:.3f}")

# ── (e) Save top-K SVD basis for ODE student to use ──
print("\n" + "-"*72)
print("(e) Saving top-K SVD basis of residual field")
K_SAVE = 128
V_top = V[:K_SAVE].cpu()     # (K, D)
torch.save({
    'basis': V_top,          # columns = top-K directions in residual space
    'singular_values': S[:K_SAVE],
    'cum_var': cum_var,
    'per_layer_dims': per_layer_dims,
    'cka_matrix': cka,
}, 'teacher_intrinsic_analysis.pt')
print(f"  Saved top {K_SAVE} basis vectors + analysis to teacher_intrinsic_analysis.pt")

# ── Synthesis ──
print("\n" + "=" * 72)
print("SYNTHESIS — what this means for novel compression")
print("=" * 72)
k4 = cum_var[3] * 100
k16 = cum_var[15] * 100
k64 = cum_var[63] * 100
print(f"  Top   4 modes explain {k4:5.2f}% of residual variance")
print(f"  Top  16 modes explain {k16:5.2f}%")
print(f"  Top  64 modes explain {k64:5.2f}%")
k_for_99 = int((cum_var < 0.99).sum()) + 1
k_for_95 = int((cum_var < 0.95).sum()) + 1
print(f"\n  Intrinsic dim @ 99%: {k_for_99}/{D} = {k_for_99/D*100:.1f}% of hidden")
print(f"  Intrinsic dim @ 95%: {k_for_95}/{D}")
print(f"\n  Prior claim 'top 4 modes = 99.7%': {'CONFIRMED' if k4 > 99 else f'NOT CONFIRMED (actual {k4:.1f}%)'}")
print("=" * 72)
