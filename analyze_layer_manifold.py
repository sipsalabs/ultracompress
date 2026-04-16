"""Analyze inter-layer weight variation in Qwen3-1.7B teacher.
Reveals the TRUE intrinsic dimensionality of layer differences."""
import torch
import numpy as np

print('Loading teacher weights...')
wd = torch.load('qwen3_1.7b_cache.pt', weights_only=True)

layer_vecs = []
for li in range(28):
    parts = []
    for key in ['self_attn.q_proj.weight', 'self_attn.k_proj.weight',
                'self_attn.v_proj.weight', 'self_attn.o_proj.weight',
                'mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight']:
        k = f'model.layers.{li}.{key}'
        if k in wd:
            parts.append(wd[k].float().flatten())
    layer_vecs.append(torch.cat(parts))

layer_matrix = torch.stack(layer_vecs)  # [28, ~50M] float32
print(f'Layer matrix shape: {layer_matrix.shape}')
print(f'Each layer: {layer_matrix.shape[1]:,} params')

mean_vec = layer_matrix.mean(dim=0)
centered = layer_matrix - mean_vec

# Use Gram matrix trick: compute 28x28 matrix, not full SVD of 28x50M
# Eigenvalues of C^T C / (n-1) = singular values squared
print('\nComputing Gram matrix (28x28) for efficient SVD...')
gram = (centered @ centered.T).numpy().astype(np.float64)
eigenvalues, eigenvectors = np.linalg.eigh(gram)
# eigh returns ascending order, flip
eigenvalues = eigenvalues[::-1]
eigenvectors = eigenvectors[:, ::-1]
# Singular values = sqrt(eigenvalues)
S = np.sqrt(np.maximum(eigenvalues, 0))
total_var = (S**2).sum()

print(f'\nSingular value spectrum:')
print(f'  Component | Singular Value | % Variance | Cumulative')
print(f'  ' + '-' * 60)
cum = 0
for i, s in enumerate(S):
    pct = (s**2) / total_var * 100
    cum += pct
    print(f'  {i+1:>9d} | {s:>14.4f} | {pct:>9.3f}% | {cum:>8.2f}%')
    if cum > 99.99:
        break

print(f'\n  HOW MANY COMPONENTS TO RECONSTRUCT?')
for target in [80, 90, 95, 99, 99.5, 99.9]:
    cum = 0
    for i, s in enumerate(S):
        cum += (s**2) / total_var * 100
        if cum >= target:
            params_needed = (i + 1) * 28
            print(f'    {target}% variance: {i+1} components ({params_needed} mixing params)')
            break

# Per-weight-matrix analysis
print(f'\n\nPER-MATRIX SVD ANALYSIS:')
for key_name, key in [('Q_proj', 'self_attn.q_proj.weight'),
                       ('K_proj', 'self_attn.k_proj.weight'),
                       ('V_proj', 'self_attn.v_proj.weight'),
                       ('O_proj', 'self_attn.o_proj.weight'),
                       ('Gate', 'mlp.gate_proj.weight'),
                       ('Up', 'mlp.up_proj.weight'),
                       ('Down', 'mlp.down_proj.weight')]:
    mats = []
    for li in range(28):
        k = f'model.layers.{li}.{key}'
        if k in wd:
            mats.append(wd[k].float().flatten().numpy())
    if not mats:
        continue
    M = torch.stack([torch.tensor(m) for m in mats])
    M_c = M - M.mean(dim=0)
    gram_m = (M_c @ M_c.T).numpy().astype(np.float64)
    ev_m, _ = np.linalg.eigh(gram_m)
    ev_m = ev_m[::-1]
    Sm = np.sqrt(np.maximum(ev_m, 0))
    tv = (Sm**2).sum()
    cum90 = 0
    for i, s in enumerate(Sm):
        cum90 += (s**2) / tv * 100
        if cum90 >= 90:
            print(f'  {key_name:>8}: 90% variance in {i+1} components, top-1={Sm[0]**2/tv*100:.1f}%')
            break

# Cosine similarities
print(f'\n  INTER-LAYER COSINE SIMILARITY:')
for i in range(min(5, len(layer_vecs)-1)):
    a, b = layer_vecs[i].numpy(), layer_vecs[i+1].numpy()
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    print(f'    Layer {i} vs {i+1}: {cos:.6f}')
a0, a27 = layer_vecs[0].numpy(), layer_vecs[-1].numpy()
cos_end = np.dot(a0, a27) / (np.linalg.norm(a0) * np.linalg.norm(a27))
print(f'    Layer 0 vs 27: {cos_end:.6f}')

# How much does the MEAN explain?
mean_norm = mean_vec.norm().item()
total_norms = np.mean([layer_matrix[i].norm().item() for i in range(28)])
residual_norms = np.mean([centered[i].norm().item() for i in range(28)])
print(f'\n  MEAN vs RESIDUAL:')
print(f'    Mean weight norm: {mean_norm:.2f}')
print(f'    Avg layer norm: {total_norms:.2f}')
print(f'    Avg residual norm: {residual_norms:.2f}')
print(f'    Mean explains: {(1 - residual_norms/total_norms)*100:.1f}% of weight magnitude')
print(f'    Residual is: {residual_norms/mean_norm*100:.1f}% of mean')

del wd
print('\nDone! This tells us exactly how many dimensions of per-layer variation exist.')
