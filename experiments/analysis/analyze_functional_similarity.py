"""
Functional Similarity Analysis using CKA (Centered Kernel Alignment).

GOAL: Complement the weight manifold analysis. We showed weights are orthogonal
(27D, cosine ~0.001). But FRR works (64% T10) → it operates in FUNCTION space.
This analysis measures how similar the layers are FUNCTIONALLY.

If functional similarity is high → FRR's shared block captures the common function.
If functional similarity is low → we need more blocks.
The eigenspectrum of the CKA matrix tells us the true number of functional modes.

CKA(X, Y) = HSIC(X, Y) / sqrt(HSIC(X,X) * HSIC(Y,Y))
where HSIC uses linear kernel: K = X @ X^T
"""
import torch
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from ultracompress.inference import ModelConfig, MiniTransformer

DEVICE = 'cpu'  # Run on CPU to not interfere with training
N_SAMPLES = 200
SEQ_LEN = 32


def centering_matrix(n: int) -> torch.Tensor:
    """H = I - 1/n J"""
    return torch.eye(n) - torch.ones(n, n) / n


def linear_CKA(X: torch.Tensor, Y: torch.Tensor) -> float:
    """CKA with linear kernel. X, Y: (n, d)"""
    n = X.shape[0]
    H = centering_matrix(n)

    # Center
    Xc = H @ X
    Yc = H @ Y

    # HSIC with linear kernel: trace(K_X @ H @ K_Y @ H) / (n-1)^2
    # = trace(X @ X^T @ H @ Y @ Y^T @ H) / (n-1)^2
    # Efficient: ||X^T @ H @ Y||_F^2 / (||X^T @ H @ X||_F * ||Y^T @ H @ Y||_F)
    # Even simpler for centered data: ||Xc^T @ Yc||_F^2 / (||Xc^T @ Xc||_F * ||Yc^T @ Yc||_F)

    XtY = Xc.T @ Yc  # (d, d') — too big if d=2048!

    # Use the kernel trick: work in sample space (n×n) not feature space (d×d)
    Kx = Xc @ Xc.T  # (n, n)
    Ky = Yc @ Yc.T  # (n, n)

    hsic_xy = (Kx * Ky).sum()  # trace(K_x @ K_y)
    hsic_xx = (Kx * Kx).sum()
    hsic_yy = (Ky * Ky).sum()

    return (hsic_xy / torch.sqrt(hsic_xx * hsic_yy + 1e-10)).item()


def main():
    print("=" * 70)
    print("FUNCTIONAL SIMILARITY ANALYSIS (CKA)")
    print(f"Will collect hidden states from {N_SAMPLES} samples × {SEQ_LEN} tokens")
    print("=" * 70)

    # Load teacher on CPU
    print("Loading Qwen3-1.7B teacher (CPU only, won't affect GPU training)...")
    wd = torch.load('qwen3_1.7b_cache.pt', weights_only=True, map_location='cpu')

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
    N_LAYERS = 28
    for li in range(N_LAYERS):
        for h, g in hf_to_gguf.items():
            k = f'model.layers.{li}.{h}'
            if k in wd:
                gd[f'blk.{li}.{g}'] = wd[k].float()
    del wd

    hidden = gd['token_embd.weight'].shape[1]
    n_heads = 16
    head_dim = hidden // n_heads
    vocab = gd['token_embd.weight'].shape[0]

    config = ModelConfig(
        n_layers=N_LAYERS, n_heads=n_heads, n_kv_heads=8,
        hidden_size=hidden, intermediate_size=hidden * 3,
        vocab_size=vocab, head_dim=head_dim,
    )
    teacher = MiniTransformer(config, 'cpu')
    teacher.load_weights(gd)
    del gd

    # Load tokens
    ALL_TOKENS = torch.load('fineweb_edu_100M_tokens.pt', weights_only=True).to(torch.long)
    print(f"  {ALL_TOKENS.numel():,} tokens loaded")

    # Collect hidden states at each layer using return_hidden=True
    print(f"\nCollecting hidden states ({N_SAMPLES} samples × {SEQ_LEN} tokens)...")
    # We'll take the MEAN hidden state per sample (average across positions)
    # This gives us (N_SAMPLES, hidden_dim) per layer
    layer_hiddens = [[] for _ in range(N_LAYERS + 1)]  # +1 for input embeddings

    t0 = time.time()
    embed_w = teacher.embed_weight.float()

    for i in range(N_SAMPLES):
        if i % 50 == 0:
            elapsed = time.time() - t0
            print(f"  Sample {i}/{N_SAMPLES} ({elapsed:.0f}s)")

        start = torch.randint(0, ALL_TOKENS.numel() - SEQ_LEN, (1,)).item()
        tokens = ALL_TOKENS[start:start + SEQ_LEN].unsqueeze(0)

        with torch.no_grad():
            # Get embedding (layer 0)
            x_emb = embed_w[tokens[0]]  # (seq_len, hidden)
            layer_hiddens[0].append(x_emb.mean(0))

            # Get all hidden states in ONE forward pass
            _, hidden_states = teacher.forward(tokens, return_hidden=True)
            for l in range(N_LAYERS):
                # hidden_states[l] is (1, seq_len, hidden)
                layer_hiddens[l + 1].append(hidden_states[l][0].mean(0))

    # Stack: each is (N_SAMPLES, hidden_dim)
    print("\nComputing CKA matrix...")
    H = []
    for l in range(N_LAYERS + 1):
        H.append(torch.stack(layer_hiddens[l]))  # (N_SAMPLES, hidden)
        del layer_hiddens[l][:]
    del layer_hiddens

    # Compute CKA matrix
    cka_matrix = torch.zeros(N_LAYERS + 1, N_LAYERS + 1)
    for i in range(N_LAYERS + 1):
        for j in range(i, N_LAYERS + 1):
            cka = linear_CKA(H[i], H[j])
            cka_matrix[i, j] = cka
            cka_matrix[j, i] = cka
        if i % 5 == 0:
            print(f"  CKA row {i}/{N_LAYERS}")

    print("\n" + "=" * 70)
    print("CKA SIMILARITY MATRIX (layers 0-28, 0=embedding)")
    print("=" * 70)

    # Print condensed: every other layer
    print("       ", end="")
    for j in range(0, N_LAYERS + 1, 4):
        print(f"  L{j:02d}", end="")
    print()
    for i in range(0, N_LAYERS + 1, 2):
        print(f"L{i:02d}: ", end="")
        for j in range(0, N_LAYERS + 1, 4):
            print(f"  {cka_matrix[i, j]:.2f}", end="")
        print()

    # Eigenspectrum of CKA matrix — THIS is the key!
    print("\n" + "=" * 70)
    print("CKA EIGENSPECTRUM — True number of functional modes")
    print("=" * 70)

    # Only use layers 1-28 (exclude embedding)
    cka_layers = cka_matrix[1:, 1:]  # 28 × 28
    eigenvalues = torch.linalg.eigvalsh(cka_layers)
    eigenvalues = eigenvalues.flip(0)  # Descending

    print("Eigenvalues (descending):")
    for i, ev in enumerate(eigenvalues):
        print(f"  EV_{i+1}: {ev:.4f}")

    # Cumulative variance
    total = eigenvalues.sum()
    cumsum = eigenvalues.cumsum(0) / total
    print(f"\nCumulative variance explained:")
    for k in [1, 2, 3, 4, 5, 8, 10, 15, 20, 27]:
        k = min(k, len(cumsum))
        print(f"  Top {k}: {cumsum[k-1]*100:.1f}%")

    n_for_90 = (cumsum >= 0.9).nonzero()[0].item() + 1 if (cumsum >= 0.9).any() else 28
    n_for_95 = (cumsum >= 0.95).nonzero()[0].item() + 1 if (cumsum >= 0.95).any() else 28
    n_for_99 = (cumsum >= 0.99).nonzero()[0].item() + 1 if (cumsum >= 0.99).any() else 28

    print(f"\n  Components for 90% variance: {n_for_90}")
    print(f"  Components for 95% variance: {n_for_95}")
    print(f"  Components for 99% variance: {n_for_99}")

    # Layer clustering
    print("\n" + "=" * 70)
    print("ADJACENT LAYER CKA (functional similarity between neighbors)")
    print("=" * 70)
    for l in range(N_LAYERS):
        cka_val = cka_matrix[l + 1, l + 2] if l + 2 <= N_LAYERS else 0
        bar = "█" * int(cka_val * 40)
        print(f"  Layer {l:2d} ↔ {l+1:2d}: CKA={cka_val:.3f}  {bar}")

    # Inter-scale similarity (average CKA within and between scales)
    print("\n" + "=" * 70)
    print("INTER-SCALE SIMILARITY (mean CKA within/between 4 scales)")
    print("=" * 70)
    for s1 in range(4):
        for s2 in range(s1, 4):
            l1_start, l1_end = s1 * 7 + 1, (s1 + 1) * 7 + 1
            l2_start, l2_end = s2 * 7 + 1, (s2 + 1) * 7 + 1
            block = cka_matrix[l1_start:l1_end, l2_start:l2_end]
            mean_cka = block.mean().item()
            print(f"  Scale {s1} ↔ Scale {s2}: {mean_cka:.3f}")

    elapsed = time.time() - t0
    print(f"\nTotal analysis time: {elapsed:.0f}s")

    # Save CKA matrix for later use
    torch.save(cka_matrix, 'cka_matrix_1.7b.pt')
    print("Saved CKA matrix to cka_matrix_1.7b.pt")


if __name__ == "__main__":
    main()
