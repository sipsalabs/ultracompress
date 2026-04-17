"""Test Tensor Train compression on Qwen3-0.6B embeddings.

Applies TTEmbedding.from_pretrained() with different TT ranks (8, 16, 32),
measures compression ratio and reconstruction error, and tests output similarity.

The vocab size 151936 has a large prime factor (1187), so we pad to 152064
which factors nicely as 8*8*8*33*9 for balanced TT decomposition.
"""
import torch
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultracompress.tensor_train import tt_decompose, tt_to_matrix

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("=" * 60)
print("TENSOR TRAIN COMPRESSION TEST -- Qwen3-0.6B Embeddings")
print("=" * 60)

# Load cached weights
print("\nLoading qwen3_0.6b_cache.pt ...")
wd = torch.load('qwen3_0.6b_cache.pt', map_location='cpu', weights_only=True)

embed_weight = wd['model.embed_tokens.weight'].float()
orig_vocab, dim = embed_weight.shape
print(f"  Embedding shape: {embed_weight.shape}")
print(f"  Original params: {embed_weight.numel():,}")
print(f"  Original size:   {embed_weight.numel() * 4 / 1024**2:.1f} MB (FP32)")

# Also load lm_head for output similarity test
lm_head = wd.get('lm_head.weight', embed_weight).float()

# Pad vocab to 152064 = 8*8*8*33*9 for balanced TT factors
# 1024 = 4*4*4*4*4 for embedding dim
PADDED_VOCAB = 8 * 8 * 8 * 33 * 9  # = 152064
assert PADDED_VOCAB >= orig_vocab
pad_rows = PADDED_VOCAB - orig_vocab
if pad_rows > 0:
    embed_padded = torch.cat([embed_weight, torch.zeros(pad_rows, dim)], dim=0)
    print(f"  Padded vocab: {orig_vocab} -> {PADDED_VOCAB} (+{pad_rows} rows)")
else:
    embed_padded = embed_weight

# TT shape: row_factors x col_factors
# rows: (8, 8, 8, 33, 9) -> product = 152064
# cols: (4, 4, 4, 4, 4)  -> product = 1024
TT_SHAPE = [(8, 4), (8, 4), (8, 4), (33, 4), (9, 4)]
n_factors = len(TT_SHAPE)

row_check = int(np.prod([s[0] for s in TT_SHAPE]))
col_check = int(np.prod([s[1] for s in TT_SHAPE]))
print(f"  TT shape: {TT_SHAPE}")
print(f"  Row product: {row_check} (need {PADDED_VOCAB})")
print(f"  Col product: {col_check} (need {dim})")
assert row_check == PADDED_VOCAB
assert col_check == dim

# Test with different TT ranks
ranks_to_test = [8, 16, 32]
results = {}

for rank in ranks_to_test:
    print(f"\n{'-' * 60}")
    print(f"TT Rank = {rank}")
    print(f"{'-' * 60}")

    tt_ranks = [1] + [rank] * (n_factors - 1) + [1]
    print(f"  Ranks: {tt_ranks}")

    # Decompose
    print("  Decomposing ...")
    cores = tt_decompose(embed_padded.float(), TT_SHAPE, tt_ranks)

    # Compression stats
    orig_params = embed_weight.numel()  # count original, not padded
    tt_params = sum(c.numel() for c in cores)
    compression = orig_params / tt_params
    tt_size_mb = tt_params * 4 / 1024**2

    print(f"  TT cores: {[tuple(c.shape) for c in cores]}")
    print(f"  TT params:       {tt_params:,}")
    print(f"  TT size:         {tt_size_mb:.4f} MB (FP32)")
    print(f"  Compression:     {compression:.1f}x")

    # Reconstruction
    print("  Reconstructing full embedding table ...")
    with torch.no_grad():
        reconstructed = tt_to_matrix(cores, TT_SHAPE)

    # Trim back to original vocab
    reconstructed = reconstructed[:orig_vocab, :dim]

    diff = embed_weight - reconstructed
    mse = (diff ** 2).mean().item()
    mae = diff.abs().mean().item()
    rel_err = (diff.norm() / embed_weight.norm()).item()
    cosine_sim = torch.nn.functional.cosine_similarity(
        embed_weight.reshape(1, -1), reconstructed.reshape(1, -1)
    ).item()

    print(f"  MSE:             {mse:.6f}")
    print(f"  MAE:             {mae:.6f}")
    print(f"  Relative error:  {rel_err:.4f}")
    print(f"  Cosine sim:      {cosine_sim:.6f}")

    # Output similarity test: pick some token IDs, get embeddings, project through lm_head
    print("  Testing output similarity ...")
    test_tokens = torch.tensor([0, 1, 100, 1000, 10000, 50000, 100000, 151000])
    test_tokens = test_tokens[test_tokens < orig_vocab]

    orig_embeds = embed_weight[test_tokens]
    tt_embeds = reconstructed[test_tokens]

    # Per-token cosine similarity
    per_token_cos = torch.nn.functional.cosine_similarity(
        orig_embeds, tt_embeds, dim=-1
    )
    print(f"  Per-token cosine (mean): {per_token_cos.mean().item():.6f}")
    print(f"  Per-token cosine (min):  {per_token_cos.min().item():.6f}")

    # Project through lm_head to get logits
    orig_logits = orig_embeds @ lm_head.T
    tt_logits = tt_embeds @ lm_head.T

    # Top-1 agreement
    orig_top1 = orig_logits.argmax(dim=-1)
    tt_top1 = tt_logits.argmax(dim=-1)
    top1_agree = (orig_top1 == tt_top1).float().mean().item()

    # Top-10 agreement
    orig_top10 = orig_logits.topk(10, dim=-1).indices
    tt_top10 = tt_logits.topk(10, dim=-1).indices
    top10_overlap = 0
    for i in range(len(test_tokens)):
        overlap = len(set(orig_top10[i].tolist()) & set(tt_top10[i].tolist()))
        top10_overlap += overlap / 10.0
    top10_overlap /= len(test_tokens)

    # Logit cosine similarity
    logit_cosine = torch.nn.functional.cosine_similarity(
        orig_logits, tt_logits, dim=-1
    ).mean().item()

    print(f"  Top-1 agreement: {top1_agree:.0%}")
    print(f"  Top-10 overlap:  {top10_overlap:.0%}")
    print(f"  Logit cosine:    {logit_cosine:.6f}")

    results[rank] = {
        'tt_params': tt_params,
        'compression': round(compression, 1),
        'tt_size_mb': round(tt_size_mb, 4),
        'mse': mse,
        'mae': mae,
        'relative_error': round(rel_err, 4),
        'cosine_sim': round(cosine_sim, 6),
        'top1_agreement': round(top1_agree, 2),
        'top10_overlap': round(top10_overlap, 2),
        'logit_cosine': round(logit_cosine, 6),
    }

# Summary
print(f"\n{'=' * 60}")
print("SUMMARY")
print(f"{'=' * 60}")
print(f"{'Rank':>6} {'Params':>12} {'Size MB':>8} {'Compress':>10} {'RelErr':>8} {'Cosine':>8} {'Top1':>6} {'Top10':>6}")
print(f"{'-' * 6} {'-' * 12} {'-' * 8} {'-' * 10} {'-' * 8} {'-' * 8} {'-' * 6} {'-' * 6}")
for rank in ranks_to_test:
    r = results[rank]
    print(f"{rank:>6} {r['tt_params']:>12,} {r['tt_size_mb']:>8.4f} {r['compression']:>9.1f}x "
          f"{r['relative_error']:>8.4f} {r['cosine_sim']:>8.4f} {r['top1_agreement']:>5.0%} {r['top10_overlap']:>5.0%}")

print(f"\nOriginal embedding: {embed_weight.numel():,} params = {embed_weight.numel() * 4 / 1024**2:.1f} MB")
