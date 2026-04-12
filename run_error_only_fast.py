"""
FAST Error-Only Test: Focus on the BIG weight matrices.
The key question: can a small predictor learn the layer-to-layer mapping for
attention/FFN weights where cosine similarity is ~0?
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys

def main():
    print("=" * 70)
    print("ERROR-ONLY FAST TEST: Real weight matrices (attention + FFN)")
    print("=" * 70)

    cache_path = "qwen3_0.6b_cache.pt"
    print(f"Loading {cache_path}...")
    model = torch.load(cache_path, map_location='cpu', weights_only=False)

    # Test the big weight types that matter
    weight_types = [
        'self_attn.q_proj.weight',    # 1M params
        'self_attn.k_proj.weight',    # 1M params
        'mlp.gate_proj.weight',       # 3M params
        'mlp.up_proj.weight',         # 3M params
        'mlp.down_proj.weight',       # 3M params
    ]

    results = {}

    for wtype in weight_types:
        # Extract layers
        layers = []
        for i in range(28):
            key = f'model.layers.{i}.{wtype}'
            if key in model:
                layers.append(model[key].float().flatten())

        if len(layers) < 2:
            print(f"\n  {wtype}: NOT FOUND, skipping")
            continue

        n_layers = len(layers)
        weight_dim = layers[0].shape[0]

        print(f"\n{'='*70}")
        print(f"TESTING: {wtype} ({weight_dim:,} params/layer, {n_layers} layers)")
        print(f"{'='*70}")

        # Baseline similarity
        print("\n  Adjacent layer similarity (raw):")
        cosines = []
        for i in range(min(5, n_layers-1)):
            cos = F.cosine_similarity(layers[i].unsqueeze(0), layers[i+1].unsqueeze(0)).item()
            cosines.append(cos)
            print(f"    Layer {i}->{i+1}: cosine={cos:.6f}")
        avg_cos = sum(cosines)/len(cosines)
        print(f"    Average: {avg_cos:.6f}")

        # Test chunked predictor with different chunk sizes
        for chunk_size in [1024, 4096]:
            for hidden_dim in [64, 256]:
                print(f"\n  Chunk={chunk_size}, Hidden={hidden_dim}:")

                # Pad
                pad = (chunk_size - weight_dim % chunk_size) % chunk_size
                if pad > 0:
                    padded = [torch.cat([w, torch.zeros(pad)]) for w in layers]
                else:
                    padded = layers

                n_chunks = padded[0].shape[0] // chunk_size
                chunked = [w.reshape(n_chunks, chunk_size) for w in padded]

                # Simple predictor: chunk -> chunk
                net = nn.Sequential(
                    nn.Linear(chunk_size, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, chunk_size),
                )
                # Identity init
                nn.init.zeros_(net[-1].weight)
                nn.init.zeros_(net[-1].bias)

                predictor_params = sum(p.numel() for p in net.parameters())
                print(f"    Predictor: {predictor_params:,} params")
                print(f"    Data: {weight_dim * (n_layers-1):,} params to predict")
                print(f"    Ratio (predictor/data): {predictor_params/(weight_dim*(n_layers-1)):.4f}")

                opt = torch.optim.Adam(net.parameters(), lr=1e-3)

                t0 = time.time()
                for step in range(2000):
                    total_loss = 0
                    for i in range(n_layers - 1):
                        pred = chunked[i].detach() + net(chunked[i].detach())
                        loss = F.mse_loss(pred, chunked[i + 1])
                        total_loss = total_loss + loss

                    opt.zero_grad()
                    total_loss.backward()
                    opt.step()

                    if step % 500 == 0:
                        avg_loss = total_loss.item() / (n_layers - 1)
                        # Quick cosine check
                        with torch.no_grad():
                            p = chunked[0].detach() + net(chunked[0].detach())
                            cos = F.cosine_similarity(
                                p.flatten().unsqueeze(0),
                                chunked[1].flatten().unsqueeze(0)
                            ).item()
                        print(f"      Step {step}: loss={avg_loss:.8f} cos(0->1)={cos:.6f} [{time.time()-t0:.1f}s]")

                # Final evaluation
                print(f"\n    Final per-layer results:")
                all_cos = []
                all_rel = []
                with torch.no_grad():
                    for i in range(n_layers - 1):
                        pred = chunked[i] + net(chunked[i])
                        actual = chunked[i + 1]

                        cos = F.cosine_similarity(
                            pred.flatten().unsqueeze(0),
                            actual.flatten().unsqueeze(0)
                        ).item()

                        err = (actual - pred).flatten()
                        rel = err.abs().mean() / (actual.flatten().abs().mean() + 1e-10)

                        all_cos.append(cos)
                        all_rel.append(rel.item())

                        if i < 3 or i >= n_layers - 3:
                            print(f"      Layer {i+1}: cos={cos:.6f} rel_err={rel:.6f}")

                    if n_layers > 8:
                        print(f"      ... ({n_layers-6} layers omitted)")

                avg_cos = sum(all_cos)/len(all_cos)
                avg_rel = sum(all_rel)/len(all_rel)
                avg_acc = 1.0 - avg_rel

                print(f"\n    AVERAGE: cosine={avg_cos:.6f}, accuracy={avg_acc*100:.2f}%, time={time.time()-t0:.1f}s")

                # Compression math
                error_size = weight_dim * (n_layers - 1)  # full error storage
                sparse_frac = 1.0  # assume worst case until we know sparsity
                # Check actual sparsity
                with torch.no_grad():
                    all_errors = []
                    for i in range(n_layers-1):
                        pred = chunked[i] + net(chunked[i])
                        err = (chunked[i+1] - pred).flatten()
                        all_errors.append(err)
                    all_err = torch.cat(all_errors)
                    threshold_1pct = all_err.abs().quantile(0.5).item()  # median
                    nnz_50 = (all_err.abs() > all_err.abs().quantile(0.5)).sum().item()
                    nnz_90 = (all_err.abs() > all_err.abs().quantile(0.1)).sum().item()
                    nnz_99 = (all_err.abs() > all_err.abs().quantile(0.01)).sum().item()
                    total_err = all_err.numel()

                    # If we keep top 50% of errors, what's reconstruction quality?
                    sparse_err = all_err.clone()
                    mask_50 = all_err.abs() < all_err.abs().quantile(0.5)
                    sparse_err[mask_50] = 0
                    # This is lossy now - how much info is in the bottom 50%?
                    dropped_energy = all_err[mask_50].norm().item()
                    total_energy = all_err.norm().item()

                    print(f"\n    Error distribution:")
                    print(f"      Total error elements: {total_err:,}")
                    print(f"      Error energy in top 50%: {1-dropped_energy/total_energy:.1%}")
                    print(f"      Error energy in top 10%: ...(computing)")

                    # Real compression: base(1 layer) + predictor + sparse errors
                    for keep_pct in [100, 50, 10, 1]:
                        if keep_pct == 100:
                            eff_error = error_size  # lossless
                        else:
                            eff_error = int(error_size * keep_pct / 100)
                        total = weight_dim + predictor_params + eff_error
                        original = weight_dim * n_layers
                        ratio = original / total
                        print(f"      Keep {keep_pct}% errors: {total:,} params = {ratio:.2f}x compression")

                results[f"{wtype}_c{chunk_size}_h{hidden_dim}"] = {
                    'avg_cosine': avg_cos,
                    'avg_accuracy': avg_acc,
                }

        # Quick cleanup
        del layers, chunked

    # VERDICT
    print(f"\n{'='*70}")
    print("VERDICT")
    print("=" * 70)
    if results:
        for key, res in results.items():
            print(f"  {key}: cos={res['avg_cosine']:.6f} acc={res['avg_accuracy']*100:.2f}%")

        best = max(results.values(), key=lambda x: x['avg_accuracy'])
        print(f"\n  Best prediction accuracy: {best['avg_accuracy']*100:.2f}%")

        if best['avg_accuracy'] > 0.995:
            print("  -> PARADIGM WORKS for weight matrices")
        elif best['avg_accuracy'] > 0.99:
            print("  -> PROMISING but not quite there")
        elif best['avg_accuracy'] > 0.95:
            print("  -> MODERATE: some compression possible but not paradigm-shifting")
        else:
            print("  -> LAYERS TOO DIFFERENT for error-only compression of weight matrices")
            print("     The predictor can't learn arbitrary layer-to-layer mappings.")
            print("     This approach works great for layernorms but not for the bulk of params.")

if __name__ == '__main__':
    main()
