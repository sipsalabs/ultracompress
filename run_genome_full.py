"""Full 28-layer genome compression — THE REAL TEST.

Previous 4-layer tests hit 14% ceiling because the original
4-layer model itself has 0% agreement with the 28-layer model.
With all 28 layers, quality should break through.

Uses progressive training: train each layer independently first,
then fine-tune jointly. ~9x faster than end-to-end.
"""
import torch
import sys
sys.path.insert(0, '.')
from ultracompress.inference import ModelConfig
from ultracompress.genome_compressor import GenomeCompressor

device = 'cuda'
wd = torch.load('qwen3_0.6b_cache.pt', weights_only=True)

config = ModelConfig(
    n_layers=28, n_heads=16, n_kv_heads=8, hidden_size=1024,
    intermediate_size=3072, vocab_size=151936, head_dim=128,
)

print("=" * 60)
print("  FULL 28-LAYER GENOME COMPRESSION")
print("  4-layer original has 0% agreement with 28-layer.")
print("  This test uses ALL 28 layers — the real quality test.")
print("=" * 60)
print()

compressor = GenomeCompressor(model_weights=wd, model_config=config, device=device)

for sd, nh in [(64, 4), (128, 4)]:
    print(f"=== Config: sd={sd}, nh={nh} ===")
    result = compressor.compress_progressive(
        small_dim=sd,
        n_heads=nh,
        n_layers=28,
        steps_per_layer=500,  # 500 per layer + 2000 joint = 16K total
        batch_size=4,
        seq_len=32,
        lr=0.001,
        eval_samples=50,
        verbose=True,
    )
    print(f"\n  RESULT: Top1={result.top1_accuracy*100:.0f}% Top10={result.top10_overlap*100:.0f}%")
    print(f"  Genome: {result.genome_size_mb:.1f} MB ({result.compression_ratio:.0f}x)")
    print(f"  BPW: {result.bpw:.4f}")
    print(f"  Time: {result.training_time:.0f}s")
    print(f"  At this ratio: 10T = {10e12 * result.bpw / 8 / 1e9:.1f} GB")
    print(f"                 1000T = {1000e12 * result.bpw / 8 / 1e9:.1f} GB")
    print()

    result.genome.save_genome(f"genome_sd{sd}_28L.pt")
    del result
    torch.cuda.empty_cache()
