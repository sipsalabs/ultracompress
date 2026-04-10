"""Full 28-layer genome compression test using GenomeCompressor."""
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

print("=== FULL 28-LAYER GENOME COMPRESSION ===")
print("Previous 4-layer tests hit 14% ceiling because")
print("4-layer original only has 0% agreement with 28-layer.")
print("With all 28 layers, the genome should do MUCH better.")
print()

compressor = GenomeCompressor(model_weights=wd, model_config=config, device=device)

for sd, nh in [(64, 4), (128, 4)]:
    print(f"--- Config: sd={sd}, nh={nh} ---")
    result = compressor.compress(
        small_dim=sd,
        n_heads=nh,
        n_layers=28,
        n_steps=5000,
        batch_size=4,  # smaller batch for 28 layers (memory)
        seq_len=32,
        lr=0.001,
        eval_every=1000,
        eval_samples=50,
        verbose=True,
    )
    print(f"\n  RESULT: Top1={result.top1_accuracy*100:.0f}% Top10={result.top10_overlap*100:.0f}%")
    print(f"  Genome: {result.genome_size_mb:.1f} MB ({result.compression_ratio:.0f}x compression)")
    print(f"  BPW: {result.bpw:.4f}")
    print(f"  At this ratio: 10T = {10e12 * result.bpw / 8 / 1e9:.1f} GB")
    print()

    # Save best genome
    result.genome.save_genome(f"genome_sd{sd}_28L.pt")
    print(f"  Saved to genome_sd{sd}_28L.pt")
    print()

    del result
    torch.cuda.empty_cache()
