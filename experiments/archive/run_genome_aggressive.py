"""Aggressive 28-layer genome training — push for 90%+ top-10."""
import torch, sys
sys.path.insert(0, '.')
from ultracompress.inference import ModelConfig
from ultracompress.genome_compressor import GenomeCompressor

device = 'cuda'
wd = torch.load('qwen3_0.6b_cache.pt', weights_only=True)

config = ModelConfig(
    n_layers=28, n_heads=16, n_kv_heads=8, hidden_size=1024,
    intermediate_size=3072, vocab_size=151936, head_dim=128,
)

print("=== AGGRESSIVE 28-LAYER: Push for 90%+ top-10 ===")
print("More steps per layer, more joint fine-tuning")
print()

compressor = GenomeCompressor(model_weights=wd, model_config=config, device=device)

# sd=128 with 2000 steps/layer + 5000 joint = 61K total steps
result = compressor.compress_progressive(
    small_dim=128,
    n_heads=4,
    n_layers=28,
    steps_per_layer=2000,
    batch_size=4,
    seq_len=32,
    lr=0.001,
    eval_samples=100,
    verbose=True,
)

print(f"\n{'='*60}")
print(f"  RESULT: Top1={result.top1_accuracy*100:.0f}% Top10={result.top10_overlap*100:.0f}%")
print(f"  Genome: {result.genome_size_mb:.1f} MB ({result.compression_ratio:.0f}x)")
print(f"  BPW: {result.bpw:.4f}")
print(f"  Time: {result.training_time:.0f}s ({result.training_time/60:.0f}min)")
print(f"  10T = {10e12 * result.bpw / 8 / 1e9:.1f} GB")
print(f"{'='*60}")

result.genome.save_genome("genome_sd128_28L_aggressive.pt")
