"""Fast genome training from cached teacher outputs."""
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

print("=== CACHED TRAINING: 10-100x faster ===")
compressor = GenomeCompressor(model_weights=wd, model_config=config, device=device)

# Build cache once (slow but only once)
cache = compressor.build_cache(n_samples=5000, batch_size=16, seq_len=32, n_layers=28)

# Train genome from cache (FAST)
for sd in [128, 256]:
    print(f"\n--- sd={sd} from cache ---")
    result = compressor.compress_from_cache(
        cache, small_dim=sd, n_heads=4, n_steps=20000,
        batch_size=64, lr=0.001, eval_every=5000, verbose=True,
    )
    print(f"  Top1={result.top1_accuracy*100:.0f}% Top10={result.top10_overlap*100:.0f}%")
    print(f"  {result.genome_size_mb:.1f} MB ({result.compression_ratio:.0f}x) BPW={result.bpw:.4f}")
    print(f"  Time: {result.training_time:.0f}s")
    result.genome.save_genome(f"genome_sd{sd}_cached.pt")
    del result; torch.cuda.empty_cache()
