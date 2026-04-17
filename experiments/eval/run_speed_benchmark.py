"""
SPEED BENCHMARK: FRR vs Teacher inference speed.
Tests the L2 cache hypothesis: FRR should be faster because the shared
block stays in cache, while the teacher loads 28 different layer weights.
"""
import lib.unbuffered
import torch, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultracompress.inference import ModelConfig, MiniTransformer
from ultracompress.moonshot import FractalModel

device = 'cuda'
WARMUP = 10
TRIALS = 100


def benchmark(model_fn, name, input_ids, n_warmup=WARMUP, n_trials=TRIALS):
    """Benchmark inference speed."""
    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            model_fn(input_ids)
    torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(n_trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model_fn(input_ids)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times) * 1000  # ms
    std = (sum((t - avg/1000)**2 for t in times) / len(times))**0.5 * 1000
    p50 = sorted(times)[len(times)//2] * 1000
    tps = input_ids.shape[1] / (avg / 1000)  # tokens per second

    print(f"  {name}:")
    print(f"    Mean: {avg:.2f} ms  Std: {std:.2f} ms  P50: {p50:.2f} ms")
    print(f"    Throughput: {tps:.0f} tokens/sec")
    return avg, tps


def main():
    print("=" * 60)
    print("SPEED BENCHMARK: FRR vs Teacher")
    print("=" * 60)

    # Load teacher
    print("\nLoading teacher...")
    wd = torch.load('qwen3_0.6b_cache.pt', weights_only=True)
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
    gd['output_norm.weight'] = wd.get('model.norm.weight', torch.ones(1024)).float()
    gd['output.weight'] = wd.get('lm_head.weight', gd['token_embd.weight']).float()
    for li in range(28):
        for h, g in hf_to_gguf.items():
            k = f'model.layers.{li}.{h}'
            if k in wd: gd[f'blk.{li}.{g}'] = wd[k].float()

    config = ModelConfig(n_layers=28, n_heads=16, n_kv_heads=8, hidden_size=1024,
                         intermediate_size=3072, vocab_size=151936, head_dim=128)
    teacher = MiniTransformer(config, device)
    teacher.load_weights(gd)
    teacher.embed_weight = teacher.embed_weight.to(device)
    if teacher.lm_head is not None: teacher.lm_head = teacher.lm_head.to(device)

    embed_w = gd['token_embd.weight'].to(device)
    norm_w = gd['output_norm.weight'].to(device)
    lm_head_w = gd['output.weight'].to(device)

    # Build FRR
    print("Building FRR...")
    frr = FractalModel(
        hidden_dim=1024, n_heads=16, n_scales=4, iters_per_scale=7,
        vocab_size=151936, ff_mult=1,
        embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w
    ).to(device)

    teacher_params = sum(v.numel() for v in gd.values())
    frr_params = sum(p.numel() for p in frr.parameters())
    print(f"Teacher: {teacher_params/1e6:.0f}M params")
    print(f"FRR: {frr_params/1e6:.1f}M params ({teacher_params/frr_params:.0f}x smaller)")

    # Benchmark at different sequence lengths
    for seq_len in [32, 64, 128, 256]:
        print(f"\n--- Sequence length: {seq_len} ---")
        input_ids = torch.randint(100, 50000, (1, seq_len), device=device)

        t_ms, t_tps = benchmark(
            lambda x: teacher.forward(x, max_layers=28),
            f"Teacher ({teacher_params/1e6:.0f}M params)",
            input_ids
        )
        f_ms, f_tps = benchmark(
            lambda x: frr(x),
            f"FRR ({frr_params/1e6:.1f}M params)",
            input_ids
        )

        speedup = t_ms / f_ms
        print(f"  Speedup: {speedup:.2f}x {'(FRR faster)' if speedup > 1 else '(Teacher faster)'}")

    # Memory comparison
    print(f"\n--- Memory ---")
    print(f"  Teacher VRAM: {teacher_params * 4 / 1e6:.0f} MB (FP32)")
    print(f"  FRR VRAM: {frr_params * 4 / 1e6:.0f} MB (FP32)")
    print(f"  FRR block only: {sum(p.numel() for p in frr.block.parameters()) * 2 / 1e6:.1f} MB (FP16)")
    print(f"  GPU L2 cache: ~96 MB (RTX 5090)")
    print(f"  Block fits in L2: YES" if sum(p.numel() for p in frr.block.parameters()) * 2 < 96e6 else "  Block fits in L2: NO")

    print("\nDone!")


if __name__ == '__main__':
    main()
