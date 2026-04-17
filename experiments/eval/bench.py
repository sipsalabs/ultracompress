"""
UltraCompress Benchmark Suite — Standardized evaluation for all approaches.

Usage:
    python bench.py --model frr_demo_model.pt --type frr
    python bench.py --model compressed.ucz --type ucz
    python bench.py --compare frr_demo_model.pt genome_best.pt

Metrics:
    - Top-1 / Top-10 token agreement with teacher
    - Perplexity on fixed prompts
    - Generation quality (side-by-side with teacher)
    - Compression ratio and model size
    - Inference speed (tokens/sec)
"""
import torch, sys, os, time, argparse, json
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultracompress.inference import ModelConfig, MiniTransformer


def load_teacher(device='cuda'):
    """Load Qwen3-0.6B teacher."""
    wd = torch.load('qwen3_0.6b_cache.pt', weights_only=True)
    hf_to_gguf = {'self_attn.q_proj.weight': 'attn_q.weight', 'self_attn.k_proj.weight': 'attn_k.weight', 'self_attn.v_proj.weight': 'attn_v.weight', 'self_attn.o_proj.weight': 'attn_output.weight', 'self_attn.q_norm.weight': 'attn_q_norm.weight', 'self_attn.k_norm.weight': 'attn_k_norm.weight', 'input_layernorm.weight': 'attn_norm.weight', 'post_attention_layernorm.weight': 'ffn_norm.weight', 'mlp.gate_proj.weight': 'ffn_gate.weight', 'mlp.up_proj.weight': 'ffn_up.weight', 'mlp.down_proj.weight': 'ffn_down.weight'}
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
    if teacher.lm_head is not None:
        teacher.lm_head = teacher.lm_head.to(device)
    return teacher, gd


def eval_token_agreement(student_fn, teacher, n=200, device='cuda'):
    """Standard eval: top-1 and top-10 token agreement."""
    t1, t10s = 0, []
    for trial in range(n):
        torch.manual_seed(trial * 13 + 9999)
        t = torch.randint(100, 50000, (1, 16), device=device)
        with torch.no_grad():
            tl = teacher.forward(t, max_layers=28)
            tp = tl[0, -1].argmax().item()
            tt10 = set(tl[0, -1].topk(10).indices.tolist())
            gl = student_fn(t)
            gp = gl[0, -1].argmax().item()
            gt10 = set(gl[0, -1].topk(10).indices.tolist())
            if tp == gp: t1 += 1
            t10s.append(len(tt10 & gt10) / 10)
    return {'top1': t1/n, 'top10': sum(t10s)/len(t10s)}


def eval_speed(student_fn, n=50, device='cuda'):
    """Measure inference speed (tokens/sec)."""
    # Warmup
    for _ in range(3):
        t = torch.randint(100, 50000, (1, 32), device=device)
        with torch.no_grad():
            student_fn(t)

    torch.cuda.synchronize()
    t0 = time.time()
    total_tokens = 0
    for _ in range(n):
        t = torch.randint(100, 50000, (1, 32), device=device)
        with torch.no_grad():
            student_fn(t)
        total_tokens += 32
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    return {'tokens_per_sec': total_tokens / elapsed, 'latency_ms': elapsed / n * 1000}


def eval_kl_divergence(student_fn, teacher, n=100, device='cuda'):
    """Average KL divergence from teacher."""
    total_kl = 0
    for trial in range(n):
        torch.manual_seed(trial * 7 + 1234)
        t = torch.randint(100, 50000, (1, 32), device=device)
        with torch.no_grad():
            tl = teacher.forward(t, max_layers=28)
            sl = student_fn(t)
            kl = F.kl_div(F.log_softmax(sl[:, -1, :], -1),
                         F.softmax(tl[:, -1, :], -1), reduction='batchmean')
            total_kl += kl.item()
    return {'avg_kl': total_kl / n}


def run_benchmark(student_fn, teacher, model_size_mb, model_name, device='cuda'):
    """Run full benchmark suite."""
    print(f"\n{'='*60}")
    print(f"BENCHMARK: {model_name}")
    print(f"{'='*60}")

    print("  Evaluating token agreement (200 trials)...")
    agreement = eval_token_agreement(student_fn, teacher, n=200, device=device)
    print(f"    Top-1: {agreement['top1']*100:.1f}%")
    print(f"    Top-10: {agreement['top10']*100:.1f}%")

    print("  Evaluating speed (50 batches)...")
    speed = eval_speed(student_fn, n=50, device=device)
    print(f"    Throughput: {speed['tokens_per_sec']:.0f} tokens/sec")
    print(f"    Latency: {speed['latency_ms']:.1f} ms/batch")

    print("  Evaluating KL divergence (100 trials)...")
    kl = eval_kl_divergence(student_fn, teacher, n=100, device=device)
    print(f"    Avg KL: {kl['avg_kl']:.4f}")

    teacher_size = 880.9  # Qwen3-0.6B layer params in MB
    compression = teacher_size / model_size_mb if model_size_mb > 0 else 0

    results = {
        'model': model_name,
        'size_mb': model_size_mb,
        'compression': compression,
        **agreement,
        **speed,
        **kl,
    }

    print(f"\n  SUMMARY: Top1={agreement['top1']*100:.0f}% Top10={agreement['top10']*100:.0f}% "
          f"Size={model_size_mb:.1f}MB ({compression:.0f}x) "
          f"Speed={speed['tokens_per_sec']:.0f}tok/s KL={kl['avg_kl']:.4f}")
    print(f"{'='*60}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Model path")
    parser.add_argument("--type", default="auto", help="Model type: frr, genome, ucz, auto")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print("Loading teacher...")
    teacher, gd = load_teacher(args.device)

    if args.model:
        print(f"Loading student: {args.model}")
        # Auto-detect type
        if args.type == "frr" or "frr" in args.model.lower():
            from ultracompress.moonshot import FractalModel
            embed = gd['token_embd.weight'].to(args.device)
            norm_w = gd['output_norm.weight'].to(args.device)
            lm_head = gd['output.weight'].to(args.device)

            ckpt = torch.load(args.model, weights_only=False)
            cfg = ckpt.get('config', {})
            model = FractalModel(
                hidden_dim=1024, n_heads=cfg.get('n_heads', 8),
                n_scales=cfg.get('n_scales', 4),
                iters_per_scale=cfg.get('iters_per_scale', 7),
                vocab_size=151936, ff_mult=cfg.get('ff_mult', 2),
                embed_weight=embed, lm_head_weight=lm_head, norm_weight=norm_w,
            ).to(args.device)
            model.block.load_state_dict(ckpt['block'])
            model.scale_gamma.data = ckpt['scale_gamma'].to(args.device)
            model.scale_beta.data = ckpt['scale_beta'].to(args.device)
            model.iter_scale.data = ckpt['iter_scale'].to(args.device)
            model.eval()

            size_mb = model.fractal_params() * 2 / 1e6
            student_fn = lambda t, _m=model: _m(t)
            run_benchmark(student_fn, teacher, size_mb, args.model, args.device)
        else:
            print(f"Unknown model type for {args.model}")
    else:
        # Benchmark teacher against itself (baseline)
        teacher_fn = lambda t: teacher.forward(t, max_layers=28)
        run_benchmark(teacher_fn, teacher, 880.9, "Teacher (Qwen3-0.6B)", args.device)
