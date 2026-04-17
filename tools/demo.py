"""
DEMO: Generate text from an FRR compressed model.

Shows FRR output side-by-side with the original teacher model.
This is what you'd show on Hacker News or to investors.

Usage:
  python demo.py                          # Use latest trained model
  python demo.py --model frr_100k_best.pt # Specific checkpoint
  python demo.py --prompt "Your prompt"   # Custom prompt
"""
import lib.unbuffered
import torch, sys, os, argparse, time
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultracompress.inference import ModelConfig, MiniTransformer
from ultracompress.moonshot import FractalModel


def generate(model_fn, input_ids, max_tokens=100, temperature=0.7, device='cuda'):
    """Generate tokens autoregressively."""
    tokens = input_ids.clone()
    for _ in range(max_tokens):
        with torch.no_grad():
            logits = model_fn(tokens)
        next_logits = logits[0, -1] / temperature
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
        if next_token.item() in (151645, 151643):  # Qwen EOS tokens
            break
    return tokens[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None, help='FRR checkpoint path')
    parser.add_argument('--prompt', default=None, help='Text prompt')
    parser.add_argument('--max-tokens', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=0.7)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Find best model
    model_path = args.model
    if model_path is None:
        for candidate in ['frr_100k_best.pt', 'frr_optimized_50k.pt', 'frr_born_again_gen2.pt']:
            if os.path.exists(candidate):
                model_path = candidate
                break
    if model_path is None or not os.path.exists(model_path):
        print("No trained FRR model found. Run run_100k_train.py first.")
        print("Generating demo with UNTRAINED FRR (random weights)...")
        model_path = None

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)

    # Load teacher
    print("Loading Qwen3-0.6B teacher...")
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

    # Load FRR
    print("Loading FRR model...")
    frr = FractalModel(
        hidden_dim=1024, n_heads=16, n_scales=4, iters_per_scale=7,
        vocab_size=151936, ff_mult=1,
        embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w
    ).to(device)

    if model_path:
        frr.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded: {model_path}")
    frr.eval()

    teacher_params = sum(v.numel() for v in gd.values())
    frr_params = sum(p.numel() for p in frr.parameters())
    frr_block = sum(p.numel() for p in frr.block.parameters())

    print(f"\n{'='*60}")
    print(f"  ULTRACOMPRESS DEMO")
    print(f"  Teacher: Qwen3-0.6B ({teacher_params/1e6:.0f}M params, {teacher_params*2/1e6:.0f} MB)")
    print(f"  FRR: {frr_params/1e6:.1f}M params ({frr_block*2/1e6:.1f} MB block)")
    print(f"  Compression: {teacher_params/frr_params:.0f}x")
    print(f"{'='*60}")

    # Prompts
    prompts = [
        "The future of artificial intelligence is",
        "In a breakthrough discovery, scientists found that",
        "The most important principle in machine learning is",
        "Once upon a time in a world where robots",
    ]
    if args.prompt:
        prompts = [args.prompt]

    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

        print(f"\n--- Prompt: '{prompt}' ---")

        # Teacher generation
        t0 = time.perf_counter()
        teacher_out = generate(
            lambda t: teacher.forward(t, max_layers=28),
            input_ids, args.max_tokens, args.temperature, device)
        t_time = (time.perf_counter() - t0) * 1000
        teacher_text = tokenizer.decode(teacher_out, skip_special_tokens=True)

        # FRR generation
        t0 = time.perf_counter()
        frr_out = generate(
            lambda t: frr(t),
            input_ids, args.max_tokens, args.temperature, device)
        f_time = (time.perf_counter() - t0) * 1000
        frr_text = tokenizer.decode(frr_out, skip_special_tokens=True)

        print(f"  Teacher ({t_time:.0f}ms): {teacher_text[:300]}")
        print(f"  FRR     ({f_time:.0f}ms): {frr_text[:300]}")
        if t_time > 0:
            print(f"  Speed: FRR is {t_time/f_time:.1f}x {'faster' if f_time < t_time else 'slower'}")

    print(f"\n{'='*60}")
    print(f"  Compression: {teacher_params/frr_params:.0f}x smaller")
    print(f"  Block in L2 cache: {'Yes' if frr_block * 2 < 96e6 else 'No'} ({frr_block*2/1e6:.1f} MB)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
