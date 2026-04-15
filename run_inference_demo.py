"""
FRR Inference Demo — Generate text from a compressed FRR model.

Demonstrates that FRR can produce coherent text despite 52x compression.
Compares FRR output with teacher output side-by-side.

Usage:
  python run_inference_demo.py                                    # Default 1.7B best checkpoint
  python run_inference_demo.py --checkpoint path/to/ckpt.pt       # Custom checkpoint
  python run_inference_demo.py --teacher 0.6b                     # Use 0.6B teacher
  python run_inference_demo.py --prompt "The meaning of life is"  # Custom prompt
  python run_inference_demo.py --interactive                      # Interactive mode
"""
import lib.unbuffered
import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from ultracompress.inference import ModelConfig, MiniTransformer
from ultracompress.moonshot import FractalModel
from transformers import AutoTokenizer


TEACHER_CONFIGS = {
    '0.6b': {
        'cache': 'qwen3_0.6b_cache.pt',
        'hidden': 1024, 'n_heads': 16, 'n_kv_heads': 8, 'n_layers': 28,
        'intermediate_size': 3072, 'vocab_size': 151936, 'head_dim': 128,
        'ff_mult': 1,
    },
    '1.7b': {
        'cache': 'qwen3_1.7b_cache.pt',
        'hidden': 2048, 'n_heads': 16, 'n_kv_heads': 8, 'n_layers': 28,
        'intermediate_size': 6144, 'vocab_size': 151936, 'head_dim': 128,
        'ff_mult': 1,
    },
}

HF_TO_GGUF = {
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


def load_models(teacher_name: str, checkpoint: str, device: str):
    """Load teacher and FRR student."""
    cfg = TEACHER_CONFIGS[teacher_name]

    # Load teacher
    print(f"Loading {teacher_name} teacher...")
    wd = torch.load(cfg['cache'], weights_only=True)
    gd = {}
    gd['token_embd.weight'] = wd['model.embed_tokens.weight'].float()
    gd['output_norm.weight'] = wd.get('model.norm.weight', torch.ones(cfg['hidden'])).float()
    gd['output.weight'] = wd.get('lm_head.weight', gd['token_embd.weight']).float()
    for li in range(cfg['n_layers']):
        for h, g in HF_TO_GGUF.items():
            k = f'model.layers.{li}.{h}'
            if k in wd:
                gd[f'blk.{li}.{g}'] = wd[k].float()
    del wd

    model_cfg = ModelConfig(
        n_layers=cfg['n_layers'], n_heads=cfg['n_heads'], n_kv_heads=cfg['n_kv_heads'],
        hidden_size=cfg['hidden'], intermediate_size=cfg['intermediate_size'],
        vocab_size=cfg['vocab_size'], head_dim=cfg['head_dim'],
    )
    teacher = MiniTransformer(model_cfg, device)
    teacher.load_weights(gd)
    teacher.embed_weight = teacher.embed_weight.to(device)
    if teacher.lm_head is not None:
        teacher.lm_head = teacher.lm_head.to(device)

    embed_w = gd['token_embd.weight'].to(device)
    norm_w = gd['output_norm.weight'].to(device)
    lm_head_w = gd['output.weight'].to(device)
    del gd

    # Load FRR student
    print(f"Loading FRR: {checkpoint}")
    model = FractalModel(
        cfg['hidden'], cfg['n_heads'], 4, 7, cfg['vocab_size'], cfg['ff_mult'],
        embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w,
    ).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_teacher_params = sum(p.numel() for p in [embed_w, norm_w, lm_head_w]) + \
        cfg['n_layers'] * (4 * cfg['hidden'] * cfg['hidden'] + 3 * cfg['hidden'] * cfg['intermediate_size'])
    print(f"FRR: {trainable:,} trainable params")
    print(f"Teacher: ~{total_teacher_params/1e6:.0f}M params")
    print(f"Compression: {total_teacher_params/trainable:.0f}x\n")

    return teacher, model, cfg


@torch.no_grad()
def generate(model_fn, tokenizer, prompt: str, max_tokens: int = 100,
             temperature: float = 0.8, top_p: float = 0.9, top_k: int = 50,
             device: str = 'cuda:0') -> tuple[str, float]:
    """Generate text autoregressively. Returns (text, tokens_per_sec)."""
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    generated = input_ids.clone()

    t0 = time.time()
    for _ in range(max_tokens):
        logits = model_fn(generated)
        next_token_logits = logits[0, -1, :] / temperature

        # Top-K filtering
        if top_k > 0:
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][-1]
            next_token_logits[indices_to_remove] = float('-inf')

        # Top-P (nucleus) filtering
        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        next_token_logits[indices_to_remove] = float('-inf')

        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

        # Stop at EOS
        if next_token.item() == tokenizer.eos_token_id:
            break

    elapsed = time.time() - t0
    new_tokens = generated.shape[1] - input_ids.shape[1]
    tps = new_tokens / elapsed if elapsed > 0 else 0

    output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return output_text, tps


def compare_generation(teacher, model, cfg, tokenizer, prompt: str,
                       max_tokens: int, device: str):
    """Generate from both teacher and FRR, compare side-by-side."""
    n_layers = cfg['n_layers']

    print(f"{'─' * 60}")
    print(f"  Prompt: \"{prompt}\"")
    print(f"  Max tokens: {max_tokens}")
    print(f"{'─' * 60}")

    # Teacher generation
    print("\n  [TEACHER]")
    teacher_text, teacher_tps = generate(
        lambda t: teacher.forward(t, max_layers=n_layers),
        tokenizer, prompt, max_tokens, device=device,
    )
    print(f"  {teacher_text}")
    print(f"  ({teacher_tps:.1f} tok/s)")

    # FRR generation
    print("\n  [FRR (compressed)]")
    frr_text, frr_tps = generate(
        lambda t: model(t),
        tokenizer, prompt, max_tokens, device=device,
    )
    print(f"  {frr_text}")
    print(f"  ({frr_tps:.1f} tok/s, {frr_tps/teacher_tps:.1f}x faster)")

    # Token-level comparison
    t_tokens = tokenizer.encode(teacher_text)
    f_tokens = tokenizer.encode(frr_text)
    shared = set(t_tokens) & set(f_tokens)
    vocab_overlap = len(shared) / max(len(set(t_tokens)), len(set(f_tokens)), 1)
    print(f"\n  Vocabulary overlap: {vocab_overlap*100:.0f}%")


def main():
    parser = argparse.ArgumentParser(description="FRR Inference Demo")
    parser.add_argument('--checkpoint', default='checkpoints_1.7b_real_text/frr_1.7b_best.pt')
    parser.add_argument('--teacher', default='1.7b', choices=['0.6b', '1.7b'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--prompt', default=None)
    parser.add_argument('--max-tokens', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--interactive', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        print("Available checkpoints:")
        for d in ['checkpoints_1.7b_real_text', 'checkpoints_1.7b_cyclic_temp',
                   'checkpoints_1.7b_multi_temp']:
            if os.path.exists(d):
                for f in os.listdir(d):
                    if f.endswith('.pt'):
                        print(f"  {d}/{f}")
        sys.exit(1)

    print("=" * 60)
    print("FRR INFERENCE DEMO")
    print("=" * 60)

    teacher, model, cfg = load_models(args.teacher, args.checkpoint, args.device)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)

    # Demo prompts
    prompts = [
        "The future of artificial intelligence is",
        "In a galaxy far away, there existed",
        "The most important scientific discovery of the 21st century was",
        "Once upon a time, a programmer decided to",
        "The key to understanding quantum mechanics is",
    ]

    if args.prompt:
        prompts = [args.prompt]

    if args.interactive:
        print("\nInteractive mode. Type your prompt (or 'quit' to exit).\n")
        while True:
            prompt = input(">>> ").strip()
            if prompt.lower() in ('quit', 'exit', 'q'):
                break
            if not prompt:
                continue
            compare_generation(teacher, model, cfg, tokenizer, prompt,
                               args.max_tokens, args.device)
            print()
    else:
        for prompt in prompts:
            compare_generation(teacher, model, cfg, tokenizer, prompt,
                               args.max_tokens, args.device)
            print()

    print("=" * 60)
    print("Demo complete.")


if __name__ == "__main__":
    main()
