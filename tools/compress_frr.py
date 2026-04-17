#!/usr/bin/env python3
"""
One-command FRR compression.

Usage:
  python compress_frr.py --model Qwen/Qwen3-0.6B --steps 50000
  python compress_frr.py --model Qwen/Qwen3-1.7B --steps 15000
  python compress_frr.py --model ./my_model/ --steps 30000 --output my_model_frr.pt

Takes any HuggingFace causal LM, distills it into FRR, saves the result.
"""
import lib.unbuffered
import argparse
import torch
import torch.nn.functional as F
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Compress any LLM with FRR")
    parser.add_argument('--model', required=True, help='HuggingFace model name or local path')
    parser.add_argument('--steps', type=int, default=15000, help='Training steps (more = better quality)')
    parser.add_argument('--output', default=None, help='Output path (default: <model>_frr.pt)')
    parser.add_argument('--device', default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--seq-len', type=int, default=32, help='Sequence length')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--ff-mult', type=int, default=1, help='FFN multiplier (1=smaller block)')
    args = parser.parse_args()

    device = args.device
    model_name = args.model
    output_path = args.output or f"{model_name.split('/')[-1]}_frr.pt"

    print(f"{'='*60}")
    print(f"  FRR COMPRESSION")
    print(f"  Model: {model_name}")
    print(f"  Steps: {args.steps}")
    print(f"  Output: {output_path}")
    print(f"{'='*60}")

    # Load model
    print(f"\nLoading {model_name}...")
    from transformers import AutoModelForCausalLM, AutoConfig
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    hidden_size = config.hidden_size
    n_heads = config.num_attention_heads
    n_kv_heads = getattr(config, 'num_key_value_heads', n_heads)
    intermediate_size = config.intermediate_size
    n_layers = config.num_hidden_layers
    vocab_size = config.vocab_size
    head_dim = hidden_size // n_heads

    print(f"Architecture: {n_layers} layers, hidden={hidden_size}, "
          f"heads={n_heads}, kv_heads={n_kv_heads}, vocab={vocab_size}")

    # Load weights
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, device_map='cpu', trust_remote_code=True)
    sd = hf_model.state_dict()

    # Convert to our format
    from ultracompress.inference import ModelConfig, MiniTransformer

    hf_to_gguf = {
        'self_attn.q_proj.weight': 'attn_q.weight', 'self_attn.k_proj.weight': 'attn_k.weight',
        'self_attn.v_proj.weight': 'attn_v.weight', 'self_attn.o_proj.weight': 'attn_output.weight',
        'self_attn.q_norm.weight': 'attn_q_norm.weight', 'self_attn.k_norm.weight': 'attn_k_norm.weight',
        'input_layernorm.weight': 'attn_norm.weight', 'post_attention_layernorm.weight': 'ffn_norm.weight',
        'mlp.gate_proj.weight': 'ffn_gate.weight', 'mlp.up_proj.weight': 'ffn_up.weight',
        'mlp.down_proj.weight': 'ffn_down.weight',
    }
    gd = {}
    gd['token_embd.weight'] = sd['model.embed_tokens.weight'].float()
    gd['output_norm.weight'] = sd.get('model.norm.weight', torch.ones(hidden_size)).float()
    gd['output.weight'] = sd.get('lm_head.weight', gd['token_embd.weight']).float()
    for li in range(n_layers):
        for h, g in hf_to_gguf.items():
            k = f'model.layers.{li}.{h}'
            if k in sd: gd[f'blk.{li}.{g}'] = sd[k].float()

    del hf_model, sd
    import gc; gc.collect()

    # Build teacher
    mc = ModelConfig(n_layers=n_layers, n_heads=n_heads, n_kv_heads=n_kv_heads,
                     hidden_size=hidden_size, intermediate_size=intermediate_size,
                     vocab_size=vocab_size, head_dim=head_dim)
    teacher = MiniTransformer(mc, device)
    teacher.load_weights(gd)
    teacher.embed_weight = teacher.embed_weight.to(device)
    if teacher.lm_head is not None: teacher.lm_head = teacher.lm_head.to(device)

    embed_w = gd['token_embd.weight'].to(device)
    norm_w = gd['output_norm.weight'].to(device)
    lm_head_w = gd['output.weight'].to(device)

    teacher_layer_params = sum(v.numel() for k, v in gd.items() if k.startswith('blk.'))
    teacher_total = sum(v.numel() for v in gd.values())

    # Build FRR
    from ultracompress.moonshot import FractalModel

    iters = n_layers // 4
    model = FractalModel(
        hidden_dim=hidden_size, n_heads=n_heads, n_scales=4, iters_per_scale=iters,
        vocab_size=vocab_size, ff_mult=args.ff_mult,
        embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    compression = teacher_layer_params / trainable
    total_size = trainable * 2 / 1e6  # FP16 MB

    print(f"\nFRR: {trainable:,} trainable params ({total_size:.1f} MB FP16)")
    print(f"Compression: {compression:.1f}x on layer params")
    print(f"Teacher: {teacher_total:,} total params ({teacher_total*2/1e6:.0f} MB FP16)")

    # Train
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)

    print(f"\nDistilling ({args.steps} steps)...")
    t0 = time.time()
    for step in range(args.steps):
        torch.manual_seed(step * 7)
        bs = min(args.batch_size, max(1, 32 // (hidden_size // 512)))
        tokens = torch.randint(100, min(50000, vocab_size), (bs, args.seq_len), device=device)

        with torch.no_grad():
            teacher_logits = teacher.forward(tokens, max_layers=n_layers)
        student_logits = model(tokens)

        T = max(2.0, 5.0 * (1 - step / args.steps))
        loss = F.kl_div(
            F.log_softmax(student_logits / T, dim=-1),
            F.softmax(teacher_logits / T, dim=-1),
            reduction='batchmean') * (T * T)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()

        if step % max(1, args.steps // 10) == 0 or step == args.steps - 1:
            elapsed = time.time() - t0
            eta = elapsed / (step + 1) * (args.steps - step - 1)
            print(f"  Step {step}/{args.steps}: loss={loss.item():.4f} "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    # Save
    torch.save(model.state_dict(), output_path)
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"\nSaved: {output_path} ({size_mb:.1f} MB)")
    print(f"Original: {teacher_total*2/1e6:.0f} MB (FP16)")
    print(f"Compressed: {size_mb:.1f} MB")
    print(f"Ratio: {teacher_total*2/1e6/size_mb:.1f}x")
    print(f"Time: {time.time()-t0:.0f}s")
    print("Done!")


if __name__ == '__main__':
    main()
