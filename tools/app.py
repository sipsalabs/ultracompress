"""
HuggingFace Spaces demo: Teacher vs FRR side-by-side.

Deploy to HuggingFace Spaces with:
  huggingface-cli upload mounnar/ultracompress-demo . --repo-type space

Requires: frr model checkpoint uploaded to the Space.
"""
try:
    import gradio as gr
    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False

import torch
import torch.nn.functional as F
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_models(device='cpu'):
    """Load teacher and FRR models."""
    from ultracompress.inference import ModelConfig, MiniTransformer
    from ultracompress.moonshot import FractalModel

    if not os.path.exists('qwen3_0.6b_cache.pt'):
        return None, None, None

    wd = torch.load('qwen3_0.6b_cache.pt', weights_only=True, map_location='cpu')
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

    frr = FractalModel(1024, 16, 4, 7, 151936, 1,
                       embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w).to(device)

    for ckpt in ['frr_100k_best.pt', 'frr_optimized_50k.pt']:
        if os.path.exists(ckpt):
            frr.load_state_dict(torch.load(ckpt, map_location=device))
            break

    frr.eval()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)

    return teacher, frr, tokenizer


def generate_text(model_fn, tokenizer, prompt, max_tokens=80, temperature=0.7):
    device = 'cpu'
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    tokens = input_ids.clone()

    t0 = time.perf_counter()
    for _ in range(max_tokens):
        with torch.no_grad():
            logits = model_fn(tokens)
        next_logits = logits[0, -1] / temperature
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
    elapsed = (time.perf_counter() - t0) * 1000

    text = tokenizer.decode(tokens[0], skip_special_tokens=True)
    return text, elapsed


def build_app():
    teacher, frr, tokenizer = load_models()

    def compare(prompt, max_tokens, temperature):
        if teacher is None:
            return "Models not loaded", "Models not loaded", ""

        t_text, t_time = generate_text(
            lambda t: teacher.forward(t, max_layers=28),
            tokenizer, prompt, int(max_tokens), float(temperature))
        f_text, f_time = generate_text(
            lambda t: frr(t),
            tokenizer, prompt, int(max_tokens), float(temperature))

        stats = (f"Teacher: {t_time:.0f}ms | FRR: {f_time:.0f}ms | "
                f"FRR is {t_time/f_time:.1f}x {'faster' if f_time < t_time else 'slower'} | "
                f"FRR size: 14.7 MB (60x smaller)")
        return t_text, f_text, stats

    with gr.Blocks(title="UltraCompress Demo") as app:
        gr.Markdown("# UltraCompress: 60x Model Compression Demo")
        gr.Markdown("Compare Qwen3-0.6B (1.5 GB) vs FRR compressed (14.7 MB)")

        with gr.Row():
            prompt = gr.Textbox(label="Prompt", value="The future of artificial intelligence is",
                              lines=2)
        with gr.Row():
            max_tokens = gr.Slider(10, 200, value=80, step=10, label="Max tokens")
            temperature = gr.Slider(0.1, 2.0, value=0.7, step=0.1, label="Temperature")

        btn = gr.Button("Generate", variant="primary")

        with gr.Row():
            teacher_out = gr.Textbox(label="Teacher (Qwen3-0.6B, 1.5 GB)", lines=8)
            frr_out = gr.Textbox(label="FRR Compressed (14.7 MB, 60x smaller)", lines=8)

        stats = gr.Textbox(label="Stats", lines=1)

        btn.click(compare, inputs=[prompt, max_tokens, temperature],
                 outputs=[teacher_out, frr_out, stats])

    return app


if __name__ == '__main__':
    if not HAS_GRADIO:
        print("Install gradio: pip install gradio")
        exit(1)
    app = build_app()
    app.launch(share=True)
