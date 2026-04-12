"""
Deployment utilities for FRR compressed models.

Load a compressed model and run inference in one line:
    from ultracompress.deploy import load_compressed
    model = load_compressed("compressed.ucz")
    output = model.generate("Hello world", max_tokens=100)
"""
import torch
import torch.nn.functional as F
import os
import sys
import json


class CompressedModel:
    """Lightweight wrapper for deployed FRR models."""

    def __init__(self, frr_model, tokenizer, device='cuda'):
        self.model = frr_model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def generate(self, prompt, max_tokens=100, temperature=0.7, top_k=50):
        """Generate text from a prompt string."""
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        tokens = input_ids.clone()

        with torch.no_grad():
            for _ in range(max_tokens):
                logits = self.model(tokens)
                next_logits = logits[0, -1] / temperature

                if top_k > 0:
                    indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][-1]
                    next_logits[indices_to_remove] = float('-inf')

                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)

                if next_token.item() in (151645, 151643):
                    break

        return self.tokenizer.decode(tokens[0], skip_special_tokens=True)

    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)

    @property
    def size_mb(self):
        return sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1e6

    @property
    def block_size_mb(self):
        return sum(p.numel() * p.element_size() for p in self.model.block.parameters()) / 1e6


def load_compressed(path, base_model="Qwen/Qwen3-0.6B", device='cuda'):
    """Load a compressed FRR model for inference.

    Supports:
      - .pt files (raw FRR state dict)
      - .ucz files (compressed archive)

    Returns a CompressedModel ready for generation.
    """
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ultracompress.moonshot import FractalModel

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    if path.endswith('.pt'):
        # Raw state dict
        from transformers import AutoModelForCausalLM
        base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float32,
                                                     device_map='cpu')
        base_sd = base.state_dict()
        embed_w = base_sd['model.embed_tokens.weight'].to(device)
        lm_head_w = base_sd.get('lm_head.weight', embed_w).to(device)
        norm_w = base_sd.get('model.norm.weight', torch.ones(embed_w.shape[1])).to(device)
        hidden_dim = embed_w.shape[1]
        vocab_size = embed_w.shape[0]
        del base, base_sd

        model = FractalModel(
            hidden_dim=hidden_dim, n_heads=16, n_scales=4, iters_per_scale=7,
            vocab_size=vocab_size, ff_mult=1,
            embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w
        ).to(device)
        model.load_state_dict(torch.load(path, map_location=device))

    elif path.endswith('.ucz'):
        # Compressed archive
        import zipfile
        import pickle
        from ultracompress.ultimate_pipeline import UltimatePipeline, UltimatePipelineConfig

        with zipfile.ZipFile(path, 'r') as zf:
            manifest = json.loads(zf.read('manifest.json'))
            compressed_block = pickle.loads(zf.read('block.compressed'))

            arch = manifest['architecture']
            hidden_dim = arch['hidden_dim']
            vocab_size = arch['vocab_size']

        # Decompress block
        pipe = UltimatePipeline(UltimatePipelineConfig(
            quant_bits=manifest['compression'].get('quant_bits', 2)))
        block_sd = pipe.decompress(compressed_block)

        # Load embeddings from base model
        from transformers import AutoModelForCausalLM
        base = AutoModelForCausalLM.from_pretrained(
            manifest.get('base_model', base_model),
            torch_dtype=torch.float32, device_map='cpu')
        base_sd = base.state_dict()
        embed_w = base_sd['model.embed_tokens.weight'].to(device)
        lm_head_w = base_sd.get('lm_head.weight', embed_w).to(device)
        norm_w = base_sd.get('model.norm.weight', torch.ones(hidden_dim)).to(device)
        del base, base_sd

        model = FractalModel(
            hidden_dim=hidden_dim, n_heads=arch.get('n_heads', 16),
            n_scales=arch.get('n_scales', 4),
            iters_per_scale=arch.get('iters_per_scale', 7),
            vocab_size=vocab_size, ff_mult=arch.get('ff_mult', 1),
            embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w
        ).to(device)

        # Load decompressed block weights
        model.block.load_state_dict({k: v.to(device) for k, v in block_sd.items()})

        # Load modulation from archive
        with zipfile.ZipFile(path, 'r') as zf:
            import numpy as np
            for name in ['scale_gamma', 'scale_beta', 'iter_scale']:
                buf = zf.read(f'modulation/{name}.bin')
                tensor = torch.from_numpy(np.frombuffer(buf, dtype=np.float16).copy())
                getattr(model, name).data = tensor.reshape(getattr(model, name).shape).float().to(device)

    else:
        raise ValueError(f"Unknown format: {path}. Use .pt or .ucz")

    return CompressedModel(model, tokenizer, device)
