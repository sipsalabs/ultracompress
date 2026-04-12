"""
HuggingFace-compatible wrapper for FRR models.
Allows FRR compressed models to be loaded via AutoModel and evaluated
on standard benchmarks (lm-eval-harness, Open LLM Leaderboard, etc.)

Usage:
    from ultracompress.hf_model import FRRForCausalLM
    model = FRRForCausalLM.from_frr("frr_100k_best.pt", "Qwen/Qwen3-0.6B")
    # Use like any HuggingFace CausalLM
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FRRForCausalLM(nn.Module):
    """HuggingFace-compatible wrapper for FRR models.

    Makes FRR models work with:
    - transformers.generate()
    - lm-eval-harness
    - Standard HF pipelines
    """

    def __init__(self, frr_model, tokenizer_name="Qwen/Qwen3-0.6B"):
        super().__init__()
        self.frr = frr_model
        self.tokenizer_name = tokenizer_name
        self.config = type('Config', (), {
            'vocab_size': 151936,
            'hidden_size': frr_model.hidden_dim,
            'is_encoder_decoder': False,
            'pad_token_id': None,
            'eos_token_id': None,
        })()

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """Forward pass compatible with HF CausalLM interface."""
        logits = self.frr(input_ids)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )

        # Return a simple namespace that looks like HF output
        return type('Output', (), {
            'loss': loss,
            'logits': logits,
        })()

    def generate(self, input_ids, max_new_tokens=50, temperature=0.7,
                 do_sample=True, top_k=50, top_p=0.9, **kwargs):
        """Simple generation compatible with HF interface."""
        generated = input_ids.clone()
        for _ in range(max_new_tokens):
            logits = self.frr(generated)
            next_logits = logits[:, -1, :] / temperature

            if do_sample:
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                    next_logits[indices_to_remove] = float('-inf')

                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)

            # Check for EOS (Qwen uses 151645)
            if next_token.item() == 151645:
                break

        return generated

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}

    @classmethod
    def from_frr(cls, frr_path, base_model="Qwen/Qwen3-0.6B", device="cuda"):
        """Load an FRR model from a saved state dict.

        Args:
            frr_path: path to .pt file with FRR state dict
            base_model: HF model name for tokenizer + embeddings
            device: cuda or cpu
        """
        from ultracompress.moonshot import FractalModel

        # Load base model embeddings
        from transformers import AutoModelForCausalLM
        base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float32)
        base_sd = base.state_dict()

        embed_w = base_sd['model.embed_tokens.weight'].to(device)
        lm_head_w = base_sd.get('lm_head.weight', embed_w).to(device)
        norm_w = base_sd.get('model.norm.weight', torch.ones(embed_w.shape[1])).to(device)

        # Determine hidden size from embeddings
        hidden_dim = embed_w.shape[1]
        vocab_size = embed_w.shape[0]

        del base, base_sd

        # Build FRR
        frr = FractalModel(
            hidden_dim=hidden_dim, n_heads=16, n_scales=4, iters_per_scale=7,
            vocab_size=vocab_size, ff_mult=1,
            embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w
        ).to(device)

        # Load trained weights
        frr.load_state_dict(torch.load(frr_path, map_location=device))

        return cls(frr, base_model)

    def num_parameters(self, only_trainable=False):
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
