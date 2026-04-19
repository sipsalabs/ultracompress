"""
Auto-detecting teacher loader for Qwen3-family HF causal LMs.

One function, one call signature -- replaces the 60 lines of hardcoded
Qwen3-1.7B boilerplate duplicated across every training and eval script
in this repo. Auto-detects n_layers, hidden_size, n_heads, n_kv_heads,
head_dim, vocab_size, and intermediate_size directly from the state
dict, so the same code path works on Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B,
Qwen3-8B, etc.

Known limitations (documented, not hidden):
  - MiniTransformer does NOT apply Qwen3's per-head Q/K RMSNorm. This
    makes our "teacher logits" differ from HF Qwen3's official logits
    by a small amount. All distillation numbers in this repo are
    relative to the no-QK-norm teacher. See REPRODUCE.md and
    KNOWN_ISSUES.md.
  - Only Qwen3-family state-dict naming is supported today. Llama-3
    works if the script is extended because the naming matches, minus
    the q_norm/k_norm keys. Phi, Gemma need their own loaders.

Usage:
    from scaling.teacher_loader import load_qwen3_teacher

    teacher, embed_w, lm_head_w, norm_w, cfg = load_qwen3_teacher(
        'qwen3_1.7b_cache.pt', device='cuda:0')

    # teacher.forward(tokens, max_layers=cfg.n_layers) -> (B, T, vocab)
"""
import os
import sys
from dataclasses import dataclass

import torch

# allow `from ultracompress.inference import ...` when caller cwd'd to repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ultracompress.inference import ModelConfig, MiniTransformer  # noqa: E402


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


@dataclass
class TeacherBundle:
    """Everything a training script needs from the teacher."""
    teacher: MiniTransformer
    embed_w: torch.Tensor   # (vocab, hidden) -- on device
    lm_head_w: torch.Tensor # (vocab, hidden) -- on device
    norm_w: torch.Tensor    # (hidden,)       -- on device
    cfg: ModelConfig
    cache_path: str
    # convenience aliases
    h_outer: int
    vocab_size: int
    n_layers: int


def _auto_detect_config(wd: dict) -> ModelConfig:
    """Derive ModelConfig from a raw HF state dict with zero hardcoded numbers."""
    embed = wd.get('model.embed_tokens.weight')
    if embed is None:
        raise ValueError("Not a Qwen3/Llama-family state dict (no model.embed_tokens.weight)")
    vocab_size, hidden = embed.shape

    layer_indices = [int(k.split('.')[2])
                     for k in wd.keys()
                     if k.startswith('model.layers.') and k.count('.') >= 2]
    if not layer_indices:
        raise ValueError("No model.layers.* keys found in state dict")
    n_layers = max(layer_indices) + 1

    q_proj_shape = wd['model.layers.0.self_attn.q_proj.weight'].shape
    k_proj_shape = wd['model.layers.0.self_attn.k_proj.weight'].shape

    # Prefer Qwen3's per-head q_norm for exact head_dim. Otherwise fall back
    # to the common Qwen/Llama value of 128, and finally to hidden // 16.
    q_norm = wd.get('model.layers.0.self_attn.q_norm.weight')
    if q_norm is not None:
        head_dim = q_norm.numel()
    elif hidden % 128 == 0:
        head_dim = 128
    else:
        head_dim = hidden // 16
    n_heads = q_proj_shape[0] // head_dim
    n_kv_heads = k_proj_shape[0] // head_dim

    intermediate = wd['model.layers.0.mlp.up_proj.weight'].shape[0]

    return ModelConfig(
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        hidden_size=hidden,
        intermediate_size=intermediate,
        vocab_size=vocab_size,
        head_dim=head_dim,
    )


def load_qwen3_teacher(cache_path: str, device: str = 'cuda:0',
                       verbose: bool = True) -> TeacherBundle:
    """Load a cached Qwen3-family HF state dict into a MiniTransformer teacher.

    Parameters
    ----------
    cache_path : str
        Path to a .pt file produced by
        `torch.save(AutoModelForCausalLM.from_pretrained(...).state_dict(), ...)`
    device : str
        Target CUDA device, e.g. 'cuda:0'.
    verbose : bool
        If True, print detected config.

    Returns
    -------
    TeacherBundle
    """
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Teacher cache not found: {cache_path}")

    wd = torch.load(cache_path, weights_only=True, map_location='cpu')
    cfg = _auto_detect_config(wd)

    # Build gguf-style dict for MiniTransformer.load_weights()
    gd = {}
    gd['token_embd.weight'] = wd['model.embed_tokens.weight'].float()
    gd['output_norm.weight'] = wd.get('model.norm.weight', torch.ones(cfg.hidden_size)).float()
    gd['output.weight'] = wd.get('lm_head.weight', gd['token_embd.weight']).float()
    for li in range(cfg.n_layers):
        for hf_key, gguf_key in HF_TO_GGUF.items():
            full = f'model.layers.{li}.{hf_key}'
            if full in wd:
                gd[f'blk.{li}.{gguf_key}'] = wd[full].float()
    del wd

    teacher = MiniTransformer(cfg, device)
    teacher.load_weights(gd)
    teacher.embed_weight = teacher.embed_weight.to(device)
    if teacher.lm_head is not None:
        teacher.lm_head = teacher.lm_head.to(device)

    embed_w = gd['token_embd.weight'].to(device)
    norm_w = gd['output_norm.weight'].to(device)
    lm_head_w = gd['output.weight'].to(device)
    del gd

    if verbose:
        print(f"[teacher_loader] {cache_path}")
        print(f"  n_layers={cfg.n_layers} hidden={cfg.hidden_size} "
              f"n_heads={cfg.n_heads} n_kv_heads={cfg.n_kv_heads} "
              f"head_dim={cfg.head_dim} ffn={cfg.intermediate_size} "
              f"vocab={cfg.vocab_size}")

    return TeacherBundle(
        teacher=teacher,
        embed_w=embed_w,
        lm_head_w=lm_head_w,
        norm_w=norm_w,
        cfg=cfg,
        cache_path=cache_path,
        h_outer=cfg.hidden_size,
        vocab_size=cfg.vocab_size,
        n_layers=cfg.n_layers,
    )
