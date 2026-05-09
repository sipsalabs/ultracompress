"""Streaming layer-wise compression runner for Track A (GSQ + V18-C).

Applies Track A compression ONE TRANSFORMER LAYER AT A TIME to sidestep the
all-at-once OOM that kills 32B/72B runs. Novel patent angle for Track A v3
supplement: layer-wise streaming compression with inter-layer hidden-state
distillation (not logit-level KL).

Architecture:
  1. Load teacher in 4-bit NF4 across both GPUs, cache per-layer hidden states
     on calibration set to disk.
  2. Free teacher entirely.
  3. For each transformer layer i:
     a. Load layer i fp16 weights from HF shards.
     b. Apply GSQ 5bpw + B=64 to each Linear.
     c. Initialize V18-C (V[r=32], U[r=32], alpha=0) per Linear with SVD warm-start.
     d. Hidden-state distillation: train V/U to minimize MSE between student
        layer output and teacher's cached hidden state for layer i+1.
     e. Save compressed layer to disk.
     f. Free everything.
  4. Eval via streaming forward (Track C pattern).
  5. Output traceability JSON.

Hardware target: dual RTX 5090 (32GB each), 128GB RAM, cuda:0 preferred.
Validated on Qwen3-8B first; 32B/72B after pipeline proven.

Usage:
  python streaming_compression_runner.py --model qwen3-8b
  python streaming_compression_runner.py --model qwen3-8b --bpw 4
"""
import argparse
import datetime
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

from safetensors import safe_open

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    'qwen3-1.7b': {
        'hf_id': 'Qwen/Qwen3-1.7B',
        'params_B': 1.7,
        'dtype': torch.bfloat16,
        'n_layers': 28,
    },
    'qwen3-8b': {
        'hf_id': 'Qwen/Qwen3-8B',
        'params_B': 8.2,
        'dtype': torch.bfloat16,
        'n_layers': 36,
    },
    'qwen3-14b': {
        'hf_id': 'Qwen/Qwen3-14B',
        'params_B': 14.8,
        'dtype': torch.bfloat16,
        'n_layers': 40,
    },
    'qwen3-32b': {
        'hf_id': 'Qwen/Qwen3-32B',
        'params_B': 32.5,
        'dtype': torch.bfloat16,
        'n_layers': 64,
    },
    'qwen25-72b': {
        'hf_id': 'Qwen/Qwen2.5-72B',
        'params_B': 72.7,
        'dtype': torch.bfloat16,
        'n_layers': 80,
    },
    # --- Non-Qwen architectures (multi-arch transfer validation) ---
    'mistral-7b-v03': {
        'hf_id': 'mistralai/Mistral-7B-v0.3',
        'params_B': 7.2,
        'dtype': torch.bfloat16,
        'n_layers': 32,
    },
    'llama-3.1-8b': {
        'hf_id': 'NousResearch/Meta-Llama-3.1-8B',  # ungated mirror
        'params_B': 8.0,
        'dtype': torch.bfloat16,
        'n_layers': 32,
    },
    'llama-3.1-70b': {
        'hf_id': 'NousResearch/Meta-Llama-3.1-70B',
        'params_B': 70.0,
        'dtype': torch.bfloat16,
        'n_layers': 80,
    },
    'hermes-3-405b': {
        'hf_id': 'NousResearch/Hermes-3-Llama-3.1-405B',
        'params_B': 405.0,
        'dtype': torch.bfloat16,
        'n_layers': 126,
    },
    'qwen3-235b-a22b': {
        'hf_id': 'Qwen/Qwen3-235B-A22B',
        'params_B': 235.0,
        'dtype': torch.bfloat16,
        'n_layers': 94,
    },
    'mixtral-8x7b': {
        'hf_id': 'mistralai/Mixtral-8x7B-v0.1',
        'params_B': 46.7,
        'dtype': torch.bfloat16,
        'n_layers': 32,
    },
    'mixtral-8x22b': {
        'hf_id': 'mistralai/Mixtral-8x22B-v0.1',
        'params_B': 141.0,
        'dtype': torch.bfloat16,
        'n_layers': 56,
    },
    'phi-3-5-moe': {
        'hf_id': 'microsoft/Phi-3.5-MoE-instruct',
        'params_B': 41.9,
        'dtype': torch.bfloat16,
        'n_layers': 32,
    },
    # --- 2026-05-08 GPU 1 conveyor belt additions (small/fast dense archs) ---
    'smollm2-1.7b': {
        'hf_id': 'HuggingFaceTB/SmolLM2-1.7B',
        'params_B': 1.7,
        'dtype': torch.bfloat16,
        'n_layers': 24,
    },
    'tinyllama-1.1b-chat': {
        'hf_id': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        'params_B': 1.1,
        'dtype': torch.bfloat16,
        'n_layers': 22,
    },
    'qwen3-0.6b': {
        'hf_id': 'Qwen/Qwen3-0.6B',
        'params_B': 0.6,
        'dtype': torch.bfloat16,
        'n_layers': 28,
    },
    'olmo-2-0425-1b': {
        'hf_id': 'allenai/OLMo-2-0425-1B',
        'params_B': 1.0,
        'dtype': torch.bfloat16,
        'n_layers': 16,
    },
    'qwen3-1.7b-base': {
        'hf_id': 'Qwen/Qwen3-1.7B-Base',
        'params_B': 1.7,
        'dtype': torch.bfloat16,
        'n_layers': 28,
    },
    'olmo-2-0425-1b-instruct': {
        'hf_id': 'allenai/OLMo-2-0425-1B-Instruct',
        'params_B': 1.0,
        'dtype': torch.bfloat16,
        'n_layers': 16,
    },
    'smollm2-1.7b-instruct': {
        'hf_id': 'HuggingFaceTB/SmolLM2-1.7B-Instruct',
        'params_B': 1.7,
        'dtype': torch.bfloat16,
        'n_layers': 24,
    },
    'phi-3-mini-4k-instruct': {
        'hf_id': 'microsoft/Phi-3-mini-4k-instruct',
        'params_B': 3.8,
        'dtype': torch.bfloat16,
        'n_layers': 32,
    },
    'yi-1.5-9b': {
        'hf_id': '01-ai/Yi-1.5-9B',
        'params_B': 8.8,
        'dtype': torch.bfloat16,
        'n_layers': 48,
    },
    'phi-2': {
        'hf_id': 'microsoft/phi-2',
        'params_B': 2.7,
        'dtype': torch.bfloat16,
        'n_layers': 32,
    },
}

TARGET_SUBS = ('q_proj', 'k_proj', 'v_proj', 'o_proj',
               'gate_proj', 'up_proj', 'down_proj',
               # MoE expert linear naming used by Mixtral and Phi-3.5-MoE
               # (block_sparse_moe.experts.<i>.w1/w2/w3)
               'w1', 'w2', 'w3',
               # State-space model (Mamba/Mamba-2/RWKV/Jamba) Linear naming
               # MambaBlock contains in_proj, x_proj, dt_proj, out_proj — bulk parameters
               # Verified compress losslessly with mean rel_l2=0.0459 on Mamba-2.8B (2026-05-08)
               'in_proj', 'x_proj', 'dt_proj', 'out_proj')

DEVICE = torch.device('cuda:0')
DATA_CANDIDATES = [
    _ROOT / 'fineweb_edu_500M_tokens.pt',
    _ROOT / 'fineweb_edu_100M_tokens.pt',
]


def get_data_candidates_for_model(model_key: str) -> list[Path]:
    """Per-model FineWeb-edu token cache (vocab compatibility for non-Qwen archs)."""
    slug = model_key.replace('-', '_').replace('.', '_').replace('/', '_')
    candidates = [
        _ROOT / f'fineweb_edu_10M_tokens_{slug}.pt',
        _ROOT / f'fineweb_edu_100M_tokens_{slug}.pt',
        _ROOT / f'fineweb_edu_500M_tokens_{slug}.pt',
    ]
    if model_key.startswith('qwen3') or model_key.startswith('qwen25'):
        candidates.extend(DATA_CANDIDATES)
    return candidates

# ---------------------------------------------------------------------------
# Quantizer: GSQ (k-means learned grid) — imported from scaling_curve_runner
# ---------------------------------------------------------------------------
_NF_LEVELS = {
    2: torch.tensor([-1.0, 1.0]),
    4: torch.tensor([-1.0, -0.318, 0.318, 1.0]),
    8: torch.tensor([-1.0, -0.5784, -0.3186, -0.1025,
                     0.1025, 0.3186, 0.5784, 1.0]),
    16: torch.tensor([
        -1.0, -0.6962, -0.5251, -0.3949, -0.2845, -0.1849, -0.0912, -0.0287,
        0.0287, 0.0912, 0.1849, 0.2845, 0.3949, 0.5251, 0.6962, 1.0,
    ]),
    32: torch.tensor([
        -1.0, -0.7979, -0.6567, -0.5549, -0.4729, -0.4019, -0.3378, -0.2786,
        -0.2225, -0.1689, -0.1168, -0.0656, -0.0219, 0.0000, 0.0219, 0.0656,
        0.1168, 0.1689, 0.2225, 0.2786, 0.3378, 0.4019, 0.4729, 0.5549,
        0.6567, 0.7979, 1.0, -0.8765, -0.9394, 0.8765, 0.9394, -0.0437,
    ]),
}


def compute_awq_scales(
    W: torch.Tensor, X_calib: torch.Tensor, alpha: float = 0.5
) -> torch.Tensor:
    """AWQ-style per-input-channel activation-aware rescaling (arxiv:2306.00978).

    Computes per-channel salience from calibration activations and returns a
    scale vector that, when applied as W' = W * diag(s), reduces quantization
    residual on salient channels.  After quantization of W', the inverse scale
    is applied: W_deq = W'_deq / s.  The scale vector is stored in the per-
    Linear codec dict for lossless reconstruction.

    Args:
        W: weight tensor, shape (out_dim, in_dim).
        X_calib: calibration activations, shape (n_samples, in_dim).
            Obtained from cached teacher hidden states (layer input).
        alpha: salience exponent.  0 = no rescaling; 1 = full salience scaling.
            Default 0.5 per AWQ paper recommendation.

    Returns:
        scale: (in_dim,) fp32 tensor, >=1e-6 clamped.

    Decision criteria (document for lab notebook):
        PPL_r <= 1.0040  -->  WIN
        1.0040 < PPL_r <= 1.0048  -->  NEUTRAL (within noise)
        PPL_r > 1.0048  -->  REFUTED
    """
    # Per-input-channel salience: mean |activation| across samples
    # X_calib may be (n_samples, seq_len, in_dim) or (n_samples, in_dim)
    salience = X_calib.float().abs().reshape(-1, W.shape[1]).mean(dim=0)
    # Normalize salience to [0, 1] range for numerical stability
    salience = salience / salience.max().clamp(min=1e-12)
    # Scale: s_c = salience_c ^ alpha.  Higher salience -> larger scale -> less
    # quantization error on that channel (the weight column is stretched before
    # quantization, giving the grid finer effective resolution).
    scale = salience.pow(alpha).clamp(min=1e-6).to(W.device)
    return scale


def gsq_quantize_weight(W: torch.Tensor, bpw: int,
                        block: int = 64, gsq_steps: int = 50,
                        return_codec: bool = False):
    """GSQ: learned scalar grid via k-means on pooled normalized weights.

    If return_codec=True, returns (Wq, grid_final, codes, absmax) tuple where:
      - Wq: dequantized weight (original return)
      - grid_final: (K,) sorted learned grid in fp32
      - codes: (out_dim, n_blocks, block) int16 tensor of grid indices (0..K-1)
      - absmax: (out_dim, n_blocks, 1) fp32 per-block scales
    These together let the v0.3 pack format reconstruct Wq exactly:
      Wq = (grid_final[codes] * absmax).reshape(W.shape)
    """
    K = 1 << bpw
    out_dim, in_dim = W.shape

    if block <= 0 or in_dim % block != 0:
        block = 128
        if in_dim % block != 0:
            # Fallback: per-row scalar quant (uniform symmetric grid)
            half = 2 ** bpw // 2
            rm = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
            Wq = ((W / rm * half).round().clamp(-half, half - 1) / half) * rm
            if return_codec:
                # Synthesize codec for fallback path: grid is uniform symmetric, codes derive from W/rm
                grid_uniform = torch.arange(-half, half, device=W.device, dtype=torch.float32) / half
                codes_full = ((W / rm * half).round().clamp(-half, half - 1) + half).to(torch.int16)
                # Reshape to (out_dim, 1, in_dim) since block_size = in_dim in fallback
                codes_full = codes_full.unsqueeze(1)
                absmax_full = rm.float().unsqueeze(-1)
                return Wq, grid_uniform.cpu(), codes_full.cpu(), absmax_full.cpu()
            return Wq

    # Per-block absmax normalization
    Wb = W.reshape(out_dim, in_dim // block, block).float()
    absmax = Wb.abs().amax(dim=-1, keepdim=True).clamp(min=1e-9)
    Wn = Wb / absmax

    # Initialize grid from NF codebook
    if K in _NF_LEVELS:
        grid = _NF_LEVELS[K][:K].clone().float().to(W.device)
    else:
        grid = torch.linspace(-1.0, 1.0, K, device=W.device)

    # Force near-zero level
    if grid.abs().min() > 0.05:
        closest_to_zero = grid.abs().argmin()
        grid[closest_to_zero] = 0.0

    Wn_flat = Wn.reshape(-1)
    N = Wn_flat.shape[0]

    # Sub-sample for centroid updates — DETERMINISTIC via local CPU generator
    # (2026-05-07: prior runs used global torch.randint which made K-means
    # non-deterministic across re-compressions; PPL drift of +13% observed.)
    sample_size = min(N, 1 << 18)
    if N > sample_size:
        g = torch.Generator().manual_seed(42)
        idx_cpu = torch.randint(0, N, (sample_size,), generator=g)
        idx_sample = idx_cpu.to(W.device)
        w_sample = Wn_flat[idx_sample]
    else:
        w_sample = Wn_flat

    # K-means iterations
    for step in range(gsq_steps):
        dists = (w_sample.unsqueeze(-1) - grid.unsqueeze(0)).abs()
        idx = dists.argmin(dim=-1)
        new_grid = grid.clone()
        for k in range(K):
            mask = (idx == k)
            count = mask.sum()
            if count > 0:
                new_grid[k] = w_sample[mask].mean()
        delta = (new_grid - grid).abs().max().item()
        grid = new_grid
        if delta < 1e-5:
            break

    # Hard assignment with learned grid (chunked)
    with torch.no_grad():
        grid_final = grid.detach().sort().values
        n_blocks = Wn.shape[1]
        Wn_q = torch.empty_like(Wn)
        # Allocate full codes tensor only when return_codec is requested (saves 4x VRAM otherwise)
        codes_full = torch.empty(out_dim, n_blocks, block, dtype=torch.int16, device=W.device) if return_codec else None
        chunk_rows = max(1, (256 * 1024 * 1024) // (n_blocks * block * K * 4))
        for r_start in range(0, out_dim, chunk_rows):
            r_end = min(r_start + chunk_rows, out_dim)
            chunk = Wn[r_start:r_end]
            dists_chunk = (chunk.unsqueeze(-1) - grid_final.view(1, 1, 1, -1)).abs()
            idx_chunk = dists_chunk.argmin(dim=-1)
            Wn_q[r_start:r_end] = grid_final[idx_chunk]
            if codes_full is not None:
                codes_full[r_start:r_end] = idx_chunk.to(torch.int16)
        Wq = Wn_q * absmax

    Wq_out = Wq.reshape(out_dim, in_dim).to(W.dtype)
    if return_codec:
        return Wq_out, grid_final.cpu(), codes_full.cpu(), absmax.cpu()
    return Wq_out


# ---------------------------------------------------------------------------
# V18-C correction module (layer-local, trainable)
# ---------------------------------------------------------------------------
class CorrectionMatrixC(nn.Module):
    """y = W_base @ x + alpha * U(V(x))"""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None,
                 rank: int = 32):
        super().__init__()
        self.register_buffer('W_base', weight.detach())
        self.register_buffer('bias_buf', bias.detach() if bias is not None else None)
        nr, hd = weight.shape
        device = weight.device
        self.V = nn.Linear(hd, rank, bias=False, device=device)
        self.U = nn.Linear(rank, nr, bias=False, device=device)
        self.alpha = nn.Parameter(torch.zeros(1, device=device))

    def init_from_svd(self, original_weight: torch.Tensor) -> None:
        """Warm-start V/U from SVD of quantization residual."""
        residual = (original_weight - self.W_base).float()
        try:
            U_svd, S_svd, Vt_svd = torch.linalg.svd(residual, full_matrices=False)
            rank = self.V.weight.shape[0]
            # V.weight: [rank, in_dim], U.weight: [out_dim, rank]
            self.V.weight.data.copy_(
                (Vt_svd[:rank] * S_svd[:rank].unsqueeze(1).sqrt()).to(self.V.weight.dtype)
            )
            self.U.weight.data.copy_(
                (U_svd[:, :rank] * S_svd[:rank].unsqueeze(0).sqrt()).to(self.U.weight.dtype)
            )
            self.alpha.data.fill_(1.0)
        except Exception:
            # Fallback: random init
            nn.init.normal_(self.V.weight, std=0.01)
            nn.init.normal_(self.U.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xd = x.dtype
        yb = F.linear(x, self.W_base.to(xd),
                      self.bias_buf.to(xd) if self.bias_buf is not None else None)
        v_out = F.linear(x, self.V.weight.to(xd))
        u_out = F.linear(v_out.float(), self.U.weight.float())
        return yb + self.alpha.to(xd) * u_out.to(xd)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def vram_gb(device: torch.device, peak: bool = False) -> float:
    if peak:
        return torch.cuda.max_memory_allocated(device) / 1e9
    return torch.cuda.memory_allocated(device) / 1e9


def vram_report() -> str:
    parts = []
    for i in range(min(2, torch.cuda.device_count())):
        a = torch.cuda.memory_allocated(i) / 1e9
        parts.append(f'GPU{i}={a:.1f}GB')
    return '  '.join(parts)


def free_memory() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def get_model_classes(hf_id: str):
    """Return (DecoderLayerClass, RotaryEmbeddingClass) for the given model.

    Supports Qwen3, Qwen2/Qwen2.5, Mistral, and Llama architectures.
    """
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(hf_id, trust_remote_code=True)
    model_type = config.model_type

    if model_type == 'qwen3':
        from transformers.models.qwen3.modeling_qwen3 import (
            Qwen3DecoderLayer,
            Qwen3RotaryEmbedding,
        )
        return Qwen3DecoderLayer, Qwen3RotaryEmbedding
    elif model_type == 'qwen3_moe':
        from transformers.models.qwen3_moe.modeling_qwen3_moe import (
            Qwen3MoeDecoderLayer,
            Qwen3MoeRotaryEmbedding,
        )
        return Qwen3MoeDecoderLayer, Qwen3MoeRotaryEmbedding
    elif model_type in ('qwen2', 'qwen2_5'):
        from transformers.models.qwen2.modeling_qwen2 import (
            Qwen2DecoderLayer,
            Qwen2RotaryEmbedding,
        )
        return Qwen2DecoderLayer, Qwen2RotaryEmbedding
    elif model_type == 'mistral':
        from transformers.models.mistral.modeling_mistral import (
            MistralDecoderLayer,
            MistralRotaryEmbedding,
        )
        return MistralDecoderLayer, MistralRotaryEmbedding
    elif model_type == 'llama':
        from transformers.models.llama.modeling_llama import (
            LlamaDecoderLayer,
            LlamaRotaryEmbedding,
        )
        return LlamaDecoderLayer, LlamaRotaryEmbedding
    elif model_type == 'mixtral':
        from transformers.models.mixtral.modeling_mixtral import (
            MixtralDecoderLayer,
            MixtralRotaryEmbedding,
        )
        return MixtralDecoderLayer, MixtralRotaryEmbedding
    elif model_type == 'phimoe':
        from transformers.models.phimoe.modeling_phimoe import (
            PhimoeDecoderLayer,
            PhimoeRotaryEmbedding,
        )
        return PhimoeDecoderLayer, PhimoeRotaryEmbedding
    elif model_type == 'phi3':
        from transformers.models.phi3.modeling_phi3 import (
            Phi3DecoderLayer,
            Phi3RotaryEmbedding,
        )
        return Phi3DecoderLayer, Phi3RotaryEmbedding
    elif model_type == 'phi':
        # Phi-1 / Phi-2 — different module than phi3
        from transformers.models.phi.modeling_phi import (
            PhiDecoderLayer,
            PhiRotaryEmbedding,
        )
        return PhiDecoderLayer, PhiRotaryEmbedding
    elif model_type == 'olmo2':
        from transformers.models.olmo2.modeling_olmo2 import (
            Olmo2DecoderLayer,
            Olmo2RotaryEmbedding,
        )
        return Olmo2DecoderLayer, Olmo2RotaryEmbedding
    elif model_type == 'olmo':
        from transformers.models.olmo.modeling_olmo import (
            OlmoDecoderLayer,
            OlmoRotaryEmbedding,
        )
        return OlmoDecoderLayer, OlmoRotaryEmbedding
    else:
        # Fallback: try Qwen3 (original behavior)
        from transformers.models.qwen3.modeling_qwen3 import (
            Qwen3DecoderLayer,
            Qwen3RotaryEmbedding,
        )
        return Qwen3DecoderLayer, Qwen3RotaryEmbedding


# ---------------------------------------------------------------------------
# Phase 1: Cache teacher hidden states
# ---------------------------------------------------------------------------
@torch.no_grad()
def cache_teacher_hidden_states(
    hf_id: str,
    dtype: torch.dtype,
    calibration_ids: list[torch.Tensor],
    cache_dir: Path,
    n_layers: int,
) -> None:
    """Load teacher in NF4 4-bit, run calibration forward, save per-layer hiddens.

    Saves:
      cache_dir/hidden_layer_{i:03d}.pt  -- [n_prompts, seq_len, hidden_dim] bf16
      for i in 0..n_layers (inclusive: layer 0 = embedding output, layer N = post-final-layer)
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check if cache already complete
    manifest_path = cache_dir / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get('hf_id') == hf_id
                and manifest.get('n_layers') == n_layers
                and manifest.get('n_prompts') == len(calibration_ids)):
            all_exist = all(
                (cache_dir / f'hidden_layer_{i:03d}.pt').exists()
                for i in range(n_layers + 1)
            )
            if all_exist:
                print(f'  Teacher hidden cache already complete at {cache_dir}')
                return

    print(f'  Loading teacher (NF4 4-bit) for hidden state caching...')
    t0 = time.time()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
    )
    max_memory = {0: '28GiB', 1: '28GiB', 'cpu': '80GiB'}

    teacher = AutoModelForCausalLM.from_pretrained(
        hf_id,
        quantization_config=bnb_config,
        device_map='auto',
        max_memory=max_memory,
        dtype=dtype,
        attn_implementation='eager',
    )
    teacher.train(False)
    print(f'  Teacher loaded in {time.time() - t0:.1f}s  {vram_report()}')

    # Determine input device
    embed_device = teacher.model.embed_tokens.weight.device
    n_prompts = len(calibration_ids)

    # Forward pass: collect hidden states after each layer
    # We do this in chunks to avoid OOM on activations
    print(f'  Caching hidden states for {n_prompts} prompts, {n_layers} layers...')

    # First pass: get embedding output (hidden_layer_000)
    all_embeds = []
    for prompt_ids in calibration_ids:
        ids = prompt_ids.unsqueeze(0).to(embed_device)
        emb = teacher.model.embed_tokens(ids).to(torch.bfloat16)
        all_embeds.append(emb.cpu())

    hidden_0 = torch.cat(all_embeds, dim=0)  # [n_prompts, seq_len, hidden]
    torch.save(hidden_0, cache_dir / 'hidden_layer_000.pt')
    print(f'    Saved hidden_layer_000.pt ({hidden_0.shape}, {hidden_0.nbytes / 1e6:.1f}MB)')
    del all_embeds

    # Full forward to get all intermediate hidden states
    # Use hooks to capture outputs
    hidden_states_per_layer: dict[int, list[torch.Tensor]] = {
        i: [] for i in range(n_layers)
    }

    hooks = []

    def make_hook(layer_idx: int):
        def hook_fn(module, input, output):
            # Qwen3DecoderLayer returns (hidden_states, ...) or just hidden_states
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            hidden_states_per_layer[layer_idx].append(h.detach().cpu().to(torch.bfloat16))
        return hook_fn

    for i in range(n_layers):
        h = teacher.model.layers[i].register_forward_hook(make_hook(i))
        hooks.append(h)

    # Run forward on each prompt individually to save memory
    for pi, prompt_ids in enumerate(calibration_ids):
        ids = prompt_ids.unsqueeze(0).long().to(embed_device)
        _ = teacher(input_ids=ids, use_cache=False, return_dict=True)
        if (pi + 1) % 20 == 0:
            print(f'    Forward {pi + 1}/{n_prompts}  {vram_report()}')

    # Remove hooks
    for h in hooks:
        h.remove()

    # Save per-layer hidden states
    for i in range(n_layers):
        layer_hidden = torch.cat(hidden_states_per_layer[i], dim=0)
        path = cache_dir / f'hidden_layer_{i + 1:03d}.pt'
        torch.save(layer_hidden, path)
        mb = path.stat().st_size / 1e6
        if (i + 1) % 10 == 0 or i == 0:
            print(f'    Saved hidden_layer_{i + 1:03d}.pt ({layer_hidden.shape}, {mb:.1f}MB)')
        del layer_hidden
        hidden_states_per_layer[i] = []

    del hidden_states_per_layer

    # Save manifest
    manifest = {
        'hf_id': hf_id,
        'n_layers': n_layers,
        'n_prompts': n_prompts,
        'seq_len': calibration_ids[0].shape[0],
        'dtype': 'bfloat16',
        'built_at': datetime.datetime.now().isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Free teacher
    del teacher
    free_memory()
    elapsed = time.time() - t0
    print(f'  Teacher hidden cache complete in {elapsed:.1f}s')


# ---------------------------------------------------------------------------
# Lazy layer loader: reads only one layer's tensors from safetensors shards
# ---------------------------------------------------------------------------
_LAYER_LOAD_MODEL_DIR_CACHE: dict[str, Path] = {}


def load_layer_state_dict(
    hf_id: str, layer_idx: int, device: str | torch.device, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    """Load only `model.layers.{layer_idx}.*` tensors from HF safetensors shards.

    Uses huggingface_hub snapshot_download to resolve the local cache path, then
    reads individual tensors via safetensors.safe_open — never loads the full
    model into RAM. Memory cost: only the target layer's parameters.
    """
    from huggingface_hub import snapshot_download

    # Cache the resolved model directory across calls (same session, same model)
    if hf_id not in _LAYER_LOAD_MODEL_DIR_CACHE:
        model_dir = Path(snapshot_download(
            hf_id, allow_patterns=["*.json", "*.safetensors"]
        ))
        _LAYER_LOAD_MODEL_DIR_CACHE[hf_id] = model_dir
    else:
        model_dir = _LAYER_LOAD_MODEL_DIR_CACHE[hf_id]

    # Read the weight-map index
    index_file = model_dir / "model.safetensors.index.json"
    if not index_file.exists():
        # Single-shard model (unlikely for 72B, but handle gracefully)
        shard_files = list(model_dir.glob("*.safetensors"))
        if not shard_files:
            raise FileNotFoundError(
                f"No safetensors files found in {model_dir}"
            )
        # Fallback: scan all shards for matching keys
        layer_prefix = f"model.layers.{layer_idx}."
        sd: dict[str, torch.Tensor] = {}
        for sf in shard_files:
            with safe_open(str(sf), framework="pt", device="cpu") as f:
                for k in f.keys():
                    if k.startswith(layer_prefix):
                        stripped = k[len(layer_prefix):]
                        sd[stripped] = f.get_tensor(k).to(device=device, dtype=dtype)
        return sd

    with open(index_file) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    layer_prefix = f"model.layers.{layer_idx}."
    layer_keys = [k for k in weight_map if k.startswith(layer_prefix)]

    if not layer_keys:
        raise KeyError(
            f"No keys found for layer {layer_idx} (prefix={layer_prefix!r}) "
            f"in {index_file}"
        )

    # Group keys by shard file
    shards: dict[str, list[str]] = {}
    for k in layer_keys:
        shards.setdefault(weight_map[k], []).append(k)

    # Load tensors from each shard (only the keys we need)
    sd = {}
    for shard_file, keys in shards.items():
        shard_path = str(model_dir / shard_file)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for k in keys:
                stripped = k[len(layer_prefix):]
                sd[stripped] = f.get_tensor(k).to(device=device, dtype=dtype)

    return sd


# ---------------------------------------------------------------------------
# Phase 2: Layer-wise compression
# ---------------------------------------------------------------------------
def compress_single_layer(
    hf_id: str,
    dtype: torch.dtype,
    layer_idx: int,
    n_layers: int,
    hidden_cache_dir: Path,
    output_dir: Path,
    bpw: int = 5,
    block_size: int = 64,
    rank: int = 32,
    train_steps: int = 200,
    train_lr: float = 1e-3,
    train_bs: int = 8,
    device: torch.device = DEVICE,
) -> dict[str, float]:
    """Compress a single transformer layer: GSQ + V18-C with hidden-state distillation.

    Returns per-layer metrics dict.

    Per-Linear adaptive train-steps (UC_ADAPTIVE_TRAIN_STEPS=1): scales train_steps
    proportionally to layer depth. Layer 0 = base_steps; final layer = 5x base_steps.
    Empirically motivated by today's observation that train_loss_final scales 5000x
    from layer 0 (0.0002) to layer 27 (1.05) on Qwen3-1.7B-Base under uniform fixed
    steps — deep layers are V18-C-undertrained at fixed schedule.
    """
    t0 = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    if os.environ.get('UC_ADAPTIVE_TRAIN_STEPS') == '1':
        # Linear ramp 1.0 → 5.0 over n_layers
        depth_frac = layer_idx / max(n_layers - 1, 1)
        scaled_steps = int(round(train_steps * (1.0 + 4.0 * depth_frac)))
        if scaled_steps != train_steps:
            print(f'    [adaptive_train_steps] layer {layer_idx}/{n_layers}: '
                  f'{train_steps} -> {scaled_steps} steps', flush=True)
            train_steps = scaled_steps

    # V3 cure: rank-redistribution at constant total V18-C parameter budget.
    # pm-agent finding 2026-05-09: deep layers (23-27 of 28) have train_loss_final
    # 4000x larger than shallow layers; they're RANK-bound not STEPS-bound.
    # Linear ramp rank_layer_i = round(16 + 32 * depth_frac) gives:
    #   layer 0/N    -> rank 16  (saturated; less rank to spare)
    #   layer N/2    -> rank 32  (default)
    #   layer N-1/N  -> rank 48  (residual-heavy; more rank to fix)
    # Total V18-C parameter budget across n_layers held at n_layers * 32. Predicted
    # PPL ratio 1.0030-1.0035 (3 sigma above noise) per RESEARCH_v3_CURE_DIRECTION.
    # Opt in via UC_RANK_REDISTRIBUTE=1.
    if os.environ.get('UC_RANK_REDISTRIBUTE') == '1':
        depth_frac = layer_idx / max(n_layers - 1, 1)
        scaled_rank = int(round(rank * 0.5 + rank * depth_frac))  # 0.5x → 1.5x of base
        if scaled_rank != rank:
            print(f'    [rank_redistribute] layer {layer_idx}/{n_layers}: '
                  f'rank {rank} -> {scaled_rank}', flush=True)
            rank = scaled_rank

    # Load teacher hidden states for this layer (input) and next (target)
    input_hidden = torch.load(
        hidden_cache_dir / f'hidden_layer_{layer_idx:03d}.pt',
        map_location='cpu', weights_only=True,
    )  # [n_prompts, seq_len, hidden_dim]
    target_hidden = torch.load(
        hidden_cache_dir / f'hidden_layer_{layer_idx + 1:03d}.pt',
        map_location='cpu', weights_only=True,
    )  # [n_prompts, seq_len, hidden_dim]

    n_prompts, seq_len, hidden_dim = input_hidden.shape

    # Load only this layer's weights from the model
    config = AutoConfig.from_pretrained(hf_id, trust_remote_code=True)
    config._attn_implementation = 'eager'

    # Lazy-load ONLY this layer's tensors from safetensors shards on disk.
    # Never loads the full model into RAM — critical for 72B (144GB bf16 > 128GB RAM).
    print(f'    Loading layer {layer_idx} weights (lazy safetensors)...')

    DecoderLayerClass, RotaryEmbClass = get_model_classes(hf_id)

    layer_sd = load_layer_state_dict(hf_id, layer_idx, device, dtype)

    # Instantiate decoder layer on meta first, then use assign=True to
    # replace meta placeholders with real tensors from state_dict.
    with torch.device('meta'):
        layer = DecoderLayerClass(config, layer_idx=layer_idx)
    layer.load_state_dict(layer_sd, strict=True, assign=True)
    # Now all parameters are on `device` with correct dtype
    layer = layer.to(device=device, dtype=dtype)
    del layer_sd
    free_memory()

    # Store original weights for SVD warm-start (before quantization)
    original_weights: dict[str, torch.Tensor] = {}
    for name, mod in layer.named_modules():
        if isinstance(mod, nn.Linear) and any(s in name for s in TARGET_SUBS):
            original_weights[name] = mod.weight.data.clone()

    # V4-A: AWQ-style per-channel activation-aware rescaling (arxiv:2306.00978).
    # Gated behind UC_AWQ_SCALING=1.  Computes per-input-channel salience from
    # teacher hidden states (this layer's input) and pre-scales W before GSQ,
    # so salient channels get finer effective quantization resolution.
    # The scale vector is stored in gsq_codecs[name]['awq_scale'] and applied
    # inversely at reconstruction: W_deq = gsq_deq(W_scaled) / s.
    use_awq = os.environ.get('UC_AWQ_SCALING') == '1'
    awq_alpha = float(os.environ.get('UC_AWQ_ALPHA', '0.5'))
    if use_awq:
        print(f'    [V4-A AWQ] enabled  alpha={awq_alpha}', flush=True)

    # Apply GSQ quantization to each Linear in the layer.
    # Capture (grid, codes, absmax) for v0.3 pack codec persistence.
    n_quantized = 0
    quant_errors: dict[str, float] = {}
    gsq_codecs: dict[str, dict[str, torch.Tensor]] = {}  # name -> {grid, codes, absmax}
    for name, mod in layer.named_modules():
        if isinstance(mod, nn.Linear) and any(s in name for s in TARGET_SUBS):
            with torch.no_grad():
                W = mod.weight.data.float()
                # Per-Linear adaptive bpw: bottleneck Linears (k_proj in GQA models)
                # carry +9.9% quant error vs other-Linear baseline at uniform bpw.
                # Promote them to bpw+1 — see docs/RESEARCH_PROPOSAL_PER_LINEAR_ADAPTIVE_BPW.
                # Storage cost ~+0.16% on a 1.7B model. Opt in via UC_ADAPTIVE_BPW=1 env var.
                this_bpw = bpw
                if os.environ.get('UC_ADAPTIVE_BPW') == '1' and 'k_proj' in name:
                    this_bpw = min(bpw + 1, 8)

                # V4-A AWQ pre-scaling: W' = W * diag(s)
                awq_scale = None
                if use_awq:
                    awq_scale = compute_awq_scales(
                        W, input_hidden.reshape(-1, hidden_dim), alpha=awq_alpha
                    )
                    W = W * awq_scale.unsqueeze(0)  # broadcast (out, in) * (1, in)

                Wq, grid, codes, absmax = gsq_quantize_weight(
                    W, this_bpw, block_size, return_codec=True
                )

                # V4-A AWQ inverse scaling: W_deq = Wq / diag(s)
                if awq_scale is not None:
                    Wq = Wq / awq_scale.unsqueeze(0)

                # Relative L2 error (measured against ORIGINAL unscaled W)
                W_orig = mod.weight.data.float()
                rel_l2 = (W_orig - Wq).norm() / W_orig.norm()
                quant_errors[name] = rel_l2.item()
                mod.weight.data.copy_(Wq.to(dtype))
                # Save codec components — these enable lossless v0.3 pack reconstruction.
                # Storage cost ≈ 1.0% of W bytes (codes) + tiny grid + per-block absmax.
                codec_entry: dict[str, torch.Tensor] = {
                    'grid': grid,                    # (K,) fp32
                    'codes': codes,                  # (out_dim, n_blocks, block) int16
                    'absmax': absmax.squeeze(-1),    # (out_dim, n_blocks) fp32
                }
                if awq_scale is not None:
                    codec_entry['awq_scale'] = awq_scale.cpu()  # (in_dim,) fp32
                gsq_codecs[name] = codec_entry
                del W, W_orig, Wq, grid, codes, absmax, awq_scale
            n_quantized += 1
    free_memory()

    # Wrap each quantized Linear with V18-C correction (SVD warm-start)
    correction_modules: dict[str, CorrectionMatrixC] = {}

    def wrap_linears_with_correction(module: nn.Module, prefix: str = '') -> None:
        for name, child in list(module.named_children()):
            full_name = f'{prefix}.{name}' if prefix else name
            if isinstance(child, nn.Linear) and any(s in name for s in TARGET_SUBS):
                cm = CorrectionMatrixC(
                    child.weight.data, child.bias.data if child.bias is not None else None,
                    rank=rank,
                )
                # SVD warm-start from residual
                if full_name in original_weights:
                    cm.init_from_svd(original_weights[full_name].to(device))
                setattr(module, name, cm)
                correction_modules[full_name] = cm
            else:
                wrap_linears_with_correction(child, full_name)

    wrap_linears_with_correction(layer)
    del original_weights
    free_memory()

    # Freeze base weights, only train V/U/alpha
    for p in layer.parameters():
        p.requires_grad = False
    for cm in correction_modules.values():
        for p in cm.V.parameters():
            p.requires_grad = True
        for p in cm.U.parameters():
            p.requires_grad = True
        cm.alpha.requires_grad = True

    # Hidden-state distillation: train V/U to minimize MSE between
    # student layer output and teacher's next-layer hidden state
    trainable_params = [p for p in layer.parameters() if p.requires_grad]
    n_train_params = sum(p.numel() for p in trainable_params)

    opt = torch.optim.Adam(trainable_params, lr=train_lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=train_steps)

    layer.train()
    losses: list[float] = []

    # We need position embeddings for the layer forward
    # Build them once
    from transformers.masking_utils import create_causal_mask

    try:
        rotary_emb = RotaryEmbClass(config=config, device=device)
    except TypeError:
        # Older signatures (e.g. PhimoeRotaryEmbedding) only take config.
        rotary_emb = RotaryEmbClass(config=config).to(device)

    for step in range(train_steps):
        # Sample a mini-batch from calibration set
        batch_idx = torch.randint(0, n_prompts, (train_bs,))
        x_batch = input_hidden[batch_idx].to(device=device, dtype=dtype)  # [bs, seq, hid]
        y_target = target_hidden[batch_idx].to(device=device, dtype=dtype)  # [bs, seq, hid]

        bs_actual, sl, _ = x_batch.shape

        # Build position state
        cache_position = torch.arange(sl, device=device)
        position_ids = cache_position.unsqueeze(0).expand(bs_actual, -1)

        # Causal mask
        causal_mask = create_causal_mask(
            config=config,
            input_embeds=x_batch,
            attention_mask=None,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids,
        )

        # RoPE
        try:
            position_embeddings = rotary_emb(x_batch, position_ids)
        except (RuntimeError, TypeError):
            # PhimoeRotaryEmbedding-style: forward(x, seq_len:int)
            position_embeddings = rotary_emb(x_batch, sl)

        # Forward through this layer
        layer_out = layer(
            x_batch,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        if isinstance(layer_out, tuple):
            h_out = layer_out[0]
        else:
            h_out = layer_out

        # MSE loss against teacher's hidden state
        loss = F.mse_loss(h_out.float(), y_target.float())

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        opt.step()
        sched.step()

        losses.append(loss.item())

        if (step + 1) % 50 == 0:
            avg_loss = sum(losses[-50:]) / len(losses[-50:])
            print(f'      step {step + 1}/{train_steps}  MSE={avg_loss:.6f}  '
                  f'{vram_report()}')

    layer.train(False)

    # Save compressed layer
    # Extract: quantized W_base + V/U/alpha per Linear + other layer params + GSQ codec
    save_dict: dict[str, Any] = {
        'layer_idx': layer_idx,
        'bpw': bpw,
        'block_size': block_size,
        'rank': rank,
        'state_dict': {},
        'corrections': {},
        'gsq_codecs': {},  # v0.3 codec persistence — enables lossless `uc pack` round-trip
    }

    # Save full layer state_dict (includes W_base buffers in CorrectionMatrixC)
    save_dict['state_dict'] = {k: v.cpu() for k, v in layer.state_dict().items()}

    # Also save correction parameters separately for clarity
    for name, cm in correction_modules.items():
        save_dict['corrections'][name] = {
            'V_weight': cm.V.weight.data.cpu(),
            'U_weight': cm.U.weight.data.cpu(),
            'alpha': cm.alpha.data.cpu().item(),
        }

    # Persist the GSQ k-means codec (grid + codes + absmax) per quantized Linear.
    # Gives `uc pack` v0.3 the components it needs for lossless reconstruction.
    save_dict['gsq_codecs'] = gsq_codecs

    output_path = output_dir / f'layer_{layer_idx:03d}.pt'
    torch.save(save_dict, output_path)

    # Metrics
    final_loss = sum(losses[-20:]) / max(1, len(losses[-20:]))
    mean_quant_error = sum(quant_errors.values()) / max(1, len(quant_errors))
    peak_vram = vram_gb(device, peak=True)
    elapsed = time.time() - t0

    metrics = {
        'layer_idx': layer_idx,
        'train_loss_final': final_loss,
        'mean_quant_rel_l2': mean_quant_error,
        'n_quantized_linears': n_quantized,
        'n_train_params': n_train_params,
        'peak_vram_gb': peak_vram,
        'compress_time_s': elapsed,
        'per_linear_quant_errors': quant_errors,
    }

    # Cleanup
    del layer, correction_modules, trainable_params, opt, sched
    del input_hidden, target_hidden, rotary_emb
    free_memory()
    torch.cuda.reset_peak_memory_stats(device)

    return metrics


# ---------------------------------------------------------------------------
# Phase 3: Streaming eval (Track C pattern)
# ---------------------------------------------------------------------------
@torch.no_grad()
def streaming_eval_ppl(
    hf_id: str,
    compressed_dir: Path,
    eval_prompts: list[torch.Tensor],
    device: torch.device = DEVICE,
) -> tuple[float, float]:
    """Evaluate PPL using streaming forward through compressed layers.

    Returns (compressed_ppl, peak_vram_gb).
    """
    config = AutoConfig.from_pretrained(hf_id, trust_remote_code=True)
    config._attn_implementation = 'eager'
    n_layers = config.num_hidden_layers
    dtype = torch.bfloat16

    DecoderLayerClass, RotaryEmbClass = get_model_classes(hf_id)
    from transformers.masking_utils import create_causal_mask

    # Load scaffold (embed, norm, lm_head) — these are small, load from HF
    print('  Loading scaffold (embed + norm + lm_head)...')
    # Phi-2 family compatibility: explicit .weight/.bias keys + embed_dropout +
    # rotary_emb. HF prefix-match has been observed to fail for some configs.
    scaffold_device_map = {
        'model.embed_tokens': str(device),
        'model.embed_tokens.weight': str(device),
        'model.norm': str(device),
        'model.norm.weight': str(device),
        'model.final_layernorm': str(device),       # Phi-2 family
        'model.final_layernorm.weight': str(device),
        'model.final_layernorm.bias': str(device),
        'model.embed_dropout': str(device),         # Phi-2
        'model.rotary_emb': str(device),            # Phi-2
        'lm_head': str(device),
        'lm_head.weight': str(device),
        'lm_head.bias': str(device),
    }
    for i in range(n_layers):
        scaffold_device_map[f'model.layers.{i}'] = 'meta'

    scaffold = AutoModelForCausalLM.from_pretrained(
        hf_id,
        device_map=scaffold_device_map,
        dtype=dtype,
        attn_implementation='eager',
        low_cpu_mem_usage=True,
    )

    embed_tokens = scaffold.model.embed_tokens
    final_norm = scaffold.model.norm
    lm_head = scaffold.lm_head

    try:
        rotary_emb = RotaryEmbClass(config=config, device=device)
    except TypeError:
        # Older signatures (e.g. PhimoeRotaryEmbedding) only take config.
        rotary_emb = RotaryEmbClass(config=config).to(device)

    # Free everything else from scaffold
    del scaffold
    free_memory()

    torch.cuda.reset_peak_memory_stats(device)

    nlls: list[float] = []

    for pi, prompt_ids in enumerate(eval_prompts):
        ids = prompt_ids.unsqueeze(0).long().to(device)
        bsz, seqlen = ids.shape

        # Embed
        hidden = embed_tokens(ids).to(dtype)

        # Position state
        cache_position = torch.arange(seqlen, device=device)
        position_ids = cache_position.unsqueeze(0).expand(bsz, -1)
        causal_mask = create_causal_mask(
            config=config,
            input_embeds=hidden,
            attention_mask=None,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids,
        )
        try:
            position_embeddings = rotary_emb(hidden, position_ids)
        except (RuntimeError, TypeError):
            # PhimoeRotaryEmbedding-style: forward(x, seq_len:int)
            position_embeddings = rotary_emb(hidden, seqlen)

        # Stream each compressed layer
        for i in range(n_layers):
            layer_path = compressed_dir / f'layer_{i:03d}.pt'
            layer_data = torch.load(layer_path, map_location=device, weights_only=False)

            # Reconstruct layer from saved state_dict.
            # The saved state_dict contains CorrectionMatrixC keys (W_base, V,
            # U, alpha) for wrapped linears and normal keys for everything else.
            # Strategy: build a fresh layer on meta, wrap target linears with
            # CorrectionMatrixC (using corrections dict for shape info), then
            # load the full state_dict with assign=True.
            corrections_data = layer_data.get('corrections', {})
            layer_sd = {k: v.to(device) for k, v in layer_data['state_dict'].items()}

            with torch.device('meta'):
                layer = DecoderLayerClass(config, layer_idx=i)

            # Wrap target linears with CorrectionMatrixC BEFORE loading state
            for name in corrections_data:
                cd = corrections_data[name]
                rank = cd['V_weight'].shape[0]
                parts = name.split('.')
                parent = layer
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                attr = parts[-1]
                mod = getattr(parent, attr)
                # Get W_base shape from state_dict
                w_base_key = f'{name}.W_base'
                w_base_shape = layer_sd[w_base_key].shape
                # Check if bias_buf exists in the saved state_dict
                bias_key = f'{name}.bias_buf'
                has_bias = bias_key in layer_sd
                # Use dummy tensors on meta for init, assign=True will replace
                with torch.device('meta'):
                    dummy_w = torch.empty(w_base_shape, dtype=dtype)
                    dummy_bias = torch.empty(w_base_shape[0], dtype=dtype) if has_bias else None
                cm = CorrectionMatrixC(dummy_w, dummy_bias, rank=rank)
                setattr(parent, attr, cm)

            # Load full state_dict (assign=True replaces meta placeholders)
            layer.load_state_dict(layer_sd, strict=True, assign=True)
            layer = layer.to(device=device, dtype=dtype)
            del layer_sd

            layer.train(False)

            # Forward
            layer_out = layer(
                hidden,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            if isinstance(layer_out, tuple):
                hidden = layer_out[0]
            else:
                hidden = layer_out

            del layer, layer_data
            torch.cuda.synchronize(device)
            free_memory()

        # Final norm + lm_head
        hidden = final_norm(hidden)
        logits = lm_head(hidden)

        # NLL
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = ids[:, 1:].contiguous()
        nll = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='mean',
        ).item()
        nlls.append(nll)

        if (pi + 1) % 10 == 0:
            running_ppl = math.exp(sum(nlls) / len(nlls))
            print(f'    eval {pi + 1}/{len(eval_prompts)}  '
                  f'running_PPL={running_ppl:.4f}  {vram_report()}')

        del hidden, logits, ids
        free_memory()

    ppl = math.exp(sum(nlls) / len(nlls))
    peak = vram_gb(device, peak=True)

    del embed_tokens, final_norm, lm_head, rotary_emb
    free_memory()

    return ppl, peak


@torch.no_grad()
def baseline_ppl(
    hf_id: str,
    dtype: torch.dtype,
    eval_prompts: list[torch.Tensor],
    device: torch.device = DEVICE,
) -> float:
    """Compute baseline fp16/bf16 PPL with full model."""
    print('  Loading baseline model for PPL...')
    max_memory = {0: '28GiB', 1: '28GiB', 'cpu': '80GiB'}
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        device_map='auto',
        max_memory=max_memory,
        dtype=dtype,
        attn_implementation='eager',
        low_cpu_mem_usage=True,
    )
    model.train(False)
    embed_dev = model.model.embed_tokens.weight.device

    nlls: list[float] = []
    for pi, prompt_ids in enumerate(eval_prompts):
        ids = prompt_ids.unsqueeze(0).long().to(embed_dev)
        logits = model(input_ids=ids, use_cache=False, return_dict=True).logits
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = ids[:, 1:].contiguous().to(shift_logits.device)
        nll = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='mean',
        ).item()
        nlls.append(nll)
        if (pi + 1) % 10 == 0:
            print(f'    baseline eval {pi + 1}/{len(eval_prompts)}  '
                  f'running_PPL={math.exp(sum(nlls) / len(nlls)):.4f}')

    del model
    free_memory()
    return math.exp(sum(nlls) / len(nlls))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description='Streaming layer-wise compression runner')
    ap.add_argument('--model', type=str, default='qwen3-8b',
                    choices=list(MODEL_REGISTRY.keys()))
    ap.add_argument('--bpw', type=int, default=5,
                    help='GSQ quantization bits per weight (default: 5)')
    ap.add_argument('--block_size', type=int, default=64,
                    help='Per-block quant group size (default: 64)')
    ap.add_argument('--rank', type=int, default=32,
                    help='V18-C correction rank (default: 32)')
    ap.add_argument('--train_steps', type=int, default=200,
                    help='Distillation steps per layer (default: 200)')
    ap.add_argument('--train_lr', type=float, default=1e-3,
                    help='Learning rate for V/U training (default: 1e-3)')
    ap.add_argument('--train_bs', type=int, default=8,
                    help='Training batch size (default: 8)')
    ap.add_argument('--n_calib', type=int, default=100,
                    help='Number of calibration prompts (default: 100)')
    ap.add_argument('--n_eval', type=int, default=50,
                    help='Number of eval prompts (default: 50)')
    ap.add_argument('--seq_len', type=int, default=128,
                    help='Sequence length (default: 128)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--skip_baseline', action='store_true',
                    help='Skip baseline PPL computation')
    ap.add_argument('--out_json', type=Path, default=None,
                    help='Output JSON path (default: artifacts/streaming_compression_*.json)')
    ap.add_argument('--track_g', type=str, default='none',
                    choices=['none', '4', '8', 'adaptive'],
                    help="Apply Track G nested-quant to V18-C V/U after compression. "
                         "Default 'none' keeps V/U at fp32. '4'/'8' = uniform int4/int8. "
                         "'adaptive' = per-tensor int4 with int8 fallback when cosine "
                         "< --track_g_cos_threshold (default 0.97).")
    ap.add_argument('--track_g_cos_threshold', type=float, default=0.97,
                    help='Adaptive Track G cosine threshold (default 0.97)')
    args = ap.parse_args()

    cfg = MODEL_REGISTRY[args.model]
    hf_id = cfg['hf_id']
    dtype = cfg['dtype']
    n_layers = cfg['n_layers']

    # Paths
    cache_dir = _HERE / f'streaming_compress_cache_{args.model}'
    output_dir = _HERE / f'streaming_compress_output_{args.model}'
    artifacts_dir = _HERE / 'artifacts'
    artifacts_dir.mkdir(exist_ok=True)

    if args.out_json is None:
        args.out_json = artifacts_dir / f'streaming_compression_{args.model}_smoke.json'

    print('=' * 78)
    print('STREAMING LAYER-WISE COMPRESSION RUNNER')
    print('=' * 78)
    print(f'  model={args.model}  hf_id={hf_id}  n_layers={n_layers}')
    print(f'  bpw={args.bpw}  block={args.block_size}  rank={args.rank}')
    print(f'  train_steps={args.train_steps}  lr={args.train_lr}  bs={args.train_bs}')
    print(f'  n_calib={args.n_calib}  n_eval={args.n_eval}  seq_len={args.seq_len}')
    print(f'  device={DEVICE}  seed={args.seed}')
    print(f'  cache_dir={cache_dir}')
    print(f'  output_dir={output_dir}')
    print(f'  out_json={args.out_json}')
    print(f'  Time: {datetime.datetime.now().isoformat()}')
    print('=' * 78)

    if not torch.cuda.is_available():
        sys.exit('ERROR: no CUDA devices visible')

    torch.manual_seed(args.seed)
    torch.cuda.set_device(DEVICE)

    # ---- Load tokenized FineWeb-edu ----
    print('\n[1/5] Loading FineWeb-edu tokens...')
    candidates = get_data_candidates_for_model(args.model)
    data_path = next((p for p in candidates if p.exists()), None)
    if data_path is None:
        sys.exit(f'ERROR: no FineWeb-edu token file for model={args.model!r} found at {candidates}\n'
                 f'  Run: python scripts/data/tokenize_fineweb_for_model.py --model {hf_id} '
                 f'--n_tokens 10000000 --output {candidates[0]}')

    all_tokens = torch.load(data_path, weights_only=True)
    total = all_tokens.numel()
    print(f'  Loaded {data_path.name} ({total / 1e6:.0f}M tokens)')

    # Split: first portion for calibration, tail for eval (no overlap)
    g = torch.Generator().manual_seed(args.seed)
    tail_size = min(50_000_000, total // 5)
    tail_start = max(args.seq_len + 1, total - tail_size)

    # Calibration prompts from body
    calib_starts = torch.randint(0, tail_start - args.seq_len - 1,
                                 (args.n_calib,), generator=g)
    calibration_ids: list[torch.Tensor] = [
        all_tokens[int(s):int(s) + args.seq_len].long()
        for s in calib_starts.tolist()
    ]

    # Eval prompts from tail (held out)
    eval_starts = torch.randint(tail_start, total - args.seq_len - 1,
                                (args.n_eval,), generator=g)
    eval_prompts: list[torch.Tensor] = [
        all_tokens[int(s):int(s) + args.seq_len].long()
        for s in eval_starts.tolist()
    ]
    print(f'  {args.n_calib} calibration prompts, {args.n_eval} eval prompts, '
          f'seq_len={args.seq_len}')

    # ---- Phase 1: Cache teacher hidden states ----
    print('\n[2/5] Caching teacher hidden states...')
    t_phase1 = time.time()
    cache_teacher_hidden_states(
        hf_id=hf_id,
        dtype=dtype,
        calibration_ids=calibration_ids,
        cache_dir=cache_dir,
        n_layers=n_layers,
    )
    phase1_time = time.time() - t_phase1
    print(f'  Phase 1 complete: {phase1_time:.1f}s')

    # ---- Phase 2: Layer-wise compression ----
    print('\n[3/5] Layer-wise streaming compression...')
    t_phase2 = time.time()
    torch.cuda.reset_peak_memory_stats(DEVICE)

    per_layer_metrics: list[dict] = []
    hardest_layer_idx = 0
    hardest_layer_loss = 0.0

    for i in range(n_layers):
        # Resume support: skip layers already saved on disk.
        # Critical for recovering from CUDA OOM mid-run on long compressions
        # (e.g. Hermes-405B Phase 2 layer 35 OOM 2026-05-08).
        existing_layer_path = output_dir / f'layer_{i:03d}.pt'
        if existing_layer_path.exists():
            print(f'\n  --- Layer {i}/{n_layers - 1} --- SKIP (already saved at {existing_layer_path.name})')
            try:
                cached = torch.load(existing_layer_path, map_location='cpu', weights_only=False)
                cached_metrics = cached.get('metrics', {'layer_idx': i, 'resumed': True})
                per_layer_metrics.append(cached_metrics)
            except Exception:
                per_layer_metrics.append({'layer_idx': i, 'resumed': True})
            continue
        print(f'\n  --- Layer {i}/{n_layers - 1} ---')
        metrics = compress_single_layer(
            hf_id=hf_id,
            dtype=dtype,
            layer_idx=i,
            n_layers=n_layers,
            hidden_cache_dir=cache_dir,
            output_dir=output_dir,
            bpw=args.bpw,
            block_size=args.block_size,
            rank=args.rank,
            train_steps=args.train_steps,
            train_lr=args.train_lr,
            train_bs=args.train_bs,
            device=DEVICE,
        )
        per_layer_metrics.append(metrics)

        if metrics['train_loss_final'] > hardest_layer_loss:
            hardest_layer_loss = metrics['train_loss_final']
            hardest_layer_idx = i

        print(f'    Layer {i} done: loss={metrics["train_loss_final"]:.6f}  '
              f'quant_l2={metrics["mean_quant_rel_l2"]:.4f}  '
              f'time={metrics["compress_time_s"]:.1f}s  '
              f'peak_vram={metrics["peak_vram_gb"]:.2f}GB')

    phase2_time = time.time() - t_phase2
    peak_vram_compress = max(m['peak_vram_gb'] for m in per_layer_metrics)
    print(f'\n  Phase 2 complete: {phase2_time:.1f}s total, '
          f'peak VRAM={peak_vram_compress:.2f}GB')
    print(f'  Hardest layer: {hardest_layer_idx} '
          f'(loss={hardest_layer_loss:.6f})')

    # ---- Phase 3: Baseline PPL ----
    if not args.skip_baseline:
        print('\n[4/5] Baseline PPL evaluation...')
        t_baseline = time.time()
        fp16_ppl = baseline_ppl(hf_id, dtype, eval_prompts, DEVICE)
        baseline_time = time.time() - t_baseline
        print(f'  Baseline PPL: {fp16_ppl:.4f} ({baseline_time:.1f}s)')
    else:
        fp16_ppl = float('nan')
        baseline_time = 0.0

    # ---- Phase 4: Compressed model eval ----
    print('\n[5/5] Streaming compressed model PPL evaluation...')
    t_eval = time.time()
    compressed_ppl, eval_peak_vram = streaming_eval_ppl(
        hf_id=hf_id,
        compressed_dir=output_dir,
        eval_prompts=eval_prompts,
        device=DEVICE,
    )
    eval_time = time.time() - t_eval
    print(f'  Compressed PPL: {compressed_ppl:.4f} ({eval_time:.1f}s)')
    print(f'  Eval peak VRAM: {eval_peak_vram:.2f}GB')

    # ---- Results ----
    ppl_ratio = compressed_ppl / fp16_ppl if not math.isnan(fp16_ppl) else float('nan')
    total_time = phase1_time + phase2_time + baseline_time + eval_time

    results = {
        'model': args.model,
        'hf_id': hf_id,
        'n_layers': n_layers,
        'bpw': args.bpw,
        'block_size': args.block_size,
        'rank': args.rank,
        'train_steps': args.train_steps,
        'train_lr': args.train_lr,
        'train_bs': args.train_bs,
        'n_calib': args.n_calib,
        'n_eval': args.n_eval,
        'seq_len': args.seq_len,
        'seed': args.seed,
        'baseline_fp16_ppl': fp16_ppl,
        'compressed_ppl': compressed_ppl,
        'ppl_ratio': ppl_ratio,
        'peak_vram_compress_gb': peak_vram_compress,
        'peak_vram_eval_gb': eval_peak_vram,
        'total_compress_time_s': phase2_time,
        'total_time_s': total_time,
        'hardest_layer_idx': hardest_layer_idx,
        'hardest_layer_loss': hardest_layer_loss,
        'per_layer_train_loss': [m['train_loss_final'] for m in per_layer_metrics],
        'per_layer_quant_rel_l2': [m['mean_quant_rel_l2'] for m in per_layer_metrics],
        'per_layer_compress_time_s': [m['compress_time_s'] for m in per_layer_metrics],
        'timestamp': datetime.datetime.now().isoformat(),
        'device': str(DEVICE),
        'cuda_device_name': torch.cuda.get_device_name(DEVICE),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(results, indent=2))

    # ---- Optional: apply Track G nested-quant post-processing ----
    if args.track_g != 'none':
        try:
            from track_g_nested_quant_postprocess import process_directory as _tg_process
        except ImportError:
            print(f'\n[track_g] WARNING: could not import track_g_nested_quant_postprocess; '
                  f'skipping Track G post-process. Run separately: '
                  f'python scripts/overlay/track_g_nested_quant_postprocess.py '
                  f'--in-dir {output_dir} --bits {args.track_g}')
        else:
            tg_bits: int | str = 'adaptive' if args.track_g == 'adaptive' else int(args.track_g)
            tg_label = 'ada' if tg_bits == 'adaptive' else str(tg_bits)
            tg_out = output_dir.parent / f'{output_dir.name}_g{tg_label}'
            print(f'\n[track_g] applying nested-quant bits={args.track_g} '
                  f'(threshold={args.track_g_cos_threshold}) -> {tg_out.name}')
            tg_summary = _tg_process(output_dir, tg_out, bits=tg_bits, verify=True,
                                     simulate_roundtrip=False,
                                     cos_threshold=args.track_g_cos_threshold)
            tg_summary_path = artifacts_dir / f'streaming_compression_{args.model}_track_g_g{tg_label}.json'
            tg_summary_path.write_text(json.dumps(tg_summary, indent=2, default=str))
            print(f'[track_g] V/U compression {tg_summary.get("vu_compression_x_total", 0):.2f}x  '
                  f'min_cos={tg_summary.get("min_cosine_across_all_layers")}  '
                  f'summary={tg_summary_path}')
            results['track_g'] = {
                'bits': args.track_g,
                'cos_threshold': args.track_g_cos_threshold,
                'out_dir': str(tg_out),
                'vu_compression_x': tg_summary.get('vu_compression_x_total'),
                'min_cosine': tg_summary.get('min_cosine_across_all_layers'),
                'summary_path': str(tg_summary_path),
            }
            args.out_json.write_text(json.dumps(results, indent=2))

    print('\n' + '=' * 78)
    print('RESULTS')
    print('=' * 78)
    print(f'  Model:              {args.model} ({hf_id})')
    print(f'  Layers:             {n_layers}')
    print(f'  Quantization:       GSQ {args.bpw}bpw B={args.block_size}')
    print(f'  Correction:         V18-C r={args.rank}')
    print(f'  Baseline PPL:       {fp16_ppl:.4f}')
    print(f'  Compressed PPL:     {compressed_ppl:.4f}')
    print(f'  PPL Ratio:          {ppl_ratio:.4f}x')
    print(f'  Peak VRAM (comp):   {peak_vram_compress:.2f} GB')
    print(f'  Peak VRAM (eval):   {eval_peak_vram:.2f} GB')
    print(f'  Total compress:     {phase2_time:.1f}s')
    print(f'  Total time:         {total_time:.1f}s')
    print(f'  Hardest layer:      {hardest_layer_idx} (loss={hardest_layer_loss:.6f})')
    print(f'  Output JSON:        {args.out_json}')
    print('=' * 78)

    # Success criterion
    if not math.isnan(ppl_ratio):
        if ppl_ratio <= 1.01:
            print('  STATUS: PASS (stretch goal: PPL_r <= 1.01x)')
        elif ppl_ratio <= 1.05:
            print('  STATUS: PASS (PPL_r <= 1.05x)')
        else:
            print(f'  STATUS: FAIL (PPL_r = {ppl_ratio:.4f}x > 1.05x)')


if __name__ == '__main__':
    main()
