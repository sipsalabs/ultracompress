"""Online distillation streaming compression runner for Track A (GSQ + V18-C).

Cure for the offline distillation distribution-shift regression: instead of
training V/U against cached teacher hiddens (which drift from the actual
compressed forward as errors compound across layers), this runner uses the
COMPRESSED model's own hidden states as input to each layer during training.

Key difference from streaming_compression_runner.py:
  - Only cache teacher hidden_0 (embed output) and teacher per-layer OUTPUT
    hiddens (targets).
  - For layer i, the INPUT hidden is produced by running compressed layers
    0..i-1 in forward (online forward), not the teacher's cached hidden_{i-1}.
  - V/U distillation target remains the teacher's hidden_i (cached).
  - This means the training distribution matches deployment distribution:
    compressed errors compound and the correction learns to compensate.

Expected improvement: 0.01-0.02 PPL_r reduction vs offline at same step count.

Hardware: cuda:1 ONLY (cuda:0 reserved for trellis Viterbi).

Usage:
  python streaming_compression_online_runner.py --model qwen3-8b
  python streaming_compression_online_runner.py --model qwen3-8b --bpw 5 --rank 32
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
}

TARGET_SUBS = ('q_proj', 'k_proj', 'v_proj', 'o_proj',
               'gate_proj', 'up_proj', 'down_proj')

# cuda:1 ONLY — cuda:0 is reserved for trellis Viterbi
DEVICE = torch.device('cuda:1')

DATA_CANDIDATES = [
    _ROOT / 'fineweb_edu_500M_tokens.pt',
    _ROOT / 'fineweb_edu_100M_tokens.pt',
]

# ---------------------------------------------------------------------------
# Quantizer: GSQ (k-means learned grid)
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


def gsq_quantize_weight(W: torch.Tensor, bpw: int,
                        block: int = 64, gsq_steps: int = 50) -> torch.Tensor:
    """GSQ: learned scalar grid via k-means on pooled normalized weights."""
    K = 1 << bpw
    out_dim, in_dim = W.shape

    if block <= 0 or in_dim % block != 0:
        block = 128
        if in_dim % block != 0:
            half = 2 ** bpw // 2
            rm = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
            return ((W / rm * half).round().clamp(-half, half - 1) / half) * rm

    Wb = W.reshape(out_dim, in_dim // block, block).float()
    absmax = Wb.abs().amax(dim=-1, keepdim=True).clamp(min=1e-9)
    Wn = Wb / absmax

    if K in _NF_LEVELS:
        grid = _NF_LEVELS[K][:K].clone().float().to(W.device)
    else:
        grid = torch.linspace(-1.0, 1.0, K, device=W.device)

    if grid.abs().min() > 0.05:
        closest_to_zero = grid.abs().argmin()
        grid[closest_to_zero] = 0.0

    Wn_flat = Wn.reshape(-1)
    N = Wn_flat.shape[0]

    sample_size = min(N, 1 << 18)
    if N > sample_size:
        idx_sample = torch.randint(0, N, (sample_size,), device=W.device)
        w_sample = Wn_flat[idx_sample]
    else:
        w_sample = Wn_flat

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

    with torch.no_grad():
        grid_final = grid.detach().sort().values
        n_blocks = Wn.shape[1]
        Wn_q = torch.empty_like(Wn)
        chunk_rows = max(1, (256 * 1024 * 1024) // (n_blocks * block * K * 4))
        for r_start in range(0, out_dim, chunk_rows):
            r_end = min(r_start + chunk_rows, out_dim)
            chunk = Wn[r_start:r_end]
            dists_chunk = (chunk.unsqueeze(-1) - grid_final.view(1, 1, 1, -1)).abs()
            idx_chunk = dists_chunk.argmin(dim=-1)
            Wn_q[r_start:r_end] = grid_final[idx_chunk]
        Wq = Wn_q * absmax

    return Wq.reshape(out_dim, in_dim).to(W.dtype)


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
            self.V.weight.data.copy_(
                (Vt_svd[:rank] * S_svd[:rank].unsqueeze(1).sqrt()).to(self.V.weight.dtype)
            )
            self.U.weight.data.copy_(
                (U_svd[:, :rank] * S_svd[:rank].unsqueeze(0).sqrt()).to(self.U.weight.dtype)
            )
            self.alpha.data.fill_(1.0)
        except Exception:
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
    """Return (DecoderLayerClass, RotaryEmbeddingClass) for the given model."""
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(hf_id, trust_remote_code=True)
    model_type = config.model_type

    if model_type == 'qwen3':
        from transformers.models.qwen3.modeling_qwen3 import (
            Qwen3DecoderLayer,
            Qwen3RotaryEmbedding,
        )
        return Qwen3DecoderLayer, Qwen3RotaryEmbedding
    elif model_type in ('qwen2', 'qwen2_5'):
        from transformers.models.qwen2.modeling_qwen2 import (
            Qwen2DecoderLayer,
            Qwen2RotaryEmbedding,
        )
        return Qwen2DecoderLayer, Qwen2RotaryEmbedding
    else:
        from transformers.models.qwen3.modeling_qwen3 import (
            Qwen3DecoderLayer,
            Qwen3RotaryEmbedding,
        )
        return Qwen3DecoderLayer, Qwen3RotaryEmbedding


# ---------------------------------------------------------------------------
# Lazy layer loader
# ---------------------------------------------------------------------------
_LAYER_LOAD_MODEL_DIR_CACHE: dict[str, Path] = {}


def load_layer_state_dict(
    hf_id: str, layer_idx: int, device: str | torch.device, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    """Load only `model.layers.{layer_idx}.*` tensors from HF safetensors shards."""
    from huggingface_hub import snapshot_download

    if hf_id not in _LAYER_LOAD_MODEL_DIR_CACHE:
        model_dir = Path(snapshot_download(
            hf_id, allow_patterns=["*.json", "*.safetensors"]
        ))
        _LAYER_LOAD_MODEL_DIR_CACHE[hf_id] = model_dir
    else:
        model_dir = _LAYER_LOAD_MODEL_DIR_CACHE[hf_id]

    index_file = model_dir / "model.safetensors.index.json"
    if not index_file.exists():
        shard_files = list(model_dir.glob("*.safetensors"))
        if not shard_files:
            raise FileNotFoundError(f"No safetensors files found in {model_dir}")
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
            f"No keys found for layer {layer_idx} (prefix={layer_prefix!r})"
        )

    shards: dict[str, list[str]] = {}
    for k in layer_keys:
        shards.setdefault(weight_map[k], []).append(k)

    sd = {}
    for shard_file, keys in shards.items():
        shard_path = str(model_dir / shard_file)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for k in keys:
                stripped = k[len(layer_prefix):]
                sd[stripped] = f.get_tensor(k).to(device=device, dtype=dtype)

    return sd


# ---------------------------------------------------------------------------
# Phase 1: Cache teacher hidden states (ALL layers — needed as targets)
# ---------------------------------------------------------------------------
@torch.no_grad()
def cache_teacher_hidden_states(
    hf_id: str,
    dtype: torch.dtype,
    calibration_ids: list[torch.Tensor],
    cache_dir: Path,
    n_layers: int,
    device: torch.device = DEVICE,
) -> None:
    """Load teacher in NF4 4-bit on cuda:1, cache per-layer hiddens.

    We still cache ALL teacher hidden states (as targets for distillation).
    The key difference from offline runner: during TRAINING, we use the
    compressed model's own forward output as INPUT, not teacher hidden_{i-1}.
    But we still need teacher hidden_i as the DISTILLATION TARGET.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

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

    print(f'  Loading teacher (NF4 4-bit) on cuda:1 for hidden state caching...')
    t0 = time.time()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
    )
    # Force teacher onto cuda:1 only
    max_memory = {0: '0GiB', 1: '28GiB', 'cpu': '80GiB'}

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

    embed_device = teacher.model.embed_tokens.weight.device
    n_prompts = len(calibration_ids)

    print(f'  Caching hidden states for {n_prompts} prompts, {n_layers} layers...')

    # Embedding output (hidden_layer_000)
    all_embeds = []
    for prompt_ids in calibration_ids:
        ids = prompt_ids.unsqueeze(0).to(embed_device)
        emb = teacher.model.embed_tokens(ids).to(torch.bfloat16)
        all_embeds.append(emb.cpu())

    hidden_0 = torch.cat(all_embeds, dim=0)
    torch.save(hidden_0, cache_dir / 'hidden_layer_000.pt')
    print(f'    Saved hidden_layer_000.pt ({hidden_0.shape}, {hidden_0.nbytes / 1e6:.1f}MB)')
    del all_embeds

    # Full forward with hooks to capture all layer outputs
    hidden_states_per_layer: dict[int, list[torch.Tensor]] = {
        i: [] for i in range(n_layers)
    }

    hooks = []

    def make_hook(layer_idx: int):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            hidden_states_per_layer[layer_idx].append(h.detach().cpu().to(torch.bfloat16))
        return hook_fn

    for i in range(n_layers):
        h = teacher.model.layers[i].register_forward_hook(make_hook(i))
        hooks.append(h)

    for pi, prompt_ids in enumerate(calibration_ids):
        ids = prompt_ids.unsqueeze(0).long().to(embed_device)
        _ = teacher(input_ids=ids, use_cache=False, return_dict=True)
        if (pi + 1) % 20 == 0:
            print(f'    Forward {pi + 1}/{n_prompts}  {vram_report()}')

    for h in hooks:
        h.remove()

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

    manifest = {
        'hf_id': hf_id,
        'n_layers': n_layers,
        'n_prompts': n_prompts,
        'seq_len': calibration_ids[0].shape[0],
        'dtype': 'bfloat16',
        'built_at': datetime.datetime.now().isoformat(),
        'mode': 'online_distillation',
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    del teacher
    free_memory()
    elapsed = time.time() - t0
    print(f'  Teacher hidden cache complete in {elapsed:.1f}s')


# ---------------------------------------------------------------------------
# Phase 2: Online layer-wise compression
#
# The key innovation: layer i's INPUT is generated by running compressed
# layers 0..i-1 in forward (online), not from teacher cache.
# The TARGET remains teacher's hidden_i (cached, fixed).
# ---------------------------------------------------------------------------
def compress_single_layer_online(
    hf_id: str,
    dtype: torch.dtype,
    layer_idx: int,
    n_layers: int,
    hidden_cache_dir: Path,
    output_dir: Path,
    compressed_input_hidden: torch.Tensor,
    bpw: int = 5,
    block_size: int = 64,
    rank: int = 32,
    train_steps: int = 200,
    train_lr: float = 1e-3,
    train_bs: int = 8,
    device: torch.device = DEVICE,
) -> tuple[dict[str, float], torch.Tensor]:
    """Compress a single transformer layer with online distillation.

    Args:
        compressed_input_hidden: [n_prompts, seq_len, hidden_dim] on CPU --
            the output of running compressed layers 0..i-1. For layer 0, this
            is the teacher's embed output.

    Returns:
        (per_layer_metrics, compressed_output_hidden) where
        compressed_output_hidden is the output of this compressed layer
        applied to compressed_input_hidden (for feeding into layer i+1).
    """
    t0 = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    # INPUT = compressed_input_hidden (online -- matches deployment distribution)
    input_hidden = compressed_input_hidden  # [n_prompts, seq_len, hidden_dim] on CPU

    # TARGET = teacher's hidden_i+1 (cached, fixed)
    target_hidden = torch.load(
        hidden_cache_dir / f'hidden_layer_{layer_idx + 1:03d}.pt',
        map_location='cpu', weights_only=True,
    )  # [n_prompts, seq_len, hidden_dim]

    n_prompts, seq_len, hidden_dim = input_hidden.shape

    # Load this layer's weights
    config = AutoConfig.from_pretrained(hf_id, trust_remote_code=True)
    config._attn_implementation = 'eager'

    print(f'    Loading layer {layer_idx} weights (lazy safetensors)...')

    DecoderLayerClass, RotaryEmbClass = get_model_classes(hf_id)
    layer_sd = load_layer_state_dict(hf_id, layer_idx, device, dtype)

    with torch.device('meta'):
        layer = DecoderLayerClass(config, layer_idx=layer_idx)
    layer.load_state_dict(layer_sd, strict=True, assign=True)
    layer = layer.to(device=device, dtype=dtype)
    del layer_sd
    free_memory()

    # Store originals for SVD warm-start
    original_weights: dict[str, torch.Tensor] = {}
    for name, mod in layer.named_modules():
        if isinstance(mod, nn.Linear) and any(s in name for s in TARGET_SUBS):
            original_weights[name] = mod.weight.data.clone()

    # Apply GSQ quantization
    n_quantized = 0
    quant_errors: dict[str, float] = {}
    for name, mod in layer.named_modules():
        if isinstance(mod, nn.Linear) and any(s in name for s in TARGET_SUBS):
            with torch.no_grad():
                W = mod.weight.data.float()
                Wq = gsq_quantize_weight(W, bpw, block_size)
                rel_l2 = (W - Wq).norm() / W.norm()
                quant_errors[name] = rel_l2.item()
                mod.weight.data.copy_(Wq.to(dtype))
                del W, Wq
            n_quantized += 1
    free_memory()

    # Wrap with V18-C correction
    correction_modules: dict[str, CorrectionMatrixC] = {}

    def wrap_linears_with_correction(module: nn.Module, prefix: str = '') -> None:
        for name, child in list(module.named_children()):
            full_name = f'{prefix}.{name}' if prefix else name
            if isinstance(child, nn.Linear) and any(s in name for s in TARGET_SUBS):
                cm = CorrectionMatrixC(
                    child.weight.data, child.bias.data if child.bias is not None else None,
                    rank=rank,
                )
                if full_name in original_weights:
                    cm.init_from_svd(original_weights[full_name].to(device))
                setattr(module, name, cm)
                correction_modules[full_name] = cm
            else:
                wrap_linears_with_correction(child, full_name)

    wrap_linears_with_correction(layer)
    del original_weights
    free_memory()

    # Freeze base weights, train V/U/alpha only
    for p in layer.parameters():
        p.requires_grad = False
    for cm in correction_modules.values():
        for p in cm.V.parameters():
            p.requires_grad = True
        for p in cm.U.parameters():
            p.requires_grad = True
        cm.alpha.requires_grad = True

    trainable_params = [p for p in layer.parameters() if p.requires_grad]
    n_train_params = sum(p.numel() for p in trainable_params)

    opt = torch.optim.Adam(trainable_params, lr=train_lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=train_steps)

    layer.train()
    losses: list[float] = []

    from transformers.masking_utils import create_causal_mask
    rotary_emb = RotaryEmbClass(config=config, device=device)

    for step in range(train_steps):
        batch_idx = torch.randint(0, n_prompts, (train_bs,))
        # ONLINE: input from compressed forward, not teacher cache
        x_batch = input_hidden[batch_idx].to(device=device, dtype=dtype)
        y_target = target_hidden[batch_idx].to(device=device, dtype=dtype)

        bs_actual, sl, _ = x_batch.shape

        cache_position = torch.arange(sl, device=device)
        position_ids = cache_position.unsqueeze(0).expand(bs_actual, -1)

        causal_mask = create_causal_mask(
            config=config,
            input_embeds=x_batch,
            attention_mask=None,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids,
        )

        position_embeddings = rotary_emb(x_batch, position_ids)

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

    # --- Compute compressed output hidden for use as next layer's input ---
    # Run the compressed+corrected layer on ALL calibration inputs
    print(f'    Computing compressed output hidden for next layer...')
    compressed_outputs = []
    with torch.no_grad():
        # Process in chunks to avoid OOM
        chunk_size = 16
        for ci in range(0, n_prompts, chunk_size):
            ce = min(ci + chunk_size, n_prompts)
            x_chunk = input_hidden[ci:ce].to(device=device, dtype=dtype)
            bs_c, sl_c, _ = x_chunk.shape

            cache_position = torch.arange(sl_c, device=device)
            position_ids = cache_position.unsqueeze(0).expand(bs_c, -1)

            causal_mask = create_causal_mask(
                config=config,
                input_embeds=x_chunk,
                attention_mask=None,
                cache_position=cache_position,
                past_key_values=None,
                position_ids=position_ids,
            )
            position_embeddings = rotary_emb(x_chunk, position_ids)

            layer_out = layer(
                x_chunk,
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

            compressed_outputs.append(h_out.cpu().to(torch.bfloat16))
            del x_chunk, h_out
            free_memory()

    compressed_output_hidden = torch.cat(compressed_outputs, dim=0)
    del compressed_outputs

    # Save compressed layer
    save_dict: dict[str, Any] = {
        'layer_idx': layer_idx,
        'bpw': bpw,
        'block_size': block_size,
        'rank': rank,
        'mode': 'online_distillation',
        'state_dict': {},
        'corrections': {},
    }

    save_dict['state_dict'] = {k: v.cpu() for k, v in layer.state_dict().items()}

    for name, cm in correction_modules.items():
        save_dict['corrections'][name] = {
            'V_weight': cm.V.weight.data.cpu(),
            'U_weight': cm.U.weight.data.cpu(),
            'alpha': cm.alpha.data.cpu().item(),
        }

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

    return metrics, compressed_output_hidden


# ---------------------------------------------------------------------------
# Phase 3: Streaming eval (Track C pattern) — same as offline runner
# ---------------------------------------------------------------------------
@torch.no_grad()
def streaming_eval_ppl(
    hf_id: str,
    compressed_dir: Path,
    eval_prompts: list[torch.Tensor],
    device: torch.device = DEVICE,
) -> tuple[float, float]:
    """Evaluate PPL using streaming forward through compressed layers."""
    config = AutoConfig.from_pretrained(hf_id, trust_remote_code=True)
    config._attn_implementation = 'eager'
    n_layers = config.num_hidden_layers
    dtype = torch.bfloat16

    DecoderLayerClass, RotaryEmbClass = get_model_classes(hf_id)
    from transformers.masking_utils import create_causal_mask

    print('  Loading scaffold (embed + norm + lm_head) on cuda:1...')
    scaffold_device_map = {
        'model.embed_tokens': str(device),
        'model.norm': str(device),
        'lm_head': str(device),
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

    rotary_emb = RotaryEmbClass(config=config, device=device)

    del scaffold
    free_memory()

    torch.cuda.reset_peak_memory_stats(device)

    nlls: list[float] = []

    for pi, prompt_ids in enumerate(eval_prompts):
        ids = prompt_ids.unsqueeze(0).long().to(device)
        bsz, seqlen = ids.shape

        hidden = embed_tokens(ids).to(dtype)

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
        position_embeddings = rotary_emb(hidden, position_ids)

        for i in range(n_layers):
            layer_path = compressed_dir / f'layer_{i:03d}.pt'
            layer_data = torch.load(layer_path, map_location=device, weights_only=False)

            corrections_data = layer_data.get('corrections', {})
            layer_sd = {k: v.to(device) for k, v in layer_data['state_dict'].items()}

            with torch.device('meta'):
                layer = DecoderLayerClass(config, layer_idx=i)

            for name in corrections_data:
                cd = corrections_data[name]
                r = cd['V_weight'].shape[0]
                parts = name.split('.')
                parent = layer
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                attr = parts[-1]
                w_base_key = f'{name}.W_base'
                w_base_shape = layer_sd[w_base_key].shape
                bias_key = f'{name}.bias_buf'
                has_bias = bias_key in layer_sd
                with torch.device('meta'):
                    dummy_w = torch.empty(w_base_shape, dtype=dtype)
                    dummy_bias = torch.empty(w_base_shape[0], dtype=dtype) if has_bias else None
                cm = CorrectionMatrixC(dummy_w, dummy_bias, rank=r)
                setattr(parent, attr, cm)

            layer.load_state_dict(layer_sd, strict=True, assign=True)
            layer = layer.to(device=device, dtype=dtype)
            del layer_sd

            layer.train(False)

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

        hidden = final_norm(hidden)
        logits = lm_head(hidden)

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
    """Compute baseline bf16 PPL with full model on cuda:1."""
    print('  Loading baseline model for PPL on cuda:1...')
    # Force to cuda:1 only
    max_memory = {0: '0GiB', 1: '28GiB', 'cpu': '80GiB'}
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
    ap = argparse.ArgumentParser(
        description='Online distillation streaming compression runner'
    )
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
    ap.add_argument('--adaptive_steps', action='store_true',
                    help='Scale train steps linearly from base at layer 0 to '
                         '2x base at final layer (compensates compounding error)')
    ap.add_argument('--skip_baseline', action='store_true',
                    help='Skip baseline PPL computation')
    ap.add_argument('--out_json', type=Path, default=None,
                    help='Output JSON path')
    args = ap.parse_args()

    cfg = MODEL_REGISTRY[args.model]
    hf_id = cfg['hf_id']
    dtype = cfg['dtype']
    n_layers = cfg['n_layers']

    # Paths — use separate cache/output dirs to avoid clobbering offline runner
    cache_dir = _HERE / f'streaming_compress_cache_{args.model}'
    output_dir = _HERE / f'streaming_compress_online_output_{args.model}'
    artifacts_dir = _HERE / 'artifacts'
    artifacts_dir.mkdir(exist_ok=True)

    if args.out_json is None:
        args.out_json = artifacts_dir / f'streaming_compression_online_{args.model}.json'

    print('=' * 78)
    print('ONLINE DISTILLATION STREAMING COMPRESSION RUNNER')
    print('=' * 78)
    print(f'  model={args.model}  hf_id={hf_id}  n_layers={n_layers}')
    print(f'  bpw={args.bpw}  block={args.block_size}  rank={args.rank}')
    print(f'  train_steps={args.train_steps}  lr={args.train_lr}  bs={args.train_bs}')
    print(f'  n_calib={args.n_calib}  n_eval={args.n_eval}  seq_len={args.seq_len}')
    print(f'  device={DEVICE}  seed={args.seed}')
    print(f'  adaptive_steps={args.adaptive_steps}')
    print(f'  cache_dir={cache_dir}')
    print(f'  output_dir={output_dir}')
    print(f'  out_json={args.out_json}')
    print(f'  Mode: ONLINE (compressed forward for layer inputs)')
    print(f'  Time: {datetime.datetime.now().isoformat()}')
    print('=' * 78)

    if not torch.cuda.is_available():
        sys.exit('ERROR: no CUDA devices visible')
    if torch.cuda.device_count() < 2:
        print('WARNING: only 1 GPU visible, proceeding on cuda:1 may fail')

    torch.manual_seed(args.seed)
    torch.cuda.set_device(DEVICE)

    # ---- Load tokenized FineWeb-edu ----
    print('\n[1/5] Loading FineWeb-edu tokens...')
    data_path = next((p for p in DATA_CANDIDATES if p.exists()), None)
    if data_path is None:
        sys.exit(f'ERROR: no FineWeb-edu token file found at {DATA_CANDIDATES}')

    all_tokens = torch.load(data_path, weights_only=True)
    total = all_tokens.numel()
    print(f'  Loaded {data_path.name} ({total / 1e6:.0f}M tokens)')

    g = torch.Generator().manual_seed(args.seed)
    tail_start = max(0, total - 50_000_000)

    calib_starts = torch.randint(0, tail_start - args.seq_len - 1,
                                 (args.n_calib,), generator=g)
    calibration_ids: list[torch.Tensor] = [
        all_tokens[int(s):int(s) + args.seq_len].long()
        for s in calib_starts.tolist()
    ]

    eval_starts = torch.randint(tail_start, total - args.seq_len - 1,
                                (args.n_eval,), generator=g)
    eval_prompts: list[torch.Tensor] = [
        all_tokens[int(s):int(s) + args.seq_len].long()
        for s in eval_starts.tolist()
    ]
    print(f'  {args.n_calib} calibration prompts, {args.n_eval} eval prompts, '
          f'seq_len={args.seq_len}')

    # ---- Phase 1: Cache teacher hidden states (reuse if available) ----
    print('\n[2/5] Caching teacher hidden states...')
    t_phase1 = time.time()
    cache_teacher_hidden_states(
        hf_id=hf_id,
        dtype=dtype,
        calibration_ids=calibration_ids,
        cache_dir=cache_dir,
        n_layers=n_layers,
        device=DEVICE,
    )
    phase1_time = time.time() - t_phase1
    print(f'  Phase 1 complete: {phase1_time:.1f}s')

    # ---- Phase 2: Online layer-wise compression ----
    print('\n[3/5] Online layer-wise streaming compression...')
    t_phase2 = time.time()
    torch.cuda.reset_peak_memory_stats(DEVICE)

    per_layer_metrics: list[dict] = []
    hardest_layer_idx = 0
    hardest_layer_loss = 0.0

    # Start with teacher embed output as the seed hidden
    compressed_hidden = torch.load(
        cache_dir / 'hidden_layer_000.pt',
        map_location='cpu', weights_only=True,
    )  # [n_prompts, seq_len, hidden_dim]
    print(f'  Seed hidden (teacher embed): {compressed_hidden.shape}, '
          f'{compressed_hidden.nbytes / 1e6:.1f}MB')

    for i in range(n_layers):
        # Adaptive step scaling: later layers get more training budget
        # because compounding compression error makes their input-target
        # gap much larger. Linear ramp from base_steps to 2*base_steps.
        if args.adaptive_steps and n_layers > 1:
            layer_steps = int(args.train_steps * (1.0 + i / (n_layers - 1)))
        else:
            layer_steps = args.train_steps

        print(f'\n  --- Layer {i}/{n_layers - 1} (ONLINE, steps={layer_steps}) ---')
        metrics, compressed_hidden = compress_single_layer_online(
            hf_id=hf_id,
            dtype=dtype,
            layer_idx=i,
            n_layers=n_layers,
            hidden_cache_dir=cache_dir,
            output_dir=output_dir,
            compressed_input_hidden=compressed_hidden,
            bpw=args.bpw,
            block_size=args.block_size,
            rank=args.rank,
            train_steps=layer_steps,
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

    del compressed_hidden
    free_memory()

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
        'mode': 'online_distillation',
        'adaptive_steps': args.adaptive_steps,
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
        'cure_description': (
            'Online distillation: layer i input is produced by compressed '
            'layers 0..i-1, not teacher cache. Eliminates distribution shift '
            'between training and deployment.'
        ),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(results, indent=2))

    print('\n' + '=' * 78)
    print('RESULTS (ONLINE DISTILLATION)')
    print('=' * 78)
    print(f'  Model:              {args.model} ({hf_id})')
    print(f'  Layers:             {n_layers}')
    print(f'  Quantization:       GSQ {args.bpw}bpw B={args.block_size}')
    print(f'  Correction:         V18-C r={args.rank}')
    print(f'  Mode:               ONLINE distillation')
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
        if ppl_ratio <= 1.02:
            print('  STATUS: PASS (stretch goal: PPL_r <= 1.02x)')
        elif ppl_ratio <= 1.025:
            print('  STATUS: PASS (PPL_r <= 1.025x)')
        elif ppl_ratio <= 1.05:
            print('  STATUS: MARGINAL (PPL_r <= 1.05x but above 1.025 target)')
        else:
            print(f'  STATUS: FAIL (PPL_r = {ppl_ratio:.4f}x > 1.05x)')


if __name__ == '__main__':
    main()
