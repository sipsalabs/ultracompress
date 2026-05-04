"""Eval-only script for streaming-compressed layers saved to disk.

Recovers PPL results when the full streaming_compression_runner crashed
during its eval phase (e.g., device_map="auto" bug under CUDA_VISIBLE_DEVICES).

Loads compressed layer .pt files from a directory, runs streaming forward
through them on a single GPU, computes compressed PPL, and optionally
computes a fresh baseline PPL (or accepts a pre-computed value).

Usage:
  python eval_compressed_only.py --model qwen3-8b --device cuda:1
  python eval_compressed_only.py --model qwen3-32b --device cuda:1 --baseline_ppl 13.766
  python eval_compressed_only.py --model qwen3-8b --compressed_dir /path/to/layers --device cuda:0
"""
import argparse
import datetime
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

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
# Import shared components from the runner (read-only, no modifications)
# ---------------------------------------------------------------------------
from streaming_compression_runner import (
    MODEL_REGISTRY,
    CorrectionMatrixC,
    get_model_classes,
    free_memory,
    vram_gb,
    vram_report,
)

DATA_CANDIDATES = [
    _ROOT / 'fineweb_edu_500M_tokens.pt',
    _ROOT / 'fineweb_edu_100M_tokens.pt',
]


# ---------------------------------------------------------------------------
# Streaming compressed eval (single-GPU, no device_map="auto")
# ---------------------------------------------------------------------------
@torch.no_grad()
def streaming_eval_ppl(
    hf_id: str,
    compressed_dir: Path,
    eval_prompts: list[torch.Tensor],
    n_layers: int,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate PPL by streaming compressed layers one at a time.

    Returns (compressed_ppl, peak_vram_gb).
    """
    config = AutoConfig.from_pretrained(hf_id, trust_remote_code=True)
    config._attn_implementation = 'eager'
    dtype = torch.bfloat16

    DecoderLayerClass, RotaryEmbClass = get_model_classes(hf_id)
    from transformers.masking_utils import create_causal_mask

    # Load scaffold (embed, norm, lm_head) to the specified single device.
    # We put transformer layers on 'meta' so they consume no memory.
    print(f'  Loading scaffold (embed + norm + lm_head) to {device}...')
    scaffold_device_map: dict[str, str] = {
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
        position_embeddings = rotary_emb(hidden, position_ids)

        # Stream each compressed layer
        for i in range(n_layers):
            layer_path = compressed_dir / f'layer_{i:03d}.pt'
            layer_data = torch.load(layer_path, map_location=device, weights_only=False)

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

                # Get shapes from state_dict
                w_base_key = f'{name}.W_base'
                w_base_shape = layer_sd[w_base_key].shape
                bias_key = f'{name}.bias_buf'
                has_bias = bias_key in layer_sd

                with torch.device('meta'):
                    dummy_w = torch.empty(w_base_shape, dtype=dtype)
                    dummy_bias = (
                        torch.empty(w_base_shape[0], dtype=dtype) if has_bias else None
                    )
                cm = CorrectionMatrixC(dummy_w, dummy_bias, rank=rank)
                setattr(parent, attr, cm)

            # Load full state_dict
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


# ---------------------------------------------------------------------------
# Baseline PPL on a single GPU (no device_map="auto")
# ---------------------------------------------------------------------------
@torch.no_grad()
def baseline_ppl_single_gpu(
    hf_id: str,
    dtype: torch.dtype,
    eval_prompts: list[torch.Tensor],
    device: torch.device,
    use_nf4: bool = False,
) -> float:
    """Compute baseline PPL on a single GPU.

    For models that don't fit in fp16/bf16 on one GPU (e.g., 32B = ~65GB),
    set use_nf4=True to load in NF4 4-bit (~10GB). Small PPL drift is
    acceptable since baseline is just a reference denominator.
    """
    print(f'  Loading baseline model to {device}' +
          (' (NF4 4-bit)' if use_nf4 else f' ({dtype})') + '...')

    load_kwargs: dict[str, Any] = {
        'torch_dtype': dtype,
        'attn_implementation': 'eager',
        'low_cpu_mem_usage': True,
    }

    if use_nf4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs['quantization_config'] = bnb_config
        # For NF4, device_map to the single target device only
        load_kwargs['device_map'] = {
            '': device,
        }
    else:
        load_kwargs['device_map'] = {
            '': device,
        }

    model = AutoModelForCausalLM.from_pretrained(hf_id, **load_kwargs)
    model.train(False)

    embed_dev = model.model.embed_tokens.weight.device
    print(f'  Baseline loaded on {embed_dev}  {vram_report()}')

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
            running = math.exp(sum(nlls) / len(nlls))
            print(f'    baseline eval {pi + 1}/{len(eval_prompts)}  '
                  f'running_PPL={running:.4f}')

    del model
    free_memory()
    return math.exp(sum(nlls) / len(nlls))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description='Eval-only for streaming-compressed layers on disk',
    )
    ap.add_argument('--model', type=str, required=True,
                    choices=list(MODEL_REGISTRY.keys()),
                    help='Model name (must match what was compressed)')
    ap.add_argument('--compressed_dir', type=Path, default=None,
                    help='Directory with layer_XXX.pt files. '
                         'Default: streaming_compress_output_{model}/')
    ap.add_argument('--n_eval', type=int, default=50,
                    help='Number of eval prompts (default: 50)')
    ap.add_argument('--seq_len', type=int, default=128,
                    help='Sequence length (default: 128)')
    ap.add_argument('--device', type=str, default='cuda:0',
                    help='GPU device for eval (default: cuda:0)')
    ap.add_argument('--seed', type=int, default=42,
                    help='Random seed (must match training run, default: 42)')
    ap.add_argument('--baseline_ppl', type=float, default=None,
                    help='Pre-computed baseline PPL (skips baseline eval)')
    ap.add_argument('--skip_baseline', action='store_true',
                    help='Skip baseline eval entirely (report ratio as N/A)')
    ap.add_argument('--use_nf4_baseline', action='store_true',
                    help='Use NF4 4-bit for baseline (for models > 1 GPU)')
    ap.add_argument('--out_json', type=Path, default=None,
                    help='Output JSON path')
    args = ap.parse_args()

    cfg = MODEL_REGISTRY[args.model]
    hf_id = cfg['hf_id']
    dtype = cfg['dtype']
    n_layers = cfg['n_layers']

    device = torch.device(args.device)

    # Resolve compressed dir
    if args.compressed_dir is None:
        args.compressed_dir = _HERE / f'streaming_compress_output_{args.model}'

    if args.out_json is None:
        artifacts_dir = _HERE / 'artifacts'
        artifacts_dir.mkdir(exist_ok=True)
        args.out_json = artifacts_dir / f'streaming_compression_{args.model}_eval_only.json'

    print('=' * 78)
    print('EVAL-ONLY: STREAMING COMPRESSED LAYERS')
    print('=' * 78)
    print(f'  model={args.model}  hf_id={hf_id}  n_layers={n_layers}')
    print(f'  compressed_dir={args.compressed_dir}')
    print(f'  n_eval={args.n_eval}  seq_len={args.seq_len}')
    print(f'  device={device}  seed={args.seed}')
    print(f'  baseline_ppl={args.baseline_ppl}  skip_baseline={args.skip_baseline}')
    print(f'  use_nf4_baseline={args.use_nf4_baseline}')
    print(f'  out_json={args.out_json}')
    print(f'  Time: {datetime.datetime.now().isoformat()}')
    print('=' * 78)

    if not torch.cuda.is_available():
        sys.exit('ERROR: no CUDA devices visible')

    # Verify compressed layers exist
    missing = []
    for i in range(n_layers):
        lp = args.compressed_dir / f'layer_{i:03d}.pt'
        if not lp.exists():
            missing.append(i)
    if missing:
        if len(missing) <= 5:
            sys.exit(f'ERROR: missing compressed layers: {missing}')
        else:
            sys.exit(f'ERROR: missing {len(missing)} compressed layers '
                     f'(first 5: {missing[:5]})')

    print(f'  All {n_layers} compressed layer files present.')

    torch.manual_seed(args.seed)

    # ---- Load eval data (same split logic as runner) ----
    print('\n[1/3] Loading FineWeb-edu eval tokens...')
    data_path = next((p for p in DATA_CANDIDATES if p.exists()), None)
    if data_path is None:
        sys.exit(f'ERROR: no FineWeb-edu token file found at {DATA_CANDIDATES}')

    all_tokens = torch.load(data_path, weights_only=True)
    total = all_tokens.numel()
    print(f'  Loaded {data_path.name} ({total / 1e6:.0f}M tokens)')

    # Reproduce the exact same eval split as the runner:
    # The runner uses seed=42 generator, draws n_calib from body, then n_eval from tail.
    # We must draw the same calibration prompts first (to advance the generator state),
    # even though we discard them, then draw eval prompts.
    n_calib = 100  # default in runner
    g = torch.Generator().manual_seed(args.seed)
    tail_start = max(0, total - 50_000_000)

    # Draw calibration starts (discarded, but advances generator)
    _ = torch.randint(0, tail_start - args.seq_len - 1, (n_calib,), generator=g)

    # Draw eval starts
    eval_starts = torch.randint(tail_start, total - args.seq_len - 1,
                                (args.n_eval,), generator=g)
    eval_prompts: list[torch.Tensor] = [
        all_tokens[int(s):int(s) + args.seq_len].long()
        for s in eval_starts.tolist()
    ]
    del all_tokens
    print(f'  {args.n_eval} eval prompts, seq_len={args.seq_len}')

    # ---- Baseline PPL ----
    if args.baseline_ppl is not None:
        fp16_ppl = args.baseline_ppl
        print(f'\n[2/3] Using pre-computed baseline PPL: {fp16_ppl:.4f}')
        baseline_time = 0.0
    elif args.skip_baseline:
        fp16_ppl = float('nan')
        print('\n[2/3] Skipping baseline PPL (--skip_baseline)')
        baseline_time = 0.0
    else:
        print('\n[2/3] Computing baseline PPL...')
        t_bl = time.time()
        fp16_ppl = baseline_ppl_single_gpu(
            hf_id, dtype, eval_prompts, device,
            use_nf4=args.use_nf4_baseline,
        )
        baseline_time = time.time() - t_bl
        print(f'  Baseline PPL: {fp16_ppl:.4f} ({baseline_time:.1f}s)')

    # ---- Compressed PPL ----
    print('\n[3/3] Streaming compressed model PPL evaluation...')
    t_eval = time.time()
    compressed_ppl, eval_peak_vram = streaming_eval_ppl(
        hf_id=hf_id,
        compressed_dir=args.compressed_dir,
        eval_prompts=eval_prompts,
        n_layers=n_layers,
        device=device,
    )
    eval_time = time.time() - t_eval
    print(f'  Compressed PPL: {compressed_ppl:.4f} ({eval_time:.1f}s)')
    print(f'  Eval peak VRAM: {eval_peak_vram:.2f}GB')

    # ---- Results ----
    if not math.isnan(fp16_ppl) and fp16_ppl > 0:
        ppl_ratio = compressed_ppl / fp16_ppl
    else:
        ppl_ratio = float('nan')

    results = {
        'model': args.model,
        'hf_id': hf_id,
        'n_layers': n_layers,
        'n_eval': args.n_eval,
        'seq_len': args.seq_len,
        'seed': args.seed,
        'baseline_ppl': fp16_ppl,
        'baseline_method': (
            'pre-computed' if args.baseline_ppl is not None
            else ('nf4_4bit' if args.use_nf4_baseline else 'bf16_single_gpu')
        ),
        'compressed_ppl': compressed_ppl,
        'ppl_ratio': ppl_ratio,
        'peak_vram_eval_gb': eval_peak_vram,
        'eval_time_s': eval_time,
        'baseline_time_s': baseline_time,
        'compressed_dir': str(args.compressed_dir),
        'device': str(device),
        'cuda_device_name': torch.cuda.get_device_name(device),
        'timestamp': datetime.datetime.now().isoformat(),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(results, indent=2))

    print('\n' + '=' * 78)
    print('RESULTS')
    print('=' * 78)
    print(f'  Model:              {args.model} ({hf_id})')
    print(f'  Layers:             {n_layers}')
    print(f'  Baseline PPL:       {fp16_ppl:.4f}')
    print(f'  Compressed PPL:     {compressed_ppl:.4f}')
    print(f'  PPL Ratio:          {ppl_ratio:.4f}x')
    print(f'  Peak VRAM (eval):   {eval_peak_vram:.2f} GB')
    print(f'  Eval time:          {eval_time:.1f}s')
    print(f'  Output JSON:        {args.out_json}')
    print('=' * 78)

    if not math.isnan(ppl_ratio):
        if ppl_ratio <= 1.01:
            print('  STATUS: PASS (stretch goal: PPL_r <= 1.01x)')
        elif ppl_ratio <= 1.05:
            print('  STATUS: PASS (PPL_r <= 1.05x)')
        else:
            print(f'  STATUS: FAIL (PPL_r = {ppl_ratio:.4f}x > 1.05x)')

    # Print machine-readable JSON to stdout for piping
    print('\n--- JSON ---')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
