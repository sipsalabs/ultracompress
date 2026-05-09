"""bench_inference_throughput.py — sales-grade inference throughput benchmark.

Measures TPS (tokens-per-second) and TTFT (time-to-first-token) of an
UltraCompress v3 packed model vs the bf16 baseline of the same model,
on the same prompt, on the same GPU.

Output: JSON summary that customers can replicate to verify perf claims.

Usage:
    python scripts/overlay/bench_inference_throughput.py \\
        --model qwen3-1.7b-base \\
        --packed-dir _packed_qwen3_1_7b_base_v3 \\
        --device cuda:1 \\
        --n-new-tokens 256 \\
        --output bench_inference_qwen3-1.7b-base.json

Notes:
- Inference uses the CorrectionMatrixC composition path (same as PPL eval),
  so the throughput number reflects exactly what customers see with the
  standard reload via streaming_compression_runner.
- Baseline is loaded from the original HF model id at bf16 on a single GPU.
- Both runs use the same prompt and the same `model.generate(...)` config
  for apples-to-apples comparison.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT))

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from streaming_compression_runner import MODEL_REGISTRY, free_memory, vram_gb


DEFAULT_PROMPT = (
    "Explain the concept of model compression for large language models in a way that "
    "a non-technical executive could understand, with one concrete example."
)


@torch.no_grad()
def benchmark_generate(model, tokenizer, prompt: str, n_new: int, device: torch.device) -> dict:
    """Run model.generate() and measure TTFT + TPS + peak VRAM.

    TTFT is approximated by the time to generate the first token (prefill + 1 decode).
    TPS is averaged over the n_new generated tokens.
    """
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    n_prompt = inputs.input_ids.shape[1]

    # Warmup pass (compile / kernel cache prewarm)
    _ = model.generate(
        **inputs, max_new_tokens=4, do_sample=False,
        pad_token_id=tokenizer.eos_token_id or 0,
    )
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    # TTFT measurement: 1 new token (prefill + 1 decode)
    t0 = time.time()
    _ = model.generate(
        **inputs, max_new_tokens=1, do_sample=False,
        pad_token_id=tokenizer.eos_token_id or 0,
    )
    torch.cuda.synchronize(device)
    ttft_s = time.time() - t0

    # Full generation: n_new tokens
    t0 = time.time()
    out_full = model.generate(
        **inputs, max_new_tokens=n_new, do_sample=False,
        pad_token_id=tokenizer.eos_token_id or 0,
    )
    torch.cuda.synchronize(device)
    full_dt_s = time.time() - t0

    n_generated = out_full.shape[1] - n_prompt
    tps_total = n_generated / full_dt_s
    decode_dt_s = max(full_dt_s - ttft_s, 1e-6)
    decode_tps = (n_generated - 1) / decode_dt_s
    peak_vram = torch.cuda.max_memory_allocated(device) / (1024**3)

    return {
        'n_prompt_tokens': n_prompt,
        'n_generated_tokens': n_generated,
        'ttft_s': round(ttft_s, 4),
        'full_generation_s': round(full_dt_s, 4),
        'tps_overall': round(tps_total, 2),
        'tps_decode_only': round(decode_tps, 2),
        'peak_vram_gb': round(peak_vram, 3),
    }


def load_baseline_bf16(hf_id: str, device: torch.device) -> torch.nn.Module:
    """Load the standard bf16 HF model on a single device."""
    print(f'[baseline] loading {hf_id} bf16 on {device}...', flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.bfloat16, device_map=device,
        trust_remote_code=True,
    )
    model.train(False)
    print(f'[baseline] loaded in {time.time()-t0:.1f}s, VRAM={vram_gb(device):.2f}GB', flush=True)
    return model


def load_packed_v3(hf_id: str, e2e_dir: Path, n_layers: int,
                    device: torch.device) -> torch.nn.Module:
    """Load packed v3 model — replaces transformer.layers[i] with reconstructed layers.

    For inference we keep ALL layers resident (no streaming) — accepts higher
    VRAM cost in exchange for fast generate(). Customers running 70B+ models
    will use the streaming_eval_ppl streaming path; this is for ≤14B real-time.
    """
    from streaming_compression_runner import (
        get_model_classes, CorrectionMatrixC, load_layer_state_dict,
    )

    print(f'[uc-v3] loading {hf_id} + {n_layers} packed layers...', flush=True)
    t0 = time.time()

    # Load full HF model bf16 (we'll replace transformer.layers[i] with packed reconstructions)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.bfloat16, device_map=device,
        trust_remote_code=True,
    )
    model.train(False)

    layers = model.model.layers
    n_replaced = 0
    for i in range(n_layers):
        layer_pt = e2e_dir / f'layer_{i:03d}.pt'
        if not layer_pt.exists():
            print(f'[uc-v3] WARN: layer {i} not found at {layer_pt}, leaving baseline weights',
                  flush=True)
            continue
        layer_sd, corrections = load_layer_state_dict(layer_pt, target_dtype=torch.bfloat16)
        for name, cd in corrections.items():
            rank = cd['V_weight'].shape[0]
            parts = name.split('.')
            parent = layers[i]
            for p in parts[:-1]:
                parent = getattr(parent, p)
            attr = parts[-1]
            existing = getattr(parent, attr)
            cm = CorrectionMatrixC(
                existing.weight.data.clone(),
                existing.bias.data.clone() if existing.bias is not None else None,
                rank=rank,
            )
            setattr(parent, attr, cm)
        layers[i].load_state_dict(layer_sd, strict=False, assign=False)
        layers[i] = layers[i].to(device=device, dtype=torch.bfloat16)
        n_replaced += 1
    free_memory()
    print(f'[uc-v3] replaced {n_replaced}/{n_layers} layers in {time.time()-t0:.1f}s, '
          f'VRAM={vram_gb(device):.2f}GB', flush=True)
    return model


def main() -> int:
    ap = argparse.ArgumentParser(description='UltraCompress v3 inference throughput benchmark')
    ap.add_argument('--model', required=True, choices=list(MODEL_REGISTRY.keys()))
    ap.add_argument('--e2e-dir', required=True, type=Path,
                    help='Path to _e2e_<arch> directory containing layer_NNN.pt files')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--n-new-tokens', type=int, default=256)
    ap.add_argument('--prompt', default=DEFAULT_PROMPT)
    ap.add_argument('--output', type=Path, required=True,
                    help='Output JSON path')
    ap.add_argument('--baseline-only', action='store_true')
    ap.add_argument('--uc-only', action='store_true')
    args = ap.parse_args()

    cfg = MODEL_REGISTRY[args.model]
    hf_id = cfg['hf_id']
    n_layers = cfg['n_layers']
    device = torch.device(args.device)

    print(f'\n=== UltraCompress v3 Inference Throughput Benchmark ===')
    print(f'  model: {args.model} ({hf_id})')
    print(f'  e2e_dir: {args.e2e_dir}')
    print(f'  device: {device}  ({torch.cuda.get_device_name(device)})')
    print(f'  prompt_len: {len(args.prompt)} chars')
    print(f'  n_new_tokens: {args.n_new_tokens}\n')

    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)

    result = {
        'model': args.model,
        'hf_id': hf_id,
        'n_layers': n_layers,
        'device': args.device,
        'cuda_device_name': torch.cuda.get_device_name(device),
        'n_new_tokens': args.n_new_tokens,
        'prompt': args.prompt,
        'timestamp': datetime.datetime.now().isoformat(),
        'baseline': None,
        'uc_v3': None,
    }

    if not args.uc_only:
        print('--- Baseline (bf16) ---', flush=True)
        baseline_model = load_baseline_bf16(hf_id, device)
        baseline_metrics = benchmark_generate(baseline_model, tokenizer,
                                               args.prompt, args.n_new_tokens, device)
        result['baseline'] = baseline_metrics
        print(f'  TTFT={baseline_metrics["ttft_s"]:.3f}s  TPS_overall={baseline_metrics["tps_overall"]:.1f}  '
              f'TPS_decode={baseline_metrics["tps_decode_only"]:.1f}  '
              f'VRAM={baseline_metrics["peak_vram_gb"]:.2f}GB', flush=True)
        del baseline_model
        free_memory()

    if not args.baseline_only:
        print('\n--- UC v3 packed (5 bpw GSQ + V18-C correction) ---', flush=True)
        uc_model = load_packed_v3(hf_id, args.e2e_dir, n_layers, device)
        uc_metrics = benchmark_generate(uc_model, tokenizer,
                                         args.prompt, args.n_new_tokens, device)
        result['uc_v3'] = uc_metrics
        print(f'  TTFT={uc_metrics["ttft_s"]:.3f}s  TPS_overall={uc_metrics["tps_overall"]:.1f}  '
              f'TPS_decode={uc_metrics["tps_decode_only"]:.1f}  '
              f'VRAM={uc_metrics["peak_vram_gb"]:.2f}GB', flush=True)
        del uc_model
        free_memory()

    if result['baseline'] and result['uc_v3']:
        b = result['baseline']
        u = result['uc_v3']
        result['comparison'] = {
            'tps_decode_uc_vs_baseline_pct': round(
                100 * (u['tps_decode_only'] - b['tps_decode_only']) / b['tps_decode_only'], 1
            ),
            'ttft_uc_vs_baseline_pct': round(
                100 * (u['ttft_s'] - b['ttft_s']) / b['ttft_s'], 1
            ),
            'vram_uc_vs_baseline_pct': round(
                100 * (u['peak_vram_gb'] - b['peak_vram_gb']) / b['peak_vram_gb'], 1
            ),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'\n[bench] saved JSON: {args.output}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
