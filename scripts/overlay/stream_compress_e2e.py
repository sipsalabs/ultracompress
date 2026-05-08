"""Stream-compress + streaming-teacher END-TO-END production runner.

Composes:
  1. streaming_teacher.cache_teacher_hidden_states_streaming  (per-layer teacher cache)
  2. streaming_compression_runner.compress_single_layer       (per-layer V18-C train)

This is the v1 frontier-scale runner. With both pieces, ANY model with a HF
safetensors index can be compressed:
  - GPU peak: scaffold + 1 layer + activations  (~10-20 GB across model sizes)
  - CPU peak: 2 layers' worth of hidden cache during teacher pass (~4 GB)
  - DISK peak: model.safetensors (already-downloaded; not re-downloaded)
               + per-layer hidden cache (~2 GB/layer for 405B = 254 GB total)
               + per-layer compressed artifacts (~120 GB for 405B at 5 bpw)

USAGE:
    # Smoke test (Qwen3-1.7B, validates pipeline end-to-end fast):
    python scripts/overlay/stream_compress_e2e.py \\
        --hf-id Qwen/Qwen3-1.7B \\
        --shard-dir scripts/overlay/_qwen3_17b_shards \\
        --output ./compressed/qwen3-1.7b-stream-e2e \\
        --bpw 5 --rank 32 --train-steps 50 \\
        --n-calib 16 --seq-len 512 \\
        --device cuda:0

    # Frontier scale (Hermes-3-405B, the headline run):
    python scripts/overlay/stream_compress_e2e.py \\
        --hf-id NousResearch/Hermes-3-Llama-3.1-405B \\
        --shard-dir /c/Users/scamd/.cache/huggingface/hub/models--NousResearch--Hermes-3-Llama-3.1-405B/snapshots/<sha> \\
        --output ./compressed/hermes-3-405b-stream-e2e \\
        --bpw 5 --rank 32 --train-steps 200 \\
        --n-calib 64 --seq-len 1024 \\
        --device cuda:0

NOTE: shard-dir must contain the HF safetensors files (use the snapshot dir from
HF cache). Future v2 will integrate BufferedShardScheduler to download+evict
shards on the fly; v1 assumes shards are already on disk.
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import torch

if sys.platform == "win32" and sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))


def main() -> int:
    ap = argparse.ArgumentParser(description="Stream-compress + streaming-teacher end-to-end (v1)")
    ap.add_argument("--hf-id", required=True)
    ap.add_argument("--shard-dir", type=Path, required=True,
                    help="Local dir with safetensors shards (use HF cache snapshot dir)")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--bpw", type=int, default=5)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--train-steps", type=int, default=200)
    ap.add_argument("--train-lr", type=float, default=1e-3)
    ap.add_argument("--train-bs", type=int, default=8)
    ap.add_argument("--n-calib", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--max-layers", type=int, default=0,
                    help="Stop after compressing N layers (0 = all)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip-cache", action="store_true",
                    help="Reuse existing teacher hidden cache (skip Phase 1)")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16

    args.output.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output / "_teacher_hidden_cache"

    # Plan
    from stream_compress import plan_compression
    plan = plan_compression(args.hf_id)
    n_layers = plan["n_layers_total"]
    print(f"[e2e] hf_id={args.hf_id}")
    print(f"[e2e] layers={n_layers}, shards={plan['n_shards']}, scaffold_keys={len(plan['scaffold_keys'])}")
    print(f"[e2e] device={device}  bpw={args.bpw}  rank={args.rank}  steps={args.train_steps}")

    # ---- Phase 1: build teacher hidden cache via streaming ----
    if args.skip_cache and (cache_dir / 'manifest.json').exists():
        print(f"\n[e2e] Phase 1 SKIPPED (--skip-cache, manifest exists at {cache_dir})")
    else:
        print(f"\n[e2e] Phase 1: streaming teacher hidden cache...")
        t0 = time.time()
        # Build calibration prompts from real FineWeb-edu tokens (model-specific cache).
        # Try several slug variants to find a model-specific tokenized cache.
        # Existing files use slugs like "llama_3_1_8b", "qwen3_8b", "mistral_7b_v03".
        from pathlib import Path as _P
        repo_root = _P(__file__).parent.parent.parent  # ultracompress/
        last_part = args.hf_id.split('/')[-1].lower()
        candidate_slugs = {
            args.hf_id.replace('/', '_').replace('-', '_').replace('.', '_').lower(),
            last_part.replace('-', '_').replace('.', '_'),
            # Variant: remove dots entirely (mistral-7b-v0.3 → mistral_7b_v03)
            last_part.replace('-', '_').replace('.', ''),
            # Variant: replace dots with empty AND hyphens with _
            last_part.replace('.', '').replace('-', '_'),
        }
        # Strip common prefixes from model name for better slug match
        for prefix in ('meta_llama_', 'meta-llama-', 'meta_'):
            if last_part.startswith(prefix):
                stripped = last_part[len(prefix):].replace('-', '_').replace('.', '_')
                candidate_slugs.add(stripped)
                candidate_slugs.add('llama_' + stripped)
        # Strip common instruct/chat suffixes for better slug match
        for suffix in ('_instruct', '-instruct', '_chat', '-chat', '-it'):
            for s in list(candidate_slugs):
                if s.endswith(suffix.replace('-', '_')):
                    candidate_slugs.add(s[:-len(suffix)])
                if s.endswith(suffix):
                    candidate_slugs.add(s[:-len(suffix)])
        # Also try the model registry style: "llama-3.1-8b" -> "llama_3_1_8b"
        for suffix_size in ('1.7b', '7b', '8b', '14b', '32b', '70b', '72b', '405b', '235b',
                            '8x7b', '8x22b'):
            for arch in ('qwen3', 'qwen25', 'mistral', 'llama_3_1', 'llama_3', 'phi_3_5', 'mixtral'):
                if suffix_size.replace('.', '_') in last_part.replace('-', '_').replace('.', '_'):
                    candidate_slugs.add(f'{arch}_{suffix_size.replace(".", "_")}')
        # Strip version suffixes like -v0.1 → mixtral_8x22b
        for suffix in ('_v0_1', '_v01', '-v0.1', '-v0_1', '_v0', '-v0'):
            for s in list(candidate_slugs):
                if s.endswith(suffix.replace('-', '_').replace('.', '_')):
                    candidate_slugs.add(s[:-len(suffix)])
                if s.endswith(suffix):
                    candidate_slugs.add(s[:-len(suffix)])
        token_paths = []
        for s in sorted(candidate_slugs):
            for size in ('500M', '100M', '10M'):
                p = repo_root / f'fineweb_edu_{size}_tokens_{s}.pt'
                if p.exists():
                    token_paths.append(p)
        token_paths.extend([
            repo_root / 'fineweb_edu_500M_tokens.pt',
            repo_root / 'fineweb_edu_100M_tokens.pt',
        ])
        token_path = next((p for p in token_paths if p.exists()), None)
        if token_path is not None:
            print(f"[e2e]   loading real calibration from {token_path.name}")
            all_tokens = torch.load(token_path, weights_only=True)
            torch.manual_seed(42)
            g = torch.Generator().manual_seed(42)
            n_tokens = all_tokens.numel()
            max_start = n_tokens - args.seq_len - 1
            starts = torch.randint(0, max_start, (args.n_calib,), generator=g)
            calibration_ids = [
                all_tokens[int(s):int(s) + args.seq_len].long() for s in starts.tolist()
            ]
            del all_tokens
        else:
            print(f"[e2e]   WARNING: no FineWeb-edu cache found — falling back to RANDOM tokens (PPL will be poor)")
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(args.hf_id, trust_remote_code=True)
            torch.manual_seed(42)
            calibration_ids = [
                torch.randint(0, tok.vocab_size, (args.seq_len,), dtype=torch.long)
                for _ in range(args.n_calib)
            ]
        from streaming_teacher import cache_teacher_hidden_states_streaming
        cache_teacher_hidden_states_streaming(
            hf_id=args.hf_id,
            calibration_ids=calibration_ids,
            shard_dir=args.shard_dir,
            cache_dir=cache_dir,
            device=device,
            dtype=dtype,
        )
        print(f"[e2e] Phase 1 done in {time.time() - t0:.1f}s")

    # ---- Phase 2: per-layer V18-C training using existing trainer ----
    print(f"\n[e2e] Phase 2: per-layer V18-C training")
    from streaming_compression_runner import compress_single_layer

    n_to_compress = args.max_layers if args.max_layers > 0 else n_layers
    layer_metrics: list[dict] = []
    t_phase2 = time.time()
    for layer_idx in range(min(n_to_compress, n_layers)):
        # SKIP-EXISTING resume support: if layer_NNN.pt already saved, reuse it
        existing_path = args.output / f'layer_{layer_idx:03d}.pt'
        if existing_path.exists():
            print(f"\n[e2e] --- Layer {layer_idx + 1}/{n_to_compress} --- SKIP (already saved at {existing_path.name})")
            try:
                cached = torch.load(existing_path, map_location='cpu', weights_only=False)
                cached_metrics = cached.get('metrics', {'layer_idx': layer_idx, 'resumed': True})
                layer_metrics.append({"layer": layer_idx, **cached_metrics})
            except Exception:
                layer_metrics.append({"layer": layer_idx, "resumed": True})
            continue
        print(f"\n[e2e] --- Layer {layer_idx + 1}/{n_to_compress} ---")
        # Note: compress_single_layer uses load_layer_state_dict internally to
        # get layer weights from the model — we don't pass them in directly.
        # It walks safetensors via HF cache, which is the same data as our
        # shard_dir. So we DO NOT need to inject anything; the existing
        # function works as long as the model is in HF cache.
        try:
            metrics = compress_single_layer(
                hf_id=args.hf_id,
                dtype=dtype,
                layer_idx=layer_idx,
                n_layers=n_layers,
                hidden_cache_dir=cache_dir,
                output_dir=args.output,
                bpw=args.bpw,
                block_size=args.block_size,
                rank=args.rank,
                train_steps=args.train_steps,
                train_lr=args.train_lr,
                train_bs=args.train_bs,
                device=device,
            )
            layer_metrics.append({"layer": layer_idx, **metrics})
            # Stringify safely — metrics keys vary between runners
            summary = ", ".join(f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}"
                                for k, v in metrics.items())
            print(f"[e2e] layer {layer_idx} done: {summary}")
        except Exception as e:
            print(f"[e2e] layer {layer_idx} FAILED: {type(e).__name__}: {e}")
            raise

    print(f"\n[e2e] DONE. {len(layer_metrics)} layers compressed in {time.time() - t_phase2:.1f}s")
    print(f"[e2e] Output: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
