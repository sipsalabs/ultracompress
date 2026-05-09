"""Streaming teacher: compute teacher logits via per-layer load -> forward -> free.

WHY THIS EXISTS:
  Conventional teacher caching (`AutoModelForCausalLM.from_pretrained` then forward)
  loads the full model into VRAM/RAM. For 14B+ dense or 47B+ MoE, that OOMs the 32 GB
  GPU and pegs the 128 GB system RAM. We need to compute teacher logits for
  distillation WITHOUT ever holding more than one decoder layer at a time.

  This module reuses the same per-layer load/forward/free pattern that
  `streaming_compression_runner.streaming_eval_ppl` uses for compressed inference,
  but loads the ORIGINAL teacher layer weights from HF safetensors (via
  `stream_compress.extract_layer_from_shards`) and CACHES the final logits.

PEAK MEMORY:
  GPU: scaffold (embed + norm + lm_head ~ a few GB) + 1 layer (~3-15 GB) + activations
  CPU: n_prompts * seq_len * vocab * 2 bytes  (e.g. 64 * 1024 * 128k * 2 = 16 GB for 405B)
  DISK: depends on caller -- typically the BufferedShardScheduler keeps ~2-3 shards.

USAGE:
    from streaming_teacher import cache_teacher_logits_streaming
    logit_cache = cache_teacher_logits_streaming(
        hf_id="NousResearch/Hermes-3-Llama-3.1-405B",
        calibration_ids=[t1, t2, ...],
        shard_dir=Path("./scratch_shards"),
        device=torch.device("cuda:0"),
    )
"""

from __future__ import annotations

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


def _get_model_classes(hf_id: str):
    """Return (DecoderLayerClass, RotaryEmbeddingClass) for supported architectures."""
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(hf_id, trust_remote_code=True)
    mt = config.model_type
    if mt == 'qwen3':
        from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding
        return Qwen3DecoderLayer, Qwen3RotaryEmbedding
    elif mt == 'qwen3_moe':
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeDecoderLayer, Qwen3MoeRotaryEmbedding
        return Qwen3MoeDecoderLayer, Qwen3MoeRotaryEmbedding
    elif mt in ('qwen2', 'qwen2_5'):
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RotaryEmbedding
        return Qwen2DecoderLayer, Qwen2RotaryEmbedding
    elif mt == 'mistral':
        from transformers.models.mistral.modeling_mistral import MistralDecoderLayer, MistralRotaryEmbedding
        return MistralDecoderLayer, MistralRotaryEmbedding
    elif mt == 'llama':
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding
        return LlamaDecoderLayer, LlamaRotaryEmbedding
    elif mt == 'mixtral':
        from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer, MixtralRotaryEmbedding
        return MixtralDecoderLayer, MixtralRotaryEmbedding
    elif mt == 'phi3':
        from transformers.models.phi3.modeling_phi3 import Phi3DecoderLayer, Phi3RotaryEmbedding
        return Phi3DecoderLayer, Phi3RotaryEmbedding
    elif mt == 'phi':
        # Phi-1 / Phi-2 — different module than phi3
        from transformers.models.phi.modeling_phi import PhiDecoderLayer, PhiRotaryEmbedding
        return PhiDecoderLayer, PhiRotaryEmbedding
    elif mt == 'phimoe':
        from transformers.models.phimoe.modeling_phimoe import PhimoeDecoderLayer, PhimoeRotaryEmbedding
        return PhimoeDecoderLayer, PhimoeRotaryEmbedding
    elif mt == 'olmo2':
        from transformers.models.olmo2.modeling_olmo2 import Olmo2DecoderLayer, Olmo2RotaryEmbedding
        return Olmo2DecoderLayer, Olmo2RotaryEmbedding
    elif mt == 'olmo':
        from transformers.models.olmo.modeling_olmo import OlmoDecoderLayer, OlmoRotaryEmbedding
        return OlmoDecoderLayer, OlmoRotaryEmbedding
    else:
        raise ValueError(f"Unsupported model_type {mt!r} for streaming teacher")


def _strip_layer_prefix(layer_idx: int, layer_tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert keys like `model.layers.27.self_attn.q_proj.weight` -> `self_attn.q_proj.weight`."""
    prefix = f"model.layers.{layer_idx}."
    out = {}
    for k, v in layer_tensors.items():
        if not k.startswith(prefix):
            raise ValueError(f"Key {k!r} does not start with prefix {prefix!r}")
        out[k[len(prefix):]] = v
    return out


@torch.no_grad()
def cache_teacher_logits_streaming(
    hf_id: str,
    calibration_ids: list[torch.Tensor],
    shard_dir: Path,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    log_fn=print,
) -> list[torch.Tensor]:
    """Compute teacher logits via per-layer streaming.

    `shard_dir` must contain the safetensors shards for ALL layers when called
    (in v0). Future v1 will integrate with BufferedShardScheduler so shards are
    downloaded in order and evicted after their layers are processed.

    Returns: list of (1, seq_len, vocab_size) logit tensors, one per prompt,
    on CPU in bf16.
    """
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.masking_utils import create_causal_mask

    from stream_compress import extract_layer_from_shards, plan_compression

    config = AutoConfig.from_pretrained(hf_id, trust_remote_code=True)
    config._attn_implementation = 'eager'
    n_layers = config.num_hidden_layers

    DecoderLayerClass, RotaryEmbClass = _get_model_classes(hf_id)

    log_fn(f"[teacher] planning {hf_id}")
    plan = plan_compression(hf_id)
    weight_map = plan["weight_map"]

    log_fn(f"[teacher] loading scaffold (embed + norm + lm_head)...")
    # Phi-2 (model_type='phi') needs additional device_map entries beyond the
    # llama default. We list both the module name AND the explicit .weight/.bias
    # keys because HF's prefix-match propagation has been observed to fail for
    # some sub-module configurations (verified empirically on Phi-2).
    scaffold_device_map = {
        'model.embed_tokens': str(device),
        'model.embed_tokens.weight': str(device),
        'model.norm': str(device),
        'model.norm.weight': str(device),
        # Phi-2 / phi-1 family — has both weight + bias on final layernorm
        'model.final_layernorm': str(device),
        'model.final_layernorm.weight': str(device),
        'model.final_layernorm.bias': str(device),
        # Phi-2 dropout module (no params but HF tracks it)
        'model.embed_dropout': str(device),
        # Phi-2 module-level RoPE (vs llama per-layer)
        'model.rotary_emb': str(device),
        'lm_head': str(device),
        'lm_head.weight': str(device),
        'lm_head.bias': str(device),
    }
    for i in range(n_layers):
        scaffold_device_map[f'model.layers.{i}'] = 'meta'

    t0 = time.time()
    scaffold = AutoModelForCausalLM.from_pretrained(
        hf_id,
        device_map=scaffold_device_map,
        dtype=dtype,
        attn_implementation='eager',
        low_cpu_mem_usage=True,
    )
    log_fn(f"[teacher] scaffold loaded in {time.time() - t0:.1f}s")

    embed_tokens = scaffold.model.embed_tokens
    # Phi-2 family uses final_layernorm; everyone else uses norm. Try both.
    final_norm = getattr(scaffold.model, 'norm', None) or getattr(scaffold.model, 'final_layernorm', None)
    if final_norm is None:
        raise RuntimeError(f"Could not find final norm layer (model.norm or model.final_layernorm) for {hf_id}")
    lm_head = scaffold.lm_head
    try:
        rotary_emb = RotaryEmbClass(config=config, device=device)
    except TypeError:
        # Older signatures (e.g. PhimoeRotaryEmbedding) only take config.
        rotary_emb = RotaryEmbClass(config=config).to(device)
    del scaffold
    torch.cuda.empty_cache()

    logit_cache: list[torch.Tensor] = []

    for pi, prompt_ids in enumerate(calibration_ids):
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
        try:
            position_embeddings = rotary_emb(hidden, position_ids)
        except (RuntimeError, TypeError):
            # PhimoeRotaryEmbedding-style: forward(x, seq_len:int)
            position_embeddings = rotary_emb(hidden, seqlen)

        # Stream each teacher layer
        for layer_idx in range(n_layers):
            layer_tensors_raw = extract_layer_from_shards(layer_idx, weight_map, shard_dir)
            layer_sd = _strip_layer_prefix(layer_idx, layer_tensors_raw)
            layer_sd = {k: v.to(device=device, dtype=dtype) for k, v in layer_sd.items()}

            with torch.device('meta'):
                layer = DecoderLayerClass(config, layer_idx=layer_idx)
            layer.load_state_dict(layer_sd, strict=False, assign=True)
            layer = layer.to(device=device, dtype=dtype)
            layer.train(False)
            del layer_sd, layer_tensors_raw

            out = layer(
                hidden,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden = out[0] if isinstance(out, tuple) else out

            del layer
            torch.cuda.empty_cache()

        hidden = final_norm(hidden)
        logits = lm_head(hidden)
        logit_cache.append(logits.to(torch.bfloat16).cpu())
        del hidden, logits, ids

        if (pi + 1) % 5 == 0 or (pi + 1) == len(calibration_ids):
            cum_bytes = sum(t.numel() * t.element_size() for t in logit_cache)
            log_fn(f"[teacher] cached prompt {pi + 1}/{len(calibration_ids)} "
                   f"(cumulative cache {cum_bytes / 1e9:.2f} GB CPU)")

    del embed_tokens, final_norm, lm_head, rotary_emb
    torch.cuda.empty_cache()
    return logit_cache


@torch.no_grad()
def cache_teacher_hidden_states_streaming(
    hf_id: str,
    calibration_ids: list[torch.Tensor],
    shard_dir: Path,
    cache_dir: Path,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    log_fn=print,
) -> None:
    """Streaming version of `streaming_compression_runner.cache_teacher_hidden_states`.

    Walks the teacher per-layer (no full-model load) and saves per-layer hidden
    states to disk in the format expected by `compress_single_layer`:

        cache_dir/hidden_layer_000.pt  ([n_prompts, seq_len, hidden_dim] bf16)  # embed output
        cache_dir/hidden_layer_001.pt  # output of layer 0
        ...
        cache_dir/hidden_layer_NNN.pt  # output of layer N-1 (= input to final norm)

    With this cache populated, the existing per-layer trainer in
    `streaming_compression_runner.compress_single_layer` works UNMODIFIED.

    Memory:
      GPU: scaffold + 1 layer + activations
      CPU: per-layer hidden cache (n_prompts * seq_len * hidden * 2 bytes per layer)
           For 405B with 64 prompts * 1024 seq * 16384 hidden * 2 bytes = 2 GB per layer.
           Saved to disk one layer at a time so peak CPU is bounded by 2 layers.
      DISK: 2 GB * (n_layers + 1) for the cache. For 405B = 254 GB.
            Acceptable since stream-compress saves ~795 GB on the model itself.
    """
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.masking_utils import create_causal_mask
    from stream_compress import extract_layer_from_shards, plan_compression

    cache_dir.mkdir(parents=True, exist_ok=True)

    config = AutoConfig.from_pretrained(hf_id, trust_remote_code=True)
    config._attn_implementation = 'eager'
    n_layers = config.num_hidden_layers

    DecoderLayerClass, RotaryEmbClass = _get_model_classes(hf_id)

    log_fn(f"[teacher-hidden] planning {hf_id}")
    plan = plan_compression(hf_id)
    weight_map = plan["weight_map"]

    log_fn(f"[teacher-hidden] loading scaffold (embed only — norm/lm_head not needed for hidden cache)...")
    scaffold_device_map = {
        'model.embed_tokens': str(device),
        'model.norm': 'meta',  # not needed; saves a few GB on big models
        'lm_head': 'meta',
    }
    for i in range(n_layers):
        scaffold_device_map[f'model.layers.{i}'] = 'meta'

    t0 = time.time()
    scaffold = AutoModelForCausalLM.from_pretrained(
        hf_id,
        device_map=scaffold_device_map,
        dtype=dtype,
        attn_implementation='eager',
        low_cpu_mem_usage=True,
    )
    log_fn(f"[teacher-hidden] scaffold loaded in {time.time() - t0:.1f}s")

    embed_tokens = scaffold.model.embed_tokens
    try:
        rotary_emb = RotaryEmbClass(config=config, device=device)
    except TypeError:
        # Older signatures (e.g. PhimoeRotaryEmbedding) only take config.
        rotary_emb = RotaryEmbClass(config=config).to(device)
    del scaffold
    torch.cuda.empty_cache()

    # Pass 1: compute hidden_layer_000 = embed output for all prompts
    log_fn(f"[teacher-hidden] caching hidden_layer_000 (embed output)...")
    embed_out_cpu = []
    for prompt_ids in calibration_ids:
        ids = prompt_ids.unsqueeze(0).long().to(device)
        emb = embed_tokens(ids).to(dtype)
        embed_out_cpu.append(emb.cpu())
    hidden_0 = torch.cat(embed_out_cpu, dim=0)  # [n_prompts, seq_len, hidden]
    torch.save(hidden_0, cache_dir / 'hidden_layer_000.pt')
    log_fn(f"[teacher-hidden]   hidden_layer_000.pt: {hidden_0.shape} {hidden_0.nbytes / 1e6:.1f}MB")

    # Pass 2: per-layer streaming forward, save output as next layer's input cache
    current_hidden = hidden_0  # CPU tensor
    for layer_idx in range(n_layers):
        log_fn(f"[teacher-hidden] layer {layer_idx + 1}/{n_layers}: load + forward + save")
        layer_tensors_raw = extract_layer_from_shards(layer_idx, weight_map, shard_dir)
        layer_sd = _strip_layer_prefix(layer_idx, layer_tensors_raw)
        layer_sd = {k: v.to(device=device, dtype=dtype) for k, v in layer_sd.items()}

        with torch.device('meta'):
            layer = DecoderLayerClass(config, layer_idx=layer_idx)
        layer.load_state_dict(layer_sd, strict=False, assign=True)
        layer = layer.to(device=device, dtype=dtype)
        layer.train(False)
        del layer_sd, layer_tensors_raw

        # Forward each prompt through this layer; collect outputs to CPU
        out_cpu = []
        for pi in range(current_hidden.shape[0]):
            h_in = current_hidden[pi:pi + 1].to(device)
            seqlen = h_in.shape[1]
            cache_position = torch.arange(seqlen, device=device)
            position_ids = cache_position.unsqueeze(0)
            causal_mask = create_causal_mask(
                config=config, input_embeds=h_in, attention_mask=None,
                cache_position=cache_position, past_key_values=None,
                position_ids=position_ids,
            )
            try:
                position_embeddings = rotary_emb(h_in, position_ids)
            except (RuntimeError, TypeError):
                # PhimoeRotaryEmbedding-style: forward(x, seq_len:int)
                position_embeddings = rotary_emb(h_in, seqlen)
            out = layer(
                h_in, attention_mask=causal_mask, position_ids=position_ids,
                past_key_values=None, use_cache=False,
                cache_position=cache_position, position_embeddings=position_embeddings,
            )
            h_out = out[0] if isinstance(out, tuple) else out
            out_cpu.append(h_out.cpu())
            del h_in, h_out, out

        del layer
        torch.cuda.empty_cache()

        next_hidden = torch.cat(out_cpu, dim=0)
        torch.save(next_hidden, cache_dir / f'hidden_layer_{layer_idx + 1:03d}.pt')
        current_hidden = next_hidden

    # Manifest for cache validity check (matches existing runner's pattern)
    import json as _json
    (cache_dir / 'manifest.json').write_text(_json.dumps({
        'hf_id': hf_id,
        'n_layers': n_layers,
        'n_prompts': len(calibration_ids),
        'method': 'streaming_teacher_hidden_cache_v0',
    }))
    log_fn(f"[teacher-hidden] DONE. {n_layers + 1} hidden caches in {cache_dir}")

    del embed_tokens, rotary_emb
    torch.cuda.empty_cache()


def main() -> int:
    """CLI: compare streaming-teacher logits vs full-teacher logits on a small model.

    For Qwen3-1.7B (small enough to load full teacher), this verifies that the
    streaming approach produces the SAME logits as the conventional approach
    (max-abs-diff should be near float-precision noise).
    """
    import argparse
    ap = argparse.ArgumentParser(description="Streaming-teacher self-check vs full-teacher")
    ap.add_argument("--hf-id", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--shard-dir", type=Path, required=True,
                    help="Local dir containing the model's safetensors shards")
    ap.add_argument("--n-prompts", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16

    print(f"[selfcheck] hf_id={args.hf_id}, n_prompts={args.n_prompts}, seq_len={args.seq_len}")

    torch.manual_seed(42)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(args.hf_id, trust_remote_code=True)
    vocab = tok.vocab_size
    calibration_ids = [
        torch.randint(0, vocab, (args.seq_len,), dtype=torch.long)
        for _ in range(args.n_prompts)
    ]

    print("[selfcheck] streaming-teacher pass...")
    streamed_logits = cache_teacher_logits_streaming(
        hf_id=args.hf_id,
        calibration_ids=calibration_ids,
        shard_dir=args.shard_dir,
        device=device,
        dtype=dtype,
    )

    print("[selfcheck] full-teacher pass (control, eager attention to match streaming)...")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.hf_id, dtype=dtype, device_map=device, attn_implementation='eager',
    )
    teacher.train(False)
    full_logits = []
    with torch.no_grad():
        for ids in calibration_ids:
            out = teacher(ids.unsqueeze(0).to(device))
            full_logits.append(out.logits.to(torch.bfloat16).cpu())
    del teacher
    torch.cuda.empty_cache()

    print("[selfcheck] comparing...")
    for i, (s, f) in enumerate(zip(streamed_logits, full_logits)):
        if s.shape != f.shape:
            print(f"  prompt {i}: SHAPE MISMATCH {s.shape} vs {f.shape}")
            continue
        diff = (s.float() - f.float()).abs()
        print(f"  prompt {i}: max-abs-diff={diff.max().item():.6f}  "
              f"mean-abs-diff={diff.mean().item():.6f}  "
              f"shape={tuple(s.shape)}")

    print("[selfcheck] done. If max-abs-diff is < 0.01 the streaming pass matches.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
