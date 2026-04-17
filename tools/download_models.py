"""
Download and cache Qwen3 models for FRR scaling tests.
Downloads to HuggingFace cache, then converts to our .pt format.
Run on CPU — no GPU needed. Can run while training is happening.
"""
import lib.unbuffered
import torch
import os
import sys
import time
import gc

def download_and_cache(model_name, cache_filename):
    """Download a model from HuggingFace and save as .pt cache."""
    if os.path.exists(cache_filename):
        size_gb = os.path.getsize(cache_filename) / 1e9
        print(f"  {cache_filename} already exists ({size_gb:.1f} GB), skipping")
        return True

    print(f"  Downloading {model_name}...")
    t0 = time.time()

    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, device_map="cpu"
        )
        state_dict = model.state_dict()

        # Count params
        total = sum(v.numel() for v in state_dict.values())
        print(f"  Loaded: {len(state_dict)} tensors, {total:,} params")

        # Save
        torch.save(state_dict, cache_filename)
        size_gb = os.path.getsize(cache_filename) / 1e9
        print(f"  Saved to {cache_filename} ({size_gb:.1f} GB) in {time.time()-t0:.0f}s")

        # Print architecture info
        config = model.config
        print(f"  Architecture: {config.num_hidden_layers} layers, "
              f"hidden={config.hidden_size}, "
              f"heads={config.num_attention_heads}, "
              f"kv_heads={config.num_key_value_heads}, "
              f"intermediate={config.intermediate_size}, "
              f"vocab={config.vocab_size}")

        del model, state_dict
        gc.collect()
        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        return False


if __name__ == '__main__':
    print("=" * 60)
    print("DOWNLOAD QWEN3 MODELS FOR SCALING TESTS")
    print("=" * 60)

    models = [
        ("Qwen/Qwen3-0.6B", "qwen3_0.6b_cache.pt"),
        ("Qwen/Qwen3-1.7B", "qwen3_1.7b_cache.pt"),
        # ("Qwen/Qwen3-4B", "qwen3_4b_cache.pt"),   # ~16GB, uncomment if needed
        # ("Qwen/Qwen3-8B", "qwen3_8b_cache.pt"),   # ~32GB, uncomment if needed
    ]

    for model_name, cache_file in models:
        print(f"\n{'='*60}")
        print(f"  {model_name} -> {cache_file}")
        print(f"{'='*60}")
        download_and_cache(model_name, cache_file)

    print("\nDone! Models cached for scaling tests.")
