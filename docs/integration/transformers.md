# Integration — Hugging Face Transformers

UltraCompress artifacts are designed to work with the existing Hugging Face Transformers ecosystem. As of v0.1, you load them via the `ultracompress` package's loader; native Transformers support is on the v0.2 roadmap.

## v0.1 (current) — using the UltraCompress loader

```python
from ultracompress_cli.loader import load_model  # available v0.1.1+
from transformers import AutoTokenizer

model_path = "./models/sipsalabs_<model-id>"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = load_model(model_path)  # returns a torch.nn.Module

inputs = tokenizer("Why is the sky blue?", return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

The loader inflates the compressed weights to a runnable representation. Inference speed will be similar to FP16 (the compression is for memory, not speed) until we ship the FP4 / NF4 / quantized-runtime kernel path in v0.2.

## v0.2 (Q3 2026 roadmap) — native AutoModel support

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "sipsalabs/<model-id>",
    quantization_config="ultracompress",
)
tokenizer = AutoTokenizer.from_pretrained("sipsalabs/<model-id>")
```

This will require:
- A `ultracompress` quantization-config registered with Transformers' BNB/HQQ-style plugin pattern
- Or a custom `from_pretrained` extension shipped in our package

We're working with the Hugging Face team on the right integration shape.

## Memory savings vs. FP16

| Variant | Memory at runtime |
|---|---|
| Qwen3-1.7B FP16 | ~3.5 GB |
| Qwen3-1.7B int8 (bitsandbytes) | ~1.8 GB |
| Qwen3-1.7B NF4 (bitsandbytes) | ~1.0 GB |
| **Qwen3-1.7B UltraCompress 5 bpw (lossless)** | **~1.1 GB** |

(All measured on a CUDA device with the loader's default inflation; "memory at runtime" is `torch.cuda.memory_allocated()` after model load.)

## Inference speed

UltraCompress v0.1 inflates compressed weights at load time, so inference speed at runtime is similar to FP16. Expected **2-3× speedup** lands in v0.2 when the quantized-runtime kernel path ships.

## Compatibility

Tested with:
- transformers >= 4.45.0
- torch >= 2.4.0
- safetensors >= 0.4.5
- huggingface_hub >= 0.24.0

Reach out if you need older-version compatibility.

## See also

- [Integration with vLLM](vllm.md)
- [Integration with TensorRT-LLM](tensorrt-llm.md)
- [Integration with llama.cpp](llamacpp.md)
- [Manifest schema](../reference/manifest-schema.md)
