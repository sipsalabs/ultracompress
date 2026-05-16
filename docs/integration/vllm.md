# Integration — vLLM

vLLM is the production-default for high-throughput LLM inference on commodity GPU. Native UltraCompress support is on the v0.2 roadmap (target Q3 2026). Until then, the integration story is:

1. Use the UltraCompress reference loader to inflate to FP16, then serve via standard vLLM (lossy on memory savings; works today)
2. Use the UltraCompress reference loader directly (no vLLM path; works today, lower throughput)
3. Wait for native vLLM support in v0.2

## Current path (v0.1, mid-2026)

### Option A — inflate then serve via vLLM

This loses the memory savings but gives you vLLM's throughput. Useful for evaluation; not the final story.

```bash
pip install "ultracompress[torch]" vllm
uc pull sipsalabs/<model-id>
```

```python
from ultracompress_cli.loader import load_model  # v0.1.1+
import torch

# Load + inflate to FP16
model = load_model("./models/sipsalabs_<model-id>").to(torch.float16)

# Save the inflated model in HF Transformers format
model.save_pretrained("./models/qwen3-1.7b-fp16-from-uc")

# Now serve with standard vLLM
# vllm serve ./models/qwen3-1.7b-fp16-from-uc
```

This is essentially "use UltraCompress to get the model and immediately throw away the compression." Use only as a vLLM evaluation path.

### Option B — direct loader, no vLLM

If you want the memory savings, run the model directly via the reference loader, accepting lower throughput than vLLM:

```python
from ultracompress_cli.loader import load_model

model = load_model("./models/sipsalabs_<model-id>").cuda()
# Single-request inference
```

This is what most pre-launch design partners are using during pilots.

## v0.2 path (Q3 2026 roadmap)

We will ship a vLLM plugin that adds UltraCompress as a recognized quantization format:

```bash
pip install "ultracompress[vllm]"   # bundles the plugin
vllm serve sipsalabs/<model-id> --dtype ultracompress
```

The plugin will:

- Register a custom layer kernel that decompresses UltraCompress weights at inference time (avoiding the FP16 inflation cost)
- Maintain vLLM's throughput characteristics (continuous batching, paged attention, etc.)
- Pass through the `ultracompress.json` provenance via `vllm.model.metadata`

## Memory footprint with v0.2 plugin

| Variant | vLLM RAM at runtime |
|---|---|
| Qwen3-1.7B FP16 (vLLM standard) | ~3.5 GB |
| Qwen3-1.7B AWQ-INT4 (vLLM AWQ path) | ~1.0 GB |
| **Qwen3-1.7B UltraCompress 5 bpw lossless (v0.2 plugin)** | **~1.1 GB** |

(Inference speed targets ~2-3× UltraCompress reference loader, comparable to vLLM's AWQ path on equivalent hardware.)

## What you can do today (mid-2026)

- **Evaluation**: use option A above to compare UltraCompress quality vs other quantization methods running through the same vLLM serving path
- **Reference deployment**: use option B for low-traffic / development / single-customer evaluation
- **Open an issue** if you're a vLLM user with a specific deployment scenario; we prioritize the v0.2 plugin work by deployed-customer-impact

## What you'll be able to do post-Q3 2026

- One-line `vllm serve` against UltraCompress artifacts directly
- Continuous-batching throughput comparable to vLLM's AWQ path
- Memory footprint comparable to UltraCompress's native artifact size

## See also

- [Integration with Hugging Face Transformers](transformers.md)
- [Integration with llama.cpp](llamacpp.md)
- [Integration with TensorRT-LLM](tensorrt-llm.md)
- [vLLM upstream](https://github.com/vllm-project/vllm)
