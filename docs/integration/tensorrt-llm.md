# Integration — NVIDIA TensorRT-LLM

NVIDIA TensorRT-LLM is the production-grade NVIDIA-GPU inference runtime, with deep integration into NVIDIA Inference Microservices (NIM), Triton Inference Server, and the broader NVIDIA serving stack.

Native UltraCompress support in TensorRT-LLM is on our v0.2 roadmap (target Q3 2026). Today, the integration story is similar to llama.cpp and vLLM: use the reference loader for evaluation, await the export path for production.

## Current path (v0.1, mid-2026)

If you're evaluating UltraCompress on NVIDIA hardware:

```bash
pip install "ultracompress[torch]"
uc pull sipsalabs/<model-id>
```

Then use the UltraCompress reference loader directly. It runs on CUDA and gives you correct (though slower than TensorRT-LLM) inference for evaluation purposes.

For production-grade throughput on NVIDIA GPUs today: inflate to FP16 and use TensorRT-LLM's standard FP16 or INT4 path. You give up the compression at runtime; you keep it on-disk + at distribution time.

## v0.2 path (Q3 2026 roadmap)

We will ship `uc export --format trtllm-engine` that produces a TensorRT-LLM-compatible inference engine:

```bash
uc export ./models/sipsalabs_<model-id> \
    --format trtllm-engine \
    --target-arch sm_90 \
    -o qwen3-uc.engine
```

Engine file is then loadable directly by TensorRT-LLM's standard runtime:

```python
from tensorrt_llm import Runner
runner = Runner.from_engine("qwen3-uc.engine")
```

The export path will:

- Convert UltraCompress weights into TensorRT-LLM's native quantization format (mostly likely INT4-AWQ-style or W4A8-INT8-style depending on target arch)
- Preserve UltraCompress provenance in the engine's metadata (`engine.metadata.ultracompress = {bpw, method, patents}`)
- Be optimized for the target SM architecture (Ada / Hopper / Blackwell)

## Memory + speed at production scale

TensorRT-LLM is the strongest existing production inference path for NVIDIA GPUs. Once we land the v0.2 export, expected numbers:

| Variant | TRT-LLM inference latency (1.7B model, batch=1, A100) | Memory |
|---|---|---|
| FP16 | ~12 ms / token | ~3.5 GB |
| INT4-AWQ (existing) | ~6 ms / token | ~0.9 GB |
| **UltraCompress export → INT4-AWQ-on-NVIDIA** | **~6 ms / token** | **~0.6 GB** |

The on-NVIDIA-runtime memory delta vs INT4-AWQ is modest (~30%); the **distribution-time** advantage is much larger (downloadable artifact is ~2.7× smaller).

For chip vendors and OEMs targeting non-NVIDIA inference paths (Snapdragon, Apple Silicon, etc.), the memory and distribution advantages are larger; the TensorRT-LLM path is mainly relevant for cloud customers.

## NVIDIA Inference Microservices (NIM)

NIM bundles TensorRT-LLM with a standardized serving API. Once we land the v0.2 TensorRT-LLM engine export, customers will be able to:

```bash
# Build a NIM container with our engine
docker build -f Dockerfile.nim -t mycorp/qwen3-uc-nim .
# Serve via NIM's standard chat-completion API
docker run --gpus all -p 8000:8000 mycorp/qwen3-uc-nim
```

This is the deployment path we'll recommend for cloud-native enterprise customers.

## What you can do today (mid-2026)

- Run UltraCompress through the reference loader on CUDA for evaluation
- Inflate to FP16 and use TensorRT-LLM's standard FP16 path for production-quality throughput (loses compression at runtime)
- Open a GitHub issue with your specific TensorRT-LLM target arch + workload so we prioritize the v0.2 export correctly

## What you'll be able to do post-Q3 2026

- `uc export --format trtllm-engine` for direct TensorRT-LLM integration
- NIM-compatible container distribution
- INT4-AWQ-comparable inference latency with smaller download artifacts

## A note on patent-licensing alignment

The UltraCompress methods are patent-pending. NVIDIA TensorRT-LLM is a closed-source NVIDIA product. Our integration works at the **export-format** level — we produce a TensorRT-LLM-compatible engine — without requiring NVIDIA to integrate or license our methods.

This is an important architectural choice. Customers using TensorRT-LLM directly are unaffected by our patents (they're using NVIDIA's existing W4A16 / INT4 AWQ paths). Customers using **our exported artifact** through TensorRT-LLM are using our patented methods at the **artifact-production** stage and need a license; the runtime is NVIDIA's, the artifact is ours.

If you're a chip vendor or hyperscaler who wants to integrate UltraCompress more deeply into your inference stack, email **legal@sipsalabs.com**.

## See also

- [Integration with Hugging Face Transformers](transformers.md)
- [Integration with llama.cpp](llamacpp.md)
- [Integration with vLLM](vllm.md)
- [TensorRT-LLM upstream](https://github.com/NVIDIA/TensorRT-LLM)
