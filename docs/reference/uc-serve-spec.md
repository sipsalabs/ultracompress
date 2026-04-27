# `uc serve` design spec (v0.2 — target Q3 2026)

**Status**: 🔵 PLANNED for v0.2.

This page describes the planned `uc serve` command — an OpenAI-compatible inference server running locally on a compressed UltraCompress artifact. **It is NOT in v0.1**; this page is the design spec we'll implement against.

This feature is planned for v0.2 (Q3 2026). To register interest, email `founder@sipsalabs.com`.

---

## Goals

1. **Single-command serving** — one CLI command brings up an HTTP server speaking the OpenAI API.
2. **Standalone** — no separate inference framework required (vLLM / TensorRT-LLM are integration paths, not requirements).
3. **Customer-deployable** — runs in customer's environment, not Sipsa Labs' cloud.
4. **Memory-efficient** — the whole point of UltraCompress; serve a 7B model on commodity 8 GB GPU.

## Synopsis (planned)

```
uc serve <path> [--port INT] [--host STR] [--max-tokens INT]
                [--device STR] [--max-batch-size INT]
                [--api-key STR] [--api-key-file PATH]
                [--cors-origin STR] [--max-concurrency INT]
```

## Options

| Option | Default | Description |
|---|---|---|
| `<path>` | required | Path to a directory or `ultracompress.json` produced by `uc pull` |
| `--port` | `8080` | Port to bind |
| `--host` | `127.0.0.1` | Bind address; `0.0.0.0` for all interfaces |
| `--max-tokens` | `2048` | Maximum new tokens per request |
| `--device` | `cuda:0` | Device for inference |
| `--max-batch-size` | `8` | Max concurrent prompts in flight |
| `--api-key STR` | (none — open) | Require this API key in `Authorization: Bearer <key>` header |
| `--api-key-file PATH` | (none) | Read API key from file (more secure than CLI flag) |
| `--cors-origin STR` | `*` | CORS Access-Control-Allow-Origin |
| `--max-concurrency INT` | `100` | Max parallel HTTP connections |

## Endpoints

OpenAI-compatible:

- `POST /v1/chat/completions` — chat completion (streaming + non-streaming)
- `POST /v1/completions` — legacy text completion (streaming + non-streaming)
- `GET /v1/models` — list available models (single entry: this artifact)
- `GET /health` — liveness probe
- `GET /metrics` — Prometheus-format metrics (request count, latency, queue depth, GPU memory)

## Example

```bash
uc serve ./models/sipsalabs_<model-id>

# In another terminal:
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sipsalabs/<model-id>",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

## Streaming

Streaming response uses Server-Sent Events (SSE) per the OpenAI API spec:

```bash
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sipsalabs/<model-id>",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'

# Response (SSE format):
data: {"id":"...","choices":[{"delta":{"content":"Hello"}}]}
data: {"id":"...","choices":[{"delta":{"content":"!"}}]}
data: [DONE]
```

## Memory budget

Approximate GPU memory for `uc serve` at the v0.2 deployment:

| Model | Loaded weights | Activations (batch=1) | Total (batch=1) | Total (batch=8) |
|---|---|---|---|---|
| Qwen3-1.7B at 2.798 bpw | ~0.6 GB | ~0.5 GB | ~1.1 GB | ~3.5 GB |
| Llama-2-7B at 2.798 bpw | ~2.5 GB | ~1.5 GB | ~4.0 GB | ~12 GB |
| Llama-2-13B at 2.798 bpw | ~4.5 GB | ~2.5 GB | ~7.0 GB | ~22 GB |

(These are rough estimates; actual numbers depend on the runtime kernel path, which we'll publish in v0.2 release notes.)

## Deployment patterns

### Single-server local serving

```bash
uc serve ./models/sipsalabs_<model-id>
```

Use case: development, testing, single-customer evaluation, demo.

### Production single-host with API key

```bash
uc serve ./models/sipsalabs_<model-id> \
    --host 0.0.0.0 \
    --port 8080 \
    --api-key-file /run/secrets/uc-api-key \
    --max-concurrency 200
```

Use case: production single-server deployment behind a reverse proxy (Cloudflare, nginx, etc.).

### Multi-host orchestrated serving

For scale-out: `uc serve` is a single-server runtime. For multi-host, run multiple `uc serve` instances behind a load balancer. Each instance is independent.

### Kubernetes / Docker

Sample Dockerfile (will be published in v0.2 release docs):

```dockerfile
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip install ultracompress[serve]
COPY models/ /app/models/
WORKDIR /app
EXPOSE 8080
CMD ["uc", "serve", "/app/models/sipsalabs_<model-id>", "--host", "0.0.0.0"]
```

## Security

- API key authentication via `--api-key` (flag) or `--api-key-file` (file; preferred for production)
- TLS termination is **not** built in; deploy behind a reverse proxy
- CORS is permissive by default; restrict via `--cors-origin` if exposing to web clients
- No request logging by default (privacy); enable via `UC_VERBOSE=1`

## Limitations

- **Not a full vLLM replacement.** vLLM has continuous batching, paged attention, and speculative decoding that we don't replicate in v0.2. For production-grade throughput at scale, integrate UltraCompress artifacts into vLLM (target same v0.2 release).
- **Single-model-per-server.** No multi-model multiplexing in v0.2. Deploy multiple servers if you need multiple models.
- **No fine-tuning surface.** Inference only; fine-tuning is via HF Transformers post-load.

## Migration path from v0.1 reference loader

Code that today uses the v0.1 reference loader will work in v0.2 unchanged. `uc serve` is an addition, not a replacement.

## Roadmap

| Feature | Target |
|---|---|
| Basic `uc serve` with non-streaming endpoints | v0.2 (Q3 2026) |
| Streaming endpoints (SSE) | v0.2 |
| Multi-prompt batching | v0.2 |
| Continuous batching (vLLM-style) | v0.2.1 or v0.3 |
| Multi-LoRA serving | v0.3 |
| Rate limiting / quotas | v0.3 |
| Multi-tenant authentication | v0.3 |
| Tracing + observability (OpenTelemetry) | v0.3 |
| WebSocket protocol | post-v0.3 |
| Kubernetes operator | post-v0.3 |

---

## Open questions

1. **Tokenization at the edge**: should `uc serve` accept raw bytes (let the model tokenize) or pre-tokenized inputs? OpenAI API standard is raw text; we'll match that.

2. **Function-calling / tool-use API**: the OpenAI API has `tools` for function-calling. Most chat models support this; we'll support it on a model-by-model basis based on what the underlying model can do.

3. **Multi-modal support**: vision-language models (LLaVA-style) need a different API surface. Defer to post-v0.3.

4. **gRPC alternative**: TensorRT-LLM offers gRPC. We'll support HTTP/JSON in v0.2; gRPC if customer demand justifies.

These are open. We'd love customer feedback — file an issue at [github.com/mounnar/ultracompress](https://github.com/mounnar/ultracompress) or email `founder@sipsalabs.com`.

---

*Last updated: 2026-04-25 evening. This is a design spec for v0.2; revise as v0.2 implementation progresses.*
