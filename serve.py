"""
UltraCompress Model Server — Ollama-compatible API.

Serves compressed models (FRR, .ucz, genome) via the same HTTP API
that Ollama uses. Athena or any Ollama client just points to this port.

Usage:
    python serve.py --model frr_demo_model.pt --port 11435

Then in Athena/Ollama config, set base URL to http://localhost:11435

API endpoints (Ollama-compatible):
    POST /api/generate  — Generate text from prompt
    POST /api/chat      — Chat completion
    GET  /api/tags      — List available models
    GET  /               — Health check
"""
import argparse
import json
import time
import sys
import os
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse, JSONResponse
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

# Fallback to basic http.server if no FastAPI
if not HAS_FASTAPI:
    from http.server import HTTPServer, BaseHTTPRequestHandler


# ============================================================
# Model loading
# ============================================================

def load_frr_model(path, device='cuda'):
    """Load a trained FRR model."""
    from ultracompress.moonshot import FractalModel

    # Load teacher weights for embed/head/norm
    wd = torch.load('qwen3_0.6b_cache.pt', weights_only=True)
    embed = wd['model.embed_tokens.weight'].float().to(device)
    norm_w = wd.get('model.norm.weight', torch.ones(1024)).float().to(device)
    lm_head = wd.get('lm_head.weight', embed).float().to(device)

    # Load FRR checkpoint
    ckpt = torch.load(path, weights_only=False)
    cfg = ckpt.get('config', {})

    model = FractalModel(
        hidden_dim=1024,
        n_heads=cfg.get('n_heads', 8),
        n_scales=cfg.get('n_scales', 4),
        iters_per_scale=cfg.get('iters_per_scale', 7),
        vocab_size=151936,
        ff_mult=cfg.get('ff_mult', 2),
        embed_weight=embed,
        lm_head_weight=lm_head,
        norm_weight=norm_w,
    ).to(device)

    model.block.load_state_dict(ckpt['block'])
    model.scale_gamma.data = ckpt['scale_gamma'].to(device)
    model.scale_beta.data = ckpt['scale_beta'].to(device)
    model.iter_scale.data = ckpt['iter_scale'].to(device)
    model.eval()

    params = model.fractal_params()
    print(f"Loaded FRR model: {params:,} params ({params*2/1e6:.1f} MB)")
    return model


def load_model(path, device='cuda'):
    """Load any supported model format."""
    if path.endswith('.ucz'):
        # TODO: load .ucz compressed model
        raise NotImplementedError(".ucz loading for serve not yet implemented")
    elif 'frr' in path.lower() or path.endswith('.pt'):
        return load_frr_model(path, device)
    else:
        raise ValueError(f"Unknown model format: {path}")


# ============================================================
# Text generation
# ============================================================

def generate_tokens(model, input_ids, max_tokens=256, temperature=0.7,
                    top_k=40, top_p=0.9, stop_tokens=None):
    """Autoregressive generation, yields one token at a time."""
    if stop_tokens is None:
        stop_tokens = {151643, 151644, 151645}  # Qwen EOS tokens

    tokens = input_ids.clone()
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(tokens)
            next_logits = logits[0, -1, :] / max(temperature, 0.01)

            # Top-k
            if top_k > 0:
                topk_vals, topk_idx = next_logits.topk(min(top_k, next_logits.shape[0]))
                next_logits = torch.full_like(next_logits, float('-inf'))
                next_logits[topk_idx] = topk_vals

            # Top-p (nucleus)
            if top_p < 1.0:
                sorted_logits, sorted_idx = next_logits.sort(descending=True)
                cumprobs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                mask = cumprobs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[mask] = float('-inf')
                next_logits = torch.zeros_like(next_logits).scatter_(0, sorted_idx, sorted_logits)

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            token_id = next_token.item()

            if token_id in stop_tokens:
                break

            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
            yield token_id


# ============================================================
# FastAPI server (Ollama-compatible)
# ============================================================

def create_app(model, tokenizer, model_name="ultracompress"):
    app = FastAPI()

    @app.get("/")
    async def health():
        return {"status": "ok", "model": model_name}

    @app.get("/api/tags")
    async def list_models():
        return {"models": [{"name": model_name, "size": 0, "format": "frr"}]}

    @app.post("/api/generate")
    async def generate(request: Request):
        body = await request.json()
        prompt = body.get("prompt", "")
        stream = body.get("stream", True)
        max_tokens = body.get("options", {}).get("num_predict", 256)
        temperature = body.get("options", {}).get("temperature", 0.7)
        top_k = body.get("options", {}).get("top_k", 40)
        top_p = body.get("options", {}).get("top_p", 0.9)

        if tokenizer:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(next(model.parameters()).device)
        else:
            input_ids = torch.tensor([[ord(c) for c in prompt]], device=next(model.parameters()).device)

        if stream:
            async def stream_response():
                full_response = ""
                for token_id in generate_tokens(model, input_ids, max_tokens, temperature, top_k, top_p):
                    if tokenizer:
                        text = tokenizer.decode([token_id])
                    else:
                        text = chr(token_id) if token_id < 128 else f"[{token_id}]"
                    full_response += text
                    chunk = json.dumps({"model": model_name, "response": text, "done": False}) + "\n"
                    yield chunk
                # Final chunk
                yield json.dumps({"model": model_name, "response": "", "done": True,
                                 "total_duration": 0, "eval_count": len(full_response)}) + "\n"
            return StreamingResponse(stream_response(), media_type="application/x-ndjson")
        else:
            full_response = ""
            for token_id in generate_tokens(model, input_ids, max_tokens, temperature, top_k, top_p):
                if tokenizer:
                    full_response += tokenizer.decode([token_id])
                else:
                    full_response += chr(token_id) if token_id < 128 else f"[{token_id}]"
            return {"model": model_name, "response": full_response, "done": True}

    @app.post("/api/chat")
    async def chat(request: Request):
        body = await request.json()
        messages = body.get("messages", [])
        prompt = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)
        body["prompt"] = prompt
        # Reuse generate
        request._body = json.dumps(body).encode()
        return await generate(request)

    return app


# ============================================================
# Fallback HTTP server (no FastAPI)
# ============================================================

class SimpleHandler(BaseHTTPRequestHandler):
    model = None
    tokenizer = None
    model_name = "ultracompress"

    def do_GET(self):
        if self.path == "/" or self.path == "/api/tags":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "model": self.model_name}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(content_length)) if content_length else {}
        prompt = body.get("prompt", "")

        if self.tokenizer:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(
                next(self.model.parameters()).device)
        else:
            input_ids = torch.tensor([[ord(c) for c in prompt]],
                                    device=next(self.model.parameters()).device)

        response = ""
        for token_id in generate_tokens(self.model, input_ids, max_tokens=256):
            if self.tokenizer:
                response += self.tokenizer.decode([token_id])
            else:
                response += chr(token_id) if token_id < 128 else f"[{token_id}]"

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({
            "model": self.model_name, "response": response, "done": True
        }).encode())

    def log_message(self, format, *args):
        print(f"[{time.strftime('%H:%M:%S')}] {args[0]}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UltraCompress Model Server")
    parser.add_argument("--model", required=True, help="Path to model (.pt or .ucz)")
    parser.add_argument("--port", type=int, default=11435, help="Port (default: 11435)")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = load_model(args.model, args.device)

    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
        print(f"Tokenizer: {tokenizer.__class__.__name__}")
    except:
        print("No tokenizer, using raw encoding")

    print(f"\nStarting server on port {args.port}")
    print(f"Ollama-compatible API: http://localhost:{args.port}/api/generate")
    print(f"Point Athena to: http://localhost:{args.port}")

    if HAS_FASTAPI:
        app = create_app(model, tokenizer)
        uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
    else:
        SimpleHandler.model = model
        SimpleHandler.tokenizer = tokenizer
        server = HTTPServer(("0.0.0.0", args.port), SimpleHandler)
        print(f"Serving on http://0.0.0.0:{args.port} (basic HTTP, install fastapi+uvicorn for streaming)")
        server.serve_forever()
