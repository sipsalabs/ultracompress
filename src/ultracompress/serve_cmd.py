"""UltraCompress `uc serve` command -- serve a .uc pack via vLLM.

Reconstructs a .uc pack to dense bf16 at startup, then serves it through
vLLM with PagedAttention, CUDA graphs, and continuous batching.

Requires: Linux (or WSL2), vLLM, PyTorch with CUDA.
The compression is throughput-neutral: a decoded .uc pack serves at full
vLLM speed with zero quality or performance penalty.

Usage:
    uc serve ./my_pack --port 8000
    uc serve ./my_pack --gpu 1 --max-model-len 4096
    uc serve SipsaLabs/qwen3-1.7b-base-uc-v3-bpw5 --port 8000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path


def _check_linux() -> None:
    if sys.platform != "linux":
        print(
            "Error: `uc serve` requires Linux (or WSL2 on Windows).\n"
            "vLLM does not support Windows or macOS.\n\n"
            "On Windows with WSL2 installed, run:\n"
            "  wsl -- uc serve ./pack --port 8000",
            file=sys.stderr,
        )
        raise SystemExit(1)


def _check_vllm() -> str:
    try:
        import vllm
        return vllm.__version__
    except ImportError:
        print(
            "Error: vLLM is not installed.\n"
            "Install it with: pip install vllm\n"
            "See https://docs.vllm.ai/en/latest/getting_started/installation.html",
            file=sys.stderr,
        )
        raise SystemExit(1)


def _resolve_pack_dir(pack_path: str) -> Path:
    p = Path(pack_path)
    if p.is_dir() and (p / "manifest.json").exists():
        return p

    if "/" in pack_path and not p.exists():
        print(f"Downloading pack from HuggingFace: {pack_path} ...")
        try:
            from huggingface_hub import snapshot_download
            local = snapshot_download(pack_path)
            return Path(local)
        except Exception as e:
            print(f"Error downloading pack: {e}", file=sys.stderr)
            raise SystemExit(1)

    print(
        f"Error: {pack_path} is not a valid pack directory.\n"
        "Expected a directory containing manifest.json + layer_*.uc files.\n"
        "Use a local path or a HuggingFace repo id (e.g. SipsaLabs/qwen3-1.7b-base-uc-v3-bpw5).",
        file=sys.stderr,
    )
    raise SystemExit(1)


def _read_manifest(pack_dir: Path) -> dict:
    manifest_path = pack_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"Error: {manifest_path} not found.", file=sys.stderr)
        raise SystemExit(1)
    return json.loads(manifest_path.read_text())


def _detect_base_model(pack_dir: Path, manifest: dict) -> str:
    config_path = pack_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        arch = config.get("architectures", [None])[0]
        if arch:
            return str(pack_dir)

    hf_id = manifest.get("base_model_hf_id")
    if hf_id:
        return hf_id

    print(
        "Warning: could not auto-detect base model architecture.\n"
        "Use --base-model to specify the HuggingFace model id.",
        file=sys.stderr,
    )
    return ""


def cmd_serve(args: argparse.Namespace) -> int:
    _check_linux()
    vllm_version = _check_vllm()

    pack_dir = _resolve_pack_dir(args.pack)
    manifest = _read_manifest(pack_dir)

    n_layers = manifest.get("n_layers", 0)
    bpw = manifest.get("bpw", "?")
    print(f"Pack: {pack_dir.name}")
    print(f"  {n_layers} layers @ {bpw} bpw")
    print(f"  vLLM: {vllm_version}")

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"  GPU: CUDA_VISIBLE_DEVICES={args.gpu}")

    base_model = args.base_model or _detect_base_model(pack_dir, manifest)
    if not base_model:
        print("Error: --base-model is required (could not auto-detect).", file=sys.stderr)
        return 1

    cache_dir = pack_dir / ".safetensors_cache"
    cached_st = cache_dir / "model.safetensors"
    cached_config = cache_dir / "config.json"

    if cached_st.exists() and cached_config.exists() and not args.no_cache:
        print(f"\n  Using cached reconstruction: {cache_dir}")
        model_dir = str(cache_dir)
    else:
        print("\n  Reconstructing .uc pack to bf16 safetensors ...")
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        try:
            import importlib
            _mod = importlib.import_module("ultracompress." + "_" + "serve" + "_" + "reconstruct")
            _fn = getattr(_mod, "reconstruct" + "_for_serving")
        except (ImportError, AttributeError):
            print(
                "Error: serve mode requires Sipsa enterprise reconstruction.\n"
                "Contact founder@sipsalabs.com for the licensed reconstruction module.",
                file=sys.stderr,
            )
            return 1

        try:
            t0 = time.time()
            model_dir = _fn(
                pack_dir=pack_dir,
                base_model=base_model,
                output_dir=cache_dir if not args.no_cache else Path(tempfile.mkdtemp()),
            )
            elapsed = time.time() - t0
            print(f"  Reconstruction complete in {elapsed:.1f}s")
        except Exception as e:
            print(
                f"Error: reconstruction failed ({type(e).__name__}).\n"
                "Contact founder@sipsalabs.com for support.",
                file=sys.stderr,
            )
            return 1

    print(f"\n  Starting vLLM server on port {args.port} ...")
    print(f"  Model: {model_dir}")
    print(f"  Max model len: {args.max_model_len}")
    print("  Dtype: bfloat16")
    print()

    from vllm.entrypoints.openai.api_server import run_server

    server_args = argparse.Namespace(
        model=model_dir,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        port=args.port,
        host=args.host,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    try:
        run_server(server_args)
    except Exception as e:
        print(f"vLLM server error: {e}", file=sys.stderr)
        return 1

    return 0


def build_serve_parser(subparsers: argparse._SubParsersAction) -> None:
    s = subparsers.add_parser(
        "serve",
        help="Serve a .uc pack via vLLM (Linux/WSL2 only)",
    )
    s.add_argument(
        "pack",
        help="Path to .uc pack directory or HuggingFace repo id",
    )
    s.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the OpenAI-compatible API server (default: 8000)",
    )
    s.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind (default: 0.0.0.0)",
    )
    s.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU index (sets CUDA_VISIBLE_DEVICES)",
    )
    s.add_argument(
        "--base-model",
        default=None,
        help="HuggingFace model id for architecture config (auto-detected if pack has config.json)",
    )
    s.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum sequence length (default: 4096)",
    )
    s.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="Fraction of GPU memory for model + KV cache (default: 0.85)",
    )
    s.add_argument(
        "--no-cache",
        action="store_true",
        help="Don't cache reconstructed safetensors to disk",
    )
