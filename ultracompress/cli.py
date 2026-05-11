"""ultracompress.cli — `uc` command-line entry point.

Public subcommands:
  uc bench  Measure Top-1 retention, decode latency, and storage footprint
            for a compressed checkpoint against an fp16/fp32 teacher.

Honest measurement only. Numbers come from real generations on the user's
GPU, not estimates. JSON output is reproducible from command line.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_PROMPTS: list[str] = [
    "The capital of France is",
    "Photosynthesis is the process by which",
    "In Python, a list comprehension is",
    "The first law of thermodynamics states that",
    "A binary search tree is a data structure where",
    "The mitochondria is the powerhouse of the",
    "When water freezes, its volume",
    "The largest planet in our solar system is",
]

REPO_ROOT = Path(__file__).resolve().parents[1]
OVERLAY_DIR = REPO_ROOT / "scripts" / "overlay"


def _load_prompts(prompts_file: str | None) -> list[str]:
    if not prompts_file:
        return DEFAULT_PROMPTS
    p = Path(prompts_file)
    if not p.exists():
        sys.exit(f"[FAILED] prompts file not found: {prompts_file}")
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        sys.exit(f"[FAILED] prompts file empty: {prompts_file}")
    return lines


def _bytes_human(n: int) -> str:
    val = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if val < 1024.0:
            return f"{val:.2f}{unit}"
        val /= 1024.0
    return f"{val:.2f}TB"


def _generate(model: Any, tokenizer: Any, prompts: list[str], device: str,
              max_new_tokens: int) -> tuple[list[list[int]], float]:
    """Greedy generate; return per-prompt token-id lists plus mean ms/token over 5 runs."""
    import torch
    pad_id = tokenizer.eos_token_id or 0
    all_ids: list[list[int]] = []
    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=pad_id)
        all_ids.append(out[0].tolist()[ids.shape[1]:])

    ids0 = tokenizer(prompts[0], return_tensors="pt").input_ids.to(device)
    use_cuda = device.startswith("cuda")
    for _ in range(2):
        with torch.no_grad():
            model.generate(ids0, max_new_tokens=max_new_tokens, do_sample=False,
                           pad_token_id=pad_id)
    if use_cuda:
        torch.cuda.synchronize(device)
    timings_ms: list[float] = []
    for _ in range(5):
        if use_cuda:
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(ids0, max_new_tokens=max_new_tokens, do_sample=False,
                           pad_token_id=pad_id)
        if use_cuda:
            torch.cuda.synchronize(device)
        timings_ms.append((time.perf_counter() - t0) * 1000.0 / max_new_tokens)
    return all_ids, sum(timings_ms) / len(timings_ms)


def _top1_retention(teacher_ids: list[list[int]], student_ids: list[list[int]]) -> float:
    matches = total = 0
    for t, s in zip(teacher_ids, student_ids):
        n = min(len(t), len(s))
        matches += sum(1 for i in range(n) if t[i] == s[i])
        total += n
    return 100.0 * matches / max(1, total)


def _bench(args: argparse.Namespace) -> int:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print(f"[FAILED] missing dependency: {e}")
        return 2

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    prompts = _load_prompts(args.prompts_file)

    print(f"model:        {args.model}")
    print(f"mode:         {args.mode}")
    print(f"target_bpw:   {args.target_bpw}")
    print(f"device:       {device} dtype={dtype}")
    print(f"prompts:      {len(prompts)}")
    print(f"max_new_tok:  {args.max_new_tokens}")

    print("\n[1/3] loading teacher (uncompressed)...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True).to(device).train(False)

    teacher_pt = REPO_ROOT / f".uc_bench_teacher_{Path(args.model).name}.pt"
    torch.save(teacher.state_dict(), teacher_pt)
    fp_size = teacher_pt.stat().st_size

    print("\n[2/3] generating with teacher...")
    t_ids, t_lat = _generate(teacher, tokenizer, prompts, device, args.max_new_tokens)

    del teacher
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    print("\n[3/3] applying compression pipeline + measuring student...")
    try:
        uc = importlib.import_module("ultracompress")
        if not (hasattr(uc, "compress") and hasattr(uc, "save")):
            raise AttributeError("uc.compress / uc.save not yet wired")
        student, report = uc.compress(args.model, mode=args.mode,
                                      target_bpw=args.target_bpw, device=device)
        comp_path = REPO_ROOT / f".uc_bench_student_{Path(args.model).name}.uc"
        uc.save(student, comp_path)
        comp_size = Path(comp_path).stat().st_size
        eff_bpw = float(report.get("effective_bpw", args.target_bpw))
    except (ImportError, AttributeError) as e:
        print(f"  (note: public API unavailable [{e.__class__.__name__}]; "
              f"using Track A v17 packed weights directly)")
        student, comp_size, eff_bpw = _fallback_student(args, device, dtype)

    s_ids, s_lat = _generate(student, tokenizer, prompts, device, args.max_new_tokens)
    top1 = _top1_retention(t_ids, s_ids)

    failed = top1 < 50.0
    summary: dict[str, Any] = {
        "model": args.model,
        "mode": args.mode,
        "target_bpw": args.target_bpw,
        "effective_bpw": eff_bpw,
        "device": device,
        "prompts_n": len(prompts),
        "max_new_tokens": args.max_new_tokens,
        "top1_retention_pct": round(top1, 4),
        "decode_latency_ms_per_token": {
            "teacher": round(t_lat, 4),
            "student": round(s_lat, 4),
            "speedup_x": round(t_lat / max(s_lat, 1e-6), 4),
        },
        "storage_bytes": {
            "uncompressed_fp16_pt": fp_size,
            "compressed_uc": comp_size,
            "ratio_x": round(fp_size / max(comp_size, 1), 4),
        },
        "status": "FAILED" if failed else "OK",
    }

    out_path = args.output or f"bench_{Path(args.model).name}_{args.mode}.json"
    Path(out_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 64)
    if failed:
        print("[FAILED] top-1 retention below 50% threshold")
    print(f"top-1 retention:        {top1:.2f}%")
    print(f"effective bpw:          {eff_bpw:.3f}")
    print(f"teacher latency:        {t_lat:.3f} ms/tok")
    print(f"student latency:        {s_lat:.3f} ms/tok")
    print(f"storage uncompressed:   {_bytes_human(fp_size)}")
    print(f"storage compressed:     {_bytes_human(comp_size)}")
    print(f"compression ratio:      {fp_size / max(comp_size, 1):.2f}x")
    print(f"json written:           {out_path}")
    print("=" * 64)

    teacher_pt.unlink(missing_ok=True)
    return 1 if failed else 0


def _fallback_student(args: argparse.Namespace, device: str, dtype: Any) -> tuple[Any, int, float]:
    """Load Track A v17 packed weights directly when public API is not yet wired."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    v17_path = REPO_ROOT / "v17_fit_qwen3_1.7b.pt"
    if not v17_path.exists():
        sys.exit(f"[FAILED] no fallback artifact at {v17_path}; public API required")

    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    student = AutoModelForCausalLM.from_config(cfg, torch_dtype=dtype, trust_remote_code=True)
    teacher_cache = REPO_ROOT / "qwen3_1.7b_cache.pt"
    sd = None
    if teacher_cache.exists():
        sd = torch.load(teacher_cache, map_location="cpu", weights_only=False)
        if "state_dict" in sd:
            sd = sd["state_dict"]
        student.load_state_dict(sd, strict=False)

    if str(OVERLAY_DIR) not in sys.path:
        sys.path.insert(0, str(OVERLAY_DIR))
    overlay_mod = importlib.import_module("eval_v17_ppl")
    substitute_v17 = getattr(overlay_mod, "substitute_v17")
    v17 = torch.load(v17_path, map_location="cpu", weights_only=False)
    substitute_v17(student, sd if sd is not None else student.state_dict(),
                   v17, device, D=8)
    student = student.to(device).train(False)
    eff_bpw = float(v17.get("global_bpw", args.target_bpw))
    return student, v17_path.stat().st_size, eff_bpw


def _load_calibration_tokens(path: str | None, tokenizer: Any, seed: int) -> Any:
    """Load calibration tokens from parquet/jsonl, or return None for random init."""
    import torch
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        sys.exit(f"[FAILED] calibration data not found: {path}")
    suffix = p.suffix.lower()
    texts: list[str] = []
    if suffix in (".jsonl", ".ndjson"):
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            t = obj.get("text") or obj.get("content") or obj.get("prompt")
            if isinstance(t, str):
                texts.append(t)
    elif suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError:
            sys.exit("[FAILED] pyarrow required for parquet calibration data")
        table = pq.read_table(str(p))
        col = "text" if "text" in table.column_names else table.column_names[0]
        texts = [str(v) for v in table.column(col).to_pylist() if v]
    else:
        sys.exit(f"[FAILED] calibration data must be .parquet or .jsonl, got {suffix!r}")
    if not texts:
        sys.exit(f"[FAILED] calibration data has no text rows: {path}")
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(texts), generator=g).tolist()
    blob = "\n\n".join(texts[i] for i in perm)
    ids = tokenizer(blob, return_tensors="pt").input_ids[0]
    return ids.long()


def _eval_t1_ppl(student: Any, teacher: Any, tokenizer: Any, prompts: list[str],
                 device: str, max_new: int) -> dict[str, float]:
    """Run T1 retention + PPL ratio + decode latency on student vs teacher."""
    import torch
    t_ids, t_lat = _generate(teacher, tokenizer, prompts, device, max_new)
    s_ids, s_lat = _generate(student, tokenizer, prompts, device, max_new)
    t1 = _top1_retention(t_ids, s_ids)
    ppls: list[float] = []
    for prompt in prompts[: min(4, len(prompts))]:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        if ids.shape[1] < 2:
            continue
        with torch.no_grad():
            t_logits = teacher(input_ids=ids).logits.float()
            s_logits = student(input_ids=ids).logits.float()
        tgt = ids[0, 1:]
        t_lp = torch.nn.functional.log_softmax(t_logits[0, :-1], dim=-1)
        s_lp = torch.nn.functional.log_softmax(s_logits[0, :-1], dim=-1)
        t_nll = -t_lp.gather(-1, tgt.unsqueeze(-1)).mean().item()
        s_nll = -s_lp.gather(-1, tgt.unsqueeze(-1)).mean().item()
        ppls.append(math_exp(s_nll) / max(math_exp(t_nll), 1e-9))
    ppl_ratio = sum(ppls) / max(1, len(ppls)) if ppls else float("nan")
    return {"t1_pct": round(t1, 4), "ppl_ratio": round(ppl_ratio, 6),
            "decode_ms_teacher": round(t_lat, 4), "decode_ms_student": round(s_lat, 4)}


def math_exp(x: float) -> float:
    import math as _m
    return _m.exp(min(x, 50.0))


def _fit(args: argparse.Namespace) -> int:
    """Fit a `.uc` artifact from a local HF model directory."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print(f"[FAILED] missing dependency: {e}")
        return 2
    import ultracompress as uc

    model_dir = Path(args.model).expanduser().resolve()
    if not model_dir.exists():
        print(f"[FAILED] model path does not exist: {model_dir}")
        return 2
    if not (model_dir / "config.json").exists():
        print(f"[FAILED] no config.json in {model_dir}; expected HF-format directory")
        return 2

    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = out_path.with_suffix(out_path.suffix + ".fit.json")

    steps = 300 if args.quick else int(args.steps)
    if args.device == "dual":
        if not (torch.cuda.is_available() and torch.cuda.device_count() >= 2):
            print("[FAILED] --device dual requires 2+ CUDA devices")
            return 2
        device = "dual"
    elif args.device == "cpu":
        device = "cpu"
    else:
        device = args.device if torch.cuda.is_available() else "cpu"

    torch.manual_seed(args.seed)
    use_cuda = device == "dual" or device.startswith("cuda")
    dtype = torch.float16 if use_cuda else torch.float32
    print(f"[uc fit] model:  {model_dir}")
    print(f"[uc fit] output: {out_path}")
    block_size = int(getattr(args, "block_size", 0) or 0)
    eff_bpw = float(args.bpw) + (16.0 / block_size if block_size > 0 else 0.0)
    print(f"[uc fit] bpw={args.bpw} block_size={block_size} eff_bpw={eff_bpw:.3f}  "
          f"rank={args.rank} steps={steps} device={device} seed={args.seed}")
    if device == "dual":
        print("[uc fit] dual-GPU dispatch via accelerate (max 28GiB/device). "
              "Tested at <=14B class. For 32B+, use scripts/overlay/scaling_curve_runner.py.")

    print("[uc fit] loading model + tokenizer (local, trust_remote_code)")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    if device == "dual":
        # Load directly with accelerate device_map; never tries to pin on cuda:0.
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), torch_dtype=dtype, trust_remote_code=True,
            device_map="auto",
            max_memory={0: "28GiB", 1: "28GiB", "cpu": "80GiB"},
        ).train(False)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), torch_dtype=dtype, trust_remote_code=True
        ).to(device).train(False)
    arch = type(model).__name__
    if "CausalLM" not in arch:
        print(f"[uc fit] WARN: architecture {arch!r} is not a transformer causal-LM; "
              "the correction layer was validated on transformer activations.")

    print("[uc fit] loading calibration tokens" if args.calibration_data
          else "[uc fit] no calibration data; correction stays at init (alpha=0)")
    tokens = _load_calibration_tokens(args.calibration_data, tokenizer, args.seed)

    t0 = time.time()
    compressed = uc.compress(
        model, mode="scalar_v18c", target_bpw=float(args.bpw),
        correction_rank=int(args.rank), train_steps=steps,
        block_size=block_size,
        n_chunks=int(getattr(args, "n_chunks", 1) or 1),
        u_weight_dtype=str(getattr(args, "u_weight_dtype", "fp32") or "fp32"),
        tokens=tokens, seed=args.seed, device=device,
    )
    fit_seconds = time.time() - t0
    uc.save(compressed, out_path)

    record: dict[str, Any] = {
        "model_path": str(model_dir),
        "architecture": arch,
        "output_path": str(out_path),
        "bpw_target": float(args.bpw),
        "correction_rank": int(args.rank),
        "steps": int(steps),
        "device": device,
        "seed": int(args.seed),
        "fit_seconds": round(fit_seconds, 2),
        "report": compressed.report.to_dict(),
        "artifact_bytes": out_path.stat().st_size,
    }

    if args.eval_prompts:
        ep = Path(args.eval_prompts).expanduser().resolve()
        if not ep.exists():
            print(f"[uc fit] WARN: eval prompts not found: {ep}")
        else:
            prompts = json.loads(ep.read_text(encoding="utf-8"))
            if not isinstance(prompts, list) or not prompts:
                print("[uc fit] WARN: eval-prompts JSON must be a non-empty list of strings")
            else:
                max_new = 16 if args.quick else 32
                # For dual dispatch, route eval inputs to whichever GPU holds
                # the embedding shard (typically cuda:0 under accelerate).
                eval_dev = device
                if device == "dual":
                    emb = getattr(getattr(model, "model", None), "embed_tokens", None)
                    if emb is not None and hasattr(emb, "weight"):
                        eval_dev = str(emb.weight.device)
                print(f"[uc fit] evaluating on {len(prompts)} prompts "
                      f"(max_new={max_new}, eval_device={eval_dev})")
                metrics = _eval_t1_ppl(compressed.model, model, tokenizer,
                                       [str(p) for p in prompts], eval_dev, max_new)
                record["eval"] = metrics
                print(f"[uc fit] eval: T1={metrics['t1_pct']:.2f}%  "
                      f"PPL_ratio={metrics['ppl_ratio']:.4f}  "
                      f"latency student={metrics['decode_ms_student']:.2f}ms/tok")

    log_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"[uc fit] done in {fit_seconds:.1f}s; artifact={out_path}; log={log_path}")
    return 0


def _load(args: argparse.Namespace) -> int:
    """Download a streaming-compressed reference checkpoint from the HF Hub.

    These are the per-layer artifact sets published under
    huggingface.co/SipsaLabs (e.g. ``SipsaLabs/qwen2.5-72b-streaming-bpw5``).
    Downloads ``layer_*.pt`` + ``manifest.json`` + ``eval_smoke.json`` +
    ``README.md`` into a local cache and prints the next-step inference
    command.

    Scaffold weights (embed_tokens, final_norm, lm_head) are not in the
    artifact - they load from the original base model at inference time.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[FAILED] huggingface_hub not installed; pip install huggingface_hub")
        return 2

    repo_id = args.repo
    if "/" not in repo_id:
        print(f"[FAILED] repo id must be 'org/name', got {repo_id!r}")
        return 2

    target = Path(args.output).expanduser().resolve() if args.output else None
    print(f"uc load: downloading {repo_id} ...")
    t0 = time.time()
    local_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=["layer_*.pt", "manifest.json", "eval_smoke.json", "README.md"],
        local_dir=str(target) if target else None,
    )
    elapsed = time.time() - t0

    # Quick inventory + size
    layer_files = sorted(Path(local_dir).glob("layer_*.pt"))
    total_bytes = sum(f.stat().st_size for f in layer_files)
    manifest_path = Path(local_dir) / "manifest.json"
    base_model = "<unknown>"
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as fh:
                manifest = json.load(fh)
            base_model = manifest.get("base_model_hf_id") or manifest.get("hf_id") or base_model
        except Exception:
            pass

    print(f"  {len(layer_files)} layer artifacts, {_bytes_human(total_bytes)} on disk")
    print(f"  base model: {base_model}")
    print(f"  cached at:  {local_dir}")
    print(f"  elapsed:    {elapsed:.1f}s")
    print()
    print("Next steps:")
    print("  - To run a smoke PPL eval against the base model:")
    print(f"    python scripts/overlay/eval_compressed_only.py \\")
    print(f"        --model {Path(base_model).name.lower()} \\")
    print(f"        --compressed_dir {local_dir} \\")
    print(f"        --device cuda:0 --n_eval 50")
    print("  - To inspect the model card:")
    print(f"    cat {Path(local_dir) / 'README.md'}")
    return 0


def _serve(args: argparse.Namespace) -> int:
    """Launch the FastAPI inference server via uvicorn.

    Sets `UC_MODEL_PATH` from `--model-path` so the lifespan validator picks
    it up. Refuses to start if the path is missing/invalid.
    """
    model_path = Path(args.model_path).expanduser().resolve(strict=False)
    if not model_path.exists():
        print(f"[FAILED] model path does not exist: {model_path}")
        return 2
    if model_path.suffix not in (".uc", ".ucz"):
        print(f"[FAILED] model path must end in .uc or .ucz, got {model_path.suffix!r}")
        return 2
    os.environ["UC_MODEL_PATH"] = str(model_path)

    try:
        import uvicorn
    except ImportError:
        print("[FAILED] uvicorn not installed; pip install uvicorn fastapi prometheus_client")
        return 2

    print(f"uc serve: model={model_path}  bind={args.host}:{args.port}")
    uvicorn.run(
        "ultracompress.server.main:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        reload=False,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="uc", description="UltraCompress CLI")
    sub = p.add_subparsers(dest="command", required=True)

    # uc bench <packed-dir> - sales-grade inference throughput benchmark
    b = sub.add_parser(
        "bench",
        help="Benchmark inference throughput on a UC v3 packed model",
    )
    b.add_argument("packed_dir", help="Path to a UC v3 packed directory (layer_*.uc + manifest.json)")
    b.add_argument("--n-prompts", type=int, default=50, dest="n_prompts",
                   help="Number of prompts to evaluate (default 50)")
    b.add_argument("--seq-len", type=int, default=1024, dest="seq_len",
                   help="Approximate prompt length in tokens (default 1024)")
    b.add_argument("--n-new-tokens", type=int, default=256, dest="n_new_tokens",
                   help="Tokens generated per prompt with greedy decoding (default 256)")
    b.add_argument("--device", default="cuda:0",
                   help="Torch device string (default cuda:0)")
    b.add_argument("--baseline", action="store_true",
                   help="Also load bf16 baseline model and compute speedup ratios")
    b.add_argument("--base-model", default=None, dest="base_model",
                   help="Override base HF model id (default: read from packed dir manifest/README)")
    b.add_argument("--out-json", default=None, dest="out_json",
                   help="Output JSON path (default: ./bench_<timestamp>.json)")

    # uc bench-compress - legacy compression-vs-teacher benchmark (kept for back-compat)
    bc = sub.add_parser(
        "bench-compress",
        help="(legacy) Benchmark compression on a HF model vs bf16 teacher",
    )
    bc.add_argument("--model", required=True, help="HuggingFace repo id (e.g. Qwen/Qwen3-1.7B)")
    bc.add_argument("--mode", choices=["track_a_full_stack", "track_a_v17_only"],
                    default="track_a_full_stack")
    bc.add_argument("--target-bpw", type=float, required=True, dest="target_bpw")
    bc.add_argument("--prompts-file", default=None, dest="prompts_file",
                    help="newline-separated prompt file (default: built-in 8-prompt set)")
    bc.add_argument("--output", default=None, help="JSON output path")
    bc.add_argument("--gpu", type=int, default=0, help="CUDA device index")
    bc.add_argument("--max-new-tokens", type=int, default=64, dest="max_new_tokens")

    f = sub.add_parser("fit", help="Fit a .uc artifact from a local HF model directory")
    f.add_argument("--model", required=True, help="Local path to HF model directory (config.json + .safetensors)")
    f.add_argument("--output", required=True, help="Output path for compressed .uc artifact")
    f.add_argument("--bpw", type=float, default=6.0, help="Scalar quantization bits per weight (default 6.0)")
    f.add_argument("--rank", type=int, default=32, help="Correction layer rank (default 32)")
    f.add_argument("--block-size", type=int, default=0, dest="block_size",
                   help="Per-block scalar quant group size. "
                        "0 = per-row absmax (legacy). Non-zero unlocks the "
                        "production tier. Adds 16/block_size bpw overhead.")
    f.add_argument("--n-chunks", type=int, default=1, dest="n_chunks",
                   help="Memory-aware mode: split correction matmul into N chunks "
                        "along output rows. 1 = legacy (parent class). 4-8 unlocks "
                        "14B+ training without OOM. Bit-exact with parent.")
    f.add_argument("--u-weight-dtype", type=str, default="fp32",
                   choices=["fp32", "bf16", "fp16"], dest="u_weight_dtype",
                   help="Correction-overlay U-projection weight dtype. fp32 (default) "
                        "matches production. bf16/fp16 = halve U.weight + skip the "
                        "fp32 inner cast. Use with --n-chunks for max memory savings.")
    f.add_argument("--steps", type=int, default=1500, help="KL distillation steps (default 1500)")
    f.add_argument("--calibration-data", default=None, dest="calibration_data",
                   help="Path to calibration tokens (.parquet or .jsonl with 'text' column)")
    f.add_argument("--eval-prompts", default=None, dest="eval_prompts",
                   help="JSON list of prompt strings; runs T1/PPL/latency eval after fit")
    f.add_argument("--device", default="cuda:0", choices=["cuda:0", "cuda:1", "dual", "cpu"],
                   help="Compute device (default cuda:0)")
    f.add_argument("--quick", action="store_true", help="Reduced steps (300) and short eval")
    f.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")

    l = sub.add_parser("load", help="Download a streaming-compressed reference model from HF Hub")
    l.add_argument("repo", help="HuggingFace repo id (e.g. SipsaLabs/qwen3-8b-streaming-bpw5)")
    l.add_argument("--output", default=None,
                   help="Local target directory (default: HF cache)")

    s = sub.add_parser("serve", help="Launch the FastAPI inference server")
    s.add_argument("--model-path", required=True, dest="model_path",
                   help="Absolute path to a compressed .uc or .ucz artifact")
    s.add_argument("--host", default="127.0.0.1", help="Bind host (default 127.0.0.1)")
    s.add_argument("--port", type=int, default=8080, help="Bind port (default 8080)")
    s.add_argument("--log-level", default="info", dest="log_level",
                   choices=["critical", "error", "warning", "info", "debug", "trace"])

    # uc pack <e2e_dir> <out_dir> — convert e2e dense bf16 layers to packed 5-bit .uc binaries
    pk = sub.add_parser("pack", help="Pack dense e2e layer artifacts to 5-bit .uc binaries (v0.2)")
    pk.add_argument("src", help="Source _e2e_* directory containing layer_*.pt files")
    pk.add_argument("dst", help="Output .uc directory")
    pk.add_argument("--bpw", type=int, default=5,
                    help="Bits per weight for scalar quantization (default 5)")
    pk.add_argument("--block-size", type=int, default=64, dest="block_size",
                    help="Per-block scale block size (default 64)")
    pk.add_argument("--include-aux", action="store_true", default=True, dest="include_aux",
                    help="Pack model-level non-Linear weights (embed_tokens, "
                         "model.norm, lm_head) into a single self-contained "
                         "aux_weights.uc — pack_format_version=3.5 (default)")
    pk.add_argument("--no-aux", action="store_false", dest="include_aux",
                    help="Skip aux_weights.uc and emit a v3.0 pack (smaller, "
                         "but customer must download base safetensors separately)")
    pk.add_argument("--base-model", default=None, dest="base_model",
                    help="HF model id used to source aux weights (e.g. Qwen/Qwen3-1.7B). "
                         "Defaults to the source dir's manifest.json base_model_hf_id field.")
    pk.add_argument("--v3", "--legacy-v3", action="store_true", dest="legacy_v3",
                    help="(legacy) Use the lossy v3 pack_layer path (reverse-derived codec). "
                         "Only useful when source layer.pt has no persisted codec state. "
                         "Default is the lossless v3 path via pack_v3.")

    # uc pack-aux <packed_dir> — retrofit an existing v3 pack with aux_weights.uc
    pka = sub.add_parser(
        "pack-aux",
        help="Retrofit an existing v3 packed dir with self-contained aux_weights.uc "
             "(no need to re-pack the layer files)",
    )
    pka.add_argument("packed_dir", help="Path to an existing v3 packed .uc directory")
    pka.add_argument("--base-model", default=None, dest="base_model",
                     help="HF model id used to source aux weights "
                          "(e.g. Qwen/Qwen3-1.7B). Defaults to manifest.json's "
                          "base_model_hf_id field if present.")

    # uc inspect <uc_dir> — read a packed dir manifest + parse one layer
    insp = sub.add_parser("inspect", help="Inspect a packed .uc directory")
    insp.add_argument("packed_dir", help="Path to packed .uc directory")
    insp.add_argument("--layer", type=int, default=None,
                      help="Print details of a specific layer index")

    # uc verify <uc_dir> — confirm v3 lossless integrity (for regulated industries)
    ver = sub.add_parser("verify", help="Verify v3 lossless pack integrity (audit-trail check)")
    ver.add_argument("packed_dir", help="Path to packed .uc directory")
    ver.add_argument("--compute-hashes", action="store_true",
                     help="Compute SHA256 of every layer file (slow, ground-truth)")
    ver.add_argument("--skip-hash", action="store_true",
                     help="Skip SHA256 integrity check entirely (fast)")

    # uc verify-org <hf_org> — auto-iterate every -uc-v3-bpw5 repo on an HF org
    # and run uc verify on each. Produces a JSON report.
    vorg = sub.add_parser(
        "verify-org",
        help="Iterate an HF org and verify every -uc-v3-bpw5 repo end-to-end",
    )
    vorg.add_argument("org", help="HuggingFace org name (e.g. SipsaLabs)")
    vorg.add_argument("--out", default="VERIFY_ALL_REPORT.json",
                      help="Output JSON report path (default: ./VERIFY_ALL_REPORT.json)")
    vorg.add_argument("--local-base", default=None,
                      help="Local cache dir for downloads (default: tempdir)")
    vorg.add_argument("--repo-suffix", default="-uc-v3-bpw5",
                      help="Only check repos ending with this suffix (default: -uc-v3-bpw5)")

    # uc status — print a one-line summary of the local pack inventory
    sub.add_parser("status", help="Print local pack inventory summary (count + total size)")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "bench":
        from .bench import cmd_bench
        return cmd_bench(args)
    if args.command == "bench-compress":
        return _bench(args)
    if args.command == "fit":
        return _fit(args)
    if args.command == "load":
        return _load(args)
    if args.command == "serve":
        return _serve(args)
    if args.command == "pack":
        from .pack import cmd_pack as cmd_pack_legacy
        from .pack_v3 import pack_e2e_dir_v3
        if getattr(args, "legacy_v3", False):
            return cmd_pack_legacy(args)
        # Default: lossless v3.x path via pack_v3 with optional aux file.
        manifest = pack_e2e_dir_v3(
            e2e_dir=args.src, out_dir=args.dst,
            bpw=int(args.bpw), block_size=int(args.block_size),
            include_aux=bool(getattr(args, "include_aux", True)),
            base_hf_id=getattr(args, "base_model", None),
        )
        print(f"\nManifest: {Path(args.dst) / 'manifest.json'}")
        print(f"Overall shrink ratio: {manifest['overall_shrink_ratio']:.2f}x")
        return 0
    if args.command == "pack-aux":
        from .pack_v3 import add_aux_to_existing_pack
        try:
            add_aux_to_existing_pack(
                packed_dir=args.packed_dir,
                base_hf_id=getattr(args, "base_model", None),
            )
            return 0
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"[FAILED] {type(exc).__name__}: {exc}")
            return 2
    if args.command == "inspect":
        from .load_uc import cmd_load
        # cmd_load expects args.packed_dir and args.layer, which we've defined above.
        return cmd_load(args)
    if args.command == "verify":
        from .verify import cmd_verify
        return cmd_verify(args)
    if args.command == "verify-org":
        from .verify_org import cmd_verify_org
        return cmd_verify_org(args)
    if args.command == "status":
        from .verify_org import cmd_status
        return cmd_status(args)
    print(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
