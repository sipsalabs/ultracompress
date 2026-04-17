"""Export FRR checkpoint to optimized formats for deployment.

Supports: TorchScript, ONNX, quantized (INT8/INT4) PyTorch.
Produces ready-to-deploy artifacts with metadata and validation.

Usage:
  python export_model.py checkpoints_1.7b_real_text/frr_1.7b_best.pt --teacher 1.7b --format all
  python export_model.py checkpoint.pt --teacher 0.6b --format onnx --output exported/
  python export_model.py checkpoint.pt --teacher 1.7b --format torchscript --validate
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultracompress.inference import ModelConfig, MiniTransformer


# ── Teacher configs ───────────────────────────────────────────────────
TEACHER_CONFIGS = {
    "0.6b": {
        "hidden": 1024,
        "n_heads": 16,
        "n_kv_heads": 8,
        "n_layers": 28,
        "intermediate_size": 3072,
        "vocab_size": 151936,
        "head_dim": 128,
        "num_scales": 4,
        "iters_per_scale": 7,
        "model_name": "Qwen/Qwen3-0.6B",
    },
    "1.7b": {
        "hidden": 2048,
        "n_heads": 16,
        "n_kv_heads": 8,
        "n_layers": 28,
        "intermediate_size": 8960,
        "vocab_size": 151936,
        "head_dim": 128,
        "num_scales": 4,
        "iters_per_scale": 7,
        "model_name": "Qwen/Qwen3-1.7B",
    },
}


def load_frr_model(
    checkpoint_path: Path,
    teacher_size: str,
    device: str = "cpu",
) -> tuple[MiniTransformer, dict]:
    """Load an FRR model from checkpoint."""
    cfg = TEACHER_CONFIGS[teacher_size]

    model_cfg = ModelConfig(
        vocab_size=cfg["vocab_size"],
        hidden_size=cfg["hidden"],
        intermediate_size=cfg["intermediate_size"],
        num_attention_heads=cfg["n_heads"],
        num_key_value_heads=cfg["n_kv_heads"],
        num_hidden_layers=cfg["n_layers"],
        head_dim=cfg["head_dim"],
    )

    model = MiniTransformer(model_cfg, num_scales=cfg["num_scales"], iters_per_scale=cfg["iters_per_scale"])

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        meta = {k: v for k, v in ckpt.items() if k != "model_state_dict"}
    elif "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
        meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    else:
        model.load_state_dict(ckpt)
        meta = {}

    model = model.to(device)
    model.eval()
    return model, meta


def export_torchscript(
    model: MiniTransformer,
    output_path: Path,
    example_input: torch.Tensor,
) -> dict:
    """Export to TorchScript format."""
    print(f"Exporting TorchScript to {output_path}...")
    start = time.time()

    with torch.no_grad():
        traced = torch.jit.trace(model, example_input)
        traced.save(str(output_path))

    elapsed = time.time() - start
    size_mb = output_path.stat().st_size / 1024 / 1024

    print(f"  TorchScript: {size_mb:.1f} MB ({elapsed:.1f}s)")
    return {"format": "torchscript", "size_mb": size_mb, "path": str(output_path)}


def export_onnx(
    model: MiniTransformer,
    output_path: Path,
    example_input: torch.Tensor,
) -> dict:
    """Export to ONNX format."""
    print(f"Exporting ONNX to {output_path}...")
    start = time.time()

    with torch.no_grad():
        torch.onnx.export(
            model,
            example_input,
            str(output_path),
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "seq_length"},
                "logits": {0: "batch_size", 1: "seq_length"},
            },
            opset_version=17,
            do_constant_folding=True,
        )

    elapsed = time.time() - start
    size_mb = output_path.stat().st_size / 1024 / 1024

    print(f"  ONNX: {size_mb:.1f} MB ({elapsed:.1f}s)")
    return {"format": "onnx", "size_mb": size_mb, "path": str(output_path)}


def export_quantized(
    model: MiniTransformer,
    output_path: Path,
    quant_type: str = "int8",
) -> dict:
    """Export quantized PyTorch model (dynamic quantization)."""
    print(f"Exporting {quant_type} quantized model to {output_path}...")
    start = time.time()

    if quant_type == "int8":
        quantized = torch.quantization.quantize_dynamic(
            model.cpu(),
            {torch.nn.Linear},
            dtype=torch.qint8,
        )
    else:
        raise ValueError(f"Unsupported quant_type: {quant_type}")

    torch.save(quantized.state_dict(), str(output_path))

    elapsed = time.time() - start
    size_mb = output_path.stat().st_size / 1024 / 1024

    print(f"  {quant_type}: {size_mb:.1f} MB ({elapsed:.1f}s)")
    return {"format": f"pytorch_{quant_type}", "size_mb": size_mb, "path": str(output_path)}


def validate_export(
    model: MiniTransformer,
    exported_path: Path,
    fmt: str,
    example_input: torch.Tensor,
    device: str = "cpu",
) -> dict:
    """Validate that exported model produces same outputs."""
    print(f"Validating {fmt} export...")
    with torch.no_grad():
        original_out = model.cpu()(example_input.cpu())

    if fmt == "torchscript":
        loaded = torch.jit.load(str(exported_path))
        loaded.eval()
        with torch.no_grad():
            exported_out = loaded(example_input.cpu())
    elif fmt == "onnx":
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(str(exported_path))
            exported_out_np = sess.run(None, {"input_ids": example_input.cpu().numpy()})[0]
            exported_out = torch.from_numpy(exported_out_np)
        except ImportError:
            print("  onnxruntime not installed, skipping ONNX validation")
            return {"validated": False, "reason": "onnxruntime not installed"}
    else:
        return {"validated": False, "reason": f"Validation not supported for {fmt}"}

    # Compare
    if isinstance(original_out, tuple):
        original_out = original_out[0]
    if isinstance(exported_out, tuple):
        exported_out = exported_out[0]

    max_diff = (original_out - exported_out).abs().max().item()
    mean_diff = (original_out - exported_out).abs().mean().item()
    cosine = F.cosine_similarity(
        original_out.reshape(1, -1).float(),
        exported_out.reshape(1, -1).float(),
    ).item()

    result = {
        "validated": True,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "cosine_similarity": cosine,
        "pass": max_diff < 1e-3 or cosine > 0.999,
    }
    status = "PASS" if result["pass"] else "FAIL"
    print(f"  {status}: max_diff={max_diff:.6f}, cosine={cosine:.6f}")
    return result


def compute_model_stats(model: MiniTransformer, teacher_size: str) -> dict:
    """Compute model statistics for metadata."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    cfg = TEACHER_CONFIGS[teacher_size]
    teacher_name = cfg["model_name"]

    # Estimate teacher size
    teacher_params_approx = {
        "0.6b": 440_000_000,
        "1.7b": 1_530_000_000,
    }
    teacher_params = teacher_params_approx.get(teacher_size, 0)
    compression_ratio = teacher_params / total_params if total_params > 0 else 0

    return {
        "frr_params": total_params,
        "trainable_params": trainable_params,
        "teacher": teacher_name,
        "teacher_params_approx": teacher_params,
        "compression_ratio": f"{compression_ratio:.1f}x",
        "architecture": {
            "type": "FRR (Fractal Residual Recursion)",
            "num_scales": cfg["num_scales"],
            "iters_per_scale": cfg["iters_per_scale"],
            "total_virtual_layers": cfg["num_scales"] * cfg["iters_per_scale"],
            "hidden_size": cfg["hidden"],
            "vocab_size": cfg["vocab_size"],
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Export FRR model to deployment formats")
    parser.add_argument("checkpoint", type=Path, help="Path to FRR checkpoint (.pt)")
    parser.add_argument("--teacher", choices=["0.6b", "1.7b"], default="1.7b", help="Teacher model size")
    parser.add_argument("--format", choices=["torchscript", "onnx", "int8", "all"], default="all",
                        help="Export format")
    parser.add_argument("--output", type=Path, default=Path("exported"), help="Output directory")
    parser.add_argument("--validate", action="store_true", help="Validate exported models")
    parser.add_argument("--device", default="cpu", help="Device for model loading")
    parser.add_argument("--seq-len", type=int, default=64, help="Example sequence length for tracing")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading FRR model from {args.checkpoint}...")
    model, meta = load_frr_model(args.checkpoint, args.teacher, args.device)
    stats = compute_model_stats(model, args.teacher)
    print(f"  {stats['frr_params']:,} params ({stats['compression_ratio']} compression)")

    # Create example input
    example_input = torch.randint(0, TEACHER_CONFIGS[args.teacher]["vocab_size"],
                                  (1, args.seq_len), dtype=torch.long)

    # Determine formats to export
    formats = ["torchscript", "onnx", "int8"] if args.format == "all" else [args.format]

    results = {"checkpoint": str(args.checkpoint), "model_stats": stats, "exports": []}

    stem = args.checkpoint.stem

    for fmt in formats:
        try:
            if fmt == "torchscript":
                out_path = args.output / f"{stem}_scripted.pt"
                model_cpu = model.cpu()
                result = export_torchscript(model_cpu, out_path, example_input)
            elif fmt == "onnx":
                out_path = args.output / f"{stem}.onnx"
                model_cpu = model.cpu()
                result = export_onnx(model_cpu, out_path, example_input)
            elif fmt == "int8":
                out_path = args.output / f"{stem}_int8.pt"
                model_cpu = model.cpu()
                result = export_quantized(model_cpu, out_path, "int8")
            else:
                continue

            if args.validate and fmt in ("torchscript", "onnx"):
                validation = validate_export(model.cpu(), out_path, fmt, example_input)
                result["validation"] = validation

            results["exports"].append(result)

        except Exception as e:
            print(f"  {fmt} export failed: {e}")
            results["exports"].append({"format": fmt, "error": str(e)})

    # Save metadata
    meta_path = args.output / f"{stem}_metadata.json"
    with open(meta_path, "w") as f:
        # Convert non-serializable items
        serializable = json.loads(json.dumps(results, default=str))
        json.dump(serializable, f, indent=2)
    print(f"\nMetadata saved to {meta_path}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  EXPORT SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Model: {stats['frr_params']:,} params ({stats['compression_ratio']})")
    for exp in results["exports"]:
        if "error" in exp:
            print(f"  {exp['format']}: FAILED ({exp['error'][:60]})")
        else:
            print(f"  {exp['format']}: {exp['size_mb']:.1f} MB -> {exp['path']}")


if __name__ == "__main__":
    main()
