"""
UltraCompress — Extreme Model Compression via Fractal Residual Recursion

One shared transformer block replaces all layers. 60x smaller, 3x faster.
Combined with quantization pipeline: 959x end-to-end compression.

Key results (Qwen3-0.6B -> 1.7B):
  - FRR 1.7B: 66% T10 at 48x compression (all-time best)
  - FRR + Q2 E2E: 53% T10 at 959x compression (proven)
  - PHM variant: 53% T10 at 239x (4x fewer params)
  - Inference: 3.1-3.4x faster (L2 cache)

Usage:
  from ultracompress.deploy import load_compressed
  model = load_compressed("compressed.ucz")
  print(model("The future of AI is"))

Or compress your own model:
  python compress_frr.py --model Qwen/Qwen3-0.6B --steps 50000
"""

__version__ = "0.3.0"
