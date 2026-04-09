"""
UltraCompress - Extreme LLM Compression Research Pipeline

Compresses large language models using tri-path hybrid compression:
  Path 1: SVD + VQ fusion — factorize, then vector-quantize the factors
  Path 2: Direct VQ — vector quantize raw weights (sub-1 BPW)
  Path 3: Scalar quantization (INT2-8) — fast fallback
  + Calibration-aware optimization using Hessian weighting

With binarization, codebook, and sparsity stages for further compression.

Current target: 235B -> 20GB (0.68 BPW)
Ultimate target: 10T -> 20GB (0.016 BPW)
"""

__version__ = "0.4.0"
