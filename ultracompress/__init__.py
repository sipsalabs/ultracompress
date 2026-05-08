"""
UltraCompress -- production model compression.

Default `uc.compress(...)` is the v3 stack: scalar symmetric per-row
quantization + learned low-rank correction, trained via KL distillation.

Empirical baseline (Qwen3-1.7B, rank=32, 1500 steps):
    T1 = 93.87%, PPL ratio = 1.0018, param overhead = 1.99%.

The v2 (v17 codebook) pipeline is kept under `ultracompress.api_v2` and is
deprecated -- it had a structural ~75% T1 ceiling and is no longer the default.

Usage:
  import ultracompress as uc
  compressed = uc.compress(model, mode='scalar_v18c',
                           target_bpw=6.0, correction_rank=32,
                           tokens=calibration_token_ids)
  uc.save(compressed, 'my_model.uc')
  reloaded = uc.load('my_model.uc', fresh_skeleton)
"""
import warnings as _warnings

__version__ = "0.5.0"

# v3 is the default
from ultracompress.api_v3 import (
    compress,
    save,
    load,
    CompressedModel,
    CompressionReport,
    SCHEMA_VERSION,
)

# Keep v2 importable but emit a deprecation warning when accessed.
from ultracompress import api_v2 as _api_v2
from ultracompress import api as _legacy_api  # noqa: F401


def _deprecated_v2_compress(*args, **kwargs):
    _warnings.warn(
        "ultracompress.api_v2.compress is deprecated; the v17 codebook stack "
        "had a structural ~75%% T1 ceiling. Use uc.compress (v3, scalar+V18-C) "
        "instead. v2 will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _api_v2.compress_v2_compat(*args, **kwargs)


# Patch a deprecation shim onto the v2 module's compress entry point.
_api_v2.compress_v2_compat = _api_v2.compress  # preserve original under new name
_api_v2.compress = _deprecated_v2_compress  # type: ignore[assignment]

__all__ = [
    "compress", "save", "load",
    "CompressedModel", "CompressionReport",
    "SCHEMA_VERSION",
]
