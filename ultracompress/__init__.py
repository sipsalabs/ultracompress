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

__version__ = "0.5.1"

# v3 is the default
from ultracompress.api_v3 import (
    compress,
    save,
    load,
    CompressedModel,
    CompressionReport,
    SCHEMA_VERSION,
)

# v2 / legacy api are optional — they reference internal research modules that
# are not always present at install time. We import them defensively so the
# customer-facing `uc` CLI (pack / load / verify) keeps working even when the
# legacy v2 dependencies are missing.
try:
    from ultracompress import api_v2 as _api_v2  # type: ignore[unused-ignore]
    from ultracompress import api as _legacy_api  # noqa: F401
    _API_V2_AVAILABLE = True
except Exception:  # pragma: no cover
    _api_v2 = None  # type: ignore[assignment]
    _API_V2_AVAILABLE = False


def _deprecated_v2_compress(*args, **kwargs):
    if _api_v2 is None:
        raise ImportError(
            "ultracompress.api_v2 is unavailable in this install (missing "
            "internal research dependencies). Use uc.compress (v3) instead."
        )
    _warnings.warn(
        "ultracompress.api_v2.compress is deprecated; the v17 codebook stack "
        "had a structural ~75%% T1 ceiling. Use uc.compress (v3, scalar+V18-C) "
        "instead. v2 will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _api_v2.compress_v2_compat(*args, **kwargs)


# Patch a deprecation shim onto the v2 module's compress entry point only when
# v2 actually loaded.
if _api_v2 is not None:
    _api_v2.compress_v2_compat = _api_v2.compress  # preserve original under new name
    _api_v2.compress = _deprecated_v2_compress  # type: ignore[assignment]

__all__ = [
    "compress", "save", "load",
    "CompressedModel", "CompressionReport",
    "SCHEMA_VERSION",
]
