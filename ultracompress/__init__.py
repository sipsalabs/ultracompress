"""
UltraCompress -- production model compression.

Default `uc.compress(...)` is the v3 stack: scalar symmetric per-row
quantization + learned low-rank correction, trained via KL distillation.

Production hyperparameters and codec internals are patent-protected
(USPTO 64/049,511 + 64/049,517). See https://github.com/sipsalabs/ultracompress
for verified PPL ratios per architecture in BENCHMARKS.json.

Legacy codebook-based pipelines are not part of this distribution; they were
deprecated in favor of the v3 default and are no longer shipped.

Usage:
  import ultracompress as uc
  compressed = uc.compress(model, mode='scalar',
                           target_bpw=5.0,
                           tokens=calibration_token_ids)
  uc.save(compressed, 'my_model.uc')
  reloaded = uc.load('my_model.uc', fresh_skeleton)
"""

__version__ = "0.6.2"

# v3 is the default
from ultracompress.api_v3 import (
    compress,
    save,
    load,
    CompressedModel,
    CompressionReport,
    SCHEMA_VERSION,
)

# Public bench API - imported lazily to avoid pulling transformers/torch eagerly
# when the user only needs `compress` / `save` / `load`.
def bench_packed(*args, **kwargs):
    """Inference throughput benchmark on a UC v3 packed model.

    See `ultracompress.bench.bench_packed` for the full signature and docs.
    """
    from ultracompress.bench import bench_packed as _impl
    return _impl(*args, **kwargs)


__all__ = [
    "compress", "save", "load",
    "CompressedModel", "CompressionReport",
    "SCHEMA_VERSION",
    "bench_packed",
]
