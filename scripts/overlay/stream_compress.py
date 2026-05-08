"""Stream-download + stream-compress: compress models you can't fully download.

WHY THIS EXISTS:
  Customers with 32 GB GPU + ~100 GB disk should be able to compress 405B
  models. The bottleneck today is: HuggingFace serves the model as ~5 GB
  safetensors shards. Conventional pipelines download ALL shards (810 GB for
  405B) before doing anything. We don't need that.

  Each safetensors shard contains 1-3 transformer layers. Stream-compress:
    1. Read the model's `model.safetensors.index.json` to map layer -> shard
    2. For each shard (in order):
       a. Download just that shard
       b. Compress all complete layers it contains
       c. Save compressed layer artifacts
       d. Delete shard
       e. Move to next
    3. Assemble final manifest

  Peak disk: ~one shard (~5 GB) + accumulated compressed artifacts (~tiny)
  Peak GPU: same as the existing per-layer streaming compression
  Peak RAM: same

  Total disk for 405B compression: ~120 GB (compressed) + ~5 GB scratch
  vs existing pipeline: 810 GB download + ~120 GB compressed = 930 GB.

DESIGN STATUS:
  v0 prototype — sequential (download then compress). Future v1 can overlap
  download and compute via a 2-stage pipeline.

USAGE:
    python scripts/overlay/stream_compress.py \\
        --hf-id NousResearch/Hermes-3-Llama-3.1-405B \\
        --output ./compressed/hermes-3-405b-uc \\
        --bpw 5 --rank 32

  Verify safety against existing pipeline (1.7B model):
    python scripts/overlay/stream_compress.py \\
        --hf-id Qwen/Qwen3-1.7B --output ./compressed/qwen3-1.7b-stream \\
        --bpw 5 --rank 32 --max-layers 4
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

if sys.platform == "win32" and sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)


def parse_layer_index_from_key(key: str) -> int | None:
    """`model.layers.27.self_attn.q_proj.weight` -> 27. None for non-layer keys."""
    # Match `model.layers.{N}.` or `layers.{N}.`
    parts = key.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                return None
    return None


def shard_to_layer_map(weight_map: dict[str, str]) -> tuple[dict[str, list[int]], dict[int, list[str]]]:
    """From a HF safetensors index's weight_map, return:
        - shard_to_layers: shard_filename -> [layer_idx, ...]
        - layer_to_shards: layer_idx -> [shard_filename, ...]  (layers can split across shards)
    """
    shard_to_layers: dict[str, set[int]] = defaultdict(set)
    layer_to_shards: dict[int, set[str]] = defaultdict(set)
    for key, shard in weight_map.items():
        layer_idx = parse_layer_index_from_key(key)
        if layer_idx is None:
            continue
        shard_to_layers[shard].add(layer_idx)
        layer_to_shards[layer_idx].add(shard)
    return (
        {s: sorted(layers) for s, layers in shard_to_layers.items()},
        {l: sorted(shards) for l, shards in layer_to_shards.items()},
    )


def fetch_safetensors_index(hf_id: str) -> dict[str, Any]:
    """Download just the index.json (small, KB-sized) to plan compression.

    For models served as a single `model.safetensors` (no shard index — common
    for sub-2B-param models like SmolLM2, TinyLlama, OLMo-2-1B), we synthesize
    a one-shard weight_map by reading the safetensors header directly.
    """
    from huggingface_hub import hf_hub_download
    # Try the safetensors index first; fall back to pytorch index
    for filename in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        try:
            path = hf_hub_download(hf_id, filename)
            return json.loads(Path(path).read_text())
        except Exception:
            continue
    # Single-file safetensors fallback: read the file's tensor list and build
    # a synthetic index pointing every tensor at the single shard.
    try:
        single_path = hf_hub_download(hf_id, "model.safetensors")
    except Exception as e:
        raise RuntimeError(
            f"Could not locate safetensors index OR single-file model.safetensors "
            f"for {hf_id!r}: {e}"
        ) from e
    from safetensors import safe_open
    with safe_open(single_path, framework="pt", device="cpu") as f:
        tensor_names = list(f.keys())
    if not tensor_names:
        raise RuntimeError(f"single-file model.safetensors at {single_path} has no tensors")
    weight_map = {name: "model.safetensors" for name in tensor_names}
    metadata = {
        "total_size": Path(single_path).stat().st_size,
        "synthetic_single_shard": True,
    }
    return {"metadata": metadata, "weight_map": weight_map}


def plan_compression(hf_id: str) -> dict[str, Any]:
    """Read model index, plan shard download order + per-shard layers.

    Returns dict with:
        weight_map: original key -> shard
        shard_to_layers: shard -> [layer_idx,...]
        layer_to_shards: layer_idx -> [shard,...]
        complete_layer_shards: shards where all that shard's layers are FULLY contained
        n_layers_total: max layer_idx + 1
        scaffold_keys: non-layer keys (embed_tokens, norm, lm_head, ...)
    """
    index = fetch_safetensors_index(hf_id)
    weight_map = index["weight_map"]
    shard_to_layers, layer_to_shards = shard_to_layer_map(weight_map)
    scaffold_keys = [k for k in weight_map if parse_layer_index_from_key(k) is None]
    n_layers = (max(layer_to_shards.keys()) + 1) if layer_to_shards else 0

    # A layer is "complete in shard X" iff all shards layer X spans == {X}.
    # If a layer spans multiple shards, we have to wait for all of them.
    complete_layers: dict[str, list[int]] = defaultdict(list)
    deferred_layers: dict[int, list[str]] = {}
    for layer_idx, shards in layer_to_shards.items():
        if len(shards) == 1:
            complete_layers[shards[0]].append(layer_idx)
        else:
            deferred_layers[layer_idx] = shards

    return {
        "weight_map": weight_map,
        "shard_to_layers": shard_to_layers,
        "layer_to_shards": layer_to_shards,
        "complete_layers_per_shard": dict(complete_layers),
        "deferred_layers": deferred_layers,
        "scaffold_keys": scaffold_keys,
        "n_layers_total": n_layers,
        "n_shards": len(shard_to_layers) + (1 if scaffold_keys else 0),
    }


class BufferedShardScheduler:
    """Adaptive scheduler for stream-compress that handles cross-shard layers.

    WHY: v0 (compress-per-shard) only works when every layer fits in one shard.
    405B-class models have layers that span 2-3 shards EACH (every layer of
    Hermes-3-405B is cross-shard). We need to:

        1. Download shards in order
        2. Buffer them until any layer's full shard set is resident
        3. Compress that layer
        4. Evict shards no longer needed

    Pure planning state machine — no I/O. Caller drives:

        sched = BufferedShardScheduler(layer_to_shards, shard_order)
        while not sched.done:
            action = sched.next_action()
            match action[0]:
                case 'download': fetch_shard_to_disk(action[1])
                case 'compress': run_layer_trainer(action[1])
                case 'evict':    delete_shard_file(action[1])

    Greedy invariants:
        - Compress as soon as a layer is satisfied
        - Evict as soon as no PENDING layer needs the shard
        - Download in caller-supplied order (typically the HF index order)
    """

    def __init__(
        self,
        layer_to_shards: dict[int, list[str]],
        shard_order: list[str],
        max_buffer_shards: int | None = None,
    ):
        # Layers we still need to compress: idx -> set of shards required
        self._pending_layer_needs: dict[int, set[str]] = {
            idx: set(shards) for idx, shards in layer_to_shards.items()
        }
        # Shards we still need to download (in order). Filter to layer-only shards
        # — scaffold/non-layer shards (embed_tokens, lm_head) are not the
        # scheduler's concern; the caller loads scaffold separately.
        layer_shards = {s for shards in layer_to_shards.values() for s in shards}
        self._download_queue: list[str] = [s for s in shard_order if s in layer_shards]
        # Shards currently in our local buffer
        self._resident: set[str] = set()
        # For eviction: which pending layers reference each resident shard
        self._max_buffer = max_buffer_shards  # None = unlimited
        # Output log of actions taken (for tests / observability)
        self.actions: list[tuple[str, Any]] = []

    @property
    def done(self) -> bool:
        # Done only after every shard is evicted — caller wants explicit cleanup signals.
        return (
            not self._pending_layer_needs
            and not self._download_queue
            and not self._resident
        )

    @property
    def buffer_size(self) -> int:
        return len(self._resident)

    @property
    def pending_layers(self) -> int:
        return len(self._pending_layer_needs)

    def _layers_satisfied_now(self) -> list[int]:
        """Layers whose full shard set is currently resident."""
        return sorted(
            idx for idx, needs in self._pending_layer_needs.items()
            if needs.issubset(self._resident)
        )

    def _shard_still_needed(self, shard: str) -> bool:
        """True if any pending layer still references this shard."""
        return any(shard in needs for needs in self._pending_layer_needs.values())

    def next_action(self) -> tuple[str, Any]:
        """Return next ('compress', layer_idx) | ('evict', shard) | ('download', shard) | ('done',)."""
        # 1. Greedy compress: any layer whose shards are all resident
        ready = self._layers_satisfied_now()
        if ready:
            layer_idx = ready[0]
            del self._pending_layer_needs[layer_idx]
            action = ("compress", layer_idx)
            self.actions.append(action)
            return action

        # 2. Greedy evict: any resident shard with no pending needs
        for shard in sorted(self._resident):
            if not self._shard_still_needed(shard):
                self._resident.remove(shard)
                action = ("evict", shard)
                self.actions.append(action)
                return action

        # 3. If buffer cap hit and we have nothing to compress/evict — stuck
        if (
            self._max_buffer is not None
            and len(self._resident) >= self._max_buffer
            and self._download_queue
        ):
            raise RuntimeError(
                f"Buffer cap {self._max_buffer} reached with {len(self._resident)} "
                f"shards resident, {self.pending_layers} pending layers, but no "
                f"layer can be completed. Cap is too small for this model's "
                f"cross-shard layer fan-in. Increase --max-buffer-shards."
            )

        # 4. Download next shard
        if self._download_queue:
            shard = self._download_queue.pop(0)
            self._resident.add(shard)
            action = ("download", shard)
            self.actions.append(action)
            return action

        action = ("done", None)
        self.actions.append(action)
        return action

    def run_simulation(self, max_steps: int = 100_000) -> dict[str, int]:
        """Drive scheduler to completion (no I/O). Returns counts of each action type."""
        counts = defaultdict(int)
        peak_buffer = 0
        for _ in range(max_steps):
            if self.done:
                break
            kind, _ = self.next_action()
            counts[kind] += 1
            peak_buffer = max(peak_buffer, len(self._resident))
            if kind == "done":
                break
        else:
            raise RuntimeError(f"Scheduler did not converge in {max_steps} steps")
        counts["peak_buffer_shards"] = peak_buffer
        return dict(counts)


def extract_layer_from_shards(
    layer_idx: int,
    weight_map: dict[str, str],
    shard_dir: Path,
) -> dict[str, "Any"]:
    """Load one transformer layer's tensors from one or more local safetensors shards.

    Used when the BufferedShardScheduler returns ('compress', layer_idx) — we
    have all of layer_idx's shards resident in `shard_dir` and need to slice
    out just that layer's parameter tensors.

    Returns dict mapping HF parameter key -> CPU tensor. Caller is responsible
    for moving to the appropriate device for compression.

    Imports `safetensors.torch.safe_open` lazily so the planner+scheduler
    remain importable without the dep.
    """
    from safetensors.torch import safe_open  # type: ignore

    keys_for_layer = [k for k, _ in weight_map.items()
                      if parse_layer_index_from_key(k) == layer_idx]
    if not keys_for_layer:
        raise KeyError(f"No keys for layer {layer_idx} in weight_map")

    # Group keys by shard so we open each shard once
    by_shard: dict[str, list[str]] = defaultdict(list)
    for k in keys_for_layer:
        by_shard[weight_map[k]].append(k)

    tensors: dict[str, Any] = {}
    for shard_name, keys in by_shard.items():
        shard_path = shard_dir / shard_name
        if not shard_path.exists():
            raise FileNotFoundError(
                f"Shard {shard_name} not resident in {shard_dir} for layer {layer_idx}. "
                f"Did the scheduler advance to 'compress' before all shards were downloaded?"
            )
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for k in keys:
                tensors[k] = f.get_tensor(k)
    return tensors


def shard_download_order(weight_map: dict[str, str]) -> list[str]:
    """HF safetensors shards in numerical order (model-00001-of-N, ...).

    Falls back to lexicographic if the naming pattern doesn't match.
    """
    shards = sorted(set(weight_map.values()))
    # Try to detect `*-NNNNN-of-MMMMM.*` pattern and sort numerically
    import re
    rx = re.compile(r"-(\d+)-of-\d+\.")
    def key(s: str):
        m = rx.search(s)
        return (0, int(m.group(1))) if m else (1, s)
    return sorted(shards, key=key)


def run_stream_compress(
    hf_id: str,
    output_dir: Path,
    compress_layer_fn,  # Callable[[int, dict[str, Tensor], Path], None]
    *,
    download_shard_fn=None,  # Callable[[str, str, Path], Path]; default uses hf_hub_download
    delete_shard_fn=None,    # Callable[[Path], None]; default uses os.unlink
    log_fn=print,
    max_buffer_shards: int | None = None,
    max_layers: int | None = None,
    skip_eviction: bool = False,  # for debug — keep all shards on disk after run
) -> dict[str, Any]:
    """End-to-end stream-compress driver. Pluggable trainer + I/O for tests.

    `compress_layer_fn(layer_idx, layer_tensors, output_dir)` is the only
    domain-specific callback. Wraps the existing per-layer V18-C trainer.

    Returns dict with run statistics: shards_downloaded, layers_compressed,
    shards_evicted, peak_buffer_shards.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir = output_dir / "_stream_scratch"
    scratch_dir.mkdir(exist_ok=True)

    plan = plan_compression(hf_id)
    weight_map = plan["weight_map"]
    layer_to_shards = plan["layer_to_shards"]

    # Default I/O backends
    if download_shard_fn is None:
        from huggingface_hub import hf_hub_download

        def download_shard_fn(hf_id_: str, shard_name: str, dst_dir: Path) -> Path:
            # `local_dir` puts the file exactly at {dst_dir}/{shard_name},
            # which is what extract_layer_from_shards expects. Without it,
            # hf_hub_download builds the cache layout {cache_dir}/models--*.
            local = hf_hub_download(
                repo_id=hf_id_,
                filename=shard_name,
                local_dir=str(dst_dir),
            )
            return Path(local)
    if delete_shard_fn is None:
        import os

        def delete_shard_fn(path: Path) -> None:
            try:
                os.unlink(path)
            except OSError as e:
                log_fn(f"[stream] warn: could not delete {path}: {e}")

    order = shard_download_order(weight_map)
    sched = BufferedShardScheduler(layer_to_shards, order, max_buffer_shards=max_buffer_shards)

    layers_done = 0
    peak_buffer = 0
    shard_paths: dict[str, Path] = {}  # shard_name -> local path

    while not sched.done:
        kind, payload = sched.next_action()
        if kind == "download":
            shard_name = payload
            log_fn(f"[stream] download {shard_name} -> {scratch_dir}")
            t0 = time.time()
            local = download_shard_fn(hf_id, shard_name, scratch_dir)
            shard_paths[shard_name] = local
            peak_buffer = max(peak_buffer, len(shard_paths))
            log_fn(f"[stream]   downloaded in {time.time() - t0:.1f}s "
                   f"(buffer now {len(shard_paths)} shards)")
        elif kind == "compress":
            layer_idx = payload
            log_fn(f"[stream] compress layer {layer_idx}")
            tensors = extract_layer_from_shards(layer_idx, weight_map, scratch_dir)
            t0 = time.time()
            compress_layer_fn(layer_idx, tensors, output_dir)
            log_fn(f"[stream]   layer {layer_idx} compressed in {time.time() - t0:.1f}s")
            layers_done += 1
            if max_layers is not None and layers_done >= max_layers:
                log_fn(f"[stream] max_layers={max_layers} reached, stopping early")
                break
        elif kind == "evict":
            shard_name = payload
            if skip_eviction:
                log_fn(f"[stream] (skip_eviction) keeping {shard_name}")
                continue
            local = shard_paths.pop(shard_name, None)
            if local is not None:
                delete_shard_fn(local)
                log_fn(f"[stream] evict {shard_name} (buffer now {len(shard_paths)} shards)")

    # Final cleanup of scratch if nothing left to evict naturally
    if not skip_eviction:
        try:
            shutil.rmtree(scratch_dir, ignore_errors=True)
        except Exception:
            pass

    return {
        "hf_id": hf_id,
        "shards_downloaded": sum(1 for a in sched.actions if a[0] == "download"),
        "layers_compressed": layers_done,
        "shards_evicted": sum(1 for a in sched.actions if a[0] == "evict"),
        "peak_buffer_shards": peak_buffer,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Stream-download + stream-compress (v0 prototype)")
    ap.add_argument("--hf-id", required=True, help="HuggingFace model id")
    ap.add_argument("--output", type=Path, required=True, help="Output dir for compressed artifact")
    ap.add_argument("--bpw", type=int, default=5)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--max-layers", type=int, default=0,
                    help="Stop after compressing N layers (default 0 = all). For testing.")
    ap.add_argument("--plan-only", action="store_true",
                    help="Print the compression plan and exit (no downloads, no compression)")
    ap.add_argument("--simulate", action="store_true",
                    help="Run the BufferedShardScheduler simulation (no I/O). Shows peak buffer + step counts.")
    ap.add_argument("--max-buffer-shards", type=int, default=0,
                    help="Cap on resident shards during streaming (0 = unlimited).")
    args = ap.parse_args()

    print(f"[stream] Planning compression for {args.hf_id}...", flush=True)
    plan = plan_compression(args.hf_id)
    print(f"[stream]   total layers: {plan['n_layers_total']}", flush=True)
    print(f"[stream]   total shards: {plan['n_shards']}", flush=True)
    print(f"[stream]   scaffold keys: {len(plan['scaffold_keys'])}", flush=True)
    print(f"[stream]   layers fully contained in single shard: "
          f"{sum(len(v) for v in plan['complete_layers_per_shard'].values())}", flush=True)
    print(f"[stream]   deferred (cross-shard) layers: {len(plan['deferred_layers'])}", flush=True)

    if args.simulate:
        print("\n[stream] SIMULATE mode — driving BufferedShardScheduler with no I/O...")
        order = shard_download_order(plan["weight_map"])
        sched = BufferedShardScheduler(
            plan["layer_to_shards"],
            order,
            max_buffer_shards=(args.max_buffer_shards or None),
        )
        try:
            counts = sched.run_simulation()
        except RuntimeError as e:
            print(f"[stream] SIMULATION FAILED: {e}", flush=True)
            return 1
        print(f"[stream]   downloads:        {counts.get('download', 0)}")
        print(f"[stream]   compressions:     {counts.get('compress', 0)}")
        print(f"[stream]   evictions:        {counts.get('evict', 0)}")
        print(f"[stream]   peak buffer:      {counts['peak_buffer_shards']} shards "
              f"(~{counts['peak_buffer_shards'] * 5:.0f} GB at 5 GB/shard estimate)")
        print(f"[stream]   total scheduler steps: {sum(v for k, v in counts.items() if k != 'peak_buffer_shards')}")
        return 0

    if args.plan_only:
        print("\n[stream] PLAN-ONLY mode. Exiting.")
        return 0

    print("\n[stream] V0 prototype — full implementation deferred. The plan above")
    print("[stream] is the foundation. Next step: implement per-shard download +")
    print("[stream] reuse the existing per-layer V18-C training loop.")
    print("\n[stream] To validate the plan logic without downloading:")
    print(f"[stream]   python scripts/overlay/stream_compress.py --hf-id {args.hf_id} --output {args.output} --plan-only")
    return 0


if __name__ == "__main__":
    sys.exit(main())
