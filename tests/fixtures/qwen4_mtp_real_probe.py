#!/usr/bin/env python3
"""Bounded real-checkpoint probe for one Qwen4 Lightning-MTP draft step.

Run only after ``runtime.memory_preflight`` passes.  This does not prefill or
verify the target model and therefore is not an acceptance/correctness score;
it measures the proposal primitive's released-weight reads, wall time, peak
Metal, selected-expert paging, and cleanup lifetime without persisting logits
or hidden tensors.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import mlx.core as mx
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.engine import RuntimeConfig, StreamingEngine
from runtime.qwen4_mtp import Qwen4MTPDrafter


def _pressure() -> dict[str, int]:
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {
        "available_bytes": int(memory.available),
        "swap_used_bytes": int(swap.used),
        "swap_out_bytes": int(swap.sout),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--token", type=int, default=1)
    args = parser.parse_args()

    before = _pressure()
    rc = RuntimeConfig(
        max_weight_cache_mb=512,
        min_weight_cache_mb=64,
        mlx_cache_limit_mb=128,
        metal_limit_mb=8500,
        pin_embeddings=False,
        pin_lm_head=False,
        embed_rows=True,
        stream_lm_head=True,
        expert_fetch_batch=16,
        decode_expert_fetch_batch=16,
        prefetch_depth=0,
        governor=True,
    )
    started = time.perf_counter()
    engine = StreamingEngine(str(args.model), rc)
    initialized = time.perf_counter()
    drafter = Qwen4MTPDrafter(engine)
    cache = drafter.new_cache()
    source = np.sin(
        np.arange(
            engine.cfg.hidden_size * engine.cfg.qwen4_hc_count,
            dtype=np.float32,
        ) / 17.0,
    ).reshape(1, 1, -1)
    hidden = mx.array(source).astype(mx.bfloat16)
    mx.eval(hidden)
    cache_before = asdict(engine.cache.stats)
    fused_before = engine.store.qwen4_fused_expert_snapshot()
    active_before = int(mx.get_active_memory())
    mx.reset_peak_memory()
    draft_started = time.perf_counter()
    logits, post_hidden = drafter.draft_step(
        hidden, args.token, cache, 0)
    proposal = int(mx.argmax(logits).item())
    mx.eval(post_hidden)
    draft_completed = time.perf_counter()
    peak = int(mx.get_peak_memory())
    cache_after = asdict(engine.cache.stats)
    fused_after = engine.store.qwen4_fused_expert_snapshot()
    release = drafter.release_round_weights()
    released = time.perf_counter()
    report = {
        "schema": "voom.qwen4-mtp-real-probe.v1",
        "model": args.model.name,
        "model_revision": (
            (args.model / ".huggingface_revision").read_text().strip()
            if (args.model / ".huggingface_revision").exists() else None),
        "startup_seconds": round(initialized - started, 6),
        "draft_seconds": round(draft_completed - draft_started, 6),
        "release_seconds": round(released - draft_completed, 6),
        "proposal_token_sha256": hashlib.sha256(
            str(proposal).encode()).hexdigest(),
        "non_expert_storage_bytes": drafter.non_expert_storage_bytes,
        "cache_read_bytes": int(
            cache_after["bytes_read"] - cache_before["bytes_read"]),
        "cache_disk_seconds": float(
            cache_after["disk_s"] - cache_before["disk_s"]),
        "fused_expert": {
            key: int(fused_after[key] - fused_before[key])
            for key in ("calls", "extents", "requested_tensors", "bytes")
        },
        "mtp_kv_bytes": int(cache.nbytes()),
        "active_before_bytes": active_before,
        "peak_metal_bytes": peak,
        "release": release,
        "pressure_before": before,
        "pressure_after": _pressure(),
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.result.write_text(encoded)
    print(encoded, end="")
    engine.close()
    mx.clear_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
