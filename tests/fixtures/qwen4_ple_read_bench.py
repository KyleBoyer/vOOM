#!/usr/bin/env python3
"""Content-blind real-row timing for Qwen4-Exp's released PLE table."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.qwen4_exp_ple_rows import Qwen4ExpPLERowStore


def _run(model: Path, *, workers: int, seed: int, tokens: int) -> dict:
    with Qwen4ExpPLERowStore(
            model, row_cache=0, read_workers=workers) as store:
        rng = np.random.default_rng(seed)
        token_ids = rng.integers(
            0, store.layout.unigram_vocab_size,
            size=tokens, dtype=np.int64).tolist()
        row_ids = store.layout.row_ids(token_ids)
        started = time.perf_counter()
        rows = store.read_rows(row_ids)
        wall = time.perf_counter() - started
        stats = store.telemetry()
        return {
            "workers": workers,
            "seed": seed,
            "tokens": tokens,
            "wall_seconds": round(wall, 6),
            "rows_sha256": hashlib.sha256(rows.tobytes()).hexdigest(),
            "rows_requested": stats["rows_requested"],
            "unique_rows_read": stats["unique_rows_read"],
            "read_extents": stats["read_extents"],
            "bytes_read": stats["bytes_read"],
            "read_microseconds": stats["read_microseconds"],
            "revision": store.identity.revision,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--parallel-workers", type=int, default=8)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.tokens <= 0:
        parser.error("tokens must be positive")
    if not 2 <= args.parallel_workers <= 16:
        parser.error("parallel-workers must be in [2, 16]")
    schedule = (
        (1, 101),
        (args.parallel_workers, 211),
        (args.parallel_workers, 307),
        (1, 401),
    )
    rows = [
        _run(args.model, workers=workers, seed=seed, tokens=args.tokens)
        for workers, seed in schedule
    ]
    serial = statistics.median(
        row["wall_seconds"] for row in rows if row["workers"] == 1)
    parallel = statistics.median(
        row["wall_seconds"] for row in rows if row["workers"] != 1)
    result = {
        "schema": "voom.qwen4-ple-read-bench.v1",
        "model_revision": rows[0]["revision"],
        "tokens_per_case": args.tokens,
        "logical_bytes_per_case": rows[0]["bytes_read"],
        "serial_median_seconds": round(serial, 6),
        "parallel_median_seconds": round(parallel, 6),
        "speedup": round(serial / parallel, 6) if parallel else None,
        "rows": rows,
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(encoded, end="")
    if args.result is not None:
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
