"""Weights-free exact GLM DSA top-k merge schedule benchmark."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mlx.core as mx

from runtime.glm_dsa import DSAState


def _legacy(scores: mx.array, tile: int, keep: int) -> mx.array:
    queries, keys = map(int, scores.shape)
    running_scores = None
    running_ids = None
    for start in range(0, keys, tile):
        end = min(start + tile, keys)
        candidate_scores = scores[:, start:end]
        candidate_ids = mx.broadcast_to(
            mx.arange(start, end, dtype=mx.int32)[None],
            (queries, end - start),
        )
        if running_scores is not None:
            candidate_scores = mx.concatenate(
                (running_scores, candidate_scores), axis=-1)
            candidate_ids = mx.concatenate(
                (running_ids, candidate_ids), axis=-1)
        id_order = mx.argsort(candidate_ids, axis=-1)
        candidate_ids = mx.take_along_axis(
            candidate_ids, id_order, axis=-1)
        candidate_scores = mx.take_along_axis(
            candidate_scores, id_order, axis=-1)
        score_order = mx.argsort(-candidate_scores, axis=-1)[..., :keep]
        running_scores = mx.take_along_axis(
            candidate_scores, score_order, axis=-1)
        running_ids = mx.take_along_axis(
            candidate_ids, score_order, axis=-1)
        mx.eval(running_scores, running_ids)
    return mx.sort(running_ids, axis=-1)


def _candidate(scores: mx.array, tile: int, keep: int) -> mx.array:
    queries, keys = map(int, scores.shape)
    running_scores = None
    running_ids = None
    for start in range(0, keys, tile):
        end = min(start + tile, keys)
        candidate_scores = scores[:, start:end]
        candidate_ids = mx.broadcast_to(
            mx.arange(start, end, dtype=mx.int32)[None],
            (queries, end - start),
        )
        if running_scores is not None:
            candidate_scores = mx.concatenate(
                (running_scores, candidate_scores), axis=-1)
            candidate_ids = mx.concatenate(
                (running_ids, candidate_ids), axis=-1)
        running_scores, running_ids = DSAState._stable_topk(
            candidate_scores, candidate_ids, keep)
        mx.eval(running_scores, running_ids)
    return running_ids


def _timed(call, repetitions: int) -> tuple[list[float], int, mx.array]:
    samples = []
    peak = 0
    output = None
    for _ in range(repetitions):
        mx.clear_cache()
        mx.reset_peak_memory()
        started = time.perf_counter()
        output = call()
        mx.eval(output)
        samples.append(time.perf_counter() - started)
        peak = max(peak, int(mx.get_peak_memory()))
    assert output is not None
    return samples, peak, output


def _legacy_selector(
        query: mx.array, head_weights: mx.array, keys: mx.array, *,
        offset: int, tile: int, keep: int,
) -> mx.array:
    batch, queries, heads, head_dim = map(int, query.shape)
    key_count = int(keys.shape[1])
    query_positions = mx.arange(
        offset, offset + queries, dtype=mx.int32)[:, None]
    running_scores = None
    running_ids = None
    for tile_index, start in enumerate(range(0, key_count, tile)):
        end = min(start + tile, key_count)
        scores = mx.einsum(
            "blje,bse->bljs", query.astype(mx.float32),
            keys[:, start:end, :].astype(mx.float32))
        scores = mx.maximum(scores * (head_dim ** -0.5), 0.0)
        scores = (scores * head_weights[..., None]).sum(axis=2)
        ids = mx.broadcast_to(
            mx.arange(start, end, dtype=mx.int32)[None, None, :],
            (batch, queries, end - start),
        )
        scores = mx.where(
            ids <= query_positions[None], scores,
            mx.array(float("-inf"), dtype=mx.float32),
        )
        if running_scores is not None:
            scores = mx.concatenate((running_scores, scores), axis=-1)
            ids = mx.concatenate((running_ids, ids), axis=-1)
        id_order = mx.argsort(ids, axis=-1)
        ids = mx.take_along_axis(ids, id_order, axis=-1)
        scores = mx.take_along_axis(scores, id_order, axis=-1)
        score_order = mx.argsort(-scores, axis=-1)[..., :keep]
        running_scores = mx.take_along_axis(
            scores, score_order, axis=-1)
        running_ids = mx.take_along_axis(ids, score_order, axis=-1)
        if (tile_index + 1) % 8 == 0:
            mx.eval(running_scores, running_ids)
    valid = running_scores > mx.array(float("-inf"), dtype=mx.float32)
    sentinel = mx.array(key_count, dtype=mx.int32)
    ordered = mx.sort(
        mx.where(valid, running_ids, sentinel), axis=-1)
    return mx.where(ordered < key_count, ordered, -1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=32)
    parser.add_argument("--keys", type=int, default=46849)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--tile", type=int, default=1024)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--score-selector", action="store_true")
    parser.add_argument("--result-json", type=Path)
    args = parser.parse_args()
    if min(args.queries, args.keys, args.topk, args.tile,
           args.repetitions) <= 0 or args.keys <= args.topk:
        parser.error("invalid selector geometry")
    if args.result_json is not None and args.result_json.exists():
        parser.error("result JSON already exists")

    # Quantized deterministic scores deliberately create many exact ties,
    # including at merge boundaries, so equality proves the secondary-ID rule
    # rather than relying on random scores being unique.
    rows = mx.arange(args.queries, dtype=mx.int32)[:, None]
    columns = mx.arange(args.keys, dtype=mx.int32)[None, :]
    scores = (((rows * 313 + columns * 17) % 257).astype(mx.float32) / 256)
    mx.eval(scores)
    expected = _legacy(scores, args.tile, args.topk)
    actual = _candidate(scores, args.tile, args.topk)
    mx.eval(expected, actual)
    identical = bool(mx.array_equal(expected, actual).item())

    legacy_samples, legacy_peak, expected = _timed(
        lambda: _legacy(scores, args.tile, args.topk), args.repetitions)
    candidate_samples, candidate_peak, actual = _timed(
        lambda: _candidate(scores, args.tile, args.topk), args.repetitions)
    identical = identical and bool(mx.array_equal(expected, actual).item())
    legacy_median = statistics.median(legacy_samples)
    candidate_median = statistics.median(candidate_samples)
    result = {
        "schema": "voom.glm-dsa-merge-bench.v1",
        "geometry": {
            "queries": args.queries,
            "keys": args.keys,
            "topk": args.topk,
            "tile": args.tile,
            "repetitions": args.repetitions,
        },
        "byte_identical_ids": identical,
        "legacy_samples_s": legacy_samples,
        "legacy_median_s": legacy_median,
        "legacy_peak_metal_bytes": legacy_peak,
        "candidate_samples_s": candidate_samples,
        "candidate_median_s": candidate_median,
        "candidate_peak_metal_bytes": candidate_peak,
        "speedup": legacy_median / candidate_median,
        "peak_reduction_bytes": legacy_peak - candidate_peak,
        "classification": (
            "lossless" if identical else "rejected-nonidentical"),
    }
    if args.score_selector:
        heads, head_dim = 32, 128
        q_values = mx.arange(
            args.queries * heads * head_dim,
            dtype=mx.int32).reshape(1, args.queries, heads, head_dim)
        query = ((q_values % 127).astype(mx.float32) / 64 - 1).astype(
            mx.bfloat16)
        k_values = mx.arange(
            args.keys * head_dim, dtype=mx.int32).reshape(
                1, args.keys, head_dim)
        keys = ((k_values % 131).astype(mx.float32) / 66 - 1).astype(
            mx.bfloat16)
        head_values = mx.arange(
            args.queries * heads, dtype=mx.int32).reshape(
                1, args.queries, heads)
        head_weights = (
            (head_values % 29).astype(mx.float32) / 14 - 1)
        mx.eval(query, keys, head_weights)
        offset = args.keys - args.queries
        cfg = SimpleNamespace(index_topk=args.topk)
        state = DSAState(cfg, key_tile_size=args.tile)
        expected_selector = _legacy_selector(
            query, head_weights, keys, offset=offset,
            tile=args.tile, keep=args.topk)
        actual_selector = state._tiled_select(
            query, head_weights, keys, offset=offset)
        mx.eval(expected_selector, actual_selector)
        selector_identical = bool(mx.array_equal(
            expected_selector, actual_selector).item())
        legacy_selector_samples, legacy_selector_peak, expected_selector = (
            _timed(
                lambda: _legacy_selector(
                    query, head_weights, keys, offset=offset,
                    tile=args.tile, keep=args.topk),
                args.repetitions,
            )
        )
        candidate_selector_samples, candidate_selector_peak, actual_selector = (
            _timed(
                lambda: state._tiled_select(
                    query, head_weights, keys, offset=offset),
                args.repetitions,
            )
        )
        selector_identical = selector_identical and bool(mx.array_equal(
            expected_selector, actual_selector).item())
        legacy_selector_median = statistics.median(legacy_selector_samples)
        candidate_selector_median = statistics.median(
            candidate_selector_samples)
        result["selector"] = {
            "byte_identical_ids": selector_identical,
            "legacy_samples_s": legacy_selector_samples,
            "legacy_median_s": legacy_selector_median,
            "legacy_peak_metal_bytes": legacy_selector_peak,
            "candidate_samples_s": candidate_selector_samples,
            "candidate_median_s": candidate_selector_median,
            "candidate_peak_metal_bytes": candidate_selector_peak,
            "speedup": legacy_selector_median / candidate_selector_median,
            "peak_reduction_bytes": (
                legacy_selector_peak - candidate_selector_peak),
        }
        identical = identical and selector_identical
        result["classification"] = (
            "lossless" if identical else "rejected-nonidentical")
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.result_json is not None:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.result_json.with_name(args.result_json.name + ".tmp")
        descriptor = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        try:
            payload = rendered.encode()
            while payload:
                written = os.write(descriptor, payload)
                payload = payload[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, args.result_json)
    print(rendered, end="")
    if not identical:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
