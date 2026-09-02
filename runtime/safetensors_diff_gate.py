"""Bounded-memory, byte-exact checkpoint tensor-difference gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import time
from typing import Any


SCHEMA = "voom.safetensors-diff-gate.v1"


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def read_header(path: Path) -> tuple[int, dict[str, dict[str, Any]]]:
    """Return payload start and validated tensor metadata."""
    file_size = path.stat().st_size
    with path.open("rb", buffering=0) as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"truncated safetensors prefix: {path}")
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size <= 0 or header_size > min(file_size - 8, 512_000_000):
            raise ValueError(f"invalid safetensors header size: {path}")
        raw_header = stream.read(header_size)
        if len(raw_header) != header_size:
            raise ValueError(f"truncated safetensors header: {path}")
    decoded = json.loads(raw_header)
    payload_start = 8 + header_size
    tensors: dict[str, dict[str, Any]] = {}
    intervals = []
    for name, metadata in decoded.items():
        if name == "__metadata__":
            continue
        if not isinstance(metadata, dict):
            raise ValueError(f"invalid tensor metadata for {name}: {path}")
        offsets = metadata.get("data_offsets")
        shape = metadata.get("shape")
        dtype = metadata.get("dtype")
        if (not isinstance(offsets, list) or len(offsets) != 2
                or not all(isinstance(value, int) for value in offsets)
                or offsets[0] < 0 or offsets[1] < offsets[0]
                or payload_start + offsets[1] > file_size
                or not isinstance(shape, list)
                or not all(isinstance(value, int) and value >= 0
                           for value in shape)
                or not isinstance(dtype, str)):
            raise ValueError(f"invalid tensor range for {name}: {path}")
        record = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": offsets,
            "bytes": offsets[1] - offsets[0],
        }
        tensors[name] = record
        intervals.append((offsets[0], offsets[1], name))
    intervals.sort()
    for previous, current in zip(intervals, intervals[1:]):
        if previous[1] > current[0]:
            raise ValueError(
                f"overlapping tensor payloads {previous[2]} and {current[2]}: "
                f"{path}")
    return payload_start, tensors


def _compare_payload(
    base_stream, candidate_stream, *, base_offset: int,
    candidate_offset: int, size: int, chunk_bytes: int,
) -> tuple[bool, str, str]:
    base_stream.seek(base_offset)
    candidate_stream.seek(candidate_offset)
    base_digest = hashlib.sha256()
    candidate_digest = hashlib.sha256()
    equal = True
    remaining = size
    while remaining:
        count = min(chunk_bytes, remaining)
        base_block = base_stream.read(count)
        candidate_block = candidate_stream.read(count)
        if len(base_block) != count or len(candidate_block) != count:
            raise ValueError("truncated tensor payload during comparison")
        base_digest.update(base_block)
        candidate_digest.update(candidate_block)
        equal = equal and base_block == candidate_block
        remaining -= count
    return equal, base_digest.hexdigest(), candidate_digest.hexdigest()


def _load_index(directory: Path) -> dict[str, str]:
    payload = json.loads(
        (directory / "model.safetensors.index.json").read_text())
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"invalid safetensors index: {directory}")
    return {str(name): str(shard) for name, shard in weight_map.items()}


def compare_checkpoints(
    base: Path, candidate: Path, *, expected_names: set[str] | None = None,
    allowed_patterns: tuple[re.Pattern[str], ...] = (),
    chunk_bytes: int = 8 * 1024 * 1024,
) -> dict[str, Any]:
    base = base.resolve()
    candidate = candidate.resolve()
    base_map = _load_index(base)
    candidate_map = _load_index(candidate)
    structural_failures = []
    if base_map != candidate_map:
        structural_failures.append("tensor-to-shard maps differ")
    shard_names = sorted(set(base_map.values()) | set(candidate_map.values()))
    changed = []
    tensor_count = 0
    tensor_bytes = 0
    compared_bytes = 0
    linked_shards = 0
    scanned_shards = 0
    for shard in shard_names:
        base_path = base / shard
        candidate_path = candidate / shard
        if not base_path.is_file() or not candidate_path.is_file():
            structural_failures.append(f"missing shard: {shard}")
            continue
        base_start, base_tensors = read_header(base_path)
        candidate_start, candidate_tensors = read_header(candidate_path)
        if set(base_tensors) != set(candidate_tensors):
            structural_failures.append(f"tensor names differ in {shard}")
            continue
        same_file = os.path.samefile(base_path, candidate_path)
        if same_file:
            linked_shards += 1
        else:
            scanned_shards += 1
        with base_path.open("rb", buffering=0) as base_stream, \
                candidate_path.open("rb", buffering=0) as candidate_stream:
            for name in sorted(base_tensors):
                base_record = base_tensors[name]
                candidate_record = candidate_tensors[name]
                tensor_count += 1
                tensor_bytes += int(base_record["bytes"])
                if (base_record["dtype"] != candidate_record["dtype"]
                        or base_record["shape"] != candidate_record["shape"]
                        or base_record["bytes"] != candidate_record["bytes"]):
                    structural_failures.append(
                        f"tensor metadata differs: {name}")
                    continue
                if same_file:
                    continue
                size = int(base_record["bytes"])
                equal, base_sha, candidate_sha = _compare_payload(
                    base_stream,
                    candidate_stream,
                    base_offset=base_start + base_record["data_offsets"][0],
                    candidate_offset=(
                        candidate_start
                        + candidate_record["data_offsets"][0]),
                    size=size,
                    chunk_bytes=chunk_bytes,
                )
                compared_bytes += size
                if not equal:
                    changed.append({
                        "tensor": name,
                        "shard": shard,
                        "dtype": base_record["dtype"],
                        "shape": base_record["shape"],
                        "bytes": size,
                        "base_sha256": base_sha,
                        "candidate_sha256": candidate_sha,
                    })

    changed_names = {item["tensor"] for item in changed}
    unexpected = sorted(
        name for name in changed_names
        if allowed_patterns and not any(
            pattern.fullmatch(name) for pattern in allowed_patterns))
    missing_expected = sorted(
        (expected_names or set()) - changed_names)
    extra_vs_expected = sorted(
        changed_names - expected_names) if expected_names is not None else []
    passed = not (
        structural_failures or unexpected or missing_expected
        or extra_vs_expected)
    return {
        "schema": SCHEMA,
        "base": str(base),
        "candidate": str(candidate),
        "passed": passed,
        "structural_failures": structural_failures,
        "tensor_count": tensor_count,
        "tensor_bytes": tensor_bytes,
        "payload_bytes_compared": compared_bytes,
        "linked_shards_skipped": linked_shards,
        "changed_shards_scanned": scanned_shards,
        "changed_tensor_count": len(changed),
        "changed_tensor_bytes": sum(item["bytes"] for item in changed),
        "changed_tensors": changed,
        "unexpected_changed_tensors": unexpected,
        "missing_expected_tensors": missing_expected,
        "extra_vs_expected_tensors": extra_vs_expected,
    }


def _expected_names(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text())
    stats = payload.get("stats")
    if not isinstance(stats, list):
        raise ValueError("expected metadata has no stats list")
    names = {str(item["tensor"]) for item in stats}
    if len(names) != len(stats):
        raise ValueError("expected metadata contains duplicate tensor names")
    claimed = payload.get("n_edited")
    if claimed is not None and int(claimed) != len(names):
        raise ValueError("expected metadata n_edited does not match stats")
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--expected-metadata", type=Path)
    parser.add_argument("--allow-regex", action="append", default=[])
    parser.add_argument("--chunk-mb", type=int, default=8)
    parser.add_argument("--result-json", type=Path, required=True)
    args = parser.parse_args()
    if args.chunk_mb <= 0 or args.chunk_mb > 256:
        parser.error("--chunk-mb must be in [1, 256]")
    if args.result_json.exists():
        parser.error("result JSON already exists")
    patterns = tuple(re.compile(value) for value in args.allow_regex)
    started = time.perf_counter()
    result = compare_checkpoints(
        args.base, args.candidate,
        expected_names=_expected_names(args.expected_metadata),
        allowed_patterns=patterns,
        chunk_bytes=args.chunk_mb * 1024 * 1024,
    )
    result["wall_seconds"] = time.perf_counter() - started
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.result_json, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
