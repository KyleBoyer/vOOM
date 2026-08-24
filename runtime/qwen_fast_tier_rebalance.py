#!/usr/bin/env python3
"""Rebalance an exact Qwen raw fast tier by measured device service rates.

No tensor file is rewritten or deleted. The candidate manifest simply stops
shadowing selected tensors, causing WeightStore to read their identical source
bytes from the external checkpoint. This is useful when the internal tier is
the measured critical path despite a nominally even byte split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
from pathlib import Path


LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(name: str) -> str:
    if name.startswith("model.language_model."):
        return "model." + name[len("model.language_model."):]
    if name.startswith("language_model.model."):
        return "model." + name[len("language_model.model."):]
    if name.startswith("language_model."):
        return name[len("language_model."):]
    return name


def _source_layer_sizes(model_dir: Path) -> dict[str, int]:
    index = json.loads(
        (model_dir / "model.safetensors.index.json").read_text())
    layouts: dict[str, dict] = {}
    for shard in sorted(set(index["weight_map"].values())):
        with (model_dir / shard).open("rb") as source:
            length_raw = source.read(8)
            if len(length_raw) != 8:
                raise IOError(f"truncated safetensors shard: {shard}")
            length = struct.unpack("<Q", length_raw)[0]
            layouts[shard] = json.loads(source.read(length))
    totals: dict[str, int] = {}
    for physical, shard in index["weight_map"].items():
        name = _canonical(physical)
        match = LAYER_RE.match(name)
        if match is None:
            continue
        start, end = layouts[shard][physical]["data_offsets"]
        layer = match.group(1)
        totals[layer] = totals.get(layer, 0) + int(end) - int(start)
    return totals


def rebalance_manifest(
    manifest: dict[str, dict], source_layer_sizes: dict[str, int],
    target_fast_fraction: float,
) -> tuple[dict[str, dict], dict]:
    if not 0 < target_fast_fraction < 1:
        raise ValueError("target_fast_fraction must be strictly between 0 and 1")
    output = dict(manifest)
    total_source = sum(source_layer_sizes.values())
    target = round(total_source * target_fast_fraction)

    def selected_by_layer():
        values: dict[str, list[tuple[str, dict]]] = {}
        for name, entry in output.items():
            match = LAYER_RE.match(name)
            if match is not None:
                values.setdefault(match.group(1), []).append((name, entry))
        return values

    removed = []
    while True:
        by_layer = selected_by_layer()
        current = sum(
            int(entry["nbytes"])
            for values in by_layer.values() for _name, entry in values)
        current_error = abs(current - target)
        choices = []
        for layer, values in by_layer.items():
            # Tiny norms/scales do not meaningfully balance a device, and each
            # layer must retain at least one substantial internal read so its
            # two-device fetch remains overlap-capable.
            substantial = [
                (name, entry) for name, entry in values
                if int(entry["nbytes"]) >= 1_000_000]
            if len(substantial) <= 1:
                continue
            name, entry = max(
                substantial, key=lambda item: int(item[1]["nbytes"]))
            size = int(entry["nbytes"])
            next_error = abs((current - size) - target)
            ratio = current_layer = sum(
                int(value["nbytes"]) for _key, value in values)
            ratio = current_layer / source_layer_sizes[layer]
            choices.append((next_error, -ratio, int(layer), name, size))
        if not choices:
            break
        next_error, _negative_ratio, _layer, name, size = min(choices)
        if next_error >= current_error:
            break
        del output[name]
        removed.append((name, size))

    selected = sum(
        int(entry["nbytes"])
        for name, entry in output.items() if LAYER_RE.match(name))
    return output, {
        "source_layer_bytes": total_source,
        "original_selected_layer_bytes": sum(
            int(entry["nbytes"])
            for name, entry in manifest.items() if LAYER_RE.match(name)),
        "candidate_selected_layer_bytes": selected,
        "candidate_fast_fraction": selected / total_source,
        "target_fast_fraction": target_fast_fraction,
        "removed_tensors": len(removed),
        "removed_bytes": sum(size for _name, size in removed),
        "removed_names": [name for name, _size in removed],
    }


def plan(
    model_dir: Path, fast_dir: Path, *, target_fast_fraction: float,
    publish: bool = False,
) -> dict:
    model_dir = model_dir.expanduser().resolve()
    fast_dir = fast_dir.expanduser().resolve()
    active = fast_dir / "fast_tier_manifest.json"
    raw = active.read_bytes()
    manifest = json.loads(raw)
    source_sizes = _source_layer_sizes(model_dir)
    candidate, report = rebalance_manifest(
        manifest, source_sizes, target_fast_fraction)
    candidate_raw = json.dumps(
        candidate, separators=(",", ":"), sort_keys=True).encode()
    candidate_path = fast_dir / "fast_tier_manifest.rebalanced.json"
    temporary = candidate_path.with_suffix(candidate_path.suffix + ".tmp")
    temporary.write_bytes(candidate_raw)
    os.replace(temporary, candidate_path)
    backup = fast_dir / "fast_tier_manifest.pre-freetoken-rebalance.json"
    if publish:
        if backup.exists() and backup.read_bytes() != raw:
            raise RuntimeError("existing rebalance backup differs from active manifest")
        if not backup.exists():
            backup.write_bytes(raw)
        active_tmp = active.with_suffix(active.suffix + ".tmp")
        active_tmp.write_bytes(candidate_raw)
        os.replace(active_tmp, active)
    return {
        "schema": "voom.qwen-fast-tier-rebalance.v1",
        "model_dir": str(model_dir),
        "fast_dir": str(fast_dir),
        "published": publish,
        "active_manifest_sha256_before": _sha256_bytes(raw),
        "candidate_manifest_sha256": _sha256_bytes(candidate_raw),
        "candidate_manifest": str(candidate_path),
        "backup_manifest": str(backup),
        **report,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("fast_dir", type=Path)
    parser.add_argument("--target-fast-fraction", type=float, required=True)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    result = plan(
        args.model_dir, args.fast_dir,
        target_fast_fraction=args.target_fast_fraction,
        publish=args.publish,
    )
    if args.result:
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
