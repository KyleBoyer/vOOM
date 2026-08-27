"""Exact, balanced Qwen4 fused-expert ranges for a second local SSD.

Qwen4-Exp stores all 512 experts of a layer in two multi-gigabyte tensors.
The generic whole-tensor mirror cannot balance a small fast-tier budget across
48 layers. This builder copies complete per-expert gate/up/down byte ranges,
unchanged, and emits the ordinary raw fast-tier map plus a source binding.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import sys
import uuid
from collections import defaultdict
from pathlib import Path

from .kimi_k3_fast_tier import (
    MANIFEST_RESERVE_BYTES,
    MAX_INTERNAL_FAST_TIER_BYTES,
    MIN_INTERNAL_FREE_BYTES,
    _existing_parent,
    _is_internal_root,
    _tree_file_bytes,
)


SCHEMA = "voom.qwen4-fused-expert-fast-tier.v1"
BINDING_NAME = "qwen4_fused_expert_fast_tier.json"
MANIFEST_NAME = "fast_tier_manifest.json"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as source:
        length_raw = source.read(8)
        if len(length_raw) != 8:
            raise EOFError(f"truncated safetensors header: {path}")
        length = struct.unpack("<Q", length_raw)[0]
        raw = source.read(length)
        if len(raw) != length:
            raise EOFError(f"truncated safetensors header: {path}")
    return json.loads(raw), 8 + length


def _source_revision(model_dir: Path) -> str:
    trees = sorted(
        (model_dir / ".cache/huggingface/trees").glob("*.json"))
    return trees[-1].stem if trees else ""


def _catalog(model_dir: Path) -> tuple[list[dict], dict]:
    config_bytes = (model_dir / "config.json").read_bytes()
    raw = json.loads(config_bytes)
    text = raw.get("text_config") or raw
    if raw.get("model_type") != "qwen4_exp":
        raise ValueError("Qwen4 fast tier requires model_type=qwen4_exp")
    hidden = int(text.get("hidden_size", 0))
    width = int(text.get("moe_intermediate_size", 0))
    experts = int(text.get("num_experts", 0))
    layers = int(text.get("num_hidden_layers", 0))
    if min(hidden, width, experts, layers) <= 0:
        raise ValueError("Qwen4 fused-expert geometry is incomplete")

    index_bytes = (model_dir / "model.safetensors.index.json").read_bytes()
    weight_map = json.loads(index_bytes)["weight_map"]
    header_cache: dict[str, tuple[dict, int]] = {}

    def physical(layer: int, suffix: str, shape: tuple[int, ...]):
        raw_name = f"model.language_model.layers.{layer}.mlp.experts.{suffix}"
        shard = weight_map.get(raw_name)
        if shard is None:
            raise ValueError(f"Qwen4 checkpoint lacks {raw_name}")
        if shard not in header_cache:
            header_cache[shard] = _header(model_dir / shard)
        header, payload_base = header_cache[shard]
        entry = header.get(raw_name)
        if not (
            isinstance(entry, dict)
            and entry.get("dtype") == "BF16"
            and tuple(int(value) for value in entry.get("shape", ())) == shape
        ):
            raise ValueError(f"unexpected fused expert metadata: {raw_name}")
        start, end = (int(value) for value in entry["data_offsets"])
        expected = 2
        for value in shape:
            expected *= value
        if end - start != expected:
            raise ValueError(f"unexpected fused expert bytes: {raw_name}")
        return shard, payload_base + start

    matrix_bytes = hidden * width * 2
    groups: list[dict] = []
    for layer in range(layers):
        gate_shard, gate_start = physical(
            layer, "gate_up_proj", (experts, 2 * width, hidden))
        down_shard, down_start = physical(
            layer, "down_proj", (experts, hidden, width))
        for expert in range(experts):
            prefix = f"model.layers.{layer}.mlp.experts.{expert}"
            gate_expert = gate_start + expert * 2 * matrix_bytes
            groups.append({
                "layer": layer,
                "expert": expert,
                "nbytes": 3 * matrix_bytes,
                "entries": (
                    {
                        "name": f"{prefix}.gate_proj.weight",
                        "source_file": gate_shard,
                        "source_offset": gate_expert,
                        "nbytes": matrix_bytes,
                        "dtype": "BF16",
                        "shape": [width, hidden],
                    },
                    {
                        "name": f"{prefix}.up_proj.weight",
                        "source_file": gate_shard,
                        "source_offset": gate_expert + matrix_bytes,
                        "nbytes": matrix_bytes,
                        "dtype": "BF16",
                        "shape": [width, hidden],
                    },
                    {
                        "name": f"{prefix}.down_proj.weight",
                        "source_file": down_shard,
                        "source_offset": down_start + expert * matrix_bytes,
                        "nbytes": matrix_bytes,
                        "dtype": "BF16",
                        "shape": [hidden, width],
                    },
                ),
            })
    return groups, {
        "source_index_sha256": _sha256(index_bytes),
        "source_config_sha256": _sha256(config_bytes),
        "source_revision": _source_revision(model_dir),
        "hidden_size": hidden,
        "expert_width": width,
        "experts": experts,
        "layers": layers,
    }


def _select(groups: list[dict], budget: int, layers: int, experts: int):
    by_pair = {(item["layer"], item["expert"]): item for item in groups}
    selected = []
    used = 0
    # 25 is coprime with 512 and spreads early selections across expert-page
    # batches rather than filling only low-numbered experts.
    step = 25 if experts % 5 else 17
    while __import__("math").gcd(step, experts) != 1:
        step += 2
    for rank in range(experts):
        expert = (rank * step) % experts
        for layer in range(layers):
            item = by_pair[(layer, expert)]
            if used + item["nbytes"] <= budget:
                selected.append(item)
                used += item["nbytes"]
    return selected


def build_qwen4_fast_tier(
    model_dir: str | Path, fast_root: str | Path, *, dry_run: bool = False,
    max_bytes: int = MAX_INTERNAL_FAST_TIER_BYTES,
    min_free_bytes: int = MIN_INTERNAL_FREE_BYTES,
) -> dict:
    model_dir = Path(model_dir).resolve()
    fast_root = Path(fast_root).expanduser().resolve()
    max_bytes = int(max_bytes)
    min_free_bytes = int(min_free_bytes)
    if max_bytes <= 0 or min_free_bytes < 0:
        raise ValueError("invalid fast-tier byte budget")
    internal = _is_internal_root(fast_root)
    if internal and max_bytes > MAX_INTERNAL_FAST_TIER_BYTES:
        raise ValueError("internal fast-tier budget exceeds 90 GB policy")
    target = fast_root / model_dir.name
    other = _tree_file_bytes(fast_root, exclude=target) if internal else 0
    available = max(0, max_bytes - other - MANIFEST_RESERVE_BYTES)
    groups, identity = _catalog(model_dir)
    selected = _select(
        groups, available, identity["layers"], identity["experts"])
    if not selected:
        raise ValueError("no complete Qwen4 expert group fits the budget")
    selected_bytes = sum(item["nbytes"] for item in selected)
    projected_free = (
        shutil.disk_usage(_existing_parent(fast_root)).free
        - selected_bytes - MANIFEST_RESERVE_BYTES)
    if not dry_run and internal and projected_free < min_free_bytes:
        raise ValueError(
            "Qwen4 fast tier would violate the 10 GB free-space floor")
    if not dry_run and target.exists():
        raise FileExistsError(f"{target} already exists")

    by_source: dict[str, list[dict]] = defaultdict(list)
    layer_counts: dict[int, int] = defaultdict(int)
    for group in selected:
        layer_counts[group["layer"]] += 1
        for entry in group["entries"]:
            by_source[entry["source_file"]].append(entry)

    staging = fast_root / f".{model_dir.name}.building-{uuid.uuid4().hex}"
    manifest: dict[str, dict] = {}
    copied = 0
    if not dry_run:
        fast_root.mkdir(parents=True, exist_ok=True)
        staging.mkdir()
    try:
        for file_index, (source_name, entries) in enumerate(
            sorted(by_source.items())):
            if dry_run:
                continue
            destination_name = f"{file_index:06d}.bin"
            destination = staging / destination_name
            entries = sorted(entries, key=lambda item: item["source_offset"])
            with (model_dir / source_name).open("rb") as source, \
                    destination.open("wb") as output:
                output_offset = 0
                for entry in entries:
                    source.seek(entry["source_offset"])
                    remaining = entry["nbytes"]
                    while remaining:
                        chunk = source.read(min(8 * 1024 * 1024, remaining))
                        if not chunk:
                            raise IOError(
                                f"truncated Qwen4 expert read: {entry['name']}")
                        output.write(chunk)
                        remaining -= len(chunk)
                    manifest[entry["name"]] = {
                        "file": destination_name,
                        "offset": output_offset,
                        "nbytes": entry["nbytes"],
                        "dtype": entry["dtype"],
                        "shape": entry["shape"],
                        "source_file": source_name,
                        "source_offset": entry["source_offset"],
                    }
                    output_offset += entry["nbytes"]
                    copied += entry["nbytes"]
            print(
                f"[{file_index + 1}/{len(by_source)}] {source_name}: "
                f"{copied / 1e9:.2f}/{selected_bytes / 1e9:.2f} GB",
                file=sys.stderr, flush=True)

        if not dry_run:
            manifest_bytes = (
                json.dumps(manifest, sort_keys=True, separators=(",", ":"))
                + "\n").encode()
            manifest_path = staging / MANIFEST_NAME
            with manifest_path.open("wb") as output:
                output.write(manifest_bytes)
                output.flush()
                os.fsync(output.fileno())
            binding = {
                "schema": SCHEMA,
                "target_model": model_dir.name,
                **identity,
                "selected_experts": len(selected),
                "selected_tensors": len(manifest),
                "selected_bytes": selected_bytes,
                "fast_manifest_sha256": _sha256(manifest_bytes),
            }
            binding_path = staging / BINDING_NAME
            with binding_path.open("w") as output:
                json.dump(binding, output, sort_keys=True)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            staged = _tree_file_bytes(staging)
            if internal and other + staged > max_bytes:
                raise RuntimeError("built Qwen4 fast tier exceeds global budget")
            os.replace(staging, target)
    except BaseException:
        if not dry_run:
            shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        "schema": SCHEMA,
        "source_revision": identity["source_revision"],
        "candidate_experts": len(groups),
        "candidate_bytes": sum(item["nbytes"] for item in groups),
        "selected_experts": len(selected),
        "selected_tensors": 3 * len(selected),
        "selected_bytes": selected_bytes,
        "selected_experts_per_layer": {
            str(layer): layer_counts.get(layer, 0)
            for layer in range(identity["layers"])
        },
        "other_fast_tier_bytes": other,
        "projected_global_fast_tier_bytes": (
            other + selected_bytes + MANIFEST_RESERVE_BYTES),
        "projected_free_bytes": projected_free,
        "max_bytes": max_bytes,
        "min_free_bytes": min_free_bytes,
        "internal_root": internal,
        "target": str(target),
    }


def validate_qwen4_fast_tier(
    model_dir: str | Path, target: str | Path,
) -> dict:
    """Compare every published fast-tier byte against its bound source."""
    model_dir = Path(model_dir).resolve()
    target = Path(target).expanduser().resolve()
    manifest_bytes = (target / MANIFEST_NAME).read_bytes()
    manifest = json.loads(manifest_bytes)
    binding = json.loads((target / BINDING_NAME).read_text())
    groups, identity = _catalog(model_dir)
    specs = {
        entry["name"]: entry
        for group in groups
        for entry in group["entries"]
    }
    index_bytes = (model_dir / "model.safetensors.index.json").read_bytes()
    config_bytes = (model_dir / "config.json").read_bytes()
    if not (
        binding.get("schema") == SCHEMA
        and binding.get("target_model") == model_dir.name
        and binding.get("source_index_sha256") == _sha256(index_bytes)
        and binding.get("source_config_sha256") == _sha256(config_bytes)
        and binding.get("fast_manifest_sha256") == _sha256(manifest_bytes)
    ):
        raise ValueError("Qwen4 fast-tier binding mismatch")

    checked = 0
    checked_bytes = 0
    for name, entry in sorted(manifest.items()):
        spec = specs.get(name)
        if not (
            spec is not None
            and entry.get("dtype") == spec["dtype"]
            and entry.get("shape") == spec["shape"]
            and int(entry.get("nbytes", -1)) == spec["nbytes"]
            and entry.get("source_file") == spec["source_file"]
            and int(entry.get("source_offset", -1)) == spec["source_offset"]
        ):
            raise ValueError(f"Qwen4 fast-tier metadata mismatch: {name}")
        fast_path = target / entry["file"]
        source_path = model_dir / spec["source_file"]
        fast_fd = os.open(fast_path, os.O_RDONLY)
        source_fd = os.open(source_path, os.O_RDONLY)
        try:
            remaining = spec["nbytes"]
            fast_offset = int(entry["offset"])
            source_offset = spec["source_offset"]
            while remaining:
                length = min(8 * 1024 * 1024, remaining)
                fast_raw = os.pread(fast_fd, length, fast_offset)
                source_raw = os.pread(source_fd, length, source_offset)
                if fast_raw != source_raw or len(fast_raw) != length:
                    raise IOError(f"Qwen4 fast-tier byte mismatch: {name}")
                fast_offset += length
                source_offset += length
                remaining -= length
        finally:
            os.close(fast_fd)
            os.close(source_fd)
        checked += 1
        checked_bytes += spec["nbytes"]
    return {
        "schema": "voom.qwen4-fused-expert-fast-tier-validation.v1",
        "source_revision": identity["source_revision"],
        "checked_tensors": checked,
        "checked_bytes": checked_bytes,
        "verdict": "PASS",
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("--fast-root", default="~/vmodel_fast_tier")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--max-bytes", type=int, default=58_000_000_000)
    parser.add_argument(
        "--min-free-bytes", type=int, default=MIN_INTERNAL_FREE_BYTES)
    args = parser.parse_args()
    report = (
        validate_qwen4_fast_tier(
            args.model_dir,
            Path(args.fast_root).expanduser() / Path(args.model_dir).name)
        if args.validate_only else
        build_qwen4_fast_tier(
            args.model_dir, args.fast_root, dry_run=args.dry_run,
            max_bytes=args.max_bytes, min_free_bytes=args.min_free_bytes)
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
