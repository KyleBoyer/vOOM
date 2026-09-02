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
import subprocess
import sys
import uuid
from collections import Counter, defaultdict
from fractions import Fraction
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
TRUNK_FIRST_SCHEMA = "voom.qwen4-trunk-first-fast-tier.v2"
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


def _canonical_name(name: str) -> str:
    if name.startswith("model.language_model."):
        return "model." + name[len("model.language_model."):]
    if name.startswith("language_model.model."):
        return "model." + name[len("language_model.model."):]
    if name.startswith("language_model."):
        return name[len("language_model."):]
    return name


def _trunk_catalog(model_dir: Path, layers: int) -> list[dict]:
    """Catalog exact, always-touched target-body tensors.

    The 51B PLE embedding and ordinary token embedding use dedicated direct-row
    stores, routed experts have their own balanced selector below, and MTP is a
    proposal phase.  Everything else in the released target layers plus its
    final hyper mixer and untied head is prompt-independent trunk traffic.
    """

    index = json.loads(
        (model_dir / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    headers: dict[str, tuple[dict, int]] = {}
    entries = []
    seen = set()
    for physical_name, shard in sorted(weight_map.items()):
        name = _canonical_name(physical_name)
        layer_member = False
        if name.startswith("model.layers."):
            pieces = name.split(".", 3)
            try:
                layer = int(pieces[2])
            except (IndexError, ValueError):
                layer = -1
            layer_member = 0 <= layer < layers
        include = (
            layer_member
            or name.startswith("model.hyper_connection_mixer.")
            or name == "lm_head.weight"
        )
        if not include:
            continue
        if (
            ".mlp.experts." in name
            or ".ple.ple_embedding." in name
        ):
            continue
        if name in seen:
            raise ValueError(f"duplicate canonical Qwen4 trunk tensor {name}")
        if shard not in headers:
            headers[shard] = _header(model_dir / shard)
        header, payload_base = headers[shard]
        metadata = header.get(physical_name)
        if not isinstance(metadata, dict):
            raise ValueError(
                f"Qwen4 trunk tensor {physical_name} missing from {shard}")
        if metadata.get("dtype") != "BF16":
            raise ValueError(
                f"Qwen4 trunk tensor is not released BF16: {physical_name}")
        shape = [int(value) for value in metadata.get("shape", ())]
        start, end = (int(value) for value in metadata["data_offsets"])
        expected = 2
        for value in shape:
            expected *= value
        if end - start != expected:
            raise ValueError(f"unexpected Qwen4 trunk bytes: {physical_name}")
        entries.append({
            "name": name,
            "source_file": shard,
            "source_offset": payload_base + start,
            "nbytes": expected,
            "dtype": "BF16",
            "shape": shape,
            "kind": "target_trunk",
        })
        seen.add(name)
    if not entries or "lm_head.weight" not in seen:
        raise ValueError("Qwen4 trunk catalog is incomplete")
    represented_layers = {
        int(entry["name"].split(".")[2])
        for entry in entries
        if entry["name"].startswith("model.layers.")
    }
    if represented_layers != set(range(layers)):
        raise ValueError("Qwen4 trunk catalog does not cover every target layer")
    return entries


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


def _trace_rankings(
    trace_paths: list[str | Path], *, model_name: str, layers: int,
    experts: int,
) -> tuple[dict[int, list[int]], dict]:
    """Rank experts per layer from an equal-request-weighted trace corpus.

    Every request contributes at most one unit of primary heat per layer,
    divided over its own target sweeps. Request support and raw occurrence
    count break exact heat ties before the content-independent spread order.
    A long-output trace therefore cannot outweigh a genuinely hotter expert
    merely because it contains more decode rounds.
    """
    from runtime.expert_plan import load_trace

    paths = [Path(path).expanduser().resolve() for path in trace_paths]
    if not paths:
        raise ValueError("trace-balanced placement requires at least one trace")
    score: dict[tuple[int, int], Fraction] = defaultdict(Fraction)
    support: Counter[tuple[int, int]] = Counter()
    hits: Counter[tuple[int, int]] = Counter()
    digests = []
    shapes = []
    total_sweeps = 0
    for path in paths:
        raw = path.read_bytes()
        document, sweeps = load_trace(path)
        declared_model = str(document.get("model", "") or "")
        if declared_model and declared_model != model_name:
            raise ValueError(
                f"expert trace model mismatch: {declared_model} != {model_name}")
        if int(document.get("num_experts", 0) or 0) != experts:
            raise ValueError("expert trace num_experts mismatch")
        per_layer_sweeps: Counter[int] = Counter()
        per_request_hits: Counter[tuple[int, int]] = Counter()
        for sweep in sweeps:
            for layer, routed in sweep.items():
                if not 0 <= layer < layers:
                    raise ValueError("expert trace layer exceeds checkpoint")
                per_layer_sweeps[layer] += 1
                per_request_hits.update((layer, expert) for expert in routed)
        if set(per_layer_sweeps) != set(range(layers)):
            raise ValueError("expert trace does not cover every target layer")
        for pair, count in per_request_hits.items():
            layer, _expert = pair
            score[pair] += Fraction(count, per_layer_sweeps[layer])
            support[pair] += 1
            hits[pair] += count
        total_sweeps += len(sweeps)
        digests.append({
            "file": path.name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "sweeps": len(sweeps),
        })
        shape = document.get("request_shape")
        if isinstance(shape, dict):
            shapes.append({
                key: shape.get(key) for key in (
                    "prompt_tokens", "requested_output_tokens",
                    "actual_output_tokens", "system_chars", "tool_count",
                    "message_count", "developer", "streaming",
                    "temperature_class",
                )
            })

    # Reuse the old coprime spread only as the final tie-break for pages that
    # were unseen in the bounded corpus; do not collapse ties toward id zero.
    step = 25 if experts % 5 else 17
    while __import__("math").gcd(step, experts) != 1:
        step += 2
    spread_rank = {(rank * step) % experts: rank for rank in range(experts)}
    rankings = {
        layer: sorted(
            range(experts),
            key=lambda expert: (
                -score[(layer, expert)],
                -support[(layer, expert)],
                -hits[(layer, expert)],
                spread_rank[expert],
            ),
        )
        for layer in range(layers)
    }
    return rankings, {
        "selection_policy": "equal-request-trace-heat-v1",
        "trace_requests": len(paths),
        "trace_sweeps": total_sweeps,
        "trace_documents": digests,
        "trace_request_shapes": shapes,
    }


def _select_trace_balanced(
    groups: list[dict], budget: int, layers: int, experts: int,
    rankings: dict[int, list[int]], hot_experts_per_layer: int = 0,
):
    """Spend the expert budget evenly without overloading the fast device.

    ``hot_experts_per_layer=0`` ranks the complete capacity by heat. A positive
    cap places only that many hot pages per layer, then fills otherwise unused
    storage from the cold end of the ranking. The latter is useful when exact
    always-touched trunk traffic already consumes most of the fast device's
    balanced per-sweep share.
    """
    if not 0 <= hot_experts_per_layer <= experts:
        raise ValueError("trace hot experts per layer must be in [0, experts]")
    by_pair = {(item["layer"], item["expert"]): item for item in groups}
    placement_order = {}
    for layer, ranking in rankings.items():
        if hot_experts_per_layer:
            hot = ranking[:hot_experts_per_layer]
            cold = list(reversed(ranking[hot_experts_per_layer:]))
            placement_order[layer] = [*hot, *cold]
        else:
            placement_order[layer] = ranking
    selected = []
    used = 0
    for rank in range(experts):
        for layer in range(layers):
            item = by_pair[(layer, placement_order[layer][rank])]
            if used + item["nbytes"] <= budget:
                selected.append(item)
                used += item["nbytes"]
    return selected


def build_qwen4_fast_tier(
    model_dir: str | Path, fast_root: str | Path, *, dry_run: bool = False,
    max_bytes: int = MAX_INTERNAL_FAST_TIER_BYTES,
    min_free_bytes: int = MIN_INTERNAL_FREE_BYTES,
    placement: str = "experts",
    trace_paths: list[str | Path] | None = None,
    trace_hot_experts_per_layer: int = 0,
    candidate_max_bytes: int = 0,
    target_name: str = "",
) -> dict:
    model_dir = Path(model_dir).resolve()
    fast_root = Path(fast_root).expanduser().resolve()
    max_bytes = int(max_bytes)
    min_free_bytes = int(min_free_bytes)
    candidate_max_bytes = int(candidate_max_bytes)
    trace_hot_experts_per_layer = int(trace_hot_experts_per_layer)
    if (max_bytes <= 0 or min_free_bytes < 0 or candidate_max_bytes < 0
            or trace_hot_experts_per_layer < 0):
        raise ValueError("invalid fast-tier byte budget")
    if placement not in ("experts", "trunk-first"):
        raise ValueError("Qwen4 fast-tier placement must be experts or trunk-first")
    internal = _is_internal_root(fast_root)
    if internal and max_bytes > MAX_INTERNAL_FAST_TIER_BYTES:
        raise ValueError("internal fast-tier budget exceeds 90 GB policy")
    target_name = str(target_name or model_dir.name)
    if (not target_name or target_name in (".", "..")
            or Path(target_name).name != target_name):
        raise ValueError(
            "Qwen4 fast-tier target name must be one path component")
    target = fast_root / target_name
    reclaimable_target_bytes = _tree_file_bytes(target) if target.exists() else 0
    other = _tree_file_bytes(fast_root, exclude=target) if internal else 0
    available = max(0, max_bytes - other - MANIFEST_RESERVE_BYTES)
    if candidate_max_bytes:
        available = min(available, candidate_max_bytes)
    groups, identity = _catalog(model_dir)
    trunk = (
        _trunk_catalog(model_dir, identity["layers"])
        if placement == "trunk-first" else []
    )
    trunk_bytes = sum(item["nbytes"] for item in trunk)
    if trunk_bytes > available:
        raise ValueError(
            "complete Qwen4 target trunk does not fit the fast-tier budget")
    trace_metadata = {"selection_policy": "uniform-coprime-v1"}
    if trace_paths:
        rankings, trace_metadata = _trace_rankings(
            list(trace_paths), model_name=model_dir.name,
            layers=identity["layers"], experts=identity["experts"])
        selected = _select_trace_balanced(
            groups, available - trunk_bytes,
            identity["layers"], identity["experts"], rankings,
            trace_hot_experts_per_layer)
        trace_metadata["trace_hot_experts_per_layer"] = (
            trace_hot_experts_per_layer)
    else:
        selected = _select(
            groups, available - trunk_bytes,
            identity["layers"], identity["experts"])
    if not selected:
        raise ValueError("no complete Qwen4 expert group fits the budget")
    selected_bytes = trunk_bytes + sum(item["nbytes"] for item in selected)
    projected_free = (
        shutil.disk_usage(_existing_parent(fast_root)).free
        + reclaimable_target_bytes
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
    for entry in trunk:
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
                "schema": (
                    TRUNK_FIRST_SCHEMA
                    if placement == "trunk-first" else SCHEMA),
                "target_model": model_dir.name,
                **identity,
                "placement": placement,
                **trace_metadata,
                "selected_trunk_tensors": len(trunk),
                "selected_trunk_bytes": trunk_bytes,
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
        "schema": (
            TRUNK_FIRST_SCHEMA if placement == "trunk-first" else SCHEMA),
        "placement": placement,
        **trace_metadata,
        "source_revision": identity["source_revision"],
        "candidate_experts": len(groups),
        "candidate_bytes": sum(item["nbytes"] for item in groups),
        "selected_experts": len(selected),
        "selected_trunk_tensors": len(trunk),
        "selected_trunk_bytes": trunk_bytes,
        "selected_tensors": len(trunk) + 3 * len(selected),
        "selected_bytes": selected_bytes,
        "selected_experts_per_layer": {
            str(layer): layer_counts.get(layer, 0)
            for layer in range(identity["layers"])
        },
        "other_fast_tier_bytes": other,
        "reclaimable_target_bytes": reclaimable_target_bytes,
        "projected_global_fast_tier_bytes": (
            other + selected_bytes + MANIFEST_RESERVE_BYTES),
        "projected_free_bytes": projected_free,
        "max_bytes": max_bytes,
        "candidate_max_bytes": candidate_max_bytes,
        "min_free_bytes": min_free_bytes,
        "internal_root": internal,
        "target": str(target),
    }


def _clone_tree(source: Path, destination: Path) -> None:
    """Create an APFS copy-on-write clone; never silently full-copy."""
    subprocess.run(
        ["cp", "-cR", str(source), str(destination)], check=True)


def build_qwen4_fast_tier_clone(
    model_dir: str | Path,
    source_model_dir: str | Path,
    source_tier: str | Path,
    fast_root: str | Path,
    *,
    target_name: str,
    max_bytes: int = MAX_INTERNAL_FAST_TIER_BYTES,
    min_free_bytes: int = MIN_INTERNAL_FREE_BYTES,
) -> dict:
    """Clone a validated tier and rewrite only candidate-different extents.

    Qwen4 ablation checkpoints commonly keep topology, gates/up projections,
    and most trunk tensors byte-identical while editing residual writers and
    expert down projections. APFS clonefile lets those identical fast-tier
    blocks share physical storage. Every manifest extent is then compared to
    the candidate checkpoint; differing chunks are rewritten into the clone,
    and the entire result is validated against the candidate before publish.
    """
    model_dir = Path(model_dir).expanduser().resolve()
    source_model_dir = Path(source_model_dir).expanduser().resolve()
    source_tier = Path(source_tier).expanduser().resolve()
    fast_root = Path(fast_root).expanduser().resolve()
    target_name = str(target_name)
    max_bytes = int(max_bytes)
    min_free_bytes = int(min_free_bytes)
    if (not target_name or target_name in (".", "..")
            or Path(target_name).name != target_name):
        raise ValueError("Qwen4 cloned tier target must be one path component")
    if max_bytes <= 0 or min_free_bytes < 0:
        raise ValueError("invalid cloned fast-tier storage policy")
    if not _is_internal_root(fast_root):
        raise ValueError("Qwen4 copy-on-write tier requires the internal APFS root")
    if max_bytes > MAX_INTERNAL_FAST_TIER_BYTES:
        raise ValueError("internal fast-tier budget exceeds 90 GB policy")
    target = fast_root / target_name
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"cloned Qwen4 fast tier already exists: {target}")
    if source_tier.parent != fast_root:
        raise ValueError(
            "Qwen4 clone source must be a direct child of the fast-tier root")
    if source_tier.stat().st_dev != _existing_parent(fast_root).stat().st_dev:
        raise ValueError("Qwen4 clone source and destination must share a volume")

    source_validation = validate_qwen4_fast_tier(
        source_model_dir, source_tier)
    source_manifest_bytes = (source_tier / MANIFEST_NAME).read_bytes()
    manifest = json.loads(source_manifest_bytes)
    source_binding = json.loads((source_tier / BINDING_NAME).read_text())
    groups, identity = _catalog(model_dir)
    schema = source_binding.get("schema")
    placement = source_binding.get("placement", "experts")
    if schema == TRUNK_FIRST_SCHEMA and placement == "trunk-first":
        trunk = _trunk_catalog(model_dir, identity["layers"])
    elif schema == SCHEMA and placement == "experts":
        trunk = []
    else:
        raise ValueError("Qwen4 cloned tier source schema is unsupported")
    specs = {
        entry["name"]: entry
        for group in groups
        for entry in group["entries"]
    }
    specs.update({entry["name"]: entry for entry in trunk})
    for name, entry in manifest.items():
        spec = specs.get(name)
        if not (
            spec is not None
            and entry.get("dtype") == spec["dtype"]
            and entry.get("shape") == spec["shape"]
            and int(entry.get("nbytes", -1)) == spec["nbytes"]
            and entry.get("source_file") == spec["source_file"]
            and int(entry.get("source_offset", -1)) == spec["source_offset"]
        ):
            raise ValueError(
                f"candidate topology differs at cloned tier tensor {name}")

    source_tree_bytes = _tree_file_bytes(source_tier)
    # ``other`` already includes the source tier. Add one more logical copy
    # for the candidate: clone sharing saves physical blocks, but the global
    # policy intentionally accounts the complete addressable tier size.
    other = _tree_file_bytes(fast_root)
    projected_global = other + source_tree_bytes
    if projected_global > max_bytes:
        raise ValueError("cloned Qwen4 tier exceeds the global 90 GB policy")
    free_before = shutil.disk_usage(_existing_parent(fast_root)).free
    if free_before - source_tree_bytes < min_free_bytes:
        raise ValueError(
            "cloned Qwen4 tier lacks a full-copy fallback safety margin")

    staging = fast_root / f".{target_name}.building-{uuid.uuid4().hex}"
    changed_tensors = 0
    rewritten_bytes = 0
    modified_files = set()
    try:
        fast_root.mkdir(parents=True, exist_ok=True)
        _clone_tree(source_tier, staging)
        for name, entry in sorted(manifest.items()):
            spec = specs[name]
            fast_path = staging / entry["file"]
            source_path = model_dir / spec["source_file"]
            fast_fd = os.open(fast_path, os.O_RDWR)
            source_fd = os.open(source_path, os.O_RDONLY)
            tensor_changed = False
            try:
                remaining = spec["nbytes"]
                fast_offset = int(entry["offset"])
                source_offset = int(spec["source_offset"])
                while remaining:
                    length = min(8 * 1024 * 1024, remaining)
                    old = os.pread(fast_fd, length, fast_offset)
                    new = os.pread(source_fd, length, source_offset)
                    if len(old) != length or len(new) != length:
                        raise IOError(f"truncated cloned tier extent: {name}")
                    if old != new:
                        written = os.pwrite(fast_fd, new, fast_offset)
                        if written != length:
                            raise IOError(f"short cloned tier rewrite: {name}")
                        rewritten_bytes += length
                        tensor_changed = True
                    fast_offset += length
                    source_offset += length
                    remaining -= length
                if tensor_changed:
                    os.fsync(fast_fd)
                    modified_files.add(entry["file"])
                    changed_tensors += 1
            finally:
                os.close(fast_fd)
                os.close(source_fd)

        binding = dict(source_binding)
        binding.update(identity)
        binding["target_model"] = model_dir.name
        binding["fast_manifest_sha256"] = _sha256(source_manifest_bytes)
        binding["clone_source_model"] = source_model_dir.name
        binding["clone_source_revision"] = source_validation[
            "source_revision"]
        binding["clone_rewritten_tensors"] = changed_tensors
        binding["clone_rewritten_bytes"] = rewritten_bytes
        binding_path = staging / BINDING_NAME
        temporary = binding_path.with_name(binding_path.name + ".tmp")
        with temporary.open("w") as output:
            json.dump(binding, output, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, binding_path)

        candidate_validation = validate_qwen4_fast_tier(model_dir, staging)
        free_after = shutil.disk_usage(_existing_parent(fast_root)).free
        staged_logical = _tree_file_bytes(staging)
        if free_after < min_free_bytes:
            raise RuntimeError(
                "cloned Qwen4 tier crossed the 10 GB actual-free floor")
        if other + staged_logical > max_bytes:
            raise RuntimeError("cloned Qwen4 tier crossed the global budget")
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        "schema": "voom.qwen4-fast-tier-clone.v1",
        "target": str(target),
        "target_model": model_dir.name,
        "source_tier": str(source_tier),
        "source_revision": source_validation["source_revision"],
        "candidate_revision": identity["source_revision"],
        "logical_bytes": _tree_file_bytes(target),
        "changed_tensors": changed_tensors,
        "rewritten_bytes": rewritten_bytes,
        "modified_files": len(modified_files),
        "free_before_bytes": free_before,
        "free_after_bytes": free_after,
        "projected_global_fast_tier_bytes": projected_global,
        "source_validation": source_validation["verdict"],
        "candidate_validation": candidate_validation["verdict"],
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
    schema = binding.get("schema")
    placement = binding.get("placement", "experts")
    if schema == TRUNK_FIRST_SCHEMA and placement == "trunk-first":
        trunk = _trunk_catalog(model_dir, identity["layers"])
    elif schema == SCHEMA and placement == "experts":
        trunk = []
    else:
        raise ValueError("Qwen4 fast-tier schema/placement mismatch")
    specs = {
        entry["name"]: entry
        for group in groups
        for entry in group["entries"]
    }
    specs.update({entry["name"]: entry for entry in trunk})
    index_bytes = (model_dir / "model.safetensors.index.json").read_bytes()
    config_bytes = (model_dir / "config.json").read_bytes()
    if not (
        binding.get("schema") in (SCHEMA, TRUNK_FIRST_SCHEMA)
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
        "schema": "voom.qwen4-fast-tier-validation.v2",
        "placement": placement,
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
    parser.add_argument(
        "--clone-from-tier", default="",
        help="APFS-clone this validated Qwen4 tier and patch candidate bytes")
    parser.add_argument(
        "--clone-source-model", default="",
        help="released model directory that binds --clone-from-tier")
    parser.add_argument(
        "--placement", choices=("experts", "trunk-first"),
        default="experts")
    parser.add_argument("--max-bytes", type=int, default=58_000_000_000)
    parser.add_argument(
        "--min-free-bytes", type=int, default=MIN_INTERNAL_FREE_BYTES)
    parser.add_argument(
        "--trace", action="append", default=[],
        help="decode-only expert trace; repeat for an equal-weighted corpus")
    parser.add_argument(
        "--trace-hot-experts-per-layer", type=int, default=0,
        help="rank only this many hot pages per layer, then fill from cold")
    parser.add_argument(
        "--candidate-max-bytes", type=int, default=0,
        help="cap this candidate while retaining the global --max-bytes cap")
    parser.add_argument(
        "--target-name", default="",
        help="publish under a distinct one-component candidate directory")
    args = parser.parse_args()
    target_name = args.target_name or Path(args.model_dir).name
    if bool(args.clone_from_tier) != bool(args.clone_source_model):
        parser.error(
            "--clone-from-tier and --clone-source-model require each other")
    if args.validate_only:
        report = validate_qwen4_fast_tier(
            args.model_dir,
            Path(args.fast_root).expanduser() / target_name)
    elif args.clone_from_tier:
        if args.dry_run or args.trace or args.candidate_max_bytes:
            parser.error("clone mode does not accept build-selection options")
        report = build_qwen4_fast_tier_clone(
            args.model_dir, args.clone_source_model,
            args.clone_from_tier, args.fast_root,
            target_name=target_name, max_bytes=args.max_bytes,
            min_free_bytes=args.min_free_bytes)
    else:
        report = build_qwen4_fast_tier(
            args.model_dir, args.fast_root, dry_run=args.dry_run,
            max_bytes=args.max_bytes, min_free_bytes=args.min_free_bytes,
            placement=args.placement, trace_paths=args.trace,
            trace_hot_experts_per_layer=(
                args.trace_hot_experts_per_layer),
            candidate_max_bytes=args.candidate_max_bytes,
            target_name=args.target_name)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
