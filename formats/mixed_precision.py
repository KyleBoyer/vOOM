"""Reproducible component-aware mixed-precision checkpoint plans.

Planning reads only JSON and safetensors headers. It does not load tensors,
allocate Metal arrays, or create an output checkpoint. The companion
``formats.quantize_mlx --precision-plan`` build path validates the complete
plan against the source again before writing any shard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
from dataclasses import asdict, dataclass
from pathlib import Path


PLAN_SCHEMA = "voom.mixed-precision-plan.v1"
MATRIX_SCHEMA = "voom.mixed-precision-matrix.v1"
_DTYPE_BYTES = {
    "BF16": 2, "F16": 2, "F32": 4,
    "I8": 1, "U8": 1, "I16": 2, "U16": 2,
    "I32": 4, "U32": 4, "I64": 8, "U64": 8,
}
_MAIN_LAYER_RE = re.compile(
    r"^(?:model\.language_model|model)\.layers\.(\d+)\.")


@dataclass(frozen=True)
class TensorLayout:
    name: str
    shard: str
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]
    source_bytes: int


@dataclass(frozen=True)
class MixedPrecisionSpec:
    attention: str = "bf16"
    last_bf16_layers: int = 0
    mtp: str = "bf16"
    body: str = "mxfp4"

    def validate(self, total_layers: int) -> None:
        if self.attention not in ("bf16", "mxfp8"):
            raise ValueError("attention precision must be bf16 or mxfp8")
        if self.last_bf16_layers not in (0, 1, 2, 4):
            raise ValueError("last_bf16_layers must be one of 0, 1, 2, 4")
        if self.last_bf16_layers > total_layers:
            raise ValueError("last_bf16_layers exceeds checkpoint depth")
        if self.mtp not in ("bf16", "mxfp8", "mxfp4"):
            raise ValueError("MTP precision must be bf16, mxfp8, or mxfp4")
        if self.body != "mxfp4":
            raise ValueError("the current mixed planner requires body=mxfp4")


def _json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_checkpoint_layout(
    source: str | Path,
) -> tuple[dict[str, TensorLayout], dict]:
    source = Path(source).expanduser().resolve()
    index_path = source / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("checkpoint index has no weight_map")
    else:
        shards = sorted(source.glob("*.safetensors"))
        if len(shards) != 1:
            raise ValueError("checkpoint requires an index or one safetensors shard")
        weight_map = {}

    headers: dict[str, dict] = {}
    header_sizes: dict[str, int] = {}
    shard_sizes: dict[str, int] = {}
    shard_names = sorted(set(weight_map.values())) if weight_map else [
        shards[0].name]
    for shard_name in shard_names:
        shard = source / shard_name
        if not shard.is_file():
            raise FileNotFoundError(f"missing checkpoint shard {shard}")
        with shard.open("rb") as handle:
            raw = handle.read(8)
            if len(raw) != 8:
                raise ValueError(f"invalid safetensors header {shard}")
            header_size = struct.unpack("<Q", raw)[0]
            header = json.loads(handle.read(header_size))
        headers[shard_name] = header
        header_sizes[shard_name] = int(header_size)
        shard_sizes[shard_name] = shard.stat().st_size
        if not weight_map:
            weight_map = {
                name: shard_name for name in header if name != "__metadata__"}

    tensors: dict[str, TensorLayout] = {}
    for name, shard_name in sorted(weight_map.items()):
        meta = headers.get(shard_name, {}).get(name)
        if not isinstance(meta, dict):
            raise ValueError(f"index tensor {name!r} is absent from {shard_name}")
        dtype = str(meta.get("dtype", ""))
        shape = tuple(int(value) for value in meta.get("shape", ()))
        if dtype not in _DTYPE_BYTES or not shape or any(value < 0 for value in shape):
            raise ValueError(f"unsupported tensor layout for {name!r}")
        values = 1
        for dimension in shape:
            values *= dimension
        source_bytes = values * _DTYPE_BYTES[dtype]
        offsets = meta.get("data_offsets")
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(not isinstance(value, int) for value in offsets)
                or offsets[0] < 0 or offsets[1] < offsets[0]
                or offsets[1] - offsets[0] != source_bytes
                or 8 + header_sizes[shard_name] + offsets[1]
                > shard_sizes[shard_name]):
            raise ValueError(f"invalid tensor extent for {name!r}")
        tensors[name] = TensorLayout(
            name=name, shard=shard_name, dtype=dtype, shape=shape,
            data_offsets=(offsets[0], offsets[1]), source_bytes=source_bytes)

    config_path = source / "config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    descriptor = {
        "config_sha256": _sha256_file(config_path) if config_path.is_file() else "",
        "index_sha256": _sha256_file(index_path) if index_path.is_file() else "",
        "shards": shard_sizes,
        "tensors": {
            name: {"shard": info.shard, "dtype": info.dtype,
                   "shape": info.shape, "data_offsets": info.data_offsets,
                   "source_bytes": info.source_bytes}
            for name, info in tensors.items()
        },
    }
    fingerprint = hashlib.sha256(json.dumps(
        descriptor, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    metadata = {
        "source": str(source),
        "layout_fingerprint": fingerprint,
        "config": config,
        "shard_sizes": shard_sizes,
        "total_source_bytes": sum(info.source_bytes for info in tensors.values()),
    }
    return tensors, metadata


def _component(name: str, total_layers: int, last_layers: int) -> tuple[str, int | None]:
    if name.startswith("mtp."):
        return "mtp", None
    match = _MAIN_LAYER_RE.match(name)
    if match:
        layer = int(match.group(1))
        if last_layers and layer >= total_layers - last_layers:
            return "final_layer", layer
        if ".self_attn." in name or ".linear_attn." in name:
            return "attention", layer
        return "body", layer
    if "embed_tokens" in name:
        return "embedding", None
    if "norm" in name:
        return "norm", None
    if name == "lm_head.weight":
        return "lm_head", None
    return "other", None


def _quantizable(info: TensorLayout) -> bool:
    return (
        len(info.shape) == 2 and info.name.endswith(".weight")
        and info.dtype in ("BF16", "F16", "F32")
        and info.shape[-1] % 32 == 0
    )


def _estimated_bytes(info: TensorLayout, storage: str) -> int:
    if storage == "bf16":
        values = 1
        for dimension in info.shape:
            values *= dimension
        return values * 2
    if storage not in ("mxfp4", "mxfp8"):
        return info.source_bytes
    bits = 4 if storage == "mxfp4" else 8
    rows = 1
    for dimension in info.shape[:-1]:
        rows *= dimension
    columns = info.shape[-1]
    return rows * columns * bits // 8 + rows * (columns // 32)


def create_plan(
    source: str | Path, spec: MixedPrecisionSpec,
) -> dict:
    tensors, metadata = inspect_checkpoint_layout(source)
    layer_ids = [
        int(match.group(1)) for name in tensors
        if (match := _MAIN_LAYER_RE.match(name))]
    total_layers = max(layer_ids) + 1 if layer_ids else int(
        metadata["config"].get("num_hidden_layers", 0) or 0)
    if total_layers <= 0:
        raise ValueError("could not determine checkpoint layer count")
    spec.validate(total_layers)

    decisions = {}
    component_bytes: dict[str, dict[str, int]] = {}
    storage_bytes: dict[str, int] = {}
    for name, info in tensors.items():
        component, layer = _component(
            name, total_layers, spec.last_bf16_layers)
        if not _quantizable(info) or component in ("embedding", "norm"):
            storage = "source"
        elif component == "final_layer":
            storage = "bf16"
        elif component == "attention":
            storage = spec.attention
        elif component == "mtp":
            storage = spec.mtp
        else:
            storage = spec.body
        estimated = _estimated_bytes(info, storage)
        decisions[name] = {
            "shard": info.shard,
            "shape": list(info.shape),
            "source_dtype": info.dtype,
            "source_bytes": info.source_bytes,
            "source_data_offsets": list(info.data_offsets),
            "component": component,
            "layer": layer,
            "storage": storage,
            "estimated_bytes": estimated,
        }
        component_entry = component_bytes.setdefault(component, {
            "source_bytes": 0, "estimated_bytes": 0, "tensors": 0})
        component_entry["source_bytes"] += info.source_bytes
        component_entry["estimated_bytes"] += estimated
        component_entry["tensors"] += 1
        storage_bytes[storage] = storage_bytes.get(storage, 0) + estimated

    estimated_total = sum(item["estimated_bytes"] for item in decisions.values())
    source_total = metadata["total_source_bytes"]
    identity = {
        "source_layout_fingerprint": metadata["layout_fingerprint"],
        "spec": asdict(spec),
    }
    plan_id = hashlib.sha256(json.dumps(
        identity, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()[:16]
    plan = {
        "schema": PLAN_SCHEMA,
        "plan_id": plan_id,
        "source": metadata["source"],
        "source_layout_fingerprint": metadata["layout_fingerprint"],
        "total_layers": total_layers,
        "spec": asdict(spec),
        "summary": {
            "tensor_count": len(decisions),
            "source_bytes": source_total,
            "estimated_bytes": estimated_total,
            "estimated_ratio": estimated_total / source_total,
            "component_bytes": component_bytes,
            "storage_bytes": storage_bytes,
        },
        "tensors": decisions,
    }
    plan["plan_digest"] = hashlib.sha256(json.dumps(
        plan, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    return plan


def validate_plan(plan: dict, source: str | Path | None = None) -> dict:
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("unsupported mixed-precision plan schema")
    recorded_digest = plan.get("plan_digest")
    unsigned = dict(plan)
    unsigned.pop("plan_digest", None)
    actual_digest = hashlib.sha256(json.dumps(
        unsigned, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    if recorded_digest != actual_digest:
        raise ValueError("mixed-precision plan digest mismatch")
    source_path = Path(source or plan.get("source", "")).expanduser().resolve()
    spec = MixedPrecisionSpec(**plan.get("spec", {}))
    expected = create_plan(source_path, spec)
    if expected != plan:
        if expected["source_layout_fingerprint"] != plan.get(
                "source_layout_fingerprint"):
            raise ValueError("mixed-precision source layout fingerprint mismatch")
        raise ValueError("mixed-precision plan decisions do not match source/spec")
    return plan


def create_matrix(source: str | Path, plan_dir: str | Path | None = None) -> dict:
    plans = []
    output_dir = Path(plan_dir) if plan_dir is not None else None
    for attention in ("bf16", "mxfp8"):
        for final_layers in (0, 1, 2, 4):
            for mtp in ("bf16", "mxfp8", "mxfp4"):
                plan = create_plan(source, MixedPrecisionSpec(
                    attention=attention,
                    last_bf16_layers=final_layers,
                    mtp=mtp,
                ))
                if output_dir is not None:
                    _json_atomic(output_dir / f"{plan['plan_id']}.json", plan)
                plans.append({
                    "plan_id": plan["plan_id"],
                    "plan_digest": plan["plan_digest"],
                    "spec": plan["spec"],
                    "summary": plan["summary"],
                    "plan_path": (
                        str((output_dir / f"{plan['plan_id']}.json").resolve())
                        if output_dir is not None else None),
                })
    return {
        "schema": MATRIX_SCHEMA,
        "source": str(Path(source).expanduser().resolve()),
        "plans": plans,
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("plan")
    one.add_argument("source", type=Path)
    one.add_argument("output", type=Path)
    one.add_argument("--attention", choices=("bf16", "mxfp8"), default="bf16")
    one.add_argument("--last-bf16-layers", type=int, choices=(0, 1, 2, 4), default=0)
    one.add_argument("--mtp", choices=("bf16", "mxfp8", "mxfp4"), default="bf16")
    matrix = commands.add_parser("matrix")
    matrix.add_argument("source", type=Path)
    matrix.add_argument("output", type=Path)
    matrix.add_argument("--plan-dir", type=Path)
    check = commands.add_parser("validate")
    check.add_argument("source", type=Path)
    check.add_argument("plan", type=Path)
    args = parser.parse_args()
    if args.command == "plan":
        value = create_plan(args.source, MixedPrecisionSpec(
            attention=args.attention,
            last_bf16_layers=args.last_bf16_layers,
            mtp=args.mtp))
        _json_atomic(args.output, value)
    elif args.command == "matrix":
        value = create_matrix(args.source, args.plan_dir)
        _json_atomic(args.output, value)
    else:
        validate_plan(json.loads(args.plan.read_text()), args.source)
        value = {"valid": True, "plan": str(args.plan.resolve())}
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()
