"""Requantize a standard-MLX affine proposal checkpoint without BF16 source.

This utility is deliberately limited to proposal-only ``all-draft`` artifacts.
It dequantizes each existing affine matrix and immediately requantizes it at a
smaller affine width.  The extra approximation is safe only because the draft
never owns an emitted token: an authoritative target must verify every
proposal.  Serving checkpoints are rejected fail-closed.

The builder is useful on storage-constrained hosts where retaining the released
BF16 draft and a second derivative at the same time would violate a free-space
floor.  It is *not* a substitute for source-BF16 quantization when measuring
standalone model quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
from pathlib import Path

import mlx.core as mx


_METADATA_FILES = {
    "added_tokens.json",
    "chat_template.json",
    "chat_template.jinja",
    "configuration.json",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _header(path: Path) -> dict:
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError(f"truncated safetensors prefix: {path}")
        size = struct.unpack("<Q", raw)[0]
        if size <= 0 or size > 256 << 20:
            raise ValueError(f"unsafe safetensors header size {size}: {path}")
        encoded = handle.read(size)
    try:
        result = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid safetensors header: {path}") from error
    if not isinstance(result, dict):
        raise ValueError(f"invalid safetensors header mapping: {path}")
    return result


def _source_shard(source: Path, index: dict) -> Path:
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("source index has no weight_map")
    shard_names = set(weight_map.values())
    if len(shard_names) != 1:
        raise ValueError(
            "bounded draft requantizer requires exactly one source shard")
    name = next(iter(shard_names))
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError(f"unsafe source shard name: {name!r}")
    shard = source / name
    if not shard.is_file():
        raise FileNotFoundError(shard)
    return shard


def _validated_source(source: Path) -> tuple[dict, dict, Path, set[str]]:
    config_path = source / "config.json"
    index_path = source / "model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        raise FileNotFoundError(
            "draft requantization requires config.json and safetensors index")
    config = json.loads(config_path.read_text())
    index = json.loads(index_path.read_text())
    quant = config.get("quantization_config") or config.get("quantization")
    if quant != {"bits": 4, "group_size": 64, "mode": "affine"}:
        raise ValueError(
            "source must be a uniform affine4/group64 MLX artifact")
    provenance = config.get("voom_quantization")
    if not isinstance(provenance, dict) or provenance.get("profile") != "all-draft":
        raise ValueError(
            "requantization is restricted to proposal-only all-draft artifacts")
    model_type = (config.get("text_config") or config).get("model_type")
    if model_type not in ("qwen3_5", "qwen3_5_text"):
        raise ValueError("draft requantizer currently requires dense Qwen3.5")
    shard = _source_shard(source, index)
    header_names = set(_header(shard)) - {"__metadata__"}
    index_names = set(index["weight_map"])
    if header_names != index_names:
        raise ValueError("source index/header tensor names differ")
    stems = {
        name[:-len(".weight")]
        for name in header_names
        if name.endswith(".weight")
        and name[:-len(".weight")] + ".scales" in header_names
    }
    if not stems:
        raise ValueError("source contains no standard-MLX quantized matrices")
    for stem in stems:
        if stem + ".biases" not in header_names:
            raise ValueError(f"affine source has no biases tensor: {stem}")
    if int(provenance.get("quantized_tensors", -1)) != len(stems):
        raise ValueError("source quantized tensor count differs from header")
    return config, index, shard, stems


def requantize_affine_draft(
    source: str | Path,
    output: str | Path,
    *,
    bits: int = 2,
    group_size: int = 64,
) -> Path:
    """Build an atomic affine2/3 proposal derivative from affine4 input."""
    source = Path(source).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if bits not in (2, 3):
        raise ValueError("draft requantization bits must be 2 or 3")
    if group_size != 64:
        raise ValueError("draft requantization group_size must be 64")
    if source == output:
        raise ValueError("source and output must differ")
    if output.exists():
        raise FileExistsError(output)

    config, index, source_shard, stems = _validated_source(source)
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    build = parent / f".{output.name}.build-{os.getpid()}"
    if build.exists():
        raise FileExistsError(build)
    build.mkdir()
    try:
        lazy = dict(mx.load(str(source_shard)))
        converted: dict[str, mx.array] = {}
        for stem in sorted(stems):
            weight = lazy.pop(stem + ".weight")
            scales = lazy.pop(stem + ".scales")
            biases = lazy.pop(stem + ".biases")
            dense = mx.dequantize(
                weight,
                scales=scales,
                biases=biases,
                group_size=64,
                bits=4,
                mode="affine",
            )
            packed = mx.quantize(
                dense,
                group_size=group_size,
                bits=bits,
                mode="affine",
            )
            mx.eval(packed)
            converted[stem + ".weight"] = packed[0]
            converted[stem + ".scales"] = packed[1]
            converted[stem + ".biases"] = packed[2]
            del dense, weight, scales, biases
        converted.update(lazy)

        output_shard = build / source_shard.name
        mx.save_safetensors(str(output_shard), converted)
        output_names = set(_header(output_shard)) - {"__metadata__"}
        if output_names != set(converted):
            raise RuntimeError("committed draft shard tensor names differ")
        weight_map = {name: output_shard.name for name in sorted(output_names)}
        total_size = sum(int(value.nbytes) for value in converted.values())

        quantization = {
            "bits": bits,
            "group_size": group_size,
            "mode": "affine",
        }
        config["quantization"] = quantization
        config["quantization_config"] = quantization
        config["voom_quantization"] = {
            "profile": "all-draft",
            "quantized_tensors": len(stems),
            "source": str(source),
        }
        config["voom_draft_requantization"] = {
            "version": 1,
            "proposal_only": True,
            "source_bits": 4,
            "source_group_size": 64,
            "source_mode": "affine",
            "source_index_sha256": _sha256(
                source / "model.safetensors.index.json"),
            "source_shard_sha256": _sha256(source_shard),
            "target_bits": bits,
            "target_group_size": group_size,
            "target_mode": "affine",
        }
        _write_json(build / "config.json", config)
        _write_json(build / "model.safetensors.index.json", {
            "metadata": {"total_size": total_size},
            "weight_map": weight_map,
        })
        for name in _METADATA_FILES:
            candidate = source / name
            if candidate.is_file():
                shutil.copy2(candidate, build / name)
        os.replace(build, output)
        return output
    except BaseException:
        shutil.rmtree(build, ignore_errors=True)
        raise
    finally:
        mx.clear_cache()


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Requantize a proposal-only affine4 MLX draft")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bits", type=int, choices=(2, 3), default=2)
    parser.add_argument("--group-size", type=int, choices=(64,), default=64)
    args = parser.parse_args()
    print(requantize_affine_draft(
        args.source, args.output, bits=args.bits,
        group_size=args.group_size))


if __name__ == "__main__":
    _main()
