"""Shard-streaming conversion to standard MLX quantized safetensors.

The converter never constructs the complete model.  It opens one HF shard
lazily, quantizes selected matrices, atomically commits one output shard, and
records a resumable manifest before advancing.  This keeps the working set
bounded for checkpoints far larger than RAM (the GLM side-quest use case).

The default ``experts`` profile preserves attention, routers, embeddings, and
the LM head while quantizing routed/shared expert projections.  It is the
quality/speed profile validated on real OLMoE-1B-7B-0924-Instruct.

    python -m formats.quantize_mlx SOURCE OUTPUT
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

import mlx.core as mx


_STATE_NAME = ".quantize-incomplete.json"
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
_FUSED_QWEN_EXPERT = re.compile(
    r"^(.*\.mlp\.experts)\.(gate_up_proj|down_proj)$")


def _write_json_atomic(path: Path, value: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _source_shards(model_dir: Path) -> list[Path]:
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        names = sorted(set(weight_map.values()))
        shards = [model_dir / name for name in names]
    else:
        shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors shards found in {model_dir}")
    missing = [str(path) for path in shards if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing source shards: {missing}")
    return shards


def _is_expert_weight(name: str) -> bool:
    return (
        name.endswith(".weight")
        and (".mlp.experts." in name or ".mlp.shared_experts." in name)
    )


def _should_quantize(name: str, value: mx.array, profile: str,
                     group_size: int) -> bool:
    if value.ndim != 2 or not name.endswith(".weight"):
        return False
    if value.shape[-1] % group_size:
        return False
    if profile == "experts":
        return _is_expert_weight(name)
    if profile == "all":
        return "embed_tokens" not in name and "norm" not in name
    if profile == "all-draft":
        # A speculative draft is never authoritative: quantizing its lookup
        # table can alter acceptance rate but not target output.  This profile
        # is therefore intentionally unavailable as a serving-model default.
        return "norm" not in name
    raise ValueError(f"unknown quantization profile: {profile!r}")


def _new_state(source: Path, *, profile: str, mode: str,
               group_size: int, bits: int,
               precision_plan_digest: str = "") -> dict:
    state = {
        "version": 2,
        "source": str(source.resolve()),
        "profile": profile,
        "mode": mode,
        "group_size": group_size,
        "bits": bits,
        "completed_shards": [],
        "weight_map": {},
        "quantized_tensors": 0,
        "total_size": 0,
    }
    # Keep the historical uniform-conversion state byte/schema compatible so
    # an interrupted v2 conversion can resume after this feature lands. Mixed
    # plans need stronger identity plus the per-tensor descriptors accumulated
    # across already committed shards, so they use a distinct v3 state.
    if precision_plan_digest:
        state.update({
            "version": 3,
            "precision_plan_digest": precision_plan_digest,
            "quantization_descriptors": {},
        })
    return state


def _load_or_create_state(source: Path, output: Path, *, profile: str,
                          mode: str, group_size: int, bits: int,
                          precision_plan_digest: str = "") -> dict:
    state_path = output / _STATE_NAME
    expected = _new_state(
        source, profile=profile, mode=mode, group_size=group_size, bits=bits,
        precision_plan_digest=precision_plan_digest)
    if state_path.exists():
        state = json.loads(state_path.read_text())
        keys = ["version", "source", "profile", "mode", "group_size", "bits"]
        if precision_plan_digest:
            keys.append("precision_plan_digest")
        for key in keys:
            if state.get(key) != expected[key]:
                raise ValueError(
                    f"resume state mismatch for {key}: "
                    f"found {state.get(key)!r}, expected {expected[key]!r}")
        return state
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"output directory {output} is non-empty and has no resumable state")
    output.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(state_path, expected)
    return expected


def _convert_shard(source_shard: Path, output_shard: Path, *, profile: str,
                   mode: str, group_size: int, bits: int,
                   tensor_plan: dict | None = None,
                   ) -> tuple[dict, int, int, dict]:
    lazy = dict(mx.load(str(source_shard)))
    output_tensors: dict[str, mx.array] = {}
    quantized_tensors = 0
    descriptors: dict[str, dict] = {}

    def add_quantized(name: str, value: mx.array, *, selected_mode: str = mode,
                      selected_bits: int = bits,
                      selected_group_size: int = group_size) -> None:
        nonlocal quantized_tensors
        packed = mx.quantize(
            value, group_size=selected_group_size,
            bits=selected_bits, mode=selected_mode)
        mx.eval(packed)
        output_tensors[name] = packed[0]
        stem = name[:-len(".weight")]
        output_tensors[f"{stem}.scales"] = packed[1]
        if len(packed) > 2:
            output_tensors[f"{stem}.biases"] = packed[2]
        descriptors[stem] = {
            "bits": selected_bits,
            "group_size": selected_group_size,
            "mode": selected_mode,
        }
        quantized_tensors += 1

    for name in sorted(tuple(lazy)):
        value = lazy.pop(name)
        if tensor_plan is not None:
            decision = tensor_plan.get(name)
            if not isinstance(decision, dict):
                raise ValueError(f"precision plan has no decision for {name!r}")
            storage = decision.get("storage")
            if storage == "source":
                output_tensors[name] = value
                continue
            if storage == "bf16":
                output_tensors[name] = value.astype(mx.bfloat16)
                continue
            if storage in ("mxfp4", "mxfp8"):
                add_quantized(
                    name, value, selected_mode=storage,
                    selected_bits=4 if storage == "mxfp4" else 8,
                    selected_group_size=32)
                continue
            raise ValueError(
                f"precision plan has unsupported storage {storage!r} for {name}")
        fused = _FUSED_QWEN_EXPERT.match(name)
        if (fused is not None and profile in ("experts", "all", "all-draft")
                and value.ndim == 3):
            experts = int(value.shape[0])
            if fused.group(2) == "gate_up_proj":
                if value.shape[1] % 2:
                    raise ValueError(
                        f"fused gate/up width must be even: {name}")
                middle = value.shape[1] // 2
                projections = (
                    ("gate_proj", value[:, :middle, :]),
                    ("up_proj", value[:, middle:, :]),
                )
            else:
                projections = (("down_proj", value),)
            for expert in range(experts):
                for projection, values in projections:
                    logical = (
                        f"{fused.group(1)}.{expert}.{projection}.weight")
                    matrix = values[expert]
                    if matrix.shape[-1] % group_size:
                        raise ValueError(
                            f"fused expert {logical} is not divisible by "
                            f"group_size={group_size}")
                    add_quantized(logical, matrix)
            continue
        if not _should_quantize(name, value, profile, group_size):
            output_tensors[name] = value
            continue
        add_quantized(name, value)

    tmp = output_shard.with_name(output_shard.stem + ".tmp.safetensors")
    mx.save_safetensors(str(tmp), output_tensors)
    os.replace(tmp, output_shard)

    # Reopen the committed shard before advancing.  This catches a missing or
    # malformed file while the resume manifest still points at the last known
    # good shard boundary.
    committed = mx.load(str(output_shard))
    if set(committed) != set(output_tensors):
        raise RuntimeError(f"committed shard key mismatch: {output_shard}")

    weight_map = {name: output_shard.name for name in output_tensors}
    total_size = sum(value.nbytes for value in output_tensors.values())
    del committed, output_tensors, lazy
    mx.clear_cache()
    return weight_map, quantized_tensors, total_size, descriptors


def convert_model(source: str | Path, output: str | Path, *,
                  profile: str = "experts", mode: str = "mxfp4",
                  group_size: int = 32, bits: int = 4,
                  precision_plan: str | Path | dict | None = None,
                  progress=None) -> Path:
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError("source and output directories must differ")
    config_path = source / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing {config_path}")

    plan = None
    plan_digest = ""
    if precision_plan is not None:
        from .mixed_precision import validate_plan

        plan = (json.loads(Path(precision_plan).read_text())
                if isinstance(precision_plan, (str, Path))
                else precision_plan)
        plan = validate_plan(plan, source)
        plan_digest = plan["plan_digest"]
        profile = f"mixed:{plan['plan_id']}"

    # Validate the tuple with MLX before creating a potentially large output.
    probe = mx.zeros((1, group_size), dtype=mx.bfloat16)
    try:
        mx.quantize(probe, group_size=group_size, bits=bits, mode=mode)
    except (ValueError, RuntimeError) as error:
        raise ValueError(
            f"unsupported MLX quantization tuple: mode={mode}, "
            f"group_size={group_size}, bits={bits}") from error

    state = _load_or_create_state(
        source, output, profile=profile, mode=mode,
        group_size=group_size, bits=bits,
        precision_plan_digest=plan_digest)
    state_path = output / _STATE_NAME
    shards = _source_shards(source)
    completed = set(state["completed_shards"])

    for index, source_shard in enumerate(shards, 1):
        if source_shard.name in completed:
            if not (output / source_shard.name).is_file():
                raise FileNotFoundError(
                    f"resume state names missing output shard {source_shard.name}")
            if progress:
                progress(index, len(shards), source_shard.name, True)
            continue
        weight_map, quantized_tensors, total_size, descriptors = _convert_shard(
            source_shard, output / source_shard.name,
            profile=profile, mode=mode, group_size=group_size, bits=bits,
            tensor_plan=plan["tensors"] if plan is not None else None)
        state["weight_map"].update(weight_map)
        state["quantized_tensors"] += quantized_tensors
        state["total_size"] += total_size
        if plan is not None:
            state["quantization_descriptors"].update(descriptors)
        state["completed_shards"].append(source_shard.name)
        _write_json_atomic(state_path, state)
        if progress:
            progress(index, len(shards), source_shard.name, False)

    config = json.loads(config_path.read_text())
    # MLX-LM requires the uniform tuple at top level and discovers selective
    # modules from physical ``.scales`` keys.  WeightStore supports the same
    # canonical descriptor.  A pure path->tuple map works in vOOM but is not a
    # portable standard-MLX checkpoint, so do not emit that looser form.
    quantization = {"bits": bits, "group_size": group_size, "mode": mode}
    if plan is not None:
        quantization.update(state["quantization_descriptors"])
    config["quantization"] = quantization
    config["quantization_config"] = quantization
    config["voom_quantization"] = {
        "profile": profile,
        "quantized_tensors": state["quantized_tensors"],
        "source": str(source.resolve()),
    }
    if plan is not None:
        config["voom_quantization"].update({
            "precision_plan_id": plan["plan_id"],
            "precision_plan_digest": plan["plan_digest"],
            "source_layout_fingerprint": plan["source_layout_fingerprint"],
            "estimated_weight_bytes": plan["summary"]["estimated_bytes"],
        })
    try:
        from runtime.lm_head_stream import lm_head_source_identity

        exact_identity = lm_head_source_identity(source)
        if exact_identity.verified_release_hash:
            config["voom_quantization"]["source_lm_head_fingerprint"] = (
                exact_identity.fingerprint)
    except (FileNotFoundError, KeyError, OSError, ValueError):
        # Expert-only/tied-head fixtures can legitimately have no standalone
        # LM head. Serving-side row reranking still requires an explicit
        # verified fingerprint and therefore cannot silently use this absence.
        pass
    _write_json_atomic(output / "config.json", config)
    _write_json_atomic(output / "model.safetensors.index.json", {
        "metadata": {"total_size": state["total_size"]},
        "weight_map": state["weight_map"],
    })
    for name in _METADATA_FILES:
        path = source / name
        if path.is_file():
            shutil.copy2(path, output / name)
    state_path.unlink()
    return output


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert HF safetensors one shard at a time to MLX quantization")
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument(
        "--profile",
        choices=("experts", "all", "all-draft"),
        default="experts",
    )
    parser.add_argument("--mode", choices=("affine", "mxfp4", "nvfp4", "mxfp8"),
                        default="mxfp4")
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument(
        "--precision-plan", type=Path,
        help="validated component-aware plan from formats.mixed_precision")
    args = parser.parse_args()

    def report(done, total, shard, resumed):
        label = "resumed" if resumed else "converted"
        print(f"[{done}/{total}] {label} {shard}", flush=True)

    result = convert_model(
        args.source, args.output, profile=args.profile, mode=args.mode,
        group_size=args.group_size, bits=args.bits,
        precision_plan=args.precision_plan, progress=report)
    print(result)


if __name__ == "__main__":
    _main()
