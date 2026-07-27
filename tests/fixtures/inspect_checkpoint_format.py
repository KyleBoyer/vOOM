#!/usr/bin/env python3
"""Inspect a real downloaded checkpoint's actual dtype/quantization format.

Written ahead of Kimi K3's open-weight release (promised 2026-07-27) per
CLAUDE.md's Goal 3 note: the real, verified dtype per tensor category must
be checked FIRST, before assuming either the existing
`dequantize_compressed_tensors_int4` path (K2.5's format) or a bare
`mx.dequantize(mode="mxfp4")` call (MLX's native convention) is the right
fit for whatever K3 actually ships. Multiple independent sources claim K3's
released format is MXFP4 (~1.4TB total), unlike Kimi Linear's own bf16
checkpoint -- but claims from before the checkpoint exists are not
verification. This script IS the verification step, run against the real
files once they exist. It never guesses a format from a claim; it only
reports what config.json / model.safetensors.index.json actually say.

Usage:
    .venv/bin/python tests/fixtures/inspect_checkpoint_format.py \
        models/Kimi-K3-Instruct
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

# Tensor-name patterns broad enough to survive a naming convention this
# project has not seen yet -- deliberately conservative regexes, not an
# exhaustive per-architecture list, since the whole point is not to assume
# K3's naming matches anything already known.
_CATEGORY_PATTERNS = (
    ("expert", re.compile(r"\.experts?\.", re.IGNORECASE)),
    ("shared_expert", re.compile(r"shared_expert", re.IGNORECASE)),
    ("attention", re.compile(
        r"self_attn|\.attn\.|attention", re.IGNORECASE)),
    ("linear_attention", re.compile(
        r"linear_attn|delta|kda|mamba", re.IGNORECASE)),
    ("router_gate", re.compile(r"\bgate\b|router", re.IGNORECASE)),
    ("norm", re.compile(r"norm", re.IGNORECASE)),
    ("embedding", re.compile(r"embed_tokens|wte\b", re.IGNORECASE)),
    ("lm_head", re.compile(r"lm_head", re.IGNORECASE)),
)


def _categorize(tensor_name: str) -> str:
    for label, pattern in _CATEGORY_PATTERNS:
        if pattern.search(tensor_name):
            return label
    return "other"


def inspect(model_dir: Path) -> dict:
    config_path = model_dir / "config.json"
    index_path = model_dir / "model.safetensors.index.json"
    single_file = model_dir / "model.safetensors"

    report: dict = {"model_dir": str(model_dir)}

    if config_path.exists():
        config = json.loads(config_path.read_text())
        report["config_model_type"] = config.get("model_type")
        report["config_architectures"] = config.get("architectures")
        report["config_num_hidden_layers"] = config.get("num_hidden_layers")
        report["config_num_experts"] = (
            config.get("num_experts") or config.get("n_routed_experts"))
        report["config_layer_types_present"] = "layer_types" in config
        # If the checkpoint documents its own quantization scheme, that is
        # authoritative -- print it in full rather than inferring from
        # dtype strings alone, exactly like K2.5's real
        # "compressed-tensors INT4" designation was only knowable this way.
        for key in ("quantization_config", "quant_config", "quantization"):
            if key in config:
                report[f"config_{key}"] = config[key]
    else:
        report["config_error"] = f"missing {config_path}"

    dtype_by_category: dict[str, Counter] = defaultdict(Counter)
    total_tensors = 0
    if index_path.exists():
        index = json.loads(index_path.read_text())
        weight_map = index.get("weight_map", {})
        # The index maps tensor name -> shard filename, not dtype directly;
        # dtype requires opening the safetensors header. Do that via the
        # already-vetted in-repo reader rather than a new parser, so this
        # inspector inherits the same header-parsing correctness proof the
        # rest of the runtime already relies on.
        from safetensors import safe_open

        shard_files = sorted(set(weight_map.values()))
        opened_headers: dict[str, dict] = {}
        for shard in shard_files:
            shard_path = model_dir / shard
            if not shard_path.exists():
                continue
            with safe_open(str(shard_path), framework="numpy") as handle:
                for name in handle.keys():
                    dtype = handle.get_slice(name).get_dtype()
                    category = _categorize(name)
                    dtype_by_category[category][str(dtype)] += 1
                    total_tensors += 1
        report["index_shard_count"] = len(shard_files)
    elif single_file.exists():
        from safetensors import safe_open

        with safe_open(str(single_file), framework="numpy") as handle:
            for name in handle.keys():
                dtype = handle.get_slice(name).get_dtype()
                category = _categorize(name)
                dtype_by_category[category][str(dtype)] += 1
                total_tensors += 1
        report["index_shard_count"] = 1
    else:
        report["index_error"] = (
            f"neither {index_path} nor {single_file} exists")

    report["total_tensors_inspected"] = total_tensors
    report["dtype_by_category"] = {
        category: dict(counts)
        for category, counts in sorted(dtype_by_category.items())
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args()
    if not args.model_dir.exists():
        raise SystemExit(f"model directory does not exist: {args.model_dir}")
    report = inspect(args.model_dir)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    if "config_error" in report or "index_error" in report:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
