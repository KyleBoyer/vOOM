"""Tiny-MLX gates for the atomic DFlash2 draft-sidecar builder.

This file intentionally does not touch the real 3.85 GB checkpoint.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mlx.core as mx
import pytest

from runtime.dflash2_adapter import inspect_runtime_sidecar
from runtime.dflash2_schema import DFlash2Config
from runtime.dflash2_sidecar import (
    MANIFEST_NAME,
    build_sidecar,
    validate_sidecar,
)


REVISION = "a" * 40
REPOSITORY = "test/qwen-dflash2"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tiny_config():
    return {
        "architectures": ["DFlash2DraftModel"],
        "model_type": "qwen3",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "is_causal": False,
        "dtype": "bfloat16",
        "hidden_act": "silu",
        "tie_word_embeddings": False,
        "hidden_size": 64,
        "intermediate_size": 64,
        "vocab_size": 128,
        "num_hidden_layers": 1,
        "num_target_layers": 3,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 1024,
        "sliding_window": 64,
        "layer_types": ["sliding_attention"],
        "rope_parameters": {"rope_theta": 10_000, "rope_type": "default"},
        "dflash_config": {
            "block_size": 4,
            "conv_kernel_size": 2,
            "conv_group_size": 16,
            "selector_rank": 64,
            "selector_top_k": 4,
            "mask_token_id": 127,
            "target_layer_ids": [0, 2],
        },
    }


def _write_source(path: Path):
    path.mkdir()
    raw = _tiny_config()
    config_bytes = json.dumps(raw, indent=2, sort_keys=True).encode()
    (path / "config.json").write_bytes(config_bytes)
    config = DFlash2Config.from_mapping(raw)
    tensors = {
        name: mx.zeros(spec.shape, dtype=mx.bfloat16)
        for name, spec in config.expected_tensor_specs().items()}
    weights = path / "model.safetensors"
    mx.save_safetensors(str(weights), tensors)
    tree = path / ".cache" / "huggingface" / "trees"
    tree.mkdir(parents=True)
    (tree / f"{REVISION}.json").write_text(json.dumps({
        "format_version": 1,
        "files": {
            "model.safetensors": {
                "lfs_size": weights.stat().st_size,
                "lfs_sha256": _sha256(weights),
                "blob_id": "fixture",
                "xet_hash": "fixture",
            },
        },
    }))
    return {
        "repository": REPOSITORY,
        "revision": REVISION,
        "expected_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "expected_weights_sha256": _sha256(weights),
        "expected_weights_bytes": weights.stat().st_size,
        "require_official_geometry": False,
    }


def test_sidecar_build_is_atomic_deterministic_and_explicitly_unsupported(tmp_path):
    source = tmp_path / "source"
    first = tmp_path / "first"
    second = tmp_path / "second"
    common = _write_source(source)

    report_a = build_sidecar(
        source, first, min_free_bytes=0, group_size=64, **common)
    report_b = build_sidecar(
        source, second, min_free_bytes=0, group_size=64, **common)

    assert report_a["passed"] is True
    assert report_a["output"]["weights_sha256"] == report_b["output"][
        "weights_sha256"]
    assert (first / "model.safetensors").read_bytes() == (
        second / "model.safetensors").read_bytes()
    config = json.loads((first / "config.json").read_text())
    assert config["vmodel_sidecar"]["runtime_supported"] is False
    assert config["vmodel_sidecar"]["enabled_by_default"] is False
    assert config["vmodel_sidecar"]["recurrent_rollback_oracle_required"] is True
    header = mx.load(str(first / "model.safetensors"))
    assert "candidate_selector.predecessor_codebook.weight" in header
    assert "candidate_selector.predecessor_codebook.scales" in header
    assert "candidate_selector.predecessor_codebook" not in header
    assert validate_sidecar(source, first, **common)["passed"] is True
    assert not list(tmp_path.glob(".first.building-*"))


def test_sidecar_validator_rejects_config_quantization_tamper(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "sidecar"
    common = _write_source(source)
    build_sidecar(source, output, min_free_bytes=0, **common)
    config_path = output / "config.json"
    config = json.loads(config_path.read_text())
    config["quantization"]["bits"] = 8
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="quantization metadata mismatch"):
        validate_sidecar(source, output, **common)


def test_runtime_header_accepts_pinned_q4_and_fails_closed_on_promotion_tamper(
    tmp_path,
):
    source = tmp_path / "source"
    output = tmp_path / "sidecar"
    common = _write_source(source)
    build_sidecar(
        source, output, min_free_bytes=0, group_size=64, bits=4,
        mode="affine", **common)

    config, physical_names, manifest = inspect_runtime_sidecar(
        output, require_official_geometry=False)
    assert config.checkpoint.target_layer_ids == (0, 2)
    assert len(physical_names) == manifest["output"]["tensor_count"]

    config_path = output / "config.json"
    raw = json.loads(config_path.read_text())
    raw["vmodel_sidecar"]["enabled_by_default"] = True
    config_path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="default-off contract"):
        inspect_runtime_sidecar(output, require_official_geometry=False)

    raw["vmodel_sidecar"]["enabled_by_default"] = False
    config_path.write_text(json.dumps(raw))
    manifest_path = output / MANIFEST_NAME
    raw_manifest = json.loads(manifest_path.read_text())
    raw_manifest["serving"]["runtime_supported"] = True
    manifest_path.write_text(json.dumps(raw_manifest))
    with pytest.raises(ValueError, match="default-off serving gate"):
        inspect_runtime_sidecar(output, require_official_geometry=False)
