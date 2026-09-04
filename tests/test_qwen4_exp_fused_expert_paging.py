from __future__ import annotations

import json
from pathlib import Path
import shutil
import struct

import mlx.core as mx
import numpy as np
import pytest

from runtime.model_loader import WeightStore


def _write_safetensor(path: Path, tensors: dict[str, tuple[np.ndarray, str]]) -> None:
    header = {}
    body = bytearray()
    for name, (value, dtype) in tensors.items():
        raw = np.ascontiguousarray(value).tobytes()
        start = len(body)
        body.extend(raw)
        header[name] = {
            "dtype": dtype,
            "shape": list(value.shape),
            "data_offsets": [start, len(body)],
        }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + body)


def _fixture(
    tmp_path: Path,
    *,
    bad_gate_shape: bool = False,
    include_mtp: bool = False,
) -> tuple[Path, dict]:
    root = tmp_path / "flash-next"
    root.mkdir()
    hidden, width, experts = 4, 3, 2
    text = {
        "model_type": "qwen4_exp_text",
        "hidden_size": hidden,
        "intermediate_size": width,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": hidden,
        "vocab_size": 32,
        "eos_token_id": 31,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10_000_000.0,
        "max_position_embeddings": 128,
        "tie_word_embeddings": False,
        "attention_bias": False,
        "dtype": "bfloat16",
        "num_experts": experts,
        "num_experts_per_tok": 1,
        "moe_intermediate_size": width,
        "shared_expert_intermediate_size": width,
        "layer_types": ["full_attention"],
        "full_attention_interval": 1,
        "linear_num_key_heads": 1,
        "linear_num_value_heads": 1,
        "linear_key_head_dim": hidden,
        "linear_value_head_dim": hidden,
        "linear_conv_kernel_dim": 4,
        "hc_count": 2,
        "hc_lowrank": 2,
        "ple_layer_ids": [1],
        "ple_embed_dim": hidden,
        "ple_conv_kernel_size": 4,
        "ngram_size": 3,
        "heads_per_ngram": 1,
        "ngram_vocab_size_base": 11,
        "make_ngram_vocab_size_divisible_by": 4,
        "split_ngram_parts": 2,
        "indexer_budget": 8,
        "indexer_compress_ratio": 2,
        "indexer_head_dim": 2,
        "indexer_kv_heads": 1,
        "indexer_n_heads": 1,
        "output_gate_type": "sigmoid",
    }
    (root / "config.json").write_text(json.dumps({
        "model_type": "qwen4_exp",
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "text_config": text,
    }))
    prefix = "model.language_model.layers.0.mlp"
    gate_shape = (experts, 2 * width, hidden)
    if bad_gate_shape:
        gate_shape = (experts, 2 * width - 1, hidden)
    gate_bits = np.arange(np.prod(gate_shape), dtype=np.uint16).reshape(gate_shape)
    down_bits = (
        1000 + np.arange(experts * hidden * width, dtype=np.uint16)
    ).reshape(experts, hidden, width)
    router_bits = (
        2000 + np.arange(experts * hidden, dtype=np.uint16)
    ).reshape(experts, hidden)
    tensors = {
        f"{prefix}.experts.gate_up_proj": (gate_bits, "BF16"),
        f"{prefix}.experts.down_proj": (down_bits, "BF16"),
        f"{prefix}.gate.weight": (router_bits, "BF16"),
        "model.language_model.layers.0.input_layernorm.weight": (
            np.arange(hidden, dtype=np.uint16) + 2500, "BF16"),
        "model.language_model.hyper_connection_mixer.hc_norm.weight": (
            np.arange(2 * hidden, dtype=np.uint16) + 2600, "BF16"),
        "language_model.lm_head.weight": (
            np.arange(32 * hidden, dtype=np.uint16).reshape(32, hidden) + 2700,
            "BF16",
        ),
    }
    mtp_gate_bits = gate_bits + 3000
    mtp_down_bits = down_bits + 3000
    if include_mtp:
        tensors.update({
            "mtp.layers.0.mlp.experts.gate_up_proj": (
                mtp_gate_bits, "BF16"),
            "mtp.layers.0.mlp.experts.down_proj": (
                mtp_down_bits, "BF16"),
            "mtp.layers.0.mlp.gate.weight": (router_bits + 3000, "BF16"),
        })
    shard = root / "model.safetensors"
    _write_safetensor(shard, tensors)
    (root / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {name: shard.name for name in tensors},
    }))
    return root, {
        "gate": gate_bits,
        "down": down_bits,
        "router": router_bits,
        "mtp_gate": mtp_gate_bits,
        "mtp_down": mtp_down_bits,
        "hidden": hidden,
        "width": width,
    }


def _per_expert_fp8_fixture(
    tmp_path: Path, *, omit_scale: bool = False,
) -> tuple[Path, dict[str, np.ndarray]]:
    root, _expected = _fixture(tmp_path)
    config = json.loads((root / "config.json").read_text())
    config["quantization_config"] = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
    }
    (root / "config.json").write_text(json.dumps(config))
    hidden, width, experts = 4, 3, 2
    prefix = "model.language_model.layers.0.mlp"
    tensors: dict[str, tuple[np.ndarray, str]] = {
        f"{prefix}.gate.weight": (
            np.arange(experts * hidden, dtype=np.uint16).reshape(
                experts, hidden),
            "BF16",
        ),
    }
    expected = {}
    for expert in range(experts):
        for projection, shape in (
            ("gate_proj", (width, hidden)),
            ("up_proj", (width, hidden)),
            ("down_proj", (hidden, width)),
        ):
            stem = f"{prefix}.experts.{expert}.{projection}"
            packed = np.full(shape, 56 + expert, dtype=np.uint8)
            scale = np.array([[1.25 + expert]], dtype=np.float32)
            tensors[f"{stem}.weight"] = (packed, "F8_E4M3")
            if not (omit_scale and expert == 1 and projection == "down_proj"):
                tensors[f"{stem}.weight_scale_inv"] = (scale, "F32")
            expected[f"model.layers.0.mlp.experts.{expert}.{projection}.weight"] = (
                packed)
    shard = root / "model.safetensors"
    _write_safetensor(shard, tensors)
    (root / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {name: shard.name for name in tensors},
    }))
    return root, expected


def _bits(value: mx.array) -> np.ndarray:
    return np.asarray(value.view(mx.uint16))


def test_fused_experts_are_virtual_exact_pages_and_physical_bodies_are_hidden(
        tmp_path):
    root, expected = _fixture(tmp_path)
    store = WeightStore(root)
    prefix = "model.layers.0.mlp.experts"
    gate = f"{prefix}.1.gate_proj.weight"
    up = f"{prefix}.1.up_proj.weight"
    down = f"{prefix}.1.down_proj.weight"

    names = store.names_with_prefix(prefix)
    assert gate in names and up in names and down in names
    assert f"{prefix}.gate_up_proj" not in names
    assert f"{prefix}.down_proj" not in names
    assert store.has(gate)
    assert store.storage_bytes([gate, up, down]) == (
        3 * expected["hidden"] * expected["width"] * 2)

    values, _seconds, nbytes = store.fetch([gate, up, down])
    np.testing.assert_array_equal(
        _bits(values[gate]), expected["gate"][1, :expected["width"]])
    np.testing.assert_array_equal(
        _bits(values[up]), expected["gate"][1, expected["width"]:])
    np.testing.assert_array_equal(_bits(values[down]), expected["down"][1])
    assert nbytes == 3 * expected["hidden"] * expected["width"] * 2
    assert store.qwen4_fused_expert_snapshot() == {
        "calls": 1,
        "extents": 2,
        "requested_tensors": 3,
        "bytes": nbytes,
        "virtual_tensors": 2 * 3,
    }


def test_virtual_and_regular_tensor_fetches_can_share_one_cache_page(tmp_path):
    root, expected = _fixture(tmp_path)
    store = WeightStore(root)
    virtual = "model.layers.0.mlp.experts.0.gate_proj.weight"
    router = "model.layers.0.mlp.gate.weight"

    values, _seconds, nbytes = store.fetch([virtual, router])

    np.testing.assert_array_equal(
        _bits(values[virtual]), expected["gate"][0, :expected["width"]])
    np.testing.assert_array_equal(_bits(values[router]), expected["router"])
    assert nbytes == values[virtual].nbytes + values[router].nbytes


def test_mtp_fused_experts_are_direct_paged_without_materializing_all_experts(
        tmp_path):
    root, expected = _fixture(tmp_path, include_mtp=True)
    store = WeightStore(root)
    prefix = "mtp.layers.0.mlp.experts.1"
    names = [
        f"{prefix}.{projection}.weight"
        for projection in ("gate_proj", "up_proj", "down_proj")
    ]

    assert all(store.has(name) for name in names)
    assert "mtp.layers.0.mlp.experts.gate_up_proj" not in (
        store.names_with_prefix("mtp.layers.0.mlp.experts"))
    values, _seconds, nbytes = store.fetch(names)

    np.testing.assert_array_equal(
        _bits(values[names[0]]),
        expected["mtp_gate"][1, :expected["width"]],
    )
    np.testing.assert_array_equal(
        _bits(values[names[1]]),
        expected["mtp_gate"][1, expected["width"]:],
    )
    np.testing.assert_array_equal(
        _bits(values[names[2]]), expected["mtp_down"][1],
    )
    assert nbytes == 3 * expected["hidden"] * expected["width"] * 2
    assert store.qwen4_fused_expert_snapshot()["virtual_tensors"] == 12


def test_unexpected_fused_expert_shape_fails_closed(tmp_path):
    root, _expected = _fixture(tmp_path, bad_gate_shape=True)
    with pytest.raises(ValueError, match="unexpected Qwen4-Exp fused"):
        WeightStore(root)


def test_per_expert_fp8_layout_is_joined_without_virtual_fused_slices(tmp_path):
    root, expected = _per_expert_fp8_fixture(tmp_path)
    store = WeightStore(root)
    name = "model.layers.0.mlp.experts.1.gate_proj.weight"

    assert store.qwen4_expert_layout == "per-expert-fp8"
    assert store.qwen4_fused_expert_snapshot()["virtual_tensors"] == 0
    assert name in store._glm53_fp8_aux
    values, _seconds, nbytes = store.fetch([name])

    assert values[name].dtype == mx.bfloat16
    assert tuple(values[name].shape) == tuple(expected[name].shape)
    assert nbytes == expected[name].size + 4
    assert store.glm53_fp8_snapshot()[1] == 1


def test_per_expert_fp8_layout_requires_every_scale(tmp_path):
    root, _expected = _per_expert_fp8_fixture(tmp_path, omit_scale=True)
    with pytest.raises(ValueError, match="per-expert FP8 layout is incomplete"):
        WeightStore(root)


def test_qwen4_virtual_fast_tier_is_byte_exact_and_source_bound(tmp_path):
    from formats.qwen4_fast_tier import build_qwen4_fast_tier

    root, expected = _fixture(tmp_path)
    fast_root = tmp_path / "fast"
    report = build_qwen4_fast_tier(
        root, fast_root, max_bytes=2_000_000 + 72)
    assert report["selected_experts"] == 1
    target = fast_root / root.name
    binding = json.loads(
        (target / "qwen4_fused_expert_fast_tier.json").read_text())
    assert binding["schema"] == "voom.qwen4-fused-expert-fast-tier.v1"

    store = WeightStore(root, fast_dirs=[fast_root])
    names = [
        f"model.layers.0.mlp.experts.0.{projection}.weight"
        for projection in ("gate_proj", "up_proj", "down_proj")
    ]
    values, _seconds, nbytes = store.fetch(names)
    np.testing.assert_array_equal(
        _bits(values[names[0]]), expected["gate"][0, :expected["width"]])
    np.testing.assert_array_equal(
        _bits(values[names[1]]), expected["gate"][0, expected["width"]:])
    np.testing.assert_array_equal(_bits(values[names[2]]), expected["down"][0])
    assert nbytes == 72
    assert store.fast_tier_bytes == 72
    assert store.archive_bytes == 0


def test_qwen4_trunk_first_fast_tier_serves_all_target_trunk_exactly(tmp_path):
    from formats.qwen4_fast_tier import (
        build_qwen4_fast_tier,
        validate_qwen4_fast_tier,
    )

    root, expected = _fixture(tmp_path)
    fast_root = tmp_path / "fast"
    # Target trunk: router 16 + norm 8 + final mixer 16 + head 256 bytes.
    trunk_bytes = 296
    report = build_qwen4_fast_tier(
        root,
        fast_root,
        max_bytes=2_000_000 + trunk_bytes + 72,
        placement="trunk-first",
    )
    assert report["selected_trunk_bytes"] == trunk_bytes
    assert report["selected_experts"] == 1
    target = fast_root / root.name
    binding = json.loads(
        (target / "qwen4_fused_expert_fast_tier.json").read_text())
    assert binding["schema"] == "voom.qwen4-trunk-first-fast-tier.v2"
    assert binding["placement"] == "trunk-first"
    assert validate_qwen4_fast_tier(root, target)["verdict"] == "PASS"

    store = WeightStore(root, fast_dirs=[fast_root])
    names = [
        "model.layers.0.mlp.gate.weight",
        "model.layers.0.input_layernorm.weight",
        "model.hyper_connection_mixer.hc_norm.weight",
        "lm_head.weight",
    ]
    values, _seconds, nbytes = store.fetch(names)
    np.testing.assert_array_equal(
        _bits(values[names[0]]), expected["router"])
    assert nbytes == trunk_bytes
    assert store.fast_tier_bytes == trunk_bytes
    assert store.archive_bytes == 0


def test_qwen4_fast_tier_clone_rewrites_only_candidate_bytes(
        tmp_path, monkeypatch):
    import formats.qwen4_fast_tier as fast_tier

    released, _expected = _fixture(tmp_path)
    candidate = tmp_path / "flash-next-abliterated"
    shutil.copytree(released, candidate)
    fast_root = tmp_path / "fast"
    source_report = fast_tier.build_qwen4_fast_tier(
        released,
        fast_root,
        max_bytes=2_000_000 + 72,
        target_name="released-tier",
    )
    source_tier = Path(source_report["target"])
    manifest = json.loads(
        (source_tier / "fast_tier_manifest.json").read_text())
    changed_name = next(
        name for name in manifest if name.endswith(".down_proj.weight"))
    changed = manifest[changed_name]
    shard = candidate / changed["source_file"]
    with shard.open("r+b") as output:
        output.seek(changed["source_offset"])
        original = output.read(1)
        output.seek(changed["source_offset"])
        output.write(bytes([original[0] ^ 0x01]))

    # Production requires an internal APFS clone. The portable fixture still
    # exercises topology checks, selective rewrites, binding, validation, and
    # runtime serving while substituting a normal tree copy for clonefile.
    monkeypatch.setattr(fast_tier, "_is_internal_root", lambda _path: True)
    monkeypatch.setattr(
        fast_tier,
        "_clone_tree",
        lambda source, destination: shutil.copytree(source, destination),
    )
    report = fast_tier.build_qwen4_fast_tier_clone(
        candidate,
        released,
        source_tier,
        fast_root,
        target_name="candidate-tier",
        max_bytes=10_000_000,
        min_free_bytes=0,
    )

    candidate_tier = Path(report["target"])
    assert report["source_validation"] == "PASS"
    assert report["candidate_validation"] == "PASS"
    assert report["changed_tensors"] == 1
    assert report["rewritten_bytes"] == changed["nbytes"]
    assert fast_tier.validate_qwen4_fast_tier(
        candidate, candidate_tier)["verdict"] == "PASS"
    assert fast_tier.validate_qwen4_fast_tier(
        released, source_tier)["verdict"] == "PASS"

    candidate_alias = tmp_path / "candidate-alias"
    candidate_alias.symlink_to(candidate, target_is_directory=True)
    store = WeightStore(candidate_alias, fast_dirs=[candidate_tier])
    values, _seconds, nbytes = store.fetch([changed_name])
    actual = np.ascontiguousarray(_bits(values[changed_name])).tobytes()
    with shard.open("rb") as source:
        source.seek(changed["source_offset"])
        expected = source.read(changed["nbytes"])
    assert actual == expected
    assert nbytes == changed["nbytes"]
    assert store.fast_tier_bytes == changed["nbytes"]
    assert store.archive_bytes == 0


def test_qwen4_trace_balanced_fast_tier_selects_observed_expert(tmp_path):
    from formats.qwen4_fast_tier import build_qwen4_fast_tier
    from runtime.expert_plan import write_trace

    root, _expected = _fixture(tmp_path)
    trace = write_trace(
        tmp_path / "decode-trace.json",
        [(0, (1,)), (0, (1,))],
        model=root.name,
        num_experts=2,
        expert_page_bytes=72,
    )
    fast_root = tmp_path / "fast"
    trunk_bytes = 296
    report = build_qwen4_fast_tier(
        root,
        fast_root,
        max_bytes=2_000_000 + trunk_bytes + 72,
        candidate_max_bytes=trunk_bytes + 72,
        target_name="trace-candidate",
        placement="trunk-first",
        trace_paths=[trace],
        trace_hot_experts_per_layer=1,
    )

    assert report["selection_policy"] == "equal-request-trace-heat-v1"
    assert report["trace_requests"] == 1
    assert report["trace_hot_experts_per_layer"] == 1
    target = fast_root / "trace-candidate"
    manifest = json.loads((target / "fast_tier_manifest.json").read_text())
    assert "model.layers.0.mlp.experts.1.gate_proj.weight" in manifest
    assert "model.layers.0.mlp.experts.0.gate_proj.weight" not in manifest
    binding = json.loads(
        (target / "qwen4_fused_expert_fast_tier.json").read_text())
    assert binding["trace_documents"][0]["file"] == trace.name


def test_qwen4_virtual_fast_and_archive_experts_overlap(tmp_path, monkeypatch):
    from formats.qwen4_fast_tier import build_qwen4_fast_tier

    root, expected = _fixture(tmp_path)
    fast_root = tmp_path / "fast"
    build_qwen4_fast_tier(root, fast_root, max_bytes=2_000_000 + 72)
    store = WeightStore(
        root, fast_dirs=[fast_root], parallel_storage_reads=True)
    monkeypatch.setattr(
        store, "_raw_fast_tier_is_independent", lambda _names: True)
    fast = "model.layers.0.mlp.experts.0.gate_proj.weight"
    slow = "model.layers.0.mlp.experts.1.gate_proj.weight"

    values, _seconds, nbytes = store.fetch([fast, slow])

    np.testing.assert_array_equal(
        _bits(values[fast]), expected["gate"][0, :expected["width"]])
    np.testing.assert_array_equal(
        _bits(values[slow]), expected["gate"][1, :expected["width"]])
    assert nbytes == 48
    snapshot = store.parallel_tier_snapshot()
    assert snapshot[0] == 1
    assert snapshot[1:3] == (24, 24)
    assert snapshot[6] >= 0
    store.close()


def test_qwen4_virtual_fast_tier_can_be_disabled_for_prefill(tmp_path):
    from formats.qwen4_fast_tier import build_qwen4_fast_tier

    root, expected = _fixture(tmp_path)
    fast_root = tmp_path / "fast"
    build_qwen4_fast_tier(root, fast_root, max_bytes=2_000_000 + 72)
    store = WeightStore(
        root, fast_dirs=[fast_root], parallel_storage_reads=True)
    store.qwen4_virtual_fast_tier_enabled = False
    names = [
        f"model.layers.0.mlp.experts.0.{projection}.weight"
        for projection in ("gate_proj", "up_proj", "down_proj")
    ]

    values, _seconds, nbytes = store.fetch(names)

    np.testing.assert_array_equal(
        _bits(values[names[0]]), expected["gate"][0, :expected["width"]])
    np.testing.assert_array_equal(
        _bits(values[names[1]]), expected["gate"][0, expected["width"]:])
    np.testing.assert_array_equal(
        _bits(values[names[2]]), expected["down"][0])
    assert nbytes == 72
    assert store.fast_tier_bytes == 0
    assert store.archive_bytes == 72
    assert store.parallel_tier_snapshot()[0] == 0
    store.close()
