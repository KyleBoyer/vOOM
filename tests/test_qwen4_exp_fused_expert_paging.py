from __future__ import annotations

import json
from pathlib import Path
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
