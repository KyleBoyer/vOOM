from __future__ import annotations

import json
import struct
from pathlib import Path

import mlx.core as mx
import numpy as np

from formats.bf16_nf12_sidecar import build_nf12_sidecar
from runtime.bf16_nf12_sidecar import decode_layer, decode_names
from runtime.engine import RuntimeConfig
from runtime.model_loader import WeightStore


def test_runtime_config_loads_nf12_direct_mode(tmp_path: Path) -> None:
    config = tmp_path / "runtime.yaml"
    config.write_text(
        "runtime:\n"
        "  bf16_nf12_sidecar_dir: /external/k3-nf12\n"
        "  bf16_nf12_uncached_reads: true\n"
        "  bf16_nf12_direct_linear: true\n"
    )

    parsed = RuntimeConfig.from_yaml(config)

    assert parsed.bf16_nf12_sidecar_dir == "/external/k3-nf12"
    assert parsed.bf16_nf12_uncached_reads
    assert parsed.bf16_nf12_direct_linear


def _write_fixture(
    model_dir: Path,
) -> tuple[str, np.ndarray, str, np.ndarray]:
    rng = np.random.default_rng(140)
    name = "model.layers.0.input_layernorm.weight"
    bits = rng.integers(0, 1 << 11, size=700, dtype=np.uint16)
    bits |= np.uint16(7 << 11)
    bits[[9, 256, 699]] = np.array(
        [0x3F80, 0x8001, 0xFFFF], dtype=np.uint16
    )
    raw = bits.astype("<u2").tobytes()
    second_name = "model.layers.0.post_attention_layernorm.weight"
    second_bits = rng.integers(
        0, 1 << 11, size=1024, dtype=np.uint16
    )
    second_bits |= np.uint16(8 << 11)
    second_bits[[0, 255, 1023]] = np.array(
        [0x0001, 0xBF80, 0x7FFF], dtype=np.uint16
    )
    second_raw = second_bits.astype("<u2").tobytes()
    header = {
        name: {
            "dtype": "BF16",
            "shape": [700],
            "data_offsets": [0, len(raw)],
        },
        second_name: {
            "dtype": "BF16",
            "shape": [1024],
            "data_offsets": [len(raw), len(raw) + len(second_raw)],
        },
    }
    header_raw = json.dumps(
        header, separators=(",", ":"), sort_keys=True
    ).encode()
    header_raw += b" " * ((-len(header_raw)) % 8)
    (model_dir / "model.safetensors").write_bytes(
        struct.pack("<Q", len(header_raw))
        + header_raw
        + raw
        + second_raw
    )
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "kimi_k3",
                "hidden_size": 700,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "vocab_size": 32,
                "rms_norm_eps": 1e-6,
                "max_position_embeddings": 128,
                "torch_dtype": "bfloat16",
            }
        )
    )
    return name, bits, second_name, second_bits


def test_metal_decoder_and_weightstore_boundary_are_bit_exact(tmp_path):
    model_dir = tmp_path / "model"
    sidecar_root = tmp_path / "sidecar"
    model_dir.mkdir()
    name, expected_bits, second_name, second_expected_bits = (
        _write_fixture(model_dir)
    )
    build_nf12_sidecar(
        model_dir,
        sidecar_root,
        layers=[0],
        enforce_external=False,
        min_free_after_bytes=0,
    )

    control = WeightStore(model_dir)
    candidate = WeightStore(
        model_dir, bf16_nf12_sidecar_dir=sidecar_root
    )
    invalidated = []
    candidate.bf16_nf12_sidecar.invalidate_layer_cache = (
        lambda layer: invalidated.append(layer) or True
    )
    reference, _, reference_bytes = control.fetch([name])
    value, _, candidate_bytes = candidate.fetch([name])

    assert mx.array_equal(
        value[name].view(mx.uint16), reference[name].view(mx.uint16)
    )
    assert np.array(value[name].view(mx.uint16)).tobytes() == (
        expected_bits.astype("<u2").tobytes()
    )
    assert candidate_bytes < reference_bytes
    read_bytes, output_bytes, decode_ns, calls = (
        candidate.bf16_nf12_snapshot()
    )
    assert read_bytes == candidate_bytes
    assert output_bytes == reference[name].nbytes
    assert decode_ns > 0
    assert calls == 1
    assert invalidated == []
    candidate.release_cache_pages((name,))
    assert invalidated == [0]
    assert candidate.bf16_nf12_invalidation_snapshot() == (1, 1, 0)

    # Exercise the public decoder with the immutable mapped tensor as well as
    # through WeightStore so shape/offset metadata is covered explicitly.
    sidecar = candidate.bf16_nf12_sidecar
    entry = sidecar.layer_entry(0)
    encoded = mx.load(str(sidecar.layer_path(0)))["encoded"]
    direct = decode_layer(encoded, entry)[name]
    assert mx.array_equal(direct.view(mx.uint16), reference[name].view(mx.uint16))
    selected = decode_names(encoded, entry, [second_name])
    assert list(selected) == [second_name]
    assert np.array(selected[second_name].view(mx.uint16)).tobytes() == (
        second_expected_bits.astype("<u2").tobytes()
    )
    compact, compact_entry, _compact_spec = (
        sidecar.read_compact_tensors(0, [second_name])[second_name]
    )
    compact_value = decode_layer(compact, compact_entry)[second_name]
    assert np.array(compact_value.view(mx.uint16)).tobytes() == (
        second_expected_bits.astype("<u2").tobytes()
    )


def test_uncached_weightstore_path_requests_f_nocache_and_stays_exact(
    tmp_path,
):
    model_dir = tmp_path / "model"
    sidecar_root = tmp_path / "sidecar"
    model_dir.mkdir()
    name, _expected_bits, second_name, _second_expected_bits = (
        _write_fixture(model_dir)
    )
    build_nf12_sidecar(
        model_dir,
        sidecar_root,
        layers=[0],
        enforce_external=False,
        min_free_after_bytes=0,
    )
    control = WeightStore(model_dir)
    candidate = WeightStore(
        model_dir,
        bf16_nf12_sidecar_dir=sidecar_root,
        bf16_nf12_uncached_reads=True,
    )
    calls = []
    original = candidate.bf16_nf12_sidecar.read_layer

    def recording_read(layer, *, uncached=False):
        calls.append((layer, uncached))
        return original(layer, uncached=uncached)

    candidate.bf16_nf12_sidecar.read_layer = recording_read
    requested = [name, second_name]
    reference, _, reference_bytes = control.fetch(requested)
    value, _, candidate_bytes = candidate.fetch(requested)

    assert calls == [(0, True)]
    assert candidate_bytes < reference_bytes
    for requested_name in requested:
        assert mx.array_equal(
            value[requested_name].view(mx.uint16),
            reference[requested_name].view(mx.uint16),
        )
    candidate.release_cache_pages(tuple(requested))
    assert candidate.bf16_nf12_invalidation_snapshot() == (0, 0, 0)


def test_direct_nf12_tensor_reshape_materializes_only_that_exact_tensor(
    tmp_path,
):
    model_dir = tmp_path / "model"
    sidecar_root = tmp_path / "sidecar"
    model_dir.mkdir()
    name = "model.layers.0.self_attn.kv_b_proj.weight"
    shape = (1024, 512)
    rng = np.random.default_rng(186)
    bits = rng.integers(0, 1 << 11, size=np.prod(shape), dtype=np.uint16)
    bits |= np.uint16(7 << 11)
    bits[[9, 256, bits.size - 1]] = np.array(
        [0x3F80, 0x8001, 0xFFFF], dtype=np.uint16
    )
    raw = bits.astype("<u2").tobytes()
    header = {
        name: {
            "dtype": "BF16",
            "shape": list(shape),
            "data_offsets": [0, len(raw)],
        }
    }
    header_raw = json.dumps(
        header, separators=(",", ":"), sort_keys=True
    ).encode()
    header_raw += b" " * ((-len(header_raw)) % 8)
    (model_dir / "model.safetensors").write_bytes(
        struct.pack("<Q", len(header_raw)) + header_raw + raw
    )
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "kimi_k3",
                "hidden_size": 512,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "vocab_size": 32,
                "rms_norm_eps": 1e-6,
                "max_position_embeddings": 128,
                "torch_dtype": "bfloat16",
            }
        )
    )
    build_nf12_sidecar(
        model_dir,
        sidecar_root,
        layers=[0],
        enforce_external=False,
        min_free_after_bytes=0,
    )

    store = WeightStore(
        model_dir,
        bf16_nf12_sidecar_dir=sidecar_root,
        bf16_nf12_direct_linear=True,
    )
    fetched, _, _ = store.fetch([name])
    compact = fetched[name]
    assert compact.__class__.__name__ == "NF12Tensor"

    reshaped = compact.reshape(2, 512, 512)
    mx.eval(reshaped)
    assert tuple(reshaped.shape) == (2, 512, 512)
    assert np.asarray(reshaped.view(mx.uint16)).tobytes() == raw
