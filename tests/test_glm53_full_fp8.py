"""Full GLM-5.3 released-FP8 compatibility gates.

The full model deliberately keeps ``model_type=glm_moe_dsa`` because it uses
the same base architecture as GLM-5.2.  These tests prevent that stable model
type from accidentally routing the new FP8 checkpoint through the old BF16
storage path.
"""

from __future__ import annotations

import json
import struct
from types import SimpleNamespace

import numpy as np
import pytest


def _config() -> dict:
    return {
        "architectures": ["GlmMoeDsaForCausalLM"],
        "model_type": "glm_moe_dsa",
        "hidden_size": 4,
        "intermediate_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "vocab_size": 16,
        "rms_norm_eps": 1e-5,
        "max_position_embeddings": 32,
        "tie_word_embeddings": False,
        "dtype": "bfloat16",
        "n_routed_experts": 2,
        "num_experts_per_tok": 1,
        "moe_intermediate_size": 2,
        "num_nextn_predict_layers": 1,
        "quantization_config": {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [2, 2],
        },
    }


def _write_safetensor(path, tensors) -> None:
    header = {}
    payload = bytearray()
    for name, (value, dtype) in tensors.items():
        raw = np.ascontiguousarray(value).tobytes()
        start = len(payload)
        payload.extend(raw)
        header[name] = {
            "dtype": dtype,
            "shape": list(value.shape),
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def test_full_glm53_fp8_store_joins_pair_and_hides_scale(tmp_path):
    import mlx.core as mx

    from runtime.model_loader import WeightStore
    from runtime.quant import dequantize_finegrained_fp8

    (tmp_path / "config.json").write_text(json.dumps(_config()))
    codes = np.arange(16, dtype=np.uint8).reshape(4, 4)
    scales = np.array([[0.25, 0.5], [1.0, 2.0]], dtype=np.float32)
    stem = "model.layers.0.mlp.gate_proj"
    _write_safetensor(tmp_path / "model.safetensors", {
        f"{stem}.weight": (codes, "F8_E4M3"),
        f"{stem}.weight_scale_inv": (scales, "F32"),
    })

    store = WeightStore(tmp_path)
    logical = f"{stem}.weight"
    assert store.has(logical)
    assert f"{stem}.weight_scale_inv" not in store.names_with_prefix(stem)
    assert store.expert_storage_bytes_per_weight == pytest.approx(2.0)

    fetched, _seconds, physical_bytes = store.fetch([logical])
    expected = dequantize_finegrained_fp8(
        mx.array(codes), mx.array(scales))
    mx.eval(fetched[logical], expected)
    assert fetched[logical].dtype == mx.bfloat16
    assert bool(mx.all(fetched[logical] == expected))
    assert physical_bytes == codes.nbytes + scales.nbytes


def test_full_glm53_fp8_layer_admission_prices_widened_page():
    from runtime.engine import StreamingEngine

    engine = StreamingEngine.__new__(StreamingEngine)
    engine.cfg = SimpleNamespace(model_type="glm_moe_dsa")
    engine.store = SimpleNamespace(
        _glm53_fp8_aux={"weight": object()},
        on_disk_quantized=False,
        bf16_nf12_sidecar=None,
        finegrained_fp8_resident_bytes=lambda names: 1_000,
    )
    engine._layer_names = lambda layer: ["weight"]

    assert engine._layer_fetch_bytes_estimate(0) == 1_050


def test_bf16_glm52_does_not_claim_full_glm53_fp8(tmp_path):
    from runtime.model_loader import WeightStore

    config = _config()
    config.pop("quantization_config")
    (tmp_path / "config.json").write_text(json.dumps(config))
    weight = np.ones((4, 4), dtype=np.float32)
    _write_safetensor(tmp_path / "model.safetensors", {
        "model.layers.0.mlp.gate_proj.weight": (weight, "F32"),
    })

    store = WeightStore(tmp_path)
    assert not store._glm53_fp8_aux
    assert store.expert_storage_bytes_per_weight == 2.0
