"""Runtime gates for an MTPLX released-BF16 MTP sidecar."""

import hashlib
import json

import mlx.core as mx

from runtime import quant
from runtime.model_loader import WeightStore
from runtime.qwen35_mtp import QwenMTPDrafter
from runtime.weight_cache import WeightCache


MTP_GATE = "mtp.layers.0.mlp.gate_proj.weight"


def _config():
    return {
        "model_type": "qwen2",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "vocab_size": 8,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 128,
        "tie_word_embeddings": True,
        "attention_bias": False,
        "torch_dtype": "bfloat16",
        "quantization": {"group_size": 32, "bits": 4, "mode": "mxfp4"},
    }


def _write_mtplx_with_stale_overlay(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    body = model / "body.safetensors"
    body_w, body_scales = mx.quantize(
        mx.ones((64, 64), dtype=mx.bfloat16),
        group_size=32, bits=4, mode="mxfp4")
    mx.save_safetensors(str(body), {
        "model.body.weight": body_w,
        "model.body.scales": body_scales,
    })
    released_gate = mx.arange(4 * 64, dtype=mx.float32).reshape(
        4, 64).astype(mx.bfloat16)
    sidecar = model / "mtp-bf16.safetensors"
    mx.save_safetensors(str(sidecar), {MTP_GATE: released_gate})
    sidecar_sha = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    (model / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {
            "mtplx_mtp_sidecar": sidecar.name,
            "mtplx_mtp_sidecar_sha256": sidecar_sha,
        },
        "weight_map": {
            "model.body.weight": body.name,
            "model.body.scales": body.name,
        },
    }))
    (model / "config.json").write_text(json.dumps(_config()))

    # Reproduce the real Huihui collision: this manifest predates sidecar
    # installation and still advertises the old all-MXFP4 MTP payload by the
    # same logical name. It must never override the explicit BF16 pointer.
    fast = tmp_path / "fast"
    fast.mkdir()
    stale = fast / "stale.safetensors"
    stale_gate = mx.ones((4, 8), dtype=mx.uint32)
    mx.save_safetensors(str(stale), {MTP_GATE: stale_gate})
    (fast / "fast_tier_manifest.json").write_text(json.dumps({
        MTP_GATE: {
            "file": stale.name,
            "offset": 0,
            "nbytes": stale_gate.nbytes,
            "dtype": "U32",
            "shape": list(stale_gate.shape),
        },
    }))
    return model, fast, released_gate


def test_sidecar_is_authoritative_over_stale_quantized_fast_tier(tmp_path):
    model, fast, expected = _write_mtplx_with_stale_overlay(tmp_path)
    store = WeightStore(model, fast_dirs=[fast])

    fetched, _seconds, _nbytes = store.fetch([MTP_GATE])
    gate = fetched[MTP_GATE]
    mx.eval(gate, expected)

    assert isinstance(gate, mx.array)
    assert gate.dtype == mx.bfloat16
    assert gate.shape == (4, 64)
    assert mx.array_equal(gate, expected)
    assert store.fast_tier_tensors == 0


def test_exact_bf16_sidecar_fast_tier_overrides_stale_quantized_entry(tmp_path):
    from runtime.qwen_mtp_bf16_fast_tier import build

    model, fast, expected = _write_mtplx_with_stale_overlay(tmp_path)
    report = build(
        model, fast, fast_fraction=0.5,
        global_fast_limit=10_000_000,
        min_internal_free=0,
    )
    store = WeightStore(model, fast_dirs=[fast])

    fetched, _seconds, _nbytes = store.fetch([MTP_GATE])
    gate = fetched[MTP_GATE]
    mx.eval(gate, expected)

    assert report["source_sha256"] == hashlib.sha256(
        (model / "mtp-bf16.safetensors").read_bytes()).hexdigest()
    assert store._mtplx_mtp_exact_fast_names == {MTP_GATE}
    assert gate.dtype == mx.bfloat16
    assert mx.array_equal(gate, expected)
    assert store.fast_tier_tensors == 1


def test_drafter_uses_isolated_untransformed_bf16_page(tmp_path):
    model, fast, expected = _write_mtplx_with_stale_overlay(tmp_path)
    store = WeightStore(model, fast_dirs=[fast])
    policy = quant.QuantPolicy(
        bits=4, group_size=32, mode="mxfp4", min_dim=0)
    cache = WeightCache(store, max_bytes=1_000_000,
                        transform=policy.transform)
    engine = type("Engine", (), {"store": store, "cache": cache})()
    drafter = QwenMTPDrafter(engine)

    released = drafter.prepare_request_weights()
    gate = released[MTP_GATE]
    projected = quant.matmul(
        mx.ones((1, 1, 64), dtype=mx.bfloat16), gate)
    mx.eval(gate, expected, projected)

    assert isinstance(gate, mx.array)
    assert not isinstance(gate, quant.QTensor)
    assert gate.shape == (4, 64)
    assert projected.shape == (1, 1, 4)
    assert cache.contains("qwen35_mtp:released-bf16")
    assert not cache.contains("qwen35_mtp")

    transformed = cache.get("qwen35_mtp", [MTP_GATE])[MTP_GATE]
    assert isinstance(transformed, quant.QTensor)
    assert released[MTP_GATE] is gate


def test_drafter_release_clears_caller_mapping_and_representation_page(tmp_path):
    model, fast, _expected = _write_mtplx_with_stale_overlay(tmp_path)
    store = WeightStore(model, fast_dirs=[fast])
    cache = WeightCache(store, max_bytes=1_000_000)
    engine = type("Engine", (), {"store": store, "cache": cache})()
    drafter = QwenMTPDrafter(engine)

    released = drafter.prepare_request_weights()
    expected_bytes = sum(value.nbytes for value in released.values())
    info = drafter.release_request_weights(released)

    assert released == {}
    assert info == {
        "resident_bytes": expected_bytes,
        "cache_discarded": 1,
    }
    assert not cache.contains("qwen35_mtp:released-bf16")
