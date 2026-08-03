from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _request_file(tmp_path: Path) -> tuple[Path, str, dict]:
    request = {
        "model": "lossy-Kimi-K3",
        "input": [
            {"role": "system", "content": "stable"},
            {"role": "developer", "content": "also stable"},
            {"role": "user", "content": "dynamic secret"},
        ],
        "tools": [{
            "type": "function", "name": "lookup",
            "description": "stable tool", "parameters": {"type": "object"},
        }],
    }
    raw = json.dumps(request, separators=(",", ":")).encode()
    path = tmp_path / "capture.json"
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest(), request


def test_startup_prefix_document_is_relative_hashed_and_strict(tmp_path):
    from runtime.prewarm import load_startup_prefixes

    request_path, digest, _request = _request_file(tmp_path)
    document = tmp_path / "prefixes.json"
    document.write_text(json.dumps({
        "schema": "voom.startup-prefixes.v1",
        "prefixes": [{
            "name": "kai-v1",
            "model": "lossy-Kimi-K3",
            "request_file": request_path.name,
            "request_sha256": digest,
            "require_persistence": True,
        }],
    }))
    entries = load_startup_prefixes([document])
    assert len(entries) == 1
    assert entries[0].request_file == request_path.resolve()
    assert entries[0].request_sha256 == digest
    assert entries[0].require_persistence

    value = json.loads(document.read_text())
    value["surprise"] = True
    document.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="unknown fields"):
        load_startup_prefixes([document])


class _CharTokenizer:
    @staticmethod
    def encode(text):
        return SimpleNamespace(ids=tuple(text.encode()))


def test_k3_static_prefix_excludes_dynamic_request_content(monkeypatch):
    from runtime.prewarm import prepare_k3_static_prefix
    from runtime.server import PreparedPrompt

    request = {
        "input": [
            {"role": "system", "content": "stable"},
            {"role": "developer", "content": "stable-2"},
            {"role": "user", "content": "never seed this"},
        ],
        "tools": [],
    }
    stem = "tools-and-system\n"

    def fake_render(_engine, _model_dir, rendered_request, _mode):
        dynamic = rendered_request["input"][2:]
        text = stem + (
            "user: never seed this\nassistant:" if dynamic else "assistant:")
        return PreparedPrompt(text, tuple(text.encode()))

    monkeypatch.setattr(
        "runtime.prewarm._render_responses_request", fake_render)
    engine = SimpleNamespace(
        cfg=SimpleNamespace(model_type="kimi_k3"),
        tokenizer=_CharTokenizer(),
    )
    prompt, metadata = prepare_k3_static_prefix(
        engine, Path("/model"), request, "fast")
    assert bytes(prompt.token_ids).startswith(stem.encode())
    assert b"never seed this" not in bytes(prompt.token_ids)
    assert metadata["dynamic_content_used_in_seed"] is False
    assert metadata["dynamic_messages"] == 1


def test_k3_server_profile_forwards_topk_hot_kv_and_disk_limits(tmp_path):
    from runtime.server import EngineManager

    captured = []

    class FakeEngine:
        def __init__(self, _path, rc):
            captured.append(rc)

        def close(self):
            pass

    cfg = SimpleNamespace(
        model_type="kimi_k3", tie_word_embeddings=False,
        index_topk=0, vision_config=None,
        num_experts_per_tok=16, num_hidden_layers=93,
    )
    environment = {
        "VMODEL_K3_WEIGHT_CACHE_MB": "150",
        "VMODEL_K3_MLX_CACHE_MB": "64",
        "VMODEL_K3_EXPERT_FETCH_BATCH": "16",
        "VMODEL_K3_DECODE_EXPERT_FETCH_BATCH": "4",
        "VMODEL_K3_EXPERT_TOP_K": "4",
        "VMODEL_K3_HOT_PROMPT_KV": "1",
        "VMODEL_K3_HOT_KV_SLOTS": "2",
        "VMODEL_K3_HOT_KV_MIN_TOKENS": "1",
        "VMODEL_K3_HOT_KV_PERSIST_DIR": str(tmp_path / "kv"),
        "VMODEL_K3_HOT_KV_PERSIST_MAX_CHECKPOINTS": "3",
        "VMODEL_K3_HOT_KV_PERSIST_MAX_MB": "2500",
        "VMODEL_K3_PREFILL_TILE_WIDTH": "128",
        "VMODEL_K3_NATIVE_FUSED_KDA_PREFILL": "1",
    }
    with patch.dict("os.environ", environment, clear=True), patch(
        "runtime.config.ModelConfig.from_dir", return_value=cfg
    ), patch(
        "runtime.path_resolver.resolve_model_dir", side_effect=lambda path: path
    ), patch(
        "runtime.engine.StreamingEngine", FakeEngine
    ):
        EngineManager().get(Path("/tmp/fake-k3"), "fast")

    rc = captured[0]
    assert rc.max_weight_cache_mb == 150
    assert rc.mlx_cache_limit_mb == 64
    assert rc.expert_fetch_batch == 16
    assert rc.decode_expert_fetch_batch == 4
    assert rc.expert_top_k_by_layer == (4,) * 93
    assert rc.hot_prompt_kv
    assert rc.hot_prompt_kv_chunk_size == 128
    assert rc.hot_prompt_kv_slots == 2
    assert rc.hot_prompt_kv_persist_dir == str(tmp_path / "kv")
    assert rc.hot_prompt_kv_persist_max_checkpoints == 3
    assert rc.hot_prompt_kv_persist_max_mb == 2500


def test_spilled_compressed_mla_is_materialized_for_durable_slice():
    import mlx.core as mx

    from runtime.hot_kv_persist import _slice_kv

    value = mx.arange(24).reshape(1, 6, 4)

    class FakeSpilledKV:
        compressed_mla = True
        keys = [None]
        values = [None]

        def materialize_latent_layer_for_persistence(self, layer):
            assert layer == 0
            self.keys[layer] = value
            return value

    arrays = _slice_kv(FakeSpilledKV(), 2, 5)
    mx.eval(arrays["k0"])
    assert arrays["k0"].shape == (1, 3, 4)
    assert arrays["k0"].tolist() == value[:, 2:5, :].tolist()


def test_spilled_kda_is_materialized_for_durable_export(tmp_path):
    import mlx.core as mx

    from runtime.kda_state import KDAStateCache

    cache = KDAStateCache(2)
    cache.enable_disk_spill(tmp_path)
    state = mx.arange(24, dtype=mx.float32).reshape(1, 2, 3, 4)
    conv = (mx.arange(6, dtype=mx.float32).reshape(1, 2, 3, 1),)
    cache.set_state(1, state)
    cache.set_conv_history(1, conv)
    assert cache.spill_layer(1)
    assert cache._state[1] is None

    arrays = cache.export_arrays()
    mx.eval(*arrays.values())
    assert arrays["kda_state_1"].tolist() == state.tolist()
    assert arrays["kda_conv_1_0"].tolist() == conv[0].tolist()
    assert cache.spill_stats()["reloads"] == 1


def test_restored_k3_endpoint_reuses_layer_stationary_spill(tmp_path):
    import mlx.core as mx

    from runtime.engine import StreamingEngine
    from runtime.kda_state import KDAStateCache
    from runtime.kv_cache import KVCache, SteppedKVCache

    mla = mx.arange(36, dtype=mx.float32).reshape(1, 3, 12)
    state = mx.arange(24, dtype=mx.float32).reshape(1, 2, 3, 4)
    base = KVCache(2)
    base.compressed_mla = True
    base.keys[1] = mla
    base.kda_cache = KDAStateCache(2)
    base.kda_cache.set_state(0, state)

    engine = object.__new__(StreamingEngine)
    engine.cfg = SimpleNamespace(
        model_type="kimi_k3", kda_layers=(0,), full_attn_layers=(1,))
    engine.rc = SimpleNamespace(
        kimi_k3_kda_spill_dir=str(tmp_path / "kda"),
        kimi_k3_mla_kv_spill_dir=str(tmp_path / "mla"),
        kimi_k3_absorbed_mla=True,
        kimi_k3_mla_key_tile_size=1024,
    )

    restored = engine._configure_restored_k3_spill(base)
    assert isinstance(restored, SteppedKVCache)
    assert restored.offset == 3
    assert restored.latent_spill_enabled
    assert restored.kda_cache.spill_enabled
    assert restored.mla_absorbed is True
    assert restored.mla_absorbed_prefill is True
    assert restored.mla_absorbed_key_tile_size == 1024

    counts = engine._respill_completed_k3_state(restored)
    assert counts == {"kda_layers": 1, "mla_layers": 1}
    assert restored.offset == 3
    assert restored.keys[1] is None
    assert restored.kda_cache._state[0] is None

    reloaded_mla = restored.materialize_latent_layer_for_persistence(1)
    reloaded_state = restored.kda_cache.state(0)
    mx.eval(reloaded_mla, reloaded_state)
    assert reloaded_mla.tolist() == mla.tolist()
    assert reloaded_state.tolist() == state.tolist()


def test_k3_one_token_dense_mlp_barriers_preserve_values(monkeypatch):
    import mlx.core as mx
    import numpy as np

    from runtime import kimi_linear

    rng = np.random.default_rng(812)
    cfg = SimpleNamespace(
        hidden_act="situ",
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
    )
    hidden = mx.array(rng.standard_normal((1, 1, 6)).astype(np.float32))
    prefix = "layer.mlp"
    weights = {
        f"{prefix}.gate_proj.weight": mx.array(
            rng.standard_normal((11, 6)).astype(np.float32)),
        f"{prefix}.up_proj.weight": mx.array(
            rng.standard_normal((11, 6)).astype(np.float32)),
        f"{prefix}.down_proj.weight": mx.array(
            rng.standard_normal((6, 11)).astype(np.float32)),
    }
    ordinary = kimi_linear._kimi_dense_mlp(
        hidden, weights, prefix, cfg)
    mx.eval(ordinary)

    real_eval = mx.eval
    barriers = []

    def recording_eval(*values):
        barriers.append(tuple(value.shape for value in values))
        return real_eval(*values)

    monkeypatch.setattr(kimi_linear.mx, "eval", recording_eval)
    staged = kimi_linear._kimi_dense_mlp(
        hidden, weights, prefix, cfg,
        synchronize_subprojections=True)
    real_eval(staged)

    assert barriers == [
        ((1, 1, 11),),
        ((1, 1, 11),),
        ((1, 1, 11),),
        ((1, 1, 6),),
    ]
    assert np.array_equal(np.asarray(staged), np.asarray(ordinary))


def test_k3_one_token_shared_expert_uses_dense_barriers(monkeypatch):
    import mlx.core as mx

    from runtime import kimi_linear

    cfg = SimpleNamespace(
        model_type="kimi_k3",
        moe_latent_hidden_size=0,
        hidden_act="situ",
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
    )
    hidden = mx.ones((1, 1, 2), dtype=mx.float32)
    shared_prefix = "layer.block_sparse_moe.shared_experts"
    weights = {
        f"{shared_prefix}.gate_proj.weight": mx.ones((3, 2)),
        f"{shared_prefix}.up_proj.weight": mx.ones((3, 2)),
        f"{shared_prefix}.down_proj.weight": mx.ones((2, 3)),
    }
    expert_weights = {
        "layer.block_sparse_moe.experts.0.w1.weight": mx.ones((3, 2)),
        "layer.block_sparse_moe.experts.0.w3.weight": mx.ones((3, 2)),
        "layer.block_sparse_moe.experts.0.w2.weight": mx.ones((2, 3)),
    }
    monkeypatch.setattr(
        kimi_linear, "_route_experts",
        lambda *_args, **_kwargs: (
            mx.array([[[0]]]), mx.array([[[1.0]]]),
        ),
    )
    real_dense = kimi_linear._kimi_dense_mlp
    synchronization = []

    def recording_dense(*args, synchronize_subprojections=False, **kwargs):
        synchronization.append(synchronize_subprojections)
        return real_dense(
            *args,
            synchronize_subprojections=synchronize_subprojections,
            **kwargs,
        )

    monkeypatch.setattr(kimi_linear, "_kimi_dense_mlp", recording_dense)
    output = kimi_linear._kimi_moe_output(
        hidden, weights, "layer", cfg, 1,
        lambda *_args, **_kwargs: {0: expert_weights},
    )
    mx.eval(output)

    assert synchronization == [True]
