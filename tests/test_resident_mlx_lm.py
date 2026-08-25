"""Admission and optional-import gates for the resident MLX-LM backend."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import pytest

from runtime.resident_mlx_lm import (
    ResidentBackendDecision,
    ResidentMLXLMEngine,
    _exact_extension_prefix,
    _exact_prompt_cache_match,
    _fork_prompt_cache,
    _parse_lossy_suffix_prefill,
    _prompt_cache_nbytes,
    _qwen35_request_incremental_bytes,
    _unique_retained_array_bytes,
    choose_resident_backend,
    import_mlx_lm,
)


def _checkpoint(
    tmp_path: Path, payload: int = 3_400_000_000, *,
    bits: int = 4, group_size: int = 32, mode: str = "mxfp4",
) -> Path:
    model_dir = tmp_path / "Qwen3.5-4B-mlx-all-mxfp4"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5",
        "quantization_config": {
            "bits": bits, "group_size": group_size, "mode": mode,
        },
        "voom_quantization": {"profile": "all", "source": "fixture"},
    }))
    (model_dir / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": payload},
        "weight_map": {},
    }))
    return model_dir


def _set_profile(model_dir: Path, profile: str) -> None:
    path = model_dir / "config.json"
    config = json.loads(path.read_text())
    config["voom_quantization"]["profile"] = profile
    path.write_text(json.dumps(config))


def _cfg(**overrides):
    values = {
        "model_type": "qwen3_5",
        "num_experts": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_auto_admits_only_measured_qwen_profile_with_bounded_headroom(tmp_path):
    model_dir = _checkpoint(tmp_path)
    decision = choose_resident_backend(
        model_dir, _cfg(), "fast",
        available_bytes=10_000_000_000, env={})

    assert decision.admitted
    assert decision.backend == "mlx-lm"
    assert decision.payload_bytes == 3_400_000_000
    assert decision.estimated_metal_bytes == 4_600_000_000
    assert decision.reason == "measured_resident_qwen_profile_admitted"


def test_all_draft_profile_is_auxiliary_only(tmp_path):
    model_dir = _checkpoint(tmp_path, payload=2_450_000_000)
    _set_profile(model_dir, "all-draft")

    served = choose_resident_backend(
        model_dir, _cfg(), "fast",
        available_bytes=10_000_000_000, env={})
    draft = choose_resident_backend(
        model_dir, _cfg(), "fast", allow_lossy_draft=True,
        available_bytes=10_000_000_000, env={})

    assert not served.admitted
    assert served.reason == "checkpoint_not_measured_all_mxfp4_profile"
    assert draft.admitted
    assert draft.payload_bytes == 2_450_000_000
    assert draft.reason == "measured_resident_qwen_draft_profile_admitted"


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_affine_all_draft_profiles_are_auxiliary_only(tmp_path, bits):
    model_dir = _checkpoint(
        tmp_path, payload=1_900_000_000,
        bits=bits, group_size=64, mode="affine")
    _set_profile(model_dir, "all-draft")

    served = choose_resident_backend(
        model_dir, _cfg(), "fast",
        available_bytes=10_000_000_000, env={})
    draft = choose_resident_backend(
        model_dir, _cfg(), "fast", allow_lossy_draft=True,
        available_bytes=10_000_000_000, env={})

    assert not served.admitted
    assert draft.admitted
    assert draft.reason == "measured_resident_qwen_draft_profile_admitted"


@pytest.mark.parametrize(
    ("bits", "group_size", "mode"),
    [(3, 32, "affine"), (3, 128, "affine"), (3, 64, "mxfp4")],
)
def test_unmeasured_all_draft_quantization_tuples_fail_closed(
    tmp_path, bits, group_size, mode,
):
    model_dir = _checkpoint(
        tmp_path, bits=bits, group_size=group_size, mode=mode)
    _set_profile(model_dir, "all-draft")

    decision = choose_resident_backend(
        model_dir, _cfg(), "fast", allow_lossy_draft=True,
        available_bytes=10_000_000_000, env={})
    assert not decision.admitted
    assert decision.reason == "checkpoint_not_measured_all_mxfp4_profile"


def test_auto_rejects_9b_when_current_system_headroom_is_too_small(tmp_path):
    model_dir = _checkpoint(tmp_path, payload=6_824_000_000)
    decision = choose_resident_backend(
        model_dir, _cfg(), "fast",
        available_bytes=8_400_000_000, env={})

    assert not decision.admitted
    assert decision.reason == "insufficient_current_system_headroom"
    assert decision.estimated_metal_bytes < decision.metal_ceiling_bytes


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"requires_vision": True}, "vision_requires_voom"),
        ({"execution_profile": "layers"}, "layer_profile_requires_voom"),
        ({"mode": "lossless"}, "auto_route_requires_fast_mode"),
        ({"cfg": _cfg(model_type="kimi_linear")}, "architecture_not_measured"),
        ({"cfg": _cfg(num_experts=256)}, "moe_checkpoint_is_out_of_core"),
    ],
)
def test_auto_fails_closed_outside_measured_envelope(
    tmp_path, kwargs, reason,
):
    model_dir = _checkpoint(tmp_path)
    decision = choose_resident_backend(
        model_dir,
        kwargs.get("cfg", _cfg()),
        kwargs.get("mode", "fast"),
        requires_vision=kwargs.get("requires_vision", False),
        execution_profile=kwargs.get("execution_profile", ""),
        available_bytes=10_000_000_000,
        env={},
    )
    assert decision.backend == "voom"
    assert decision.reason == reason


def test_operator_can_force_voom_and_invalid_backend_is_rejected(tmp_path):
    model_dir = _checkpoint(tmp_path)
    forced = choose_resident_backend(
        model_dir, _cfg(), "fast", available_bytes=10_000_000_000,
        env={"VMODEL_RESIDENT_BACKEND": "voom"})
    assert forced.reason == "operator_forced_voom"

    with pytest.raises(ValueError, match="VMODEL_RESIDENT_BACKEND"):
        choose_resident_backend(
            model_dir, _cfg(), "fast", available_bytes=10_000_000_000,
            env={"VMODEL_RESIDENT_BACKEND": "surprise"})


def test_optional_mlx_lm_import_uses_pinned_transformers_compat():
    if importlib.util.find_spec("mlx_lm") is None:
        pytest.skip("optional mlx-lm dependency is not installed")
    module = import_mlx_lm()
    assert module.__name__ == "mlx_lm"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        ("", None),
        ("off", None),
        ("0", None),
        ("8:384", (8, 384)),
    ],
)
def test_lossy_suffix_prefill_schedule_is_explicit(raw, expected):
    assert _parse_lossy_suffix_prefill(raw) == expected


@pytest.mark.parametrize("raw", ["8", "8:384:2", "x:384", "0:384", "8:0"])
def test_lossy_suffix_prefill_schedule_rejects_invalid_values(raw):
    with pytest.raises(ValueError, match="LOSSY_SUFFIX_PREFILL"):
        _parse_lossy_suffix_prefill(raw)


def test_layerwise_qwen_forward_uses_each_hybrid_cache_for_its_mask(
    monkeypatch,
):
    mask_calls = []

    class FakeBase:
        @staticmethod
        def create_ssm_mask(hidden, cache):
            mask_calls.append(("ssm", cache))
            return None

        @staticmethod
        def create_attention_mask(hidden, cache):
            mask_calls.append(("attention", cache))
            return None

    class FakeLayer:
        def __init__(self, is_linear):
            self.is_linear = is_linear

        def __call__(self, hidden, mask, cache):
            assert mask is None
            return hidden

    layers = [FakeLayer(True), FakeLayer(False), FakeLayer(True)]
    core = SimpleNamespace(
        layers=layers,
        embed_tokens=lambda inputs: mx.ones(
            (*inputs.shape, 2), dtype=mx.float32),
        norm=lambda hidden: hidden,
    )
    text_model = SimpleNamespace(
        model=core,
        args=SimpleNamespace(tie_word_embeddings=False),
        lm_head=lambda hidden: hidden,
    )
    engine = object.__new__(ResidentMLXLMEngine)
    engine.model = SimpleNamespace(language_model=text_model)
    monkeypatch.setattr(
        "runtime.resident_mlx_lm.importlib.import_module",
        lambda _name: FakeBase)
    caches = [object(), object(), object()]

    logits = engine._qwen35_layerwise_forward(
        mx.array([[1]], dtype=mx.int32), caches)
    mx.eval(logits)

    assert mask_calls == [
        ("ssm", caches[0]),
        ("attention", caches[1]),
        ("ssm", caches[2]),
    ]
    assert logits.shape == (1, 1, 2)


def test_lossy_suffix_prefill_runs_full_shallow_chunks_and_one_deep_suffix(
    monkeypatch,
):
    class FakeBase:
        @staticmethod
        def create_ssm_mask(_hidden, _cache):
            return None

        create_attention_mask = create_ssm_mask

    class FakeCache:
        state = [mx.zeros((1,), dtype=mx.float32)]

    class FakeLayer:
        is_linear = True

        def __init__(self):
            self.lengths = []

        def __call__(self, hidden, mask, cache):
            assert mask is None
            self.lengths.append(hidden.shape[1])
            return hidden + 1

    layers = [FakeLayer() for _ in range(4)]
    core = SimpleNamespace(
        layers=layers,
        embed_tokens=lambda inputs: inputs[..., None].astype(mx.float32),
        norm=lambda hidden: hidden,
    )
    text_model = SimpleNamespace(
        model=core,
        args=SimpleNamespace(tie_word_embeddings=False),
        lm_head=lambda hidden: hidden,
    )
    engine = object.__new__(ResidentMLXLMEngine)
    engine.model = SimpleNamespace(language_model=text_model)
    monkeypatch.setattr(
        "runtime.resident_mlx_lm.importlib.import_module",
        lambda _name: FakeBase)
    progress = []

    logits = engine._qwen35_lossy_suffix_prefill(
        mx.array([1, 2, 3, 4, 5], dtype=mx.int32),
        [FakeCache() for _ in layers],
        early_layers=2,
        suffix_tokens=2,
        prefill_step_size=2,
        progress=lambda done, total: progress.append((done, total)),
    )
    mx.eval(logits)

    assert layers[0].lengths == [2, 2, 1]
    assert layers[1].lengths == [2, 2, 1]
    assert layers[2].lengths == [2]
    assert layers[3].lengths == [2]
    assert progress == [(2, 5), (4, 5), (5, 5)]
    assert logits.shape == (1,)


def test_decision_shape_is_stable_for_server_telemetry():
    decision = ResidentBackendDecision(
        "mlx-lm", "auto", "measured", payload_bytes=1,
        estimated_metal_bytes=2, available_bytes=3,
        metal_ceiling_bytes=4)
    assert decision.admitted
    assert decision.reason == "measured"


def test_prompt_cache_bytes_prefers_cache_owned_accounting():
    class Cache:
        nbytes = 123

        @property
        def state(self):
            raise AssertionError("authoritative nbytes should avoid state views")

    assert _prompt_cache_nbytes([Cache(), Cache()]) == 246


def test_retained_array_bytes_deduplicates_shared_checkpoint_payloads():
    shared = mx.zeros((8,), dtype=mx.float16)
    distinct = mx.zeros((4,), dtype=mx.float32)

    class ArraysCache:
        cache = [shared]

    class KVCache:
        keys = shared
        values = distinct

    assert _unique_retained_array_bytes(
        [ArraysCache(), KVCache()],
        {"prompt_logits": shared},
        [(4, [ArraysCache(), KVCache()])],
    ) == shared.nbytes + distinct.nbytes


def test_qwen35_request_projection_includes_attention_state_and_scratch():
    cfg = SimpleNamespace(
        num_hidden_layers=32, full_attention_interval=4,
        num_key_value_heads=4, head_dim=256)
    assert _qwen35_request_incremental_bytes(
        cfg, 14_375) == 1_735_040_000


def test_prompt_cache_reuse_requires_a_strict_exact_token_extension():
    assert _exact_extension_prefix([1, 2, 3], [1, 2, 3, 4]) == 3
    assert _exact_extension_prefix([1, 2, 3], [1, 2, 3]) == 0
    assert _exact_extension_prefix([1, 2, 3], [1, 9, 3, 4]) == 0
    assert _exact_extension_prefix([], [1, 2]) == 0


def test_prompt_endpoint_match_allows_only_exact_or_forward_extension():
    assert _exact_prompt_cache_match([1, 2, 3], [1, 2, 3]) == (3, "exact")
    assert _exact_prompt_cache_match(
        [1, 2, 3], [1, 2, 3, 4]) == (3, "extension")
    assert _exact_prompt_cache_match(
        [1, 2, 3], [1, 2, 9, 4]) == (0, "miss")
    assert _exact_prompt_cache_match(
        [1, 2, 3], [1, 2]) == (0, "miss")
    assert _exact_prompt_cache_match([], [1]) == (0, "miss")


def test_prompt_cache_fork_copies_wrappers_but_shares_array_payloads():
    class ArraysCache:
        def __init__(self):
            self.cache = [mx.array([1]), mx.array([2])]

    class KVCache:
        def __init__(self):
            self.keys = mx.zeros((1, 1, 8, 2))
            self.values = mx.zeros((1, 1, 8, 2))
            self.offset = 3

    recurrent = ArraysCache()
    attention = KVCache()
    forked_recurrent, forked_attention = _fork_prompt_cache(
        [recurrent, attention])

    assert forked_recurrent is not recurrent
    assert forked_recurrent.cache is not recurrent.cache
    assert forked_recurrent.cache[0] is recurrent.cache[0]
    forked_recurrent.cache[0] = mx.array([9])
    assert recurrent.cache[0].item() == 1

    assert forked_attention is not attention
    assert forked_attention.keys is attention.keys
    assert forked_attention.values is attention.values
    forked_attention.offset += 1
    assert attention.offset == 3


def test_engine_manager_returns_admitted_mlx_lm_backend_before_voom_load(
    tmp_path,
):
    from runtime.server import EngineManager

    model_dir = _checkpoint(tmp_path)
    cfg = SimpleNamespace(
        model_type="qwen3_5", tie_word_embeddings=True,
        index_topk=0, vision_config={"depth": 1}, num_experts=0,
        hidden_size=2560, intermediate_size=9216,
        num_hidden_layers=32, num_attention_heads=16,
        num_key_value_heads=4, head_dim=256, vocab_size=248320,
        attention_bias=False)
    decision = ResidentBackendDecision(
        "mlx-lm", "auto", "measured_resident_qwen_profile_admitted",
        payload_bytes=3_400_000_000,
        estimated_metal_bytes=4_600_000_000,
        available_bytes=10_000_000_000,
        metal_ceiling_bytes=8_300_000_000)
    made = []

    class FakeResident:
        backend_name = "mlx-lm"

        def __init__(self, path, config, rc, admission):
            self.path = Path(path)
            self.cfg = config
            self.rc = rc
            self._resident_backend_decision = admission
            self._load_s = 0.01
            self.closes = 0
            made.append(self)

        def close(self):
            self.closes += 1

    manager = EngineManager()
    with patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir",
               side_effect=lambda path: path), \
         patch("runtime.resident_mlx_lm.choose_resident_backend",
               return_value=decision), \
         patch("runtime.resident_mlx_lm.ResidentMLXLMEngine", FakeResident), \
         patch("runtime.engine.StreamingEngine",
               side_effect=AssertionError("vOOM must not load after admission")):
        engine = manager.get(model_dir, "fast")
        assert manager.get(model_dir, "fast") is engine

    assert engine is made[0]
    assert engine.backend_name == "mlx-lm"
    assert engine.path == model_dir
    manager.close()
    assert engine.closes == 1
