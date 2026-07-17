from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

from runtime import layer_runner
from runtime.kv_cache import KVCache
from tests.fixtures.lossy_fastkv_tsp_qwen_v0 import (
    StateIsolationError,
    TSPProfile,
    UnsupportedFastKVTSP,
    engine_state_snapshot,
    generate_greedy,
    isolated_engine_call,
    run_prefill,
    select_positions,
    validate_admission,
)
from tests.fixtures.qwen_lossy_fastkv_tsp_v0_gate import (
    PREREGISTERED_MAX_NLL_DELTA,
    preregistered_nll_gate,
)


class _Cache:
    def __init__(self, weights):
        self.weights = weights

    def get(self, _key, _names):
        return self.weights


class _TinyQwen:
    def __init__(self):
        self.cfg = SimpleNamespace(
            model_type="qwen2",
            num_experts=0,
            vision_config=None,
            layer_types=(),
            rope_interleave=False,
            num_hidden_layers=3,
            hidden_size=8,
            intermediate_size=16,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            rope_theta=10_000.0,
            rms_norm_eps=1e-6,
            max_position_embeddings=128,
            eos_token_ids=(),
        )
        self.rc = SimpleNamespace()
        self.effective_max_position_embeddings = 128
        self._rope_freqs = None
        self._mscale = 1.0
        self.embedding = (
            mx.arange(32 * 8, dtype=mx.float32).reshape(32, 8) / 100)
        weights = {}
        for layer in range(self.cfg.num_hidden_layers):
            prefix = f"model.layers.{layer}"
            shift = layer + 1
            weights.update({
                f"{prefix}.input_layernorm.weight": mx.ones((8,)),
                f"{prefix}.post_attention_layernorm.weight": mx.ones((8,)),
                f"{prefix}.self_attn.q_proj.weight": (
                    mx.arange(8 * 8, dtype=mx.float32).reshape(8, 8)
                    / (180 + shift)),
                f"{prefix}.self_attn.k_proj.weight": (
                    mx.arange(4 * 8, dtype=mx.float32).reshape(4, 8)
                    / (170 + shift)),
                f"{prefix}.self_attn.v_proj.weight": (
                    mx.arange(4 * 8, dtype=mx.float32).reshape(4, 8)
                    / (190 + shift)),
                f"{prefix}.self_attn.o_proj.weight": (
                    mx.arange(8 * 8, dtype=mx.float32).reshape(8, 8)
                    / (210 + shift)),
                f"{prefix}.mlp.gate_proj.weight": (
                    mx.arange(16 * 8, dtype=mx.float32).reshape(16, 8)
                    / (230 + shift)),
                f"{prefix}.mlp.up_proj.weight": (
                    mx.arange(16 * 8, dtype=mx.float32).reshape(16, 8)
                    / (250 + shift)),
                f"{prefix}.mlp.down_proj.weight": (
                    mx.arange(8 * 16, dtype=mx.float32).reshape(8, 16)
                    / (270 + shift)),
            })
        self.cache = _Cache(weights)
        self._norm_w = mx.ones((8,))
        self._head = (
            mx.arange(32 * 8, dtype=mx.float32).reshape(32, 8) / 290)

        # Request-owned runtime state must remain untouched by the fixture.
        self.last_kv = None
        self._hot_prompt_slots = []
        self._prompt_kv_store = None
        self._hot_kv_persist = None
        self._vision_prompt_cache = None
        self._position_free_pool = None
        self._provisional = None
        self._h_window = None
        self._h_last = None
        self._tap_hidden = {}
        self.tokenizer = SimpleNamespace(
            decode=lambda values: " ".join(str(value) for value in values))

    def _embed(self, tokens):
        return self.embedding[mx.array(tokens)][None]

    @staticmethod
    def _layer_key(layer):
        return f"layer.{layer}"

    @staticmethod
    def _layer_names(_layer):
        return []

    def _lm_head_weight(self):
        return self._head


def _profile(**changes):
    values = {
        "tsp_layer": 0,
        "retention": 0.5,
        "recent_window": 2,
        "pool_width": 3,
        "query_chunk": 3,
    }
    values.update(changes)
    return TSPProfile(**values)


def test_selector_is_sorted_bounded_deterministic_and_keeps_endpoint_window():
    salience = [0.01, 0.9, 0.02, 0.03, 0.8, 0.04, 0.05, 0.06, 0.0, 0.0]
    first, pooled = select_positions(
        salience, retention=0.5, recent_window=2, pool_width=3)
    second, _ = select_positions(
        salience, retention=0.5, recent_window=2, pool_width=3)

    assert first == second
    assert first == tuple(sorted(set(first)))
    assert len(first) == 5
    assert first[-2:] == (8, 9)
    assert min(first) >= 0 and max(first) < len(salience)
    assert pooled.shape == (len(salience),)


@pytest.mark.parametrize("mutation, match", [
    (lambda engine: setattr(engine.cfg, "model_type", "llama"), "model_type"),
    (lambda engine: setattr(engine.cfg, "num_experts", 2), "MoE"),
    (lambda engine: setattr(engine.cfg, "vision_config", {"depth": 1}), "vision"),
    (lambda engine: setattr(engine.rc, "hot_prompt_kv", True), "hot_prompt_kv"),
    (lambda engine: setattr(engine.rc, "tool_pic", True), "tool_pic"),
    (lambda engine: setattr(engine, "last_kv", object()), "request/speculative"),
])
def test_admission_fails_closed_for_unsupported_architecture_or_state(
        mutation, match):
    engine = _TinyQwen()
    mutation(engine)
    with pytest.raises(UnsupportedFastKVTSP, match=match):
        validate_admission(engine, _profile(), prompt_length=10)


def test_two_stage_prefill_has_per_layer_positions_and_no_engine_state_leak():
    engine = _TinyQwen()
    before = engine_state_snapshot(engine)
    prefill = run_prefill(engine, list(range(1, 11)), _profile())

    assert engine_state_snapshot(engine) == before
    assert [layer.count for layer in prefill.state.layers] == [10, 5, 5]
    assert prefill.state.layers[0].positions.tolist() == list(range(10))
    assert prefill.state.layers[1].positions.tolist() == list(
        prefill.selected_positions)
    assert prefill.selected_positions[-2:] == (8, 9)
    metadata = prefill.state.metadata()
    assert metadata == {
        "profile": _profile().name,
        "approximate": True,
        "exact": False,
        "persistent": False,
        "reusable": False,
        "cache_scope": "fixture-local-request-only",
    }

    generated = generate_greedy(engine, prefill, max_tokens=2)
    assert engine_state_snapshot(engine) == before
    # The final emitted token is deliberately not fed. One decode position was
    # appended independently to each layer's existing position vector.
    assert [layer.count for layer in generated.state.layers] == [11, 6, 6]
    assert generated.state.layers[0].positions[-1].item() == 10
    assert generated.state.layers[1].positions[-1].item() == 10


def test_dense_fixture_matches_standard_qwen_block_math():
    engine = _TinyQwen()
    tokens = [1, 2, 3, 4, 5, 6]
    candidate = run_prefill(
        engine, tokens, _profile(retention=1.0, query_chunk=2))

    reference_kv = KVCache(engine.cfg.num_hidden_layers)
    x = engine._embed(tokens)
    for layer in range(engine.cfg.num_hidden_layers):
        x = layer_runner.run_block(
            x, engine.cache.weights, f"model.layers.{layer}", engine.cfg,
            reference_kv, layer, 0,
            mlp_last_only=(layer == engine.cfg.num_hidden_layers - 1))
        mx.eval(x)
    reference_logits = layer_runner.final_logits(
        x, engine._norm_w, engine._head, engine.cfg.rms_norm_eps)
    mx.eval(reference_logits, candidate.logits)

    assert mx.allclose(candidate.logits, reference_logits, rtol=2e-5, atol=2e-5)
    for layer in range(engine.cfg.num_hidden_layers):
        assert mx.allclose(
            candidate.state.layers[layer].keys, reference_kv.keys[layer],
            rtol=2e-5, atol=2e-5)
        assert mx.allclose(
            candidate.state.layers[layer].values, reference_kv.values[layer],
            rtol=2e-5, atol=2e-5)


def test_isolation_guard_detects_even_fixture_callback_mutation():
    engine = _TinyQwen()

    def mutate():
        engine._hot_prompt_slots.append(object())

    with pytest.raises(StateIsolationError, match="_hot_prompt_slots"):
        isolated_engine_call(engine, mutate)


def test_preregistered_nll_gate_cannot_silently_widen_after_observation():
    assert PREREGISTERED_MAX_NLL_DELTA == 0.02
    assert preregistered_nll_gate(0.02)
    assert preregistered_nll_gate(-0.01)
    assert not preregistered_nll_gate(0.0200001)
    assert not preregistered_nll_gate(float("nan"))
