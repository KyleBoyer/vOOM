"""Small real-math gates for experimental multi-request Qwen scheduling."""

from __future__ import annotations

from dataclasses import replace
import math
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from runtime.config import ModelConfig
from runtime.kda_state import KDAStateCache
from runtime.kv_cache import KVCache, fork_hybrid_kv_endpoint
from runtime.lm_head_stream import StreamedLMHead
from runtime.qwen35 import final_logits, run_qwen35_block
from runtime.qwen35_multi_request import (
    QwenLayerStationaryRequest,
    QwenLayerStationaryScheduler,
)
from runtime.qwen35_multi_request_server import (
    QwenMultiRequestItem,
    run_qwen_multi_request_batch,
)
from runtime.sampler import SamplingParams, sample


HIDDEN = 16
HEADS = 2
KV_HEADS = 1
HEAD_DIM = 8
KEY_HEADS = 1
VALUE_HEADS = 2
LINEAR_DIM = 8
INTERMEDIATE = 12
EXPERTS = 4
TOP_K = 2
VOCAB = 29


def _config(*, moe: bool) -> ModelConfig:
    return ModelConfig(
        model_type="qwen3_5_moe" if moe else "qwen3_5",
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=2,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        vocab_size=VOCAB,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
        max_position_embeddings=256,
        tie_word_embeddings=False,
        attention_bias=False,
        head_dim=HEAD_DIM,
        eos_token_ids=(),
        torch_dtype="float32",
        num_experts=EXPERTS if moe else 0,
        num_experts_per_tok=TOP_K if moe else 0,
        moe_intermediate_size=INTERMEDIATE,
        layer_types=("linear_attention", "full_attention"),
        linear_num_key_heads=KEY_HEADS,
        linear_num_value_heads=VALUE_HEADS,
        linear_key_head_dim=LINEAR_DIM,
        linear_value_head_dim=LINEAR_DIM,
        linear_conv_kernel_dim=4,
        shared_expert_intermediate_size=INTERMEDIATE if moe else 0,
        partial_rotary_factor=0.5,
        attn_output_gate=True,
        rope_scaling={"mrope_section": [1, 1, 0]},
    )


def _array(rng, shape, scale=0.08):
    return mx.array(rng.normal(0.0, scale, shape).astype(np.float32))


def _linear_weights(rng, prefix: str) -> dict[str, mx.array]:
    mixed = 2 * KEY_HEADS * LINEAR_DIM + VALUE_HEADS * LINEAR_DIM
    value_width = VALUE_HEADS * LINEAR_DIM
    return {
        f"{prefix}.input_layernorm.weight": _array(rng, (HIDDEN,)),
        f"{prefix}.post_attention_layernorm.weight": _array(rng, (HIDDEN,)),
        f"{prefix}.linear_attn.in_proj_qkv.weight": _array(
            rng, (mixed, HIDDEN)),
        f"{prefix}.linear_attn.conv1d.weight": _array(rng, (mixed, 1, 4)),
        f"{prefix}.linear_attn.in_proj_z.weight": _array(
            rng, (value_width, HIDDEN)),
        f"{prefix}.linear_attn.in_proj_b.weight": _array(
            rng, (VALUE_HEADS, HIDDEN)),
        f"{prefix}.linear_attn.in_proj_a.weight": _array(
            rng, (VALUE_HEADS, HIDDEN)),
        f"{prefix}.linear_attn.dt_bias": _array(rng, (VALUE_HEADS,)),
        f"{prefix}.linear_attn.A_log": mx.array(
            np.log(rng.uniform(1.0, 2.0, VALUE_HEADS)).astype(np.float32)),
        f"{prefix}.linear_attn.norm.weight": _array(
            rng, (LINEAR_DIM,)),
        f"{prefix}.linear_attn.out_proj.weight": _array(
            rng, (HIDDEN, value_width)),
    }


def _attention_weights(rng, prefix: str) -> dict[str, mx.array]:
    return {
        f"{prefix}.input_layernorm.weight": _array(rng, (HIDDEN,)),
        f"{prefix}.post_attention_layernorm.weight": _array(rng, (HIDDEN,)),
        f"{prefix}.self_attn.q_proj.weight": _array(
            rng, (HEADS * 2 * HEAD_DIM, HIDDEN)),
        f"{prefix}.self_attn.k_proj.weight": _array(
            rng, (KV_HEADS * HEAD_DIM, HIDDEN)),
        f"{prefix}.self_attn.v_proj.weight": _array(
            rng, (KV_HEADS * HEAD_DIM, HIDDEN)),
        f"{prefix}.self_attn.o_proj.weight": _array(
            rng, (HIDDEN, HEADS * HEAD_DIM)),
        f"{prefix}.self_attn.q_norm.weight": _array(rng, (HEAD_DIM,)),
        f"{prefix}.self_attn.k_norm.weight": _array(rng, (HEAD_DIM,)),
    }


def _dense_mlp_weights(rng, prefix: str) -> dict[str, mx.array]:
    return {
        f"{prefix}.mlp.gate_proj.weight": _array(
            rng, (INTERMEDIATE, HIDDEN)),
        f"{prefix}.mlp.up_proj.weight": _array(
            rng, (INTERMEDIATE, HIDDEN)),
        f"{prefix}.mlp.down_proj.weight": _array(
            rng, (HIDDEN, INTERMEDIATE)),
    }


def _moe_weights(rng, prefix: str):
    trunk = {
        f"{prefix}.mlp.gate.weight": _array(rng, (EXPERTS, HIDDEN)),
        f"{prefix}.mlp.shared_expert.gate_proj.weight": _array(
            rng, (INTERMEDIATE, HIDDEN)),
        f"{prefix}.mlp.shared_expert.up_proj.weight": _array(
            rng, (INTERMEDIATE, HIDDEN)),
        f"{prefix}.mlp.shared_expert.down_proj.weight": _array(
            rng, (HIDDEN, INTERMEDIATE)),
        f"{prefix}.mlp.shared_expert_gate.weight": _array(rng, (1, HIDDEN)),
    }
    experts = {}
    for expert in range(EXPERTS):
        expert_prefix = f"{prefix}.mlp.experts.{expert}"
        experts[expert] = {
            f"{expert_prefix}.gate_proj.weight": _array(
                rng, (INTERMEDIATE, HIDDEN)),
            f"{expert_prefix}.up_proj.weight": _array(
                rng, (INTERMEDIATE, HIDDEN)),
            f"{expert_prefix}.down_proj.weight": _array(
                rng, (HIDDEN, INTERMEDIATE)),
        }
    return trunk, experts


class _Cache:
    def __init__(self, layers):
        self.layers = layers
        self.stats = SimpleNamespace(
            hits=0, misses=0, evictions=0, disk_s=0.0, bytes_read=0)
        self.get_counts = {}

    def contains(self, _key):
        return False

    def prepare_for(self, _incoming):
        pass

    def get(self, key, _names):
        self.get_counts[key] = self.get_counts.get(key, 0) + 1
        self.stats.misses += 1
        weights = self.layers[int(key.removeprefix("layer."))]
        self.stats.bytes_read += sum(value.nbytes for value in weights.values())
        return weights

    def reset(self):
        self.stats = SimpleNamespace(
            hits=0, misses=0, evictions=0, disk_s=0.0, bytes_read=0)
        self.get_counts = {}


class _Engine:
    def __init__(self, *, moe: bool):
        self.cfg = _config(moe=moe)
        rng = np.random.default_rng(940 if moe else 941)
        layer0 = _linear_weights(rng, "model.layers.0")
        layer1 = _attention_weights(rng, "model.layers.1")
        self.experts = {}
        for layer, weights in enumerate((layer0, layer1)):
            prefix = f"model.layers.{layer}"
            if moe:
                trunk, experts = _moe_weights(rng, prefix)
                weights.update(trunk)
                self.experts[layer] = experts
            else:
                weights.update(_dense_mlp_weights(rng, prefix))
        self.cache = _Cache((layer0, layer1))
        self.embedding = _array(rng, (VOCAB, HIDDEN), scale=0.15)
        self._norm_w = _array(rng, (HIDDEN,))
        self.head = _array(rng, (VOCAB, HIDDEN), scale=0.1)
        self._iter_expert_batches = None
        self.governor = None
        self.timer = None
        self._layer_transient = 0
        self._layer_transient_margin = 0
        self.rc = SimpleNamespace(
            zmlx_fused_deltanet_decode=False,
            native_fused_deltanet_decode=False,
            qwen_chunked_delta_prefill=False,
            qwen_compiled_delta_prefill=False,
            qwen_native_fused_delta_prefill=False,
        )

    def new_kv(self):
        kv = KVCache(self.cfg.num_hidden_layers)
        kv.kda_cache = KDAStateCache(self.cfg.num_hidden_layers)
        return kv

    def _embed(self, tokens):
        return mx.take(self.embedding, mx.array(tokens), axis=0)[None, :, :]

    def _layer_key(self, layer):
        return f"layer.{layer}"

    def _layer_names(self, layer):
        return list(self.cache.layers[layer])

    def _layer_fetch_bytes_estimate(self, layer):
        return sum(value.nbytes for value in self.cache.layers[layer].values())

    def _get_experts(self, layer, ids, positions=None):
        selected = {expert: self.experts[layer][expert] for expert in ids}
        self.cache.stats.bytes_read += sum(
            value.nbytes
            for expert in selected.values()
            for value in expert.values()
        )
        return selected

    def _select_serial_verify_layer_transient(self, _count, _layer):
        return 0

    def _record_serial_verify_layer_transient(
        self, _count, _layer, measured,
    ):
        return measured

    def _note_true_peak(self):
        pass

    def _lm_head_weight(self):
        return self.head

    def _final_logits(self, hidden, head=None):
        return final_logits(
            hidden,
            self._norm_w,
            self.head if head is None else head,
            self.cfg.rms_norm_eps,
        )


def _serial_advance(engine: _Engine, token: int, kv, *, positions3=None):
    hidden = engine._embed([token])
    offset = kv.offset
    for layer in range(engine.cfg.num_hidden_layers):
        weights = engine.cache.get(
            engine._layer_key(layer), engine._layer_names(layer))
        hidden = run_qwen35_block(
            hidden,
            weights,
            f"model.layers.{layer}",
            engine.cfg,
            kv,
            layer,
            offset,
            engine._get_experts,
            positions3=positions3,
        )
        mx.eval(hidden)
    logits = engine._final_logits(hidden)
    mx.eval(logits)
    return logits


def _snapshot(kv):
    result = []
    for value in (*kv.keys, *kv.values):
        result.append(None if value is None else np.array(value))
    for layer in range(len(kv.keys)):
        state = kv.kda_cache.state(layer)
        result.append(None if state is None else np.array(state))
        history = kv.kda_cache.conv_history(layer)
        if history is None:
            result.append(None)
        else:
            result.append(tuple(np.array(value) for value in history))
    return result


def _assert_snapshot_equal(left, right):
    assert len(left) == len(right)
    for actual, expected in zip(left, right, strict=True):
        if actual is None or expected is None:
            assert actual is expected
        elif isinstance(actual, tuple):
            assert len(actual) == len(expected)
            for avalue, evalue in zip(actual, expected, strict=True):
                assert np.array_equal(avalue, evalue)
        else:
            assert np.array_equal(actual, expected)


@pytest.mark.parametrize("moe", [False, True])
def test_multi_request_logits_and_hybrid_state_are_byte_exact(moe):
    engine = _Engine(moe=moe)
    base_a, base_b = engine.new_kv(), engine.new_kv()
    _serial_advance(engine, 2, base_a)
    _serial_advance(engine, 7, base_b)
    base_a_snapshot, base_b_snapshot = _snapshot(base_a), _snapshot(base_b)

    serial_a = fork_hybrid_kv_endpoint(base_a)
    serial_b = fork_hybrid_kv_endpoint(base_b)
    engine.cache.reset()
    expected_a = _serial_advance(engine, 11, serial_a)
    expected_b = _serial_advance(engine, 13, serial_b)
    assert engine.cache.get_counts == {"layer.0": 2, "layer.1": 2}

    batch_a = fork_hybrid_kv_endpoint(base_a)
    batch_b = fork_hybrid_kv_endpoint(base_b)
    engine.cache.reset()
    positions = np.array([[1], [1], [1]], dtype=np.int32)
    result = QwenLayerStationaryScheduler(engine, max_requests=2).advance((
        QwenLayerStationaryRequest("a", 11, batch_a, positions),
        QwenLayerStationaryRequest("b", 13, batch_b),
    ))
    by_id = result.by_request_id()

    assert np.array_equal(np.array(by_id["a"].logits), np.array(expected_a))
    assert np.array_equal(np.array(by_id["b"].logits), np.array(expected_b))
    assert by_id["a"].greedy_token == int(mx.argmax(expected_a).item())
    assert by_id["b"].greedy_token == int(mx.argmax(expected_b).item())
    _assert_snapshot_equal(_snapshot(batch_a), _snapshot(serial_a))
    _assert_snapshot_equal(_snapshot(batch_b), _snapshot(serial_b))

    # Advancing forks cannot mutate either stable source or one another.
    _assert_snapshot_equal(_snapshot(base_a), base_a_snapshot)
    _assert_snapshot_equal(_snapshot(base_b), base_b_snapshot)
    assert batch_a is not batch_b
    assert batch_a.kda_cache is not batch_b.kda_cache
    assert batch_a.kda_cache.state(0) is not batch_b.kda_cache.state(0)

    telemetry = result.telemetry
    assert engine.cache.get_counts == {"layer.0": 1, "layer.1": 1}
    assert telemetry.request_count == 2
    assert telemetry.layer_count == 2
    assert telemetry.request_tokens == 2
    assert telemetry.layer_page_get_calls == 2
    assert telemetry.serial_equivalent_layer_page_get_calls == 4
    assert telemetry.layer_page_get_call_savings == 2
    assert telemetry.layer_page_get_call_reduction_fraction == 0.5
    assert telemetry.target_layer_page_bytes_read > 0
    assert telemetry.total_cache_bytes_read >= telemetry.target_layer_page_bytes_read
    assert telemetry.expert_or_other_bytes_read == (
        telemetry.total_cache_bytes_read
        - telemetry.target_layer_page_bytes_read)
    assert telemetry.bytes_read_per_request_token == (
        telemetry.total_weight_bytes_read / 2)
    assert telemetry.request_tokens_per_second > 0
    if moe:
        assert telemetry.expert_or_other_bytes_read > 0
    else:
        assert telemetry.expert_or_other_bytes_read == 0


def test_request_permutation_does_not_change_outputs_or_state():
    engine = _Engine(moe=True)
    base = {"a": engine.new_kv(), "b": engine.new_kv(), "c": engine.new_kv()}
    for token, kv in zip((2, 3, 5), base.values(), strict=True):
        _serial_advance(engine, token, kv)
    tokens = {"a": 17, "b": 19, "c": 23}

    def run(order):
        endpoints = {
            request_id: fork_hybrid_kv_endpoint(base[request_id])
            for request_id in order
        }
        engine.cache.reset()
        result = QwenLayerStationaryScheduler(engine, max_requests=3).advance([
            QwenLayerStationaryRequest(
                request_id, tokens[request_id], endpoints[request_id])
            for request_id in order
        ])
        return result.by_request_id(), {
            request_id: _snapshot(kv) for request_id, kv in endpoints.items()
        }

    forward, forward_state = run(("a", "b", "c"))
    reverse, reverse_state = run(("c", "b", "a"))
    for request_id in tokens:
        assert np.array_equal(
            np.array(forward[request_id].logits),
            np.array(reverse[request_id].logits),
        )
        assert forward[request_id].greedy_token == reverse[request_id].greedy_token
        _assert_snapshot_equal(
            forward_state[request_id], reverse_state[request_id])


def test_streamed_lm_head_is_shared_without_batching_row_matmuls(
        tmp_path, monkeypatch):
    import runtime.lm_head_stream as lm_head_module

    engine = _Engine(moe=False)
    base_a, base_b = engine.new_kv(), engine.new_kv()
    _serial_advance(engine, 2, base_a)
    _serial_advance(engine, 7, base_b)
    expected_a_kv = fork_hybrid_kv_endpoint(base_a)
    expected_b_kv = fork_hybrid_kv_endpoint(base_b)
    expected_a = _serial_advance(engine, 11, expected_a_kv)
    expected_b = _serial_advance(engine, 13, expected_b_kv)

    mx.save_safetensors(
        str(tmp_path / "model.safetensors"),
        {"lm_head.weight": engine.head},
    )
    streamed = StreamedLMHead(
        tmp_path,
        {"lm_head.weight": "model.safetensors"},
        block_rows=5,
    )
    engine.head = streamed
    read_calls = 0
    original = lm_head_module._pread_exact

    def counted(*args, **kwargs):
        nonlocal read_calls
        read_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(lm_head_module, "_pread_exact", counted)
    try:
        result = QwenLayerStationaryScheduler(engine, max_requests=2).advance((
            QwenLayerStationaryRequest(
                "a", 11, fork_hybrid_kv_endpoint(base_a)),
            QwenLayerStationaryRequest(
                "b", 13, fork_hybrid_kv_endpoint(base_b)),
        ))
        actual = result.by_request_id()
        assert np.array_equal(np.array(actual["a"].logits), np.array(expected_a))
        assert np.array_equal(np.array(actual["b"].logits), np.array(expected_b))
        assert read_calls == math.ceil(VOCAB / 5)
        assert result.telemetry.shared_streamed_lm_head
        assert result.telemetry.streamed_lm_head_read_calls == read_calls
        assert result.telemetry.streamed_lm_head_bytes_read == VOCAB * HIDDEN * 4
        assert result.telemetry.total_weight_bytes_read == (
            result.telemetry.total_cache_bytes_read + VOCAB * HIDDEN * 4)
    finally:
        streamed.close()


def test_scheduler_fails_closed_on_unbounded_or_shared_state():
    engine = _Engine(moe=False)
    with pytest.raises(ValueError, match="max_requests"):
        QwenLayerStationaryScheduler(engine, max_requests=0)
    with pytest.raises(ValueError, match="max_requests"):
        QwenLayerStationaryScheduler(engine, max_requests=17)

    scheduler = QwenLayerStationaryScheduler(engine, max_requests=1)
    first, second = engine.new_kv(), engine.new_kv()
    with pytest.raises(ValueError, match="configured maximum"):
        scheduler.advance((
            QwenLayerStationaryRequest("a", 1, first),
            QwenLayerStationaryRequest("b", 2, second),
        ))

    scheduler = QwenLayerStationaryScheduler(engine, max_requests=2)
    with pytest.raises(ValueError, match="share a KV"):
        scheduler.advance((
            QwenLayerStationaryRequest("a", 1, first),
            QwenLayerStationaryRequest("b", 2, first),
        ))
    shared_kda = second.kda_cache
    first.kda_cache = shared_kda
    with pytest.raises(ValueError, match="share a KDA"):
        scheduler.advance((
            QwenLayerStationaryRequest("a", 1, first),
            QwenLayerStationaryRequest("b", 2, second),
        ))
    with pytest.raises(ValueError, match="outside"):
        scheduler.advance((QwenLayerStationaryRequest("a", True, first),))

    other = _Engine(moe=False)
    other.cfg = replace(other.cfg, model_type="qwen2")
    with pytest.raises(ValueError, match="supports only"):
        QwenLayerStationaryScheduler(other).advance((
            QwenLayerStationaryRequest("a", 1, other.new_kv()),
        ))


class _NumericTokenizer:
    def decode(self, tokens):
        return "|".join(str(int(value)) for value in tokens)


def test_generation_coordinator_handles_heterogeneous_requests_exactly():
    engine = _Engine(moe=False)
    engine.tokenizer = _NumericTokenizer()
    base_a, base_b = engine.new_kv(), engine.new_kv()
    _serial_advance(engine, 2, base_a)
    _serial_advance(engine, 7, base_b)
    base_snapshots = (_snapshot(base_a), _snapshot(base_b))
    params_a = SamplingParams(temperature=0, repetition_penalty=1.1)
    params_b = SamplingParams(temperature=0)
    items = (
        QwenMultiRequestItem(
            "a", "prompt-a", (2,), 2, params_a),
        QwenMultiRequestItem(
            "b", "a different prompt", (7, 3), 3, params_b),
    )

    sources = {"prompt-a": base_a, "a different prompt": base_b}
    first_tokens = {"prompt-a": 11, "a different prompt": 13}

    def bootstrap(prompt, max_tokens, **_kwargs):
        assert max_tokens == 1
        engine.last_kv = fork_hybrid_kv_endpoint(sources[str(prompt)])
        return {
            "tokens": [first_tokens[str(prompt)]],
            "termination_reason": "length",
            "stop_sequence": None,
        }

    # Independent serial oracles use the same ordinary one-row sampler and
    # per-request history as the coordinator.
    serial_a = fork_hybrid_kv_endpoint(base_a)
    logits_a = _serial_advance(engine, 11, serial_a)
    expected_a = [11, sample(logits_a, params_a, history=(2, 11))]
    serial_b = fork_hybrid_kv_endpoint(base_b)
    logits_b1 = _serial_advance(engine, 13, serial_b)
    expected_b = [13, sample(logits_b1, params_b, history=(7, 3, 13))]
    logits_b2 = _serial_advance(engine, expected_b[-1], serial_b)
    expected_b.append(sample(
        logits_b2, params_b, history=(7, 3, *expected_b)))

    engine.cache.reset()
    result = run_qwen_multi_request_batch(
        engine, items, max_requests=2, bootstrap_generate=bootstrap)
    by_id = {choice["id"]: choice for choice in result["choices"]}
    assert by_id["a"]["tokens"] == expected_a
    assert by_id["b"]["tokens"] == expected_b
    assert by_id["a"]["completion_tokens"] == 2
    assert by_id["b"]["completion_tokens"] == 3
    assert by_id["a"]["prompt_tokens"] == 1
    assert by_id["b"]["prompt_tokens"] == 2

    telemetry = result["telemetry"]
    assert telemetry["scheduler_rounds"] == 2
    assert telemetry["scheduled_request_tokens"] == 3
    assert telemetry["layer_page_get_calls"] == 4
    assert telemetry["serial_equivalent_layer_page_get_calls"] == 6
    assert telemetry["layer_page_get_call_savings"] == 2
    assert telemetry["private_kv_endpoints"] == 2
    assert telemetry["private_kda_endpoints"] == 2
    assert telemetry["cache_identity_policy"] == (
        "private-kv-and-kda-per-request")
    assert telemetry["prompt_cache_policy"] == "disabled-during-bootstrap"
    assert engine.last_kv is None
    _assert_snapshot_equal(_snapshot(base_a), base_snapshots[0])
    _assert_snapshot_equal(_snapshot(base_b), base_snapshots[1])
