from __future__ import annotations

import json
import math
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from runtime.config import ModelConfig
from runtime.qwen4_exp import (
    qwen4_mlp_from_group_batches,
    qwen4_mlp_from_groups,
    qwen4_rms_norm,
    qwen4_mlp_route,
    qwen4_mlp_route_window_exact,
)
from runtime.qwen4_mtp import (
    Qwen4MTPDrafter,
    Qwen4MTPSpeculativeEngine,
    _REQUIRED_NON_EXPERT_NAMES,
    _verify_stochastic_token,
)
from runtime.sampler import SamplingParams


def _config() -> ModelConfig:
    return ModelConfig(
        model_type="qwen4_exp",
        hidden_size=2,
        intermediate_size=3,
        num_hidden_layers=2,
        num_attention_heads=1,
        num_key_value_heads=1,
        vocab_size=3,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
        max_position_embeddings=128,
        tie_word_embeddings=False,
        attention_bias=False,
        head_dim=2,
        eos_token_ids=(2,),
        torch_dtype="bfloat16",
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=2,
        layer_types=("linear_attention", "full_attention"),
        qwen4_hc_count=2,
        qwen4_hc_lowrank=1,
        qwen4_ple_layers=(0,),
        qwen4_indexer_budget=2,
        qwen4_indexer_compress_ratio=1,
        qwen4_indexer_head_dim=2,
        qwen4_indexer_kv_heads=1,
        qwen4_indexer_n_heads=1,
    )


class _Store:
    def __init__(self, *, complete: bool = True):
        names = set(_REQUIRED_NON_EXPERT_NAMES)
        if not complete:
            names.remove("mtp.fc_hidden.weight")
        for expert in range(2):
            names.update(
                f"mtp.layers.0.mlp.experts.{expert}.{projection}.weight"
                for projection in ("gate_proj", "up_proj", "down_proj")
            )
        self.names = tuple(sorted(names))

    def names_with_prefix(self, prefix):
        return [name for name in self.names if name.startswith(prefix)]

    def storage_bytes(self, names):
        return 2 * len(names)


class _Cache:
    def __init__(self, weights):
        self.weights = weights
        self.total_bytes = 0
        self.discards = []

    def get(self, key, names, *, apply_transform=True):
        assert key == "qwen4_mtp:released-bf16"
        assert not apply_transform
        return {name: self.weights[name] for name in names}

    def get_many(self, items):
        result = {}
        for key, names in items:
            result[key] = {
                name: mx.zeros((1, 1), dtype=mx.bfloat16)
                for name in names
            }
        self.total_bytes += 2 * sum(len(names) for _, names in items)
        return result

    def discard(self, key, names=()):
        self.discards.append((key, tuple(names)))
        return True


def _weights(cfg):
    values = {
        name: mx.zeros((1,), dtype=mx.bfloat16)
        for name in _REQUIRED_NON_EXPERT_NAMES
    }
    values.update({
        "mtp.pre_fc_norm_embedding.weight": mx.zeros(
            (cfg.hidden_size,), dtype=mx.bfloat16),
        "mtp.pre_fc_norm_hidden.weight": mx.zeros(
            (cfg.hidden_size * cfg.qwen4_hc_count,), dtype=mx.bfloat16),
        "mtp.fc_embedding.weight": mx.array(
            [[1.0, 0.0], [0.0, 1.0]], dtype=mx.bfloat16),
        "mtp.fc_hidden.weight": mx.array(
            [[0.0, 1.0], [1.0, 0.0]], dtype=mx.bfloat16),
        "mtp.hyper_connection_mixer.hc_norm.weight": mx.zeros(
            (cfg.hidden_size * cfg.qwen4_hc_count,), dtype=mx.bfloat16),
        "mtp.hyper_connection_mixer.input_mix_weight_down.weight": mx.zeros(
            (cfg.qwen4_hc_lowrank,
             cfg.hidden_size * cfg.qwen4_hc_count), dtype=mx.bfloat16),
        "mtp.hyper_connection_mixer.input_mix_weight_up.weight": mx.zeros(
            (cfg.hidden_size * cfg.qwen4_hc_count,
             cfg.qwen4_hc_lowrank), dtype=mx.bfloat16),
    })
    return values


def _engine(*, complete=True):
    cfg = _config()
    weights = _weights(cfg)
    embeddings = mx.array(
        [[0.25, 0.75], [1.0, -0.5], [-0.25, 0.5]],
        dtype=mx.bfloat16,
    )
    cache = _Cache(weights)
    return SimpleNamespace(
        cfg=cfg,
        store=_Store(complete=complete),
        cache=cache,
        _embed_rows=SimpleNamespace(
            lookup=lambda tokens: embeddings[mx.array(tokens)][None]),
        _embed_weight=lambda: embeddings,
        _lm_head_weight=lambda: mx.array(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.5]],
            dtype=mx.bfloat16,
        ),
    ), weights


def test_fuse_inputs_projects_each_hidden_stream_with_shared_fc():
    engine, weights = _engine()
    drafter = Qwen4MTPDrafter(engine)
    embedding = engine._embed_rows.lookup([1])
    hidden = mx.array(
        [[[1.0, 2.0, 3.0, 4.0]]], dtype=mx.bfloat16)

    actual = drafter.fuse_inputs(embedding, hidden, weights)

    projected_embedding = qwen4_rms_norm(
        embedding,
        weights["mtp.pre_fc_norm_embedding.weight"],
        engine.cfg.rms_norm_eps,
    )
    streams = qwen4_rms_norm(
        hidden,
        weights["mtp.pre_fc_norm_hidden.weight"],
        engine.cfg.rms_norm_eps,
    ).reshape(1, 1, 2, 2)
    swapped = streams[..., ::-1]
    expected = (projected_embedding[..., None, :] + swapped).reshape(
        hidden.shape)
    np.testing.assert_array_equal(
        np.asarray(actual.view(mx.uint16)),
        np.asarray(expected.view(mx.uint16)),
    )


def test_batched_expert_lifetime_keeps_per_row_bf16_accumulation_exact():
    mx.random.seed(73)
    hidden = 8
    intermediate = 5
    routes = []
    group_sets = ((0, 2, 4), (1, 2, 5), (0, 3, 5), (1, 3, 4))
    for row, groups in enumerate(group_sets):
        mixed = mx.random.normal((1, 1, hidden)).astype(mx.bfloat16)
        hyper_input = mx.random.normal((1, 1, hidden)).astype(mx.bfloat16)
        injection = mx.random.normal((1, 1, 1)).astype(mx.bfloat16)
        routes.append((
            mixed,
            hyper_input,
            injection,
            {
                expert: [(0, 0.125 * (row + expert + 1))]
                for expert in groups
            },
        ))
    experts = {
        expert: {
            f"model.layers.0.mlp.experts.{expert}.gate_proj.weight":
                mx.random.normal((intermediate, hidden)).astype(mx.bfloat16),
            f"model.layers.0.mlp.experts.{expert}.up_proj.weight":
                mx.random.normal((intermediate, hidden)).astype(mx.bfloat16),
            f"model.layers.0.mlp.experts.{expert}.down_proj.weight":
                mx.random.normal((hidden, intermediate)).astype(mx.bfloat16),
        }
        for expert in range(6)
    }
    weights = {
        "model.layers.0.mlp.shared_expert.gate_proj.weight":
            mx.random.normal((intermediate, hidden)).astype(mx.bfloat16),
        "model.layers.0.mlp.shared_expert.up_proj.weight":
            mx.random.normal((intermediate, hidden)).astype(mx.bfloat16),
        "model.layers.0.mlp.shared_expert.down_proj.weight":
            mx.random.normal((hidden, intermediate)).astype(mx.bfloat16),
        "model.layers.0.mlp.shared_expert_gate.weight":
            mx.random.normal((1, hidden)).astype(mx.bfloat16),
    }
    expected = [
        qwen4_mlp_from_groups(
            route, experts, weights, "model.layers.0")
        for route in routes
    ]
    mx.eval(*expected)

    batches = iter((
        ([0, 1], {expert: experts[expert] for expert in (0, 1)}),
        ([2, 3], {expert: experts[expert] for expert in (2, 3)}),
        ([4, 5], {expert: experts[expert] for expert in (4, 5)}),
    ))
    actual = qwen4_mlp_from_group_batches(
        routes, batches, weights, "model.layers.0")
    mx.eval(*actual)

    for expected_row, actual_row in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(
            np.asarray(expected_row.view(mx.uint16)),
            np.asarray(actual_row.view(mx.uint16)),
        )


def test_exact_bf16_mlp_window_matches_singleton_routes_and_experts():
    cfg = ModelConfig(
        model_type="qwen4_exp",
        hidden_size=8,
        intermediate_size=4,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        vocab_size=32,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
        max_position_embeddings=128,
        tie_word_embeddings=False,
        attention_bias=False,
        head_dim=8,
        eos_token_ids=(2,),
        torch_dtype="bfloat16",
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=4,
        layer_types=("linear_attention",),
        qwen4_hc_count=4,
        qwen4_hc_lowrank=4,
    )
    mx.random.seed(113)
    prefix = "model.layers.0"
    width = cfg.hidden_size * cfg.qwen4_hc_count
    weights = {
        f"{prefix}.mlp_hyper_connection.hc_norm.weight":
            mx.random.normal((width,)).astype(mx.bfloat16),
        f"{prefix}.mlp_hyper_connection.input_mix_weight_down.weight":
            mx.random.normal((cfg.qwen4_hc_lowrank, width)).astype(
                mx.bfloat16),
        f"{prefix}.mlp_hyper_connection.input_mix_weight_up.weight":
            mx.random.normal((width, cfg.qwen4_hc_lowrank)).astype(
                mx.bfloat16),
        f"{prefix}.mlp_hyper_connection.block_inject_weight.weight":
            mx.random.normal((cfg.qwen4_hc_count, width)).astype(mx.bfloat16),
        f"{prefix}.mlp.gate.weight":
            mx.random.normal((cfg.num_experts, cfg.hidden_size)).astype(
                mx.bfloat16),
        f"{prefix}.mlp.shared_expert.gate_proj.weight":
            mx.random.normal((cfg.moe_intermediate_size, cfg.hidden_size)).astype(
                mx.bfloat16),
        f"{prefix}.mlp.shared_expert.up_proj.weight":
            mx.random.normal((cfg.moe_intermediate_size, cfg.hidden_size)).astype(
                mx.bfloat16),
        f"{prefix}.mlp.shared_expert.down_proj.weight":
            mx.random.normal((cfg.hidden_size, cfg.moe_intermediate_size)).astype(
                mx.bfloat16),
        f"{prefix}.mlp.shared_expert_gate.weight":
            mx.random.normal((1, cfg.hidden_size)).astype(mx.bfloat16),
    }
    experts = {}
    for expert in range(cfg.num_experts):
        page = {
            f"{prefix}.mlp.experts.{expert}.gate_proj.weight":
                mx.random.normal((
                    cfg.moe_intermediate_size, cfg.hidden_size)).astype(
                        mx.bfloat16),
            f"{prefix}.mlp.experts.{expert}.up_proj.weight":
                mx.random.normal((
                    cfg.moe_intermediate_size, cfg.hidden_size)).astype(
                        mx.bfloat16),
            f"{prefix}.mlp.experts.{expert}.down_proj.weight":
                mx.random.normal((
                    cfg.hidden_size, cfg.moe_intermediate_size)).astype(
                        mx.bfloat16),
        }
        experts[expert] = page
    hidden_rows = [
        mx.random.normal((1, 1, width)).astype(mx.bfloat16)
        for _ in range(4)
    ]
    singleton_routes = [
        qwen4_mlp_route(row, weights, prefix, cfg, 0)
        for row in hidden_rows
    ]
    expected = [
        qwen4_mlp_from_groups(route, experts, weights, prefix)
        for route in singleton_routes
    ]
    stats = {}
    exact_routes = qwen4_mlp_route_window_exact(
        hidden_rows, weights, prefix, cfg, 0, stats=stats)
    actual = qwen4_mlp_from_group_batches(
        exact_routes,
        iter(((list(range(cfg.num_experts)), experts),)),
        weights,
        prefix,
        exact_bf16=True,
        exact_stats=stats,
    )
    mx.eval(*expected, *actual)
    assert [route[3] for route in exact_routes] == [
        route[3] for route in singleton_routes]
    for expected_row, actual_row in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(
            np.asarray(expected_row.view(mx.uint16)),
            np.asarray(actual_row.view(mx.uint16)),
        )
    assert stats["calls"] > 0
    assert stats["rows"] >= 4


def test_draft_step_uses_qsa_only_mtp_layer_and_sparse_expert_pages(monkeypatch):
    engine, weights = _engine()
    drafter = Qwen4MTPDrafter(engine)
    cache = drafter.new_cache()
    hidden = mx.array(
        [[[1.0, 2.0, 3.0, 4.0]]], dtype=mx.bfloat16)
    seen = {}

    def fake_block(
        fused, input_ids, _weights, prefix, cfg, actual_cache, layer,
        offset, _get_experts, **kwargs,
    ):
        seen.update({
            "input_ids": tuple(input_ids),
            "prefix": prefix,
            "layer_types": cfg.layer_types,
            "ple_layers": cfg.qwen4_ple_layers,
            "cache": actual_cache,
            "layer": layer,
            "offset": offset,
        })
        batches = list(kwargs["iter_expert_batches"](
            0, [1], positions={1: [0]}))
        assert batches[0][0] == [1]
        assert len(batches[0][1][1]) == 3
        return fused

    monkeypatch.setattr("runtime.qwen4_mtp.run_qwen4_block", fake_block)
    logits, post_hidden = drafter.draft_step(
        hidden, 1, cache, 17, weights=weights)

    assert tuple(logits.shape) == (engine.cfg.vocab_size,)
    assert tuple(post_hidden.shape) == tuple(hidden.shape)
    assert seen == {
        "input_ids": (1,),
        "prefix": "mtp.layers.0",
        "layer_types": ("full_attention",),
        "ple_layers": (),
        "cache": cache,
        "layer": 0,
        "offset": 17,
    }
    assert drafter.proposal_steps == 1
    assert drafter.proposal_expert_pages == 1

    report = drafter.release_round_weights()
    assert report["released_pages"] == 2
    assert {key for key, _names in engine.cache.discards} == {
        "qwen4_mtp.expert.1",
        "qwen4_mtp:released-bf16",
    }


def test_mtp_adapter_fails_closed_on_incomplete_checkpoint():
    engine, _weights_value = _engine(complete=False)
    with pytest.raises(ValueError, match="checkpoint is incomplete"):
        Qwen4MTPDrafter(engine)


class _FakeRecurrent:
    def __init__(self, marker="full"):
        self.marker = marker

    def synchronize(self):
        pass

    def fork(self):
        return _FakeRecurrent(self.marker)


class _FakeFactorWindow:
    def __init__(self, width):
        self.width = width
        self.commits = []

    def commit_prefix(self, base, count):
        assert count <= self.width
        self.commits.append((base.marker, count))
        return _FakeRecurrent(f"factor-{count}")


class _FakeAux:
    def __init__(self):
        self.restores = []
        self.trims = []

    def restore_recurrent_prefix(self, endpoint, length):
        self.restores.append((endpoint.marker, length))

    def trim(self, length):
        self.trims.append(length)

    def synchronize(self):
        pass


class _FakeKV:
    def __init__(self, offset):
        self._lengths = [offset]
        self.kda_cache = _FakeRecurrent()
        self.qwen4_cache = _FakeAux()

    @property
    def offset(self):
        return self._lengths[0]

    def layer_lengths(self):
        return tuple(self._lengths)

    def grow(self, count):
        self._lengths[0] += count

    def trim_layer_lengths(self, lengths):
        self._lengths = list(lengths)

    def nbytes(self):
        return 0


class _FakeTokenizer:
    def __init__(self, ids=(1, 2)):
        self.ids = tuple(ids)

    def encode(self, _prompt):
        return SimpleNamespace(ids=list(self.ids))

    def decode(self, tokens):
        return " ".join(str(token) for token in tokens)


class _FakeTarget:
    def __init__(self, target_rows):
        self.cfg = SimpleNamespace(
            model_type="qwen4_exp", eos_token_ids=(255,), vocab_size=256)
        self.tokenizer = _FakeTokenizer()
        self.effective_max_position_embeddings = 128
        self.target_rows = list(target_rows)
        self.last_kv = None
        self._hot_prompt_slots = []
        self._true_peak_metal_bytes = 0
        self.governor = None
        self.cache = SimpleNamespace()
        self.verify_calls = []
        self._kda_endpoints = None
        self._kda_factors = None
        self._aux_endpoints = None
        self.idle_head_releases = 0
        self._request_profiler = None

    def generate(self, _prompt, max_tokens, **_kwargs):
        assert max_tokens == 1
        self.last_kv = _FakeKV(len(self.tokenizer.encode(_prompt).ids))
        self._h_last = mx.ones((1, 1, 4), dtype=mx.bfloat16)
        self._h_window = self._h_last
        return {
            "text": "10", "tokens": [10], "prefill_s": 1.0,
            "first_token_s": 1.0, "termination_reason": "length",
            "path_stats": {},
        }

    def forward_tokens_serial_positions(self, tokens, kv, **kwargs):
        assert kwargs in ({
            "capture_kda_endpoints": True,
            "capture_kda_factors": False,
            "capture_qwen4_endpoints": True,
        }, {
            "capture_kda_endpoints": False,
            "capture_kda_factors": True,
            "capture_qwen4_endpoints": True,
        })
        self.verify_calls.append(tuple(tokens))
        width = len(tokens)
        kv.grow(width)
        self._h_window = mx.arange(
            width * 4, dtype=mx.float32).reshape(1, width, 4).astype(
                mx.bfloat16)
        self._h_last = self._h_window[:, -1:]
        self._kda_endpoints = [
            _FakeRecurrent(f"kda-{index}") for index in range(1, width)
        ]
        self._kda_factors = (
            _FakeFactorWindow(width)
            if kwargs["capture_kda_factors"] else None)
        self._serial_kda_factor_retained_bytes = (
            11 * width if self._kda_factors is not None else 0)
        self._serial_kda_endpoint_retained_bytes = (
            101 * (width - 1) if self._kda_factors is None else 0)
        self._aux_endpoints = [
            SimpleNamespace(marker=f"aux-{index}")
            for index in range(1, width)
        ]
        rows = []
        for token in self.target_rows[:width]:
            row = mx.full((256,), -100.0)
            row = row.at[int(token)].add(200.0)
            rows.append(row)
        return mx.stack(rows)

    def consume_serial_kda_endpoint(self, count):
        endpoints, self._kda_endpoints = self._kda_endpoints, None
        return None if count is None else endpoints[count - 1]

    def consume_serial_kda_factors(self):
        factors, self._kda_factors = self._kda_factors, None
        self._serial_kda_factor_retained_bytes = 0
        return factors

    def consume_serial_qwen4_endpoint(self, count):
        endpoints, self._aux_endpoints = self._aux_endpoints, None
        return None if count is None else endpoints[count - 1]

    def _append_hot_prompt_slot(self, _slot):
        pass

    def _suspend_qwen4_phase_lm_head(self):
        self.idle_head_releases += 1
        return 123

    def close(self):
        pass


class _FakeDraftCache(_FakeKV):
    def __init__(self):
        super().__init__(0)


class _FakeDrafter:
    def __init__(self, draft_tokens):
        self.draft_tokens = list(draft_tokens)
        self.index = 0
        self.non_expert_storage_bytes = 181
        self.proposal_steps = 0

    def new_cache(self):
        return _FakeDraftCache()

    def _weights(self):
        return {}

    def draft_step(self, hidden, _token, cache, _offset, *, weights):
        assert weights == {}
        token = self.draft_tokens[self.index % len(self.draft_tokens)]
        self.index += 1
        self.proposal_steps += 1
        cache.grow(1)
        row = mx.full((256,), -100.0)
        row = row.at[token].add(200.0)
        return row, hidden + 1

    def release_round_weights(self):
        return {
            "proposal_expert_pages": self.proposal_steps,
            "proposal_expert_bytes": self.proposal_steps * 98,
        }


class _ConfidenceDrafter(_FakeDrafter):
    def __init__(self, draft_rows):
        super().__init__([token for token, _probability in draft_rows])
        self.draft_rows = list(draft_rows)

    def draft_step(self, hidden, _token, cache, _offset, *, weights):
        assert weights == {}
        token, probability = self.draft_rows[
            self.index % len(self.draft_rows)]
        self.index += 1
        self.proposal_steps += 1
        cache.grow(1)
        row = mx.full((256,), -100.0)
        competitor = (token + 1) % 256
        margin = math.log(probability / (1.0 - probability))
        row = row.at[competitor].add(100.0)
        row = row.at[token].add(100.0 + margin)
        return row, hidden + 1


@pytest.fixture
def _cache_io_noop(monkeypatch):
    monkeypatch.setattr("runtime.engine._cache_io_snapshot", lambda _target: {})
    monkeypatch.setattr(
        "runtime.engine._record_cache_io_delta", lambda *_args, **_kwargs: None)


def test_speculative_controller_full_accept_emits_bonus_in_one_target_sweep(
        _cache_io_noop):
    target = _FakeTarget([11, 12, 13])
    engine = Qwen4MTPSpeculativeEngine(
        target, depth=2, drafter=_FakeDrafter([11, 12]))

    result = engine.generate(
        "prompt", max_tokens=4,
        sampling=SamplingParams(temperature=0.0))

    assert result["tokens"] == [10, 11, 12, 13]
    assert target.verify_calls == [(10, 11, 12)]
    assert result["path_stats"]["qwen4_mtp_target_sweeps"] == 1
    assert result["path_stats"]["qwen4_mtp_target_sweeps_avoided"] == 2
    assert result["path_stats"]["qwen4_mtp_round_outcomes"] == "A2"
    assert result["path_stats"]["qwen4_mtp_idle_head_release_bytes"] == 123
    assert target.idle_head_releases == 1
    assert target.last_kv.offset == 5


def test_speculative_controller_returns_request_profiler_result(
        _cache_io_noop):
    class _Profiler:
        def __init__(self):
            self.calls = []

        def result(self, wall_s):
            self.calls.append(wall_s)
            return {"schema_version": 1, "level": "layers"}

    target = _FakeTarget([11, 12, 13])
    profiler = _Profiler()
    target._request_profiler = profiler
    engine = Qwen4MTPSpeculativeEngine(
        target, depth=2, drafter=_FakeDrafter([11, 12]))

    result = engine.generate(
        "prompt", max_tokens=4,
        sampling=SamplingParams(temperature=0.0))

    assert result["execution_profile"] == {
        "schema_version": 1, "level": "layers"}
    assert len(profiler.calls) == 1
    assert profiler.calls[0] == pytest.approx(result["total_s"])


def test_confidence_adaptive_width_keeps_low_confidence_token_and_verifies_it(
        _cache_io_noop):
    target = _FakeTarget([11, 12, 13])
    engine = Qwen4MTPSpeculativeEngine(
        target,
        depth=4,
        min_draft_probability=0.8,
        drafter=_ConfidenceDrafter([(11, 0.99), (12, 0.60), (99, 0.99)]),
    )

    result = engine.generate(
        "prompt", max_tokens=4,
        sampling=SamplingParams(temperature=0.0))

    assert result["tokens"] == [10, 11, 12, 13]
    assert target.verify_calls == [(10, 11, 12)]
    stats = result["path_stats"]
    assert stats["qwen4_mtp_adaptive_width_enabled"] == 1
    assert stats["qwen4_mtp_adaptive_truncations"] == 1
    assert stats["qwen4_mtp_round_widths"] == "2"
    assert stats["qwen4_mtp_proposed"] == 2
    assert stats["qwen4_mtp_selected_probability_min"] == pytest.approx(0.6)
    assert stats["qwen4_mtp_selected_probability_max"] == pytest.approx(0.99)
    assert float(stats["qwen4_mtp_truncation_probabilities"]) == pytest.approx(
        0.6)
    assert stats["qwen4_mtp_target_prefix_rollbacks"] == 0


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), True, "bad"])
def test_confidence_adaptive_width_rejects_invalid_threshold(value):
    target = _FakeTarget([11])
    with pytest.raises(ValueError, match="minimum draft probability"):
        Qwen4MTPSpeculativeEngine(
            target, depth=2, min_draft_probability=value,
            drafter=_FakeDrafter([11]))


def test_ngram_first_bypasses_native_draft_and_keeps_target_authoritative(
        _cache_io_noop):
    target = _FakeTarget([11, 5, 13])
    target.tokenizer = _FakeTokenizer((5, 10, 11, 5))
    drafter = _FakeDrafter([99])
    engine = Qwen4MTPSpeculativeEngine(
        target,
        depth=3,
        ngram_first=True,
        ngram_max_draft_tokens=2,
        drafter=drafter,
    )

    result = engine.generate(
        "prompt", max_tokens=4,
        sampling=SamplingParams(temperature=0.0))

    assert result["tokens"] == [10, 11, 5, 13]
    assert target.verify_calls == [(10, 11, 5)]
    assert drafter.proposal_steps == 0
    stats = result["path_stats"]
    assert stats["qwen4_mtp_ngram_first_attempts"] == 1
    assert stats["qwen4_mtp_ngram_first_matches"] == 1
    assert stats["qwen4_mtp_ngram_first_proposed"] == 2
    assert stats["qwen4_mtp_ngram_first_accepted"] == 2
    assert stats["qwen4_mtp_ngram_first_rejected"] == 0
    assert stats["qwen4_mtp_ngram_first_native_draft_bypasses"] == 1
    assert stats["qwen4_mtp_proposal_sources"] == "N"
    assert stats["qwen4_mtp_round_outcomes"] == "N:A2"
    assert stats["qwen4_mtp_target_sweeps"] == 1


def test_ngram_first_stochastic_point_mass_q_is_exactly_verified(
        _cache_io_noop):
    target = _FakeTarget([11, 5, 13])
    target.tokenizer = _FakeTokenizer((5, 10, 11, 5))
    drafter = _FakeDrafter([99])
    engine = Qwen4MTPSpeculativeEngine(
        target,
        depth=3,
        ngram_first=True,
        ngram_max_draft_tokens=2,
        drafter=drafter,
    )

    result = engine.generate(
        "prompt", max_tokens=4,
        sampling=SamplingParams(temperature=1.0, seed=17))

    assert result["tokens"] == [10, 11, 5, 13]
    assert drafter.proposal_steps == 0
    stats = result["path_stats"]
    assert stats["qwen4_mtp_stochastic_verified"] == 2
    assert stats["qwen4_mtp_expected_acceptance"] == pytest.approx(1.0)
    assert stats["qwen4_mtp_proposal_sources"] == "N"


def test_ngram_first_rejection_restores_exact_hybrid_target_prefix(
        _cache_io_noop):
    target = _FakeTarget([11, 255, 13])
    target.tokenizer = _FakeTokenizer((5, 10, 11, 5))
    drafter = _FakeDrafter([77])
    engine = Qwen4MTPSpeculativeEngine(
        target,
        depth=3,
        ngram_first=True,
        ngram_max_draft_tokens=2,
        drafter=drafter,
    )

    result = engine.generate(
        "prompt", max_tokens=4,
        sampling=SamplingParams(temperature=1.0, seed=41))

    assert result["tokens"] == [10, 11, 255]
    assert target.verify_calls == [(10, 11, 5)]
    assert target.last_kv.offset == 6
    assert target.last_kv.kda_cache.marker == "kda-2"
    assert target.last_kv.qwen4_cache.restores == [("aux-2", 6)]
    assert drafter.proposal_steps == 0
    stats = result["path_stats"]
    assert stats["qwen4_mtp_ngram_first_accepted"] == 1
    assert stats["qwen4_mtp_ngram_first_rejected"] == 1
    assert stats["qwen4_mtp_round_outcomes"] == "N:A1R"
    assert stats["qwen4_mtp_target_prefix_rollbacks"] == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ngram_first": 1},
        {"ngram_min_ngram": 0},
        {"ngram_min_ngram": 3, "ngram_max_ngram": 2},
        {"ngram_max_draft_tokens": 1},
        {"ngram_max_draft_tokens": 8},
        {"ngram_first": True, "min_draft_probability": 0.1},
        {"compact_kda_rollback": 1},
    ],
)
def test_ngram_first_rejects_invalid_configuration(kwargs):
    target = _FakeTarget([11])
    with pytest.raises((TypeError, ValueError)):
        Qwen4MTPSpeculativeEngine(
            target, depth=2, drafter=_FakeDrafter([11]), **kwargs)


def test_speculative_controller_partial_reject_restores_all_target_state(
        _cache_io_noop):
    target = _FakeTarget([11, 99, 13])
    engine = Qwen4MTPSpeculativeEngine(
        target, depth=2, drafter=_FakeDrafter([11, 12]))

    result = engine.generate(
        "prompt", max_tokens=3,
        sampling=SamplingParams(temperature=0.0))

    assert result["tokens"] == [10, 11, 99]
    assert result["path_stats"]["qwen4_mtp_round_outcomes"] == "A1R"
    assert target.last_kv.offset == 4
    assert target.last_kv.kda_cache.marker == "kda-2"
    assert target.last_kv.qwen4_cache.restores == [("aux-2", 4)]
    assert result["path_stats"]["qwen4_mtp_aux_endpoint_restores"] == 1


def test_compact_kda_rollback_replays_exact_factors_instead_of_endpoints(
        _cache_io_noop):
    target = _FakeTarget([11, 99, 13])
    engine = Qwen4MTPSpeculativeEngine(
        target,
        depth=2,
        compact_kda_rollback=True,
        drafter=_FakeDrafter([11, 12]),
    )

    result = engine.generate(
        "prompt", max_tokens=3,
        sampling=SamplingParams(temperature=0.0))

    stats = result["path_stats"]
    assert result["tokens"] == [10, 11, 99]
    assert target.last_kv.kda_cache.marker == "factor-2"
    assert stats["qwen4_mtp_compact_kda_rollback_enabled"] == 1
    assert stats["qwen4_mtp_kda_factor_restores"] == 1
    assert stats["qwen4_mtp_kda_factor_capture_bytes"] == 33
    assert stats["qwen4_mtp_kda_endpoint_capture_bytes"] == 0
    assert stats["qwen4_mtp_kda_factor_restore_s"] >= 0.0


def test_speculative_controller_all_reject_keeps_final_token_unfed(
        _cache_io_noop):
    target = _FakeTarget([77, 12])
    engine = Qwen4MTPSpeculativeEngine(
        target, depth=2, drafter=_FakeDrafter([11, 12]))

    result = engine.generate(
        "prompt", max_tokens=2,
        sampling=SamplingParams(temperature=0.0))

    assert result["tokens"] == [10, 77]
    assert target.verify_calls == [(10, 11)]
    assert target.last_kv.offset == 3
    assert target.last_kv.kda_cache.marker == "kda-1"
    assert target.last_kv.qwen4_cache.restores == [("aux-1", 3)]
    assert result["path_stats"]["qwen4_mtp_round_outcomes"] == "R"


def test_stochastic_verifier_uses_positive_part_rejection_correction():
    sampling = SamplingParams(temperature=1.0, seed=17)
    sampling.seed_rng()
    accepted, token, probabilities, overlap = _verify_stochastic_token(
        0,
        mx.array([1.0, 0.0, 0.0]),
        mx.array([-100.0, 100.0, -100.0]),
        sampling,
        history=[2],
    )

    assert not accepted
    assert token == 1
    assert probabilities.tolist() == [0.0, 1.0, 0.0]
    assert overlap == pytest.approx(0.0)


def test_stochastic_controller_full_accept_uses_exact_target_verifier(
        _cache_io_noop):
    target = _FakeTarget([11, 12, 13])
    engine = Qwen4MTPSpeculativeEngine(
        target, depth=2, drafter=_FakeDrafter([11, 12]))

    result = engine.generate(
        "prompt", max_tokens=4,
        sampling=SamplingParams(temperature=1.0, seed=29))

    assert result["tokens"] == [10, 11, 12, 13]
    assert target.verify_calls == [(10, 11, 12)]
    stats = result["path_stats"]
    assert stats["qwen4_mtp_used"] == 1
    assert stats["qwen4_mtp_stochastic"] == 1
    assert stats["qwen4_mtp_stochastic_verified"] == 2
    assert stats["qwen4_mtp_expected_acceptance"] == pytest.approx(1.0)


def test_stochastic_controller_reports_observed_path_q_calibration(
        _cache_io_noop):
    target = _FakeTarget([11, 12, 13])
    engine = Qwen4MTPSpeculativeEngine(
        target,
        depth=2,
        q_calibration_scales=(0.5, 1.0, 1.5),
        drafter=_FakeDrafter([11, 12]),
    )

    result = engine.generate(
        "prompt", max_tokens=4,
        sampling=SamplingParams(temperature=1.0, seed=29))

    stats = result["path_stats"]
    calibration = json.loads(stats["qwen4_mtp_q_calibration"])
    assert stats["qwen4_mtp_q_calibration_rows"] == 2
    assert calibration["schema"] == "voom.qwen4-mtp-q-calibration.v1"
    assert calibration["scope"] == "observed-native-draft-path"
    assert calibration["scales"] == [0.5, 1.0, 1.5]
    assert calibration["counts"] == [2, 2, 2]
    assert calibration["by_step_counts"] == [[1, 1, 1], [1, 1, 1]]
    assert calibration["overall"][1] == pytest.approx(
        stats["qwen4_mtp_expected_acceptance"])


@pytest.mark.parametrize(
    "scales",
    [(-1,), (0.2,), (4.1,), (float("nan"),), (True,), (1, 1), range(10)],
)
def test_q_calibration_scales_are_strictly_bounded(scales):
    target = _FakeTarget([11])
    with pytest.raises(ValueError, match="q-calibration"):
        Qwen4MTPSpeculativeEngine(
            target,
            depth=1,
            q_calibration_scales=scales,
            drafter=_FakeDrafter([11]),
        )


def test_stochastic_controller_all_reject_restores_exact_target_state(
        _cache_io_noop):
    target = _FakeTarget([77, 12])
    engine = Qwen4MTPSpeculativeEngine(
        target, depth=2, drafter=_FakeDrafter([11, 12]))

    result = engine.generate(
        "prompt", max_tokens=2,
        sampling=SamplingParams(temperature=1.0, seed=41))

    assert result["tokens"] == [10, 77]
    assert target.last_kv.offset == 3
    assert target.last_kv.kda_cache.marker == "kda-1"
    assert target.last_kv.qwen4_cache.restores == [("aux-1", 3)]
    assert result["path_stats"]["qwen4_mtp_round_outcomes"] == "R"
