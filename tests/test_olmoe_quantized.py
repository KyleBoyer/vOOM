"""Regression gates for the generic OLMoE lossy path."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from runtime import layer_runner, quant
from runtime.config import effective_expert_top_k
from runtime.engine import (
    RuntimeConfig,
    _apply_runtime_expert_top_k,
    _quantization_cache_identity,
)
from tests.fixtures.olmoe_sidequest_gate import (
    FIBONACCI_20,
    _make_expert_top_k_schedule,
    _parse_layer_selection,
    _task_quality,
)


class _OneStepKV:
    def update(self, _layer, keys, values):
        return keys, values


def test_sidequest_task_quality_rejects_overgeneration():
    sequence = ", ".join(str(value) for value in FIBONACCI_20)
    assert _task_quality(1, '\n{"result": 54}\nextra prose', 128)
    assert not _task_quality(1, '{"result": 55}', 128)
    assert _task_quality(3, sequence, 128)
    assert not _task_quality(3, sequence + ", 6765", 128)


def test_layer_top_k_schedule_is_complete_and_fails_closed():
    cfg = SimpleNamespace(
        num_hidden_layers=4,
        num_experts=64,
        num_experts_per_tok=8,
        expert_top_k_by_layer=(),
    )
    assert effective_expert_top_k(cfg, 2) == 8

    cfg.expert_top_k_by_layer = (7, 8, 7, 8)
    assert [effective_expert_top_k(cfg, layer) for layer in range(4)] == [
        7, 8, 7, 8]

    for invalid in ((7, 8), (7, 8, 0, 8), (7, 8, 9, 8)):
        cfg.expert_top_k_by_layer = invalid
        try:
            effective_expert_top_k(cfg, 0 if len(invalid) != 4 else 2)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid top-k schedule was accepted: {invalid}")


def test_runtime_top_k_schedule_defaults_exact_and_is_olmoe_only():
    cfg = SimpleNamespace(
        model_type="olmoe",
        num_hidden_layers=4,
        num_experts=64,
        num_experts_per_tok=8,
        expert_top_k_by_layer=(),
    )
    default = RuntimeConfig()
    _apply_runtime_expert_top_k(default, cfg)
    assert default.expert_top_k_by_layer == ()
    assert cfg.expert_top_k_by_layer == ()

    selected = RuntimeConfig(expert_top_k_by_layer=[7, 7, 8, 8])
    _apply_runtime_expert_top_k(selected, cfg)
    assert selected.expert_top_k_by_layer == (7, 7, 8, 8)
    assert cfg.expert_top_k_by_layer == (7, 7, 8, 8)

    try:
        _apply_runtime_expert_top_k(
            RuntimeConfig(expert_top_k_by_layer=(7,)), cfg)
    except ValueError as error:
        assert "exactly 4 entries" in str(error)
    else:
        raise AssertionError("an incomplete runtime schedule was accepted")

    dense = SimpleNamespace(model_type="qwen2")
    try:
        _apply_runtime_expert_top_k(
            RuntimeConfig(expert_top_k_by_layer=(7,)), dense)
    except ValueError as error:
        assert "only for OLMoE" in str(error)
    else:
        raise AssertionError("a non-OLMoE schedule was accepted")


def test_runtime_top_k_schedule_round_trips_yaml_and_changes_kv_identity(tmp_path):
    yaml_path = tmp_path / "runtime.yaml"
    yaml_path.write_text(
        "runtime:\n  expert_top_k_by_layer: [7, 7, 8, 8]\n")
    scheduled = RuntimeConfig.from_yaml(yaml_path)
    assert scheduled.expert_top_k_by_layer == (7, 7, 8, 8)

    yaml_path.write_text("runtime:\n  expert_top_k_by_layer: early\n")
    try:
        RuntimeConfig.from_yaml(yaml_path)
    except ValueError as error:
        assert "YAML sequence" in str(error)
    else:
        raise AssertionError("a scalar YAML schedule was accepted")

    store = SimpleNamespace(on_disk_quantized=False)
    released = _quantization_cache_identity(RuntimeConfig(), store)
    selective = _quantization_cache_identity(scheduled, store)
    assert released != selective
    assert selective.endswith("+olmoe-topk-7.7.8.8")


def test_sidequest_layer_selection_supports_named_and_explicit_subsets():
    assert _parse_layer_selection("early", 16) == tuple(range(5))
    assert _parse_layer_selection("middle", 16) == tuple(range(5, 10))
    assert _parse_layer_selection("late", 16) == tuple(range(10, 16))
    assert _parse_layer_selection("even", 16) == tuple(range(0, 16, 2))
    assert _parse_layer_selection("1-3,8,10-11", 16) == (1, 2, 3, 8, 10, 11)
    schedule, selected = _make_expert_top_k_schedule(6, 8, 7, "1-2,5")
    assert selected == (1, 2, 5)
    assert schedule == (8, 7, 7, 8, 8, 7)


def _q4(weight: mx.array) -> quant.QTensor:
    packed = mx.quantize(weight, group_size=32, bits=4, mode="mxfp4")
    mx.eval(packed)
    return quant.QTensor(
        packed[0], packed[1], None, 4, 32, mode="mxfp4")


def test_quantized_olmoe_router_runs_through_real_moe_block():
    """Fast mode quantizes ``mlp.gate``; routing must dispatch QTensor matmul."""
    hidden = 64
    prefix = "model.layers.0"
    cfg = SimpleNamespace(
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=hidden,
        rms_norm_eps=1e-5,
        num_experts=2,
        num_experts_per_tok=2,
        num_hidden_layers=1,
        expert_top_k_by_layer=(),
        norm_topk_prob=False,
        rope_theta=10_000.0,
    )
    identity = mx.eye(hidden, dtype=mx.bfloat16)
    router = mx.stack([
        mx.ones((hidden,), dtype=mx.bfloat16),
        -mx.ones((hidden,), dtype=mx.bfloat16),
    ])
    weights = {
        f"{prefix}.input_layernorm.weight": mx.ones((hidden,)),
        f"{prefix}.post_attention_layernorm.weight": mx.ones((hidden,)),
        f"{prefix}.self_attn.q_proj.weight": identity,
        f"{prefix}.self_attn.k_proj.weight": identity,
        f"{prefix}.self_attn.v_proj.weight": identity,
        f"{prefix}.self_attn.o_proj.weight": identity,
        f"{prefix}.mlp.gate.weight": _q4(router),
    }
    experts = {}
    for expert in range(cfg.num_experts):
        expert_prefix = f"{prefix}.mlp.experts.{expert}"
        experts[expert] = {
            f"{expert_prefix}.gate_proj.weight": _q4(identity * (expert + 1)),
            f"{expert_prefix}.up_proj.weight": _q4(identity * (expert + 1)),
            f"{expert_prefix}.down_proj.weight": _q4(identity / (expert + 1)),
        }

    output = layer_runner.run_moe_block(
        mx.ones((1, 1, hidden), dtype=mx.bfloat16),
        weights,
        prefix,
        cfg,
        _OneStepKV(),
        layer=0,
        offset=0,
        get_experts=lambda _layer, ids, positions=None: {
            expert: experts[expert] for expert in ids
        },
    )
    fused = {
        projection: layer_runner.stack_expert_weights([
            experts[expert][
                f"{prefix}.mlp.experts.{expert}.{projection}.weight"]
            for expert in range(cfg.num_experts)
        ])
        for projection in ("gate_proj", "up_proj", "down_proj")
    }
    fused_output = layer_runner.run_fused_moe_block(
        mx.ones((1, 1, hidden), dtype=mx.bfloat16),
        weights,
        fused,
        prefix,
        cfg,
        _OneStepKV(),
        layer=0,
        offset=0,
    )
    compiled_output = layer_runner.run_fused_moe_block(
        mx.ones((1, 1, hidden), dtype=mx.bfloat16),
        weights,
        fused,
        prefix,
        cfg,
        _OneStepKV(),
        layer=0,
        offset=0,
        fused_swiglu=True,
    )
    mlx_output = layer_runner.run_fused_moe_block(
        mx.ones((1, 1, hidden), dtype=mx.bfloat16),
        weights,
        fused,
        prefix,
        cfg,
        _OneStepKV(),
        layer=0,
        offset=0,
        mlx_router_semantics=True,
    )
    mx.eval(output, fused_output, compiled_output, mlx_output)

    assert isinstance(weights[f"{prefix}.mlp.gate.weight"], quant.QTensor)
    assert output.shape == (1, 1, hidden)
    assert bool(mx.all(mx.isfinite(output)))
    assert mx.array_equal(output, fused_output)
    assert mx.array_equal(output, compiled_output)
    assert bool(mx.all(mx.isfinite(mlx_output)))

    cfg.expert_top_k_by_layer = (1,)
    routed = []

    def get_selective_experts(_layer, ids, positions=None):
        routed.append(tuple(ids))
        return {expert: experts[expert] for expert in ids}

    pageable_selective = layer_runner.run_moe_block(
        mx.ones((1, 1, hidden), dtype=mx.bfloat16),
        weights,
        prefix,
        cfg,
        _OneStepKV(),
        layer=0,
        offset=0,
        get_experts=get_selective_experts,
    )
    resident_selective = layer_runner.run_fused_moe_block(
        mx.ones((1, 1, hidden), dtype=mx.bfloat16),
        weights,
        fused,
        prefix,
        cfg,
        _OneStepKV(),
        layer=0,
        offset=0,
    )
    mx.eval(pageable_selective, resident_selective)
    assert routed and all(len(request) == 1 for request in routed)
    assert mx.array_equal(pageable_selective, resident_selective)


def test_pageable_olmoe_bulk_routes_preserve_position_and_topk_order():
    """The pageable route transfer must retain the old nested-loop ordering."""
    hidden = 64
    prefix = "model.layers.0"
    cfg = SimpleNamespace(
        rms_norm_eps=1e-5,
        num_experts=3,
        num_experts_per_tok=2,
        num_hidden_layers=1,
        expert_top_k_by_layer=(),
        norm_topk_prob=False,
    )
    router = mx.zeros((3, hidden), dtype=mx.float32)
    router[:, :3] = mx.array([
        [3.0, 2.0, 1.0],
        [2.0, 3.0, 0.0],
        [1.0, 0.0, 3.0],
    ])
    weights = {
        f"{prefix}.post_attention_layernorm.weight": mx.ones((hidden,)),
        f"{prefix}.mlp.gate.weight": router,
    }
    identity = mx.eye(hidden, dtype=mx.float32)
    experts = {}
    for expert in range(3):
        expert_prefix = f"{prefix}.mlp.experts.{expert}"
        experts[expert] = {
            f"{expert_prefix}.gate_proj.weight": identity,
            f"{expert_prefix}.up_proj.weight": identity,
            f"{expert_prefix}.down_proj.weight": identity,
        }

    routed = []

    def get_experts(_layer, ids, positions=None):
        routed.append((tuple(ids), positions))
        return {expert: experts[expert] for expert in ids}

    states = mx.zeros((1, 3, hidden), dtype=mx.float32)
    states[0, 0, 0] = 1.0
    states[0, 1, 1] = 1.0
    states[0, 2, 2] = 1.0
    output = layer_runner.run_moe_mlp(
        states, weights, prefix, cfg, layer=0, get_experts=get_experts)
    mx.eval(output)

    assert routed == [(
        (0, 1, 2),
        {0: [0, 1, 2], 1: [0, 1], 2: [2]},
    )]
    assert output.shape == states.shape


def test_reranked_mxfp4_head_recovers_exact_greedy_candidate():
    mx.random.seed(9917)
    exact = (mx.random.normal((64, 64)) * 0.08).astype(mx.bfloat16)
    hidden = (mx.random.normal((1, 1, 64)) * 0.2).astype(mx.bfloat16)
    head = quant.make_reranked_q_head(
        exact, candidates=8, group_size=32, bits=4, mode="mxfp4")

    exact_logits = quant.matmul(hidden, exact)
    approx_logits = quant.matmul(hidden, head.approx)
    candidate_ids = mx.argpartition(
        -approx_logits, kth=head.candidates - 1, axis=-1
    )[..., :head.candidates]
    exact_token = int(mx.argmax(exact_logits))
    # This seed is a deterministic candidate-recall fixture. The second assert
    # gates the exact gather_mm rerank and sparse-logit assembly independently.
    assert exact_token in candidate_ids.tolist()[0][0]

    reranked_logits = quant.matmul(hidden, head)
    mx.eval(reranked_logits)
    assert int(mx.argmax(reranked_logits)) == exact_token
    assert mx.array_equal(
        mx.take_along_axis(reranked_logits, candidate_ids, axis=-1),
        mx.take_along_axis(exact_logits, candidate_ids, axis=-1),
    )
    assert int(mx.sum(mx.isfinite(reranked_logits))) == head.candidates


def test_reranked_head_rejects_an_impossible_candidate_count():
    exact = mx.zeros((32, 64), dtype=mx.bfloat16)
    try:
        quant.make_reranked_q_head(exact, candidates=33)
    except ValueError as error:
        assert "must be in [1, 32]" in str(error)
    else:
        raise AssertionError("candidate count larger than vocabulary was accepted")
