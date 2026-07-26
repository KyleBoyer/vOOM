"""Request-local phase/layer execution attribution."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from runtime.engine import RuntimeConfig, StreamingEngine
from runtime.telemetry import RequestProfiler


MODEL = str(
    Path(__file__).resolve().parent.parent / "models" / "SmolLM2-135M")
GLM_FIXTURE = str(
    Path(__file__).resolve().parent.parent / "models" / "glm-fixture-tiny")


def test_profiler_aggregates_nested_expert_and_substep_metrics():
    profiler = RequestProfiler("ops")
    cache = SimpleNamespace(stats=SimpleNamespace(
        hits=1, misses=2, evictions=3, bytes_read=4, disk_s=0.5))
    before = profiler.cache_snapshot(cache)
    profiler.set_phase("prefill")
    profiler.begin_sweep(8, path="streamed")
    profiler.record_expert_fetch(
        2, pages=8, misses=7, wall_s=0.25)
    profiler.record_substep("attention", 2, 0.5, positions=8)
    cache.stats.hits += 2
    cache.stats.misses += 3
    cache.stats.evictions += 1
    cache.stats.bytes_read += 4096
    cache.stats.disk_s += 0.125
    profiler.record_layer(
        2, positions=8, weight_wait_s=1.0, compute_s=2.0,
        cache_before=before, cache_after=profiler.cache_snapshot(cache),
        layer_type="linear_attention")

    result = profiler.result(4.0)
    assert result["schema_version"] == 1
    assert result["level"] == "ops"
    assert result["phases"]["prefill"]["positions"] == 8
    row = result["layers"][0]
    assert row["total_s"] == 3.0
    assert row["expert_fetch_s"] == 0.25
    assert row["expert_pages"] == 8
    assert row["cache_hits"] == 2
    assert row["cache_misses"] == 3
    assert row["store_bytes_read"] == 4096
    assert row["substeps"]["attention"]["wall_s"] == 0.5
    assert "do not add" in result["semantics"]["expert_fetch_s"]


def test_resident_stack_does_not_double_count_its_sweep_path():
    profiler = RequestProfiler("layers")
    profiler.set_phase("decode")
    profiler.begin_sweep(1, path="resident_fast_stack")
    profiler.record_stack(
        positions=1, path="resident_fast_stack", wall_s=0.125)

    result = profiler.result(0.25)
    phase = result["phases"]["decode"]
    assert phase["sweeps"] == 1
    assert phase["paths"] == {"resident_fast_stack": 1}
    assert phase["stack_calls"] == 1
    assert phase["stack_wall_s"] == 0.125


def test_execution_profile_keeps_greedy_tokens_identical():
    plain = StreamingEngine(MODEL, RuntimeConfig())
    profiled = StreamingEngine(
        MODEL, RuntimeConfig(execution_profile="layers"))
    try:
        baseline = plain.generate(
            "The capital of France is", max_tokens=4)
        measured = profiled.generate(
            "The capital of France is", max_tokens=4)
    finally:
        plain.close()
        profiled.close()
        mx.clear_cache()

    assert measured["tokens"] == baseline["tokens"]
    profile = measured["execution_profile"]
    assert profile["level"] == "layers"
    assert profile["phases"]["prefill"]["sweeps"] >= 1
    assert profile["phases"]["decode"]["sweeps"] >= 1
    assert profile["layers"]
    assert all(row["total_s"] >= 0 for row in profile["layers"])
    assert sum(row["store_bytes_read"] for row in profile["layers"]) >= 0
    assert "execution_profile" not in baseline


def test_invalid_execution_profile_fails_before_model_load():
    with pytest.raises(ValueError, match="execution_profile"):
        StreamingEngine("unused", RuntimeConfig(execution_profile="maybe"))


def test_ops_profile_attributes_glm_attention_router_mlp_and_expert_io():
    engine = StreamingEngine(
        GLM_FIXTURE,
        RuntimeConfig(
            execution_profile="ops", max_weight_cache_mb=200,
            min_weight_cache_mb=100, context_bound=32))
    try:
        result = engine.generate("instrument this request", max_tokens=2)
    finally:
        engine.close()
        mx.clear_cache()

    profile = result["execution_profile"]
    assert profile["level"] == "ops"
    assert "adds synchronization" in profile["semantics"]["substeps"]
    assert any(
        "attention" in row.get("substeps", {})
        and "mlp" in row.get("substeps", {})
        for row in profile["layers"])
    assert any(
        "router" in row.get("substeps", {})
        and row["expert_fetch_calls"] > 0
        and row["expert_pages"] > 0
        for row in profile["layers"])
