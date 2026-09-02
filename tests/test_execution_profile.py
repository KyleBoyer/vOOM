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
    stages = [1_000_000, 1, 256, 1_024]
    scale_stages = [2_048, 8_192, 500_000, 3]
    parallel_stages = [1, 2_000, 3_000, 4_000_000, 5_000_000, 6_000_000,
                       7_000_000]
    cache = SimpleNamespace(
        stats=SimpleNamespace(
            hits=1, misses=2, evictions=3, bytes_read=4, disk_s=0.5),
        store=SimpleNamespace(
            stage_snapshot=lambda: tuple(stages),
            k3_scale_sidecar_snapshot=lambda: tuple(scale_stages),
            parallel_tier_snapshot=lambda: tuple(parallel_stages),
        ))
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
    stages[0] += 2_000_000
    stages[1] += 2
    stages[2] += 512
    stages[3] += 2_048
    scale_stages[0] += 4_096
    scale_stages[1] += 16_384
    scale_stages[2] += 1_500_000
    scale_stages[3] += 2
    parallel_stages[0] += 2
    parallel_stages[1] += 8_192
    parallel_stages[2] += 4_096
    parallel_stages[3] += 2_000_000_000
    parallel_stages[4] += 1_250_000_000
    parallel_stages[5] += 1_500_000_000
    parallel_stages[6] += 750_000_000
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
    assert row["parallel_tier_fetches"] == 2
    assert row["parallel_tier_fast_bytes"] == 8192
    assert row["parallel_tier_archive_bytes"] == 4096
    assert row["parallel_tier_wall_s"] == 2.0
    assert row["parallel_tier_fast_service_s"] == 1.25
    assert row["parallel_tier_archive_service_s"] == 1.5
    assert row["parallel_tier_hidden_s"] == 0.75
    assert row["ct_mxfp4_transform_s"] == 0.002
    assert row["ct_mxfp4_transform_calls"] == 2
    assert row["ct_mxfp4_input_bytes"] == 512
    assert row["ct_mxfp4_resident_bytes"] == 2048
    assert row["k3_scale_sidecar_read_bytes"] == 4096
    assert row["k3_scale_sidecar_output_bytes"] == 16384
    assert row["k3_scale_sidecar_decode_s"] == 0.0015
    assert row["k3_scale_sidecar_decode_calls"] == 2
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
