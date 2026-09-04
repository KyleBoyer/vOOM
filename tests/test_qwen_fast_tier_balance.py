import hashlib
import json
from pathlib import Path

from runtime.qwen_fast_tier_balance import plan


def _trace(
    path: Path, *, hot: int, cold: int, measured: bool,
    measured_fast_bytes: int = 240,
) -> None:
    document = {
        "schema": "vmodel.expert-trace.v1",
        "model": "Qwen-test",
        "num_experts": 4,
        "expert_page_bytes": 60,
        "request_shape": {"prompt_tokens": 8},
        "target_sweeps": 1,
        "sweeps": [{
            "index": 0,
            "routes": [
                {"layer": 0, "experts": [hot, cold]},
                {"layer": 1, "experts": [hot, cold]},
            ],
        }],
        "baseline_io": {
            "parallel_tier_fast_bytes": measured_fast_bytes if measured else 0,
            "parallel_tier_fast_service_ns": 120 if measured else 0,
            "parallel_tier_archive_bytes": 360 if measured else 0,
            "parallel_tier_archive_service_ns": 360 if measured else 0,
        },
    }
    path.write_text(json.dumps(document))


def test_plan_prefers_complete_hot_capacity_and_reports_holdouts(tmp_path):
    fast = tmp_path / "fast"
    fast.mkdir()
    manifest = {}
    for layer in range(2):
        for expert in (2, 3):
            for projection in ("gate", "up", "down"):
                name = (
                    f"model.layers.{layer}.mlp.experts.{expert}."
                    f"{projection}_proj.weight")
                manifest[name] = {"nbytes": 20}
    manifest_raw = json.dumps(manifest).encode()
    (fast / "fast_tier_manifest.json").write_bytes(manifest_raw)
    binding = {
        "target_model": "Qwen-test",
        "layers": 2,
        "experts": 4,
        "selected_bytes": 240,
        "trace_hot_experts_per_layer": 1,
        "fast_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
    }
    (fast / "qwen4_fused_expert_fast_tier.json").write_text(
        json.dumps(binding))
    traces = []
    for index, pair in enumerate(((0, 2), (0, 3), (1, 3))):
        path = tmp_path / f"trace-{index}.json"
        _trace(path, hot=pair[0], cold=pair[1], measured=index > 0)
        traces.append(path)

    result = plan(fast, traces)

    assert result["schema"] == "voom.qwen4-fast-tier-balance-plan.v1"
    assert result["selected_experts"] == 4
    assert result["storage_expert_bytes"] == 60
    assert result["trace_expert_page_bytes"] == 60
    assert result["recommended"]["trace_hot_experts_per_layer"] == 0
    assert result["recommended"]["selection_mode"] == "all-ranked-hot"
    assert len(result["leave_one_out"]) == 3
    assert result["measured_service_bytes_per_second"] == {
        "fast": 2_000_000_000.0,
        "archive": 1_000_000_000.0,
    }


def test_plan_rejects_single_trace(tmp_path):
    fast = tmp_path / "fast"
    fast.mkdir()
    try:
        plan(fast, [tmp_path / "one.json"])
    except ValueError as error:
        assert "at least two" in str(error)
    else:
        raise AssertionError("single-trace plan unexpectedly succeeded")


def test_plan_can_score_larger_capacity_from_active_tier(tmp_path):
    fast = tmp_path / "fast"
    fast.mkdir()
    manifest = {}
    for layer in range(2):
        for expert in (2, 3):
            for projection in ("gate", "up", "down"):
                name = (
                    f"model.layers.{layer}.mlp.experts.{expert}."
                    f"{projection}_proj.weight")
                manifest[name] = {"nbytes": 20}
    manifest_raw = json.dumps(manifest).encode()
    (fast / "fast_tier_manifest.json").write_bytes(manifest_raw)
    (fast / "qwen4_fused_expert_fast_tier.json").write_text(json.dumps({
        "target_model": "Qwen-test",
        "layers": 2,
        "experts": 4,
        "selected_bytes": 240,
        "trace_hot_experts_per_layer": 1,
        "fast_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
    }))
    traces = []
    for index, pair in enumerate(((0, 2), (0, 3), (1, 3))):
        path = tmp_path / f"trace-{index}.json"
        _trace(path, hot=pair[0], cold=pair[1], measured=index > 0)
        traces.append(path)

    result = plan(fast, traces, evaluated_selected_experts=6)

    assert result["selected_experts"] == 4
    assert result["evaluated_selected_experts"] == 6
    assert result["evaluated_selected_bytes"] == 360
    assert result["recommended"]["selection_mode"] == "all-ranked-hot"


def test_plan_refuses_stale_tier_timing_without_losing_route_scores(tmp_path):
    fast = tmp_path / "fast"
    fast.mkdir()
    manifest = {}
    for layer in range(2):
        for expert in (2, 3):
            for projection in ("gate", "up", "down"):
                name = (
                    f"model.layers.{layer}.mlp.experts.{expert}."
                    f"{projection}_proj.weight")
                manifest[name] = {"nbytes": 20}
    manifest_raw = json.dumps(manifest).encode()
    (fast / "fast_tier_manifest.json").write_bytes(manifest_raw)
    (fast / "qwen4_fused_expert_fast_tier.json").write_text(json.dumps({
        "target_model": "Qwen-test",
        "layers": 2,
        "experts": 4,
        "selected_bytes": 240,
        "trace_hot_experts_per_layer": 1,
        "fast_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
    }))
    traces = []
    for index in range(3):
        path = tmp_path / f"trace-{index}.json"
        # The active tier covers expert 2 on both layers (120 routed bytes),
        # so 60 measured fast bytes cannot belong to this placement.
        _trace(
            path, hot=0, cold=2, measured=True,
            measured_fast_bytes=60)
        traces.append(path)

    result = plan(fast, traces, evaluated_selected_experts=6)

    assert result["incompatible_calibration_traces"] == [
        path.name for path in traces]
    assert result["recommended"]["mean_calibrated_critical_seconds"] is None
    assert result["recommended"]["mean_route_hit_fraction"] > 0
