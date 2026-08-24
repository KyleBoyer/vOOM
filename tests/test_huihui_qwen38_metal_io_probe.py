from __future__ import annotations

import json
from pathlib import Path

from experiments.huihui_qwen38_metal_io_probe import classify_stop


ROOT = Path(__file__).resolve().parent.parent
RESULT = ROOT / "logs/huihui_qwen38_metal_io_probe.json"
PREFLIGHT = ROOT / "logs/huihui_qwen38_metal_io_probe.preflight.json"


def test_recorded_metal_io_stop_has_honest_bounded_evidence():
    result = json.loads(RESULT.read_text())
    preflight = json.loads(PREFLIGHT.read_text())

    assert result["schema"] == "voom.metal-io-conditional-probe-summary.v1"
    assert result["decision"]["verdict"] == "STOP"
    assert result["bounded_inputs"]["raw_probe_bytes"] == 512_000_000
    assert not result["bounded_inputs"]["model_inference"]
    assert len(set(result["bounded_inputs"]["independent_device_ids"])) == 2
    assert preflight["passed"] and preflight["verdict"] == "PASS"
    assert preflight["swap_growth_bytes"] == 0
    assert preflight["swap_out_growth_bytes"] == 0
    evidence = result["existing_evidence"]
    assert evidence["decode_weight_store_bytes_read"] / \
        evidence["decode_seconds"] / 1e9 == \
        result["measured"]["existing_fast_decode_logical_GBps"]

    decision = classify_stop(result["measured"])
    assert decision["verdict"] == "STOP"
    assert decision["max_stage_fraction"] < 0.10
    assert decision["perfect_stage_elimination_speedup_ceiling"] < 0.10
    assert result["measured"]["concurrent_uncached_aggregate_GBps"] > 3.0
    assert result["measured"]["external_uncached_raw_GBps"] > 1.5


def test_metal_io_decision_reopens_at_threshold():
    measured = json.loads(RESULT.read_text())["measured"]
    measured["external_stage_fraction"] = 0.10
    assert classify_stop(measured)["verdict"] == "REOPEN"
