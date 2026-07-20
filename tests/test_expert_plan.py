import json
import sys

import pytest

from runtime.expert_plan import (
    build_physical_orders,
    compile_plan,
    evaluate_adjacent_sweep_persistence,
    evaluate_transition_predictor,
    load_trace,
    simulate_layout,
    split_sweeps,
    write_trace,
)


PAGE = 100


def _training_sweeps():
    return [
        {0: (0, 1), 1: (4, 5)},
        {0: (0, 1), 1: (4, 5)},
        {0: (0, 2), 1: (4, 6)},
    ]


def test_trace_round_trip_and_reset_split(tmp_path):
    trace = [(3, [2, 1, 2]), (4, [5]), (3, [1]), (4, [6])]
    assert split_sweeps(trace) == [
        {3: (1, 2), 4: (5,)},
        {3: (1,), 4: (6,)},
    ]
    path = write_trace(
        tmp_path / "routes.json", trace, model="fixture", num_experts=8,
        expert_page_bytes=PAGE,
    )
    metadata, sweeps = load_trace(path)
    assert metadata["model"] == "fixture"
    assert sweeps == split_sweeps(trace)


def test_trace_loader_rejects_non_monotonic_layer_order(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({
        "schema": "vmodel.expert-trace.v1",
        "sweeps": [{"index": 0, "routes": [
            {"layer": 4, "experts": [0]},
            {"layer": 3, "experts": [1]},
        ]}],
    }))
    with pytest.raises(ValueError, match="layers must increase"):
        load_trace(path)


def test_coactivation_order_reduces_requests_without_extra_bytes():
    sweeps = [
        {0: (0, 3)},
        {0: (0, 3)},
        {0: (1, 2)},
        {0: (1, 2)},
    ]
    orders = build_physical_orders(sweeps, num_experts=4)
    identity = simulate_layout(
        sweeps, orders["identity"], expert_page_bytes=PAGE,
        bandwidth_mbps=100, coalesce_gap_pages=0,
    )
    planned = simulate_layout(
        sweeps, orders["coactivation"], expert_page_bytes=PAGE,
        bandwidth_mbps=100, coalesce_gap_pages=0,
    )
    assert planned.requests < identity.requests
    assert planned.physical_bytes == planned.demanded_bytes
    assert planned.read_amplification == 1.0


def test_fixed_bundles_expose_read_amplification():
    sweeps = [{0: (0,)}]
    result = simulate_layout(
        sweeps, {0: (0, 1, 2, 3)}, expert_page_bytes=PAGE,
        bandwidth_mbps=100, bundle_pages=4,
    )
    assert result.requests == 1
    assert result.demanded_bytes == PAGE
    assert result.physical_bytes == 4 * PAGE
    assert result.read_amplification == 4.0


def test_dynamic_gap_coalescing_charges_unused_gap_page():
    result = simulate_layout(
        [{0: (0, 2)}], {0: (0, 1, 2)}, expert_page_bytes=PAGE,
        bandwidth_mbps=100, coalesce_gap_pages=1,
    )
    assert result.requests == 1
    assert result.demanded_bytes == 2 * PAGE
    assert result.physical_bytes == 3 * PAGE


def test_layout_cache_replays_hits_and_only_charges_misses():
    result = simulate_layout(
        [{0: (0, 1)}, {0: (0, 1)}], {0: (0, 1)},
        expert_page_bytes=PAGE, bandwidth_mbps=100,
        coalesce_gap_pages=0, cache_pages=2,
    )
    assert result.accesses == 4
    assert result.cache_hits == 2
    assert result.cache_misses == 2
    assert result.demanded_bytes == 2 * PAGE
    assert result.physical_bytes == 2 * PAGE
    assert result.requests == 1


def test_transition_predictor_is_trained_only_on_prior_sweeps():
    held_out = [{0: (0, 1), 1: (4, 5)}]
    result = evaluate_transition_predictor(
        _training_sweeps(), held_out, top_m=2,
        opportunistic_pages=2, expert_page_bytes=PAGE,
    )
    assert result.transitions == 1
    assert result.useful == 2
    assert result.wasted == 0
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.demand_stall_bytes_saved == 2 * PAGE


def test_opportunistic_budget_can_disable_prefetch():
    result = evaluate_transition_predictor(
        _training_sweeps(), [{0: (0, 1), 1: (4, 5)}], top_m=2,
        opportunistic_pages=0, expert_page_bytes=PAGE,
    )
    assert result.predictions == 0
    assert result.extra_device_bytes == 0
    assert result.demand_stall_bytes_saved == 0


def test_adjacent_sweep_persistence_scores_same_layer_overlap():
    result = evaluate_adjacent_sweep_persistence([
        {0: (0, 1), 1: (4, 5)},
        {0: (0, 2), 1: (4, 6)},
    ])
    assert result.comparisons == 2
    assert result.predicted == 4
    assert result.useful == 2
    assert result.precision == 0.5
    assert result.recall == 0.5


def test_compiled_plan_is_deterministic_and_logical_only():
    sweeps = _training_sweeps()
    first = compile_plan(sweeps, num_experts=8)
    second = compile_plan(sweeps, num_experts=8)
    assert first == second
    assert first["application_status"].startswith("logical-only")
    assert set(first["orders"]) == {"identity", "heat", "coactivation"}


def test_engine_export_uses_stable_trace_schema(tmp_path):
    from types import SimpleNamespace

    from runtime.engine import StreamingEngine

    fake = SimpleNamespace(
        expert_trace=[(3, (1, 2)), (4, (5,)), (3, (2, 3))],
        _model_dir=tmp_path / "model",
        _expert_storage_page_bytes=75500000,
        cfg=SimpleNamespace(num_experts=256),
    )
    path = StreamingEngine.export_expert_trace(fake, tmp_path / "trace.json")
    document, sweeps = load_trace(path)
    assert document["num_experts"] == 256
    assert document["expert_page_bytes"] == 75500000
    assert len(sweeps) == 2


def test_cli_writes_held_out_machine_readable_report(tmp_path, monkeypatch):
    from runtime.expert_plan import main

    trace = []
    # First sweep models prefill and should be skipped by the CLI default.
    for sweep in [
        {0: (0, 1, 2, 3), 1: (4, 5, 6, 7)},
        *_training_sweeps(),
        {0: (0, 1), 1: (4, 5)},
        {0: (0, 1), 1: (4, 5)},
    ]:
        trace.extend(sorted(sweep.items()))
    trace_path = write_trace(
        tmp_path / "routes.json", trace, model="fixture", num_experts=8,
        expert_page_bytes=PAGE,
    )
    report_path = tmp_path / "report.json"
    plan_path = tmp_path / "plan.json"
    monkeypatch.setattr(sys, "argv", [
        "expert_plan.py", "--trace", str(trace_path),
        "--bandwidth-mbps", "100", "--request-overhead-ms", "1",
        "--cache-pages", "2",
        "--opportunistic-pages", "0",
        "--report-out", str(report_path), "--plan-out", str(plan_path),
    ])
    main()
    report = json.loads(report_path.read_text())
    plan = json.loads(plan_path.read_text())
    assert report["layout_evaluation"] == "held-out"
    assert report["skipped_sweeps"] == 1
    assert report["cache_pages"] == 2
    assert report["opportunistic_predictor"]["extra_device_bytes"] == 0
    assert plan["application_status"].startswith("logical-only")


def test_predictive_expert_prefetch_is_separate_and_fail_safe_by_default(tmp_path):
    from runtime.engine import RuntimeConfig

    default = RuntimeConfig()
    assert default.expert_predictive_prefetch is False
    assert default.expert_prefetch_idle_only is True

    config = tmp_path / "runtime.yaml"
    config.write_text(
        "runtime:\n"
        "  expert_predictive_prefetch: true\n"
        "  expert_prefetch_idle_only: false\n"
    )
    aggressive = RuntimeConfig.from_yaml(config)
    assert aggressive.expert_predictive_prefetch is True
    assert aggressive.expert_prefetch_idle_only is False


def test_route_recording_only_schedules_explicit_predictive_prefetch():
    from types import SimpleNamespace

    from runtime.engine import StreamingEngine

    class Predictor:
        def __init__(self):
            self.observed = []

        def observe(self, layer, experts):
            self.observed.append((layer, tuple(experts)))

        def predict(self, _layer, _experts, top_m):
            assert top_m == 2
            return [7]

    class Prefetcher:
        def __init__(self):
            self.calls = []

        def schedule(self, key, names, **kwargs):
            self.calls.append((key, tuple(names), kwargs))

    predictor = Predictor()
    prefetcher = Prefetcher()
    fake = SimpleNamespace(
        _provisional=None,
        expert_usage={},
        expert_trace=[],
        predictor=predictor,
        prefetcher=prefetcher,
        rc=SimpleNamespace(
            expert_predictive_prefetch=False,
            expert_prefetch_idle_only=True,
        ),
        cfg=SimpleNamespace(
            num_experts_per_tok=2,
            moe_expert_prefix="mlp.experts",
        ),
        store=SimpleNamespace(
            names_with_prefix=lambda prefix: [prefix + "weight"]),
    )
    StreamingEngine._record_expert_route(fake, 0, [1, 2])
    assert predictor.observed == [(0, (1, 2))]
    assert prefetcher.calls == []

    fake.rc.expert_predictive_prefetch = True
    StreamingEngine._record_expert_route(fake, 1, [3, 4])
    assert prefetcher.calls == [(
        "layer.2.expert.7",
        ("model.layers.2.mlp.experts.7.weight",),
        {"only_if_idle": True},
    )]


def test_vpack2_expert_storage_estimate_uses_compressed_extents():
    from types import SimpleNamespace

    from runtime.model_loader import WeightStore

    names = [
        "model.layers.3.mlp.experts.0.gate_proj.weight",
        "model.layers.3.mlp.experts.0.up_proj.weight",
        "model.layers.3.mlp.experts.1.gate_proj.weight",
        "model.layers.3.self_attn.q_proj.weight",
    ]
    fake = SimpleNamespace(
        _names=names,
        _real_name={},
        vpack2=SimpleNamespace(index={
            names[0]: {"len": 100},
            names[1]: {"len": 140},
            names[2]: {"len": 80},
            names[3]: {"len": 999},
        }),
    )
    estimated = WeightStore.estimate_expert_storage_page_bytes(
        fake, "mlp.experts", fallback=1_000)
    assert estimated == 160  # expert pages: (100+140) and 80
