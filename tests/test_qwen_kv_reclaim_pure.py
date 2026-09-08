"""Admission recovery policy and real engine hook; no MLX import or model I/O."""

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from runtime import qwen_kv_reclaim as recovery
from tests.test_qwen35_serial_refusal_pure import hook, target as refusal_target
from tests.test_governor_reserve_pure import load_pressure, make_governor


class FakePaged:
    def __init__(self, calls, *, released=60, spill_error=None):
        self.stats = SimpleNamespace(spills=0, spill_s=0.0)
        self.max_bytes = 256_000_000
        self.resident = 90
        self.calls, self.released, self.spill_error = calls, released, spill_error

    def nbytes(self):
        return self.resident

    def reclaim_closed_pages(self, requested, *, protected_layer):
        self.calls.append(("spill", requested, protected_layer))
        if self.spill_error:
            raise self.spill_error
        released = self.released if requested else 0
        self.resident -= released
        if released:
            self.stats.spills += 1
            self.stats.spill_s += 0.01
        return released


@pytest.fixture
def setup(monkeypatch):
    module = ModuleType("runtime.kv_paged")
    module.PagedKVCache = FakePaged
    monkeypatch.setitem(sys.modules, "runtime.kv_paged", module)
    calls = []
    active = iter([100, 40])
    metal = SimpleNamespace(get_active_memory=lambda: next(active))
    monkeypatch.setattr(recovery.psutil, "virtual_memory", lambda: SimpleNamespace(available=1000))
    gov = SimpleNamespace(_metal_ceiling=lambda a, v: 150,
        reserve=lambda n, **kw: calls.append(("reserve", n, kw)))
    engine = SimpleNamespace(rc=SimpleNamespace(qwen35_serial_kv_reclaim=True),
        cfg=SimpleNamespace(model_type="qwen3_5"), governor=gov,
        _layer_transient=90, _layer_transient_margin=20)
    return engine, FakePaged(calls), metal, calls


def run(engine, kv, metal):
    return recovery.recover_serial_kv_admission(
        engine, kv, metal, layer=3, positions=5, offset=5046)


@pytest.mark.parametrize("enabled", [False, None, 1, "1", "auto"])
def test_only_explicit_boolean_true_enables_and_disabled_does_not_read(enabled):
    engine = SimpleNamespace(rc=SimpleNamespace(qwen35_serial_kv_reclaim=enabled))
    assert run(engine, object(), object()) is False
    assert not hasattr(engine, "_qwen35_serial_kv_reclaim_stats")


@pytest.mark.parametrize("kind", ["other_model", "no_governor", "not_paged"])
def test_inapplicable_never_spills_or_reads(setup, kind):
    engine, kv, _, calls = setup
    if kind == "other_model":
        engine.cfg.model_type = "glm5_next"
    elif kind == "no_governor":
        engine.governor = None
    else:
        kv = object()
    assert run(engine, kv, object()) is False and not calls


def test_content_independent_live_deficit_protected_layer_and_fresh_reserve(setup):
    engine, kv, metal, calls = setup
    assert run(engine, kv, metal)
    assert calls == [("spill", 60, 3),
        ("reserve", 90, dict(margin=20, reason="serial-verify-transient"))]
    stats = engine._qwen35_serial_kv_reclaim_stats
    row = stats["records"][0]
    assert row["requested_bytes"] == row["logical_reclaimed_bytes"] == 60
    assert row["metal_active_released_bytes"] == 60
    assert row["before"]["deficit_bytes"] == 60 and row["after_reclaim"]["deficit_bytes"] == 0
    assert row["outcome"] == "admitted" and row["reservation_retried"] is True
    assert stats["attempts"] == stats["admitted"] == row["spill_pages"] == 1
    assert kv.max_bytes == 256_000_000


def test_aliases_never_turn_logical_reclamation_into_admission(setup):
    engine, kv, _, calls = setup
    metal = SimpleNamespace(get_active_memory=lambda: 100)
    error = MemoryError("aliases retain memory")
    def refuse(*args, **kwargs):
        calls.append("refuse")
        raise error
    engine.governor.reserve = refuse
    assert run(engine, kv, metal) is False
    row = engine._qwen35_serial_kv_reclaim_stats["records"][0]
    assert row["logical_reclaimed_bytes"] == 60 and row["metal_active_released_bytes"] == 0
    assert row["outcome"] == "refused" and calls == [("spill", 60, 3), "refuse"]


def test_no_candidates_keeps_refusal_without_second_settle_loop(setup):
    engine, _, metal, calls = setup
    kv = FakePaged(calls, released=0)
    assert run(engine, kv, metal) is False
    assert calls == [("spill", 60, 3)]
    row = engine._qwen35_serial_kv_reclaim_stats["records"][0]
    assert row["outcome"] == "no_candidates" and not row["reservation_retried"]


def test_zero_new_deficit_still_requires_real_admission(setup):
    engine, kv, metal, calls = setup
    engine.governor._metal_ceiling = lambda a, v: 300
    assert run(engine, kv, metal)
    assert calls[0] == ("spill", 0, 3) and calls[1][0] == "reserve"
    assert kv.resident == 90


@pytest.mark.parametrize("stage", ["spill", "reserve"])
def test_io_and_device_errors_propagate_before_compute(setup, stage):
    engine, kv, metal, calls = setup
    error = OSError("disk failed")
    def fail(*args, **kwargs):
        raise error
    if stage == "spill":
        kv.spill_error = error
    else:
        engine.governor.reserve = fail
    with pytest.raises(OSError) as raised:
        run(engine, kv, metal)
    assert raised.value is error
    row = engine._qwen35_serial_kv_reclaim_stats["records"][0]
    assert row["outcome"] == "error" and row["error_type"] == "OSError"
    if stage == "spill":
        assert calls == [("spill", 60, 3)] and kv.resident == 90


def test_trace_is_bounded_without_losing_aggregate_attempts(setup):
    engine, _, metal, calls = setup
    engine._qwen35_serial_kv_reclaim_stats = {"records": [{}] * 64}
    assert run(engine, FakePaged(calls), metal)
    stats = engine._qwen35_serial_kv_reclaim_stats
    assert len(stats["records"]) == 64 and stats["records_dropped"] == 1
    assert stats["attempts"] == stats["admitted"] == 1


def test_partial_spill_progress_is_retained_when_later_write_fails(setup):
    engine, kv, metal, calls = setup
    error = OSError("second page failed")
    def partial(*args, **kwargs):
        kv.resident -= 10
        kv.stats.spills += 1
        kv.stats.spill_s += 0.01
        raise error
    kv.reclaim_closed_pages = partial
    with pytest.raises(OSError) as raised:
        run(engine, kv, metal)
    assert raised.value is error and not calls
    row = engine._qwen35_serial_kv_reclaim_stats["records"][0]
    assert row["logical_reclaimed_bytes"] == 10 and row["spill_pages"] == 1
    assert row["logical_after_bytes"] == 80 and row["metal_active_released_bytes"] is None
    assert row["outcome"] == "error" and row["reservation_retried"] is False


@pytest.mark.parametrize("invalid", [-1, True, None, 1.5])
def test_unavailable_or_invalid_observation_fails_before_any_spill(setup, invalid):
    engine, kv, _, calls = setup
    with pytest.raises(ValueError):
        run(engine, kv, SimpleNamespace(get_active_memory=lambda: invalid))
    assert not calls


@pytest.mark.parametrize("recovered", [False, True])
def test_actual_engine_hook_rethrows_original_or_continues_only_on_recovery(monkeypatch, recovered):
    error = MemoryError("original")
    engine, calls = refusal_target(error)
    seen = []
    def recover(*args, **kw):
        seen.append(kw)
        return recovered
    monkeypatch.setattr(recovery, "recover_serial_kv_admission", recover)
    invoke, _ = hook()
    if recovered:
        invoke(engine, 3, 5, 5046, object())
    else:
        with pytest.raises(MemoryError) as raised:
            invoke(engine, 3, 5, 5046, object())
        assert raised.value is error
    assert len(calls) == 1 and seen == [dict(layer=3, positions=5, offset=5046)]


def test_real_governor_not_logical_credit_owns_final_decision(setup, monkeypatch):
    engine, kv, _, calls = setup
    module, metal = load_pressure(100)
    governor = make_governor(module, metal, cache_max=1, floor=1)
    governor.metal_limit = 150
    governor.critical = 0
    monkeypatch.setattr(module.time, "sleep", lambda *a: None)
    engine.governor = governor
    # The fake spill changes only logical accounting, not Metal ownership.
    assert run(engine, kv, metal) is False
    assert governor.reservation_calls == governor.reservation_failures == 1
    assert engine._qwen35_serial_kv_reclaim_stats["refused"] == 1


def test_recovery_stays_before_prefetch_and_attention_and_is_default_off():
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "runtime/engine.py").read_text())
    engine = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "StreamingEngine")
    fn = next(n for n in engine.body if isinstance(n, ast.FunctionDef) and n.name == "forward_tokens_serial_positions")
    source = ast.unparse(fn)
    start = source.index("reason='serial-verify-transient'")
    assert start < source.index("recover_serial_kv_admission(", start) < source.index("self.prefetcher.schedule(", start) < source.index("layer_compute_t0 =", start)
    config = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RuntimeConfig")
    field = next(n for n in config.body if isinstance(n, ast.AnnAssign) and n.target.id == "qwen35_serial_kv_reclaim")
    assert ast.literal_eval(field.value) is False


def test_overlay_only_adds_recovery_and_protocol_preserves_structured_phase_records():
    from runtime.profiles import discover_runtime_profiles, resolve_runtime_profiles
    from runtime.server import _cache_phase_telemetry, _vision_protocol_timing
    root = Path(__file__).resolve().parents[1]
    catalog = discover_runtime_profiles((root / "profiles",))
    base = "huihui-qwen38-27b-full-workflow-lifetime-audit"
    _, baseline = resolve_runtime_profiles((base,), catalog)
    _, candidate = resolve_runtime_profiles((base, "qwen35-serial-kv-reclaim"), catalog)
    assert candidate == {**baseline, "VMODEL_QWEN35_SERIAL_KV_RECLAIM": "1"}
    for phase in ("gateway_decision", "gateway_execution"):
        stats = {"qwen35_serial_kv_reclaim_enabled": 1,
                 "qwen35_serial_kv_reclaim": {"attempts": 1, "records": [{"outcome": "admitted"}]}}
        result = dict(path_stats=stats)
        for output in (_cache_phase_telemetry(phase, result), _vision_protocol_timing(result)):
            assert all(output[k] == v for k, v in stats.items())
    assert "qwen35_serial_kv_reclaim" not in _cache_phase_telemetry("gateway_execution", {})
    assert "qwen35_serial_kv_reclaim" not in _vision_protocol_timing({})
