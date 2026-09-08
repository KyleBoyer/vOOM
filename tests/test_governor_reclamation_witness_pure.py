"""No MLX/native calls: temporal brackets, observation-only order and fail-closed coverage."""

import ast
from collections import deque
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from runtime import process_memory_witness as witness
from runtime.profiles import apply_runtime_profiles
from tests.fixtures.captured_transition_tracking_gate import native_pressure_summary


def valid_row():
    return dict(system_available_bytes=5_200_000_000,
        system_swap_used_bytes=0, system_swap_out_bytes=0,
        process=dict(available=True, physical_footprint_bytes=100,
            internal_compressed_ledger_bytes=20,
            monotonic_start_ns=50, monotonic_end_ns=60),
        reclamation_alignment=dict(schema="voom.governor-reclamation-alignment.v1",
            available=True, atomic=False, input_sample_start_ns=10,
            input_sample_end_ns=20, observation_start_ns=30, observation_end_ns=40,
            system_available_bytes=6_200_000_000, metal_active_bytes=50, metal_cache_bytes=0))


def test_profile_is_only_an_explicit_observation_overlay():
    env = {}
    apply_runtime_profiles(["governor-reclamation-witness"], environ=env)
    assert env == {witness.FLAG: "1", witness.RECLAMATION_FLAG: "1"}


@pytest.mark.parametrize("value", [None, "", "0", "true", "1"])
def test_alignment_strict_flag_does_not_sample(monkeypatch, value):
    monkeypatch.delenv(witness.RECLAMATION_FLAG, raising=False)
    if value is not None:
        monkeypatch.setenv(witness.RECLAMATION_FLAG, value)
    monkeypatch.setattr(witness, "sample_reclamation_alignment", lambda *a: pytest.fail("sampled"))
    assert witness.GovernorProcessMemoryObserver().reclamation_enabled is (value == "1")


@pytest.mark.parametrize("bounds", [None, (1,), (1, 2, 3), (True, 2), (-1, 2), (3, 2), (1, 1001)])
def test_invalid_input_brackets_do_not_read_or_fabricate_zero(monkeypatch, bounds):
    monkeypatch.setattr(witness.time, "monotonic_ns", lambda: 1000)
    monkeypatch.setattr(witness.psutil, "virtual_memory", lambda: pytest.fail("sampled"))
    row = witness.sample_reclamation_alignment(object(), bounds)
    assert not row["available"] and row["reason"] == "invalid-input-bracket"
    assert all(row[k] is None for k in ("system_available_bytes", "metal_active_bytes", "metal_cache_bytes"))


@pytest.mark.parametrize("mode", ["raises", "negative", "bool"])
def test_failed_reader_is_explicit_unavailable_and_redacted(monkeypatch, mode):
    monkeypatch.setattr(witness.time, "monotonic_ns", lambda: 1000)
    monkeypatch.setattr(witness.psutil, "virtual_memory", lambda: SimpleNamespace(available=100))
    def read():
        if mode == "raises":
            raise RuntimeError("PRIVATE")
        return -1 if mode == "negative" else True
    row = witness.sample_reclamation_alignment(SimpleNamespace(
        get_active_memory=read, get_cache_memory=lambda: 0), (1, 2))
    assert not row["available"]
    assert all(row[k] is None for k in ("system_available_bytes", "metal_active_bytes", "metal_cache_bytes"))
    assert "PRIVATE" not in json.dumps(row)


@pytest.mark.parametrize("mutation", ["missing", "error", "clock", "native_clock", "bool", "cap"])
def test_missing_error_or_unordered_coverage_is_not_a_pass(mutation):
    row = valid_row()
    if mutation == "missing":
        row.pop("reclamation_alignment")
    elif mutation == "error":
        row["reclamation_alignment"]["available"] = False
    elif mutation == "clock":
        row["reclamation_alignment"]["observation_start_ns"] = 15
    elif mutation == "native_clock":
        row["process"]["monotonic_start_ns"] = 35
    elif mutation == "bool":
        row["reclamation_alignment"]["metal_active_bytes"] = True
    else:
        row = dict(schema="voom.process-memory-trace-end.v1", coverage_complete=False)
    assert not witness.summarize_reclamation_alignment([valid_row(), row])["passed"]
    assert not witness.summarize_reclamation_alignment([])["passed"]


def test_alignment_cannot_replace_the_original_low_available_pressure_sample():
    row = valid_row()
    result = native_pressure_summary('[process-memory] ' + json.dumps(row))
    assert result["minimum_available_bytes"] == 5_200_000_000
    assert not result["passed"]
    assert result["reclamation_alignment"] == dict(available=True, passed=True, samples=1,
        maximum_input_sample_span_ns=10, maximum_after_response_delay_ns=10,
        maximum_observation_span_ns=30)


@pytest.mark.parametrize("enabled", [False, True])
def test_alignment_uses_existing_rate_cap_and_has_no_reads_when_disabled(monkeypatch, capsys, enabled):
    monkeypatch.setenv(witness.RECLAMATION_FLAG, "1" if enabled else "0")
    monkeypatch.delenv("VMODEL_HOST_ACTIVITY_WITNESS", raising=False)
    monkeypatch.setattr(witness, "MAX_SAMPLES", 2)
    reads = []
    monkeypatch.setattr(witness, "sample_reclamation_alignment",
        lambda *a: reads.append("aligned") or valid_row()["reclamation_alignment"])
    monkeypatch.setattr(witness, "sample_self_memory",
        lambda: reads.append("native") or valid_row()["process"])
    observer = witness.GovernorProcessMemoryObserver()
    for now in (0, 1, 2, 3, 4, 10):
        observer.record(governor_monotonic_s=now, system_available_bytes=5_200_000_000,
            system_swap_used_bytes=0, system_swap_out_bytes=0, metal_active_bytes=100,
            cache_budget_bytes_after_response=64, swap_pressure_response=False)
    assert reads == (["aligned", "native"] * 2 if enabled else ["native"] * 2)
    rows = [json.loads(line.removeprefix('[process-memory] '))
            for line in capsys.readouterr().out.splitlines()]
    assert len(rows) == 3 and rows[-1]["coverage_complete"] is False
    assert ("reclamation_alignment" in rows[0]) is enabled
    assert not witness.summarize_reclamation_alignment(rows)["passed"]


def test_alignment_failure_preserves_native_and_original_inputs(monkeypatch, capsys):
    monkeypatch.setenv(witness.RECLAMATION_FLAG, "1")
    monkeypatch.delenv("VMODEL_HOST_ACTIVITY_WITNESS", raising=False)
    monkeypatch.setattr(witness, "sample_self_memory", lambda: valid_row()["process"])
    witness.GovernorProcessMemoryObserver().record(governor_monotonic_s=1,
        system_available_bytes=5_200_000_000, system_swap_used_bytes=0,
        system_swap_out_bytes=0, metal_active_bytes=100,
        cache_budget_bytes_after_response=64, swap_pressure_response=False)
    row = json.loads(capsys.readouterr().out.removeprefix('[process-memory] '))
    assert row["process"] == valid_row()["process"]
    assert row["system_available_bytes"] == 5_200_000_000
    assert not row["reclamation_alignment"]["available"]


def test_real_poll_preserves_inputs_and_samples_alignment_only_after_safety(monkeypatch, capsys):
    monkeypatch.setenv(witness.RECLAMATION_FLAG, "1")
    monkeypatch.delenv("VMODEL_HOST_ACTIVITY_WITNESS", raising=False)
    actions = []
    clock = iter(range(10, 100, 10))
    monkeypatch.setattr(witness.time, "monotonic_ns", lambda: next(clock))
    available = iter((5_200_000_000, 6_200_000_000))
    def virtual_memory():
        actions.append("vm")
        return SimpleNamespace(available=next(available))
    monkeypatch.setattr(witness.psutil, "virtual_memory", virtual_memory)
    active = iter((200, 50))
    def metal_active():
        actions.append("active")
        return next(active)
    def native():
        actions.append("native")
        return dict(available=True, physical_footprint_bytes=100,
            internal_compressed_ledger_bytes=0,
            monotonic_start_ns=next(clock), monotonic_end_ns=next(clock))
    monkeypatch.setattr(witness, "sample_self_memory", native)
    metal = SimpleNamespace(get_active_memory=metal_active,
        get_cache_memory=lambda: actions.append("allocator") or 5,
        clear_cache=lambda: actions.append("clear"))
    def swap():
        actions.append("swap")
        return SimpleNamespace(used=0, sout=0)
    namespace = dict(mx=metal, psutil=SimpleNamespace(virtual_memory=virtual_memory, swap_memory=swap),
        time=SimpleNamespace(monotonic=lambda: 2, monotonic_ns=lambda: next(clock)),
        _SWAP_WINDOW_SECONDS=30, _swap_growth=lambda *a: (False, 0, 0),
        _swap_restore_ready=lambda *a: False)
    tree = ast.parse((Path(__file__).resolve().parents[1] / "runtime/pressure.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MemoryGovernor")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_run")
    exec(compile(ast.Module(body=[method], type_ignores=[]), "pressure.py", "exec"), namespace)
    waits = iter((False, True))
    cache = SimpleNamespace(max_bytes=1000)
    def shrink(value):
        actions.append("shrink")
        cache.max_bytes = value
    gov = SimpleNamespace(_stop=SimpleNamespace(wait=lambda _: next(waits)), poll_s=2,
        _swap_samples=deque([(0, 0, 0)]), _last_swap_pressure_at=None,
        _metal_ceiling=lambda *a: 8000, _peak_lock=threading.Lock(),
        _request_peak_metal_bytes=0, cache=cache, floor=100, shrink_step=0.15,
        shrinks=0, _set_cache_max=shrink, _pause_prefetch=lambda _: actions.append("pause"),
        _green_streak=10, critical=5_600_000_000, warn=5_800_000_000, green=6_000_000_000,
        _process_memory_observer=witness.GovernorProcessMemoryObserver())
    namespace["_run"](gov)
    assert actions == ["vm", "active", "swap", "shrink", "pause", "clear",
                       "vm", "active", "allocator", "native"]
    raw = next(line for line in capsys.readouterr().out.splitlines() if line.startswith('[process-memory] '))
    row = json.loads(raw.removeprefix('[process-memory] '))
    assert row["system_available_bytes"] == 5_200_000_000 and row["metal_active_bytes"] == 200
    assert row["reclamation_alignment"]["system_available_bytes"] == 6_200_000_000
    assert row["reclamation_alignment"]["metal_active_bytes"] == 50
    assert witness.summarize_reclamation_alignment([row])["passed"]
    assert gov.shrinks == 1 and gov._green_streak == 0


def test_sampler_does_not_import_or_evaluate_model_or_mutate_allocator():
    tree = ast.parse(Path(witness.__file__).read_text())
    assert not any(isinstance(n, ast.Attribute) and n.attr in {
        "synchronize", "eval", "async_eval", "clear_cache", "reset_peak_memory", "sleep"
    } for n in ast.walk(tree))
