"""Native ABI decoding and governor-order tests with no native call or MLX."""

import ast
import ctypes
import json
from pathlib import Path
import threading
from types import SimpleNamespace
from collections import deque

import pytest

from runtime import process_memory_witness as witness
from runtime import phase_head_witness


@pytest.fixture
def native(monkeypatch):
    state = SimpleNamespace(code=0, count=witness.REV1_COUNT, calls=0)
    monkeypatch.setattr(witness.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(witness.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(witness, "_self_task_port", lambda lib: 773)

    def call(port, flavor, output, count):
        assert port == 773 and flavor == 22
        assert ctypes.cast(count, ctypes.POINTER(ctypes.c_uint32)).contents.value == 38
        state.calls += 1
        info = ctypes.cast(output, ctypes.POINTER(witness.TaskVMInfoRev1)).contents
        for index, (_, field) in enumerate(witness.FIELDS.items()):
            setattr(info, field, 2**48 + index)
        ctypes.cast(count, ctypes.POINTER(ctypes.c_uint32)).contents.value = state.count
        return state.code

    monkeypatch.setattr(witness, "_bindings", lambda: (object(), call))
    return state


def test_rev1_prefix_layout_and_all_large_scalar_decodes(native):
    assert ctypes.sizeof(witness.TaskVMInfoRev1) == 152
    assert ctypes.alignment(witness.TaskVMInfoRev1) == 4
    assert witness.TaskVMInfoRev1.resident_size.offset == 16
    assert witness.TaskVMInfoRev1.compressed.offset == 120
    assert witness.TaskVMInfoRev1.phys_footprint.offset == 144
    row = witness.sample_self_memory()
    assert row["available"] is True and row["reason"] is None
    assert row["returned_natural_count"] == 38 and row["kernel_return_code"] == 0
    for index, key in enumerate(witness.FIELDS):
        assert row[key] == 2**48 + index
    assert row["scope"] == "current_process_only" and row["atomic"] is False
    assert row["monotonic_end_ns"] >= row["monotonic_start_ns"]
    assert row["observation_seconds"] >= 0 and native.calls == 1
    assert "virtual_size" not in row and "port" not in row
    json.dumps(row, allow_nan=False)


@pytest.mark.parametrize("code,count,reason", [
    (5, 38, "task-info-failed"), (0, 0, "unexpected-record-count"),
    (0, 36, "unexpected-record-count"), (0, 40, "unexpected-record-count"),
])
def test_failed_or_short_native_record_never_fabricates_zero(native, code, count, reason):
    native.code, native.count = code, count
    row = witness.sample_self_memory()
    assert row["available"] is False and row["reason"] == reason
    assert all(row[key] is None for key in witness.FIELDS)


@pytest.mark.parametrize("system,machine", [("Linux", "x86_64"), ("Darwin", "unknown")])
def test_unsupported_platform_does_not_call_native(native, monkeypatch, system, machine):
    monkeypatch.setattr(witness.platform, "system", lambda: system)
    monkeypatch.setattr(witness.platform, "machine", lambda: machine)
    row = witness.sample_self_memory()
    assert row["reason"] == "unsupported-platform" and native.calls == 0
    assert all(row[key] is None for key in witness.FIELDS)


def test_native_load_failure_redacts_details(native, monkeypatch):
    def fail():
        raise OSError("PRIVATE path or credentials")
    monkeypatch.setattr(witness, "_bindings", fail)
    row = witness.sample_self_memory()
    assert row["reason"] == "observation-error"
    assert all(row[key] is None for key in witness.FIELDS)
    assert "PRIVATE" not in json.dumps(row)


@pytest.mark.parametrize("flag", [None, "", "0", "true", "invalid", "1"])
def test_strict_flag_and_factory_do_not_probe(native, monkeypatch, flag):
    monkeypatch.delenv(witness.FLAG, raising=False)
    if flag is not None:
        monkeypatch.setenv(witness.FLAG, flag)
    observer = witness.process_memory_observer_from_environment()
    assert (observer is not None) is (flag == "1")
    assert native.calls == 0


def inputs(now):
    return dict(governor_monotonic_s=now, system_available_bytes=6000,
                system_swap_used_bytes=100, system_swap_out_bytes=400,
                metal_active_bytes=3000, cache_budget_bytes_after_response=500,
                swap_pressure_response=False)


def test_periodic_trace_is_rate_and_count_bounded_with_visible_truncation(native, monkeypatch, capsys):
    monkeypatch.setattr(witness, "MAX_SAMPLES", 2)
    observer = witness.GovernorProcessMemoryObserver()
    for now in (0, 1, 1.999, 2, 3, 4, 6, 100):
        observer.record(**inputs(now))
    rows = [json.loads(line.removeprefix('[process-memory] '))
            for line in capsys.readouterr().out.splitlines()]
    assert len(rows) == 3 and native.calls == 2
    assert [r["sample_index"] for r in rows[:2]] == [1, 2]
    assert rows[-1]["reason"] == "sample-limit"
    assert rows[-1]["coverage_complete"] is False
    assert rows[0]["system_swap_out_bytes"] == 400
    assert "samples" not in vars(observer)  # no in-memory timeline


def test_periodic_reader_failure_is_an_unavailable_record(monkeypatch, capsys):
    def fail():
        raise RuntimeError("PRIVATE")
    monkeypatch.setattr(witness, "sample_self_memory", fail)
    witness.GovernorProcessMemoryObserver().record(**inputs(1))
    text = capsys.readouterr().out
    row = json.loads(text.removeprefix('[process-memory] '))
    assert row["process"] == {"available": False, "reason": "observation-error"}
    assert "PRIVATE" not in text


@pytest.mark.parametrize("observer_mode", ["off", "on", "raises"])
def test_governor_actions_precede_observation_and_survive_its_failure(observer_mode):
    tree = ast.parse((Path(__file__).resolve().parents[1] / "runtime/pressure.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MemoryGovernor")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_run")
    actions = []
    def record(**kwargs):
        assert actions == [("cache", 850), ("pause", True), "clear"]
        assert kwargs["cache_budget_bytes_after_response"] == 850
        assert kwargs["swap_pressure_response"] is True
        actions.append("observe")
        if observer_mode == "raises":
            raise RuntimeError("telemetry cannot roll back a safety response")
    namespace = dict(
        mx=SimpleNamespace(get_active_memory=lambda: 3000,
                           clear_cache=lambda: actions.append("clear")),
        psutil=SimpleNamespace(virtual_memory=lambda: SimpleNamespace(available=6000),
                               swap_memory=lambda: SimpleNamespace(used=200, sout=500)),
        time=SimpleNamespace(monotonic=lambda: 2), _SWAP_WINDOW_SECONDS=30,
        _swap_growth=lambda *a: (True, 100, 100),
        _swap_restore_ready=lambda *a: False)
    exec(compile(ast.Module(body=[method], type_ignores=[]), "pressure.py", "exec"), namespace)
    waits = iter((False, True))
    cache = SimpleNamespace(max_bytes=1000)
    gov = SimpleNamespace(
        _stop=SimpleNamespace(wait=lambda interval: next(waits)), poll_s=2,
        _swap_samples=deque([(0, 100, 400)]), _last_swap_pressure_at=None,
        swap_pressure_events=0, swap_pressure_used_growth_bytes=0,
        swap_pressure_out_growth_bytes=0, _metal_ceiling=lambda *a: 8000,
        _peak_lock=threading.Lock(), _request_peak_metal_bytes=10,
        cache=cache, floor=100, shrink_step=0.15, shrinks=0,
        _pause_prefetch=lambda value: actions.append(("pause", value)),
        _green_streak=10, critical=100, warn=200, green=300,
        _process_memory_observer=None if observer_mode == "off" else SimpleNamespace(record=record))
    def shrink(value):
        cache.max_bytes = value
        actions.append(("cache", value))
    gov._set_cache_max = shrink
    namespace["_run"](gov)
    assert actions[:3] == [("cache", 850), ("pause", True), "clear"]
    assert len(actions) == (3 if observer_mode == "off" else 4)
    assert gov.shrinks == 1 and gov._green_streak == 0
    assert gov.swap_pressure_events == 1 and gov._request_peak_metal_bytes == 3000


@pytest.mark.parametrize("flag", ["0", "1"])
def test_phase_observer_adds_only_nested_self_snapshot(native, monkeypatch, flag):
    monkeypatch.setenv(witness.FLAG, flag)
    monkeypatch.setattr(phase_head_witness.psutil, "virtual_memory", lambda: SimpleNamespace(available=10))
    monkeypatch.setattr(phase_head_witness.psutil, "swap_memory", lambda: SimpleNamespace(used=20))
    target = SimpleNamespace(cache=SimpleNamespace(total_bytes=30, pinned_bytes=4, max_bytes=50),
                             _true_peak_metal_bytes=60)
    metal = SimpleNamespace(get_active_memory=lambda: 70, get_cache_memory=lambda: 8,
                            get_peak_memory=lambda: 90)
    row = phase_head_witness.sample_phase_head_memory(target, metal)
    assert row["available"] is True and row["system_available_bytes"] == 10
    assert ("process_memory" in row) is (flag == "1")
    assert native.calls == int(flag)


def test_source_has_no_model_or_privilege_or_new_thread_dependency():
    source = Path(witness.__file__).read_text()
    tree = ast.parse(source)
    imports = [alias.name for n in ast.walk(tree) if isinstance(n, ast.Import) for alias in n.names]
    assert not set(imports) & {"mlx", "torch", "threading", "subprocess"}
    assert not any(isinstance(n, ast.Attribute) and n.attr in
                   {"synchronize", "clear_cache", "task_for_pid", "reset_peak_memory"}
                   for n in ast.walk(tree))
