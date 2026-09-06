"""Actual release hook and scalar helper, without importing MLX or a model."""

import ast
import copy
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from runtime import phase_head_witness as witness


@pytest.fixture
def harness(monkeypatch):
    events = []
    state = {"resident": 1300, "pinned": 1200, "active": 1600,
             "allocator": 30, "system": 6000, "swap": 100}

    def read(name, value):
        events.append(name)
        return value

    class Cache:
        max_bytes = 2000

        @property
        def total_bytes(self):
            return read("resident", state["resident"])

        @property
        def pinned_bytes(self):
            return read("pinned", state["pinned"])

    def release():
        events.append("release")
        state.update(resident=100, pinned=0, active=400, allocator=1230,
                     system=5900, swap=120)
        return 1200

    target = SimpleNamespace(cache=Cache(), _true_peak_metal_bytes=1900,
                             _suspend_qwen4_phase_lm_head=release)
    metal = SimpleNamespace(
        get_active_memory=lambda: read("active", state["active"]),
        get_cache_memory=lambda: read("allocator", state["allocator"]),
        get_peak_memory=lambda: read("peak", 1700))
    monkeypatch.setattr(witness, "psutil", SimpleNamespace(
        virtual_memory=lambda: SimpleNamespace(available=read("system", state["system"])),
        swap_memory=lambda: SimpleNamespace(used=read("swap", state["swap"]))))
    source = Path(__file__).resolve().parents[1] / "runtime/qwen4_mtp.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef)
               and node.name == "Qwen4MTPSpeculativeEngine")
    names = {"_idle_phase_head_memory_sample", "_release_idle_phase_head"}
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef)
               and node.name in names]
    assert len(methods) == len(names)
    namespace = {"__package__": "runtime", "os": os, "time": time, "mx": metal}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"),
         namespace)
    hook = type("ReleaseHarness", (), {name: namespace[name] for name in names})()
    hook.target = target
    monkeypatch.delenv("VMODEL_GENERATION_WITNESS", raising=False)
    return SimpleNamespace(hook=hook, target=target, metal=metal, events=events,
                           state=state)


@pytest.mark.parametrize("flag", [None, "", "0", "true", "invalid"])
def test_default_off_does_no_observer_reads(harness, monkeypatch, flag):
    if flag is not None:
        monkeypatch.setenv("VMODEL_GENERATION_WITNESS", flag)
    result = {"tokens": [7, 8], "text": "private", "path_stats": {"existing": 9}}
    harness.hook._release_idle_phase_head(result)
    assert harness.events == ["release"]
    assert result == {"tokens": [7, 8], "text": "private", "path_stats": {
        "existing": 9, "qwen4_mtp_idle_head_release_calls": 1,
        "qwen4_mtp_idle_head_release_bytes": 1200}}


def test_samples_bracket_exactly_one_release_and_preserve_historical_values(harness, monkeypatch):
    monkeypatch.setenv("VMODEL_GENERATION_WITNESS", "1")
    result = {"tokens": [7, 8], "text": "private", "generation_witness": {"old": True},
              "path_stats": {"weight_cache_pinned_bytes": 1200}}
    original = copy.deepcopy(result)
    harness.hook._release_idle_phase_head(result)
    events = ["resident", "pinned", "active", "allocator", "peak", "system", "swap"]
    assert harness.events == events + ["release"] + events
    stats = result["path_stats"]
    observation = stats["qwen4_mtp_idle_head_memory_witness"]
    assert observation["schema"] == "voom.phase-head-memory-witness.v1"
    assert observation["scope"] == "inside_generation_around_idle_head_release"
    assert observation["atomic"] is observation["synchronizes_device"] is False
    assert observation["release_callable"] is True
    assert observation["released_logical_bytes"] == 1200
    assert observation["release_seconds"] >= 0
    before, after = observation["before"], observation["after"]
    assert before["available"] is after["available"] is True
    assert before["weight_cache_pinned_bytes"] == 1200
    assert after["weight_cache_pinned_bytes"] == 0
    assert before["metal_allocator_cache_bytes"] == 30
    assert after["metal_allocator_cache_bytes"] == 1230
    assert before["system_available_bytes"] == 6000
    assert after["system_available_bytes"] == 5900  # no claimed/clamped reclaim
    assert after["system_swap_used_bytes"] == 120
    assert after["metal_peak_since_last_reset_bytes"] == 1700
    assert after["request_true_peak_metal_bytes"] == 1900
    for sample in (before, after):
        assert sample["observation_seconds"] >= 0
        assert sample["unavailable_fields"] == []
    assert "private" not in json.dumps(observation, allow_nan=False)
    assert result["tokens"] == original["tokens"]
    assert result["text"] == original["text"]
    assert result["generation_witness"] == original["generation_witness"]
    assert stats["weight_cache_pinned_bytes"] == 1200  # earlier sample unchanged


@pytest.mark.parametrize("failed_call", [1, 2])
def test_observation_failure_never_skips_or_repeats_release(harness, monkeypatch, failed_call):
    monkeypatch.setenv("VMODEL_GENERATION_WITNESS", "1")
    calls = 0
    original_sample = witness.sample_phase_head_memory

    def sample(*args):
        nonlocal calls
        calls += 1
        if calls == failed_call:
            raise RuntimeError("PRIVATE failure details")
        return original_sample(*args)

    monkeypatch.setattr(witness, "sample_phase_head_memory", sample)
    result = {"tokens": [7], "text": "original"}
    harness.hook._release_idle_phase_head(result)
    observation = result["path_stats"]["qwen4_mtp_idle_head_memory_witness"]
    assert calls == 2 and harness.events.count("release") == 1
    failed = observation["before" if failed_call == 1 else "after"]
    assert failed["available"] is False and failed["reason"] == "observation-error"
    assert "PRIVATE" not in json.dumps(observation, allow_nan=False)
    assert result["tokens"] == [7] and result["text"] == "original"
    assert result["path_stats"]["qwen4_mtp_idle_head_release_bytes"] == 1200


def test_real_release_exception_is_not_swallowed_or_replaced(harness, monkeypatch):
    monkeypatch.setenv("VMODEL_GENERATION_WITNESS", "1")
    error = RuntimeError("original release exception")

    def release():
        harness.events.append("release")
        raise error

    harness.target._suspend_qwen4_phase_lm_head = release
    result = {"tokens": [7]}
    with pytest.raises(RuntimeError) as raised:
        harness.hook._release_idle_phase_head(result)
    assert raised.value is error
    assert harness.events[-1] == "release"
    assert harness.events.count("release") == 1
    assert result == {"tokens": [7]}


@pytest.mark.parametrize("failed_call", [1, 2])
def test_sample_method_failure_does_not_escape_diagnostic_guard(harness, monkeypatch, failed_call):
    monkeypatch.setenv("VMODEL_GENERATION_WITNESS", "1")
    original = harness.hook._idle_phase_head_memory_sample
    calls = 0

    def sample():
        nonlocal calls
        calls += 1
        if calls == failed_call:
            raise MemoryError("observer timing/annotation failed")
        return original()

    monkeypatch.setattr(harness.hook, "_idle_phase_head_memory_sample", sample)
    result = {"tokens": [7], "text": "original"}
    harness.hook._release_idle_phase_head(result)
    observation = result["path_stats"]["qwen4_mtp_idle_head_memory_witness"]
    assert observation["before" if failed_call == 1 else "after"] is None
    assert calls == 2 and harness.events.count("release") == 1
    assert result["tokens"] == [7] and result["text"] == "original"
    assert result["path_stats"]["qwen4_mtp_idle_head_release_bytes"] == 1200


def test_optional_attachment_failure_does_not_fail_completed_generation(harness, monkeypatch):
    monkeypatch.setenv("VMODEL_GENERATION_WITNESS", "1")

    class RefuseWitness(dict):
        def __setitem__(self, key, value):
            if key == "qwen4_mtp_idle_head_memory_witness":
                raise MemoryError("optional annotation failed")
            super().__setitem__(key, value)

    result = {"tokens": [7], "text": "original", "path_stats": RefuseWitness()}
    harness.hook._release_idle_phase_head(result)
    assert harness.events.count("release") == 1
    assert result["tokens"] == [7] and result["text"] == "original"
    assert result["path_stats"] == {
        "qwen4_mtp_idle_head_release_calls": 1,
        "qwen4_mtp_idle_head_release_bytes": 1200}


def test_observer_clock_failure_does_not_prevent_release(harness, monkeypatch):
    monkeypatch.setenv("VMODEL_GENERATION_WITNESS", "1")

    def fail():
        raise RuntimeError("observer clock failed")

    globals_ = harness.hook._release_idle_phase_head.__func__.__globals__
    monkeypatch.setitem(globals_, "time", SimpleNamespace(perf_counter=fail))
    result = {"tokens": [7]}
    harness.hook._release_idle_phase_head(result)
    observation = result["path_stats"]["qwen4_mtp_idle_head_memory_witness"]
    assert observation["before"] is observation["after"] is None
    assert observation["release_seconds"] is None
    assert harness.events == ["release"]
    assert result["tokens"] == [7]


def test_no_release_method_is_distinct_from_zero_byte_release(harness, monkeypatch):
    monkeypatch.setenv("VMODEL_GENERATION_WITNESS", "1")
    del harness.target._suspend_qwen4_phase_lm_head
    result = {}
    harness.hook._release_idle_phase_head(result)
    observation = result["path_stats"]["qwen4_mtp_idle_head_memory_witness"]
    assert observation["release_callable"] is False
    assert observation["released_logical_bytes"] == 0
    assert "release" not in harness.events


@pytest.mark.parametrize("bad", [None, -1, True, 1.5, "12", float("nan")])
def test_invalid_count_is_unavailable_not_zero(harness, bad):
    harness.target.cache.max_bytes = bad
    observation = witness.sample_phase_head_memory(harness.target, harness.metal)
    assert observation["available"] is False
    assert observation["weight_cache_budget_bytes"] is None
    assert observation["unavailable_fields"] == ["weight_cache_budget_bytes"]
    assert observation["metal_active_bytes"] == 1600
    json.dumps(observation, allow_nan=False)


def test_missing_metal_getter_does_not_drop_other_fields(harness):
    del harness.metal.get_cache_memory
    observation = witness.sample_phase_head_memory(harness.target, harness.metal)
    assert observation["available"] is False
    assert observation["metal_allocator_cache_bytes"] is None
    assert observation["unavailable_fields"] == ["metal_allocator_cache_bytes"]
    assert observation["weight_cache_resident_bytes"] == 1300


def test_legitimate_zero_bytes_remain_available(harness):
    harness.target.cache.max_bytes = 0
    observation = witness.sample_phase_head_memory(harness.target, harness.metal)
    assert observation["available"] is True
    assert observation["weight_cache_budget_bytes"] == 0
