"""Real scalar and server hooks, fake device only; no MLX/model imports."""

import ast
import copy
import json
import os
from pathlib import Path
import sys
import time
from types import ModuleType, SimpleNamespace
import weakref

import pytest

from runtime import phase_head_witness as witness


FLAG = "VMODEL_QWEN4_POST_GENERATION_MEMORY"


@pytest.fixture
def api(monkeypatch):
    events = []
    metal = ModuleType("mlx.core")
    metal.synchronize = lambda: events.append("sync")
    metal.clear_cache = lambda: events.append("clear")
    mlx = ModuleType("mlx")
    mlx.core = metal
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", metal)

    class Wrapper:
        def __init__(self, target):
            self.target = target

    mtp = ModuleType("runtime.qwen4_mtp")
    mtp.Qwen4MTPSpeculativeEngine = Wrapper
    monkeypatch.setitem(sys.modules, "runtime.qwen4_mtp", mtp)
    sample_count = 0

    def sample(target, device):
        nonlocal sample_count
        assert device is metal
        sample_count += 1
        events.append("sample")
        # A falling reading must never be clamped into an alleged reclaim win.
        return {"available": True, "system_available_bytes": 6000 - sample_count,
                "metal_active_bytes": 10, "metal_allocator_cache_bytes": 30}

    monkeypatch.setattr(witness, "sample_phase_head_memory", sample)
    source = Path(__file__).resolve().parents[1] / "runtime/server.py"
    names = {"_observe_qwen4_post_generation_memory", "_engine_generate", "_has_own_method"}
    nodes = [n for n in ast.parse(source.read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    namespace = {"__package__": "runtime", "os": os, "time": time,
                 "_request_expert_trace_target": lambda engine: None,
                 "_persist_request_expert_trace": lambda *args: events.append("trace"),
                 "_attach_generation_witness": lambda *args: events.append("tokens")}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    monkeypatch.delenv(FLAG, raising=False)
    monkeypatch.delenv("VMODEL_DEBUG_ENGINE_REPORT", raising=False)
    target = SimpleNamespace(cfg=SimpleNamespace(model_type="qwen4_exp"),
                             last_kv=object(), _hot_prompt_slots=[object()],
                             _h_last=object(), cache=object(), rng=object())
    return SimpleNamespace(events=events, metal=metal, Wrapper=Wrapper,
                           target=target, namespace=namespace,
                           observe=namespace["_observe_qwen4_post_generation_memory"],
                           generate=namespace["_engine_generate"])


@pytest.mark.parametrize("mode", [None, "", "0", "1", "true", "invalid"])
def test_default_off_never_touches_target_or_device(api, monkeypatch, mode):
    if mode is not None:
        monkeypatch.setenv(FLAG, mode)
    api.observe(object(), {})
    assert api.events == []


@pytest.mark.parametrize("mode,expected", [
    ("observe", ["sample"]),
    ("synchronize_clear_cache", ["sample", "sync", "sample", "clear", "sample"]),
])
def test_boundary_order_preserves_every_owner_and_payload(api, monkeypatch, mode, expected):
    monkeypatch.setenv(FLAG, mode)
    owners = dict(vars(api.target))
    result = {"tokens": [1, 2], "text": "PRIVATE", "total_s": 7,
              "path_stats": {"true_peak": 123, "cache": 456}}
    original = copy.deepcopy(result)
    api.observe(api.Wrapper(api.target), result)
    obs = result["path_stats"].pop("qwen4_post_generation_memory_witness")
    assert result == original
    assert vars(api.target) == owners
    assert api.events == expected
    assert obs["atomic"] is False
    assert obs["disposes_request_or_reuse_state"] is False
    assert obs["included_in_engine_total_s"] is False
    assert obs["included_in_http_wall_s"] is True
    assert obs["synchronizes_device"] is (mode != "observe")
    assert obs["synchronization_scope"] == (
        "default_stream_of_default_device" if mode != "observe" else "none")
    assert obs["clears_allocator_cache"] is (mode != "observe")
    assert obs["before"]["system_available_bytes"] == 5999
    if mode != "observe":
        assert obs["after_synchronize"]["system_available_bytes"] == 5998
        assert obs["after_clear_cache"]["system_available_bytes"] == 5997
        assert obs["synchronize_seconds"] >= 0
        assert obs["clear_cache_seconds"] >= 0
    else:
        assert "after_clear_cache" not in obs and "after_synchronize" not in obs
    assert obs["wall_seconds"] >= 0
    assert "PRIVATE" not in json.dumps(obs, allow_nan=False)


@pytest.mark.parametrize("kind", ["direct", "subclass", "proxy", "other-model"])
def test_strict_concrete_engine_gate(api, monkeypatch, kind):
    monkeypatch.setenv(FLAG, "synchronize_clear_cache")
    if kind == "direct":
        engine = api.target
    elif kind == "subclass":
        engine = type("Subclass", (api.Wrapper,), {})(api.target)
    elif kind == "proxy":
        engine = SimpleNamespace(target=api.target)
    else:
        api.target.cfg.model_type = "glm5_next"
        engine = api.Wrapper(api.target)
    api.observe(engine, {})
    assert not api.events


@pytest.mark.parametrize("operation", ["synchronize", "clear_cache"])
def test_device_errors_propagate_without_false_completion(api, monkeypatch, operation):
    monkeypatch.setenv(FLAG, "synchronize_clear_cache")
    error = RuntimeError("device operation failed")

    def fail():
        raise error

    setattr(api.metal, operation, fail)
    result = {"tokens": [1]}
    with pytest.raises(RuntimeError) as raised:
        api.observe(api.Wrapper(api.target), result)
    assert raised.value is error and result == {"tokens": [1]}
    if operation == "synchronize":
        assert "clear" not in api.events


@pytest.mark.parametrize("failed_sample", [1, 2, 3])
def test_observation_failure_does_not_skip_or_repeat_barrier(api, monkeypatch, failed_sample):
    monkeypatch.setenv(FLAG, "synchronize_clear_cache")
    original = witness.sample_phase_head_memory
    count = 0

    def sample(*args):
        nonlocal count
        count += 1
        if count == failed_sample:
            raise MemoryError("PRIVATE observation failure")
        return original(*args)

    monkeypatch.setattr(witness, "sample_phase_head_memory", sample)
    result = {}
    api.observe(api.Wrapper(api.target), result)
    obs = result["path_stats"]["qwen4_post_generation_memory_witness"]
    key = ("before", "after_synchronize", "after_clear_cache")[failed_sample - 1]
    assert obs[key]["available"] is False
    assert api.events.count("sync") == api.events.count("clear") == 1
    assert count == 3 and "PRIVATE" not in json.dumps(obs)


def test_attachment_failure_does_not_repeat_operations(api, monkeypatch):
    monkeypatch.setenv(FLAG, "synchronize_clear_cache")

    class Refuse(dict):
        def __setitem__(self, key, value):
            raise MemoryError("observation attachment failed")

    result = {"tokens": [1], "path_stats": Refuse()}
    api.observe(api.Wrapper(api.target), result)
    assert result == {"tokens": [1], "path_stats": {}}
    assert api.events.count("sync") == api.events.count("clear") == 1


def test_clock_failure_does_not_turn_operation_time_into_zero(api, monkeypatch):
    def fail():
        raise RuntimeError("clock failure")

    monkeypatch.setattr(witness, "time", SimpleNamespace(perf_counter=fail))
    obs = witness.post_generation_memory_witness(api.target, api.metal, barrier=True)
    assert obs["wall_seconds"] is None and obs["synchronize_seconds"] is None
    assert obs["clear_cache_seconds"] is None
    assert api.events.count("sync") == api.events.count("clear") == 1


@pytest.mark.parametrize("generation_fails", [False, True])
def test_actual_server_hook_follows_frame_teardown_before_protocol_observers(
        api, monkeypatch, generation_fails):
    monkeypatch.setenv(FLAG, "synchronize_clear_cache")
    refs = []
    class Temporary:
        pass

    def generate(self, *args):
        temporary = Temporary()
        refs.append(weakref.ref(temporary))
        api.events.append("generate")
        if generation_fails:
            raise ValueError("generation failed")
        return {"tokens": [5], "text": "untouched", "total_s": 4}

    def sync():
        assert refs[0]() is None
        api.events.append("sync")

    api.Wrapper.generate = generate
    api.metal.synchronize = sync
    engine = api.Wrapper(api.target)
    if generation_fails:
        with pytest.raises(ValueError):
            api.generate(engine, "prompt", 1024)
        assert api.events == ["generate"]
    else:
        result = api.generate(engine, "prompt", 1024)
        assert result["tokens"] == [5] and result["total_s"] == 4
        assert api.events == ["generate", "sample", "sync", "sample", "clear",
                              "sample", "tokens", "trace"]
