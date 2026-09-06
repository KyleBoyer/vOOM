"""Serving-only reference detachment and protocol order; no MLX imports."""

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

from runtime.request_state import detach_unretained_qwen4_endpoint as detach
from runtime import phase_head_witness


FLAG = "VMODEL_QWEN4_RELEASE_UNRETAINED_ENDPOINT"


class Plain:
    def __init__(self, payload=None):
        self.payload = payload if payload is not None else [object()]

    def nbytes(self):
        return 123456

    def release(self):
        raise AssertionError("even an available release() must never be called")


def target():
    return SimpleNamespace(
        cfg=SimpleNamespace(model_type="qwen4_exp"),
        rc=SimpleNamespace(hot_prompt_kv=True, qwen4_hot_kv_tile_aligned=True),
        last_kv=Plain(), _hot_prompt_slots=[SimpleNamespace(kv=Plain(),
            tokens=(1, 2, 3), logits=object(), hidden=object(), metadata={"x": 7})],
        _h_last=object(), _h_window=object(), _provisional=None,
        _serial_kda_endpoints=None, _serial_qwen4_endpoints=None,
        _serial_kda_factors=None, _hot_kv_persist=None, _prompt_kv_store=None,
        _vision_prompt_cache=None, _glm53_vision_prompt_cache=None,
        _vision_embedding_cache={}, _glm53_vision_embedding_cache={},
    )


def test_drop_only_engine_root_preserves_slot_identity_payload_and_hidden_owners():
    owner = target()
    # Different cache containers deliberately share payload storage.
    slot = owner._hot_prompt_slots[0]
    owner.last_kv.payload = slot.kv.payload
    endpoint_ref = weakref.ref(owner.last_kv)
    before = {k: v for k, v in vars(owner).items() if k != "last_kv"}
    payload = slot.kv.payload
    decision = detach(owner, Plain)
    assert decision["detached"] is True
    assert decision["detached_endpoint_logical_bytes"] == 123456
    assert decision["physical_reclamation_proven"] is False
    assert decision["calls_endpoint_release"] is False
    assert owner.last_kv is None and endpoint_ref() is None
    assert {k: v for k, v in vars(owner).items() if k != "last_kv"} == before
    assert owner._hot_prompt_slots[0] is slot and slot.kv.payload is payload
    assert payload and slot.tokens == (1, 2, 3) and slot.metadata == {"x": 7}
    assert detach(owner, Plain)["reason"] == "no-plain-endpoint"
    assert owner._hot_prompt_slots[0] is slot


def test_external_alias_remains_usable_no_claim_that_object_was_freed():
    owner = target()
    external = owner.last_kv
    assert detach(owner, Plain)["detached"] is True
    assert external.nbytes() == 123456 and external.payload


@pytest.mark.parametrize("index", [0, 1, 2])
def test_any_retained_alias_prevents_detachment(index):
    owner = target()
    owner._hot_prompt_slots = [SimpleNamespace(kv=Plain()) for _ in range(3)]
    owner._hot_prompt_slots[index].kv = owner.last_kv
    before = dict(vars(owner))
    assert detach(owner, Plain)["reason"] == "endpoint-is-retained-slot"
    assert vars(owner) == before


@pytest.mark.parametrize("field", ["_hot_kv_persist", "_prompt_kv_store",
    "_vision_prompt_cache", "_glm53_vision_prompt_cache", "_provisional",
    "_serial_kda_endpoints", "_serial_qwen4_endpoints", "_serial_kda_factors",
    "_vision_embedding_cache", "_glm53_vision_embedding_cache"])
def test_competing_owner_blocks_detachment(field):
    owner = target()
    setattr(owner, field, object())
    endpoint = owner.last_kv
    assert detach(owner, Plain)["reason"] == "other-state-owner"
    assert owner.last_kv is endpoint


@pytest.mark.parametrize("change", ["model", "hot-off", "aligned-off", "unknown-policy",
    "missing-slots", "empty-slots", "tuple-slots", "unknown-slot", "paged-slot",
    "missing-endpoint", "paged-endpoint"])
def test_unknown_or_unsupported_policy_and_resource_types_are_untouched(change):
    owner = target()
    derived = type("Paged", (Plain,), {})
    if change == "model":
        owner.cfg.model_type = "glm5_next"
    elif change == "hot-off":
        owner.rc.hot_prompt_kv = False
    elif change == "aligned-off":
        owner.rc.qwen4_hot_kv_tile_aligned = False
    elif change == "unknown-policy":
        owner.rc = SimpleNamespace()
    elif change == "missing-slots":
        del owner._hot_prompt_slots
    elif change == "empty-slots":
        owner._hot_prompt_slots = []
    elif change == "tuple-slots":
        owner._hot_prompt_slots = tuple(owner._hot_prompt_slots)
    elif change == "unknown-slot":
        owner._hot_prompt_slots = [object()]
    elif change == "paged-slot":
        owner._hot_prompt_slots[0].kv = derived()
    elif change == "missing-endpoint":
        owner.last_kv = None
    else:
        owner.last_kv = derived()
    before = dict(vars(owner))
    assert detach(owner, Plain)["detached"] is False
    assert vars(owner) == before


@pytest.mark.parametrize("size", [-1, True, 1.5, None])
def test_invalid_logical_count_never_mutates_owner(size):
    owner = target()
    endpoint = owner.last_kv
    endpoint.nbytes = lambda: size
    assert detach(owner, Plain)["reason"] == "invalid-logical-byte-count"
    assert owner.last_kv is endpoint


@pytest.fixture
def api(monkeypatch):
    events = []
    owner = target()
    endpoint_ref = weakref.ref(owner.last_kv)
    metal = ModuleType("mlx.core")

    def clear_cache():
        assert owner.last_kv is None and endpoint_ref() is None
        events.append("clear")

    metal.clear_cache = clear_cache
    mlx = ModuleType("mlx")
    mlx.core = metal
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", metal)
    kv_module = ModuleType("runtime.kv_cache")
    kv_module.KVCache = Plain
    monkeypatch.setitem(sys.modules, "runtime.kv_cache", kv_module)

    class Wrapper:
        def __init__(self, target):
            self.target = target

        def generate(self, *args, **kwargs):
            events.append("generate")
            return {"tokens": [7, 8], "text": "PRIVATE", "total_s": 5,
                    "kv_bytes": 123456, "path_stats": {"old": 3}}

        def report(self):
            assert owner.last_kv is not None
            events.append("report")
            return ""

    mtp = ModuleType("runtime.qwen4_mtp")
    mtp.Qwen4MTPSpeculativeEngine = Wrapper
    monkeypatch.setitem(sys.modules, "runtime.qwen4_mtp", mtp)

    def sample(target_, metal_):
        assert target_ is owner and metal_ is metal
        events.append("sample")
        return {"available": True, "metal_active_bytes": 100 if owner.last_kv else 20}

    monkeypatch.setattr(phase_head_witness, "sample_phase_head_memory", sample)
    source = Path(__file__).resolve().parents[1] / "runtime/server.py"
    names = {"_release_qwen4_unretained_endpoint", "_engine_generate", "_has_own_method"}
    nodes = [n for n in ast.parse(source.read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    ns = {"__package__": "runtime", "os": os, "time": time,
          "_request_expert_trace_target": lambda e: None,
          "_observe_qwen4_post_generation_memory": lambda *a: events.append("boundary"),
          "_attach_generation_witness": lambda *a: events.append("tokens")}

    def trace(*args):
        assert owner.last_kv is not None
        events.append("trace")

    ns["_persist_request_expert_trace"] = trace
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), ns)
    monkeypatch.delenv(FLAG, raising=False)
    monkeypatch.delenv("VMODEL_DEBUG_ENGINE_REPORT", raising=False)
    return SimpleNamespace(owner=owner, metal=metal, Wrapper=Wrapper, events=events,
        release=ns["_release_qwen4_unretained_endpoint"], generate=ns["_engine_generate"])


@pytest.mark.parametrize("flag", [None, "", "0", "true", "invalid"])
def test_default_off_touches_neither_engine_nor_device(api, monkeypatch, flag):
    if flag is not None:
        monkeypatch.setenv(FLAG, flag)
    api.release(object(), {})
    assert not api.events


@pytest.mark.parametrize("kind", ["direct", "subclass", "proxy"])
def test_concrete_serving_wrapper_required(api, monkeypatch, kind):
    monkeypatch.setenv(FLAG, "1")
    engine = (api.owner if kind == "direct" else
              type("Subclass", (api.Wrapper,), {})(api.owner) if kind == "subclass"
              else SimpleNamespace(target=api.owner))
    api.release(engine, {})
    assert api.owner.last_kv is not None and not api.events


def test_server_detaches_after_consumers_without_altering_result(api, monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv("VMODEL_DEBUG_ENGINE_REPORT", "1")
    result = api.generate(api.Wrapper(api.owner), "prompt", 1024)
    obs = result["path_stats"].pop("qwen4_unretained_endpoint_release")
    assert result == {"tokens": [7, 8], "text": "PRIVATE", "total_s": 5,
                      "kv_bytes": 123456, "path_stats": {"old": 3}}
    assert api.events == ["generate", "boundary", "report", "tokens", "trace",
                          "sample", "sample", "clear", "sample"]
    assert obs["before"]["metal_active_bytes"] == 100
    assert obs["after_detach"]["metal_active_bytes"] == 20
    assert obs["decision"]["detached"] is True
    assert obs["included_in_http_wall_s"] is True
    assert obs["included_in_engine_total_s"] is obs["synchronizes_device"] is False
    assert obs["wall_seconds"] >= obs["clear_cache_seconds"] >= 0
    assert "PRIVATE" not in json.dumps(obs, allow_nan=False)


def test_skip_does_not_clear_allocator_or_touch_retained_alias(api, monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    api.owner._hot_prompt_slots[0].kv = api.owner.last_kv
    result = {}
    api.release(api.Wrapper(api.owner), result)
    obs = result["path_stats"]["qwen4_unretained_endpoint_release"]
    assert obs["decision"]["reason"] == "endpoint-is-retained-slot"
    assert obs["after_clear_cache"] is None
    assert api.events == ["sample", "sample"]


def test_failed_generation_never_reaches_disposal(api, monkeypatch):
    monkeypatch.setenv(FLAG, "1")

    def fail(*args, **kwargs):
        raise RuntimeError("original generation failure")

    api.Wrapper.generate = fail
    with pytest.raises(RuntimeError, match="original generation failure"):
        api.generate(api.Wrapper(api.owner), "prompt", 1024)
    assert api.owner.last_kv is not None and not api.events


def test_observation_failure_does_not_prevent_reference_disposal(api, monkeypatch):
    monkeypatch.setenv(FLAG, "1")

    def fail(*args):
        raise MemoryError("PRIVATE")

    monkeypatch.setattr(phase_head_witness, "sample_phase_head_memory", fail)
    result = {}
    api.release(api.Wrapper(api.owner), result)
    obs = result["path_stats"]["qwen4_unretained_endpoint_release"]
    assert all(obs[key]["available"] is False for key in
               ("before", "after_detach", "after_clear_cache"))
    assert api.owner.last_kv is None and api.events == ["clear"]
    assert "PRIVATE" not in json.dumps(obs)


def test_device_failure_is_not_a_successful_reclamation_witness(api, monkeypatch):
    monkeypatch.setenv(FLAG, "1")

    def fail():
        raise RuntimeError("device failure")

    api.metal.clear_cache = fail
    result = {"tokens": [1]}
    with pytest.raises(RuntimeError, match="device failure"):
        api.release(api.Wrapper(api.owner), result)
    assert result == {"tokens": [1]}
