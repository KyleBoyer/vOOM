"""Pure AST-extracted serving ownership tests; no MLX/model imports.

These execute the real helper and complete HTTP lock/dispatch methods with
fake protocol consumers. They prove reference disposal and hook ordering,
not physical Metal reclamation, model equivalence, or a pressure-gate pass.
"""

from __future__ import annotations

import ast
import math
import os
from pathlib import Path
import sys
import time
from types import ModuleType, SimpleNamespace
import weakref

import pytest


ROOT = Path(__file__).resolve().parents[1]
FLAG = "VMODEL_QWEN4_RELEASE_IDLE_REQUEST_STATE"


@pytest.fixture
def source_api(monkeypatch):
    tree = ast.parse((ROOT / "runtime/server.py").read_text())
    helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == "_release_qwen4_idle_request_state")
    handler = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "Handler")
    methods = [node for node in handler.body if isinstance(node, ast.FunctionDef)
               and node.name in ("do_POST", "_do_post_locked")]

    class Wrapper:
        def __init__(self, target):
            self.target = target

        def __getattr__(self, name):
            return getattr(self.target, name)

    fake_mtp = ModuleType("runtime.qwen4_mtp")
    fake_mtp.Qwen4MTPSpeculativeEngine = Wrapper
    monkeypatch.setitem(sys.modules, "runtime.qwen4_mtp", fake_mtp)
    # The package and cleanup helper are dependency-free. Do not import the
    # real server or MTP controller just to exercise their serving boundary.
    state_tree = ast.parse((ROOT / "runtime/request_state.py").read_text())
    namespace = {
        "__package__": "runtime", "os": os, "math": math, "time": time,
        "RequestValidationError": type("RequestValidationError", (ValueError,), {}),
        "_DEFAULT_RESPONSE_WRITE_TIMEOUT_SECONDS": 30,
    }
    exec(compile(state_tree, str(ROOT / "runtime/request_state.py"), "exec"), namespace)
    exec(compile(ast.Module(body=[helper, *methods], type_ignores=[]),
                 str(ROOT / "runtime/server.py"), "exec"), namespace)
    monkeypatch.delenv(FLAG, raising=False)
    monkeypatch.delenv("VMODEL_CAPTURE_REQUESTS", raising=False)
    monkeypatch.delenv("VMODEL_RESPONSE_WRITE_TIMEOUT_SECONDS", raising=False)
    return SimpleNamespace(namespace=namespace, Wrapper=Wrapper,
                           cleanup=namespace[helper.name])


def _target(api, events=None):
    events = events if events is not None else []

    class State:
        def nbytes(self):
            return 1_031_050_560

        def release(self):
            events.append("endpoint-release")

    target = SimpleNamespace(
        cfg=SimpleNamespace(model_type="qwen4_exp"),
        rc=SimpleNamespace(hot_prompt_kv=False),
        _hot_prompt_slots=[], _hot_kv_persist=None, _prompt_kv_store=None,
        last_kv=State(), _h_window=State(), _h_last=State(), _provisional=State(),
        _serial_kda_endpoints=State(), _serial_qwen4_endpoints=State(),
        _serial_kda_factors=State(),
    )

    def consume(name, _position=None):
        events.append(name)
        setattr(target, name, None)

    target.consume_serial_kda_endpoint = lambda position: consume(
        "_serial_kda_endpoints", position)
    target.consume_serial_qwen4_endpoint = lambda position: consume(
        "_serial_qwen4_endpoints", position)
    target.consume_serial_kda_factors = lambda: consume("_serial_kda_factors")

    def release():
        events.append("request-release")
        api.namespace["release_generation_state"](target)

    target.release_request_state = release
    return target


@pytest.mark.parametrize("value", [None, "0", "", "true", "invalid"])
def test_cleanup_is_strictly_opt_in(source_api, monkeypatch, value):
    target = _target(source_api)
    if value is not None:
        monkeypatch.setenv(FLAG, value)
    endpoint = target.last_kv
    source_api.cleanup(source_api.Wrapper(target))
    assert target.last_kv is endpoint


def test_cleanup_drops_all_endpoint_and_interrupted_capture_owners(
        source_api, monkeypatch, capsys):
    monkeypatch.setenv(FLAG, "1")
    events = []
    target = _target(source_api, events)
    names = ("last_kv", "_h_window", "_h_last", "_provisional",
             "_serial_kda_endpoints", "_serial_kda_factors", "_serial_qwen4_endpoints")
    refs = [weakref.ref(getattr(target, name)) for name in names]
    source_api.cleanup(source_api.Wrapper(target))
    assert all(getattr(target, name) is None for name in names)
    assert all(reference() is None for reference in refs)
    assert events == ["_serial_kda_endpoints", "_serial_kda_factors",
                      "_serial_qwen4_endpoints", "request-release", "endpoint-release"]
    output = capsys.readouterr().out
    assert "endpoint_logical_bytes=1031050560" in output
    assert "physical_reclamation=unmeasured" in output
    # Repeating idle disposal cannot release the endpoint twice.
    source_api.cleanup(source_api.Wrapper(target))
    assert events.count("endpoint-release") == 1


@pytest.mark.parametrize("owner", ["hot-policy", "hot-slot", "unknown-slots",
                                  "unknown-policy", "hot-persistence", "prompt-store",
                                  "_vision_prompt_cache", "_glm53_vision_prompt_cache",
                                  "_vision_embedding_cache", "_glm53_vision_embedding_cache"])
def test_cleanup_preserves_every_reuse_owner(source_api, monkeypatch, owner):
    monkeypatch.setenv(FLAG, "1")
    events = []
    target = _target(source_api, events)
    if owner == "hot-policy":
        target.rc.hot_prompt_kv = True
    elif owner == "hot-slot":
        target._hot_prompt_slots = [SimpleNamespace(kv=target.last_kv)]
    elif owner == "unknown-slots":
        del target._hot_prompt_slots
    elif owner == "unknown-policy":
        target.rc = SimpleNamespace()
    elif owner == "hot-persistence":
        target._hot_kv_persist = object()
    elif owner == "prompt-store":
        target._prompt_kv_store = object()
    else:
        setattr(target, owner, {"reusable": object()})
    endpoint = target.last_kv
    source_api.cleanup(source_api.Wrapper(target))
    assert target.last_kv is endpoint
    assert not events


def test_cleanup_releases_one_endpoint_aliased_by_hidden_and_provisional_owners(
        source_api, monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    events = []
    target = _target(source_api, events)
    target._h_window = target._h_last = target._provisional = target.last_kv
    ref = weakref.ref(target.last_kv)
    source_api.cleanup(source_api.Wrapper(target))
    assert ref() is None
    assert events.count("endpoint-release") == 1


def test_empty_vision_embedding_caches_do_not_block_text_cleanup(source_api, monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    target = _target(source_api)
    target._vision_embedding_cache = {}
    target._glm53_vision_embedding_cache = {}
    source_api.cleanup(source_api.Wrapper(target))
    assert target.last_kv is None


@pytest.mark.parametrize("kind", ["direct", "subclass", "proxy", "other-model", "absent"])
def test_cleanup_requires_the_concrete_qwen4_mtp_wrapper(source_api, monkeypatch, kind):
    monkeypatch.setenv(FLAG, "1")
    events = []
    target = _target(source_api, events)
    if kind == "direct":
        engine = target
    elif kind == "subclass":
        engine = type("Subclass", (source_api.Wrapper,), {})(target)
    elif kind == "proxy":
        engine = SimpleNamespace(target=target, cfg=target.cfg)
    elif kind == "other-model":
        target.cfg.model_type = "glm5_next"
        engine = source_api.Wrapper(target)
    else:
        engine = None
    source_api.cleanup(engine)
    assert target.last_kv is not None
    assert not events


@pytest.mark.parametrize("broken_log", [False, True])
def test_cleanup_failure_cannot_fail_a_sent_response(
        source_api, monkeypatch, capsys, broken_log):
    monkeypatch.setenv(FLAG, "1")
    target = _target(source_api)

    def fail():
        raise RuntimeError("cleanup failed")

    target.release_request_state = fail
    if broken_log:
        source_api.namespace["print"] = lambda *args, **kwargs: (
            (_ for _ in ()).throw(BrokenPipeError("log sink closed")))
    source_api.cleanup(source_api.Wrapper(target))
    if not broken_log:
        assert "cleanup_failed error_type=RuntimeError" in capsys.readouterr().out


def test_pre_engine_validation_finally_does_not_dispose_another_idle_engine(
        source_api, monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    target = _target(source_api)
    endpoint = target.last_kv
    cleanups = []

    def cleanup(engine):
        cleanups.append(engine)
        source_api.cleanup(engine)

    source_api.namespace["_release_qwen4_idle_request_state"] = cleanup
    handler = SimpleNamespace(
        _parsed_request=(b"{}", {"model": 42}, 2),
        _route=lambda: "/responses",
        _json=lambda status, body: (status, body),
    )
    status, body = source_api.namespace["_do_post_locked"](handler)
    assert status == 400
    assert body == {"error": "model must be a non-empty string"}
    assert cleanups == [None]
    assert target.last_kv is endpoint


@pytest.mark.parametrize("route,stream", [("/responses", False), ("/responses", True),
                                          ("/messages", False), ("/messages", True)])
@pytest.mark.parametrize("outcome", ["success", "disconnect", "validation", "failure"])
def test_real_handler_finally_runs_after_protocol_consumers_before_lock_release(
        source_api, monkeypatch, route, stream, outcome):
    monkeypatch.setenv(FLAG, "1")
    events = []
    target = _target(source_api, events)
    engine = source_api.Wrapper(target)
    ns = source_api.namespace
    lock = SimpleNamespace(held=False)

    def acquire(**kwargs):
        assert not lock.held
        lock.held = True
        events.append("lock-acquire")

    def release_lock():
        assert lock.held
        lock.held = False
        events.append("lock-release")

    original_release = target.release_request_state

    def release_state():
        assert lock.held
        original_release()

    target.release_request_state = release_state
    ns.update({
        "INFER_LOCK": SimpleNamespace(acquire=acquire, release=release_lock),
        "_tool_request_controls": lambda *args: ([], "auto", False),
        "_validate_generation_controls": lambda *args: SimpleNamespace(),
        "_structured_output_request": lambda *args: None,
        "_request_reasoning_controls": lambda *args: ("none", False, False, None),
        "split_model_mode": lambda value: (value, "lossless"),
        "_positive_token_limit": lambda value, _field: int(value),
        "_resolve": lambda model_id: ROOT,
        "MANAGER": SimpleNamespace(get=lambda *args, **kwargs: engine),
        "ModelDownloading": type("ModelDownloading", (Exception,), {}),
        "ModelDownloadFailed": type("ModelDownloadFailed", (Exception,), {}),
    })
    req = {"model": "qwen", "input": "hello", "messages": [],
           "max_tokens": 32, "stream": stream}

    def protocol(*args):
        assert lock.held
        # Stand in for multiple hidden-gateway generations and their final
        # cache/trace/metadata consumers, all of which still need the endpoint.
        for event in ("generation-1", "generation-2", "cache", "telemetry", "response"):
            assert target.last_kv is not None
            events.append(event)
        if outcome == "disconnect":
            raise BrokenPipeError("peer closed")
        if outcome == "validation":
            raise ns["RequestValidationError"]("invalid response control")
        if outcome == "failure":
            raise RuntimeError("generation failed")
        return "sent"

    def json_response(code, body):
        assert target.last_kv is not None
        events.append(f"json-{code}")
        return code

    handler_type = type("Handler", (), {
        "do_POST": ns["do_POST"], "_do_post_locked": ns["_do_post_locked"],
    })
    handler = handler_type()
    handler.headers = {}
    handler.connection = SimpleNamespace(gettimeout=lambda: None,
                                         settimeout=lambda value: None)
    handler._read_json_request = lambda: (b"{}", req, 2)
    handler._preflight_nested_request = lambda parsed: ([], [], [])
    handler._route = lambda: route
    handler._do_responses = protocol
    handler._do_anthropic_messages = protocol
    handler._json = json_response
    result = handler.do_POST()
    assert result == {"success": "sent", "disconnect": None,
                      "validation": 400, "failure": None}[outcome]
    assert target.last_kv is None
    assert events.index("response") < events.index("request-release")
    assert events.index("request-release") < events.index("lock-release")
    assert events.count("request-release") == 1
    assert not lock.held
