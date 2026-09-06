"""Execute actual AST-selected engine hooks with pure fake dependencies."""

import ast
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest

from runtime import qwen4_prefix_capture as helper


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "runtime/engine.py").read_text()
TREE = ast.parse(SOURCE)


def function(name):
    return next(node for node in ast.walk(TREE)
                if isinstance(node, ast.FunctionDef) and node.name == name)


def execute(nodes, state):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "engine-hook", "exec"), state)
    return state


def assigned(name):
    return next(node for node in ast.walk(function("generate"))
                if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in node.targets))


class KV:
    pass


def eligibility(**changes):
    rc = NS(qwen4_fused_aligned_prefix=True, qwen4_compact_retained_conv=True,
            prefill_chunk_size=1024, prefill_last_token_separate=False,
            prefill_checkpoint_every=0, paged_kv_persist=False)
    kv = KV()
    kv.kda_cache = NS(spill_enabled=False, _factor_capture=None)
    kv.qwen4_cache = NS(qsa_pool_cache_enabled=False)
    state = dict(boundary_layer_stationary=True, aligned_qwen4_retention=True,
                 hot_eligible=True, pos=0, matched=0, kv=kv, KVCache=KV,
                 stable_boundary=1024, boundary_chunk=1024, kv_store=None,
                 prompt_state_approximate=False,
                 self=NS(rc=rc, cfg=NS(model_type="qwen4_exp"), _hot_kv_persist=None))
    for name, value in changes.items():
        if name.startswith("rc_"):
            setattr(rc, name[3:], value)
        elif name == "model_type":
            state["self"].cfg.model_type = value
        elif name == "persist_owner":
            state["self"]._hot_kv_persist = value
        elif name == "pool":
            kv.qwen4_cache.qsa_pool_cache_enabled = value
        elif name == "spill":
            kv.kda_cache.spill_enabled = value
        elif name == "factor":
            kv.kda_cache._factor_capture = value
        else:
            state[name] = value
    execute([assigned("fuse_qwen4_prefix")], state)
    return state


def test_actual_cold_predicate_and_deferral_do_not_fork_or_advance_empty_source():
    state = eligibility()
    assert state["fuse_qwen4_prefix"]
    state.update(boundary_fork_kv=None, boundary_fork_tokens=0, deferred_qwen4_prefix_tokens=0)
    branch = next(node for node in ast.walk(function("generate"))
                  if isinstance(node, ast.If) and ast.unparse(node.test) == "fuse_qwen4_prefix")
    execute(branch.body, state)
    assert state["deferred_qwen4_prefix_tokens"] == 1024
    assert state["pos"] == state["boundary_fork_tokens"] == 0
    assert state["boundary_fork_kv"] is None
    bookkeeping = next(node for node in ast.walk(function("generate"))
                       if isinstance(node, ast.If) and ast.unparse(node.test) == (
                           "not fuse_qwen_boundary_scaffold and (not fuse_qwen4_prefix)"))
    assert not eval(compile(ast.Expression(bookkeeping.test), "bookkeeping", "eval"),
                    state | {"fuse_qwen_boundary_scaffold": False})


@pytest.mark.parametrize("changes", [
    dict(boundary_layer_stationary=False), dict(aligned_qwen4_retention=False),
    dict(hot_eligible=False), dict(rc_qwen4_fused_aligned_prefix=False),
    dict(rc_qwen4_compact_retained_conv=False), dict(pos=1024, matched=1024),
    dict(pos=0, matched=1024), dict(kv=object()), dict(model_type="qwen3_5"),
    dict(rc_prefill_chunk_size=0), dict(stable_boundary=1606),
    dict(rc_prefill_last_token_separate=True), dict(rc_prefill_checkpoint_every=1024),
    dict(kv_store=object()), dict(rc_paged_kv_persist=True), dict(persist_owner=object()),
    dict(prompt_state_approximate=True), dict(pool=True), dict(spill=True), dict(factor=[]),
])
def test_actual_predicate_keeps_unsupported_modes_on_existing_path(changes):
    assert not eligibility(**changes)["fuse_qwen4_prefix"]


@pytest.mark.parametrize("deferred,eligible,raises", [(0, False, False), (1024, True, False), (1024, False, True)])
def test_lost_stationary_eligibility_fails_before_fallback(deferred, eligible, raises):
    node = next(node for node in ast.walk(function("generate")) if isinstance(node, ast.If)
                and ast.unparse(node.test) == "deferred_qwen4_prefix_tokens and (not layer_stationary_eligible)")
    state = dict(deferred_qwen4_prefix_tokens=deferred, layer_stationary_eligible=eligible)
    if raises:
        with pytest.raises(RuntimeError): execute([node], state)
    else:
        execute([node], state)


@pytest.mark.parametrize("captured", [True, False])
def test_actual_tile_hook_runs_after_attention_eval_and_before_route_or_overwrite(captured):
    sweep = function("_layer_stationary_qwen4_sweep")
    hooks = [node for node in ast.walk(sweep) if isinstance(node, ast.If)
             and ast.unparse(node.test) == "prefix_capture is not None"]
    assert len(hooks) == 2
    begin, tile = sorted(hooks, key=lambda node: node.lineno)
    text = ast.get_source_segment(SOURCE, sweep)
    assert text.index("prefix_capture.begin_sweep") < text.index("self.cache.get")
    assert text.index("mx.eval(post_attention)") < text.index("prefix_capture.observe_tile")
    assert text.index("prefix_capture.observe_tile") < text.index("hidden_host[:, pos:end] =")
    events = []
    class Capture:
        def observe_tile(self, source, **kw):
            events.append(("observe", source, kw))
            return captured
    source = object()
    execute([tile], dict(prefix_capture=Capture(), kv=source, i=4, pos=0, end=1024,
                         spool_peak_host_bytes=123,
                         note_spool=lambda *a, **kw: events.append(("sample", a, kw))))
    assert events[0] == ("observe", source, dict(layer=4, start=0, end=1024))
    assert len(events) == (2 if captured else 1)
    if captured: assert events[1][1] == ("retained_prefix",)


@pytest.mark.parametrize("failure", [None, "sweep", "eval", "cap", "finish", "interrupt"])
def test_actual_wrapper_finishes_after_hidden_and_always_aborts(monkeypatch, failure):
    events = []
    class Capture:
        def __init__(self, *args, **kwargs):
            events.append("construct")
            kwargs["reserve"](12)
        def finish(self, source):
            events.append("finish")
            if failure == "finish": raise ValueError("failed finish")
            return "prefix", {"capture": 1}
        def abort(self): events.append("abort")
    monkeypatch.setattr(helper, "AlignedPrefixCapture", Capture)
    for name, attribute in (("runtime.kda_state", "KDAStateCache"),
                            ("runtime.qwen4_exp_state", "Qwen4ExpStateCache")):
        module = ModuleType(name)
        setattr(module, attribute, type(attribute, (), {}))
        monkeypatch.setitem(sys.modules, name, module)
    def sweep(*args, **kwargs):
        events.append("sweep")
        assert isinstance(kwargs["prefix_capture"], Capture)
        if failure == "sweep": raise MemoryError("weights")
        if failure == "interrupt": raise KeyboardInterrupt()
        return "hidden"
    def evaluate(value):
        events.append("eval")
        if failure == "eval": raise RuntimeError("restore")
    mx = NS(bfloat16="bf16", float32="fp32", int32="i32", eval=evaluate,
            get_active_memory=lambda: 2_000_000 if failure == "cap" else 0,
            get_peak_memory=lambda: 0)
    state = dict(mx=mx, KVCache=KV, __package__="runtime")
    execute([function("_layer_stationary_qwen4_capture_sweep")], state)
    engine = NS(cfg=NS(num_hidden_layers=4), rc=NS(metal_limit_mb=1),
                governor=NS(reserve=lambda size, **kw: events.append(("reserve", size, kw))),
                _layer_stationary_qwen4_sweep=sweep, _note_true_peak=lambda: events.append("peak"))
    call = state["_layer_stationary_qwen4_capture_sweep"]
    if failure:
        with pytest.raises((MemoryError, ValueError, RuntimeError, KeyboardInterrupt)):
            call(engine, NS(shape=(1, 7, 8)), object(), offset=0, tile_width=4, prefix_tokens=4)
    else:
        assert call(engine, NS(shape=(1, 7, 8)), object(), offset=0, tile_width=4, prefix_tokens=4) == (
            "hidden", "prefix", {"capture": 1})
        assert events[-4:] == ["eval", "peak", "finish", "abort"]
    assert events[-1] == "abort"
    if failure in ("sweep", "eval", "cap", "interrupt"):
        assert "finish" not in events


def dispatch(**changes):
    node = next(node for node in ast.walk(function("generate")) if isinstance(node, ast.If)
                and ast.unparse(node.test) == "deferred_qwen4_prefix_tokens")
    calls = []
    state = dict(deferred_qwen4_prefix_tokens=1024, pos=0, stop_before=1611,
                 tokens=[0] * 1611, chunk=1024, boundary_fork_kv=None,
                 boundary_fork_tokens=0, path_stats={}, xc="input", kv="source",
                 on_progress=None, self=NS(rc=NS(prefill_chunk_size=1024),
                 _layer_stationary_qwen4_capture_sweep=lambda *a, **kw: (
                     calls.append("capture") or ("output", "prefix", {"layers": 48})),
                 _layer_stationary_qwen4_sweep=lambda *a, **kw: calls.append("ordinary") or "output"))
    state.update(changes)
    execute([node], state)
    return state, calls


def test_actual_full_dispatch_publishes_only_completed_capture():
    state, calls = dispatch()
    assert calls == ["capture"] and state["xc"] == "output"
    assert state["boundary_fork_kv"] == "prefix" and state["boundary_fork_tokens"] == 1024
    assert state["path_stats"] == {"layers": 48, "hot_prompt_boundary_fork_tokens": 1024}
    assert "captured_prefix" not in state


def test_actual_matched_repeat_preserves_existing_ordinary_suffix_dispatch():
    state, calls = dispatch(deferred_qwen4_prefix_tokens=0, pos=1024,
                            boundary_fork_kv="existing", boundary_fork_tokens=1024)
    assert calls == ["ordinary"] and state["boundary_fork_kv"] == "existing"
    assert not state["path_stats"]


@pytest.mark.parametrize("changes", [dict(pos=1), dict(stop_before=1610), dict(chunk=512),
                                     dict(boundary_fork_kv="premature")])
def test_full_dispatch_refuses_contract_drift(changes):
    with pytest.raises(ValueError): dispatch(**changes)


def test_explicit_runtime_yaml_identity_and_typed_telemetry():
    server = (ROOT / "runtime/server.py").read_text()
    assert "qwen4_fused_aligned_prefix: bool = False" in SOURCE
    assert 'qwen4_fused_aligned_prefix=run.get("qwen4_fused_aligned_prefix", False)' in SOURCE
    assert '("VMODEL_QWEN4_FUSED_ALIGNED_PREFIX", "0")' in server
    assert 'rc.qwen4_fused_aligned_prefix = qwen4_request_identity[38] == "1"' in server
    assert 'if qwen4_request_identity[38] not in ("0", "1"):' in server
    ints = server[server.index("    optional_integer_fields = ("):server.index("    optional_float_fields = (")]
    floats = server[server.index("    optional_float_fields = ("):]
    for suffix in ("layers", "tokens", "arrays_copied", "bytes_copied", "scratch_peak_bytes"):
        assert f'"qwen4_fused_prefix_capture_{suffix}"' in ints
    assert '"qwen4_fused_prefix_capture_seconds"' in floats
