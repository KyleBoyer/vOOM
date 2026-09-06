"""Pure production ownership/admission/call-site tests; no model imports."""

from pathlib import Path
from types import SimpleNamespace as NS
import textwrap

import pytest

from runtime.qwen4_retained_history import (
    compact_retained_history, retained_qsa_backing_allowance)


class Array:
    def __init__(self, shape, dtype="bf16", payload=b"bits"):
        self.shape, self.dtype, self.payload = shape, dtype, payload
        self.nbytes = 2
        for size in shape:
            self.nbytes *= size


class KV:
    pass


def setup():
    cfg = NS(model_type="qwen4_exp", num_hidden_layers=4,
             layer_types=("linear_attention",) * 3 + ("full_attention",),
             qwen4_ple_layers=(1,), linear_num_key_heads=1,
             linear_key_head_dim=1, linear_num_value_heads=2,
             linear_value_head_dim=2, linear_conv_kernel_dim=4,
             qwen4_ple_conv_kernel_size=2, qwen4_ngram_size=3,
             qwen4_hc_count=2, hidden_size=4,
             qwen4_indexer_n_heads=4, qwen4_indexer_head_dim=2)
    endpoint = KV()
    endpoint.offset, endpoint.compressed_mla = 4, False
    endpoint.keys = [None] * 3 + [Array((1, 1, 4, 2))]
    endpoint.values = [None] * 3 + [Array((1, 1, 4, 2))]
    endpoint._starts, endpoint._windows = [0] * 4, [None] * 4
    endpoint.kda_cache = NS(
        _state=[object(), object(), object(), None],
        _conv=[(Array((1, 3, 6)),) for _ in range(3)] + [None],
        spill_enabled=False, _spill_meta={}, _factor_capture=None)
    endpoint.qwen4_cache = NS(
        qsa_keys=[None] * 3 + [Array((1, 4, 2))],
        qsa_positions=[None] * 3 + [Array((1, 4), "int32")],
        qsa_pooled_keys=[None] * 4, qsa_pool_cache_enabled=False,
        ple_conv=[None, Array((1, 3, 8)), None, None],
        ple_context=[None, (7, 8), None, None], ple_lengths=[0, 4, 0, 0])
    fork = KV()
    for name, value in vars(endpoint).items():
        if isinstance(value, list): value = list(value)
        if isinstance(value, NS):
            value = NS(**{key: list(item) if isinstance(item, list) else item
                          for key, item in vars(value).items()})
        setattr(fork, name, value)
    return cfg, endpoint, fork


def run(cfg, endpoint, fork, *, copier=None, reserve=None):
    return compact_retained_history(
        fork, endpoint, cfg, prefix_tokens=4, kv_type=KV, expected_dtype="bf16",
        copy_bits=copier or (lambda a: Array(a.shape, a.dtype, a.payload)),
        reserve=reserve or (lambda size: None), active_bytes=lambda: 100)


def test_only_histories_change_and_original_keeps_old_arrays():
    cfg, endpoint, fork = setup()
    old_ple = endpoint.qwen4_cache.ple_conv[1]
    original_histories = list(endpoint.kda_cache._conv)
    reserved = []
    report = run(cfg, endpoint, fork, reserve=reserved.append)
    assert report["qwen4_retained_conv_compact_arrays"] == 4
    assert report["qwen4_retained_conv_compact_bytes"] == 3 * 36 + 48
    assert reserved == [2 * (3 * 36 + 48)]
    assert endpoint.qwen4_cache.ple_conv[1] is old_ple
    assert fork.qwen4_cache.ple_conv[1] is not old_ple
    for layer in range(3):
        assert endpoint.kda_cache._conv[layer] is original_histories[layer]
        assert fork.kda_cache._conv[layer][0] is not original_histories[layer][0]
        assert fork.kda_cache._conv[layer][0].payload == original_histories[layer][0].payload
        assert fork.kda_cache._state[layer] is endpoint.kda_cache._state[layer]
    assert fork.keys[3] is endpoint.keys[3]
    assert fork.qwen4_cache.qsa_keys[3] is endpoint.qwen4_cache.qsa_keys[3]


@pytest.mark.parametrize("failure", ["reserve", "copy", "alias_copy", "shape_copy"])
def test_failure_is_atomic_and_does_not_publish_partial_histories(failure):
    cfg, endpoint, fork = setup()
    old = list(fork.kda_cache._conv), list(fork.qwen4_cache.ple_conv)
    copies = []

    def copy(a):
        copies.append(a)
        if failure == "copy" and len(copies) == 3: raise MemoryError("copy failed")
        if failure == "alias_copy": return a
        if failure == "shape_copy": return Array((1,))
        return Array(a.shape, a.dtype, a.payload)

    def reserve(size):
        if failure == "reserve": raise MemoryError("not enough scratch")

    with pytest.raises((MemoryError, ValueError)):
        run(cfg, endpoint, fork, copier=copy, reserve=reserve)
    assert fork.kda_cache._conv == old[0]
    assert fork.qwen4_cache.ple_conv == old[1]
    if failure == "reserve": assert not copies


@pytest.mark.parametrize("change", [
    "same_kv", "wrong_prefix", "wrong_model", "same_kda", "same_aux",
    "shared_conv_list", "shared_state_list", "shared_qsa_list", "missing_layer",
    "spill", "factor", "mutable_history", "wrong_history_dtype",
    "wrong_history_shape", "missing_history", "new_not_shared_history",
    "wrong_ple_length", "wrong_context", "missing_ple", "wrong_full_length",
    "wrong_start", "wrong_pool_policy", "compressed",
    "missing_qsa", "wrong_qsa_length", "unexpected_kda_state",
    "unexpected_linear_kv", "unexpected_linear_qsa",
])
def test_invalid_forks_fail_before_reservation_or_copy(change):
    cfg, endpoint, fork = setup()
    if change == "same_kv": fork = endpoint
    elif change == "wrong_prefix": fork.offset = 3
    elif change == "wrong_model": cfg.model_type = "qwen3_5"
    elif change == "same_kda": fork.kda_cache = endpoint.kda_cache
    elif change == "same_aux": fork.qwen4_cache = endpoint.qwen4_cache
    elif change == "shared_conv_list": fork.kda_cache._conv = endpoint.kda_cache._conv
    elif change == "shared_state_list": fork.kda_cache._state = endpoint.kda_cache._state
    elif change == "shared_qsa_list": fork.qwen4_cache.qsa_keys = endpoint.qwen4_cache.qsa_keys
    elif change == "missing_layer": fork.kda_cache._conv.pop()
    elif change == "spill": fork.kda_cache.spill_enabled = True
    elif change == "factor": fork.kda_cache._factor_capture = []
    elif change == "mutable_history": fork.kda_cache._conv[0] = list(fork.kda_cache._conv[0])
    elif change == "wrong_history_dtype": fork.kda_cache._conv[0][0].dtype = "fp32"
    elif change == "wrong_history_shape": fork.kda_cache._conv[0][0].shape = (1, 2, 6)
    elif change == "missing_history": fork.kda_cache._conv[0] = None
    elif change == "new_not_shared_history": fork.kda_cache._conv[0] = (Array((1, 3, 6)),)
    elif change == "wrong_ple_length": fork.qwen4_cache.ple_lengths[1] = 3
    elif change == "wrong_context": fork.qwen4_cache.ple_context[1] = (8,)
    elif change == "missing_ple": fork.qwen4_cache.ple_conv[1] = None
    elif change == "wrong_full_length": fork.keys[3].shape = (1, 1, 3, 2)
    elif change == "wrong_start": fork._starts[3] = 1
    elif change == "wrong_pool_policy": fork.qwen4_cache.qsa_pool_cache_enabled = True
    elif change == "compressed": fork.compressed_mla = True
    elif change == "missing_qsa":
        fork.qwen4_cache.qsa_keys[3] = endpoint.qwen4_cache.qsa_keys[3] = None
    elif change == "wrong_qsa_length": fork.qwen4_cache.qsa_positions[3].shape = (1, 3)
    elif change == "unexpected_kda_state":
        fork.kda_cache._state[3] = endpoint.kda_cache._state[3] = object()
    elif change == "unexpected_linear_kv":
        fork.keys[0] = endpoint.keys[0] = Array((1, 1, 4, 2))
    elif change == "unexpected_linear_qsa":
        fork.qwen4_cache.qsa_keys[0] = endpoint.qwen4_cache.qsa_keys[0] = Array((1, 4, 2))
    with pytest.raises(ValueError):
        run(cfg, endpoint, fork, reserve=lambda size: pytest.fail("reserved before validation"))


def test_qsa_allowance_covers_query_head_backing_for_only_one_tile():
    cfg, _, _ = setup()
    assert retained_qsa_backing_allowance(cfg, prefix_tokens=4, tile_tokens=4) == 64
    assert retained_qsa_backing_allowance(cfg, prefix_tokens=8, tile_tokens=4) == 64


def call_site(monkeypatch, **changes):
    import runtime.qwen4_retained_history as helper
    source = (Path(__file__).resolve().parents[1] / "runtime/engine.py").read_text()
    start = source.index("            if (boundary_fork_kv is not None and aligned_qwen4_retention")
    end = source.index("            ckpt =", start)
    calls = []
    monkeypatch.setattr(helper, "compact_retained_history", lambda *a, **kw: calls.append((a, kw)) or {"compact_called": 1})
    cfg, endpoint, fork = setup()
    state = dict(boundary_fork_kv=fork, kv=endpoint, aligned_qwen4_retention=True,
                 hot_eligible=True, force_adaptive_paged=False, boundary_fork_tokens=4,
                 stable_boundary_positions=4, deferred_qwen_boundary_tokens=0,
                 prompt_state_approximate=False, path_stats={}, KVCache=KV,
                 mx=NS(bfloat16="bf16", get_active_memory=lambda: 100),
                 self=NS(rc=NS(qwen4_compact_retained_conv=True), cfg=cfg,
                         governor=None, _hot_kv_persist=None, last_kv=None),
                 __package__="runtime")
    state.update(changes)
    exec(compile(textwrap.dedent(source[start:end]), "engine-compact-callsite", "exec"), state)
    return state, calls


@pytest.mark.parametrize("changes", [
    {}, {"aligned_qwen4_retention": False}, {"hot_eligible": False},
    {"force_adaptive_paged": True}, {"boundary_fork_kv": None},
])
def test_actual_call_site_guards_and_single_invocation(monkeypatch, changes):
    state, calls = call_site(monkeypatch, **changes)
    assert len(calls) == (0 if changes else 1)
    assert bool(state["path_stats"]) is (not bool(changes))


@pytest.mark.parametrize("changes", [
    {"prompt_state_approximate": True}, {"deferred_qwen_boundary_tokens": 4},
    {"stable_boundary_positions": 3},
])
def test_actual_call_site_refuses_incomplete_boundary_ownership(monkeypatch, changes):
    with pytest.raises(ValueError): call_site(monkeypatch, **changes)


def test_runtime_default_identity_and_placement_are_explicit():
    root = Path(__file__).resolve().parents[1]
    engine = (root / "runtime/engine.py").read_text()
    server = (root / "runtime/server.py").read_text()
    assert "qwen4_compact_retained_conv: bool = False" in engine
    assert 'qwen4_compact_retained_conv=run.get("qwen4_compact_retained_conv", False)' in engine
    assert '("VMODEL_QWEN4_COMPACT_RETAINED_CONV", "0")' in server
    assert 'rc.qwen4_compact_retained_conv = qwen4_request_identity[37] == "1"' in server
    hook = engine.index("            if (boundary_fork_kv is not None and aligned_qwen4_retention")
    assert engine.index("boundary_fork_kv = matched_boundary_fork") < hook
    assert engine.rindex("boundary_fork_kv = fork_hybrid_kv_endpoint(kv)", 0, hook) < hook
    assert hook < engine.index("            ckpt =", hook)


@pytest.mark.parametrize("enabled,eligible,expected", [(True, True, 164),
                                                       (False, True, 0),
                                                       (True, False, 0)])
def test_actual_admission_adds_full_retained_endpoint_not_suffix_delta(enabled, eligible, expected):
    source = (Path(__file__).resolve().parents[1] / "runtime/engine.py").read_text()
    start = source.index("        if (aligned_qwen4_retention and hot_eligible")
    end = source.index("        admission_done = False", start)
    cfg, _, _ = setup()
    state = dict(aligned_qwen4_retention=True, hot_eligible=eligible,
                 stable_boundary_positions=4, required_total_kv_bytes=500,
                 path_stats={}, __package__="runtime",
                 self=NS(cfg=cfg, rc=NS(qwen4_compact_retained_conv=enabled,
                                       prefill_chunk_size=4),
                         _project_dense_text_kv_bytes=lambda positions: 100))
    exec(compile(textwrap.dedent(source[start:end]), "engine-retained-admission", "exec"), state)
    assert state["required_total_kv_bytes"] == 500 + expected
    if expected:
        assert state["path_stats"]["qwen4_retained_projected_logical_bytes"] == 100
        assert state["path_stats"]["qwen4_retained_qsa_backing_allowance_bytes"] == 64


def test_call_site_default_off_does_not_copy(monkeypatch):
    cfg, _, _ = setup()
    _, calls = call_site(monkeypatch, self=NS(
        cfg=cfg, rc=NS(qwen4_compact_retained_conv=False), governor=None,
        _hot_kv_persist=None, last_kv=None))
    assert not calls
