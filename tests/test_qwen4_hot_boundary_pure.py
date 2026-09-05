"""Complete-tile cache eligibility and actual call-site tests, without MLX."""

import ast
from pathlib import Path
import textwrap
from types import SimpleNamespace

import pytest

from runtime.qwen4_hot_boundary import plan_tile_retention, slot_matches_tile_retention


ROOT = Path(__file__).resolve().parents[1]


def config(**changes):
    return SimpleNamespace(**{
        "prefill_chunk_size": 1024, "hot_prompt_kv_chunk_size": 1024,
        "hot_prompt_kv_min_tokens": 16, "hot_prompt_kv": True,
        "layer_stationary_prefill": True, "qwen4_hot_kv_tile_aligned": True,
        "max_kv_mb": 0, "paged_kv_persist": False, **changes})


@pytest.mark.parametrize("boundary,expected", [
    (0, 0), (1023, 0), (1024, 1024), (1025, 1024), (1606, 1024),
    (2047, 1024), (2048, 2048), (32766, 31744),
])
def test_rounds_down_content_blind_and_never_keeps_partial_tile(boundary, expected):
    result = plan_tile_retention(config(), requested_boundary=boundary,
                                 prompt_tokens=boundary + 5)
    assert result["effective"] == expected
    assert result["eligible"] is bool(expected)
    assert result["requested"] == boundary


@pytest.mark.parametrize("changes", [
    {"prefill_chunk_size": 0}, {"prefill_chunk_size": True},
    {"hot_prompt_kv_chunk_size": 512}, {"hot_prompt_kv": False},
    {"layer_stationary_prefill": False}, {"hot_prompt_kv_min_tokens": 2048},
    *[{name: True} for name in (
        "adaptive_chunk_size", "max_kv_mb", "adaptive_kv_spill_mb",
        "paged_kv_persist", "hot_prompt_kv_persist_dir",
        "prefill_checkpoint_every", "prefill_last_token_separate",
        "qwen4_global_expert_rows", "qwen4_sparse_expert_batch_rows")],
])
def test_unsupported_config_disables_cache_not_only_hint(changes):
    result = plan_tile_retention(config(**changes), requested_boundary=1606,
                                 prompt_tokens=1611)
    assert result["eligible"] is False and result["effective"] == 0


@pytest.mark.parametrize("boundary,length", [
    (-1, 1611), (1611, 1611), (1612, 1611), (True, 1611),
    (1024.0, 1611), (1024, True), (1024, 0),
])
def test_invalid_prefix_cannot_enter_raw_endpoint_fallback(boundary, length):
    result = plan_tile_retention(config(), requested_boundary=boundary,
                                 prompt_tokens=length)
    assert result["eligible"] is False and result["effective"] == 0


def test_forced_paging_disables_candidate():
    result = plan_tile_retention(config(), requested_boundary=1606,
                                 prompt_tokens=1611, force_paged=True)
    assert result["eligible"] is False


@pytest.mark.parametrize("changes,eligible", [
    ({}, True), ({"tokens": tuple(range(2048))}, False),
    ({"tokens": tuple(range(1023))}, False), ({"tokens": ()}, False),
    ({"qwen4_retention_tile": 0}, False), ({"chunk_size": 512}, False),
    ({"logits": object()}, False), ({"prompt_logits": object()}, False),
    ({"exact_hidden": object()}, False), ({"approximate": True}, False),
])
def test_only_compatible_forks_at_or_before_current_boundary_reuse(changes, eligible):
    slot = SimpleNamespace(**{
        "tokens": tuple(range(1024)), "chunk_size": 1024,
        "qwen4_retention_tile": 1024, "logits": None, "prompt_logits": None,
        "exact_hidden": None, "approximate": False, **changes})
    assert slot_matches_tile_retention(slot, effective_boundary=1024,
                                       tile=1024) is eligible


def serving_plan(*, boundary=1606, enabled=True, model="qwen4_exp", disable=False):
    source = (ROOT / "runtime/engine.py").read_text()
    start = source.index("        recurrent_exact_only = self.cfg.model_type in (")
    end = source.index("        resident_prompt_kv_bytes =", start)
    prompt = SimpleNamespace(stable_boundary_tokens=boundary,
                             disable_hot_prompt_kv=disable)
    namespace = {
        "__package__": "runtime", "self": SimpleNamespace(
            cfg=SimpleNamespace(model_type=model),
            rc=config(qwen4_hot_kv_tile_aligned=enabled)),
        "tokens": tuple(range(1611)), "prompt": prompt,
        "force_adaptive_paged": False, "path_stats": {},
    }
    exec(compile(textwrap.dedent(source[start:end]), str(ROOT / "runtime/engine.py"),
                 "exec"), namespace)
    assert prompt.stable_boundary_tokens == boundary
    assert namespace["tokens"] == tuple(range(1611))
    return namespace


@pytest.mark.parametrize("boundary,expected,eligible", [
    (0, 0, False), (1023, 0, False), (1606, 1024, True),
])
def test_real_engine_call_site_applies_effective_boundary_and_eligibility(
        boundary, expected, eligible):
    state = serving_plan(boundary=boundary)
    assert state["stable_boundary_positions"] == expected
    assert state["hot_eligible"] is eligible
    assert state["path_stats"]["qwen4_hot_boundary_requested"] == boundary


@pytest.mark.parametrize("options", [{"enabled": False}, {"model": "qwen3_5"}])
def test_default_off_and_other_model_paths_unchanged(options):
    state = serving_plan(**options)
    assert state["stable_boundary_positions"] == 1606
    assert state["hot_eligible"] is True
    assert state["path_stats"] == {}


def test_alignment_does_not_reenable_explicitly_disabled_request():
    state = serving_plan(disable=True)
    assert state["hot_eligible"] is False
    assert state["path_stats"]["qwen4_hot_boundary_eligible"] is False
    assert state["path_stats"]["qwen4_hot_boundary_policy_eligible"] is True


@pytest.mark.parametrize("forked,enabled,marker", [
    (True, True, 1024), (True, False, 0), (False, True, 0), (False, False, 0),
])
def test_real_slot_constructor_marks_only_candidate_boundary_fork(forked, enabled, marker):
    path = ROOT / "runtime/engine.py"
    tree = ast.parse(path.read_text())
    owner = next(node for node in tree.body
                 if isinstance(node, ast.ClassDef) and node.name == "StreamingEngine")
    constructor = next(node for node in owner.body
                       if isinstance(node, ast.FunctionDef)
                       and node.name == "_new_hot_prompt_slot")
    namespace = {"_HotPromptSlot": lambda **kw: SimpleNamespace(
        **{"qwen4_retention_tile": 0, **kw})}
    exec(compile(ast.Module(body=[constructor], type_ignores=[]), str(path), "exec"), namespace)
    retained, endpoint = object(), object()
    state = SimpleNamespace(cfg=SimpleNamespace(model_type="qwen4_exp"),
                            rc=config(qwen4_hot_kv_tile_aligned=enabled),
                            _h_last=object())
    slot = namespace["_new_hot_prompt_slot"](
        state, recurrent_exact_only=True,
        boundary_fork_kv=retained if forked else None,
        boundary_fork_tokens=1024, tokens=tuple(range(1611)),
        full_tokens=tuple(range(1680)), kv=endpoint, logits=object(),
        prompt_endpoint_logits=object(), reusable_watermark=1024,
        prompt_state_approximate=False, tool_capsules=(), segment_chain=(),
        cache_namespace="default")
    assert slot.qwen4_retention_tile == marker
    assert slot.kv is (retained if forked else endpoint)
    assert len(slot.tokens) == (1024 if forked else 1680)
    assert (slot.exact_hidden is None) is forked


def test_mtp_bootstrap_preserves_boundary_and_disable_hint():
    path = ROOT / "runtime/qwen4_mtp.py"
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.ClassDef) and n.name == "_Qwen4MTPBootstrapPrompt")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    prompt = SimpleNamespace(stable_boundary_tokens=1606, disable_hot_prompt_kv=True)
    copy = namespace["_Qwen4MTPBootstrapPrompt"](prompt, (7, 8, 9))
    assert copy.stable_boundary_tokens == 1606
    assert copy.disable_hot_prompt_kv is True
    assert copy.token_ids == (7, 8, 9)


def test_policy_has_runtime_default_yaml_mapping_engine_identity_and_no_second_hint_read():
    source = (ROOT / "runtime/engine.py").read_text()
    assert "qwen4_hot_kv_tile_aligned: bool = False" in source
    assert 'qwen4_hot_kv_tile_aligned=run.get("qwen4_hot_kv_tile_aligned", False)' in source
    assert "stable_boundary = stable_boundary_positions" in source
    assert "and not slot_matches_tile_retention(" in source
    assert source.count("qwen4_retention_tile=(") == 1
    constructor = source[source.index("    def _new_hot_prompt_slot("):
                         source.index("    def _retain_interrupted_prefill(")]
    assert constructor.index("qwen4_retention_tile=(") < constructor.index(
        "tokens=full_tokens")
    server = (ROOT / "runtime/server.py").read_text()
    assert '("VMODEL_QWEN4_HOT_KV_TILE_ALIGNED", "0")' in server
    assert 'rc.qwen4_hot_kv_tile_aligned = qwen4_request_identity[36] == "1"' in server
