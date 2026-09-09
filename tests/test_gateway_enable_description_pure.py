"""Prompt-only opt-in contract, not a model stopping or quality evaluation."""

import ast
import copy
from pathlib import Path

import pytest

from runtime import server
from runtime.profiles import discover_runtime_profiles, resolve_runtime_profiles

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("value,expected", [("0", "legacy"), ("1", "external-only-v1")])
def test_explicit_description_policy(value, expected):
    assert server._hidden_gateway_enable_description_policy(value) == expected


@pytest.mark.parametrize("value", [None, False, True, 0, 1, "", "auto", "true", " 1", "1 "])
def test_description_policy_rejects_ambiguous_values(value):
    with pytest.raises(server.RequestValidationError, match="must be 0 or 1"):
        server._hidden_gateway_enable_description_policy(value)


@pytest.mark.parametrize("value", [None, 0, 1, "0", "1"])
def test_private_helper_requires_actual_boolean(value):
    with pytest.raises(ValueError, match="boolean"):
        server._hidden_gateway_virtual_pairs(enable_external_only=value)


def test_default_catalog_is_unchanged_and_only_enable_prose_differs():
    default = server._hidden_gateway_virtual_pairs()
    assert default == server._hidden_gateway_virtual_pairs(enable_external_only=False)
    before = copy.deepcopy(default)
    candidate = server._hidden_gateway_virtual_pairs(enable_external_only=True)
    for baseline, changed in zip(default, candidate, strict=True):
        assert len(baseline) == len(changed) == 2
        assert baseline[0] == changed[0]  # Search tool is unchanged.
        old, new = (row.get("function", row) for row in (baseline[1], changed[1]))
        assert "interpreting a prior tool result" in old["description"]
        assert "does not require enabling tools" in new["description"]
        assert "different external capability is still needed" in new["description"]
        assert old != new
        assert {k:v for k,v in old.items() if k != "description"} == {
            k:v for k,v in new.items() if k != "description"}
    assert default == before  # No mutation of a retained catalog.
    assert server._hidden_gateway_enable_description_policy() == "legacy"


@pytest.mark.parametrize("more,expected", [(False, "auto"), (True, "specific:vmodel_enable_tools")])
@pytest.mark.parametrize("followup", [False, True])
def test_description_option_does_not_force_a_completed_page_to_stop(more, expected, followup):
    messages = [dict(role="user", content="List every invoice." + (
        " Then save the total to the accounting application." if followup else "")),
        dict(role="tool", content='{"hasMore": ' + str(more).lower() + '}')]
    before = copy.deepcopy(messages)
    for enabled in (False, True):
        server._hidden_gateway_virtual_pairs(enable_external_only=enabled)
        reason = server._hidden_gateway_force_reason(messages)
        assert server._hidden_gateway_decision_choice("auto", reason, True) == expected
    assert messages == before


def test_overlay_changes_only_the_description_flag():
    catalog = discover_runtime_profiles((ROOT / "profiles",))
    base = ("huihui-qwen38-27b-full-workflow-lifetime-audit", "generation-witness",
            "host-activity-witness", "qwen35-serial-kv-reclaim", "qwen35-serial-kv-reclaim-topup")
    _, old = resolve_runtime_profiles(base, catalog)
    _, new = resolve_runtime_profiles((*base, "gateway-enable-external-only"), catalog)
    assert "VMODEL_FAST_TOOL_GATEWAY_ENABLE_EXTERNAL_ONLY" not in old
    assert new == {**old, "VMODEL_FAST_TOOL_GATEWAY_ENABLE_EXTERNAL_ONLY": "1"}


def test_serving_path_passes_explicit_policy_and_exposes_phase_metadata():
    tree = ast.parse((ROOT / "runtime/server.py").read_text())
    handler = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Handler")
    method = next(n for n in handler.body if isinstance(n, ast.FunctionDef) and n.name == "_do_responses")
    calls = [n for n in ast.walk(method) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "_hidden_gateway_virtual_pairs"]
    assert len(calls) == 1
    assert [(k.arg, ast.unparse(k.value)) for k in calls[0].keywords] == [
        ("enable_external_only", "gateway_enable_description_profile == 'external-only-v1'")]
    fields = [(key.value, ast.unparse(value)) for n in ast.walk(method)
              if isinstance(n, ast.Dict) for key, value in zip(n.keys, n.values)
              if isinstance(key, ast.Constant) and key.value == "gateway_enable_description_profile"]
    assert fields == [("gateway_enable_description_profile", "gateway_enable_description_profile")]


@pytest.mark.parametrize("observed,expected", [
    ("external-only-v1", True), ("legacy", False), (None, False), (True, False)])
def test_captured_gate_requires_actual_selected_description_metadata(observed, expected):
    from tests.fixtures.huihui_captured_action_gate import acceptance
    from tests.test_huihui_captured_action_gate_pure import row

    checks = acceptance(row(), {"vmodel_tool_selection": {
        "gateway_enable_description_profile": observed}}, {
            "profiles": [], "profile_digest": "unused",
            "gateway_enable_description_profile": "external-only-v1"}, initial_action=False)
    assert checks["gateway_enable_description_profile"] is expected
