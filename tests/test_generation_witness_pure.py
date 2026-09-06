"""Real serving helper/hook tests without importing MLX or a model."""

import ast
import copy
import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def api(monkeypatch):
    source = Path(__file__).resolve().parents[1] / "runtime/server.py"
    tree = ast.parse(source.read_text())
    names = {"_attach_generation_witness", "_engine_generate", "_has_own_method",
             "_vision_protocol_timing"}
    nodes = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == len(names)
    namespace = {"os": os, "hashlib": hashlib, "json": json, "time": time,
                 "__package__": "runtime",
                 "_request_expert_trace_target": lambda engine: None,
                 "_persist_request_expert_trace": lambda *args: None}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    monkeypatch.delenv("VMODEL_GENERATION_WITNESS", raising=False)
    monkeypatch.delenv("VMODEL_DEBUG_ENGINE_REPORT", raising=False)
    return SimpleNamespace(**namespace)


@pytest.mark.parametrize("flag", [None, "", "0", "true", "invalid"])
def test_witness_is_strictly_opt_in(api, monkeypatch, flag):
    if flag is not None:
        monkeypatch.setenv("VMODEL_GENERATION_WITNESS", flag)
    result = {"tokens": [1], "text": "private"}
    before = copy.deepcopy(result)
    api._attach_generation_witness(object(), result)
    assert result == before


def test_witness_hashes_real_ids_and_preserves_result(api, monkeypatch):
    monkeypatch.setenv("VMODEL_GENERATION_WITNESS", "1")
    prompt = SimpleNamespace(token_ids=(4, 50))
    result = {"tokens": [1, 23], "text": "private café", "path_stats": {"x": 3}}
    before = copy.deepcopy(result)
    api._attach_generation_witness(prompt, result)
    witness = result.pop("generation_witness")
    raw_calls = result.pop("tool_call_text_witness")
    assert raw_calls["available"] is True and raw_calls["framed_call_objects"] == 0
    assert result == before
    assert witness["available"] is True
    assert witness["generated_token_count"] == 2
    assert witness["generated_token_ids_sha256"] == hashlib.sha256(b"[1,23]").hexdigest()
    assert witness["prepared_prompt_token_ids_sha256"] == hashlib.sha256(b"[4,50]").hexdigest()
    assert witness["prepared_prompt_token_count"] == 2
    assert witness["engine_text_sha256"] == hashlib.sha256(before["text"].encode()).hexdigest()
    assert witness["engine_text_bytes"] == len(before["text"].encode())
    assert "private" not in json.dumps(witness)


def test_same_text_different_ids_cannot_fake_token_equivalence(api, monkeypatch):
    monkeypatch.setenv("VMODEL_GENERATION_WITNESS", "1")
    results = [{"tokens": [1, 23], "text": "same"},
               {"tokens": [12, 3], "text": "same"}]
    for result in results:
        api._attach_generation_witness("unprepared", result)
    left, right = [result["generation_witness"] for result in results]
    assert left["engine_text_sha256"] == right["engine_text_sha256"]
    assert left["generated_token_ids_sha256"] != right["generated_token_ids_sha256"]
    assert left["prepared_prompt_token_ids_sha256"] is None
    assert left["prepared_prompt_token_count"] is None


@pytest.mark.parametrize("tokens", [None, "12", [True], [-1], [1.5], {1: 2}])
def test_invalid_generated_ids_are_unavailable_not_reencoded(api, monkeypatch, tokens):
    monkeypatch.setenv("VMODEL_GENERATION_WITNESS", "1")
    result = {"tokens": tokens, "text": "private text"}
    api._attach_generation_witness("prompt", result)
    assert result["generation_witness"] == {
        "schema": "voom.generation-witness.v1", "available": False,
        "error_type": "ValueError",
    }


@pytest.mark.parametrize("bad_source", ["text", "prompt_ids"])
def test_invalid_other_source_data_is_explicitly_unavailable(api, monkeypatch, bad_source):
    monkeypatch.setenv("VMODEL_GENERATION_WITNESS", "1")
    result = {"tokens": [1], "text": None if bad_source == "text" else "valid"}
    prompt = SimpleNamespace(token_ids=[False] if bad_source == "prompt_ids" else [1])
    api._attach_generation_witness(prompt, result)
    assert result["generation_witness"]["available"] is False


@pytest.mark.parametrize("enabled", [False, True])
def test_real_generation_hook_and_protocol_projection_preserve_output(api, monkeypatch, enabled):
    monkeypatch.setenv("VMODEL_GENERATION_WITNESS", "1" if enabled else "0")
    events = []
    result = {"tokens": [7, 8], "text": "unparsed tool delimiters", "path_stats": {}}
    class Engine:
        def generate(self, prompt, **kwargs):
            events.append("generated")
            return result
    output = api._engine_generate(Engine(), SimpleNamespace(token_ids=(1, 2)), max_tokens=512)
    assert output is result
    assert output["tokens"] == [7, 8]
    assert events == ["generated"]
    timing = api._vision_protocol_timing(result)
    assert ("generation_witness" in timing) is enabled
    assert ("tool_call_text_witness" in timing) is enabled
    if enabled:
        assert timing["generation_witness"] == result["generation_witness"]
        assert timing["generation_witness"]["generated_token_count"] == 2


def test_empty_generation_is_distinct_from_missing_generation(api, monkeypatch):
    monkeypatch.setenv("VMODEL_GENERATION_WITNESS", "1")
    result = {"tokens": [], "text": ""}
    api._attach_generation_witness(SimpleNamespace(token_ids=()), result)
    assert result["generation_witness"]["available"] is True
    assert result["generation_witness"]["generated_token_count"] == 0


def test_raw_tool_observer_projects_before_protocol_without_changing_ids_or_text(api, monkeypatch):
    monkeypatch.setenv("VMODEL_GENERATION_WITNESS", "1")
    text = '<tool_call>{"name":"private_tool","arguments":{"q":"PRIVATE"}}</tool_call>'
    result = {"tokens": [10, 11], "text": text + text}
    api._attach_generation_witness(SimpleNamespace(token_ids=[1]), result)
    timing = api._vision_protocol_timing(result)
    raw = timing["tool_call_text_witness"]
    assert raw["framed_call_objects"] == 2 and raw["canonical_duplicate_count"] == 1
    assert raw["observation_seconds"] >= 0
    assert raw["raw_text_sha256"] == timing["generation_witness"]["engine_text_sha256"]
    assert result["tokens"] == [10, 11] and result["text"] == text + text
    assert "PRIVATE" not in json.dumps(raw) and "private_tool" not in json.dumps(raw)


def test_raw_tool_observer_failure_cannot_invalidate_original_generation_witness(api, monkeypatch):
    import runtime.tool_call_witness as observer
    monkeypatch.setenv("VMODEL_GENERATION_WITNESS", "1")

    def fail(text):
        raise RuntimeError("private diagnostic failure")

    monkeypatch.setattr(observer, "hermes_text_witness", fail)
    result = {"tokens": [7], "text": "answer"}
    api._attach_generation_witness(SimpleNamespace(token_ids=[1]), result)
    assert result["generation_witness"]["available"] is True
    assert result["tool_call_text_witness"] == {
        "schema": "voom.hermes-text-witness.v1", "available": False,
        "error_type": "RuntimeError"}
    assert result["tokens"] == [7] and result["text"] == "answer"


@pytest.mark.parametrize("source", ["path_stats", "top_level"])
def test_phase_head_memory_protocol_preserves_nested_unavailable_and_zero(api, source):
    key = "qwen4_mtp_idle_head_memory_witness"
    observation = {
        "schema": "voom.phase-head-memory-witness.v1",
        "atomic": False, "release_callable": True,
        "before": {"available": False, "weight_cache_pinned_bytes": None},
        "after": {"available": True, "weight_cache_pinned_bytes": 0,
                  "observation_seconds": 0.001},
    }
    fields = {key: observation}
    result = {"path_stats": fields} if source == "path_stats" else fields
    original = copy.deepcopy(result)
    timing = api._vision_protocol_timing(result)
    assert timing[key] == observation
    assert result == original
    assert key not in api._vision_protocol_timing({"path_stats": {}})
    json.dumps(timing[key], allow_nan=False)


@pytest.mark.parametrize("source", ["path_stats", "top_level"])
@pytest.mark.parametrize("reason,eligible,policy_eligible", [
    ("tile-aligned", True, True),
    ("no-admissible-complete-tile", False, False),
    ("tile-aligned", False, True),
])
def test_aligned_boundary_protocol_keeps_reason_string_and_typed_counts(
        api, source, reason, eligible, policy_eligible):
    fields = {
        "qwen4_hot_boundary_requested": 1606,
        "qwen4_hot_boundary_effective": 1024 if policy_eligible else 0,
        "qwen4_hot_boundary_tile": 1024,
        "qwen4_hot_boundary_eligible": eligible,
        "qwen4_hot_boundary_policy_eligible": policy_eligible,
        "qwen4_hot_boundary_reason": reason,
    }
    result = {"path_stats": fields} if source == "path_stats" else fields
    timing = api._vision_protocol_timing(result)
    assert timing["qwen4_hot_boundary_reason"] == reason
    for key, value in fields.items():
        if key != "qwen4_hot_boundary_reason":
            assert type(timing[key]) is int
            assert timing[key] == int(value)


@pytest.mark.parametrize("source", ["path_stats", "top_level"])
def test_retained_compact_protocol_preserves_copy_timing_and_integer_bytes(api, source):
    fields = {
        "qwen4_retained_conv_compact_calls": 1,
        "qwen4_retained_conv_compact_arrays": 37,
        "qwen4_retained_conv_compact_bytes": 2396160,
        "qwen4_retained_conv_compact_scratch_bytes": 4792320,
        "qwen4_retained_conv_compact_prefix_tokens": 1024,
        "qwen4_retained_conv_compact_active_before_bytes": 1552511544,
        "qwen4_retained_conv_compact_active_after_bytes": 1554907704,
        "qwen4_retained_projected_logical_bytes": 144003072,
        "qwen4_retained_qsa_backing_allowance_bytes": 12582912,
        "qwen4_retained_conv_compact_seconds": 0.035678,
    }
    result = {"path_stats": fields} if source == "path_stats" else fields
    timing = api._vision_protocol_timing(result)
    for key, value in fields.items():
        if key.endswith("_seconds"):
            assert timing[key] == pytest.approx(value, abs=0.0001)
        else:
            assert type(timing[key]) is int and timing[key] == value
