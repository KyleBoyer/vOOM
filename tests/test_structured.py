"""JSON Schema and XGrammar adapter tests."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

from runtime.structured import (GrammarConstraint, JSONSchemaValidationError,
                                _required_tool_grammar, effective_tool_schema,
                                tool_call_json_schema,
                                validate_json_schema)


WEATHER = {"type": "function", "function": {
    "name": "weather",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "additionalProperties": False,
    },
}}


def test_schema_validation_accepts_valid_and_rejects_wrong_arguments():
    schema = WEATHER["function"]["parameters"]
    validate_json_schema({"city": "Chicago"}, schema)
    with pytest.raises(JSONSchemaValidationError, match="city"):
        validate_json_schema({}, schema)
    with pytest.raises(JSONSchemaValidationError, match="Additional properties"):
        validate_json_schema({"city": "Chicago", "units": "C"}, schema)


def test_tool_call_union_binds_name_to_its_own_argument_schema():
    clock = {"type": "function", "function": {
        "name": "clock", "parameters": {
            "type": "object", "properties": {"tz": {"type": "string"}},
            "required": ["tz"], "additionalProperties": False}}}
    schema = tool_call_json_schema([WEATHER, clock])
    validate_json_schema(
        {"name": "weather", "arguments": {"city": "Paris"}}, schema)
    with pytest.raises(JSONSchemaValidationError):
        validate_json_schema(
            {"name": "weather", "arguments": {"tz": "UTC"}}, schema)


def test_required_tool_grammar_has_canonical_json_and_exact_markers():
    grammar = str(_required_tool_grammar(
        tool_call_json_schema([WEATHER]), allow_parallel=True))
    tool_rule = next(
        line for line in grammar.splitlines() if line.startswith("tool_call ::="))
    calls_rule = next(
        line for line in grammar.splitlines() if line.startswith("tool_calls ::="))
    assert '[ \\n\\t]*' not in tool_rule
    assert '"<tool_call>" tool_json "</tool_call>"' in tool_rule
    assert '[ \\n\\t]*' not in calls_rule


def test_effective_tool_schema_honors_x_optional_without_mutating_wire_schema():
    wire = {
        "type": "object",
        "properties": {
            "path": {"type": ["string", "null"]},
            "depth": {"type": ["integer", "null"]},
            "query": {"type": "string"},
        },
        "required": ["path", "depth", "query"],
        "x-optional": ["path", "depth"],
        "additionalProperties": False,
    }
    effective = effective_tool_schema(wire)
    assert effective["required"] == ["query"]
    assert "x-optional" not in effective
    assert wire["required"] == ["path", "depth", "query"]
    assert wire["x-optional"] == ["path", "depth"]
    validate_json_schema({"query": "files"}, effective)
    with pytest.raises(JSONSchemaValidationError):
        validate_json_schema({"query": "files", "depth": "deep"}, effective)


def test_effective_tool_schema_rejects_unknown_x_optional_property():
    with pytest.raises(JSONSchemaValidationError, match="unknown properties"):
        effective_tool_schema({
            "type": "object", "properties": {}, "x-optional": ["missing"],
        })


def test_xgrammar_constraint_accepts_complete_qwen_json_sequence():
    # This is a real tokenizer/compiler integration but does not load weights.
    from runtime.config import ModelConfig

    model = __import__("pathlib").Path.home() / "models/Qwen2.5-1.5B-Instruct-mlx-mxfp4"
    if not (model / "config.json").exists():
        pytest.skip("local Qwen tokenizer is not installed")
    from tokenizers import Tokenizer

    engine = SimpleNamespace(
        _model_dir=model,
        cfg=ModelConfig.from_dir(model),
        tokenizer=Tokenizer.from_file(str(model / "tokenizer.json")),
    )
    constraint = GrammarConstraint.json(
        engine, WEATHER["function"]["parameters"])
    ids = engine.tokenizer.encode('{"city":"Paris"}').ids
    for token in ids:
        masked = constraint.mask_logits(mx.zeros((engine.cfg.vocab_size,)))
        assert float(masked[token]) == 0.0
        constraint.accept_token(token)
    assert constraint.completed
