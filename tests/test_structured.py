"""JSON Schema and XGrammar adapter tests."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

from runtime.structured import (GrammarConstraint, JSONSchemaValidationError,
                                _compiler, _grammar_compatible_schema,
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


def test_grammar_schema_distributes_conditional_required_fields():
    source = {
        "type": "object",
        "properties": {
            "core": {"type": "string"},
            "root": {"type": "string"},
            "section": {"type": "string"},
        },
        "required": ["core"],
        "anyOf": [
            {"required": ["root"]},
            {"required": ["section"]},
        ],
        "additionalProperties": False,
    }
    compatible = _grammar_compatible_schema(source)
    assert "anyOf" in compatible
    assert all("core" in branch["required"]
               for branch in compatible["anyOf"])
    validate_json_schema({"core": "x", "root": "Kids"}, compatible)
    validate_json_schema({"core": "x", "section": "Kids"}, compatible)
    with pytest.raises(JSONSchemaValidationError):
        validate_json_schema({"root": "Kids"}, compatible)
    with pytest.raises(JSONSchemaValidationError):
        validate_json_schema({"core": "x"}, compatible)
    assert source["required"] == ["core"]


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


def test_xgrammar_compiler_accepts_kimi_local_tiktoken_code():
    """Kimi checkpoints expose their exact vocabulary through local code."""
    from pathlib import Path

    from runtime.config import ModelConfig

    model = Path(__file__).resolve().parents[1] / "models" / "Kimi-K3"
    if not (model / "tiktoken.model").exists():
        pytest.skip("local Kimi K3 tokenizer is not installed")
    engine = SimpleNamespace(
        _model_dir=model,
        cfg=ModelConfig.from_dir(model),
        _xgrammar_compiler=None,
    )

    assert _compiler(engine) is engine._xgrammar_compiler


class _FakeMatcher:
    """Stands in for xgr.GrammarMatcher, letting tests choose exactly which
    bits fill_next_token_bitmask writes and whether accept_token rejects,
    without a real compiled grammar."""

    def __init__(self, *, allow_all: bool, accept: bool = True):
        self._allow_all = allow_all
        self._accept = accept

    def fill_next_token_bitmask(self, bitmask) -> bool:
        bitmask.fill_(-1 if self._allow_all else 0)
        return True

    def accept_token(self, token: int) -> bool:
        return self._accept

    def is_completed(self) -> bool:
        return False

    def fork(self):
        return _FakeMatcher(
            allow_all=self._allow_all, accept=self._accept)


def test_dead_grammar_state_stops_generation_instead_of_crashing():
    """An under-tuned/untrained model (no native tool-call special tokens,
    e.g. OLMoE) can commit to a tool-call span it can never complete
    validly, walking the grammar into a state with genuinely zero legal
    next tokens. Every position is then -inf, so ANY sampler (greedy argmax
    included) returns an arbitrary index that the matcher would reject
    regardless of which one it picked. This must degrade to a clean stop,
    not a crashed request.
    """
    constraint = GrammarConstraint(
        matcher=_FakeMatcher(allow_all=False, accept=False),
        vocab_size=8, profile="test")
    masked = constraint.mask_logits(mx.zeros((8,)))
    assert bool(mx.all(masked == float("-inf")).item())
    # Whatever index a sampler picked from an all -inf row is irrelevant --
    # none of them were ever going to be legal. Must not raise.
    constraint.accept_token(0)
    assert constraint.completed


def test_rejection_despite_available_tokens_also_stops_cleanly():
    """Live-reproduced 2026-07-22 against OLMoE-1B-7B: _compiler() builds
    xgrammar's own vocabulary view via transformers.AutoTokenizer, separate
    from the raw tokenizers.Tokenizer this engine actually samples/decodes
    with -- for a checkpoint whose vocabulary the two disagree on (OLMoE's
    id 0 is an unusual non-BOS/EOS/pad special token, "|||IP_ADDRESS|||"),
    mask_logits can report real tokens as legal, greedy argmax can pick one
    of them, and the grammar matcher can still reject it. This is not
    resampling-recoverable mid-generation either -- it must ALSO degrade to
    a clean stop rather than crash the whole request, exactly like a fully
    dead grammar state, even though plenty of tokens were "allowed"."""
    constraint = GrammarConstraint(
        matcher=_FakeMatcher(allow_all=True, accept=False),
        vocab_size=8, profile="test")
    masked = constraint.mask_logits(mx.zeros((8,)))
    assert not bool(mx.any(masked == float("-inf")).item())
    constraint.accept_token(3)
    assert constraint.completed


def test_accepted_token_completes_normally():
    constraint = GrammarConstraint(
        matcher=_FakeMatcher(allow_all=True, accept=True),
        vocab_size=8, profile="test", stop_on_complete=False)
    constraint.mask_logits(mx.zeros((8,)))
    constraint.accept_token(3)
    assert not constraint.completed


def test_grammar_constraint_fork_is_independent():
    constraint = GrammarConstraint(
        matcher=_FakeMatcher(allow_all=True, accept=True),
        vocab_size=8, profile="test", stop_on_complete=False)
    forked = constraint.fork()

    assert forked is not constraint
    assert forked.matcher is not constraint.matcher
    forked.accept_token(3)
    assert not constraint.completed
    assert not forked.completed
