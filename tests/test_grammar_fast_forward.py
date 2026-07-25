"""Grammar fast-forward (token-level exact) and string-level jump-forward
(lossy) decoding for constrained generation.

Three layers of proof, cheapest first:
1. Matcher-level unit tests (no model weights): forced_run's behavior on a
   real compiled tool grammar, including the auto-profile terminated-matcher
   crash regression (xgrammar hard-fails find_jump_forward_string() after a
   stop token is accepted -- live-crashed on the real Plex capture
   2026-07-23 because the auto profile terminates WITHOUT setting
   GrammarConstraint.completed).
2. Real-model exact A/B (token-level mode): byte-identical greedy tokens.
3. Real-model lossy A/B (string-level mode): identical RENDERED TEXT with a
   real forced-token count (token ids may differ by design -- that is the
   documented lossy tradeoff, gated to the fast profile in server.py).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from runtime.engine import RuntimeConfig, StreamingEngine
from runtime.sampler import SamplingParams
from runtime.structured import GrammarConstraint

_REAL_MODEL_DIR = (
    Path(__file__).resolve().parent.parent / "models" / "Qwen3.5-4B")
_skip = pytest.mark.skipif(
    not _REAL_MODEL_DIR.exists(),
    reason="real Qwen3.5-4B checkpoint not present on this machine")

_TOOLS = [{
    "type": "function",
    "function": {
        "name": "plex_list_library",
        "description": "List media items in the Plex library with filters.",
        "parameters": {
            "type": "object",
            "properties": {
                "mediaType": {"type": "string", "enum": ["all", "movie", "show"]},
                "limit": {"type": "integer"},
                "offset": {"type": "integer"},
            },
            "required": ["mediaType"],
        },
    },
}]

_PROMPT = (
    "user: List every movie and show rated PG-13 or below, all media "
    "types, first page, limit 50.\nassistant:"
)


def _fake_engine():
    """Enough of an engine for GrammarConstraint construction: the compiler
    needs only _model_dir + cfg.vocab_size/eos_token_ids (tokenizer files,
    no weights)."""
    return SimpleNamespace(
        _model_dir=_REAL_MODEL_DIR,
        cfg=SimpleNamespace(vocab_size=248320, eos_token_ids=[248044]),
    )


@_skip
def test_forced_run_terminated_auto_matcher_returns_empty_not_crash():
    constraint = GrammarConstraint.tools(
        _fake_engine(), _TOOLS, required=False, allow_parallel=True)
    matcher = constraint.matcher
    assert matcher.accept_string(
        '<tool_call>{"name": "plex_list_library", '
        '"arguments": {"mediaType": "all"}}</tool_call>')
    stop_ids = list(matcher.stop_token_ids)
    if stop_ids:
        matcher.accept_token(stop_ids[0])
    if not matcher.is_terminated():
        pytest.skip("matcher did not terminate on this grammar/version")
    assert constraint.forced_run(32) == []
    assert constraint.forced_run(
        32, encode=lambda text: []) == []


@_skip
def test_forced_run_string_level_jumps_required_tool_scaffold():
    constraint = GrammarConstraint.tools(
        _fake_engine(), _TOOLS, required=True, allow_parallel=False)
    jump = constraint.matcher.find_jump_forward_string()
    assert jump.startswith('<tool_call>{"name": "plex_list_library"'), jump


def _run(model_dir: Path, *, token_level=False, string_level=False,
         max_tokens: int = 96):
    rc = RuntimeConfig(
        prefill_chunk_size=512,
        grammar_fast_forward=token_level,
        grammar_jump_forward_lossy=string_level)
    engine = StreamingEngine(str(model_dir), rc)
    try:
        constraint = GrammarConstraint.tools(
            engine, _TOOLS, required=True, allow_parallel=False)
        return engine.generate(
            _PROMPT, max_tokens=max_tokens,
            sampling=SamplingParams(temperature=0.0),
            constraint=constraint)
    finally:
        engine.close()


@_skip
def test_token_level_fast_forward_is_byte_identical():
    baseline = _run(_REAL_MODEL_DIR)
    fast = _run(_REAL_MODEL_DIR, token_level=True)
    assert fast["tokens"] == baseline["tokens"]
    assert fast["text"] == baseline["text"]


@_skip
def test_string_level_jump_forward_preserves_text_and_actually_forces():
    baseline = _run(_REAL_MODEL_DIR)
    jumped = _run(_REAL_MODEL_DIR, string_level=True)
    # Token ids may legitimately differ (documented lossy tradeoff); the
    # rendered text of the DETERMINED spans must not. For this single-tool
    # required grammar the full call is text-stable in practice.
    assert jumped["text"] == baseline["text"]
    assert jumped["path_stats"].get("grammar_fast_forward_tokens", 0) >= 10, (
        "string-level jump-forward should force a large scaffold run for a "
        "required single-tool grammar; if not, the mechanism regressed"
    )


def test_canonical_whitespace_extends_json_schema_forced_run():
    """`GrammarConstraint.json`'s default (`any_whitespace=True`, matching
    server.py's non-lossy path) leaves almost nothing determined, since
    arbitrary whitespace could legally precede every key/punctuation mark.
    `canonical_whitespace=True` (server.py wires this to
    `grammar_jump_forward_lossy`) collapses that ambiguity the same way the
    required-tool grammar already does, and should force well past the
    opening brace. No model weights needed -- this is matcher/compiler-only.
    """
    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "count": {"type": "integer"},
        },
        "required": ["title", "count"],
    }
    loose = GrammarConstraint.json(_fake_engine(), schema, strict=True)
    strict = GrammarConstraint.json(
        _fake_engine(), schema, strict=True, canonical_whitespace=True)
    loose_jump = loose.matcher.find_jump_forward_string()
    strict_jump = strict.matcher.find_jump_forward_string()
    assert strict_jump.startswith('{"title"'), strict_jump
    assert len(strict_jump) > len(loose_jump) + 5, (
        loose_jump, strict_jump)


@_skip
def test_canonical_whitespace_json_schema_stays_valid_and_forces_more():
    import json as _json

    rc_common = dict(prefill_chunk_size=512, grammar_jump_forward_lossy=True)
    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "count": {"type": "integer"},
        },
        "required": ["title", "count"],
    }
    prompt = (
        "user: Give me a JSON object with a short title and a count of 3.\n"
        "assistant:"
    )

    def _generate(canonical_whitespace: bool):
        engine = StreamingEngine(str(_REAL_MODEL_DIR), RuntimeConfig(**rc_common))
        try:
            constraint = GrammarConstraint.json(
                engine, schema, strict=True,
                canonical_whitespace=canonical_whitespace)
            return engine.generate(
                prompt, max_tokens=64,
                sampling=SamplingParams(temperature=0.0),
                constraint=constraint)
        finally:
            engine.close()

    loose = _generate(False)
    strict = _generate(True)
    for result in (loose, strict):
        parsed = _json.loads(result["text"])
        assert set(parsed) == {"title", "count"}
    assert (strict["path_stats"].get("grammar_fast_forward_tokens", 0)
            > loose["path_stats"].get("grammar_fast_forward_tokens", 0)), (
        loose["path_stats"], strict["path_stats"])
