"""F11: prompt-lookup (n-gram) speculative decoding for qwen3_5, real
greedy-token correctness proof against the real Qwen3.5-4B checkpoint.

Uses a genuinely repetitive prompt so the n-gram lookup actually finds
matches (exercising accept/partial-accept/reject/restore, not just the
trivial always-empty-proposal case an unrepetitive prompt would give).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from runtime.engine import RuntimeConfig, StreamingEngine
from runtime.qwen35_ngram import QwenNgramSpeculativeEngine
from runtime.sampler import SamplingParams

_REAL_MODEL_DIR = (
    Path(__file__).resolve().parent.parent / "models" / "Qwen3.5-4B")
_skip = pytest.mark.skipif(
    not _REAL_MODEL_DIR.exists(),
    reason="real Qwen3.5-4B checkpoint not present on this machine")

_REPETITIVE_PROMPT = (
    "Repeat the following phrase exactly six times, separated by spaces, "
    "and do not say anything else: "
    "The quick brown fox jumps over the lazy dog. "
    "The quick brown fox jumps over the lazy dog. "
    "The quick brown fox jumps over the lazy dog. "
)


def _plain_engine():
    rc = RuntimeConfig(prefill_chunk_size=512)
    return StreamingEngine(str(_REAL_MODEL_DIR), rc)


@_skip
def test_ngram_speculative_matches_plain_greedy_byte_identical():
    plain = _plain_engine()
    try:
        baseline = plain.generate(
            _REPETITIVE_PROMPT, max_tokens=40,
            sampling=SamplingParams(temperature=0.0))
    finally:
        plain.close()

    spec_target = _plain_engine()
    try:
        spec = QwenNgramSpeculativeEngine(spec_target, k=8)
        result = spec.generate(
            _REPETITIVE_PROMPT, max_tokens=40,
            sampling=SamplingParams(temperature=0.0))
    finally:
        spec_target.close()

    assert result["tokens"] == baseline["tokens"], (
        "n-gram speculative decoding must produce byte-identical greedy "
        "output to plain generation for a real repetitive prompt"
    )
    assert result["text"] == baseline["text"]
    stats = result["path_stats"]
    assert stats["qwen_ngram_used"] == 1
    assert stats["qwen_ngram_proposed"] > 0, (
        "this prompt is deliberately repetitive -- the n-gram lookup should "
        "find at least one real match; if not, the test's own repetition "
        "no longer exercises the accept/reject path this test is for"
    )


@_skip
def test_ngram_speculative_falls_back_for_stochastic_sampling():
    target = _plain_engine()
    try:
        spec = QwenNgramSpeculativeEngine(target, k=8)
        result = spec.generate(
            "Hello", max_tokens=4,
            sampling=SamplingParams(temperature=0.7))
        assert result["path_stats"]["qwen_ngram_used"] == 0
        assert result["path_stats"]["qwen_ngram_fallback_reason"] == (
            "stochastic-sampling")
    finally:
        target.close()
