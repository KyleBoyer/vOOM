"""F112: first real speed measurement of QwenNgramSpeculativeEngine
(prompt-lookup n-gram speculation for Qwen3.5's hybrid DeltaNet layers).

tests/test_qwen35_ngram_speculative.py already proves byte-identical
correctness against the real Qwen3.5-4B checkpoint, but this feature had
never been measured for real speed. Tests against the real Qwen3.5-9B
checkpoint (NOT fully resident on a 16GB machine, confirmed by its own
plain baseline elsewhere in this session's real measurements taking
~140s for 48 tokens vs Qwen3.5-4B's ~9s for the same generation) --
the same disk-paged regime where F110's native MTP showed its biggest
win (6.08x), using the same repetitive "capital of X is Y" prompt that
gives the n-gram proposer strong recurring-template matches.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from runtime.engine import RuntimeConfig, StreamingEngine
from runtime.qwen35_ngram import QwenNgramSpeculativeEngine
from runtime.sampler import SamplingParams

_REAL_MODEL_DIR = (
    Path(__file__).resolve().parent.parent / "models" / "Qwen3.5-9B-mlx-all-mxfp4")
_model_skip = pytest.mark.skipif(
    not _REAL_MODEL_DIR.exists(),
    reason="real Qwen3.5-9B-mlx-all-mxfp4 checkpoint not present on this machine")

_PROMPT = (
    "The capital of France is Paris. The capital of Germany is Berlin. "
    "The capital of Italy is Rome. The capital of Spain is"
)


def _run(use_ngram: bool, max_tokens: int = 12):
    rc = RuntimeConfig(prefill_chunk_size=512)
    engine = StreamingEngine(str(_REAL_MODEL_DIR), rc)
    driver = QwenNgramSpeculativeEngine(engine) if use_ngram else engine
    try:
        result = driver.generate(
            _PROMPT, max_tokens=max_tokens,
            sampling=SamplingParams(temperature=0.0))
    finally:
        engine.close()
    return result


@_model_skip
def test_qwen35_ngram_matches_plain_target_byte_identical():
    baseline = _run(use_ngram=False)
    ngram = _run(use_ngram=True)
    assert ngram["tokens"] == baseline["tokens"], (
        "QwenNgramSpeculativeEngine's verified-draft scheme must produce "
        "byte-identical greedy output to the plain target engine"
    )
    assert ngram["text"] == baseline["text"]


@_model_skip
def test_qwen35_ngram_actually_engages_on_a_repetitive_prompt():
    result = _run(use_ngram=True, max_tokens=24)
    stats = result["path_stats"]
    assert stats.get("qwen_ngram_enabled") == 1
    assert stats.get("qwen_ngram_used") == 1, (
        "expected n-gram speculation to actually engage for a real greedy "
        "request against a real checkpoint with a repetitive prompt, not "
        "silently fall back"
    )
    assert stats.get("qwen_ngram_proposed", 0) > 0
    assert stats.get("qwen_ngram_accepted", 0) > 0
