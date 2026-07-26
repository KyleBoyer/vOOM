"""F113 follow-on (2026-07-25): suffix/prompt-lookup speculative decoding
(rc.suffix_decoding) now works correctly for dense qwen3_5 targets too, via
the same fork/restore pattern runtime/qwen35_ngram.py already proved safe
for its own (simpler, single-request) round loop.

Background: fallback_reason() blanket-excluded any kda_cache-bearing
target ("recurrent-state-target"), since KVCache.trim() has no kda_cache
branch and a partially-accepted round would silently corrupt the
DeltaNet/KDA recurrent state. run_shared_prefill_suffix_decode now forks
kv.kda_cache before each round's speculative feed and, whenever the
actually-committed feed is shorter than the full window fed to the
verifier (a partial reject, or a stop/eos mid-window), restores the fork
and re-feeds exactly the committed tokens -- mirroring qwen35_ngram.py's
own proven pattern. forward_tokens_serial_positions was separately
verified (tests/test_f113_glm_serial_positions_oracle.py's sibling
diagnostic, run standalone against real Qwen3.5-9B) to produce
byte-identical logits AND kda_cache state vs. true sequential decode.

Real A/B against the real Qwen3.5-9B checkpoint, repetitive prompt (to
force real prompt-lookup engagement, including partial rejects where the
draft diverges from the true continuation -- exactly where the new
refeed logic is exercised).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from runtime.engine import RuntimeConfig, StreamingEngine
from runtime.sampler import SamplingParams

_REAL_MODEL_DIR = (
    Path(__file__).resolve().parent.parent / "models" / "Qwen3.5-9B-mlx-all-mxfp4"
)
_model_skip = pytest.mark.skipif(
    not _REAL_MODEL_DIR.exists(),
    reason="real Qwen3.5-9B checkpoint not present on this machine")

_PROMPT = (
    "The capital of France is Paris. The capital of Germany is Berlin. "
    "The capital of Italy is Rome. The capital of Spain is Madrid. "
    "The capital of Portugal is"
)


def _run(use_suffix: bool, max_tokens: int = 24):
    rc = RuntimeConfig(
        prefill_chunk_size=256, min_weight_cache_mb=200, max_weight_cache_mb=4000,
        suffix_decoding=use_suffix,
    )
    engine = StreamingEngine(str(_REAL_MODEL_DIR), rc)
    try:
        result = engine.generate(
            _PROMPT, max_tokens=max_tokens,
            sampling=SamplingParams(temperature=0.0))
    finally:
        engine.close()
    return result


@_model_skip
def test_suffix_decoding_matches_plain_target_and_actually_engages_for_qwen35():
    """Two real Qwen3.5-9B sessions (baseline, suffix) in one test,
    mirroring test_f113_glm_suffix_decoding_real.py's consolidation
    rationale -- keep real-model sessions per pytest process low."""
    baseline = _run(use_suffix=False)
    suffix = _run(use_suffix=True)
    assert suffix["tokens"] == baseline["tokens"], (
        "suffix decoding must produce byte-identical greedy output to "
        "plain decode for a real dense qwen3_5 target now that its "
        "kda_cache is fork/restore-safe across speculative rounds"
    )
    assert suffix["text"] == baseline["text"]

    stats = suffix["path_stats"]
    assert stats.get("suffix_decoding_enabled") == 1
    assert stats.get("suffix_decoding_used") == 1, (
        "expected suffix decoding to actually engage for a real greedy "
        "request against a real dense qwen3_5 checkpoint with a "
        "repetitive prompt, not silently fall back"
    )
    assert stats.get("suffix_decoding_proposed", 0) > 0
    assert stats.get("suffix_decoding_accepted", 0) > 0
    # A real partial-reject (proposed > accepted) is what actually
    # exercises the new kda_cache fork/restore refeed path -- assert it
    # actually happened rather than merely being a passive by-product of
    # a perfect-match run that never touches the new code.
    assert stats["suffix_decoding_proposed"] > stats["suffix_decoding_accepted"], (
        "expected at least one real partial reject in this run so the "
        "new kda_cache fork/restore refeed logic is actually exercised, "
        "not just present but untested"
    )
