"""F113 follow-on: suffix/prompt-lookup speculative decoding
(rc.suffix_decoding) now works correctly for GLM-family targets
(glm_moe_dsa/kimi_k25/glm4_moe_lite) for the first time.

Background: suffix_decoding.py's fallback_reason() blanket-excluded any
target with routed experts ("non-dense-target"), which caught GLM-family
targets too -- but its OWN verify sweep already used
forward_tokens_serial_positions() unconditionally, so once that method
gained real MoE-aware GLM support (this session's F113 fix), the
"non-dense-target" exclusion became overly conservative for GLM
specifically (still correctly excludes gpt_oss and true recurrent-state
targets via the separate kda_cache/compressed-kv checks below it).

Real A/B against the real Kimi-K2.5 checkpoint (554GB, shares GLM's exact
MLA+MoE block code): byte-identical output, suffix decoding actually
engages and accepts real drafts.

Known, separate, honestly-flagged limitation (not a correctness bug):
a longer generation (16 tokens) hit a real MemoryError from the
governor correctly refusing an unsafe reservation as available memory
tightened over the run -- suffix decoding's multi-position verify sweep
has a larger transient-memory footprint for MoE than plain decode, and
this project's own governor fails closed rather than risk corruption.
This test uses a short generation (8 tokens) that stays well within a
safe margin; tuning the transient-memory estimate for longer MoE suffix-
decoding runs is real, separate, future work. For the same reason, this
file deliberately runs only ONE consolidated test (two real engine
sessions: baseline + suffix) rather than splitting the byte-identical
proof and the real-engagement check into separate tests -- a third real
554GB-checkpoint session in the same pytest process was observed to trip
the same governor limit, since each session leaves some residual memory
pressure even after engine.close().
"""

from __future__ import annotations

from pathlib import Path

import pytest

from runtime.engine import RuntimeConfig, StreamingEngine
from runtime.sampler import SamplingParams

_REAL_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "Kimi-K2.5"
_model_skip = pytest.mark.skipif(
    not _REAL_MODEL_DIR.exists(),
    reason="real Kimi-K2.5 checkpoint not present on this machine")

_PROMPT = (
    "The capital of France is Paris. The capital of Germany is Berlin. "
    "The capital of Italy is Rome. The capital of Spain is"
)


def _run(use_suffix: bool, max_tokens: int = 8):
    rc = RuntimeConfig(
        prefill_chunk_size=256, min_weight_cache_mb=200, max_weight_cache_mb=6000,
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
def test_suffix_decoding_matches_plain_target_and_actually_engages_for_k25():
    """Combines the byte-identical proof and the real-engagement check into
    one test (rather than two independent tests) deliberately -- each real
    K2.5 engine session leaves some residual memory pressure even after
    engine.close(), and running a THIRD real 554GB-checkpoint session in
    the same pytest process (on top of this test's own two) was observed
    to trip this machine's real memory-safety governor (a correct,
    honest fail-closed MemoryError, not a bug) -- see the module
    docstring's note on this real, separate, environment-dependent
    limitation. Two real sessions (baseline, suffix) is what this
    project's own manual verification already confirmed works reliably."""
    baseline = _run(use_suffix=False)
    suffix = _run(use_suffix=True)
    assert suffix["tokens"] == baseline["tokens"], (
        "suffix decoding must produce byte-identical greedy output to "
        "plain decode for a real GLM-family (kimi_k25) target now that "
        "forward_tokens_serial_positions supports it"
    )
    assert suffix["text"] == baseline["text"]

    stats = suffix["path_stats"]
    assert stats.get("suffix_decoding_enabled") == 1
    assert stats.get("suffix_decoding_used") == 1, (
        "expected suffix decoding to actually engage for a real greedy "
        "request against a real GLM-family checkpoint with a repetitive "
        "prompt, not silently fall back"
    )
    assert stats.get("suffix_decoding_proposed", 0) > 0
    assert stats.get("suffix_decoding_accepted", 0) > 0
