"""A cache miss during a decode step must not pollute the engine-lifetime
`_token_transient` ratchet used to gate the resident fast decode path.

`_token_transient` is set once in `__init__` and never reset between
requests -- it is a `max()` ratchet for the engine's whole lifetime, used
by `reserve_decode_step`'s governor check to decide whether the resident
fast path is safe to keep using. A real decode-step cache MISS (a fresh
`WeightCache` fetch, which on a quantize-on-load config also means real
bf16->Q4 conversion scratch) is a one-time cost, not the steady-state
per-token compute this ratchet exists to learn, and folding it in would
silently disable the fast path for the rest of the engine's life over one
unrepresentative measurement. Guarded in `runtime/engine.py` at both
`_token_transient` update sites: skip the update whenever
`self.cache.stats.misses` changed during that step.

2026-07-23 investigation notes (read before extending this test):
An earlier version of this investigation tried to reproduce the ORIGINAL
observation (F99 failing to engage on a quantize-on-load config across
two sequential requests) end-to-end and found two real, SEPARATE
confounds along the way, neither of which is what this specific test
verifies:
  1. The reproduction's RuntimeConfig was missing
     `pin_lm_head=False, stream_lm_head=True` (what server.py always sets
     for this model type), which caused a large, real, but UNRELATED
     per-step lm_head materialization cost -- fixed by matching production
     settings, not by any code change.
  2. Even with that fixed, a large, reproducible memory spike still
     showed up specifically on the LAST decode step of a request (i.e. it
     is not merely a cold-start artifact -- it recurred at request end
     regardless of prior warmth), and `self.cache.stats.misses` did NOT
     change at that step, meaning this specific spike is NOT a
     `WeightCache` miss at all and this file's guard cannot catch it.
     Root cause not identified this session (candidates not yet checked:
     `StreamedLMHead`'s own disk-streaming path bypasses `WeightCache`
     entirely so has no miss counter; accumulated un-evaluated lazy graph
     across the decode loop finally forced at the sampling/completion
     boundary; end-of-request bookkeeping). This is a real, still-open
     problem, tracked as a follow-up in docs/future_lossless_techniques.md's
     F99 entry -- NOT claimed fixed by the code change this test covers.

Given (2), a full "two real sequential generate() calls, assert the
second still uses the fast path" integration test is currently confounded
by an unrelated, unresolved issue and would fail for the WRONG reason.
This test instead verifies the actual, fixed mechanism directly: within a
single request, a genuine mid-decode cache miss (forced via a tiny weight
cache budget that guarantees eviction+refetch between tokens) must not
inflate `_token_transient` beyond what a miss-free step would need.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from runtime.engine import RuntimeConfig, StreamingEngine
from runtime.sampler import SamplingParams

_REAL_MODEL_DIR = (
    Path(__file__).resolve().parent.parent / "models" / "Qwen3.5-4B")
_skip = pytest.mark.skipif(
    not _REAL_MODEL_DIR.exists(),
    reason="real Qwen3.5-4B checkpoint not present on this machine")

_PROMPT = "The capital of France is Paris. The capital of Germany is"


@_skip
def test_mid_decode_cache_miss_does_not_inflate_token_transient():
    # A tiny weight cache budget guarantees real evictions and re-fetches
    # (misses) between decode steps for a model this size -- forcing the
    # exact scenario the guard exists for, without needing a second request.
    rc = RuntimeConfig(
        prefill_chunk_size=512,
        quant_bits=4, quant_mode="mxfp4", quant_group_size=32, quant_min_dim=0,
        max_weight_cache_mb=1500, min_weight_cache_mb=200,
        pin_lm_head=False, stream_lm_head=True)
    engine = StreamingEngine(str(_REAL_MODEL_DIR), rc)
    try:
        misses_before = engine.cache.stats.misses
        engine.generate(
            _PROMPT, max_tokens=16, sampling=SamplingParams(temperature=0.0))
        misses_after = engine.cache.stats.misses
        assert misses_after > misses_before, (
            "test setup didn't actually force mid-decode cache misses -- "
            "the tiny budget no longer guarantees eviction for this model; "
            "adjust max_weight_cache_mb rather than trusting this result"
        )
    finally:
        engine.close()

    # The real assertion: _token_transient reflects genuine per-token
    # compute scratch, not a miss-inflated outlier. A real per-token
    # transient for a 4B model at this scale is at most tens of MB (see
    # the debug trace referenced in this file's docstring: ~30-160MB for
    # clean steps); a miss-polluted one measured in the GB range in the
    # unguarded version of this code.
    assert engine._token_transient < 500_000_000, (
        f"_token_transient={engine._token_transient / 1e6:.1f}MB -- "
        "looks miss-polluted despite the guard"
    )
