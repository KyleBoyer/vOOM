"""Hybrid resident fast decode for dense qwen3_5 (F99): byte-identical A/B.

The ordinary decode loop pays ~56 GPU sync points per token for this hybrid
(32 per-layer mx.eval(x) + 24 per-DeltaNet-layer mx.eval(state)). The
resident fast branch in StreamingEngine._sweep runs every layer lazily
(defer_state_eval) and batch-evals all updated recurrent state in ONE call
at the sweep boundary -- identical arithmetic, different eval boundaries
only. This test proves byte-identical greedy output against the real
Qwen3.5-4B checkpoint and that the fast path actually engaged (fast-sweep
counter), skipping with a clear message if live memory pressure forced the
graceful streamed fallback instead (a legitimate runtime outcome, but one
under which this test would not be exercising the code it exists for).

Also a regression anchor for the latent crash this work fixed: the fast
branch used to call layer_runner.run_block unconditionally -- a plain
dense-transformer block that looks up self_attn.* tensor names
linear_attention layers do not have -- and only never crashed because
server.py never set resident_fast_decode for qwen3_5.
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

_PROMPT = (
    "The capital of France is Paris. The capital of Germany is Berlin. "
    "The capital of Italy is Rome. The capital of Spain is"
)


def _run(fast: bool, max_tokens: int = 24):
    rc = RuntimeConfig(
        prefill_chunk_size=512,
        quant_bits=4, quant_mode="mxfp4", quant_group_size=32, quant_min_dim=0,
        max_weight_cache_mb=7000,
        resident_fast_decode=fast,
    )
    engine = StreamingEngine(str(_REAL_MODEL_DIR), rc)
    try:
        result = engine.generate(
            _PROMPT, max_tokens=max_tokens,
            sampling=SamplingParams(temperature=0.0))
        sweeps = engine._resident_fast_decode_sweeps
    finally:
        engine.close()
    return result, sweeps


@_skip
def test_hybrid_resident_fast_decode_byte_identical_and_engaged():
    baseline, _ = _run(fast=False)
    fast, fast_sweeps = _run(fast=True)
    assert fast["tokens"] == baseline["tokens"], (
        "hybrid resident fast decode must be byte-identical to the ordinary "
        "per-layer loop -- it only moves eval boundaries, never arithmetic"
    )
    if fast_sweeps == 0:
        if fast["path_stats"].get("resident_fast_memory_fallback"):
            pytest.skip(
                "live memory pressure forced the graceful streamed fallback; "
                "byte-identity still held but the fast path was not exercised")
        pytest.fail(
            "fast path never engaged and no memory fallback was recorded -- "
            "eligibility logic regressed")
