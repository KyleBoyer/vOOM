"""SQ26: zmlx's fused DeltaNet decode kernels, real greedy-token quality gate.

The earlier survey (docs/future_sidequest_techniques.md SQ26) found a real
bf16 precision difference in isolation (0.03125/0.0625 max abs diff vs this
project's own float32-accumulated `_causal_depthwise_conv1d`/
`_silu_gated_rms_norm`) but explicitly deferred the decisive question: does
that per-call noise ever change an actual greedy-generated token? This test
answers it directly against the real Qwen3.5-4B checkpoint, matching this
project's own "greedy A/B, byte-identical tokens" standard everywhere else.
"""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    import zmlx  # noqa: F401
    _ZMLX_AVAILABLE = True
except ImportError:
    _ZMLX_AVAILABLE = False

from runtime.engine import RuntimeConfig, StreamingEngine
from runtime.sampler import SamplingParams

_REAL_MODEL_DIR = (
    Path(__file__).resolve().parent.parent / "models" / "Qwen3.5-4B")
_skip = pytest.mark.skipif(
    not _REAL_MODEL_DIR.exists() or not _ZMLX_AVAILABLE,
    reason="real Qwen3.5-4B checkpoint and/or zmlx package not present")

_PROMPT = (
    "The capital of France is Paris. The capital of Germany is Berlin. "
    "The capital of Italy is Rome. The capital of Spain is"
)


def _run(zmlx_fused: bool, max_tokens: int = 24):
    rc = RuntimeConfig(
        prefill_chunk_size=512, zmlx_fused_deltanet_decode=zmlx_fused)
    engine = StreamingEngine(str(_REAL_MODEL_DIR), rc)
    try:
        return engine.generate(
            _PROMPT, max_tokens=max_tokens,
            sampling=SamplingParams(temperature=0.0))
    finally:
        engine.close()


@_skip
def test_zmlx_fused_decode_matches_baseline_byte_identical():
    baseline = _run(zmlx_fused=False)
    fused = _run(zmlx_fused=True)
    assert fused["tokens"] == baseline["tokens"], (
        "zmlx's fused DeltaNet decode kernels must produce byte-identical "
        "greedy output to the existing float32-accumulated implementation "
        "-- a real per-call bf16 precision difference exists (SQ26) but "
        "must not change any emitted token"
    )
    assert fused["text"] == baseline["text"]
