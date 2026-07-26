"""F113 follow-on (2026-07-26): suffix/prompt-lookup speculative decoding
now works correctly for qwen3_5_moe targets too, closing the gap
F114/F116's own docs flagged as open ("qwen3_5_moe is NOT exempted...
per-position MoE routing safety for Qwen hasn't been separately
verified the way GLM's was").

A standalone diagnostic against the real Qwen3.5-35B-A3B-mlx-expert-
mxfp4 checkpoint (256 experts) exercised the actual MoE routing
(_route_experts / _qwen35_mlp_residual's MoE branch) through
forward_tokens_serial_positions' existing qwen_family dispatch and came
back byte-identical (0.0 max abs diff, logits AND all 30 kda_cache
layers) vs true sequential decode -- the same standard kimi_linear's own
diagnostic met (F116). fallback_reason()'s "non-dense-target" and
"recurrent-state-target" checks were both relaxed for qwen3_5_moe
specifically. No round-loop changes needed -- fork/restore only ever
checks kv.kda_cache presence, never model_type.

Real A/B against the real checkpoint, repetitive prompt (to force
genuine partial rejects). One consolidated test (two real sessions),
matching this project's own large real-model-checkpoint testing
convention.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from runtime.engine import RuntimeConfig, StreamingEngine
from runtime.sampler import SamplingParams

_REAL_MODEL_DIR = (
    Path(__file__).resolve().parent.parent
    / "models" / "Qwen3.5-35B-A3B-mlx-expert-mxfp4"
)
_model_skip = pytest.mark.skipif(
    not _REAL_MODEL_DIR.exists(),
    reason="real Qwen3.5-35B-A3B-mlx-expert-mxfp4 checkpoint not present on this machine")

_PROMPT = (
    "The capital of France is Paris. The capital of Germany is Berlin. "
    "The capital of Italy is Rome. The capital of Spain is Madrid. "
    "The capital of Portugal is"
)


def _run(use_suffix: bool, max_tokens: int = 16):
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
def test_suffix_decoding_matches_plain_target_and_actually_engages_for_qwen35_moe():
    """Two real Qwen3.5-35B-A3B sessions (baseline, suffix) in one test."""
    baseline = _run(use_suffix=False)
    suffix = _run(use_suffix=True)
    assert suffix["tokens"] == baseline["tokens"], (
        "suffix decoding must produce byte-identical greedy output to "
        "plain decode for a real qwen3_5_moe target now that its "
        "kda_cache is fork/restore-safe across speculative rounds"
    )
    assert suffix["text"] == baseline["text"]

    stats = suffix["path_stats"]
    assert stats.get("suffix_decoding_enabled") == 1
    assert stats.get("suffix_decoding_used") == 1, (
        "expected suffix decoding to actually engage for a real greedy "
        "request against a real qwen3_5_moe checkpoint with a repetitive "
        "prompt, not silently fall back"
    )
    assert stats.get("suffix_decoding_proposed", 0) > 0
    assert stats.get("suffix_decoding_accepted", 0) > 0
