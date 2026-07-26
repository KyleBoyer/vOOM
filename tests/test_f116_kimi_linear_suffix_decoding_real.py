"""F113 follow-on (2026-07-26, Kimi K3 readiness): suffix/prompt-lookup
speculative decoding (rc.suffix_decoding) now works correctly for
kimi_linear targets too -- the same fork/restore fix built for dense
qwen3_5 (F114) generalizes to Kimi Linear's own KDA+MLA hybrid layout
without any kimi_linear-specific round-loop code, since
run_shared_prefill_suffix_decode's fork/restore only ever checked for
`kv.kda_cache` presence, never branched on model_type.

Background: forward_tokens_serial_positions gained a real kimi_family
per-position dispatch, reusing _kimi_linear_attention_residual/
_kimi_linear_mlp_residual (already split out for F35-prep) which
internally dispatch KDA vs MLA per layer via cfg.kda_layers/
full_attn_layers -- verified byte-identical (logits AND kda_cache state,
all 20 KDA layers) against the real Kimi-Linear-48B-A3B-Instruct
checkpoint via a standalone diagnostic. fallback_reason()'s
"non-dense-target" and "recurrent-state-target" checks were both
relaxed for kimi_linear specifically (stronger evidence than
qwen3_5_moe currently has: the diagnostic exercised the real MoE
routing directly, not just dense/full-attention layers, and still
matched byte-identical).

Real A/B against the real 98GB checkpoint, repetitive prompt (to force
genuine partial rejects, exercising the fork/restore refeed path) --
one consolidated test (two real engine sessions), mirroring
test_f113_glm_suffix_decoding_real.py's and
test_f114_qwen35_suffix_decoding_real.py's own consolidation rationale
for large real-model checkpoints in one pytest process.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from runtime.engine import RuntimeConfig, StreamingEngine
from runtime.sampler import SamplingParams

_REAL_MODEL_DIR = (
    Path(__file__).resolve().parent.parent / "models" / "Kimi-Linear-48B-A3B-Instruct"
)
_model_skip = pytest.mark.skipif(
    not _REAL_MODEL_DIR.exists(),
    reason="real Kimi-Linear-48B-A3B-Instruct checkpoint not present on this machine")

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
def test_suffix_decoding_matches_plain_target_and_actually_engages_for_kimi_linear():
    """Two real Kimi-Linear-48B sessions (baseline, suffix) in one test,
    mirroring this project's own established consolidation rationale for
    real, large checkpoints -- keep real-model sessions per pytest
    process low."""
    baseline = _run(use_suffix=False)
    suffix = _run(use_suffix=True)
    assert suffix["tokens"] == baseline["tokens"], (
        "suffix decoding must produce byte-identical greedy output to "
        "plain decode for a real kimi_linear target now that its "
        "kda_cache is fork/restore-safe across speculative rounds"
    )
    assert suffix["text"] == baseline["text"]

    stats = suffix["path_stats"]
    assert stats.get("suffix_decoding_enabled") == 1
    assert stats.get("suffix_decoding_used") == 1, (
        "expected suffix decoding to actually engage for a real greedy "
        "request against a real kimi_linear checkpoint with a "
        "repetitive prompt, not silently fall back"
    )
    assert stats.get("suffix_decoding_proposed", 0) > 0
    assert stats.get("suffix_decoding_accepted", 0) > 0
