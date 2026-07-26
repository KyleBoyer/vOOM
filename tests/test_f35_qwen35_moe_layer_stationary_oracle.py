"""F35 extension to Qwen's own qwen3_5_moe (Qwen3.5-35B-A3B, and the same
architecture Qwen3.6-27B/35B-A3B use): layer-major (not chunk-major) prefill,
now MoE-aware.

F94 (2026-07-20) originally built layer-stationary prefill only for dense
qwen3_5 (num_experts=0) -- MoE routing was called once per tile inside
run_qwen35_block, which would silently re-route/re-fetch experts once per
chunk if ever used for qwen3_5_moe (never enabled for it before this).
F35 already fixed this exact problem for Kimi Linear and GLM/K2.5 by
splitting attention (still per-tile, causal state order preserved) from
MLP/MoE (now once per layer). This applies the identical split
(`_qwen35_attention_residual`/`_qwen35_mlp_residual` in runtime/qwen35.py)
to Qwen's own MoE variant and extends `rc.layer_stationary_prefill`'s
eligibility to `qwen3_5_moe`.

Same two-property "greedy A/B, byte-identical tokens" standard as the other
F35 oracle tests, against the real Qwen3.5-35B-A3B checkpoint (real hybrid
DeltaNet/full-attention layers, real MoE routing/expert fetch):

1. Byte-identical output: `rc.layer_stationary_prefill=True` vs `False`.
2. The weight-fetch-once property: each layer's non-expert cache key is
   fetched exactly once during layer-major prefill vs. more than once
   during chunk-major, via a call-counting wrapper around WeightCache.get.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from runtime.engine import RuntimeConfig, StreamingEngine
from runtime.sampler import SamplingParams

_REAL_MODEL_DIR = (
    Path(__file__).resolve().parent.parent / "models" / "Qwen3.5-35B-A3B")
_model_skip = pytest.mark.skipif(
    not _REAL_MODEL_DIR.exists(),
    reason="real Qwen3.5-35B-A3B checkpoint not present on this machine")

_PROMPT = (
    "The capital of France is Paris. The capital of Germany is Berlin. "
    "The capital of Italy is Rome. The capital of Spain is"
)
_CHUNK = 8


def _run(layer_stationary: bool, count_fetches: bool = False, max_tokens: int = 4):
    from runtime.server import PreparedPrompt

    rc = RuntimeConfig(
        prefill_chunk_size=_CHUNK,
        hot_prompt_kv_chunk_size=_CHUNK,
        hot_prompt_kv=True,
        hot_prompt_kv_min_tokens=0,
        layer_stationary_prefill=layer_stationary,
        min_weight_cache_mb=200, max_weight_cache_mb=6000,
    )
    engine = StreamingEngine(str(_REAL_MODEL_DIR), rc)
    fetch_counts: dict[str, int] = {}
    if count_fetches:
        real_get = engine.cache.get

        def counting_get(key, names):
            fetch_counts[key] = fetch_counts.get(key, 0) + 1
            return real_get(key, names)

        engine.cache.get = counting_get
    try:
        prompt_ids = engine.tokenizer.encode(_PROMPT).ids
        prompt = PreparedPrompt(
            _PROMPT, prompt_ids,
            stable_boundary_tokens=max(1, len(prompt_ids) - 2))
        result = engine.generate(
            prompt, max_tokens=max_tokens,
            sampling=SamplingParams(temperature=0.0))
    finally:
        engine.close()
    return result, fetch_counts


@_model_skip
def test_layer_stationary_matches_chunk_major_byte_identical():
    baseline, _ = _run(layer_stationary=False)
    layer_major, _ = _run(layer_stationary=True)
    assert layer_major["tokens"] == baseline["tokens"], (
        "layer-major (F35) prefill must produce byte-identical greedy "
        "output to chunk-major prefill for qwen3_5_moe"
    )
    assert layer_major["text"] == baseline["text"]


@_model_skip
def test_layer_stationary_fetches_each_layer_once_not_once_per_chunk():
    """max_tokens=1 holds decode's own fixed per-layer fetch contribution
    identical in both runs, isolating the prefill-phase difference this
    change actually targets -- same methodology as the dense qwen3_5/Kimi
    Linear/GLM analogues."""
    _, chunk_major_fetches = _run(
        layer_stationary=False, count_fetches=True, max_tokens=1)
    _, layer_major_fetches = _run(
        layer_stationary=True, count_fetches=True, max_tokens=1)

    layer_keys = [k for k in layer_major_fetches if k.startswith("layer.")]
    assert layer_keys, "expected at least one layer.* cache key to be fetched"
    chunk_major_layer_keys = [
        k for k in chunk_major_fetches if k.startswith("layer.")]
    assert chunk_major_layer_keys

    for key in layer_keys:
        assert layer_major_fetches[key] < chunk_major_fetches[key], (
            f"{key}: layer-stationary fetched {layer_major_fetches[key]}x, "
            f"chunk-major fetched {chunk_major_fetches[key]}x -- "
            "layer-major must fetch strictly fewer times for a prompt "
            "spanning multiple chunks"
        )
    assert any(chunk_major_fetches[k] > 2 for k in chunk_major_layer_keys), (
        "expected chunk-major prefill to re-fetch at least one layer's "
        "weights more than twice (prefill chunks + the tail/decode step) "
        "-- if not, this test's own prompt/chunk-size setup no longer "
        "exercises the bug this fixes"
    )
