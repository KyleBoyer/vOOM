"""F35 extension: layer-major (not chunk-major) prefill for GLM's MLA+MoE
block shape (glm_moe_dsa / kimi_k25 / glm4_moe_lite), the same technique
tests/test_f35_kimi_linear_layer_stationary_oracle.py proves for Kimi
Linear's KDA/MLA hybrid, and tests/test_f94_qwen35_layer_stationary_oracle.py
proves for dense qwen3_5.

Kimi K2.5 (real, locally downloaded, model_type="kimi_k25") shares GLM-5.2's
exact block math (real q_lora MLA, noaux_tc MoE -- see runtime/engine.py's
own comment at the run_glm_block dispatch site and CLAUDE.md's Goal 3 notes)
and is the only real, complete, lossless checkpoint of this shape available
on this machine -- GLM-5.2 itself is 1.49TB bf16, far beyond what's stored
here (only a quantized GLM-5.2-q4e side-quest artifact exists locally, which
is out of scope for a lossless correctness oracle). Proving the technique
against K2.5's real weights validates the same runtime/glm.py code path
GLM-5.2 itself would use.

Same two properties as the Kimi Linear test, same "greedy A/B, byte-identical
tokens" standard:

1. Byte-identical output: rc.layer_stationary_prefill=True vs False, same
   prompt, a small chunk width so the prompt spans several chunks.
2. The weight-fetch-once property: each layer's non-expert cache key is
   fetched exactly once during layer-major prefill vs. more than once during
   chunk-major, via a call-counting wrapper around the real WeightCache.get.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from runtime.engine import RuntimeConfig, StreamingEngine
from runtime.sampler import SamplingParams

_REAL_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "Kimi-K2.5"
_model_skip = pytest.mark.skipif(
    not (_REAL_MODEL_DIR / "modeling_deepseek.py").exists(),
    reason="real Kimi-K2.5 checkpoint not present on this machine",
)

_PROMPT = (
    "The capital of France is Paris. The capital of Germany is Berlin. "
    "The capital of Italy is Rome. The capital of Spain is"
)
_CHUNK = 8


def _run(layer_stationary: bool, count_fetches: bool = False, max_tokens: int = 4):
    rc = RuntimeConfig(
        prefill_chunk_size=_CHUNK,
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
        result = engine.generate(
            _PROMPT, max_tokens=max_tokens,
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
        "output to chunk-major prefill for the same prompt"
    )
    assert layer_major["text"] == baseline["text"]


@_model_skip
def test_layer_stationary_fetches_each_layer_once_not_once_per_chunk():
    """max_tokens=1 holds decode's own fixed per-layer fetch contribution
    identical in both runs, isolating the prefill-phase difference this
    change actually targets -- same methodology as the qwen3_5/Kimi Linear
    analogues."""
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
