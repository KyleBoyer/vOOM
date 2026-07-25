"""F35-prep: layer-major (not chunk-major) prefill for Kimi Linear, the MoE
analogue of F94's dense-only layer-stationary prefill.

Quantified motivation (docs/future_lossless_techniques.md F92, 2026-07-24):
a real per-layer routed-expert footprint check found each MoE layer's TRUE
unique-expert union across a whole 291-token prefill is only ~90-117/256,
but chunk-major's measured average weight-cache misses per layer was
~194 -- roughly double, the signature of re-routing/re-fetching
overlapping-but-not-identical expert sets once per chunk instead of once
per layer. This fixes it by construction (route once per layer over the
WHOLE prompt, not once per chunk) rather than any new "union of experts
across chunks" bookkeeping -- see `StreamingEngine.
_layer_stationary_kimi_linear_sweep`'s own docstring for the full
correctness argument.

Proves two things against the REAL Kimi-Linear-48B-A3B-Instruct checkpoint
(real KDA/MLA hybrid layers, real MoE routing), matching this project's own
"greedy A/B, byte-identical tokens" standard exactly like
tests/test_f94_qwen35_layer_stationary_oracle.py does for the dense case:

1. Byte-identical output: `rc.layer_stationary_prefill=True` vs `False`,
   same prompt, a small chunk width so the prompt spans several chunks --
   if reordering the (layer, chunk) loop nesting, or routing over the
   whole prompt instead of per-chunk, changed a single token, this would
   catch it.
2. The weight-fetch-once property: each layer's non-expert cache key is
   fetched exactly once during layer-major prefill vs. more than once
   during chunk-major, via a call-counting wrapper around the real
   WeightCache.get.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from runtime.engine import RuntimeConfig, StreamingEngine
from runtime.sampler import SamplingParams

_REAL_MODEL_DIR = (
    Path(__file__).resolve().parent.parent / "models" / "Kimi-Linear-48B-A3B-Instruct")
_model_skip = pytest.mark.skipif(
    not _REAL_MODEL_DIR.exists(),
    reason="real Kimi-Linear-48B-A3B-Instruct checkpoint not present on this machine")

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
        "layer-major (F35-prep) prefill must produce byte-identical greedy "
        "output to chunk-major prefill for the same prompt"
    )
    assert layer_major["text"] == baseline["text"]


@_model_skip
def test_layer_stationary_fetches_each_layer_once_not_once_per_chunk():
    """Same methodology as test_f94_qwen35_layer_stationary_oracle.py's
    analogous test: max_tokens=1 holds decode's own fixed per-layer fetch
    contribution identical in both runs, isolating the prefill-phase
    difference this change actually targets."""
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
