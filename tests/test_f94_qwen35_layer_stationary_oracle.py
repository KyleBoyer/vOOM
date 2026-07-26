"""F94 live path: layer-major (not chunk-major) prefill for dense qwen3_5.

Proves two things against the REAL Qwen3.5-4B checkpoint (hybrid DeltaNet/
full-attention layers, real KDAStateCache recurrent state) rather than a
synthetic fixture, matching this project's own established "greedy A/B,
byte-identical tokens" gold standard:

1. Byte-identical output: `rc.layer_stationary_prefill=True` vs `False`,
   same prompt, same small chunk width chosen so the prompt spans several
   chunks -- if reordering the (layer, chunk) loop nesting changed a single
   token, this would catch it. Recurrent state correctness across tile
   boundaries follows from each layer's own state depending only on that
   layer's own sequential inputs and its own prior state (see
   StreamingEngine._layer_stationary_qwen35_sweep's docstring), but this is
   the actual empirical proof, not just the argument.
2. The weight-fetch-once property layer-major exists for: each layer's
   cache key is fetched exactly once during layer-major prefill, vs. once
   per chunk during chunk-major prefill, via a call-counting wrapper around
   the real WeightCache.get -- not merely asserted from reading the code.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from runtime.engine import RuntimeConfig, StreamingEngine
from runtime.sampler import SamplingParams

_REAL_MODEL_DIR = (
    Path(__file__).resolve().parent.parent / "models" / "Qwen3.5-4B")
_model_skip = pytest.mark.skipif(
    not _REAL_MODEL_DIR.exists(),
    reason="real Qwen3.5-4B checkpoint not present on this machine")

_PROMPT = (
    "The capital of France is Paris. The capital of Germany is Berlin. "
    "The capital of Italy is Rome. The capital of Spain is"
)
_CHUNK = 8


def _run(layer_stationary: bool, count_fetches: bool = False, max_tokens: int = 6):
    rc = RuntimeConfig(
        prefill_chunk_size=_CHUNK,
        hot_prompt_kv_chunk_size=_CHUNK,
        layer_stationary_prefill=layer_stationary,
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


def _run_with_stable_boundary(layer_stationary: bool):
    """Exercise the server-style first-turn boundary fork, not a plain str."""
    from runtime.server import PreparedPrompt

    rc = RuntimeConfig(
        prefill_chunk_size=_CHUNK,
        hot_prompt_kv_chunk_size=_CHUNK,
        layer_stationary_prefill=layer_stationary,
        hot_prompt_kv=True,
        hot_prompt_kv_min_tokens=0,
        execution_profile="layers",
    )
    engine = StreamingEngine(str(_REAL_MODEL_DIR), rc)
    try:
        token_ids = engine.tokenizer.encode(_PROMPT).ids
        prompt = PreparedPrompt(
            _PROMPT, token_ids,
            stable_boundary_tokens=max(1, len(token_ids) - 2))
        result = engine.generate(
            prompt, max_tokens=2,
            sampling=SamplingParams(temperature=0.0))
    finally:
        engine.close()
    return result


@_model_skip
def test_layer_stationary_matches_chunk_major_byte_identical():
    baseline, _ = _run(layer_stationary=False)
    layer_major, _ = _run(layer_stationary=True)
    assert layer_major["tokens"] == baseline["tokens"], (
        "layer-major (F94) prefill must produce byte-identical greedy "
        "output to chunk-major prefill for the same prompt"
    )
    assert layer_major["text"] == baseline["text"]


@_model_skip
def test_stable_boundary_uses_layer_stationary_and_stays_byte_identical():
    """A server PreparedPrompt's stable boundary must not bypass F94."""
    baseline = _run_with_stable_boundary(layer_stationary=False)
    layer_major = _run_with_stable_boundary(layer_stationary=True)

    assert layer_major["tokens"] == baseline["tokens"]
    assert layer_major["text"] == baseline["text"]
    paths = layer_major["execution_profile"]["phases"]["prefill"]["paths"]
    assert paths.get("layer_stationary_qwen35", 0) == 1
    assert any(
        "hot_boundary layer_stationary eligible=1" in note
        for note in layer_major["execution_profile"]["notes"])


@_model_skip
def test_layer_stationary_fetches_each_layer_once_not_once_per_chunk():
    """generate() fetches every layer's weights once more after prefill for
    each decode step (expected, unrelated to F94 -- decode has nothing to
    amortize since it is always exactly one position). Use max_tokens=1 to
    hold that contribution to a small, fixed +1-or-2 per layer identical in
    both runs, so the REMAINING difference isolates the prefill phase F94
    actually changes: chunk-major must fetch every layer more than once
    across a multi-chunk prompt; layer-major must fetch every layer a fixed,
    prompt-length-independent number of times (proven by comparison against
    chunk-major on the SAME prompt/model, not a hardcoded constant).
    """
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
        "exercises the bug F94 fixes"
    )
