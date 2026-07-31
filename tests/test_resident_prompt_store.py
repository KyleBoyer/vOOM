"""Durability and identity gates for exact resident MLX prompt endpoints."""

from __future__ import annotations

import json

import mlx.core as mx

from runtime.resident_prompt_store import ResidentPromptStore


def _endpoint():
    from runtime.resident_mlx_lm import import_mlx_lm
    import_mlx_lm()
    from mlx_lm.models.cache import ArraysCache, KVCache

    recurrent = ArraysCache(2)
    recurrent.cache = [
        mx.arange(12, dtype=mx.float16).reshape(1, 3, 4),
        mx.ones((1, 2, 2), dtype=mx.float32),
    ]
    attention = KVCache()
    attention.state = (
        mx.arange(24, dtype=mx.float16).reshape(1, 2, 3, 4),
        mx.arange(24, dtype=mx.float16).reshape(1, 2, 3, 4) + 1,
    )
    logits = mx.array([0.5, -1.0, 2.0], dtype=mx.float32)
    mx.eval(logits, recurrent.state, attention.state)
    return [recurrent, attention], logits


def _arrays(value):
    if isinstance(value, mx.array):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _arrays(item)


def test_round_trip_restores_mixed_recurrent_and_attention_state(tmp_path):
    store = ResidentPromptStore(tmp_path, "a" * 64, max_bytes=1_000_000)
    prompt = [1, 7, 9, 42]
    cache, logits = _endpoint()

    generation_tokens = [11, 12, 13]
    generation_logits = [
        logits,
        logits + 1,
        logits + 2,
    ]
    saved = store.save(
        prompt, cache, logits, generation_tokens, generation_logits)
    (
        loaded_cache,
        loaded_logits,
        loaded_tokens,
        loaded_generation_logits,
        loaded,
    ) = store.load(prompt)

    assert saved["saved"] == 1
    assert loaded["hit"] == 1
    assert len(loaded_cache) == len(cache)
    assert all(
        mx.array_equal(expected, actual).item()
        for original, restored in zip(cache, loaded_cache)
        for expected, actual in zip(
            _arrays(original.state), _arrays(restored.state))
    )
    assert mx.array_equal(logits, loaded_logits).item()
    assert loaded_tokens == tuple(generation_tokens)
    assert len(loaded_generation_logits) == len(generation_logits)
    assert all(
        mx.array_equal(expected, actual).item()
        for expected, actual in zip(
            generation_logits, loaded_generation_logits)
    )


def test_key_covers_every_token_and_runtime_fingerprint(tmp_path):
    first = ResidentPromptStore(tmp_path, "a" * 64)
    second = ResidentPromptStore(tmp_path, "b" * 64)

    assert first.key([1, 2, 3]) != first.key([1, 2, 4])
    assert first.key([1, 2, 3]) != first.key([1, 2, 3, 4])
    assert first.key([1, 2, 3]) != second.key([1, 2, 3])


def test_corrupt_payload_misses_and_next_save_repairs_entry(tmp_path):
    store = ResidentPromptStore(tmp_path, "c" * 64, max_bytes=1_000_000)
    prompt = [4, 3, 2, 1]
    cache, logits = _endpoint()
    saved = store.save(prompt, cache, logits)
    cache_path = store.directory / f"{saved['key']}.cache.safetensors"
    cache_path.write_bytes(b"torn")

    assert store.load(prompt) is None
    repaired = store.save(prompt, cache, logits)
    loaded = store.load(prompt)

    assert repaired["saved"] == 1
    assert loaded is not None
    assert loaded[4]["hit"] == 1


def test_invalid_manifest_is_never_visible(tmp_path):
    store = ResidentPromptStore(tmp_path, "d" * 64, max_bytes=1_000_000)
    prompt = [10, 20]
    key = store.key(prompt)
    manifest_path = store.directory / f"{key}.json"
    manifest_path.write_text(json.dumps({
        "format": "resident-mlx-prompt-v2",
        "fingerprint": store.fingerprint,
        "key": key,
        "prompt_tokens": len(prompt),
    }))

    assert store.load(prompt) is None


def test_byte_budget_is_global_across_fingerprints(tmp_path):
    first = ResidentPromptStore(tmp_path, "e" * 64, max_bytes=1_000_000)
    cache, logits = _endpoint()
    first_saved = first.save([1, 2, 3], cache, logits)
    one_entry_budget = (
        first_saved["cache_bytes"] + first_saved["logits_bytes"] + 1_000)

    second = ResidentPromptStore(
        tmp_path, "f" * 64, max_bytes=one_entry_budget)
    second.save([7, 8, 9], cache, logits)

    assert first.load([1, 2, 3]) is None
    assert second.load([7, 8, 9]) is not None
