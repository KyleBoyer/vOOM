"""Exact disk persistence for Qwen3.5/3.6 hybrid stable prefixes.

These tests use tiny MLX tensors but the production cache layout: conventional
full-attention K/V on those layers plus DeltaNet matrix and convolution history
on every linear-attention layer.  No checkpoint weights are loaded.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from runtime.hot_kv_persist import HotPromptKVPersistence
from runtime.kda_state import KDAStateCache
from runtime.kv_cache import KVCache


_LAYER_TYPES = (
    "linear_attention",
    "full_attention",
    "linear_attention",
    "full_attention",
)
_BOUNDARIES = (1, 31, 32, 33, 64, 65)


def _config(model_type: str = "qwen3_5"):
    return SimpleNamespace(
        model_type=model_type,
        layer_types=_LAYER_TYPES,
        kda_layers=(),
    )


def _journal(
    path,
    fingerprint="qwen-hybrid-test",
    *,
    limit=64,
    model_type="qwen3_5",
):
    return HotPromptKVPersistence(
        path,
        fingerprint,
        32,
        max_checkpoints=limit,
        config=_config(model_type),
        require_recurrent=True,
    )


def _hybrid_state(length: int) -> KVCache:
    kv = KVCache(len(_LAYER_TYPES))
    for layer in (1, 3):
        # Position-major construction keeps every shorter state's KV a true
        # byte-for-byte prefix of a longer state's KV on both heads.
        base = mx.transpose(
            mx.arange(length * 6, dtype=mx.float32).reshape(length, 2, 3),
            (1, 0, 2),
        )[None, ...]
        kv.keys[layer] = base + layer * 1000
        kv.values[layer] = base + layer * 1000 + 500

    recurrent = KDAStateCache(len(_LAYER_TYPES))
    for layer in (0, 2):
        recurrent.set_state(
            layer,
            mx.arange(24, dtype=mx.float32).reshape(1, 2, 3, 4)
            + length * 100 + layer,
        )
        recurrent.set_conv_history(
            layer,
            (mx.arange(12, dtype=mx.float32).reshape(1, 2, 2, 3)
             + length * 10 + layer,),
        )
    kv.kda_cache = recurrent
    recurrent.synchronize()
    mx.eval(*[
        value for value in (*kv.keys, *kv.values) if value is not None
    ])
    return kv


def _save_stable(journal, tokens, kv, *, parent=(), covered=0):
    return journal.save(
        tuple(parent),
        covered,
        tokens,
        kv,
        logits=None,
        prompt_logits=None,
        prompt_length=len(tokens),
        reusable_prefix=len(tokens),
        checkpoint_kind="stable_prefix",
    )


def _assert_hybrid_equal(actual: KVCache, expected: KVCache) -> None:
    assert actual.offset == expected.offset
    for layer in (1, 3):
        assert np.array_equal(
            np.array(actual.keys[layer]), np.array(expected.keys[layer]))
        assert np.array_equal(
            np.array(actual.values[layer]), np.array(expected.values[layer]))
    for layer in (0, 2):
        assert np.array_equal(
            np.array(actual.kda_cache.state(layer)),
            np.array(expected.kda_cache.state(layer)),
        )
        assert np.array_equal(
            np.array(actual.kda_cache.conv_history(layer)[0]),
            np.array(expected.kda_cache.conv_history(layer)[0]),
        )


@pytest.mark.parametrize("length", _BOUNDARIES)
@pytest.mark.parametrize("model_type", ("qwen3_5", "qwen3_5_moe"))
def test_stable_prefix_round_trip_is_exact_at_block_boundaries(
        tmp_path, length, model_type):
    journal = _journal(tmp_path, model_type=model_type)
    tokens = list(range(length))
    expected = _hybrid_state(length)

    chain = _save_stable(journal, tokens, expected)

    assert len(chain) == math.ceil(length / 32)
    # A state-only stable prefix must never masquerade as an exact-logit hit.
    assert journal.find_best_match(tokens, 32) is None
    match = journal.find_best_match(tokens + [1000, 1001], 32)
    assert match is not None
    assert match["checkpoint_kind"] == "stable_prefix"
    assert match["case"] == "extension"
    assert match["matched"] == length
    loaded = journal.load_matched_chain(match, len(_LAYER_TYPES))
    assert loaded is not None
    loaded_tokens, actual, exact_logits = loaded
    assert loaded_tokens == tuple(tokens)
    assert exact_logits is None
    _assert_hybrid_equal(actual, expected)


@pytest.mark.parametrize("model_type", ("qwen3_5", "qwen3_5_moe"))
def test_restart_selects_longest_exact_stable_prefix(tmp_path, model_type):
    writer = _journal(tmp_path, model_type=model_type)
    chain = ()
    for length in (32, 64, 96):
        tokens = list(range(length))
        chain = _save_stable(
            writer, tokens, _hybrid_state(length),
            parent=chain, covered=length - 32)

    # A new journal object has no process-local segment index or tensor state.
    reader = _journal(tmp_path, model_type=model_type)
    query = list(range(96)) + [200, 201]
    match = reader.find_best_match(query, 32)
    assert match is not None
    assert match["matched"] == 96
    assert match["checkpoint_kind"] == "stable_prefix"

    loaded = reader.load_matched_chain(match, len(_LAYER_TYPES))
    assert loaded is not None
    assert loaded[0] == tuple(range(96))
    _assert_hybrid_equal(loaded[1], _hybrid_state(96))
    assert reader.find_best_match(
        query, 32, min_matched_exclusive=96,
    ) is None

    # Make the shortest branch the most recently used on disk. Restart preload
    # still keeps the longest stable prefix, so a touched ancestor cannot hide
    # its more useful descendant in a one-slot RAM tier.
    short_query = list(range(32)) + [8000]
    short_match = reader.find_best_match(short_query, 32)
    assert short_match is not None and short_match["matched"] == 32
    assert reader.load_matched_chain(
        short_match, len(_LAYER_TYPES),
    ) is not None

    # Restart preload prioritizes the useful stable boundary over a newer full
    # endpoint when the bounded RAM tier has only one slot.
    logits = mx.array([[1.0, 2.0, 3.0]], dtype=mx.float32)
    endpoint_tokens = list(range(96)) + [999]
    endpoint_chain = writer.save(
        chain, 96, endpoint_tokens, _hybrid_state(97), logits, logits,
        prompt_length=97, reusable_prefix=96)
    assert endpoint_chain[:len(chain)] == chain
    assert len(endpoint_chain) == len(chain) + 1
    preloaded = _journal(
        tmp_path, model_type=model_type,
    ).load_all(len(_LAYER_TYPES), limit=1)
    assert len(preloaded) == 1
    assert preloaded[0][0] == tuple(range(96))
    assert preloaded[0][2] is None
    assert preloaded[0][4] is None


def test_corrupt_stale_and_incomplete_hybrid_snapshots_fail_closed(tmp_path):
    tokens = list(range(33))
    writer = _journal(tmp_path)
    _save_stable(writer, tokens, _hybrid_state(len(tokens)))
    checkpoint = next(tmp_path.glob("*.ckpt.json"))
    checkpoint_id = checkpoint.name[:-len(".ckpt.json")]
    payload = writer._checkpoint_payload_path(checkpoint_id)

    damaged = bytearray(payload.read_bytes())
    damaged[len(damaged) // 2] ^= 1
    payload.write_bytes(damaged)
    assert _journal(tmp_path).find_best_match(tokens + [90], 32) is None

    # The same files under a different model/runtime/config fingerprint are
    # stale data, not a candidate for best-effort loading.
    assert _journal(
        tmp_path, fingerprint="different-model-runtime-config",
    ).find_best_match(tokens + [90], 32) is None

    incomplete_dir = tmp_path / "incomplete"
    incomplete = _journal(incomplete_dir)
    kv = _hybrid_state(len(tokens))
    kv.keys[3] = None
    kv.values[3] = None
    with pytest.raises(ValueError, match="full-attention KV"):
        _save_stable(incomplete, tokens, kv)


def test_endpoint_identity_and_logits_remain_content_addressed(tmp_path):
    journal = _journal(tmp_path)
    tokens = list(range(33))
    kv = _hybrid_state(len(tokens))
    logits = mx.array([[1.25, -2.0, 7.5]], dtype=mx.float32)

    first_chain = journal.save(
        (), 0, tokens, kv, logits, logits,
        prompt_length=len(tokens), reusable_prefix=len(tokens))
    first_ids = {
        path.name[:-len(".ckpt.json")]
        for path in tmp_path.glob("*.ckpt.json")
    }
    second_chain = journal.save(
        (), 0, tokens, kv, logits, logits,
        prompt_length=len(tokens), reusable_prefix=len(tokens))
    second_ids = {
        path.name[:-len(".ckpt.json")]
        for path in tmp_path.glob("*.ckpt.json")
    }
    assert second_chain == first_chain
    assert second_ids == first_ids

    match = journal.find_best_match(tokens, 32)
    assert match is not None
    assert match["checkpoint_kind"] == "endpoint"
    loaded = journal.load_matched_chain(match, len(_LAYER_TYPES))
    assert loaded is not None
    assert np.array_equal(np.array(loaded[2]), np.array(logits))

    changed_tokens = tokens[:-1] + [5000]
    journal.save(
        (), 0, changed_tokens, kv, logits, logits,
        prompt_length=len(changed_tokens), reusable_prefix=len(changed_tokens))
    changed_ids = {
        path.name[:-len(".ckpt.json")]
        for path in tmp_path.glob("*.ckpt.json")
    }
    assert len(changed_ids) == len(first_ids) + 1
    manifests = [json.loads(path.read_text())
                 for path in tmp_path.glob("*.ckpt.json")]
    assert all(item["checkpoint_kind"] == "endpoint" for item in manifests)
