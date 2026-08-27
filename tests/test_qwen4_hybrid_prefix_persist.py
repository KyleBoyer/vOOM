"""Exact durable Qwen3.8-Flash-Next hybrid endpoint coverage."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from runtime.hot_kv_persist import HotPromptKVPersistence
from runtime.kda_state import KDAStateCache
from runtime.kv_cache import KVCache
from runtime.qwen4_exp_state import Qwen4ExpStateCache


_LAYER_TYPES = (
    "linear_attention", "full_attention",
    "linear_attention", "full_attention",
)


def _config():
    return SimpleNamespace(
        model_type="qwen4_exp",
        layer_types=_LAYER_TYPES,
        kda_layers=(),
        qwen4_ple_layers=(0,),
        qwen4_indexer_head_dim=4,
        qwen4_ngram_size=3,
        qwen4_ple_conv_kernel_size=4,
        qwen4_ple_embed_dim=8,
    )


def _journal(path):
    return HotPromptKVPersistence(
        path, "qwen4-hybrid-test", 32, max_checkpoints=4,
        config=_config(), require_recurrent=True)


def _state(length: int) -> KVCache:
    kv = KVCache(len(_LAYER_TYPES))
    for layer in (1, 3):
        values = mx.arange(length * 6, dtype=mx.float32).reshape(
            1, 2, length, 3).astype(mx.bfloat16)
        kv.keys[layer] = values + layer
        kv.values[layer] = values + layer + 10
    recurrent = KDAStateCache(len(_LAYER_TYPES))
    for layer in (0, 2):
        recurrent.set_state(
            layer,
            mx.arange(24, dtype=mx.float32).reshape(1, 2, 3, 4) + length,
        )
        recurrent.set_conv_history(
            layer,
            (mx.arange(12, dtype=mx.float32).reshape(1, 2, 2, 3)
             .astype(mx.bfloat16),),
        )
    kv.kda_cache = recurrent
    auxiliary = Qwen4ExpStateCache(len(_LAYER_TYPES))
    for layer in (1, 3):
        auxiliary.qsa_keys[layer] = mx.arange(
            length * 4, dtype=mx.float32).reshape(1, length, 4).astype(
                mx.bfloat16)
        auxiliary.qsa_positions[layer] = mx.arange(
            length, dtype=mx.int32)[None]
    auxiliary.ple_conv[0] = mx.arange(
        72, dtype=mx.float32).reshape(1, 9, 8).astype(mx.bfloat16)
    auxiliary.ple_context[0] = (max(0, length - 2), max(0, length - 1))
    auxiliary.ple_lengths[0] = length
    kv.qwen4_cache = auxiliary
    recurrent.synchronize()
    auxiliary.synchronize()
    mx.eval(*[
        value for value in (*kv.keys, *kv.values) if value is not None
    ])
    return kv


def _assert_equal(actual: KVCache, expected: KVCache) -> None:
    def host(value):
        return np.asarray(
            value.view(mx.uint16) if value.dtype == mx.bfloat16 else value)

    assert actual.offset == expected.offset
    for layer in (1, 3):
        for left, right in (
            (actual.keys[layer], expected.keys[layer]),
            (actual.values[layer], expected.values[layer]),
            (actual.qwen4_cache.qsa_keys[layer],
             expected.qwen4_cache.qsa_keys[layer]),
            (actual.qwen4_cache.qsa_positions[layer],
             expected.qwen4_cache.qsa_positions[layer]),
        ):
            assert np.array_equal(host(left), host(right))
    for layer in (0, 2):
        assert np.array_equal(
            host(actual.kda_cache.state(layer)),
            host(expected.kda_cache.state(layer)))
    assert np.array_equal(
        host(actual.qwen4_cache.ple_conv[0]),
        host(expected.qwen4_cache.ple_conv[0]))
    assert actual.qwen4_cache.ple_context == expected.qwen4_cache.ple_context
    assert actual.qwen4_cache.ple_lengths == expected.qwen4_cache.ple_lengths


@pytest.mark.parametrize("length", (1, 31, 32, 33, 64, 65))
def test_qwen4_stable_prefix_round_trip_is_exact(tmp_path, length):
    journal = _journal(tmp_path)
    tokens = list(range(length))
    expected = _state(length)
    journal.save(
        (), 0, tokens, expected, None, None,
        prompt_length=length, reusable_prefix=length,
        checkpoint_kind="stable_prefix")

    assert journal.find_best_match(tokens, 32) is None
    match = journal.find_best_match(tokens + [9000], 32)
    assert match is not None
    assert match["matched"] == length
    loaded = _journal(tmp_path).load_matched_chain(
        match, len(_LAYER_TYPES))
    assert loaded is not None and loaded[2] is None
    _assert_equal(loaded[1], expected)


def test_qwen4_incomplete_or_noncanonical_auxiliary_state_fails_closed(tmp_path):
    journal = _journal(tmp_path)
    broken = _state(33)
    broken.qwen4_cache.qsa_keys[3] = None
    with pytest.raises(ValueError, match="incomplete QSA keys"):
        journal.save(
            (), 0, list(range(33)), broken, None, None,
            prompt_length=33, reusable_prefix=33,
            checkpoint_kind="stable_prefix")

    broken = _state(33)
    broken.qwen4_cache.qsa_positions[1] = mx.full(
        (1, 33), 7, dtype=mx.int32)
    with pytest.raises(ValueError, match="not canonical"):
        journal.save(
            (), 0, list(range(33)), broken, None, None,
            prompt_length=33, reusable_prefix=33,
            checkpoint_kind="stable_prefix")
