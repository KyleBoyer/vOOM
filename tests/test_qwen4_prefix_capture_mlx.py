"""Tiny synthetic MLX state/conv oracles, not released-model serving proof.

Requires the usual fresh memory preflight. No model weights are read and no
runtime/profile opts into the new helper. The independent prefix is rebuilt
from deterministic inputs, never taken from the helper's staged records.
"""

import hashlib

import mlx.core as mx
import numpy as np
import pytest

from runtime.kda_state import KDAStateCache
from runtime.kv_cache import KVCache
from runtime.qwen4_exp_state import Qwen4ExpStateCache
from runtime.qwen4_prefix_capture import AlignedPrefixCapture
from runtime.qwen4_retained_history import copy_history_bits
from tests.test_qwen4_prefix_capture_pure import config, fill_layer


def empty_cache(count):
    result = KVCache(count)
    result.kda_cache = KDAStateCache(count)
    result.qwen4_cache = Qwen4ExpStateCache(count)
    return result


def array_factory(activation_dtype):
    def array(shape, dtype="bf16", payload=""):
        target_dtype = {"bf16": activation_dtype, "fp32": mx.float32, "int32": mx.int32}[dtype]
        seed = int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], "little")
        rng = np.random.default_rng(seed)
        # Convolution tails are intentionally noncontiguous views of a larger
        # padded owner, matching the storage hazard without a huge allocation.
        if payload.endswith(("c", "e")):
            raw = rng.integers(0, 65536, size=(shape[0], shape[1] + 32, shape[2] * 2), dtype=np.uint16)
            return mx.array(raw).view(target_dtype)[:, -shape[1]:, ::2]
        if target_dtype == mx.int32:
            return mx.arange(shape[-1], dtype=mx.int32)[None]
        if target_dtype == mx.float32:
            # Ordinary finite matrices; continuation identity checks their
            # bits and immutable ownership, not a synthetic KDA recurrence.
            return mx.array(rng.normal(size=shape).astype(np.float32))
        return mx.array(rng.integers(0, 65536, size=shape, dtype=np.uint16)).view(target_dtype)
    return array


def bits(value):
    unsigned = mx.uint32 if value.dtype in (mx.float32, mx.int32) else mx.uint16
    return np.asarray(value.view(unsigned)).copy()


def assert_same_cache(actual, expected):
    assert actual.offset == expected.offset
    assert actual._starts == expected._starts
    assert actual._windows == expected._windows
    assert actual.compressed_mla == expected.compressed_mla
    for owner_name, fields in (
        (None, ("keys", "values")),
        ("kda_cache", ("_state",)),
        ("qwen4_cache", ("qsa_keys", "qsa_positions", "qsa_pooled_keys", "ple_conv")),
    ):
        left = actual if owner_name is None else getattr(actual, owner_name)
        right = expected if owner_name is None else getattr(expected, owner_name)
        for field in fields:
            for a, b in zip(getattr(left, field), getattr(right, field), strict=True):
                if b is None:
                    assert a is None
                else:
                    assert a.dtype == b.dtype and a.shape == b.shape
                    assert np.array_equal(bits(a), bits(b))
    for a, b in zip(actual.kda_cache._conv, expected.kda_cache._conv, strict=True):
        if b is None:
            assert a is None
        else:
            assert type(a) is tuple and len(a) == len(b) == 1
            assert np.array_equal(bits(a[0]), bits(b[0]))
    for field in ("ple_lengths", "ple_context", "qsa_pool_cache_enabled"):
        assert getattr(actual.qwen4_cache, field) == getattr(expected.qwen4_cache, field)


def make_capture(source, cfg, prefix, total, tile, dtype, reservations):
    return AlignedPrefixCapture(
        source, cfg, prefix_tokens=prefix, total_tokens=total, tile_tokens=tile,
        cache_types=(KVCache, KDAStateCache, Qwen4ExpStateCache),
        empty_cache=lambda: empty_cache(cfg.num_hidden_layers),
        activation_dtype=dtype, state_dtype=mx.float32, position_dtype=mx.int32,
        copy_bits=copy_history_bits, evaluate=mx.eval, reserve=reservations.append)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("prefix,total,tile", [(4, 7, 4), (8, 13, 4), (1024, 1031, 1024)])
def test_complete_raw_state_and_metadata_match_independent_prefix_and_endpoint(dtype, prefix, total, tile):
    cfg, reservations = config(count=8), []
    source, reference_prefix, reference_endpoint = (empty_cache(8) for _ in range(3))
    array = array_factory(dtype)
    capture = make_capture(source, cfg, prefix, total, tile, dtype, reservations)
    capture.begin_sweep(source, offset=0, total_tokens=total, tile_tokens=tile)
    shared, copied = [], []
    for layer in range(8):
        fill_layer(reference_prefix, cfg, layer, prefix, array)
        fill_layer(reference_endpoint, cfg, layer, total, array)
        for start in range(0, total, tile):
            end = min(start + tile, total)
            fill_layer(source, cfg, layer, end, array)
            if end == prefix:
                shared.append((source.keys[layer], source.kda_cache._state[layer],
                               source.qwen4_cache.qsa_keys[layer]))
                copied.append((source.kda_cache._conv[layer], source.qwen4_cache.ple_conv[layer]))
            capture.observe_tile(source, layer=layer, start=start, end=end)
    result, stats = capture.finish(source)
    assert_same_cache(result, reference_prefix)
    assert_same_cache(source, reference_endpoint)
    for layer, (key, state, qkey) in enumerate(shared):
        assert result.keys[layer] is key
        assert result.kda_cache._state[layer] is state
        assert result.qwen4_cache.qsa_keys[layer] is qkey
        conv, pconv = copied[layer]
        if conv is not None:
            assert result.kda_cache._conv[layer] is not conv
            assert result.kda_cache._conv[layer][0] is not conv[0]
        if pconv is not None:
            assert result.qwen4_cache.ple_conv[layer] is not pconv
    assert stats["qwen4_fused_prefix_capture_arrays_copied"] == 7
    assert stats["qwen4_fused_prefix_capture_bytes_copied"] == 6 * 60 + 48
    assert sum(reservations) == 2 * (6 * 60 + 48)
    # The retained branch can advance its append-only state without changing
    # the authoritative full endpoint or another independently reconstructed
    # prefix. All mutable owner/list boundaries are genuinely distinct.
    for owner_name, fields in (
        (None, ("keys", "values", "_starts", "_windows")),
        ("kda_cache", ("_state", "_conv")),
        ("qwen4_cache", ("qsa_keys", "qsa_positions", "qsa_pooled_keys", "ple_conv", "ple_context", "ple_lengths")),
    ):
        a = result if owner_name is None else getattr(result, owner_name)
        b = source if owner_name is None else getattr(source, owner_name)
        assert a is not b
        assert all(getattr(a, field) is not getattr(b, field) for field in fields)
    result.update(3, mx.zeros((1, 2, 1, 4), dtype=dtype), mx.zeros((1, 2, 1, 4), dtype=dtype))
    result.qwen4_cache.update_qsa(3, mx.zeros((1, 1, 2), dtype=dtype), mx.array([[prefix]], dtype=mx.int32))
    assert_same_cache(source, reference_endpoint)
    assert reference_prefix.offset == prefix


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_real_causal_conv_continues_from_captured_prefix_after_source_advances(dtype):
    from runtime.kimi_linear import _causal_depthwise_conv1d

    cfg, source, reservations = config(), empty_cache(4), []
    capture = make_capture(source, cfg, 4, 7, 4, dtype, reservations)
    capture.begin_sweep(source, offset=0, total_tokens=7, tile_tokens=4)
    rng, continuations = np.random.default_rng(91407), {}
    for layer in range(4):
        x = mx.array(rng.normal(size=(1, 7, 10)).astype(np.float32)).astype(dtype)
        weight = mx.array(rng.normal(size=(10, 1, 4)).astype(np.float32)).astype(dtype)
        original_history = None
        for start, end in ((0, 4), (4, 7)):
            fill_layer(source, cfg, layer, end, array_factory(dtype))
            if cfg.layer_types[layer] == "linear_attention":
                output, history = _causal_depthwise_conv1d(x[:, start:end], weight, original_history, 4)
                mx.eval(output, history)  # Existing attention evaluation boundary.
                source.kda_cache._conv[layer] = (history,)
                if start == 0:
                    original_history = history
                else:
                    continuations[layer] = (x[:, 4:7], weight, output, history)
            capture.observe_tile(source, layer=layer, start=start, end=end)
    result, _ = capture.finish(source)
    for layer, (suffix, weight, expected_output, expected_history) in continuations.items():
        output, history = _causal_depthwise_conv1d(suffix, weight, result.kda_cache._conv[layer][0], 4)
        assert np.array_equal(bits(output), bits(expected_output))
        assert np.array_equal(bits(history), bits(expected_history))
        assert np.array_equal(bits(source.kda_cache._conv[layer][0]), bits(expected_history))
