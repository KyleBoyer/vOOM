"""Real MLX bit-copy checks; require the normal memory preflight first."""

import mlx.core as mx
import numpy as np
import pytest

from tests.fixtures.qwen4_retained_fork_diagnostic import copy_ple_bits


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_detachment_preserves_all_16bit_patterns_and_shape(dtype):
    # Includes signed zeros, every NaN payload, infinities and subnormals.
    raw = np.arange(65536, dtype=np.uint16).reshape(1, 256, 256)
    source = mx.array(raw).view(dtype)
    detached = copy_ple_bits(source, array_module=mx, numpy_module=np)
    assert detached is not source and detached.dtype == source.dtype
    assert detached.shape == source.shape
    assert np.array_equal(np.asarray(detached.view(mx.uint16)), raw)
    assert np.array_equal(np.asarray(source.view(mx.uint16)), raw)


def test_detachment_preserves_noncontiguous_tail_view():
    raw = np.arange(2048, dtype=np.uint16).reshape(1, 256, 8)
    source = mx.array(raw).view(mx.bfloat16)[:, -9:, ::2]
    detached = copy_ple_bits(source, array_module=mx, numpy_module=np)
    assert np.array_equal(np.asarray(detached.view(mx.uint16)), raw[:, -9:, ::2])


def test_wrong_dtype_is_not_silently_cast():
    with pytest.raises(ValueError, match="16-bit"):
        copy_ple_bits(mx.array([1], dtype=mx.float32), array_module=mx, numpy_module=np)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("prefix_length,suffix_length", [(1, 1), (32, 1), (32, 17)])
def test_real_deltanet_conv_continuation_is_bit_exact_after_detach(
        dtype, prefix_length, suffix_length):
    from runtime.kimi_linear import _causal_depthwise_conv1d
    rng = np.random.default_rng(8128)
    prefix = mx.array(rng.normal(size=(1, prefix_length, 8)).astype(np.float32)).astype(dtype)
    suffix = mx.array(rng.normal(size=(1, suffix_length, 8)).astype(np.float32)).astype(dtype)
    weight = mx.array(rng.normal(size=(8, 1, 4)).astype(np.float32)).astype(dtype)
    _, retained = _causal_depthwise_conv1d(prefix, weight, None, 4)
    mx.eval(retained)
    detached = copy_ple_bits(retained, array_module=mx, numpy_module=np)
    expected, expected_history = _causal_depthwise_conv1d(suffix, weight, retained, 4)
    actual, actual_history = _causal_depthwise_conv1d(suffix, weight, detached, 4)
    for left, right in ((expected, actual), (expected_history, actual_history)):
        assert left.dtype == right.dtype == dtype
        assert np.array_equal(np.asarray(left.view(mx.uint16)),
                              np.asarray(right.view(mx.uint16)))
