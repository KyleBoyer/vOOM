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
