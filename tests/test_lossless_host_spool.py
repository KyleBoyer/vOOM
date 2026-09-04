import mlx.core as mx
import numpy as np
import pytest

from runtime.engine import (
    _lossless_16bit_host_spool,
    _restore_lossless_16bit_host_spool,
)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_lossless_host_spool_round_trips_every_sample_bit(dtype):
    bits = np.array(
        [0x0000, 0x8000, 0x0001, 0x03FF, 0x3C00, 0x7BFF, 0x7C00,
         0xFC00, 0x7E01, 0xFFFF],
        dtype=np.uint16,
    )
    source = mx.array(bits, dtype=mx.uint16).view(dtype)

    host = _lossless_16bit_host_spool(source, expected_dtype=dtype)
    restored = _restore_lossless_16bit_host_spool(host, dtype=dtype)

    assert host.dtype == np.uint16
    assert host.flags.owndata
    assert restored.dtype == dtype
    np.testing.assert_array_equal(np.asarray(restored.view(mx.uint16)), bits)


def test_lossless_host_spool_rejects_midstream_dtype_change():
    source = mx.array([1.0], dtype=mx.float16)

    with pytest.raises(TypeError, match="activation dtype changed"):
        _lossless_16bit_host_spool(source, expected_dtype=mx.bfloat16)


def test_lossless_host_spool_rejects_non_16bit_float_contract():
    source = mx.array([1.0], dtype=mx.float32)

    with pytest.raises(TypeError, match="only FP16/BF16"):
        _lossless_16bit_host_spool(source, expected_dtype=mx.float32)
    with pytest.raises(TypeError, match="only FP16/BF16"):
        _restore_lossless_16bit_host_spool(
            np.array([0], dtype=np.uint16), dtype=mx.float32)
