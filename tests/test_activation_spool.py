"""Exact raw-bit gates for the external-NVMe activation spool."""

from __future__ import annotations

import numpy as np
import mlx.core as mx
import pytest

from runtime.activation_spool import DiskBacked16BitTileSpool


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_disk_spool_round_trips_every_16_bit_pattern(tmp_path, dtype):
    raw = np.arange(1 << 16, dtype=np.uint16).reshape(1, 256, 256)
    source = mx.array(raw, dtype=mx.uint16).view(dtype)
    spans = [(0, 31), (31, 129), (129, 256)]
    spool = DiskBacked16BitTileSpool(
        tmp_path, shape=source.shape, spans=spans, dtype=dtype)
    directory = spool.directory

    for index, (start, end) in enumerate(spans):
        assert spool.store(index, source[:, start:end]) == index
    for index, (start, end) in enumerate(spans):
        restored = spool.load(index)
        actual = np.asarray(restored.view(mx.uint16))
        np.testing.assert_array_equal(actual, raw[:, start:end])
        assert restored.dtype == dtype
        assert tuple(restored.shape) == (1, end - start, 256)

    stats = spool.stats()
    assert stats["logical_bytes"] == raw.nbytes
    assert stats["bytes_written"] == raw.nbytes
    assert stats["bytes_read"] == raw.nbytes
    assert stats["write_calls"] == len(spans)
    assert stats["read_calls"] == len(spans)
    assert stats["write_s"] >= 0.0
    assert stats["read_s"] >= 0.0
    assert stats["uncached_descriptors"] in (0, 1)
    assert directory.exists()

    spool.close()
    assert not directory.exists()
    spool.close()
    with pytest.raises(RuntimeError, match="closed"):
        spool.load(0)


def test_disk_spool_validates_dtype_shape_and_partition(tmp_path):
    with pytest.raises(TypeError, match="FP16/BF16"):
        DiskBacked16BitTileSpool(
            tmp_path, shape=(1, 4, 8), spans=[(0, 4)], dtype=mx.float32)
    with pytest.raises(ValueError, match="shape"):
        DiskBacked16BitTileSpool(
            tmp_path, shape=(1, 0, 8), spans=[], dtype=mx.bfloat16)
    with pytest.raises(ValueError, match="partition"):
        DiskBacked16BitTileSpool(
            tmp_path, shape=(1, 4, 8), spans=[(0, 2), (3, 4)],
            dtype=mx.bfloat16)

    spool = DiskBacked16BitTileSpool(
        tmp_path, shape=(1, 4, 8), spans=[(0, 4)], dtype=mx.bfloat16)
    with pytest.raises(IndexError, match="outside"):
        spool.load(1)
    with pytest.raises(ValueError, match="shape"):
        spool.store(0, mx.zeros((1, 3, 8), dtype=mx.bfloat16))
    with pytest.raises(TypeError, match="dtype"):
        spool.store(0, mx.zeros((1, 4, 8), dtype=mx.float16))
    spool.close()
