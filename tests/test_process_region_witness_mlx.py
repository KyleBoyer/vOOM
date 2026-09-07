"""Tiny native map observation around real MLX bits/RNG, not a model test."""

import hashlib

import pytest


@pytest.mark.parametrize('dtype_name', ['bfloat16', 'float16'])
def test_native_map_walk_does_not_change_live_bits_or_rng(dtype_name):
    import mlx.core as mx
    import numpy as np
    from runtime.process_region_witness import sample_self_regions

    raw = mx.array(np.arange(65536, dtype=np.uint16)).view(getattr(mx, dtype_name))
    state = mx.arange(1024, dtype=mx.float32).reshape(32, 32)
    mx.eval(raw, state)
    def hashes():
        return [hashlib.sha256(np.asarray(a.view(mx.uint8)).tobytes()).hexdigest()
                for a in (raw, state)]
    expected = hashes()
    mx.random.seed(713)
    expected_rng = mx.random.uniform(shape=(32,))
    mx.eval(expected_rng)
    mx.random.seed(713)
    snapshot = sample_self_regions()
    actual_rng = mx.random.uniform(shape=(32,))
    mx.eval(actual_rng)
    assert snapshot['available'] and snapshot['coverage_complete']
    assert hashes() == expected
    assert expected_rng.tolist() == actual_rng.tolist()
