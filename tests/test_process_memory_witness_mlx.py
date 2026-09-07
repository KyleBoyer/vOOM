"""Tiny real-MLX bits/RNG gate; requires a fresh 30-second memory preflight.

No model weights, active compression, physical reclaim or serving-speed proof.
"""

import hashlib
import json
import time

import pytest

from runtime.process_memory_witness import (
    GovernorProcessMemoryObserver, sample_self_memory,
)


@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_native_observer_preserves_live_bits_and_mlx_rng(dtype, capsys):
    import mlx.core as mx
    import numpy as np
    import psutil

    raw = mx.array(np.arange(65536, dtype=np.uint16)).view(getattr(mx, dtype))
    state = mx.arange(1024, dtype=mx.float32).reshape(32, 32)
    arrays = (raw, state)
    def hashes():
        return [hashlib.sha256(np.asarray(a.view(mx.uint8)).tobytes()).hexdigest()
                for a in arrays]
    expected_bits = hashes()
    mx.random.seed(930)
    expected_rng = mx.random.uniform(shape=(32,))
    mx.eval(expected_rng)
    mx.random.seed(930)
    sample = sample_self_memory()
    assert sample['available'] is True
    swap = psutil.swap_memory()
    GovernorProcessMemoryObserver().record(
        governor_monotonic_s=time.monotonic(),
        system_available_bytes=psutil.virtual_memory().available,
        system_swap_used_bytes=swap.used, system_swap_out_bytes=swap.sout,
        metal_active_bytes=mx.get_active_memory(),
        cache_budget_bytes_after_response=0, swap_pressure_response=False)
    actual_rng = mx.random.uniform(shape=(32,))
    mx.eval(actual_rng)
    assert actual_rng.tolist() == expected_rng.tolist()
    assert hashes() == expected_bits
    line = capsys.readouterr().out.strip()
    record = json.loads(line.removeprefix('[process-memory] '))
    assert record['process']['available'] is True
