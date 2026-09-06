"""Small real-device check; requires the same fresh memory preflight as MLX jobs.

Not a model/state oracle or a physical-memory reclamation benchmark.
"""

from types import SimpleNamespace

import pytest

from runtime.phase_head_witness import post_generation_memory_witness


@pytest.mark.parametrize("barrier", [False, True])
@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
def test_real_device_boundary_preserves_live_bits_owners_and_rng(barrier, dtype_name):
    import mlx.core as mx

    dtype = getattr(mx, dtype_name)
    live = mx.arange(4096).reshape(64, 64).astype(dtype)
    pending = live @ mx.eye(64, dtype=dtype)
    cache = SimpleNamespace(total_bytes=16384, pinned_bytes=8192, max_bytes=32768)
    slot = SimpleNamespace(kv=live, logits=pending, tokens=(1, 2, 3))
    target = SimpleNamespace(cache=cache, last_kv=live, _hot_prompt_slots=[slot],
                             _h_last=pending, _true_peak_metal_bytes=0)
    mx.eval(live, pending)
    expected_live_bits = live.view(mx.uint8).tolist()
    expected_pending_bits = pending.view(mx.uint8).tolist()
    mx.random.seed(7321)
    first = mx.random.uniform(shape=(32,))
    second = mx.random.uniform(shape=(32,))
    mx.eval(first, second)
    expected = second.tolist()
    mx.random.seed(7321)
    first = mx.random.uniform(shape=(32,))
    mx.eval(first)
    observation = post_generation_memory_witness(target, mx, barrier=barrier)
    actual = mx.random.uniform(shape=(32,))
    mx.eval(actual)
    assert actual.tolist() == expected
    assert live.view(mx.uint8).tolist() == expected_live_bits
    assert pending.view(mx.uint8).tolist() == expected_pending_bits
    assert target.last_kv is live and target._h_last is pending
    assert target.cache is cache and target._hot_prompt_slots == [slot]
    assert slot.kv is live and slot.logits is pending and slot.tokens == (1, 2, 3)
    assert observation["before"]["available"] is True
    assert observation["synchronizes_device"] is barrier
    assert observation["synchronization_scope"] == (
        "default_stream_of_default_device" if barrier else "none")
    if barrier:
        assert observation["after_clear_cache"]["available"] is True
        assert observation["after_clear_cache"]["metal_allocator_cache_bytes"] == 0
