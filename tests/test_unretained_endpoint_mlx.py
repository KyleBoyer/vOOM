"""Real plain-KV/KDA/QSA/PLE fork bits; needs fresh 30s MLX preflight.

Synthetic small states, not model outputs or a physical-reclamation benchmark.
"""

import hashlib
from types import SimpleNamespace
import weakref

import pytest

from runtime.request_state import detach_unretained_qwen4_endpoint


@pytest.mark.parametrize("dtype_name", ["bfloat16", "float16"])
@pytest.mark.parametrize("retained_alias", [False, True])
def test_real_shared_fork_bits_metadata_and_rng_survive_endpoint_disposal(
        dtype_name, retained_alias):
    import mlx.core as mx
    import numpy as np
    from runtime.kv_cache import KVCache
    from runtime.kda_state import KDAStateCache
    from runtime.qwen4_exp_state import Qwen4ExpStateCache

    raw = mx.array(np.arange(65536, dtype=np.uint16)).view(getattr(mx, dtype_name))
    prefix = KVCache(2)
    prefix.keys[0] = raw.reshape(1, 1, 256, 256)
    prefix.values[0] = prefix.keys[0]
    prefix.kda_cache = KDAStateCache(2)
    prefix.kda_cache._state[1] = mx.arange(256, dtype=mx.float32).reshape(1, 1, 16, 16)
    prefix.kda_cache._conv[1] = (raw[:768].reshape(1, 3, 256), None)
    prefix.qwen4_cache = Qwen4ExpStateCache(2)
    aux = prefix.qwen4_cache
    aux.qsa_keys[0] = raw.reshape(1, 256, 256)
    aux.qsa_positions[0] = mx.arange(256, dtype=mx.int32)
    aux.qsa_pooled_keys[0] = raw[:1024].reshape(1, 1, 4, 256)
    aux.qsa_pool_cache_enabled = True
    aux.ple_conv[1] = raw[-768:].reshape(1, 3, 256)
    aux.ple_context[1] = (7, 8, 9)
    aux.ple_lengths[1] = 256

    def fingerprint(kv):
        arrays = list(kv.keys) + list(kv.values) + list(kv.kda_cache._state)
        for history in kv.kda_cache._conv:
            arrays.extend(history or ())
        q = kv.qwen4_cache
        arrays.extend(q.qsa_keys + q.qsa_positions + q.qsa_pooled_keys + q.ple_conv)
        bits = [None if a is None else (str(a.dtype), tuple(a.shape),
                hashlib.sha256(np.asarray(a.view(mx.uint8)).tobytes()).hexdigest())
                for a in arrays]
        return (bits, kv.offset, tuple(kv._starts), tuple(kv._windows),
                tuple(q.ple_context), tuple(q.ple_lengths), q.qsa_pool_cache_stats())

    expected = fingerprint(prefix)
    slot = SimpleNamespace(kv=prefix, tokens=tuple(range(256)),
                           metadata={"tile": 256}, logits=None)
    owner = SimpleNamespace(
        cfg=SimpleNamespace(model_type="qwen4_exp"),
        rc=SimpleNamespace(hot_prompt_kv=True, qwen4_hot_kv_tile_aligned=True),
        _hot_prompt_slots=[slot], last_kv=prefix if retained_alias else prefix.fork(),
        _h_last=raw[:16], _h_window=raw[:32],
    )
    endpoint_ref = weakref.ref(owner.last_kv)
    if not retained_alias:
        # Advance only the endpoint's ordinary KV, exercising shared immutable
        # companion arrays plus a separately owned changed attention buffer.
        suffix = mx.zeros((1, 1, 1, 256), dtype=getattr(mx, dtype_name))
        owner.last_kv.update(0, suffix, suffix)
        mx.eval(*[a for a in owner.last_kv.keys if a is not None])
    hidden, window = owner._h_last, owner._h_window
    mx.random.seed(884)
    expected_random = mx.random.uniform(shape=(16,))
    mx.eval(expected_random)
    mx.random.seed(884)
    decision = detach_unretained_qwen4_endpoint(owner, KVCache)
    if not retained_alias:
        assert endpoint_ref() is None and owner.last_kv is None
    else:
        assert endpoint_ref() is prefix and owner.last_kv is prefix
    mx.clear_cache()
    actual_random = mx.random.uniform(shape=(16,))
    mx.eval(actual_random)
    assert actual_random.tolist() == expected_random.tolist()
    assert decision["detached"] is (not retained_alias)
    assert fingerprint(prefix) == expected
    assert owner._hot_prompt_slots == [slot] and slot.kv is prefix
    assert slot.tokens == tuple(range(256)) and slot.metadata == {"tile": 256}
    assert owner._h_last is hidden and owner._h_window is window

    # A subsequent plain-KV fork/update still leaves the retained prefix valid.
    continuation = prefix.fork()
    suffix = mx.zeros((1, 1, 2, 256), dtype=getattr(mx, dtype_name))
    continuation.update(0, suffix, suffix)
    mx.eval(continuation.keys[0], continuation.values[0])
    assert continuation.offset == 258 and prefix.offset == 256
    assert fingerprint(prefix) == expected
