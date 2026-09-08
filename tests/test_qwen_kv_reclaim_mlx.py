"""Actual engine admission hook, real paging/governor/SDPA; synthetic activations.

The artificially tight device ceiling exercises failure deterministically; it
is not production host pressure or a full-model token/latency proof.
"""

import gc
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from runtime import pressure
from runtime.kda_state import KDAStateCache
from runtime.kv_paged import PagedKVCache
from tests.test_governor_reserve_pure import make_governor
from tests.test_qwen35_serial_refusal_pure import hook, target as refusal_target


def _values(layer, start, width):
    raw = mx.arange(start * 1024, (start + width) * 1024, dtype=mx.uint32)
    k = ((raw.reshape(1, width, 4, 256).transpose(0, 2, 1, 3) + layer * 17) % 251).astype(mx.bfloat16)
    return k, (k + 3).astype(mx.bfloat16)


def _bits(value):
    return np.array(value.view(mx.uint16))


@pytest.mark.parametrize("mode", ["disabled", "recover", "aliases",
    "one_pass_moving", "topup_moving", "topup_aliases", "topup_exhausted"])
def test_real_hook_exact_pages_and_governor_recheck(tmp_path, monkeypatch, mode):
    kv = PagedKVCache(8, 256_000_000, tmp_path, page_positions=256, resident_pages=1)
    kv.kda_cache = KDAStateCache(8)
    kv.kda_cache.set_state(0, mx.full((1, 2, 3, 3), 0.125))
    kv.kda_cache.synchronize()
    recurrent = kv.kda_cache.state(0)
    length = 4103
    for layer in (3, 7):
        for start in range(0, length, 128):
            k, v = _values(layer, start, min(128, length - start))
            kv.append_for_online_attention(layer, k, v)
        del k, v
    lengths = kv.layer_lengths()
    protected = [(p.k, p.v) for p in kv._pages[7]]
    tails = tuple(kv._tail_k), tuple(kv._tail_v)
    aliases = [(p.k, p.v) for pages in kv._pages for p in pages if p.resident] if mode in ("aliases", "topup_aliases") else []
    engine, _ = refusal_target()
    engine.rc = SimpleNamespace(qwen35_serial_kv_reclaim=mode != "disabled",
        qwen35_serial_kv_reclaim_topup=mode.startswith("topup_"))
    engine.cfg = SimpleNamespace(model_type="qwen3_5")
    engine._layer_transient = 8_388_608
    engine._layer_transient_margin = 0
    governor = make_governor(pressure, mx, cache_max=1, floor=1)
    governor.critical = 5_600_000_000
    engine.governor = governor
    # Test only: avoid twelve artificial settle waits for the controlled
    # device-limit failure. Production constants/policy are never changed.
    monkeypatch.setattr(pressure, "_RESERVE_SETTLE_ATTEMPTS", 0)
    gc.collect()
    mx.clear_cache()
    before_active = mx.get_active_memory()
    governor.metal_limit = before_active + engine._layer_transient - 2_100_000
    if mode in ("one_pass_moving", "topup_moving", "topup_exhausted"):
        # A controlled post-spill ceiling drop models headroom moving while
        # real arrays retire. No host-app allocations or production policy
        # constants are changed. Retain the actual governor's second check.
        initial_ceiling = governor.metal_limit
        original_reclaim = kv.reclaim_closed_pages
        def moving_reclaim(*args, **kwargs):
            released = original_reclaim(*args, **kwargs)
            governor.metal_limit = initial_ceiling - (
                20_000_000 if mode == "topup_exhausted" else 2_000_000)
            return released
        monkeypatch.setattr(kv, "reclaim_closed_pages", moving_reclaim)
    invoke, _ = hook()
    invoke.__globals__["mx"] = mx
    try:
        if mode in ("recover", "topup_moving"):
            invoke(engine, 7, 5, length, kv)
            row = engine._qwen35_serial_kv_reclaim_stats["records"][0]
            assert row["outcome"] == "admitted" and row["reservation_retried"]
            assert row["logical_reclaimed_bytes"] >= 2_100_000
            assert row["metal_active_released_bytes"] >= 2_100_000
            assert governor.reservation_calls == 2 and governor.reservation_failures == 1
            if mode == "topup_moving":
                from tests.fixtures.qwen_kv_reclaim_witness import valid_trace
                assert len(row['reclaim_passes']) == 2
                assert row['topup_check']['deficit_bytes'] > 0
                assert all(p['metal_active_released_bytes'] > 0 for p in row['reclaim_passes'])
                assert valid_trace(engine._qwen35_serial_kv_reclaim_stats,
                    budget_bytes=256_000_000)
        else:
            with pytest.raises(MemoryError, match="unsafe Metal reservation"):
                invoke(engine, 7, 5, length, kv)
            if mode in ("aliases", "topup_aliases"):
                row = engine._qwen35_serial_kv_reclaim_stats["records"][0]
                assert row["outcome"] == "refused" and row["reservation_retried"]
                assert row["logical_reclaimed_bytes"] >= 2_100_000
                assert row["metal_active_released_bytes"] < 2_100_000
                assert governor.reservation_calls == governor.reservation_failures == 2
                if mode == "topup_aliases":
                    assert len(row['reclaim_passes']) == 1 and row['topup_check'] is None
            elif mode in ("one_pass_moving", "topup_exhausted"):
                row = engine._qwen35_serial_kv_reclaim_stats['records'][0]
                assert row['outcome'] == 'refused' and row['after_reclaim']['deficit_bytes'] > 0
                assert governor.reservation_calls == governor.reservation_failures == 2
                if mode == "topup_exhausted":
                    assert len(row['reclaim_passes']) == 2
                else:
                    assert row['schema'].endswith('.v1')
            else:
                assert kv.stats.spills == 0 and governor.reservation_calls == 1
        assert kv.layer_lengths() == lengths and kv.offset == length and kv.max_bytes == 256_000_000
        assert kv.kda_cache.state(0) is recurrent
        assert all(p.k is k and p.v is v for p, (k, v) in zip(kv._pages[7], protected))
        assert all(a is b for a, b in zip(kv._tail_k, tails[0]))
        assert all(a is b for a, b in zip(kv._tail_v, tails[1]))
        # Readback and attention comparison happen only after the controlled
        # hook test has ended; no compute is admitted after a refused hook.
        for layer in (3, 7):
            expected_k, expected_v = _values(layer, 0, length)
            actual_k, actual_v = kv.materialize_layer(layer)
            assert np.array_equal(_bits(actual_k), _bits(expected_k))
            assert np.array_equal(_bits(actual_v), _bits(expected_v))
            query = mx.ones((1, 4, 1, 256), dtype=mx.bfloat16)
            expected = mx.fast.scaled_dot_product_attention(query, expected_k, expected_v, scale=0.0625)
            actual = mx.fast.scaled_dot_product_attention(query, actual_k, actual_v, scale=0.0625)
            assert bool(mx.array_equal(actual, expected).item())
        kv.trim_layer_lengths((0, 0, 0, 257, 0, 0, 0, 257))
        for layer in (3, 7):
            k, v = kv.materialize_layer(layer)
            ek, ev = _values(layer, 0, 257)
            assert np.array_equal(_bits(k), _bits(ek)) and np.array_equal(_bits(v), _bits(ev))
        assert kv.offset == 257 and kv.kda_cache.state(0) is recurrent
    finally:
        kv.release()


@pytest.mark.parametrize('topup', [False, True])
def test_mtp_exports_post_bootstrap_recovery_stats(monkeypatch, topup):
    from tests.test_qwen_mtp_scalar_rollback import _FactorTarget, _engine
    target = _FactorTarget(2)
    target.rc.qwen35_serial_kv_reclaim = True
    target.rc.qwen35_serial_kv_reclaim_topup = topup
    original = target.forward_tokens_serial_positions
    expected = dict(attempts=1, admitted=1, records=[dict(outcome="admitted")])
    def verify(*args, **kwargs):
        target._qwen35_serial_kv_reclaim_stats = expected
        return original(*args, **kwargs)
    monkeypatch.setattr(target, "forward_tokens_serial_positions", verify)
    result = _engine(target, True).generate("x", 4)
    assert result["path_stats"]["qwen35_serial_kv_reclaim_enabled"] == 1
    assert result["path_stats"]["qwen35_serial_kv_reclaim_topup_enabled"] == int(topup)
    assert result["path_stats"]["qwen35_serial_kv_reclaim"] == expected
