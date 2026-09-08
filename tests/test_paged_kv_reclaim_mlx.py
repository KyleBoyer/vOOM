"""Exact paging/reclamation storage and attention tests; supervised MLX job only."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from runtime.kv_paged import PagedKVCache


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
@pytest.mark.parametrize("compressed", [False, True])
def test_reclaim_preserves_bits_attention_tails_companion_and_rollback(tmp_path, dtype, compressed):
    kv = PagedKVCache(8, 1_000_000, tmp_path, page_positions=4,
                      resident_pages=1, compress_spill=compressed)
    marker = kv.kda_cache = SimpleNamespace(state=object())
    def bits(value):
        return np.array(value.view(mx.uint16 if dtype == mx.bfloat16 else mx.uint32))
    source = mx.array(np.arange(192, dtype=np.float32).reshape(1, 2, 12, 8) / 17 - 3).astype(dtype)
    values = (-source).astype(dtype)
    for layer in (3, 7):
        for start in range(0, 11, 3):
            kv.append_for_online_attention(layer, source[:, :, start:min(start+3, 11)],
                values[:, :, start:min(start+3, 11)])
    lengths = kv.layer_lengths()
    tails = tuple(kv._tail_k), tuple(kv._tail_v)
    query = mx.ones((1, 2, 1, 8), dtype=dtype)
    keys, vals = kv.materialize_layer(3)
    expected = mx.fast.scaled_dot_product_attention(query, keys, vals, scale=0.25)
    mx.eval(expected)
    if compressed and dtype != mx.bfloat16:
        # The existing compressed format records no dtype and is BF16-only.
        # Reject incompatible input before writing or dropping any references.
        with pytest.raises(ValueError, match="requires BF16"):
            kv.reclaim_closed_pages(1, protected_layer=7)
        assert kv.stats.spills == 0 and kv.layer_lengths() == lengths
        assert all(p.resident for pages in kv._pages for p in pages)
        kv.release()
        return
    assert kv.reclaim_closed_pages(1, protected_layer=7) > 0
    assert kv.max_bytes == 1_000_000 and kv.layer_lengths() == lengths and kv.offset == 11
    assert kv.kda_cache is marker
    assert all(a is b for a, b in zip(kv._tail_k, tails[0]))
    assert all(a is b for a, b in zip(kv._tail_v, tails[1]))
    assert all(p.resident for p in kv._pages[7]) and kv._pages[3][-1].resident
    for layer in (3, 7):
        keys, vals = kv.materialize_layer(layer)
        assert np.array_equal(bits(keys), bits(source[:, :, :11]))
        assert np.array_equal(bits(vals), bits(values[:, :, :11]))
        actual = mx.fast.scaled_dot_product_attention(query, keys, vals, scale=0.25)
        assert bool(mx.array_equal(actual, expected).item())
        kv.append_for_online_attention(layer, source[:, :, 11:], values[:, :, 11:])
    # Roll back across a closed/spilled page and keep exact partial-page bits.
    kv.trim_layer_lengths((0, 0, 0, 5, 0, 0, 0, 5))
    for layer in (3, 7):
        keys, vals = kv.materialize_layer(layer)
        assert np.array_equal(bits(keys), bits(source[:, :, :5]))
        assert np.array_equal(bits(vals), bits(values[:, :, :5]))
    assert kv.offset == 5 and kv.kda_cache is marker
    kv.release()


def test_spill_failure_keeps_page_references_and_stats(tmp_path, monkeypatch):
    kv = PagedKVCache(1, 1_000_000, tmp_path, page_positions=4, resident_pages=0)
    kv.append_for_online_attention(0, mx.ones((1, 1, 4, 2)), mx.zeros((1, 1, 4, 2)))
    page = kv._pages[0][0]
    keys, values = page.k, page.v
    failure = OSError("test disk write failure")
    def fail(*args, **kwargs):
        raise failure
    monkeypatch.setattr(mx, "save_safetensors", fail)
    with pytest.raises(OSError) as raised:
        kv.reclaim_closed_pages(1)
    assert raised.value is failure and page.k is keys and page.v is values
    assert page.path is None and kv.stats.spills == 0 and kv.offset == 4
    kv.release()


def test_spill_paths_are_cache_local_and_release_preserves_other_cache(tmp_path):
    caches = [PagedKVCache(1, 1_000_000, tmp_path, page_positions=4, resident_pages=0)
              for _ in range(2)]
    for index, kv in enumerate(caches):
        kv.append_for_online_attention(0, mx.full((1, 1, 4, 2), index, dtype=mx.bfloat16),
            mx.full((1, 1, 4, 2), index+1, dtype=mx.bfloat16))
        assert kv.reclaim_closed_pages(1) > 0
    paths = [kv._pages[0][0].path for kv in caches]
    assert paths[0] != paths[1]
    caches[0].release()
    assert paths[1].exists()
    keys, values = caches[1].materialize_layer(0)
    assert bool(mx.all(keys == 1).item()) and bool(mx.all(values == 2).item())
    caches[1].release()
