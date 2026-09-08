"""Exercise the real page selection without MLX, serialization or model I/O."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


def reclaim():
    source = Path(__file__).resolve().parents[1] / "runtime/kv_paged.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PagedKVCache")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "reclaim_closed_pages")
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["reclaim_closed_pages"]


def cache(lengths=(3, 2, 4), recent=1):
    pages = [[SimpleNamespace(resident=True, nbytes=10) for _ in range(n)] for n in lengths]
    events = []
    def spill(layer, index):
        page = pages[layer][index]
        if not page.resident:
            return 0
        events.append((layer, index))
        page.resident = False
        return page.nbytes
    value = SimpleNamespace(_pages=pages, num_layers=len(pages), resident_pages=recent,
        max_bytes=256_000_000, _offset=100, _tail_k=[object() for _ in pages],
        _tail_v=[object() for _ in pages], kda_cache=object(), _spill_resident_page=spill)
    return value, events


def test_oldest_first_excludes_protected_and_recent_pages_and_stops_at_target():
    kv, events = cache()
    original = (kv.max_bytes, kv._offset, tuple(kv._tail_k), tuple(kv._tail_v), kv.kda_cache)
    assert reclaim()(kv, 25, protected_layer=1) == 30
    assert events == [(0, 0), (2, 0), (0, 1)]
    assert original == (kv.max_bytes, kv._offset, tuple(kv._tail_k), tuple(kv._tail_v), kv.kda_cache)
    assert all(p.resident for p in kv._pages[1])
    assert all(layer[-1].resident for layer in kv._pages)


def test_insufficient_candidates_and_repeat_are_best_effort_not_allocation_credit():
    kv, events = cache()
    assert reclaim()(kv, 1000, protected_layer=1) == 50
    assert events == [(0, 0), (2, 0), (0, 1), (2, 1), (2, 2)]
    assert reclaim()(kv, 1000, protected_layer=1) == 0
    assert len(events) == 5 and kv.max_bytes == 256_000_000


@pytest.mark.parametrize("recent,expected", [(0, 90), (1, 60), (4, 0)])
def test_existing_resident_page_policy_is_honored(recent, expected):
    kv, _ = cache(recent=recent)
    assert reclaim()(kv, 1000) == expected


def test_zero_or_empty_candidates_are_noop():
    kv, events = cache()
    assert reclaim()(kv, 0, protected_layer=0) == 0 and not events
    empty, _ = cache(lengths=())
    assert reclaim()(empty, 1) == 0


@pytest.mark.parametrize("requested", [-1, 1.0, True, None, "1"])
def test_bad_requested_bytes_rejected_before_spilling(requested):
    kv, events = cache()
    with pytest.raises(ValueError, match="requested_bytes"):
        reclaim()(kv, requested)
    assert not events


@pytest.mark.parametrize("protected", [-1, 3, 1.0, True, "1"])
def test_bad_protected_layer_rejected_before_spilling(protected):
    kv, events = cache()
    with pytest.raises(ValueError, match="protected_layer"):
        reclaim()(kv, 10, protected_layer=protected)
    assert not events


def test_spill_failure_propagates_without_budget_or_length_changes():
    kv, events = cache()
    failure = OSError("disk unavailable")
    def fail(*args):
        raise failure
    kv._spill_resident_page = fail
    with pytest.raises(OSError) as raised:
        reclaim()(kv, 10)
    assert raised.value is failure and not events
    assert kv.max_bytes == 256_000_000 and kv._offset == 100
    assert all(p.resident for layer in kv._pages for p in layer)
