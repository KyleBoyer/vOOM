"""F197: output must not depend on how much memory the cache was given.

The C reference implementation treats "byte-identical output at every memory
budget" as its load-bearing invariant, and it is a stronger gate than a single
paired A/B: residency budget changes eviction order, fetch batching, page
lifetimes, and pin sets, so any budget-dependent divergence means one of those
mechanisms is altering arithmetic rather than only altering scheduling.

Run against the architecture-faithful tiny GLM fixture (real MoE routing and
expert paging, sub-second), so this is a mechanism gate, not a released-model
claim.

  .venv/bin/python -m pytest tests/test_f197_budget_invariance.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FIXTURE = Path(__file__).resolve().parent.parent / "models" / "glm-fixture-tiny"

PROMPT = ("The quick brown fox jumps over the lazy dog while the curious cat "
          "watches from atop the old wooden fence nearby")
MAX_TOKENS = 8

# Spans pass-through (too small to hold one page) through comfortably resident.
BUDGETS_MB = [1, 4, 16, 64, 256, 2000]


def _ensure_fixture():
    from tests.fixtures.build_glm_fixture import build, is_current
    if not is_current(FIXTURE):
        build(FIXTURE)


def _run(**overrides) -> dict:
    from runtime.engine import RuntimeConfig, StreamingEngine

    config = dict(
        max_weight_cache_mb=256,
        min_weight_cache_mb=1,
        pin_lm_head=True,
        mla_compressed_kv=True,
    )
    config.update(overrides)
    engine = StreamingEngine(str(FIXTURE), RuntimeConfig(**config))
    try:
        result = engine.generate(PROMPT, MAX_TOKENS)
        return {
            "tokens": list(result["tokens"]),
            "incomplete_pages": engine.cache.stats.incomplete_pages,
            "hits": engine.cache.stats.hits,
            "misses": engine.cache.stats.misses,
            "bytes_read": engine.cache.stats.bytes_read,
        }
    finally:
        engine.close()


@pytest.fixture(scope="module")
def reference():
    _ensure_fixture()
    return _run()


@pytest.mark.parametrize("budget_mb", BUDGETS_MB)
def test_tokens_are_identical_at_every_budget(reference, budget_mb):
    got = _run(max_weight_cache_mb=budget_mb)
    assert got["tokens"] == reference["tokens"], (
        f"budget {budget_mb}MB produced {got['tokens']}, "
        f"reference produced {reference['tokens']}")
    assert got["incomplete_pages"] == 0


def test_budget_actually_changes_cache_behavior(reference):
    """Guard against the invariant passing because nothing varied.

    If every budget produced the same hit/miss counts, the parametrized test
    above would be asserting nothing.  A tight budget must read strictly more
    bytes than a generous one.
    """
    tight = _run(max_weight_cache_mb=1)
    generous = _run(max_weight_cache_mb=2000)
    assert tight["bytes_read"] > generous["bytes_read"], (
        f"budget had no effect on I/O: tight read {tight['bytes_read']}, "
        f"generous read {generous['bytes_read']}")
    assert generous["hits"] > tight["hits"]
    assert tight["tokens"] == generous["tokens"] == reference["tokens"]


def test_pinned_trunk_prefix_does_not_change_output(reference):
    """Prefix pinning is a residency policy; it may not move a single token."""
    for pinned in (1, 2):
        got = _run(pin_first_layers=pinned)
        assert got["tokens"] == reference["tokens"], (
            f"pin_first_layers={pinned} diverged: {got['tokens']}")
        assert got["incomplete_pages"] == 0


def test_budget_derived_trunk_pin_does_not_change_output(reference):
    """The F197 planner must also be output-neutral, at several budgets."""
    from runtime.engine import RuntimeConfig, StreamingEngine

    planned = []
    for budget_mb in (1, 8, 64, 512):
        engine = StreamingEngine(str(FIXTURE), RuntimeConfig(
            max_weight_cache_mb=256, min_weight_cache_mb=1, pin_lm_head=True,
            mla_compressed_kv=True, pin_trunk_budget_mb=budget_mb))
        try:
            tokens = list(engine.generate(PROMPT, MAX_TOKENS)["tokens"])
            planned.append((budget_mb, engine.planned_trunk_pin_layers))
            assert engine.cache.stats.incomplete_pages == 0
        finally:
            engine.close()
        assert tokens == reference["tokens"], (
            f"pin_trunk_budget_mb={budget_mb} diverged: {tokens}")

    counts = [count for _budget, count in planned]
    assert counts == sorted(counts), (
        f"pinned prefix must grow monotonically with budget: {planned}")
    assert counts[-1] > counts[0], (
        f"budget had no effect on the planned prefix: {planned}")


# ---- incomplete-page detection ------------------------------------------
#
# A page served without every requested tensor is the one storage failure that
# cannot be seen downstream: a routed expert whose weights never arrived just
# contributes zero to the MoE sum, so the request succeeds and returns a
# plausible wrong answer. These tests prove the cache converts that into an
# error instead of admitting it.


class _StubTensor:
    __slots__ = ("nbytes",)

    def __init__(self, nbytes: int = 1024):
        self.nbytes = nbytes


class _DroppingStore:
    """Returns every requested tensor except the ones named in ``drop``."""

    def __init__(self, drop: set[str]):
        self.drop = drop

    def fetch(self, names):
        tensors = {name: _StubTensor() for name in names
                   if name not in self.drop}
        return tensors, 0.0, 1024 * len(tensors)


def _cache(store):
    from runtime.weight_cache import WeightCache

    return WeightCache(store, max_bytes=10_000_000)


def test_short_page_from_get_is_refused_and_counted():
    cache = _cache(_DroppingStore({"b"}))
    with pytest.raises(KeyError, match="incomplete page"):
        cache.get("page", ["a", "b", "c"])
    assert cache.stats.incomplete_pages == 1
    assert "INCOMPLETE PAGES" in cache.stats.summary()


def test_short_page_from_get_many_is_refused_and_counted():
    cache = _cache(_DroppingStore({"layer.1.expert.4.w"}))
    with pytest.raises(KeyError, match="incomplete page"):
        cache.get_many([
            ("layer.1.expert.3", ["layer.1.expert.3.w"]),
            ("layer.1.expert.4", ["layer.1.expert.4.w"]),
        ])
    assert cache.stats.incomplete_pages == 1


def test_short_page_from_warm_tier_is_refused():
    """The warm tier previously admitted its pages with no name check."""
    class _ShortWarm:
        def take(self, key):
            return {"a": _StubTensor()}  # missing "b"

        def admit(self, key, tensors):
            pass

    from runtime.weight_cache import WeightCache

    cache = WeightCache(_DroppingStore(set()), max_bytes=10_000_000,
                        warm=_ShortWarm())
    with pytest.raises(KeyError, match="incomplete page"):
        cache.get("page", ["a", "b"])
    assert cache.stats.incomplete_pages == 1


def test_complete_pages_never_increment_the_counter():
    cache = _cache(_DroppingStore(set()))
    cache.get("page", ["a", "b", "c"])
    cache.get_many([("other", ["d", "e"])])
    assert cache.stats.incomplete_pages == 0
    assert "INCOMPLETE" not in cache.stats.summary()
