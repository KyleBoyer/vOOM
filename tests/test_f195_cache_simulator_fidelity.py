"""F195: the offline capacity simulator must match the live cache exactly.

``runtime.expert_plan.simulate_layout`` is used to size the expert cache without
paying for a full request per data point.  That is only sound if its replay
produces the hit/miss counts the runtime would actually have produced.  These
tests drive the real ``WeightCache`` over a recorded trace with a stub store and
require the counts to agree exactly -- not approximately.

They also cover the ordering primitive itself (``runtime.cache_policy``), which
both sides now import instead of each keeping a transcription of it.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from runtime.cache_policy import CacheEntry, rank_victims  # noqa: E402
from runtime.expert_plan import (  # noqa: E402
    capacity_sweep, identity_orders, simulate_layout,
)
from runtime.weight_cache import WeightCache  # noqa: E402


PAGE_BYTES = 1024


class StubTensor:
    """Minimal stand-in for an mx.array: the cache only reads ``nbytes``."""

    __slots__ = ("nbytes",)

    def __init__(self, nbytes: int):
        self.nbytes = nbytes


class StubStore:
    """Counts what the cache actually asked the storage layer to read."""

    def __init__(self, page_bytes: int = PAGE_BYTES):
        self.page_bytes = page_bytes
        self.fetch_calls = 0
        self.fetched_names: list[str] = []

    def fetch(self, names):
        self.fetch_calls += 1
        self.fetched_names.extend(names)
        tensors = {name: StubTensor(self.page_bytes) for name in names}
        return tensors, 0.0, self.page_bytes * len(names)


def make_trace(*, sweeps: int, layers: int, num_experts: int, top_k: int,
               seed: int) -> list[dict[int, tuple[int, ...]]]:
    """Build a routed trace with realistic partial reuse between sweeps.

    Each sweep re-selects experts from a layer-local biased pool, so adjacent
    sweeps overlap substantially without ever being identical -- the shape the
    real K3 measurements show (F187 measured 38.59% adjacent-route overlap).
    """
    rng = random.Random(seed)
    hot = {
        layer: rng.sample(range(num_experts), k=min(num_experts, top_k * 3))
        for layer in range(layers)
    }
    trace = []
    for _ in range(sweeps):
        sweep = {}
        for layer in range(layers):
            picks = set()
            while len(picks) < top_k:
                if rng.random() < 0.7:
                    picks.add(rng.choice(hot[layer]))
                else:
                    picks.add(rng.randrange(num_experts))
            sweep[layer] = tuple(sorted(picks))
        trace.append(sweep)
    return trace


def replay_on_real_cache(trace, *, cache_pages: int, fetch_batch: int = 0
                         ) -> tuple[int, int, int]:
    """Run the recorded trace through the live ``WeightCache``.

    Mirrors ``StreamingEngine._get_experts``: one ``get_many`` per layer with
    the whole routed union, which is the shape ``simulate_layout`` models.
    Returns ``(hits, misses, pages_fetched)``.
    """
    store = StubStore()
    cache = WeightCache(
        store,
        max_bytes=cache_pages * PAGE_BYTES,
        max_fetch_batch=fetch_batch,
    )
    for sweep in trace:
        for layer, experts in sweep.items():
            items = [
                (f"layer.{layer}.expert.{expert}",
                 [f"model.layers.{layer}.experts.{expert}.weight"])
                for expert in experts
            ]
            cache.get_many(items)
    return cache.stats.hits, cache.stats.misses, len(store.fetched_names)


@pytest.mark.parametrize("cache_pages", [0, 1, 4, 16, 64, 256, 4096])
@pytest.mark.parametrize("fetch_batch", [0, 4])
def test_simulator_matches_live_cache_exactly(cache_pages, fetch_batch):
    trace = make_trace(sweeps=12, layers=6, num_experts=48, top_k=8, seed=7)
    orders = identity_orders(trace, num_experts=48)

    simulated = simulate_layout(
        trace, orders,
        expert_page_bytes=PAGE_BYTES,
        bandwidth_mbps=1000.0,
        coalesce_gap_pages=-1,
        cache_pages=cache_pages,
        fetch_batch=fetch_batch,
    )
    hits, misses, pages_fetched = replay_on_real_cache(
        trace, cache_pages=cache_pages, fetch_batch=fetch_batch)

    assert (simulated.cache_hits, simulated.cache_misses) == (hits, misses)
    # The simulator's demanded page count is what the store is asked to read.
    assert simulated.demanded_bytes == pages_fetched * PAGE_BYTES


def test_capacity_sweep_is_monotone_and_bounded():
    """More capacity never hurts, and the curve saturates at the working set."""
    trace = make_trace(sweeps=20, layers=8, num_experts=64, top_k=8, seed=11)
    working_set = len({(layer, expert)
                       for sweep in trace
                       for layer, experts in sweep.items()
                       for expert in experts})

    points = capacity_sweep(
        trace, [0, 8, 32, 128, 512, working_set, working_set * 2],
        expert_page_bytes=PAGE_BYTES, num_experts=64)

    rates = [point.hit_rate for point in points]
    assert rates == sorted(rates), f"hit rate not monotone in capacity: {rates}"
    assert points[0].hit_rate == 0.0  # zero capacity disables the cache
    by_pages = {point.cache_pages: point for point in points}
    # Once every distinct page fits, nothing is ever evicted, so every access
    # after the first per page is a hit and physical bytes equal the working set.
    saturated = by_pages[working_set * 2]
    assert saturated.misses == working_set
    assert saturated.physical_bytes == working_set * PAGE_BYTES
    assert by_pages[working_set].hit_rate == saturated.hit_rate


def test_capacity_sweep_matches_live_cache_at_every_point():
    trace = make_trace(sweeps=10, layers=5, num_experts=40, top_k=6, seed=3)
    capacities = [0, 2, 7, 23, 90, 1000]
    points = capacity_sweep(
        trace, capacities, expert_page_bytes=PAGE_BYTES, num_experts=40)
    for point in points:
        hits, misses, _pages = replay_on_real_cache(
            trace, cache_pages=point.cache_pages)
        assert (point.hits, point.misses) == (hits, misses), (
            f"capacity {point.cache_pages} pages: simulator "
            f"{point.hits}/{point.misses} vs runtime {hits}/{misses}")


def test_fetch_batch_never_changes_hit_rate():
    """Sub-batching bounds peak residency; it cannot change what is retained.

    Eviction always removes the globally lowest-ranked page, so admitting a
    routed union in groups removes exactly the pages a single post-admission
    pass would have removed -- just earlier.  Asserting this keeps the
    capacity sweep honest: its hit-rate curve is independent of
    ``max_fetch_batch``, so a sweep run at one batch size may be read at any
    other.  A search over 1,440 (seed, top-k, expert-count, capacity, batch)
    combinations found no counterexample.
    """
    trace = make_trace(sweeps=10, layers=4, num_experts=32, top_k=12, seed=5)
    orders = identity_orders(trace, num_experts=32)
    for cache_pages in (2, 5, 9, 32):
        reference = None
        for fetch_batch in (0, 1, 3, 8):
            result = simulate_layout(
                trace, orders,
                expert_page_bytes=PAGE_BYTES, bandwidth_mbps=1000.0,
                coalesce_gap_pages=-1, cache_pages=cache_pages,
                fetch_batch=fetch_batch)
            counts = (result.cache_hits, result.cache_misses)
            if reference is None:
                reference = counts
            assert counts == reference, (
                f"capacity {cache_pages}, batch {fetch_batch}: {counts} "
                f"!= {reference}")


def test_fetch_batch_does_change_request_shape():
    """What sub-batching *does* affect is coalescing granularity.

    Smaller groups issue more requests but read fewer unused gap pages.  This
    is the tradeoff ``max_fetch_batch`` actually controls, and the simulator
    has to represent it for coalescing plans to be scored correctly.
    """
    trace = make_trace(sweeps=6, layers=3, num_experts=32, top_k=12, seed=5)
    orders = identity_orders(trace, num_experts=32)
    shapes = {}
    for fetch_batch in (0, 2, 4):
        result = simulate_layout(
            trace, orders,
            expert_page_bytes=PAGE_BYTES, bandwidth_mbps=1000.0,
            coalesce_gap_pages=2, cache_pages=8, fetch_batch=fetch_batch)
        shapes[fetch_batch] = (result.requests, result.physical_bytes)
    assert shapes[2][0] > shapes[0][0], "smaller groups must issue more requests"
    assert shapes[2][1] < shapes[0][1], "smaller groups must read fewer bytes"
    assert shapes[4] != shapes[0] and shapes[4] != shapes[2]


def test_rank_victims_protects_pinned_and_prefetched():
    entries = [
        CacheEntry("cold-ordinary"),
        CacheEntry("pinned", pinned=True),
        CacheEntry("prefetched", prefetched=True),
        CacheEntry("hot-ordinary"),
    ]
    frequencies = {"cold-ordinary": 1, "hot-ordinary": 9, "prefetched": 1,
                   "pinned": 1}
    victims = rank_victims(entries, frequencies)
    assert "pinned" not in victims
    # Every ordinary page goes before any unconsumed prefetched page, even when
    # the prefetched page is colder.
    assert victims == ["cold-ordinary", "hot-ordinary", "prefetched"]


def test_rank_victims_breaks_frequency_ties_by_recency():
    entries = [CacheEntry("oldest"), CacheEntry("middle"), CacheEntry("newest")]
    victims = rank_victims(entries, {"oldest": 2, "middle": 2, "newest": 2})
    assert victims == ["oldest", "middle", "newest"]
