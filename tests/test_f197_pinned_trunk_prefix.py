"""F197: prefix pinning for the strictly cyclic trunk.

A decoder trunk is read once per layer per sweep, in order.  Under any
recency- or frequency-ordered eviction, a layer is the coldest resident page
exactly when it is about to be needed again, so a budget smaller than the whole
trunk returns a zero percent hit rate however large it is.  These tests prove
that pathology against the **real** ``WeightCache`` rather than asserting it,
then prove that pinning a prefix of the same size converts it into a hit rate
equal to the pinned fraction.

``plan_pinned_prefix`` sizes that prefix from a byte budget while leaving room
for one streaming layer plus whatever else shares the cache.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from runtime.cache_policy import plan_pinned_prefix  # noqa: E402
from runtime.weight_cache import WeightCache  # noqa: E402


LAYER_BYTES = 1_000_000


class StubTensor:
    __slots__ = ("nbytes",)

    def __init__(self, nbytes: int):
        self.nbytes = nbytes


class StubStore:
    def __init__(self, layer_bytes: int = LAYER_BYTES):
        self.layer_bytes = layer_bytes
        self.bytes_read = 0

    def fetch(self, names):
        tensors = {name: StubTensor(self.layer_bytes // len(names))
                   for name in names}
        read = sum(t.nbytes for t in tensors.values())
        self.bytes_read += read
        return tensors, 0.0, read


def layer_names(layer: int) -> list[str]:
    return [f"model.layers.{layer}.self_attn.weight",
            f"model.layers.{layer}.mlp.weight"]


def sweep_trunk(cache: WeightCache, *, layers: int, sweeps: int) -> None:
    """Read every layer once per sweep, in released order."""
    for _ in range(sweeps):
        for layer in range(layers):
            cache.get(f"layer.{layer}", layer_names(layer))


def test_cyclic_trunk_gets_zero_hits_without_pinning():
    """The pathology the C reference calls out, measured on our own cache."""
    layers, sweeps = 32, 6
    for budget_layers in (1, 4, 8, 16, 31):
        store = StubStore()
        cache = WeightCache(store, max_bytes=budget_layers * LAYER_BYTES)
        sweep_trunk(cache, layers=layers, sweeps=sweeps)
        assert cache.stats.hits == 0, (
            f"budget {budget_layers}/{layers} layers scored "
            f"{cache.stats.hits} hits; the cyclic sweep should defeat "
            f"recency/frequency ordering entirely")
        assert cache.stats.misses == layers * sweeps


def test_budget_that_holds_the_whole_trunk_hits_everything_after_sweep_one():
    layers, sweeps = 32, 6
    store = StubStore()
    cache = WeightCache(store, max_bytes=layers * LAYER_BYTES)
    sweep_trunk(cache, layers=layers, sweeps=sweeps)
    assert cache.stats.misses == layers
    assert cache.stats.hits == layers * (sweeps - 1)


@pytest.mark.parametrize("pinned", [1, 4, 8, 16])
def test_pinned_prefix_converts_dead_budget_into_hits(pinned):
    """Same bytes, but pinned: the hit rate becomes the pinned fraction."""
    layers, sweeps = 32, 6
    store = StubStore()
    # One extra layer of room so the unpinned remainder can still stream.
    cache = WeightCache(store, max_bytes=(pinned + 1) * LAYER_BYTES)
    for layer in range(pinned):
        cache.pin(f"layer.{layer}", layer_names(layer))
    sweep_trunk(cache, layers=layers, sweeps=sweeps)

    # Pinned layers hit on every sweep including the first (pin() loaded them).
    assert cache.stats.hits == pinned * sweeps
    assert cache.stats.misses == (layers - pinned) * sweeps
    achieved = cache.stats.hits / (cache.stats.hits + cache.stats.misses)
    assert achieved == pytest.approx(pinned / layers)


def test_pinned_prefix_reduces_bytes_read_proportionally():
    layers, sweeps, pinned = 32, 6, 8
    unpinned_store = StubStore()
    unpinned = WeightCache(unpinned_store,
                           max_bytes=(pinned + 1) * LAYER_BYTES)
    sweep_trunk(unpinned, layers=layers, sweeps=sweeps)

    pinned_store = StubStore()
    pinned_cache = WeightCache(pinned_store,
                               max_bytes=(pinned + 1) * LAYER_BYTES)
    for layer in range(pinned):
        pinned_cache.pin(f"layer.{layer}", layer_names(layer))
    sweep_trunk(pinned_cache, layers=layers, sweeps=sweeps)

    # The pinned arm reads each pinned layer once instead of once per sweep.
    expected_saved = pinned * (sweeps - 1) * LAYER_BYTES
    assert unpinned_store.bytes_read - pinned_store.bytes_read == expected_saved


def test_plan_pinned_prefix_leaves_room_to_stream_and_for_experts():
    layers = [LAYER_BYTES] * 32

    # Budget for 9 layers: pin 8, leave one layer of streaming room.
    assert plan_pinned_prefix(layers, 9 * LAYER_BYTES) == 8
    # Reserving 4 layers' worth for routed experts shrinks the prefix.
    assert plan_pinned_prefix(
        layers, 9 * LAYER_BYTES, reserve_bytes=4 * LAYER_BYTES) == 4
    # Too small to hold even one pinned layer plus a streaming one.
    assert plan_pinned_prefix(layers, LAYER_BYTES) == 0
    # Budget covering the entire trunk pins all of it: nothing is left to
    # stream, so no streaming allowance is required.
    assert plan_pinned_prefix(layers, 32 * LAYER_BYTES) == 32


def test_plan_pinned_prefix_handles_uneven_layers():
    """The streaming allowance must track the largest *unpinned* layer."""
    layers = [1, 1, 1, 10, 1, 1]  # layer 3 is the outlier
    # Pinning 3 costs 3 and must still fit the 10-byte layer 3 streaming.
    assert plan_pinned_prefix(layers, 13) == 3
    assert plan_pinned_prefix(layers, 12) == 2
    # Once layer 3 is itself pinned the allowance collapses to 1, so a budget
    # that could not hold 3 pinned layers can hold 4.
    assert plan_pinned_prefix(layers, 14) == 4


def test_plan_pinned_prefix_rejects_nothing_fitting():
    assert plan_pinned_prefix([], 1_000) == 0
    assert plan_pinned_prefix([LAYER_BYTES] * 4, 0) == 0
