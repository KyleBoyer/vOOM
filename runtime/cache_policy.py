"""The single eviction ordering shared by the live cache and its simulator.

``WeightCache`` and ``runtime.expert_plan.simulate_layout`` used to carry two
independent transcriptions of the same policy.  A capacity plan is only worth
acting on if the simulator evicts exactly what the runtime would have evicted,
and two hand-kept copies cannot be trusted to agree -- so the ordering lives
here once and both callers import it.

Deliberately free of MLX, threading, and I/O: ``expert_plan`` is a pure
planning module and must stay importable without a device or a checkpoint.

The policy is cumulative-frequency-first with recency as the tie-break, and it
protects two classes:

* pinned pages are never victims;
* pages prefetched but not yet demanded are evicted only after every ordinary
  unpinned page is gone, so the prefetcher cannot thrash its own work.
"""

from __future__ import annotations

from typing import Hashable, Iterable, Mapping, NamedTuple, Sequence


class CacheEntry(NamedTuple):
    """One resident page as the eviction policy sees it.

    ``key`` identifies the page, ``pinned`` and ``prefetched`` select its
    protection class.  Entries must be supplied in ascending recency order
    (least recently used first); position in that sequence is the tie-break.

    The runtime keys pages by string; the offline simulator keys them by
    ``(layer, expert)``.  Recency position is unique per entry, so ordering
    never compares two keys and either type works.
    """

    key: Hashable
    pinned: bool = False
    prefetched: bool = False


def rank_victims(
    entries: Iterable[CacheEntry],
    frequencies: Mapping[Hashable, int],
) -> list[Hashable]:
    """Return evictable keys in the exact order the runtime would remove them.

    Pinned pages are excluded entirely.  Ordinary pages come first, then
    unconsumed prefetched pages; within each class the lowest cumulative
    frequency wins and recency order breaks ties.  Callers stop consuming the
    list as soon as their budget is satisfied.
    """
    ordinary: list[tuple[int, int, Hashable]] = []
    prefetched: list[tuple[int, int, Hashable]] = []
    for age, entry in enumerate(entries):
        if entry.pinned:
            continue
        candidate = (int(frequencies.get(entry.key, 0)), age, entry.key)
        (prefetched if entry.prefetched else ordinary).append(candidate)
    return [key for _f, _a, key in sorted(ordinary) + sorted(prefetched)]


def plan_pinned_prefix(
    layer_bytes: Sequence[int],
    budget_bytes: int,
    *,
    reserve_bytes: int = 0,
) -> int:
    """Return how many leading trunk layers to pin under a byte budget.

    A decoder trunk is read strictly cyclically: every layer, in order, exactly
    once per sweep.  That is the worst case for any recency- or
    frequency-ordered policy -- by the time a layer comes round again it is the
    coldest thing resident, so a budget smaller than the whole trunk yields a
    **zero** percent hit rate no matter how large it is.  Pinning a prefix
    instead converts the same bytes into a guaranteed hit rate equal to the
    pinned fraction, and the layers that do not fit stream through the
    remaining room.

    ``reserve_bytes`` is capacity that must stay available to everything else
    the cache serves (routed expert pages, in particular).  Room for one
    streaming trunk layer is added on top of it automatically: the largest
    unpinned layer must still be materializable.  Because that streaming
    requirement shrinks as more of the trunk is pinned, the answer is the
    largest ``n`` satisfying the constraint rather than a single pass -- the
    same fixed point a ring-plus-pinned-prefix layout has to solve.

    Returns zero when nothing fits, which leaves the ordinary policy in charge.
    """
    sizes = [max(0, int(value)) for value in layer_bytes]
    budget = max(0, int(budget_bytes))
    reserve = max(0, int(reserve_bytes))
    best = 0
    prefix_total = 0
    for count in range(1, len(sizes) + 1):
        prefix_total += sizes[count - 1]
        streaming = max(sizes[count:], default=0)
        if prefix_total + streaming + reserve <= budget:
            best = count
    return best
