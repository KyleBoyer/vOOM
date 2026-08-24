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
    pin_limit_bytes: int | None = None,
    prefetch_depth: int = 0,
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

    ``pin_limit_bytes`` is a separate cap on just the pinned prefix. This lets
    callers account for already-pinned head/embedding pages in
    ``budget_bytes`` without letting the trunk consume the entire remaining
    cache. ``prefetch_depth`` reserves that many additional unpinned layer
    pages alongside the one demanded layer, preventing a valid pin plan from
    silently making every configured prefetch fail admission.

    Returns zero when nothing fits, which leaves the ordinary policy in charge.
    """
    sizes = [max(0, int(value)) for value in layer_bytes]
    budget = max(0, int(budget_bytes))
    reserve = max(0, int(reserve_bytes))
    pin_limit = (budget if pin_limit_bytes is None
                 else max(0, int(pin_limit_bytes)))
    depth = max(0, int(prefetch_depth))
    best = 0
    prefix_total = 0
    for count in range(1, len(sizes) + 1):
        prefix_total += sizes[count - 1]
        streaming = max(sizes[count:], default=0)
        if (prefix_total <= pin_limit
                and prefix_total + streaming * (1 + depth) + reserve <= budget):
            best = count
    return best


def prefetch_starvation_warning(pinned_bytes: int, cache_bytes: int,
                                layer_bytes: int, expert_batch_bytes: int,
                                prefetch_depth: int) -> str | None:
    """Warn when a trunk pin leaves the prefetcher no budget to work in.

    A pin is charged against the SAME cache budget the prefetcher checks
    before accepting a page, so a pin that approaches ``max_weight_cache_mb``
    makes every scheduled prefetch fail its budget check. Nothing errors and
    nothing is slower in a way that points at the cause: the run simply loses
    the overlap it was configured for.

    Found live, and it cost a full round of investigation. Pinning 4.367GB
    into an 1800MB cache produced prefetch_hits 0 at every depth with
    byte-identical timings, while disk was 71% of decode; the same pin with
    the cache raised to 5500MB landed 168 hits and 10.5% wall.

    Returns a message to print, or None when there is room. Advisory only --
    the operator's explicit budgets are not overridden.
    """
    if prefetch_depth <= 0 or pinned_bytes <= 0:
        return None
    # A prefetch must fit alongside the pin AND the demand traffic already
    # competing for the same budget: one layer page in flight plus the routed
    # expert batch that layer will ask for.
    needed = layer_bytes * (1 + max(prefetch_depth, 1)) + expert_batch_bytes
    headroom = cache_bytes - pinned_bytes
    if headroom >= needed:
        return None
    suggested = (pinned_bytes + needed + 999_999) // 1_000_000
    return (
        f"[engine] WARNING: trunk pin ({pinned_bytes / 1e9:.3f}GB) leaves "
        f"{headroom / 1e9:.3f}GB of the {cache_bytes / 1e9:.3f}GB weight "
        f"cache, but prefetch depth {prefetch_depth} needs "
        f"{needed / 1e9:.3f}GB to accept a page. Every prefetch will be "
        f"refused and the run will silently lose I/O/compute overlap. "
        f"Raise max_weight_cache_mb to at least {suggested}, or lower "
        f"pin_trunk_budget_mb.")
