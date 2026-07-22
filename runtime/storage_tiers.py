"""Read-only planning for static weight placement over heterogeneous disks.

The runtime already supports fastest-first ``WeightStore.fast_dirs`` overlays,
but the old staging scripts choose a prefix of layers or rank one expert tier by
heat.  This module answers the missing policy question before copying anything:
which exact pages belong on internal SSD, external NVMe, USB, or the source
archive, given measured bandwidth, capacity, reuse, decode cost, and whether the
reader can genuinely overlap independent devices.

It never copies or deletes files.  A resulting plan must still pass the usual
free-space, integrity, and real wall-clock gates before publication.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StorageTier:
    name: str
    bytes_per_second: float
    capacity_bytes: int | None = None
    root: str = ""

    def __post_init__(self):
        if self.bytes_per_second <= 0:
            raise ValueError("bytes_per_second must be positive")
        if self.capacity_bytes is not None and self.capacity_bytes < 0:
            raise ValueError("capacity_bytes must be non-negative")


@dataclass(frozen=True)
class PageDemand:
    name: str
    size_bytes: int
    expected_accesses: float = 1.0
    decode_seconds_per_access: float = 0.0

    def __post_init__(self):
        if self.size_bytes <= 0:
            raise ValueError("size_bytes must be positive")
        if self.expected_accesses < 0:
            raise ValueError("expected_accesses must be non-negative")
        if self.decode_seconds_per_access < 0:
            raise ValueError("decode_seconds_per_access must be non-negative")

    @property
    def weighted_bytes(self) -> float:
        return float(self.size_bytes) * float(self.expected_accesses)


@dataclass(frozen=True)
class TierPlacementPlan:
    assignments: dict[str, str]
    physical_bytes_by_tier: dict[str, int]
    expected_bytes_by_tier: dict[str, float]
    seconds_by_tier: dict[str, float]
    baseline_seconds: float
    estimated_seconds: float
    estimated_speedup: float
    parallel_reads: bool
    excluded_tiers: tuple[str, ...]


def _page_seconds(page: PageDemand, tier: StorageTier) -> float:
    return (page.weighted_bytes / float(tier.bytes_per_second)
            + page.expected_accesses * page.decode_seconds_per_access)


def _fits(tier: StorageTier, used: int, page: PageDemand) -> bool:
    return (tier.capacity_bytes is None
            or used + page.size_bytes <= tier.capacity_bytes)


def plan_static_placement(
    pages: list[PageDemand],
    source: StorageTier,
    overlays: list[StorageTier],
    *,
    parallel_reads: bool = False,
) -> TierPlacementPlan:
    """Plan exact static copies without assuming unsupported I/O overlap.

    In today's serial fetch path, a slower overlay can never help and is
    explicitly excluded.  With ``parallel_reads=True``, longest-processing-time
    scheduling minimizes the maximum normalized device load: a slower independent
    USB device may then help by carrying a proportional minority of traffic.
    That mode is a projection until the serving path proves concurrent reads.
    """
    names = [page.name for page in pages]
    if len(names) != len(set(names)):
        raise ValueError("page names must be unique")
    tier_names = [source.name, *(tier.name for tier in overlays)]
    if len(tier_names) != len(set(tier_names)):
        raise ValueError("tier names must be unique")

    if parallel_reads:
        eligible = list(overlays)
        excluded: list[StorageTier] = []
    else:
        eligible = [tier for tier in overlays
                    if tier.bytes_per_second > source.bytes_per_second]
        excluded = [tier for tier in overlays if tier not in eligible]

    all_tiers = [source, *eligible]
    used = {tier.name: 0 for tier in all_tiers}
    expected = {tier.name: 0.0 for tier in all_tiers}
    seconds = {tier.name: 0.0 for tier in all_tiers}
    assignments: dict[str, str] = {}

    ordered_pages = sorted(
        pages, key=lambda page: (-page.weighted_bytes, page.name))
    for page in ordered_pages:
        candidates = [tier for tier in all_tiers
                      if _fits(tier, used[tier.name], page)]
        if not candidates:
            raise ValueError(f"page {page.name!r} fits no storage tier")

        def objective(tier: StorageTier):
            projected = dict(seconds)
            projected[tier.name] += _page_seconds(page, tier)
            total = max(projected.values()) if parallel_reads else sum(projected.values())
            # Deterministic tie-break: preserve the faster device for a page
            # only when the modeled objective really ties.
            return total, -tier.bytes_per_second, tier.name

        selected = min(candidates, key=objective)
        assignments[page.name] = selected.name
        used[selected.name] += page.size_bytes
        expected[selected.name] += page.weighted_bytes
        seconds[selected.name] += _page_seconds(page, selected)

    for tier in excluded:
        used[tier.name] = 0
        expected[tier.name] = 0.0
        seconds[tier.name] = 0.0

    baseline = sum(_page_seconds(page, source) for page in pages)
    estimated = max(seconds.values()) if parallel_reads else sum(seconds.values())
    speedup = baseline / estimated if estimated > 0 else 1.0
    return TierPlacementPlan(
        assignments=assignments,
        physical_bytes_by_tier=used,
        expected_bytes_by_tier=expected,
        seconds_by_tier=seconds,
        baseline_seconds=baseline,
        estimated_seconds=estimated,
        estimated_speedup=speedup,
        parallel_reads=parallel_reads,
        excluded_tiers=tuple(tier.name for tier in excluded),
    )
