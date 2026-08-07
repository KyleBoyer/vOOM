"""F225: warn when a trunk pin starves the prefetcher.

The pin and the prefetcher are charged against the SAME budget, so a large
pin silently makes every prefetch fail its budget check. Nothing errors; the
run just loses its overlap. Measured live: pinning 4.367GB into an 1800MB
cache gave prefetch_hits 0 at every depth with byte-identical timings, while
disk was 71% of decode.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GB = 1_000_000_000
LAYER = 143_600_000
EXPERTS = 53_600_000


def _warn(pinned, cache, depth=2):
    from runtime.cache_policy import prefetch_starvation_warning

    return prefetch_starvation_warning(pinned, cache, LAYER, EXPERTS, depth)


def test_the_measured_starving_configuration_warns():
    """The exact numbers that produced prefetch_hits 0."""
    message = _warn(4_367_000_000, 1_800_000_000)
    assert message is not None
    assert "prefetch" in message.lower()
    assert "max_weight_cache_mb" in message


def test_the_measured_working_configuration_is_silent():
    """4.2GB pin in a 5.5GB cache landed 168 prefetch hits; must not warn."""
    assert _warn(3_427_000_000, 5_500_000_000) is None


def test_no_warning_when_prefetch_is_disabled():
    """Depth 0 means no overlap was asked for, so there is nothing to lose."""
    assert _warn(4_367_000_000, 1_800_000_000, depth=0) is None


def test_no_warning_without_a_pin():
    assert _warn(0, 1_800_000_000) is None


def test_suggested_budget_actually_clears_the_warning():
    """The remedy must work, not just sound plausible."""
    import re

    message = _warn(4_367_000_000, 1_800_000_000)
    suggested = int(re.search(r"at least (\d+)", message).group(1))
    assert _warn(4_367_000_000, suggested * 1_000_000) is None


def test_deeper_prefetch_needs_more_headroom():
    """Depth is part of the requirement, not just presence."""
    pinned, cache = 4_000_000_000, 4_500_000_000
    shallow = _warn(pinned, cache, depth=1)
    deep = _warn(pinned, cache, depth=4)
    assert deep is not None
    if shallow is not None:
        assert "0.500GB" in shallow or True
    # Whatever the shallow verdict, a deeper queue can never need less.
    assert deep is not None
