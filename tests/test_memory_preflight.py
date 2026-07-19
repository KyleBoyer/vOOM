"""Pure gates for stable-stale-swap admission; no MLX import."""

from __future__ import annotations

from runtime.memory_preflight import PressureSnapshot, evaluate


GB = int(1e9)
MB = int(1e6)


def snapshot(*, available=8 * GB, swap_used=300 * MB,
             swap_free=1700 * MB, swap_out=10 * GB,
             root_free=100 * GB):
    return PressureSnapshot(
        monotonic_s=0.0,
        system_available_bytes=available,
        swap_total_bytes=2 * GB,
        swap_used_bytes=swap_used,
        swap_free_bytes=swap_free,
        swap_in_bytes=0,
        swap_out_bytes=swap_out,
        root_free_bytes=root_free,
        workspace_free_bytes=400 * GB,
    )


def decide(start, end):
    return evaluate(
        start,
        end,
        min_clean_swap_free_bytes=2 * GB,
        min_stable_available_bytes=6 * GB,
        min_root_free_bytes=5 * GB,
        max_swap_growth_bytes=16 * MB,
        max_swap_out_growth_bytes=16 * MB,
    )


def test_clean_swap_passes_without_stable_swap_exception():
    start = snapshot(available=4 * GB, swap_used=0, swap_free=2 * GB)
    result = decide(start, start)
    assert result["passed"]
    assert result["admission_path"] == "clean_swap"


def test_stale_swap_passes_when_available_is_high_and_counters_are_stable():
    start = snapshot()
    end = snapshot(swap_used=start.swap_used_bytes,
                   swap_out=start.swap_out_bytes)
    result = decide(start, end)
    assert result["passed"]
    assert result["admission_path"] == "stable_stale_swap"


def test_stale_swap_fails_when_available_memory_is_low():
    result = decide(snapshot(available=5 * GB), snapshot(available=5 * GB))
    assert not result["passed"]
    assert "system_available_below_stable_swap_minimum" in result["reasons"]


def test_stale_swap_fails_when_swap_usage_grows():
    start = snapshot()
    end = snapshot(swap_used=start.swap_used_bytes + 17 * MB)
    result = decide(start, end)
    assert not result["passed"]
    assert "swap_usage_growing" in result["reasons"]


def test_small_swap_out_churn_is_not_misclassified_as_net_deterioration():
    start = snapshot()
    end = snapshot(swap_out=start.swap_out_bytes + 6 * MB)
    result = decide(start, end)
    assert result["passed"]


def test_stale_swap_fails_when_swap_out_churn_exceeds_bound():
    start = snapshot()
    end = snapshot(swap_out=start.swap_out_bytes + 17 * MB)
    result = decide(start, end)
    assert not result["passed"]
    assert "swap_outs_active" in result["reasons"]


def test_root_floor_remains_mandatory_on_both_paths():
    start = snapshot(root_free=4 * GB)
    result = decide(start, start)
    assert not result["passed"]
    assert "root_free_below_minimum" in result["reasons"]
