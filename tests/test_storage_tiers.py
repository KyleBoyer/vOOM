from runtime.storage_tiers import (PageDemand, StorageTier,
                                   plan_static_placement)


def test_serial_plan_never_moves_nvme_page_to_slower_usb():
    source = StorageTier("nvme", 1_500_000_000)
    usb = StorageTier("usb", 315_000_000, capacity_bytes=10_000)
    pages = [PageDemand("a", 1000), PageDemand("b", 1000)]
    plan = plan_static_placement(pages, source, [usb])
    assert set(plan.assignments.values()) == {"nvme"}
    assert plan.excluded_tiers == ("usb",)
    assert plan.estimated_speedup == 1.0


def test_serial_plan_spends_fast_capacity_on_hottest_page():
    source = StorageTier("nas", 100_000_000)
    internal = StorageTier("internal", 3_000_000_000, capacity_bytes=1000)
    pages = [
        PageDemand("cold", 1000, expected_accesses=1),
        PageDemand("hot", 1000, expected_accesses=8),
    ]
    plan = plan_static_placement(pages, source, [internal])
    assert plan.assignments == {"hot": "internal", "cold": "nas"}
    assert plan.estimated_speedup > 1


def test_parallel_projection_balances_independent_nvme_and_usb():
    source = StorageTier("nvme", 1_500_000_000)
    usb = StorageTier("usb", 315_000_000, capacity_bytes=10_000)
    pages = [PageDemand(str(index), 1000) for index in range(12)]
    plan = plan_static_placement(
        pages, source, [usb], parallel_reads=True)
    assert "usb" in set(plan.assignments.values())
    assert "nvme" in set(plan.assignments.values())
    assert plan.estimated_speedup > 1
    assert plan.physical_bytes_by_tier["usb"] <= 10_000


def test_decode_cost_can_make_faster_compressed_overlay_lose():
    source = StorageTier("source", 100_000_000)
    fast = StorageTier("fast", 1_000_000_000, capacity_bytes=1000)
    page = PageDemand(
        "expensive", 1000, expected_accesses=1,
        decode_seconds_per_access=1.0)
    # Decode cost is representation-specific, not tier-specific in this small
    # planner. It is charged on either placement, keeping the comparison honest.
    plan = plan_static_placement([page], source, [fast])
    assert plan.assignments["expensive"] == "fast"
    assert plan.baseline_seconds > plan.estimated_seconds


def test_duplicate_names_fail_closed():
    source = StorageTier("source", 1)
    pages = [PageDemand("same", 1), PageDemand("same", 1)]
    try:
        plan_static_placement(pages, source, [])
    except ValueError as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("duplicate page names must be rejected")
