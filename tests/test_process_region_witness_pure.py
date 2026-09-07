"""Fake native-map regression gates; no native reads, MLX or other tasks."""

from collections import deque
import ctypes
import json

import pytest

from runtime import process_region_witness as region


@pytest.fixture
def native(monkeypatch):
    rows = deque()
    calls = []
    monkeypatch.setattr(region.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(region.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(region, "_self_task_port", lambda lib: 991)

    def fn(port, addr, size, depth, output, count):
        assert port == 991
        def pointer(value, kind):
            return ctypes.cast(value, ctypes.POINTER(kind)).contents
        a, s, d, c = (pointer(addr, ctypes.c_uint64), pointer(size, ctypes.c_uint64),
                       pointer(depth, ctypes.c_uint32), pointer(count, ctypes.c_uint32))
        assert c.value == region.INFO_COUNT
        calls.append((a.value, d.value))
        if not rows:
            return 1
        row = rows.popleft()
        a.value = row.get("address", a.value)
        s.value = row.get("size", 4096)
        d.value = row.get("depth", d.value)
        c.value = row.get("count", region.INFO_COUNT)
        info = pointer(output, region.RegionInfo64)
        for k, v in row.get("info", {}).items():
            setattr(info, k, v)
        return row.get("code", 0)
    monkeypatch.setattr(region, "_bindings", lambda: (object(), fn))
    return rows, calls


def test_sdk_v2_layout():
    assert ctypes.sizeof(region.RegionInfo64) == 76
    assert region.INFO_COUNT == 19
    assert region.RegionInfo64.offset.offset == 12
    assert region.RegionInfo64.user_tag.offset == 20
    assert region.RegionInfo64.object_id_full.offset == 68


def test_leaf_groups_preserve_unknown_tags_and_large_sums(native):
    rows, calls = native
    rows.extend([{"size": 1 << 40, "info": {"user_tag": 3, "pages_resident": 10}},
                 {"info": {"user_tag": 3, "pages_resident": 7}},
                 {"info": {"user_tag": 999, "external_pager": 1, "pages_swapped_out": 5}}])
    result = region.sample_self_regions()
    assert result["available"] and result["coverage_complete"]
    assert result["leaf_regions"] == 3 and result["calls"] == 4
    assert result["groups"][0]["mapped_bytes"] == (1 << 40) + 4096
    assert result["groups"][0]["pages_resident"] == 17
    assert result["groups"][1]["tag_label"] == "UNKNOWN"
    assert result["groups"][1]["pages_swapped_out"] == 5
    assert not result["swapped_out_pages_are_disk_io"]
    serialized = json.dumps(result)
    assert all(key not in serialized for key in ("object_id", '"address"', '"port"'))


def test_nested_submaps_are_entered_without_counting_parent(native):
    rows, calls = native
    rows.extend([{"info": {"is_submap": 1}}, {"info": {"user_tag": 21}},
                 {"depth": 0, "info": {"user_tag": 3}}])
    r = region.sample_self_regions()
    assert r["coverage_complete"] and r["submaps_entered"] == 1
    assert r["leaf_regions"] == 2 and calls == [(0, 0), (0, 1), (4096, 1), (8192, 0)]


@pytest.mark.parametrize("record,reason", [
    ({"code": 5}, "region-query-failed"),
    ({"count": 18}, "unexpected-record-count"),
    ({"size": 0}, "invalid-region-metadata"),
    ({"address": (1 << 64) - 2}, "invalid-region-metadata"),
    ({"info": {"is_submap": 2}}, "invalid-region-metadata"),
    ({"info": {"external_pager": 2}}, "invalid-region-metadata"),
    ({"depth": 33}, "invalid-region-metadata"),
    ({"depth": 32, "info": {"is_submap": 1}}, "submap-depth-limit"),
])
def test_failures_do_not_fabricate_complete_zeroes(native, record, reason):
    native[0].append(record)
    r = region.sample_self_regions()
    assert not r["available"] and not r["coverage_complete"]
    assert r["reason"] == reason and r["groups"] is None


def test_empty_walk_is_unavailable(native):
    r = region.sample_self_regions()
    assert not r["available"] and r["groups"] is None


def test_partial_call_limit_retains_explicit_partial_groups(native):
    native[0].append({"info": {"user_tag": 3}})
    r = region.sample_self_regions(max_calls=1)
    assert r["reason"] == "call-limit" and r["groups"] is None
    assert r["partial_groups"][0]["user_tag"] == 3
    assert not r["coverage_complete"]


def test_group_cardinality_limit_never_silently_drops_unknown_tags(native):
    native[0].extend({'info': {'user_tag': tag}} for tag in range(257))
    r = region.sample_self_regions()
    assert r['reason'] == 'group-limit' and not r['coverage_complete']
    assert r['groups'] is None and len(r['partial_groups']) == 256


def test_time_limit_before_syscall_is_unavailable(native, monkeypatch):
    ticks = iter([0, 2_000_000_000, 2_000_000_001])
    monkeypatch.setattr(region.time, "monotonic_ns", lambda: next(ticks))
    r = region.sample_self_regions()
    assert r["reason"] == "time-limit" and not native[1]


def test_backward_region_is_rejected(native):
    native[0].extend([{}, {"address": 0}])
    r = region.sample_self_regions()
    assert r["reason"] == "invalid-region-metadata" and r["groups"] is None
    assert len(r["partial_groups"]) == 1


def test_exceptions_are_redacted(native, monkeypatch):
    def fail():
        raise OSError("PRIVATE address/path")
    monkeypatch.setattr(region, "_bindings", fail)
    r = region.sample_self_regions()
    assert r["reason"] == "observation-error" and "PRIVATE" not in json.dumps(r)


def test_other_platform_never_calls_native(native, monkeypatch):
    monkeypatch.setattr(region.platform, "system", lambda: "Linux")
    assert region.sample_self_regions()["reason"] == "unsupported-platform"
    assert not native[1]


@pytest.mark.parametrize("kwargs", [
    {"max_calls": True}, {"max_calls": 0}, {"max_calls": 16385},
    {"max_seconds": True}, {"max_seconds": 0}, {"max_seconds": float("nan")},
    {"max_seconds": 6}, {"max_seconds": "1"},
])
def test_bad_limits_raise_without_native_work(native, kwargs):
    with pytest.raises(ValueError):
        region.sample_self_regions(**kwargs)
    assert not native[1]
