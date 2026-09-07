"""Explicit Darwin self-map diagnostic, never enabled by serving defaults.

Aggregate leaf mappings by kernel user tag and external-pager flag. These are
non-atomic mapping sums, not deduplicated physical ownership. The kernel's
pages_swapped_out includes compressor-backed pages: it is NOT disk swap I/O.
No addresses, object IDs, paths, payloads, MLX arrays or other tasks are exposed.
"""

from __future__ import annotations

import ctypes
from functools import lru_cache
import math
import os
import platform
import time


class RegionInfo64(ctypes.Structure):
    _layout_ = "ms"
    _pack_ = 4
    _fields_ = [
        ("protection", ctypes.c_int32), ("max_protection", ctypes.c_int32),
        ("inheritance", ctypes.c_int32), ("offset", ctypes.c_uint64),
        *[(name, ctypes.c_uint32) for name in (
            "user_tag", "pages_resident", "pages_shared_now_private",
            "pages_swapped_out", "pages_dirtied", "ref_count")],
        ("shadow_depth", ctypes.c_uint16), ("external_pager", ctypes.c_uint8),
        ("share_mode", ctypes.c_uint8), ("is_submap", ctypes.c_int32),
        ("behavior", ctypes.c_int32), ("object_id", ctypes.c_uint32),
        ("user_wired_count", ctypes.c_uint16), ("flags", ctypes.c_uint16),
        ("pages_reusable", ctypes.c_uint32), ("object_id_full", ctypes.c_uint64),
    ]


INFO_COUNT = ctypes.sizeof(RegionInfo64) // ctypes.sizeof(ctypes.c_uint32)
PAGE_FIELDS = (
    "pages_resident", "pages_shared_now_private", "pages_swapped_out",
    "pages_dirtied", "pages_reusable",
)
# Labels are SDK constants, not an assertion that a particular library owns
# every region with the tag. Unknown tags remain numeric and are never dropped.
TAG_LABELS = {
    1: "MALLOC", 2: "MALLOC_SMALL", 3: "MALLOC_LARGE", 4: "MALLOC_HUGE",
    7: "MALLOC_TINY", 8: "MALLOC_LARGE_REUSABLE", 9: "MALLOC_LARGE_REUSED",
    11: "MALLOC_NANO", 12: "MALLOC_MEDIUM", 13: "MALLOC_PROB_GUARD",
    21: "IOKIT", 88: "IOSURFACE", 100: "IOACCELERATOR",
}


@lru_cache(maxsize=1)
def _bindings():
    lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    fn = lib.mach_vm_region_recurse
    fn.argtypes = [
        ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_uint32),
    ]
    fn.restype = ctypes.c_int32
    return lib, fn


def _self_task_port(lib):
    return ctypes.c_uint32.in_dll(lib, "mach_task_self_").value


def sample_self_regions(*, max_calls=16384, max_seconds=1.0) -> dict:
    """Walk current-task metadata only, with explicit partial/error coverage.

    Time is checked BETWEEN syscalls; this cannot preempt a slow kernel call.
    No task_for_pid, process fork/suspension, mapping changes or memory reads.
    May still add kernel work; use at diagnostic boundaries, not each layer.
    """
    if type(max_calls) is not int or not 1 <= max_calls <= 16384:
        raise ValueError("max_calls must be an integer in 1..16384")
    if (isinstance(max_seconds, bool) or not isinstance(max_seconds, (int, float))
            or not math.isfinite(max_seconds) or not 0 < max_seconds <= 5):
        raise ValueError("max_seconds must be finite in (0, 5]")
    started = time.monotonic_ns()
    result = {
        "schema": "voom.process-regions.v1", "pid": os.getpid(),
        "scope": "current_process_leaf_mapping_sums_not_unique_physical_bytes",
        "source": "Darwin mach_vm_region_recurse V2", "atomic": False,
        "available": False, "coverage_complete": False, "reason": None,
        "unix_time_ns": time.time_ns(), "monotonic_start_ns": started,
        "calls": 0, "leaf_regions": 0, "submaps_entered": 0,
        "kernel_return_code": None, "returned_natural_count": None,
        "groups": None, "partial_groups": None,
        "limits": {"max_calls": max_calls, "max_seconds": max_seconds,
                   "time_bound_checked_between_syscalls": True},
        "swapped_out_pages_are_disk_io": False,
    }
    groups = {}
    try:
        if (platform.system() != "Darwin"
                or platform.machine() not in ("arm64", "x86_64")
                or ctypes.sizeof(ctypes.c_void_p) != 8):
            result["reason"] = "unsupported-platform"
        else:
            lib, fn = _bindings()
            port = _self_task_port(lib)
            address = ctypes.c_uint64(0)
            depth = ctypes.c_uint32(0)
            while True:
                if result["calls"] >= max_calls:
                    result["reason"] = "call-limit"
                    break
                if (time.monotonic_ns() - started) / 1e9 >= max_seconds:
                    result["reason"] = "time-limit"
                    break
                previous = int(address.value)
                size = ctypes.c_uint64(0)
                info = RegionInfo64()
                count = ctypes.c_uint32(INFO_COUNT)
                code = int(fn(port, ctypes.byref(address), ctypes.byref(size),
                              ctypes.byref(depth),
                              ctypes.cast(ctypes.byref(info), ctypes.POINTER(ctypes.c_int32)),
                              ctypes.byref(count)))
                result["calls"] += 1
                result["kernel_return_code"] = code
                result["returned_natural_count"] = int(count.value)
                # KERN_INVALID_ADDRESS is the native map-walk terminator,
                # not a successful empty observation when no leaf was read.
                if code == 1 and result["leaf_regions"]:
                    result.update(available=True, coverage_complete=True)
                    break
                if code != 0:
                    result["reason"] = "region-query-failed"
                    break
                if count.value != INFO_COUNT:
                    result["reason"] = "unexpected-record-count"
                    break
                if (address.value < previous or size.value == 0
                        or address.value + size.value > (1 << 64) - 1
                        or info.is_submap not in (0, 1)
                        or info.external_pager not in (0, 1)
                        or depth.value > 32):
                    result["reason"] = "invalid-region-metadata"
                    break
                if info.is_submap:
                    if depth.value >= 32:
                        result["reason"] = "submap-depth-limit"
                        break
                    depth.value += 1
                    result["submaps_entered"] += 1
                    continue
                key = (int(info.user_tag), int(info.external_pager))
                if key not in groups and len(groups) >= 256:
                    result["reason"] = "group-limit"
                    break
                group = groups.setdefault(key, {
                    "user_tag": key[0], "tag_label": TAG_LABELS.get(key[0], "UNKNOWN"),
                    "external_pager": bool(key[1]), "regions": 0,
                    "mapped_bytes": 0, **dict.fromkeys(PAGE_FIELDS, 0),
                })
                group["regions"] += 1
                group["mapped_bytes"] += int(size.value)
                for field in PAGE_FIELDS:
                    group[field] += int(getattr(info, field))
                result["leaf_regions"] += 1
                address.value += size.value
    except Exception:
        result["reason"] = "observation-error"
    rows = [groups[key] for key in sorted(groups)]
    if result["coverage_complete"]:
        result["groups"] = rows
    elif rows:
        result["partial_groups"] = rows
    result["monotonic_end_ns"] = time.monotonic_ns()
    result["observation_seconds"] = (result["monotonic_end_ns"] - started) / 1e9
    return result
