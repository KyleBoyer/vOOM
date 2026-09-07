"""Opt-in, scalar-only Darwin self-task memory observations; no ML framework.

ABI: Apple's mach/task_info.h, TASK_VM_INFO through REV1, pack(4).
compressed is the internal_compressed ledger balance, NOT physical compressor
storage or this process's disk swap-out bytes. Footprint/RSS/Metal are overlapping
views; never add them or infer global swap ownership from their correlation.
"""

from __future__ import annotations

import ctypes
from functools import lru_cache
import json
import os
import platform
import time


FLAG = "VMODEL_PROCESS_MEMORY_WITNESS"
TASK_VM_INFO = 22
MAX_SAMPLES = 4096
MIN_INTERVAL_SECONDS = 2.0


class TaskVMInfoRev1(ctypes.Structure):
    # Explicit packed layout in Python 3.14+. Integer-only REV1 ABI is checked
    # independently against the installed Darwin SDK by the CPU probe.
    _layout_ = "ms"
    _pack_ = 4
    _fields_ = [
        ("virtual_size", ctypes.c_uint64),
        ("region_count", ctypes.c_int32), ("page_size", ctypes.c_int32),
        *[(name, ctypes.c_uint64) for name in (
            "resident_size", "resident_size_peak", "device", "device_peak",
            "internal", "internal_peak", "external", "external_peak",
            "reusable", "reusable_peak", "purgeable_volatile_pmap",
            "purgeable_volatile_resident", "purgeable_volatile_virtual",
            "compressed", "compressed_peak", "compressed_lifetime",
            "phys_footprint")],
    ]


REV1_COUNT = ctypes.sizeof(TaskVMInfoRev1) // ctypes.sizeof(ctypes.c_uint32)
FIELDS = {
    "resident_bytes": "resident_size",
    "physical_footprint_bytes": "phys_footprint",
    "internal_bytes": "internal", "external_bytes": "external",
    "reusable_bytes": "reusable",
    "internal_compressed_ledger_bytes": "compressed",
    "internal_compressed_ledger_peak_bytes": "compressed_peak",
    "internal_compressed_ledger_lifetime_bytes": "compressed_lifetime",
}


@lru_cache(maxsize=1)
def _bindings():
    # Fixed system library; no shell, compiler, task_for_pid or privilege request.
    lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    fn = lib.task_info
    fn.argtypes = [ctypes.c_uint32, ctypes.c_uint32,
                   ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_uint32)]
    fn.restype = ctypes.c_int32
    return lib, fn


def _self_task_port(lib):
    return ctypes.c_uint32.in_dll(lib, "mach_task_self_").value


def sample_self_memory() -> dict:
    """Read only the current task. Failed/short records stay null, never zero."""
    started = time.monotonic_ns()
    result = {
        "schema": "voom.process-memory.v1", "scope": "current_process_only",
        "pid": os.getpid(), "unix_time_ns": time.time_ns(),
        "monotonic_start_ns": started, "atomic": False,
        "source": "Darwin TASK_VM_INFO REV1",
        "available": False, "reason": None,
        "kernel_return_code": None, "returned_natural_count": None,
        **dict.fromkeys(FIELDS),
    }
    try:
        if (platform.system() != "Darwin"
                or platform.machine() not in ("arm64", "x86_64")
                or ctypes.sizeof(ctypes.c_void_p) != 8):
            result["reason"] = "unsupported-platform"
        else:
            lib, fn = _bindings()
            # Read the self port afresh; no cached PID/port that can outlive fork.
            port = _self_task_port(lib)
            info = TaskVMInfoRev1()
            count = ctypes.c_uint32(REV1_COUNT)
            code = fn(port, TASK_VM_INFO,
                      ctypes.cast(ctypes.byref(info), ctypes.POINTER(ctypes.c_int32)),
                      ctypes.byref(count))
            result.update(kernel_return_code=int(code),
                          returned_natural_count=int(count.value))
            if code != 0:
                result["reason"] = "task-info-failed"
            elif count.value != REV1_COUNT:
                result["reason"] = "unexpected-record-count"
            else:
                result.update({key: int(getattr(info, field))
                               for key, field in FIELDS.items()})
                result["available"] = True
    except Exception:
        result["reason"] = "observation-error"
    result["monotonic_end_ns"] = time.monotonic_ns()
    result["observation_seconds"] = (result["monotonic_end_ns"] - started) / 1e9
    return result


class GovernorProcessMemoryObserver:
    """Bounded JSONL to the existing server log, on the existing governor thread.

    Call AFTER the safety response, not before making an admission decision.
    No new thread, engine references, tensor traversal, reset, sync or clearance.
    Existing governor inputs precede the self sample and are explicitly non-atomic.
    A hard cap emits a terminal marker; missing coverage is not a pressure pass.
    """

    def __init__(self):
        self.count = 0
        self.last_monotonic = None
        self.capped = False

    def record(self, *, governor_monotonic_s, system_available_bytes,
               system_swap_used_bytes, system_swap_out_bytes, metal_active_bytes,
               cache_budget_bytes_after_response, swap_pressure_response):
        now = float(governor_monotonic_s)
        if self.capped or (self.last_monotonic is not None
                          and now - self.last_monotonic < MIN_INTERVAL_SECONDS):
            return
        self.last_monotonic = now
        if self.count >= MAX_SAMPLES:
            self.capped = True
            print('[process-memory] ' + json.dumps({
                "schema": "voom.process-memory-trace-end.v1", "pid": os.getpid(),
                "reason": "sample-limit", "samples": self.count,
                "governor_monotonic_s": now, "coverage_complete": False}), flush=True)
            return
        self.count += 1
        try:
            memory = sample_self_memory()
        except Exception:
            memory = {"available": False, "reason": "observation-error"}
        row = {
            "schema": "voom.governor-process-memory.v1", "sample_index": self.count,
            "atomic": False, "governor_monotonic_s": now,
            "system_available_bytes": system_available_bytes,
            "system_swap_used_bytes": system_swap_used_bytes,
            "system_swap_out_bytes": system_swap_out_bytes,
            "metal_active_bytes": metal_active_bytes,
            "cache_budget_bytes_after_response": cache_budget_bytes_after_response,
            "swap_pressure_response": bool(swap_pressure_response),
            "process": memory,
        }
        print('[process-memory] ' + json.dumps(row, allow_nan=False,
                                             separators=(",", ":")), flush=True)


def process_memory_observer_from_environment():
    return GovernorProcessMemoryObserver() if os.environ.get(FLAG) == "1" else None
