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

import psutil


FLAG = "VMODEL_PROCESS_MEMORY_WITNESS"
RECLAMATION_FLAG = "VMODEL_GOVERNOR_RECLAMATION_WITNESS"
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


def sample_reclamation_alignment(metal, input_sample_bounds_ns) -> dict:
    """Read-only post-response comparison; never replace admission inputs.

    The two brackets identify non-atomic input and subsequent observation
    intervals. Differences are not causal attribution or proof that GPU work
    has retired. No evaluation, synchronization or allocator mutation occurs.
    """
    started = time.monotonic_ns()
    fields = ("system_available_bytes", "metal_active_bytes", "metal_cache_bytes")
    result = dict(schema="voom.governor-reclamation-alignment.v1", atomic=False,
                  available=False, reason=None,
                  input_sample_start_ns=None, input_sample_end_ns=None,
                  observation_start_ns=started, **dict.fromkeys(fields))
    try:
        valid = (isinstance(input_sample_bounds_ns, tuple)
                 and len(input_sample_bounds_ns) == 2
                 and all(type(v) is int for v in input_sample_bounds_ns)
                 and 0 <= input_sample_bounds_ns[0] <= input_sample_bounds_ns[1] <= started)
        if not valid:
            result["reason"] = "invalid-input-bracket"
        else:
            result.update(input_sample_start_ns=input_sample_bounds_ns[0],
                          input_sample_end_ns=input_sample_bounds_ns[1])
            values = (psutil.virtual_memory().available,
                      metal.get_active_memory(), metal.get_cache_memory())
            if not all(type(v) is int and v >= 0 for v in values):
                result["reason"] = "invalid-observation"
            else:
                result.update(zip(fields, values))
                result["available"] = True
    except Exception:
        result["reason"] = "observation-error"
    result["observation_end_ns"] = time.monotonic_ns()
    return result


def summarize_reclamation_alignment(records) -> dict:
    """Missing/error/capped coverage is never a successful aligned trace."""
    rows = list(records)
    result = dict(available=False, passed=False, samples=len(rows))
    if not rows:
        return result
    delays, input_spans, observations = [], [], []
    for row in rows:
        alignment = row.get("reclamation_alignment")
        process = row.get("process")
        if not isinstance(alignment, dict) or not isinstance(process, dict):
            return result
        if (alignment.get("schema") != "voom.governor-reclamation-alignment.v1"
                or alignment.get("available") is not True
                or alignment.get("atomic") is not False
                or process.get("available") is not True):
            return result
        stamps = [alignment.get(key) for key in (
            "input_sample_start_ns", "input_sample_end_ns",
            "observation_start_ns", "observation_end_ns")]
        stamps += [process.get("monotonic_start_ns"), process.get("monotonic_end_ns")]
        if (not all(type(v) is int and v >= 0 for v in stamps)
                or stamps != sorted(stamps)
                or not all(type(alignment.get(key)) is int and alignment[key] >= 0
                           for key in ("system_available_bytes", "metal_active_bytes", "metal_cache_bytes"))):
            return result
        input_spans.append(stamps[1] - stamps[0])
        delays.append(stamps[2] - stamps[1])
        observations.append(stamps[5] - stamps[2])
    result.update(available=True, passed=True,
                  maximum_input_sample_span_ns=max(input_spans),
                  maximum_after_response_delay_ns=max(delays),
                  maximum_observation_span_ns=max(observations))
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
        self.reclamation_enabled = os.environ.get(RECLAMATION_FLAG) == "1"
        self.host_activity_sample = None
        if os.environ.get('VMODEL_HOST_ACTIVITY_WITNESS') == '1':
            from .host_activity_witness import sample_known_transcoders
            self.host_activity_sample = sample_known_transcoders

    def record(self, *, governor_monotonic_s, system_available_bytes,
               system_swap_used_bytes, system_swap_out_bytes, metal_active_bytes,
               cache_budget_bytes_after_response, swap_pressure_response,
               input_sample_bounds_ns=None, metal=None):
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
        alignment = None
        if self.reclamation_enabled:
            alignment = sample_reclamation_alignment(metal, input_sample_bounds_ns)
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
        if alignment is not None:
            row["reclamation_alignment"] = alignment
        if self.host_activity_sample is not None:
            try:
                row['known_transcoders'] = self.host_activity_sample()
            except Exception:
                row['known_transcoders'] = {'available': False, 'reason': 'inventory-error'}
        print('[process-memory] ' + json.dumps(row, allow_nan=False,
                                             separators=(",", ":")), flush=True)


def process_memory_observer_from_environment():
    return GovernorProcessMemoryObserver() if os.environ.get(FLAG) == "1" else None
