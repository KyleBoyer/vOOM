"""Explicit request-boundary reclamation of unreachable/unused CPU storage.

Darwin ABI: apple-oss-distributions/libmalloc/include/malloc/malloc.h.
malloc_zone_pressure_relief(NULL, 0) examines this process's zones for unused
storage; it does not free live allocations. Returned bytes are the allocator's
report, not proven physical RAM or process-attributed system swap. No MLX,
model/cache/state mutation, admission-policy change, or background thread.
"""
from __future__ import annotations

import ctypes
from functools import lru_cache
import gc
import json
import os
import platform
import time

import psutil

from .process_memory_witness import sample_self_memory

FLAG = 'VMODEL_HOST_ALLOCATOR_RELIEF'
PREFIX = '[host-allocator-relief] '
SCHEMA = 'voom.host-allocator-relief.v1'


@lru_cache(maxsize=1)
def _bindings():
    if platform.system() != 'Darwin' or ctypes.sizeof(ctypes.c_void_p) != 8:
        raise RuntimeError('unsupported host allocator')
    lib = ctypes.CDLL('/usr/lib/libSystem.B.dylib')
    fn = lib.malloc_zone_pressure_relief
    fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    fn.restype = ctypes.c_size_t
    return lib, fn


def _sample():
    return dict(monotonic_ns=time.monotonic_ns(),
        system_available_bytes=int(psutil.virtual_memory().available),
        process=sample_self_memory())


def reclaim():
    """Call under the serving lock, with no live generation frame to unwind."""
    result = dict(schema=SCHEMA, available=False, error_type=None,
        scope='current-process unused CPU allocator storage; non-atomic samples',
        pid=os.getpid(), started_ns=time.monotonic_ns(),
        gc_collected=None, allocator_reported_released_bytes=None,
        before=None, after_gc=None, after_relief=None)
    try:
        _lib, relief = _bindings()
        result['before'] = _sample()
        result['gc_collected'] = int(gc.collect())
        result['after_gc'] = _sample()
        result['allocator_reported_released_bytes'] = int(relief(None, 0))
        result['after_relief'] = _sample()
        result['available'] = all(result[k]['process'].get('available') is True
            for k in ('before', 'after_gc', 'after_relief'))
    except Exception as error:
        # Failure never authorizes an allocation or replaces the governor.
        result['error_type'] = type(error).__name__
    result['ended_ns'] = time.monotonic_ns()
    return result


def run_if_enabled(environ=None):
    env = os.environ if environ is None else environ
    if env.get(FLAG, '0') != '1':
        return None
    result = reclaim()
    print(PREFIX + json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return result


def summarize(log_text, *, expected_count):
    """Coverage does not imply that reclamation helped or a request completed."""
    try:
        records = [json.loads(line[len(PREFIX):]) for line in log_text.splitlines()
            if line.startswith(PREFIX)]
        assert type(expected_count) is int and expected_count > 0
        assert len(records) == expected_count
        for r in records:
            assert r['schema'] == SCHEMA and r['available'] is True and r['error_type'] is None
            assert all(type(r[k]) is int and r[k] >= 0 for k in (
                'started_ns','ended_ns','gc_collected','allocator_reported_released_bytes','pid'))
            previous = r['started_ns']
            for name in ('before','after_gc','after_relief'):
                sample = r[name]
                assert type(sample['monotonic_ns']) is int and previous <= sample['monotonic_ns']
                assert type(sample['system_available_bytes']) is int and sample['system_available_bytes'] >= 0
                native = sample['process']
                assert native['available'] is True and native['pid'] == r['pid']
                assert sample['monotonic_ns'] <= native['monotonic_start_ns'] <= native['monotonic_end_ns']
                previous = native['monotonic_end_ns']
            assert previous <= r['ended_ns']
        return dict(passed=True, samples=len(records),
            allocator_reported_released_bytes=sum(r['allocator_reported_released_bytes'] for r in records),
            gc_collected=sum(r['gc_collected'] for r in records),
            wall_seconds=sum(r['ended_ns']-r['started_ns'] for r in records)/1e9)
    except (AssertionError, KeyError, TypeError, ValueError):
        return dict(passed=False, reason='missing-invalid-or-incomplete-coverage')
