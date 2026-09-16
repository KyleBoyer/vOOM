"""Source-labeled system swap counters, distinct from Darwin file paging.

psutil7.2.2 on this Mac reports VM pageins/pageouts as sin/sout. Darwin's
HOST_VM_INFO64 REV1 has separate compressor-segment swapins/swapouts.
No memory mutation, allocation policy, cache trimming, or subprocess calls.
"""
import ctypes
from functools import lru_cache
import os
import platform
from types import SimpleNamespace

import psutil

SOURCE = 'Darwin HOST_VM_INFO64 REV1 swapins/swapouts'
FLAG = 'VMODEL_ACTUAL_SWAP_WITNESS'


class VMStatistics64Rev1(ctypes.Structure):
    _fields_ = [
        *[(n,ctypes.c_uint32) for n in ('free_count','active_count','inactive_count','wire_count')],
        *[(n,ctypes.c_uint64) for n in ('zero_fill_count','reactivations','pageins','pageouts',
            'faults','cow_faults','lookups','hits','purges')],
        ('purgeable_count',ctypes.c_uint32),('speculative_count',ctypes.c_uint32),
        *[(n,ctypes.c_uint64) for n in ('decompressions','compressions','swapins','swapouts')],
        *[(n,ctypes.c_uint32) for n in ('compressor_page_count','throttled_count','external_page_count','internal_page_count')],
        ('total_uncompressed_pages_in_compressor',ctypes.c_uint64),
    ]


REV1_COUNT=ctypes.sizeof(VMStatistics64Rev1)//ctypes.sizeof(ctypes.c_int32)


@lru_cache(maxsize=1)
def _bindings():
    lib=ctypes.CDLL('/usr/lib/libSystem.B.dylib')
    lib.mach_host_self.argtypes=[];lib.mach_host_self.restype=ctypes.c_uint32
    lib.host_statistics64.argtypes=[ctypes.c_uint32,ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),ctypes.POINTER(ctypes.c_uint32)]
    lib.host_statistics64.restype=ctypes.c_int32
    lib.mach_port_deallocate.argtypes=[ctypes.c_uint32,ctypes.c_uint32]
    lib.mach_port_deallocate.restype=ctypes.c_int32
    return lib


def decode(raw, *, returned_count, kernel_code, page_size):
    if kernel_code != 0 or returned_count < REV1_COUNT:
        raise RuntimeError('unavailable or short Darwin VM statistics')
    if type(page_size) is not int or not 1024<=page_size<=1048576 or page_size&(page_size-1):
        raise ValueError('invalid native page size')
    return dict(available=True,source=SOURCE,page_size_bytes=page_size,
        swap_in_bytes=int(raw.swapins)*page_size,swap_out_bytes=int(raw.swapouts)*page_size,
        page_in_bytes=int(raw.pageins)*page_size,page_out_bytes=int(raw.pageouts)*page_size)


def native_counters():
    if platform.system()!='Darwin' or ctypes.sizeof(ctypes.c_void_p)!=8:
        raise RuntimeError('native Darwin swap counters unavailable')
    lib=_bindings();host=lib.mach_host_self()
    try:
        raw=VMStatistics64Rev1();count=ctypes.c_uint32(REV1_COUNT)
        code=lib.host_statistics64(host,4,ctypes.cast(ctypes.byref(raw),
            ctypes.POINTER(ctypes.c_int32)),ctypes.byref(count))
        return decode(raw,returned_count=count.value,kernel_code=code,
            page_size=int(os.sysconf('SC_PAGE_SIZE')))
    finally:
        task=ctypes.c_uint32.in_dll(lib,'mach_task_self_').value
        lib.mach_port_deallocate(task,host)


def sample_native_counters():
    try:
        return native_counters()
    except Exception as error:
        return dict(available=False,source=SOURCE,error_type=type(error).__name__)


def swap_memory():
    """Real swap counters on Darwin; fail closed, never silently substitute paging."""
    raw=psutil.swap_memory()
    if platform.system()=='Darwin':
        counters=native_counters()
    else:
        counters=dict(source='psutil platform swap counters',swap_in_bytes=int(raw.sin),
            swap_out_bytes=int(raw.sout),page_in_bytes=None,page_out_bytes=None)
    return SimpleNamespace(total=raw.total,used=raw.used,free=raw.free,
        sin=counters['swap_in_bytes'],sout=counters['swap_out_bytes'],
        source=counters['source'],page_in_bytes=counters['page_in_bytes'],
        page_out_bytes=counters['page_out_bytes'])


def summarize_native(records):
    """Require complete consistent native coverage, including monotonicity."""
    try:
        assert records
        for row in records:
            assert row['available'] is True and row['source']==SOURCE
            assert type(row['page_size_bytes']) is int and row['page_size_bytes']>0
            assert row['page_size_bytes']==records[0]['page_size_bytes']
            for key in ('swap_in_bytes','swap_out_bytes','page_in_bytes','page_out_bytes'):
                assert type(row[key]) is int and row[key]>=0
                assert row[key]%row['page_size_bytes']==0
        for left,right in zip(records,records[1:]):
            assert all(right[k]>=left[k] for k in ('swap_in_bytes','swap_out_bytes','page_in_bytes','page_out_bytes'))
        return dict(available=True,source=SOURCE,samples=len(records),
            swap_out_growth_bytes=records[-1]['swap_out_bytes']-records[0]['swap_out_bytes'],
            page_out_growth_bytes=records[-1]['page_out_bytes']-records[0]['page_out_bytes'])
    except (AssertionError,KeyError,TypeError,ValueError):
        return dict(available=False,source=SOURCE,reason='missing-invalid-or-incomplete-native-swap-coverage')


def http_identity(before,after):
    return (all(row.get('swap_counter_source')==SOURCE
        and all(type(row.get(k)) is int and row[k]>=0 for k in ('swap_out_bytes','page_out_bytes'))
        for row in (before,after)) and after['swap_out_bytes']>=before['swap_out_bytes']
        and after['page_out_bytes']>=before['page_out_bytes'])
