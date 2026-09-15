"""Exact, ephemeral disk ownership for a scalar-factor rollback base.

Only an immutable proposal-verification snapshot is spilled. The live target
state is never modified. Restoring a prefix still uses the existing recurrence.
"""
import hashlib
from contextlib import ExitStack
from contextvars import ContextVar
from functools import wraps
import math
import os
from pathlib import Path
import tempfile
import time

import mlx.core as mx
import psutil

from .kda_state import KDAStateCache
from .uncached_io import set_darwin_nocache

FLAG = 'VMODEL_QWEN_MTP_FACTOR_BASE_DISK'
ROOT = Path('.kv_spill/qwen-mtp-factor-base')
_owner = ContextVar('qwen_factor_base_owner', default=None)


def managed_bases(function):
    """Close owned snapshots on every synchronous generation exit, including errors."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        with ExitStack() as stack:
            token = _owner.set(stack)
            try:
                return function(*args, **kwargs)
            finally:
                _owner.reset(token)
    return wrapped


class DiskFactorBase:
    def __init__(self, source, governor, stats, root):
        started = time.perf_counter()
        if governor is None or not isinstance(source, KDAStateCache):
            raise ValueError('disk factor base requires governed native KDA state')
        if source.spill_enabled or source.factor_capture_active:
            raise ValueError('disk factor base requires idle resident source')
        self.logical_bytes = source.nbytes()
        if not 0 < self.logical_bytes <= 512_000_000:
            raise ValueError('factor base outside bounded snapshot capacity')
        self._state = [None] * len(source._state)
        self._records = {}
        self._governor = governor
        self._stats = stats
        self._temporary = None
        self._closed = False
        self._maximum_array_bytes = 0
        root = Path(root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        if (psutil.disk_usage(root).free < 10_000_000_000 + 2*self.logical_bytes
                or psutil.disk_usage('/').free < 10_000_000_000):
            raise MemoryError('insufficient free disk for bounded factor snapshot')
        self._temporary = tempfile.TemporaryDirectory(prefix='round-',dir=root)
        try:
            for layer, state in enumerate(source._state):
                history = source._conv[layer]
                if state is None:
                    if history is not None:
                        raise ValueError('conv-only factor snapshot unsupported')
                    continue
                values = [('state',state)]
                if history is not None:
                    values += [(f'conv_{i}',value) for i,value in enumerate(history)
                               if value is not None]
                record = dict(history_length=None if history is None else len(history),
                              arrays={})
                for name,value in values:
                    payload,dtype = KDAStateCache._array_payload(value)
                    self._maximum_array_bytes = max(self._maximum_array_bytes,len(payload))
                    path = Path(self._temporary.name)/f'{layer}-{name}.bin'
                    with path.open('xb',buffering=0) as output:
                        self._add('uncached_descriptors',int(set_darwin_nocache(output.fileno())))
                        if output.write(payload) != len(payload):
                            raise IOError('short factor snapshot write')
                    record['arrays'][name] = (path,tuple(value.shape),dtype,len(payload),
                                              hashlib.sha256(payload).digest())
                    self._add('bytes_written',len(payload))
                    del payload
                self._records[layer] = record
            self._add('rounds',1)
            self._add('write_s', time.perf_counter() - started)
            stats['logical_bytes_peak'] = max(stats.get('logical_bytes_peak',0),self.logical_bytes)
            stats['snapshot_resident_bytes_peak'] = max(stats.get('snapshot_resident_bytes_peak',0),self.nbytes())
        except BaseException:
            self.close()
            raise

    def _add(self,key,value):
        self._stats[key] = self._stats.get(key,0)+value

    def has_state(self,layer):
        return layer in self._records

    def nbytes(self):
        # Metadata and files are not resident Metal arrays or allocation credit.
        return 0

    def fork(self):
        started = time.perf_counter()
        if self._closed:
            raise ValueError('factor snapshot is closed')
        # Match the existing factor-restore admission: one full endpoint plus
        # bounded per-layer scratch. Never repopulate this disk base itself.
        self._governor.reserve(self.logical_bytes+6*self._maximum_array_bytes,
                               margin=0,reason='qwen-mtp-base-reload')
        result=KDAStateCache(len(self._state))
        for layer,record in self._records.items():
            loaded={}
            for name,(path,shape,dtype,size,digest) in record['arrays'].items():
                itemsize={'bf16':2,'f16':2,'f32':4}[dtype]
                if math.prod(shape)*itemsize != size:
                    raise ValueError('factor snapshot geometry mismatch')
                with path.open('rb',buffering=0) as source:
                    self._add('uncached_descriptors',int(set_darwin_nocache(source.fileno())))
                    if os.fstat(source.fileno()).st_size != size:
                        raise IOError('factor snapshot size mismatch')
                    payload=source.read(size+1)
                if len(payload)!=size or hashlib.sha256(payload).digest()!=digest:
                    raise IOError('factor snapshot checksum mismatch')
                loaded[name]=KDAStateCache._array_from_payload(payload,shape,dtype)
                mx.eval(loaded[name])
                self._add('bytes_read',len(payload))
                del payload
            result.set_state(layer,loaded['state'])
            if record['history_length'] is not None:
                result.set_conv_history(layer,tuple(loaded.get(f'conv_{i}')
                    for i in range(record['history_length'])))
        self._add('reloads',1)
        self._add('reload_s', time.perf_counter() - started)
        return result

    def close(self):
        if self._closed:
            return
        self._closed=True
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary=None
        self._records.clear()
        self._add('closed',1)


def create_base(source,target):
    value=os.environ.get(FLAG,'0')
    if value not in ('0','1'):
        raise ValueError(FLAG+' must be 0 or 1')
    if value=='0':
        return source.fork()
    owner = _owner.get()
    if owner is None:
        raise RuntimeError('disk factor base requires a generation owner')
    stats=getattr(target,'_qwen_mtp_base_spill_stats',None)
    if stats is None:
        stats=target._qwen_mtp_base_spill_stats={}
    base = DiskFactorBase(source,target.governor,stats,ROOT)
    owner.callback(base.close)
    return base


def base_bytes(base):
    return base.logical_bytes if isinstance(base,DiskFactorBase) else base.nbytes()


def has_state(base,layer):
    return base.has_state(layer) if isinstance(base,DiskFactorBase) else base.state(layer) is not None


def close_base(base):
    if isinstance(base,DiskFactorBase):
        base.close()
