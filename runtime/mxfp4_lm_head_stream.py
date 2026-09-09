"""Unselected Huihui MXFP4 head primitive; engine/profile wiring is not enabled.

Row copies preserve packed values and group scales. Actual-head synthetic-input
bit evidence is not a full-model/state or serving qualification. Keep singleton
rank-three contractions, not StreamedLMHead's BF16 rank-two contractions.
"""

import importlib.metadata
from pathlib import Path
import time

import mlx.core as mx
import numpy as np

from .lm_head_stream import StreamedLMHead
from .mxfp4_head_rows import HeadRows
from .quant import QTensor, matmul


class MXFP4StreamedLMHead(StreamedLMHead):
    """Explicit primitive for the locally tested native MXFP4 head geometry.

    A reserve callback is mandatory and must use the ordinary live governor.
    It receives host-staging plus tile bytes; no cache release is RAM credit.
    This does not select a WeightStore overlay, provide reranked/shortlist
    logits, or make the existing phase-scoped whole-head lease compatible.
    """

    def __init__(self, model_dir, *, reserve, block_rows=32768, observe=None):
        if type(block_rows) is not int or block_rows not in (8192, 32768, 65536):
            raise ValueError('MXFP4 head requires an explicitly tested row size')
        if not callable(reserve):
            raise ValueError('MXFP4 head requires an ordinary governor reservation callback')
        if observe is not None and not callable(observe):
            raise ValueError('MXFP4 head observer must be callable or None')
        if importlib.metadata.version('mlx') != '0.32.0':
            raise ValueError('MXFP4 head requires the tested MLX 0.32.0 backend')
        self.model_dir = Path(model_dir).resolve()
        self._reader = HeadRows(self.model_dir)
        try:
            if (self._reader.vocab, self._reader.hidden) != (248320, 5120):
                raise ValueError('MXFP4 head requires the tested Huihui geometry')
            self.vocab, self.hidden = self._reader.vocab, self._reader.hidden
            self.row_bytes = sum(e.row_bytes for e in self._reader.extents)
            self.block_rows = block_rows
            self._reserve = reserve
            self._observe = observe
            self.name = self.real_name = 'lm_head.weight'
            self.path = self._reader.path
            for name in ('full_scan_calls', 'full_read_extents', 'full_bytes_read',
                         'full_read_ns', 'full_scan_ns', 'completed_scan_calls',
                         'failed_scan_calls', 'reservation_calls', 'reservation_ns',
                         'upload_ns', 'projection_ns', 'candidate_read_calls',
                         'candidate_read_extents', 'candidate_rows_requested',
                         'candidate_unique_rows_read', 'candidate_bytes_read',
                         'candidate_recall_full_scan_calls', 'candidate_recall_full_scan_bytes'):
                setattr(self, name, 0)
        except BaseException:
            self._reader.close()
            raise

    def close(self):
        self._reader.close()

    def _validate_hidden(self, h, *, singleton):
        if (len(h.shape) != 3 or h.shape[0] != 1 or h.shape[-1] != self.hidden
                or not 1 <= h.shape[1] <= 64 or (singleton and h.shape[1] != 1)
                or h.dtype != mx.bfloat16):
            raise ValueError('MXFP4 head requires BF16 (1, positions, 5120); '
                             'ordinary logits is singleton, serial windows are 1..64 positions')
        # 64 bounds one verifier projection window, NOT request output tokens.
        self._reader.check_unchanged()

    def _project_block(self, h, start, stop):
        # Function scope drops EVERY packed/scales/QTensor local after evaluated
        # logits, before the next block's reservation. Do not yield a live head.
        incoming = 2 * (stop - start) * self.row_bytes
        t0 = time.perf_counter_ns()
        self.reservation_calls += 1
        try:
            self._reserve(incoming, reason='mxfp4-head-row-block')
        finally:
            self.reservation_ns += time.perf_counter_ns() - t0
        arrays = []
        for index, extent in enumerate(self._reader.extents):
            t0 = time.perf_counter_ns()
            try:
                raw = self._reader.read_component(index, start, stop)
            finally:
                self.full_read_ns += time.perf_counter_ns() - t0
            self.full_read_extents += 1
            self.full_bytes_read += len(raw)
            t0 = time.perf_counter_ns()
            try:
                host = np.frombuffer(raw, dtype='<u4' if index == 0 else 'u1')
                value = mx.array(host.reshape(stop-start, extent.columns))
                mx.eval(value)
                arrays.append(value)
            finally:
                self.upload_ns += time.perf_counter_ns() - t0
            del raw, host, value
        head = QTensor(arrays[0], arrays[1], None, 4, 32, 'mxfp4')
        if self._observe is not None:
            self._observe('loaded', start, stop)
        t0 = time.perf_counter_ns()
        try:
            values = []
            for position in range(h.shape[1]):
                value = matmul(h[:, position:position+1, :], head)
                mx.eval(value)
                values.append(value)
            result = mx.concatenate(values, axis=1)
            mx.eval(result)
            return result
        finally:
            self.projection_ns += time.perf_counter_ns() - t0

    def logits(self, h):
        self._validate_hidden(h, singleton=True)
        return self.logits_serial_rows(h)

    def logits_serial_rows(self, h):
        self._validate_hidden(h, singleton=False)
        started = time.perf_counter_ns()
        self.full_scan_calls += 1
        try:
            mx.eval(h)
            chunks = []
            for start in range(0, self.vocab, self.block_rows):
                stop = min(self.vocab, start+self.block_rows)
                chunks.append(self._project_block(h, start, stop))
                if self._observe is not None:
                    self._observe('released', start, stop)
            result = mx.concatenate(chunks, axis=-1)
            mx.eval(result)
            self._reader.check_unchanged()
            self.completed_scan_calls += 1
            return result
        except BaseException:
            self.failed_scan_calls += 1
            raise
        finally:
            self.full_scan_ns += time.perf_counter_ns() - started

    def full_scan_telemetry(self):
        return dict(super().full_scan_telemetry(),
            completed_scan_calls=self.completed_scan_calls,
            failed_scan_calls=self.failed_scan_calls,
            reservation_calls=self.reservation_calls,
            reservation_ns=self.reservation_ns, upload_ns=self.upload_ns,
            projection_ns=self.projection_ns, block_rows=self.block_rows)

    def candidate_logits(self, h, indices):
        raise NotImplementedError('native MXFP4 head streaming requires full vocabulary')
