"""Real primitive control flow with symbolic arrays; never an MLX bit proof."""

import ast
from pathlib import Path
import time
from types import SimpleNamespace
import weakref

import pytest


ROOT = Path(__file__).resolve().parents[1]


class Array:
    def __init__(self, shape, dtype='bf16'):
        self.shape, self.dtype = tuple(shape), dtype

    def __getitem__(self, slices):
        shape = tuple(len(range(*s.indices(n))) for n, s in zip(self.shape, slices, strict=True))
        return Array(shape, self.dtype)

    def reshape(self, *shape):
        return Array(shape, self.dtype)


class Raw:
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size


def primitive(*, failure=None, geometry=(248320, 5120), version='0.32.0'):
    events, live_weights, readers = [], [], []
    class Reader:
        def __init__(self, directory):
            self.path = directory/'head.safetensors'
            self.vocab, self.hidden = geometry
            self.extents = [SimpleNamespace(row_bytes=2560, columns=640),
                            SimpleNamespace(row_bytes=160, columns=160)]
            self.closed = False
            readers.append(self)
        def check_unchanged(self):
            if self.closed:
                raise ValueError('closed')
        def read_component(self, index, start, stop):
            self.check_unchanged()
            events.append(('read', index, start, stop))
            if failure == 'read' and index == 1:
                raise EOFError('read failed')
            return Raw((stop-start)*self.extents[index].row_bytes)
        def close(self):
            self.closed = True
    def array(host):
        events.append(('upload', host.shape))
        if failure == 'upload':
            raise RuntimeError('upload failed')
        result = Array(host.shape, 'packed')
        live_weights.append(weakref.ref(result))
        return result
    def concatenate(values, axis):
        shape = list(values[0].shape)
        shape[axis] = sum(v.shape[axis] for v in values)
        events.append(('concat', tuple(shape), axis))
        return Array(shape)
    def quantized(wq, scales, biases, bits, group_size, mode):
        assert biases is None and (bits, group_size, mode) == (4, 32, 'mxfp4')
        assert wq.shape[0] == scales.shape[0] and wq.shape[1] == 4*scales.shape[1]
        return SimpleNamespace(wq=wq, scales=scales)
    def matmul(h, head):
        events.append(('matmul', h.shape, head.wq.shape[0]))
        assert h.shape == (1, 1, 5120)
        if failure == 'project':
            raise RuntimeError('projection failed')
        return Array((1, 1, head.wq.shape[0]))
    class Base:
        def full_scan_telemetry(self):
            return {k:getattr(self, k) for k in ('full_scan_calls', 'full_read_extents',
                'full_bytes_read', 'full_read_ns', 'full_scan_ns')}
    namespace = dict(Path=Path, time=time, StreamedLMHead=Base, HeadRows=Reader,
        importlib=SimpleNamespace(metadata=SimpleNamespace(version=lambda _: version)),
        mx=SimpleNamespace(bfloat16='bf16', array=array, concatenate=concatenate,
            eval=lambda *values: events.append(('eval', tuple(v.shape for v in values)))),
        np=SimpleNamespace(frombuffer=lambda raw, dtype: Array((len(raw),), dtype)),
        QTensor=quantized, matmul=matmul)
    source = ast.parse((ROOT/'runtime/mxfp4_lm_head_stream.py').read_text())
    node = next(n for n in source.body if isinstance(n, ast.ClassDef))
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<actual-mxfp4-head-primitive>', 'exec'), namespace)
    def reserve(size, **kwargs):
        # Successful previous block frames must not keep any packed arrays alive.
        assert not any(ref() is not None for ref in live_weights)
        events.append(('reserve', size, kwargs))
        if failure == 'reserve':
            raise MemoryError('ordinary governor refused')
    return namespace['MXFP4StreamedLMHead'], reserve, events, live_weights, readers


@pytest.mark.parametrize('block_rows', [8192, 32768, 65536])
@pytest.mark.parametrize('positions', [1, 6, 64])
def test_serial_windows_read_each_extent_once_and_keep_singleton_contractions(block_rows, positions):
    cls, reserve, events, live, _ = primitive()
    head = cls(ROOT, reserve=reserve, block_rows=block_rows)
    result = head.logits_serial_rows(Array((1, positions, 5120)))
    assert result.shape == (1, positions, 248320)
    starts = list(range(0, 248320, block_rows))
    reads = [e for e in events if e[0] == 'read']
    assert reads == [('read', i, start, min(248320, start+block_rows))
                     for start in starts for i in (0, 1)]
    calls = [e for e in events if e[0] == 'matmul']
    assert len(calls) == len(starts)*positions and all(e[1] == (1, 1, 5120) for e in calls)
    reservations = [e for e in events if e[0] == 'reserve']
    assert reservations == [('reserve', 2*min(block_rows, 248320-s)*2720,
                             {'reason':'mxfp4-head-row-block'}) for s in starts]
    assert not any(ref() is not None for ref in live)
    stats = head.full_scan_telemetry()
    assert stats['full_scan_calls'] == stats['completed_scan_calls'] == 1
    assert stats['failed_scan_calls'] == 0
    assert stats['full_read_extents'] == 2*len(starts)
    assert stats['full_bytes_read'] == 675430400
    assert stats['reservation_calls'] == len(starts)
    assert all(stats[key] >= 0 for key in ('reservation_ns','upload_ns','projection_ns','full_scan_ns'))
    head.close(); head.close()


def test_ordinary_singleton_and_multiple_scans_accumulate_actual_read_counters():
    cls, reserve, _, _, _ = primitive()
    head = cls(ROOT, reserve=reserve)
    assert head.logits(Array((1, 1, 5120))).shape == (1, 1, 248320)
    head.logits_serial_rows(Array((1, 6, 5120)))
    assert head.completed_scan_calls == head.full_scan_calls == 2
    assert head.full_bytes_read == 2*675430400
    with pytest.raises(ValueError):
        head.logits(Array((1, 6, 5120)))
    assert head.full_scan_calls == 2


@pytest.mark.parametrize('shape,dtype', [((1,5120),'bf16'), ((2,1,5120),'bf16'),
    ((1,0,5120),'bf16'), ((1,65,5120),'bf16'), ((1,1,4096),'bf16'), ((1,1,5120),'f32')])
def test_unsupported_input_never_reads_or_reserves(shape, dtype):
    cls, reserve, events, _, _ = primitive()
    head = cls(ROOT, reserve=reserve)
    with pytest.raises(ValueError, match='BF16'):
        head.logits_serial_rows(Array(shape, dtype))
    assert events == [] and head.full_scan_calls == 0


@pytest.mark.parametrize('failure,error', [('reserve', MemoryError), ('read', EOFError),
    ('upload', RuntimeError), ('project', RuntimeError)])
def test_failure_is_not_a_completed_scan_and_never_retries(failure, error):
    cls, reserve, events, _, readers = primitive(failure=failure)
    head = cls(ROOT, reserve=reserve, block_rows=65536)
    with pytest.raises(error):
        head.logits_serial_rows(Array((1, 6, 5120)))
    stats = head.full_scan_telemetry()
    assert stats['full_scan_calls'] == stats['failed_scan_calls'] == 1
    assert stats['completed_scan_calls'] == 0 and stats['reservation_calls'] == 1
    assert all(e[2] == 0 for e in events if e[0] == 'read')
    if failure == 'reserve':
        assert stats['full_read_extents'] == stats['full_bytes_read'] == 0
    if failure == 'read':
        assert stats['full_read_extents'] == 1 and stats['full_bytes_read'] == 65536*2560
    head.close()
    assert readers[0].closed


@pytest.mark.parametrize('block_rows', [True, 0, -1, 16384, 65537, None, '32768'])
def test_unqualified_block_size_never_opens_source(block_rows):
    cls, reserve, _, _, readers = primitive()
    with pytest.raises(ValueError, match='row size'):
        cls(ROOT, reserve=reserve, block_rows=block_rows)
    assert readers == []


def test_backend_geometry_and_reservation_guards_fail_closed():
    cls, reserve, _, _, readers = primitive(version='0.33.0')
    with pytest.raises(ValueError, match='backend'):
        cls(ROOT, reserve=reserve)
    assert readers == []
    cls, reserve, _, _, readers = primitive(geometry=(248319, 5120))
    with pytest.raises(ValueError, match='geometry'):
        cls(ROOT, reserve=reserve)
    assert readers[0].closed
    cls, _, _, _, readers = primitive()
    with pytest.raises(ValueError, match='reservation'):
        cls(ROOT, reserve=None)
    assert readers == []


def test_closed_descriptor_and_shortlist_never_load_payload():
    cls, reserve, events, _, _ = primitive()
    head = cls(ROOT, reserve=reserve)
    with pytest.raises(NotImplementedError, match='full vocabulary'):
        head.candidate_logits(Array((1,1,5120)), None)
    head.close()
    with pytest.raises(ValueError, match='closed'):
        head.logits(Array((1,1,5120)))
    assert events == []
