"""Oracle adapter selection, unchanged bytes and fail-closed evidence; no MLX."""

import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from tests.fixtures import huihui_mxfp4_head_rows_gate as gate


def install_fake(monkeypatch, *, stats_change=None, fail=False):
    instances = []
    output = object()
    input_value = object()
    governor = SimpleNamespace(reserve=lambda *a, **kw: None)
    raw = [b'packed bytes', b'scale bytes']
    class Head:
        def __init__(self, directory, *, reserve, block_rows, observe):
            assert directory == gate.MODEL and reserve is governor.reserve
            assert block_rows == 32768
            self.observer = observe
            self.closed = False
            self.calls = 0
            self.reads = []
            self.stats = dict(full_scan_calls=0, completed_scan_calls=0, failed_scan_calls=0,
                full_bytes_read=0, full_read_extents=0, block_rows=block_rows)
            self._reader = SimpleNamespace(identity=(1,2,3,4,5), read_component=self.read)
            instances.append(self)
        def read(self, index, start, stop):
            self.reads.append((index,start,stop))
            self.stats['full_bytes_read'] += len(raw[index])
            self.stats['full_read_extents'] += 1
            return raw[index]
        def logits_serial_rows(self, h):
            assert h is input_value
            self.calls += 1
            self.stats['full_scan_calls'] += 1
            for index in (0,1):
                value = self._reader.read_component(index,0,17)
                assert value is raw[index]  # witness must return original bytes unchanged
            if fail:
                self.stats['failed_scan_calls'] += 1
                raise MemoryError('native failure')
            self.observer('loaded',0,17)
            self.observer('released',0,17)
            self.stats['completed_scan_calls'] += 1
            return output
        def full_scan_telemetry(self):
            return dict(self.stats, **(stats_change or {}))
        def close(self):
            self.closed = True
    monkeypatch.setitem(sys.modules,'runtime.mxfp4_lm_head_stream',
        SimpleNamespace(MXFP4StreamedLMHead=Head))
    return input_value, output, governor, raw, instances


def test_adapter_uses_primitive_once_hashes_actual_returned_bytes_and_closes(monkeypatch):
    h, output, governor, raw, instances = install_fake(monkeypatch)
    seen, document = [], {}
    result, hashes, stats = gate.runtime_candidate(h,32768,governor,seen.append,document,(1,2,3,4,5))
    assert result is output
    assert [x.hexdigest() for x in hashes] == [hashlib.sha256(x).hexdigest() for x in raw]
    assert instances[0].closed and instances[0].calls == 1
    assert seen == ['rows32768:0:loaded','rows32768:0:released']
    assert document['runtime_candidate_last_stats'] == stats


@pytest.mark.parametrize('change',[{'full_scan_calls':0},{'completed_scan_calls':0},
    {'failed_scan_calls':1},{'full_bytes_read':0},{'full_read_extents':0}])
def test_incomplete_or_inconsistent_primitive_stats_cannot_pass(monkeypatch,change):
    h, _, governor, _, instances = install_fake(monkeypatch,stats_change=change)
    document = {}
    with pytest.raises(AssertionError):
        gate.runtime_candidate(h,32768,governor,lambda _:None,document,(1,2,3,4,5))
    assert instances[0].closed and instances[0].calls == 1
    assert all(document['runtime_candidate_last_stats'][k] == v for k,v in change.items())


def test_source_identity_mismatch_is_before_payload_or_projection(monkeypatch):
    h, _, governor, _, instances = install_fake(monkeypatch)
    with pytest.raises(AssertionError):
        gate.runtime_candidate(h,32768,governor,lambda _:None,{},(9,2,3,4,5))
    assert instances[0].closed and instances[0].calls == 0 and not instances[0].reads


def test_native_failure_keeps_partial_stats_closes_and_does_not_retry(monkeypatch):
    h, _, governor, raw, instances = install_fake(monkeypatch,fail=True)
    document = {}
    with pytest.raises(MemoryError,match='native failure'):
        gate.runtime_candidate(h,32768,governor,lambda _:None,document,(1,2,3,4,5))
    assert instances[0].closed and instances[0].calls == 1
    stats = document['runtime_candidate_last_stats']
    assert stats['failed_scan_calls'] == 1 and stats['completed_scan_calls'] == 0
    assert stats['full_bytes_read'] == sum(map(len,raw))


def test_unknown_candidate_kind_is_rejected_before_preflight_read(tmp_path):
    with pytest.raises(AssertionError):
        gate.run(tmp_path/'missing.json',tmp_path/'out.json','unqualified')


@pytest.mark.parametrize('window', [None, {}, {'complete':False,'minimum_available_bytes':8_000_000_000},
    {'complete':True,'minimum_available_bytes':6_699_999_999}])
def test_runtime_oracle_requires_periodic_workspace_evidence_before_mlx(tmp_path,window):
    pre=dict(passed=True, sample_seconds=30, known_transcoders={'passed':True},
        end=dict(monotonic_s=gate.time.monotonic(),root_free_bytes=20_000_000_000))
    if window is not None:
        pre['pressure_window']=window
    path=tmp_path/'preflight.json';path.write_text(json.dumps(pre))
    with pytest.raises(AssertionError):
        gate.run(path,tmp_path/'result.json','runtime')
    assert not (tmp_path/'result.json').exists()


def test_telemetry_failure_still_closes_primitive_descriptor(monkeypatch):
    h, _, governor, _, instances = install_fake(monkeypatch)
    fake_class=sys.modules['runtime.mxfp4_lm_head_stream'].MXFP4StreamedLMHead
    def fail(self):
        raise OSError('telemetry failed')
    monkeypatch.setattr(fake_class,'full_scan_telemetry',fail)
    with pytest.raises(OSError,match='telemetry failed'):
        gate.runtime_candidate(h,32768,governor,lambda _:None,{},(1,2,3,4,5))
    assert instances[0].closed and instances[0].calls == 1
