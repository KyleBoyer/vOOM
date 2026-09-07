"""Real hook ordering and bounded read-only observations, without MLX."""

import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from runtime import prefill_phase_memory_witness as witness
from runtime.profiles import apply_runtime_profiles

ROOT = Path(__file__).resolve().parents[1]


def source_method():
    tree = ast.parse((ROOT/'runtime/engine.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'StreamingEngine')
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_layer_stationary_qwen4_sweep')


def good_sample(*args):
    return {'available': True, 'metal_active_bytes': 100,
            'process_memory': {'available': True, 'physical_footprint_bytes': 200}}


def observer(**kwargs):
    rows = []
    value = witness.PrefillPhaseMemoryObserver(total_tokens=16, total_layers=1,
        tile_width=4, emit=rows.append, sample=kwargs.pop('sample', good_sample), **kwargs)
    return value, rows


def complete(value):
    for phase, layer, tokens in [('initial_hidden', 0, 16), ('layer_enter', 0, 0),
        ('attention_tile', 0, 16), ('expert_batch', 0, 16), ('output_tile', 0, 16),
        ('layer_complete', 1, 16), ('layer_released', 0, 16)]:
        value.record(None, None, phase=phase, layer_marker=layer, completed_tokens=tokens)
    value.finish()


def test_complete_boundary_coverage_and_no_retained_engine_reference():
    value, rows = observer()
    complete(value)
    assert rows[-1]['coverage_complete'] and rows[-1]['missing_required_boundaries'] == 0
    assert rows[-1]['samples'] == 7 and len(rows) == 8
    assert [r['sample_index'] for r in rows[:-1]] == list(range(1, 8))
    assert not rows[-1]['synchronizes_device'] and not rows[-1]['clears_allocator_cache']
    assert 'target' not in value.__dict__ and 'metal' not in value.__dict__
    json.dumps(rows, allow_nan=False)
    value.finish(); value.record(None, None, phase='layer_enter', layer_marker=0)
    assert len(rows) == 8


def test_partial_trace_cannot_claim_complete_coverage():
    value, rows = observer()
    value.record(None, None, phase='initial_hidden', layer_marker=0, completed_tokens=16)
    value.finish()
    assert not rows[-1]['coverage_complete'] and rows[-1]['missing_required_boundaries'] == 6


def test_middle_tiles_do_not_trigger_observer_reads():
    value, rows = observer(sample=lambda *a: pytest.fail('must not sample'))
    for phase in ('attention_tile', 'output_tile'):
        value.record(None, None, phase=phase, layer_marker=0, completed_tokens=8)
    assert not rows and value.count == 0


def test_sample_cap_is_explicit_and_not_complete(monkeypatch):
    monkeypatch.setattr(witness, 'MAX_SAMPLES', 2)
    value, rows = observer()
    complete(value)
    assert len(rows) == 3 and rows[-1]['capped'] and not rows[-1]['coverage_complete']


@pytest.mark.parametrize('sample', [None, {}, {'available': False}, {'available': True},
    {'available': True, 'process_memory': {'available': False}}])
def test_unavailable_native_or_scalar_sample_cannot_certify_trace(sample):
    value, rows = observer(sample=lambda *a: sample)
    complete(value)
    assert rows[-1]['observation_failed'] and not rows[-1]['coverage_complete']


def test_sampler_exception_is_redacted_and_nonfatal():
    def fail(*args): raise RuntimeError('PRIVATE')
    value, rows = observer(sample=fail)
    complete(value)
    assert not rows[-1]['coverage_complete'] and 'PRIVATE' not in json.dumps(rows)


def test_sink_failure_is_nonfatal_and_disqualifies_coverage():
    value, rows = observer()
    def fail_once(row):
        if row.get('sample_index') == 1: raise OSError('PRIVATE')
        rows.append(row)
    value.emit = fail_once
    complete(value)
    assert rows[-1]['observation_failed'] and not rows[-1]['coverage_complete']


@pytest.mark.parametrize('kwargs', [{'phase': 'PRIVATE'}, {'layer_marker': True},
    {'layer_marker': 2}, {'completed_tokens': 17}, {'completed_tokens': -1},
    {'reported_host_spool_peak_bytes': -1}])
def test_invalid_metadata_does_not_leak_or_change_execution(kwargs):
    value, rows = observer()
    args = dict(phase='layer_enter', layer_marker=0)
    args.update(kwargs)
    value.record(None, None, **args)
    value.finish()
    assert rows[-1]['observation_failed'] and not rows[-1]['coverage_complete']
    assert 'PRIVATE' not in json.dumps(rows)


def test_profile_only_enables_phase_and_required_native_memory_witness():
    env = {}
    apply_runtime_profiles(['prefill-phase-memory-witness'], environ=env)
    assert env == {'VMODEL_PROCESS_MEMORY_WITNESS': '1', witness.FLAG: '1'}


def actual_note_hook(events, phase_memory, *, limit=100, active=20):
    fn = copy.deepcopy(next(n for n in source_method().body
                            if isinstance(n, ast.FunctionDef) and n.name == 'note_spool'))
    factory = ast.parse('def make(self, mx, phase_memory, metal_limit_bytes, on_progress):\n spool_samples=0\n total=16\n return None').body[0]
    factory.body[-1:] = [fn, ast.Return(ast.Name('note_spool', ast.Load()))]
    ns = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[factory], type_ignores=[])), '<real-note-spool>', 'exec'), ns)
    target = SimpleNamespace(cfg=SimpleNamespace(num_hidden_layers=1),
                             _note_true_peak=lambda: events.append('true_peak'))
    metal = SimpleNamespace(get_active_memory=lambda: active, get_peak_memory=lambda: active)
    return ns['make'](target, metal, phase_memory, limit, lambda row: events.append('progress'))


def test_actual_hard_cap_runs_before_new_diagnostics():
    events = []
    phase = SimpleNamespace(record=lambda *a, **k: events.append('observer'))
    hook = actual_note_hook(events, phase, limit=10, active=20)
    with pytest.raises(MemoryError): hook('initial_hidden', layer=0, completed_tokens=16)
    assert events == ['true_peak', 'progress']


@pytest.mark.parametrize('enabled,publish', [(False, True), (True, False), (True, True)])
def test_actual_hook_only_observes_enabled_published_boundary(enabled, publish):
    events = []
    phase = SimpleNamespace(record=lambda *a, **k: events.append('observer')) if enabled else None
    hook = actual_note_hook(events, phase)
    hook('initial_hidden', layer=0, completed_tokens=16, publish=publish)
    assert events == ['true_peak'] + (['progress'] if publish else []) + (['observer'] if enabled and publish else [])


@pytest.mark.parametrize('flag', [None, '0', '', 'true', '1'])
def test_actual_factory_default_off_and_lazy(monkeypatch, flag):
    method = source_method()
    begin = next(i for i,n in enumerate(method.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'total' for t in n.targets))
    end = next(i for i,n in enumerate(method.body) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'input_ids' for t in n.targets))
    factory = ast.parse('def factory(x, self, tile_width):\n import os as _phase_os\n return None').body[0]
    factory.body[-1:] = copy.deepcopy(method.body[begin:end]) + [ast.Return(ast.Name('phase_memory', ast.Load()))]
    ns = {'__package__': 'runtime'}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[factory], type_ignores=[])), '<real-observer-factory>', 'exec'), ns)
    calls = []
    sentinel = object()
    monkeypatch.setattr(witness, 'PrefillPhaseMemoryObserver', lambda **k: calls.append(k) or sentinel)
    if flag is None: monkeypatch.delenv(witness.FLAG, raising=False)
    else: monkeypatch.setenv(witness.FLAG, flag)
    result = ns['factory'](SimpleNamespace(shape=(1,16,4)), SimpleNamespace(cfg=SimpleNamespace(num_hidden_layers=1)), 4)
    assert result is (sentinel if flag == '1' else None)
    assert len(calls) == int(flag == '1')


def test_observer_stripped_sweep_retains_exact_preexisting_operation_tree():
    class StripObserver(ast.NodeTransformer):
        def visit_Import(self, node):
            return None if any(a.asname == '_phase_os' for a in node.names) else node
        def visit_Assign(self, node):
            return None if any(isinstance(t, ast.Name) and t.id == 'phase_memory' for t in node.targets) else self.generic_visit(node)
        def visit_If(self, node):
            return None if any(isinstance(n, ast.Name) and n.id in ('phase_memory', '_phase_os') for n in ast.walk(node.test)) else self.generic_visit(node)
    fn = StripObserver().visit(source_method())
    digest = hashlib.sha256(ast.dump(fn, include_attributes=False).encode()).hexdigest()
    # Method on e873ead, before observer-only edits: no altered model operations,
    # tile boundaries, traversal, eval/clear/reset, routing or scheduling calls.
    assert digest == '904de4d986118add7b47d82ee2a8e368abcb41dc93a7b9e984df832e77bc5823'
