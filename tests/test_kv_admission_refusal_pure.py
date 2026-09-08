"""Execute the real KV-admission method with no MLX import or model I/O."""

import ast
import copy
import json
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pytest

from runtime import phase_head_witness


ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def method_node():
    tree = ast.parse((ROOT / 'runtime/engine.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == 'StreamingEngine')
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                and n.name == '_evict_hot_slots_for_admission')


def invoke(error=None, *, margin=400_000_000, system_floor=5600, keep=None):
    calls = []
    def reserve(incoming, **kwargs):
        calls.append((incoming, kwargs))
        if error is not None:
            raise error
    metal = SimpleNamespace(get_active_memory=lambda: 10_280,
                            clear_cache=lambda: calls.append('clear'))
    namespace = {'__package__': 'runtime', 'mx': metal,
        'psutil': SimpleNamespace(virtual_memory=lambda: SimpleNamespace(available=5_660_000_000))}
    module = ast.fix_missing_locations(ast.Module(
        body=[copy.deepcopy(method_node())], type_ignores=[]))
    exec(compile(module, str(ROOT / 'runtime/engine.py'), 'exec'), namespace)
    engine = SimpleNamespace(
        rc=SimpleNamespace(hot_prompt_kv_min_available_mb=system_floor),
        governor=SimpleNamespace(reserve=reserve, critical=5_600_000_000,
            current_ceiling=lambda: 60_010_280, reservations=0),
        _hot_prompt_slots=[], _resident_fast_layers=None,
        _layer_transient_margin=margin,
        _kv_nbytes=lambda value: value.accounted_bytes)
    def run(required=256_000_000, transient=0):
        return namespace['_evict_hot_slots_for_admission'](
            engine, required, keep, 'not-logged', transient_bytes=transient)
    return run, calls, namespace


@pytest.mark.parametrize('margin,transient,kept', [
    (400_000_000, 0, 0), (0, 4_000_000, 0), (0, 17_000_000, 21_000_000)])
def test_exact_projection_components_and_original_exception(monkeypatch, capsys, margin, transient, kept):
    memory = {'available': False, 'metal_active_bytes': 10_280,
              'system_available_bytes': None, 'unavailable_fields': ['system_available_bytes']}
    monkeypatch.setattr(phase_head_witness, 'sample_phase_head_memory', lambda *args: memory)
    error = MemoryError('original refusal')
    keep = SimpleNamespace(accounted_bytes=kept) if kept else None
    run, calls, _ = invoke(error, margin=margin, keep=keep)
    with pytest.raises(MemoryError) as raised:
        run(transient=transient)
    assert raised.value is error
    incoming = 256_000_000 - kept
    assert calls == [(incoming + transient, {'margin': margin})]
    raw = capsys.readouterr().out
    assert raw.startswith('[kv-admission-refusal] ')
    result = json.loads(raw.split('] ', 1)[1])
    assert result['required_total_kv_bytes'] == 256_000_000
    assert result['kept_kv_accounted_bytes'] == kept
    assert result['kept_kv_type'] == ('SimpleNamespace' if keep else None)
    assert result['projected_incoming_bytes'] == incoming
    assert result['projected_transient_bytes'] == transient
    assert result['reservation_bytes'] == incoming + transient
    assert result['selected_compute_margin_bytes'] == margin
    assert result['reservation_margin_bytes'] == margin
    assert result['system_floor_bytes'] == result['governor_critical_bytes'] == 5_600_000_000
    assert result['remaining_hot_slots'] == result['evicted_slots'] == result['evicted_accounted_bytes'] == 0
    assert result['memory'] == memory
    assert 'not-logged' not in raw


def test_operator_floor_margin_remains_authoritative(monkeypatch, capsys):
    monkeypatch.setattr(phase_head_witness, 'sample_phase_head_memory', lambda *args: {})
    error = MemoryError('same')
    run, calls, _ = invoke(error, margin=0, system_floor=6100)
    with pytest.raises(MemoryError) as raised:
        run()
    assert raised.value is error
    assert calls == [(256_000_000, {'margin': 500_000_000})]
    result = json.loads(capsys.readouterr().out.split('] ', 1)[1])
    assert result['selected_compute_margin_bytes'] == 0
    assert result['reservation_margin_bytes'] == 500_000_000


def test_success_has_no_new_observation_or_logging(monkeypatch, capsys):
    def unexpected(*args):
        raise AssertionError('success must not observe')
    monkeypatch.setattr(phase_head_witness, 'sample_phase_head_memory', unexpected)
    run, calls, _ = invoke()
    result = run()
    assert result['projected_incoming_bytes'] == 256_000_000
    assert calls == [(256_000_000, {'margin': 400_000_000})]
    assert capsys.readouterr().out == ''


@pytest.mark.parametrize('kind', ['sample', 'serialization', 'emit'])
def test_observer_failure_never_masks_original_error(monkeypatch, capsys, kind):
    def sample(*args):
        if kind == 'sample':
            raise RuntimeError('observer failed')
        return {'invalid': float('nan')} if kind == 'serialization' else {}
    monkeypatch.setattr(phase_head_witness, 'sample_phase_head_memory', sample)
    error = MemoryError('same')
    run, calls, namespace = invoke(error)
    if kind == 'emit':
        def fail_print(*args, **kwargs):
            raise OSError('closed log')
        namespace['print'] = fail_print
    with pytest.raises(MemoryError) as raised:
        run()
    assert raised.value is error
    assert calls == [(256_000_000, {'margin': 400_000_000})]
    assert capsys.readouterr().out == ''


def test_other_reservation_errors_are_unchanged(monkeypatch, capsys):
    error = ValueError('not a memory refusal')
    run, calls, _ = invoke(error)
    with pytest.raises(ValueError) as raised:
        run()
    assert raised.value is error and len(calls) == 1
    assert capsys.readouterr().out == ''


def test_failure_only_observer_has_no_device_or_policy_mutations():
    node = method_node()
    guarded = next(n for n in ast.walk(node) if isinstance(n, ast.Try)
        and any(isinstance(s, ast.ExceptHandler) and ast.unparse(s.type) == 'MemoryError'
                for s in n.handlers))
    assert ast.unparse(guarded.body[0]) == 'reserve(reservation_bytes, margin=reserve_margin)'
    handler = ast.unparse(guarded.handlers[0])
    for forbidden in ('synchronize(', 'clear_cache(', 'reset_peak_memory(', 'mx.eval(',
                      'reserve(', 'release(', 'sleep('):
        assert forbidden not in handler
