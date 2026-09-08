"""Execute the real failure-only verifier hook without MLX or model I/O."""

import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from runtime import phase_head_witness


ROOT = Path(__file__).resolve().parents[1]


def hook():
    tree = ast.parse((ROOT / 'runtime/engine.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
        and n.name == 'StreamingEngine')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
        and n.name == 'forward_tokens_serial_positions')
    node = next(n for n in ast.walk(method) if isinstance(n, ast.Try)
        and any(isinstance(c, ast.Constant) and c.value == 'serial-verify-transient'
            for c in ast.walk(ast.Module(body=n.body, type_ignores=[]))))
    fn = ast.parse('def invoke(self, layer, verifier_positions, offset, kv, qwen_family=True): pass').body[0]
    fn.body = [copy.deepcopy(node)]
    namespace = {'__package__': 'runtime', 'mx': object()}
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    exec(compile(module, str(ROOT / 'runtime/engine.py'), 'exec'), namespace)
    return namespace['invoke'], node


def target(error=None):
    calls = []
    def reserve(incoming, **kwargs):
        calls.append((incoming, kwargs))
        if error is not None:
            raise error
    return SimpleNamespace(governor=SimpleNamespace(reserve=reserve),
        _layer_transient=330_000_000, _layer_transient_margin=0,
        _transient_layer_signature=lambda layer: 'linear_attention+dense',
        _serial_verify_layer_transient_counts={(5, 'linear_attention+dense'): 7},
        _serial_verify_layer_transient={(5, 'linear_attention+dense'): 330_000_000},
        _layer_transient_by_signature={(1, 'linear_attention+dense'): 12_000_000}), calls


def kv():
    return SimpleNamespace(nbytes=lambda: 255_000_000, max_bytes=256_000_000)


def test_success_never_samples_and_preserves_selected_margin(monkeypatch, capsys):
    def unexpected(*args):
        raise AssertionError('no success-path observation')
    monkeypatch.setattr(phase_head_witness, 'sample_phase_head_memory', unexpected)
    engine, calls = target()
    invoke, _ = hook()
    invoke(engine, 60, 5, 6010, SimpleNamespace(nbytes=unexpected))
    assert calls == [(330_000_000, dict(margin=0, reason='serial-verify-transient'))]
    assert capsys.readouterr().out == ''


def test_failure_observes_actual_layer_width_kv_and_keeps_original_error(monkeypatch, capsys):
    memory = dict(available=False, metal_active_bytes=792454568,
        system_available_bytes=None, unavailable_fields=['system_available_bytes'])
    monkeypatch.setattr(phase_head_witness, 'sample_phase_head_memory', lambda *args: memory)
    error = MemoryError('original')
    engine, calls = target(error)
    original = dict(engine._serial_verify_layer_transient)
    invoke, _ = hook()
    with pytest.raises(MemoryError) as raised:
        invoke(engine, 60, 5, 6010, kv())
    assert raised.value is error and len(calls) == 1
    assert engine._serial_verify_layer_transient == original
    raw = capsys.readouterr().out
    assert raw.startswith('[qwen35-serial-transient-admission] ')
    observed = json.loads(raw.split('] ', 1)[1])
    assert (observed['layer'], observed['verifier_positions'], observed['start_offset']) == (60, 5, 6010)
    assert observed['selected_compute_scratch_bytes'] == 330_000_000
    assert observed['selected_compute_margin_bytes'] == 0
    assert observed['matching_verify_observations'] == 7
    assert observed['matching_verify_scratch_bytes'] == 330_000_000
    assert observed['one_position_scratch_bytes'] == 12_000_000
    assert observed['target_kv_resident_logical_bytes'] == 255_000_000
    assert observed['target_kv_budget_bytes'] == 256_000_000
    assert observed['memory'] == memory


def test_missing_matching_width_is_null_not_a_fabricated_sample(monkeypatch, capsys):
    monkeypatch.setattr(phase_head_witness, 'sample_phase_head_memory', lambda *args: {})
    engine, _ = target(MemoryError())
    invoke, _ = hook()
    with pytest.raises(MemoryError):
        invoke(engine, 60, 3, 6010, kv())
    observed = json.loads(capsys.readouterr().out.split('] ', 1)[1])
    assert observed['matching_verify_observations'] is None
    assert observed['matching_verify_scratch_bytes'] is None
    assert observed['one_position_scratch_bytes'] == 12_000_000


@pytest.mark.parametrize('kind', ['sample', 'kv', 'serialization', 'emit'])
def test_observer_failure_never_masks_memory_refusal(monkeypatch, capsys, kind):
    def sample(*args):
        if kind == 'sample':
            raise RuntimeError('observer')
        return dict(bad=float('nan')) if kind == 'serialization' else {}
    monkeypatch.setattr(phase_head_witness, 'sample_phase_head_memory', sample)
    invoke, _ = hook()
    if kind == 'emit':
        def fail(*args, **kwargs):
            raise OSError('closed')
        invoke.__globals__['print'] = fail
    error = MemoryError('keep')
    engine, calls = target(error)
    with pytest.raises(MemoryError) as raised:
        invoke(engine, 60, 5, 6010, object() if kind == 'kv' else kv())
    assert raised.value is error and len(calls) == 1
    assert capsys.readouterr().out == ''


@pytest.mark.parametrize('error,family', [(ValueError('other'), True), (MemoryError('other model'), False)])
def test_unrelated_exception_or_model_is_not_observed(error, family, monkeypatch, capsys):
    def unexpected(*args):
        raise AssertionError('not a Qwen memory refusal')
    monkeypatch.setattr(phase_head_witness, 'sample_phase_head_memory', unexpected)
    engine, calls = target(error)
    invoke, _ = hook()
    with pytest.raises(type(error)) as raised:
        invoke(engine, 60, 5, 6010, object(), family)
    assert raised.value is error and len(calls) == 1
    assert capsys.readouterr().out == ''


def test_observer_has_no_device_mutation_or_second_reservation():
    _, node = hook()
    text = ast.unparse(node)
    assert text.count('self.governor.reserve(') == 1
    for forbidden in ('synchronize(', 'clear_cache(', 'reset_peak_memory(',
        'mx.eval(', 'prepare_for(', 'set_budget(', 'sleep(', 'materialize_layer('):
        assert forbidden not in text
