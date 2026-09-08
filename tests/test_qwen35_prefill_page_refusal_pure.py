"""Execute the real failure-only hook without MLX, weights, or allocations."""

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
                  and n.name == '_layer_stationary_qwen35_sweep')
    node = next(n for n in ast.walk(method) if isinstance(n, ast.Try)
        and any(isinstance(c, ast.Constant) and c.value == 'qwen-prefill-layer-page'
                for c in ast.walk(ast.Module(body=n.body, type_ignores=[]))))
    fn = ast.parse('def invoke(self, incoming_page, i, transient_shape_positions, tile_width): pass').body[0]
    fn.body = [copy.deepcopy(node)]
    namespace = {'__package__': 'runtime', 'mx': object(),
        'kv': SimpleNamespace(nbytes=lambda: 250_000_000, max_bytes=256_000_000,
            kda_cache=SimpleNamespace(nbytes=lambda: 150_000_000)),
        'x': SimpleNamespace(nbytes=60_000_000)}
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    exec(compile(module, str(ROOT / 'runtime/engine.py'), 'exec'), namespace)
    return namespace['invoke'], method


def target(error=None):
    calls = []
    def reserve(incoming, **kwargs):
        calls.append((incoming, kwargs))
        if error is not None:
            raise error
    return SimpleNamespace(governor=SimpleNamespace(reserve=reserve),
        _transient_layer_signature=lambda i: 'full_attention+dense',
        _layer_transient=73_000_000, _layer_transient_margin=400_000_000,
        _layer_transient_observation_counts={(111, 'full_attention+dense'): 6}), calls


def test_success_has_no_sampling_logging_or_margin_change(monkeypatch, capsys):
    def unexpected(*args):
        raise AssertionError('success must not observe')
    monkeypatch.setattr(phase_head_witness, 'sample_phase_head_memory', unexpected)
    invoke, _ = hook()
    engine, calls = target()
    invoke(engine, 210_000_000, 27, 111, 128)
    assert calls == [(210_000_000, {'reason': 'qwen-prefill-layer-page'})]
    assert capsys.readouterr().out == ''


def test_refusal_records_matching_shape_and_preserves_original_error(monkeypatch, capsys):
    memory = {'available': False, 'metal_active_bytes': 800_000_000,
              'system_available_bytes': None, 'unavailable_fields': ['system_available_bytes']}
    monkeypatch.setattr(phase_head_witness, 'sample_phase_head_memory', lambda *args: memory)
    error = MemoryError('original refusal')
    engine, calls = target(error)
    counts = dict(engine._layer_transient_observation_counts)
    invoke, _ = hook()
    with pytest.raises(MemoryError) as raised:
        invoke(engine, 210_000_000, 27, 111, 128)
    assert raised.value is error
    assert len(calls) == 1 and 'margin' not in calls[0][1]
    assert engine._layer_transient_observation_counts == counts
    raw = capsys.readouterr().out
    assert raw.startswith('[qwen35-prefill-page-admission] ')
    result = json.loads(raw.split('] ', 1)[1])
    assert result['layer'] == 27 and result['positions'] == 111
    assert result['tile_width'] == 128 and result['matching_compute_observations'] == 6
    assert result['estimated_page_bytes'] == 210_000_000
    assert result['selected_compute_scratch_bytes'] == 73_000_000
    assert result['selected_compute_margin_bytes'] == 400_000_000
    assert result['page_margin_source'] == 'governor_default'
    assert result['target_kv_resident_logical_bytes'] == 250_000_000
    assert result['target_kv_budget_bytes'] == 256_000_000
    assert result['target_recurrent_logical_bytes'] == 150_000_000
    assert result['hidden_logical_bytes'] == 60_000_000
    assert result['memory'] == memory  # missing observations are not zeroed


def test_unknown_optional_ownership_is_null_not_zero(monkeypatch, capsys):
    monkeypatch.setattr(phase_head_witness, 'sample_phase_head_memory', lambda *args: {})
    invoke, _ = hook()
    invoke.__globals__['kv'] = SimpleNamespace(nbytes=lambda: 13)
    error = MemoryError('unchanged')
    engine, _ = target(error)
    with pytest.raises(MemoryError) as raised:
        invoke(engine, 21, 3, 19, 8)
    assert raised.value is error
    result = json.loads(capsys.readouterr().out.split('] ', 1)[1])
    assert result['target_kv_resident_logical_bytes'] == 13
    assert result['target_kv_budget_bytes'] is None
    assert result['target_recurrent_logical_bytes'] is None


@pytest.mark.parametrize('kind', ['sample', 'serialization', 'emit'])
def test_optional_diagnostic_failure_never_masks_allocation_refusal(monkeypatch, capsys, kind):
    def sample(*args):
        if kind == 'sample':
            raise RuntimeError('observer failed')
        return {'invalid': float('nan')} if kind == 'serialization' else {}
    monkeypatch.setattr(phase_head_witness, 'sample_phase_head_memory', sample)
    invoke, _ = hook()
    if kind == 'emit':
        def fail_print(*args, **kwargs):
            raise OSError('log closed')
        invoke.__globals__['print'] = fail_print
    error = MemoryError('keep me')
    engine, calls = target(error)
    with pytest.raises(MemoryError) as raised:
        invoke(engine, 210_000_000, 27, 111, 128)
    assert raised.value is error and len(calls) == 1
    assert capsys.readouterr().out == ''


def test_other_reservation_errors_propagate_without_observation(monkeypatch, capsys):
    invoke, _ = hook()
    error = ValueError('not a memory refusal')
    engine, calls = target(error)
    with pytest.raises(ValueError) as raised:
        invoke(engine, 210_000_000, 27, 111, 128)
    assert raised.value is error and len(calls) == 1
    assert capsys.readouterr().out == ''


def test_refusal_hook_remains_before_fetch_and_has_no_device_mutations():
    _, method = hook()
    loop = next(n for n in method.body if isinstance(n, ast.For))
    branches = [n for n in loop.body if isinstance(n, ast.If)
                and 'self.cache.contains(layer_key)' in ast.unparse(n.test)]
    assert len(branches) == 1
    branch = branches[0]
    following = loop.body[loop.body.index(branch) + 1]
    assert 'self.cache.get(layer_key, layer_names)' in ast.unparse(following)
    text = ast.unparse(branch)
    assert text.index('self.cache.prepare_for(') < text.index('self.governor.reserve(')
    for forbidden in ('synchronize(', 'clear_cache(', 'reset_peak_memory(', 'mx.eval(',
                      'sleep(', 'margin='):
        assert forbidden not in text
