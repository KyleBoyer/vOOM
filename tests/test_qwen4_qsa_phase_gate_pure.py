"""Diagnostic source preservation, exact-reference failures, bounded phase timing."""
import ast
import copy
from pathlib import Path

import pytest

from tests.fixtures.qwen4_qsa_phase_gate import PHASES, PhaseTimer, compare_reference, instrument_attention

SOURCE = Path(__file__).resolve().parents[1]/'runtime/qwen4_exp.py'


def test_injected_boundaries_preserve_every_original_attention_expression():
    source = SOURCE.read_text()
    original = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == '_qsa_attention')
    transformed = instrument_attention(source).body[0]
    markers = [n for n in transformed.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
               and ast.unparse(n.value.func) == '_phase_probe.mark']
    assert tuple(n.value.args[0].value for n in markers) == PHASES
    stripped = copy.deepcopy(transformed)
    stripped.body = [n for n in stripped.body if not (isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == '_phase_probe.mark')]
    assert isinstance(stripped.body[-2], ast.Assign)
    stripped.body[-2:] = [ast.Return(value=stripped.body[-2].value)]
    assert ast.dump(stripped, include_attributes=False) == ast.dump(original, include_attributes=False)


@pytest.mark.parametrize('old,new', [('_qsa_attention(', '_different_attention('),
    ('qsa_mask = _qsa_selection_mask(', 'changed_mask = _qsa_selection_mask('),
    ('keys, values = kv.update(', 'keys, wrong = kv.update('),
    ('if qsa_mask is None:', 'if qsa_mask is not None:'),
    ('if sdpa_query_tile:', 'if not sdpa_query_tile:'),
    ('return _linear(attended, weights,', 'return different(attended, weights,')])
def test_source_boundary_drift_fails_closed(old, new):
    with pytest.raises(ValueError):
        instrument_attention(SOURCE.read_text().replace(old, new))


def test_timer_evaluates_only_non_none_values_and_keeps_only_scalars():
    events = []; ticks = iter(range(30)); marker = object()
    timer = PhaseTimer(lambda *a: events.append(a), lambda: {'active_bytes': 9}, lambda: next(ticks))
    timer.begin(layer=3, offset=0, query_tokens=1024, tile=256)
    for phase in PHASES:
        timer.mark(phase, None, marker)
    timer.finish()
    assert events == [(marker,)]*5 and timer.active is None
    assert [row['seconds'] for row in timer.calls[0]['phases']] == [1]*5
    assert all(row['metal_after'] == {'active_bytes': 9} for row in timer.calls[0]['phases'])
    assert marker not in timer.__dict__.values()
    with pytest.raises(ValueError): timer.finish()
    with pytest.raises(ValueError): timer.mark(PHASES[0], marker)


def test_timer_refuses_nested_partial_out_of_order_and_excess_calls():
    timer = PhaseTimer(lambda *a: None, lambda: {})
    timer.begin(layer=3, offset=0, query_tokens=17, tile=0)
    with pytest.raises(ValueError): timer.begin(layer=3, offset=0, query_tokens=17, tile=0)
    with pytest.raises(ValueError): timer.mark(PHASES[1])
    with pytest.raises(ValueError): timer.finish()
    timer.active = None; timer.calls = [None]*128
    with pytest.raises(ValueError): timer.begin(layer=3, offset=0, query_tokens=17, tile=0)


def case():
    step = dict(start=0, end=3, input='input-hash', input_unchanged=True,
                output='output-hash', state={'key':'key-hash'}, seconds=1)
    return dict(layer=3, total_tokens=3, seed=5, activation_dtype='bf16', loaded_weight_hashes={'w':'hash'},
                arms=[dict(tile_queries=256, steps=[step], continuation=[dict(position=3, output='o', state='s')])])


def test_reference_excludes_timing_but_not_actual_inputs_outputs_weights_or_state():
    ref = case(); actual = copy.deepcopy(ref)
    actual['arms'][0]['steps'][0]['seconds'] = 9
    assert compare_reference([actual], [ref])
    actual['arms'][0]['steps'][0]['state']['key'] = 'different'
    assert not compare_reference([actual], [ref])
    assert not compare_reference([], [])
    with pytest.raises(ValueError): compare_reference([ref], [])
