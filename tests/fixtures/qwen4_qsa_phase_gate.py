#!/usr/bin/env python3
"""Synchronized real-QSA phase diagnostic, never a serving latency benchmark.

Injects five evaluation boundaries into a copy of the actual attention AST.
No model expression is replaced. Evaluation changes scheduling and memory;
every output/state is therefore checked against an uninstrumented artifact.
The installed runtime is patched only inside this isolated one-job process.
"""
from __future__ import annotations

import argparse
import ast
import copy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PHASES = ('indexer_selection_update', 'qkv_projection_norm_rope_update',
          'additive_mask', 'sdpa', 'output_gate_projection')


def instrument_attention(source):
    """Fail closed if any of the five actual-source boundaries is missing."""
    tree = ast.parse(source)
    matches = [n for n in tree.body if isinstance(n, ast.FunctionDef)
               and n.name == '_qsa_attention']
    if len(matches) != 1:
        raise ValueError('one actual _qsa_attention definition required')
    fn = copy.deepcopy(matches[0])
    if sum(isinstance(n, ast.Return) for n in ast.walk(fn)) != 1:
        raise ValueError('one terminal attention return required')
    body, seen = [], []
    for node in fn.body:
        label = arrays = None
        if isinstance(node, ast.Assign):
            targets = [ast.unparse(n) for n in node.targets]
            if targets == ['qsa_mask']:
                label, arrays = PHASES[0], 'qsa_mask, state.qsa_keys[layer], state.qsa_positions[layer]'
            elif targets == ['(keys, values)']:
                label, arrays = PHASES[1], 'query, keys, values, output_gate'
        elif isinstance(node, ast.If):
            if ast.unparse(node.test) == 'qsa_mask is None':
                label, arrays = PHASES[2], 'mask'
            elif ast.unparse(node.test) == 'sdpa_query_tile':
                label, arrays = PHASES[3], 'attended'
        elif isinstance(node, ast.Return):
            if node is not fn.body[-1]:
                raise ValueError('attention return must be terminal')
            if not isinstance(node.value, ast.Call) or ast.unparse(node.value.func) != '_linear':
                raise ValueError('original output projection required')
            body.append(ast.Assign(targets=[ast.Name(id='_phase_output', ctx=ast.Store())],
                                   value=node.value))
            label, arrays = PHASES[4], '_phase_output'
        if not isinstance(node, ast.Return):
            body.append(node)
        if label:
            seen.append(label)
            body.extend(ast.parse(f'_phase_probe.mark({label!r}, {arrays})').body)
        if isinstance(node, ast.Return):
            body.append(ast.Return(value=ast.Name(id='_phase_output', ctx=ast.Load())))
    if tuple(seen) != PHASES:
        raise ValueError('actual attention boundary layout changed')
    fn.body = body
    return ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))


class PhaseTimer:
    """Bounded scalar-only records; never retain model inputs or outputs."""
    def __init__(self, evaluate, memory, clock=time.perf_counter):
        self.evaluate, self.memory, self.clock = evaluate, memory, clock
        self.calls = []
        self.active = None

    def begin(self, *, layer, offset, query_tokens, tile):
        if self.active is not None or len(self.calls) >= 128:
            raise ValueError('nested call or diagnostic call cap')
        self.active = dict(layer=layer, offset=offset, query_tokens=query_tokens,
                           tile=tile, phases=[])
        self.calls.append(self.active)
        self.previous = self.clock()

    def mark(self, label, *arrays):
        if self.active is None or len(self.active['phases']) >= len(PHASES):
            raise ValueError('unclaimed phase or phase cap')
        if label != PHASES[len(self.active['phases'])]:
            raise ValueError('unexpected phase order')
        self.evaluate(*(array for array in arrays if array is not None))
        finished = self.clock()
        self.active['phases'].append(dict(phase=label, seconds=finished-self.previous,
                                         metal_after=self.memory()))
        self.previous = self.clock()

    def finish(self):
        if self.active is None or len(self.active['phases']) != len(PHASES):
            raise ValueError('incomplete phase coverage')
        self.active = None


def compare_reference(actual, reference):
    """Compare every measured input/weight/output/KV/QSA bit, not timing fields."""
    checks = []
    for case, ref in zip(actual, reference, strict=True):
        for name in ('layer', 'total_tokens', 'seed', 'activation_dtype', 'loaded_weight_hashes'):
            checks.append(case[name] == ref[name])
        for arm, old in zip(case['arms'], ref['arms'], strict=True):
            checks.append(arm['tile_queries'] == old['tile_queries'])
            for row, prior in zip(arm['steps'], old['steps'], strict=True):
                checks.append(all(row[k] == prior[k] for k in
                                  ('start', 'end', 'input', 'input_unchanged', 'output', 'state')))
            for row, prior in zip(arm['continuation'], old['continuation'], strict=True):
                checks.append(all(row[k] == prior[k] for k in ('position', 'output', 'state')))
    return bool(checks) and all(checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--preflight', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--reference-sha256', required=True)
    args = parser.parse_args()
    from tests.fixtures.qwen4_real_qsa_tiling_gate import file_sha, run_probe
    from tests.fixtures.qwen4_hot_boundary_http_probe import _atomic_write_private
    pre = json.loads(args.preflight.read_text())
    if not (pre['passed'] and pre['sample_seconds'] >= 30
            and pre['known_transcoders']['passed']
            and 0 <= time.monotonic()-pre['end']['monotonic_s'] < 120):
        parser.error('fresh passing30s preflight required')
    if args.result.exists() or file_sha(args.reference) != args.reference_sha256:
        parser.error('fresh result and pinned uninstrumented reference required')
    reference = json.loads(args.reference.read_text())
    if not reference.get('passed'):
        parser.error('passing uninstrumented reference required')
    source = (ROOT/'runtime/qwen4_exp.py').read_text()
    tree = instrument_attention(source)
    document = dict(schema='voom.qwen4-real-qsa-phase-diagnostic.v1', passed=False,
        failures=[], cases=[], preflight=pre, preflight_sha256=file_sha(args.preflight),
        model=str(args.model), reference=str(args.reference), reference_sha256=args.reference_sha256,
        scope='Actual installed attention weights; synthetic hidden inputs at3079/32799. Added evaluation boundaries. No generated tokens, whole model/harness/Plex or serving pressure/speed proof.',
        timing_scope='Non-overlapping synchronized construction+evaluation phases. Observation overhead excluded from phase times but included in whole step/wall. Barriers alter scheduling and allocator lifetimes; do not compare these phase times as serving latency.',
        extra_synchronization=True, runtime_files_modified=False,
        uninstrumented_source_sha256=file_sha(ROOT/'runtime/qwen4_exp.py'),
        model_metadata_sha256={name:file_sha(args.model/name) for name in reference['model_metadata_sha256']})
    assert document['model_metadata_sha256'] == reference['model_metadata_sha256']
    started = time.perf_counter()
    import mlx.core as mx
    from runtime import qwen4_exp
    probe = PhaseTimer(mx.eval, lambda: dict(active_bytes=int(mx.get_active_memory()),
                                           cumulative_peak_bytes=int(mx.get_peak_memory())))
    namespace = dict(qwen4_exp.__dict__, _phase_probe=probe)
    exec(compile(tree, '<actual-qsa-with-diagnostic-boundaries>', 'exec'), namespace)
    instrumented = namespace['_qsa_attention']
    original = qwen4_exp._qsa_attention
    def wrapper(*a, **kw):
        probe.begin(layer=int(a[5]), offset=int(a[6]), query_tokens=int(a[0].shape[1]),
                    tile=int(kw.get('sdpa_query_tile', 0)))
        out = instrumented(*a, **kw)
        probe.finish()
        return out
    try:
        qwen4_exp._qsa_attention = wrapper
        run_probe(document, args.model)
        document['matches_uninstrumented_reference'] = compare_reference(document['cases'], reference['cases'])
        if not document['matches_uninstrumented_reference']:
            document['failures'].append('uninstrumented output/state reference mismatch')
    except BaseException as error:
        document['failures'].append(type(error).__name__)
    finally:
        qwen4_exp._qsa_attention = original
        document['runtime_function_restored'] = qwen4_exp._qsa_attention is original
        document['phase_calls'] = probe.calls
        document['phase_coverage_complete'] = (len(probe.calls) == 86 and probe.active is None
            and all(tuple(row['phase'] for row in call['phases']) == PHASES for call in probe.calls))
        if not document['phase_coverage_complete']:
            document['failures'].append('incomplete phase coverage')
        document['wall_seconds'] = time.perf_counter()-started
        document['passed'] = not document['failures']
        _atomic_write_private(args.result, document)
    print(json.dumps({k:document.get(k) for k in ('passed', 'failures', 'wall_seconds',
                     'matches_uninstrumented_reference', 'phase_coverage_complete')}), flush=True)
    return 0 if document['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
