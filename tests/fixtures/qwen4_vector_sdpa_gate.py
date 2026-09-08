#!/usr/bin/env python3
"""Stop-gated native vector-SDPA prefill experiment; never a serving profile.

The installed MLX0.32 head256 full-query path falls back, whereas1/2 query rows
can select native vector attention. That changes reduction/rounding: unchanged
weights and key order do NOT establish numerical equivalence. Refuse escalation
to larger cases on the first completed shape with any output-bit mismatch.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CASES = ((17, 257, 68329), (33, 4099, 47981), (129, 32768, 95317))
ORDER = (0, 1, 2, 2, 1, 0)


def vector_plan(total, tile, query_heads, kv_heads):
    if any(type(v) is not int for v in (total, tile, query_heads, kv_heads)):
        raise ValueError('integer geometry required')
    if not 9 <= total <= 1024 or tile not in (1, 2):
        raise ValueError('bounded fallback-prefill query count and vector tile1/2 required')
    if min(query_heads, kv_heads) <= 0 or query_heads % kv_heads:
        raise ValueError('positive divisible GQA heads required')
    if tile * (query_heads // kv_heads) > 32:
        raise ValueError('MLX0.32 vector dispatch GQA bound exceeded')
    return tuple((i, min(i+tile, total)) for i in range(0, total, tile))


def vector_prefill(mx, q, k, v, *, scale, mask, tile):
    if (q.ndim != 4 or k.ndim != 4 or v.ndim != 4
            or q.shape[0] != k.shape[0] or k.shape != v.shape
            or any(x.shape[-1] != 256 for x in (q, k, v))
            or q.dtype not in (mx.bfloat16, mx.float16)
            or q.dtype != k.dtype or q.dtype != v.dtype):
        raise ValueError('same-format head256 FP16/BF16 geometry required')
    spans = vector_plan(q.shape[2], tile, q.shape[1], k.shape[1])
    if k.shape[2] < tile:
        raise ValueError('vector query count must not exceed key count')
    if mask is not None:
        shape = getattr(mask, 'shape', ())
        if (len(shape) not in (2, 4) or shape[-2] not in (1, q.shape[2])
                or shape[-1] not in (1, k.shape[2])
                or len(shape) == 4 and (shape[0] not in (1, q.shape[0])
                                       or shape[1] not in (1, q.shape[1]))):
            raise ValueError('explicit offset-correct rank2/4 array mask required')
    outputs = []
    for start, end in spans:
        part_mask = (mask if mask is None or mask.shape[-2] == 1
                     else mask[start:end, :] if mask.ndim == 2
                     else mask[:, :, start:end, :])
        out = mx.fast.scaled_dot_product_attention(q[:, :, start:end, :], k, v,
                                                 scale=scale, mask=part_mask)
        mx.eval(out)
        outputs.append(out)
    return mx.concatenate(outputs, axis=2)


def should_stop(cases):
    return any(not arm['exact'] for case in cases for arm in case['arms'])


def run_probe(document):
    import mlx.core as mx
    import numpy as np
    from tests.fixtures.qwen4_sdpa_tiling_gate import pressure_sample, pressure_failures
    from runtime.host_activity_witness import sample_known_transcoders, summarize_known_transcoders
    samples = [pressure_sample()]; activity = [sample_known_transcoders()]
    previous_limit = mx.set_cache_limit(0); peak = 0
    def bits(a):
        return np.asarray(a.view(mx.uint16)).copy()
    def digest(a):
        return hashlib.sha256(bits(a).tobytes(order='C')).hexdigest()
    try:
        for length, key_length, seed in CASES:
            for dtype_name in ('bfloat16', 'float16'):
                if pressure_failures(samples, peak):
                    document['failures'].append('pressure before next case');return
                rng = np.random.default_rng(seed); dtype = getattr(mx, dtype_name)
                def random(n, heads):
                    return mx.array(rng.standard_normal((1, n, heads, 256), dtype=np.float32)).astype(dtype).transpose(0, 2, 1, 3)
                q, k, v = random(length, 24), random(key_length, 2), random(key_length, 2)
                positions = mx.arange(key_length-length, key_length)[:, None]
                keys = mx.arange(key_length)[None, :]
                allowed = keys <= positions
                if key_length > 2048:
                    allowed = allowed & (((keys//4 + positions*13 + seed) % 16 == 0) | (keys == positions))
                mask = mx.where(allowed, 0., -mx.inf).astype(dtype)
                if dtype_name == 'float16': mask = mask[None, None]
                mx.eval(q, k, v, mask)
                inputs = (q, k, v, mask); hashes = [digest(x) for x in inputs]
                reference = mx.fast.scaled_dot_product_attention(q, k, v, scale=.0625, mask=mask)
                mx.eval(reference); reference_bits = bits(reference)
                row = dict(query_tokens=length, key_tokens=key_length, seed=seed, dtype=dtype_name,
                    mask_rank=mask.ndim, input_sha256=hashes,
                    reference_sha256=hashlib.sha256(reference_bits.tobytes()).hexdigest(), arms=[])
                document['cases'].append(row)
                for tile in ORDER:
                    activity.append(sample_known_transcoders())
                    if not summarize_known_transcoders(activity)['passed']:
                        document['failures'].append('known-transcoder isolation');return
                    peak = max(peak, int(mx.get_peak_memory()))
                    mx.clear_cache(); mx.reset_peak_memory()
                    started = time.perf_counter()
                    out = (vector_prefill(mx, q, k, v, scale=.0625, mask=mask, tile=tile)
                           if tile else mx.fast.scaled_dot_product_attention(q, k, v, scale=.0625, mask=mask))
                    mx.eval(out); seconds = time.perf_counter()-started
                    peak = max(peak, int(mx.get_peak_memory()))
                    actual = bits(out); mismatches = int(np.count_nonzero(actual != reference_bits))
                    row['arms'].append(dict(tile=tile, seconds=seconds, output_values=int(actual.size),
                        mismatches=mismatches, exact=mismatches == 0,
                        output_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
                        metal_peak_bytes=int(mx.get_peak_memory()),
                        vector_eligible_sdpa_calls=(len(vector_plan(length, tile, 24, 2)) if tile else 0)))
                    samples.append(pressure_sample()); del out, actual
                    if pressure_failures(samples, peak):
                        document['failures'].append('pressure during case');return
                row['inputs_unchanged'] = [digest(x) for x in inputs] == hashes
                if not row['inputs_unchanged']:
                    document['failures'].append('mutated inputs');return
                del q, k, v, mask, inputs, reference, reference_bits
            if should_stop(document['cases']):
                document['failures'].append('bit mismatch; larger shapes not attempted')
                document['stopped_before_larger_shapes'] = True
                break
    finally:
        peak = max(peak, int(mx.get_peak_memory()))
        samples.append(pressure_sample()); activity.append(sample_known_transcoders())
        mx.set_cache_limit(previous_limit)
        document.update(pressure_samples=samples, pressure_failures=pressure_failures(samples, peak),
            true_peak_metal_bytes=peak, known_transcoders=summarize_known_transcoders(activity),
            cache_limit_restored=True)
        document['failures'].extend(document['pressure_failures'])
        if not document['known_transcoders']['passed']:
            document['failures'].append('known-transcoder isolation')
    document['passed'] = not document['failures']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    pre = json.loads(args.preflight.read_text())
    if (args.result.exists() or not pre['passed'] or pre['sample_seconds'] < 30
            or not pre['known_transcoders']['passed']
            or not 0 <= time.monotonic()-pre['end']['monotonic_s'] < 120):
        parser.error('fresh result and passing30s preflight required')
    document = dict(schema='voom.qwen4-vector-sdpa-stop-gate.v1', passed=False, failures=[], cases=[],
        planned_cases=CASES, order=ORDER, preflight=pre,
        preflight_sha256=hashlib.sha256(args.preflight.read_bytes()).hexdigest(),
        versions={name:importlib.metadata.version(name) for name in ('mlx', 'numpy')},
        dispatch_basis='Version-tagged MLX0.32.0 source and actual call geometry; not a binary GPU kernel trace.',
        scope='Synthetic head256 SDPA only, full shared KV and fixed causal/block mask. No model weights, tokens, QSA selection/state, whole-model/harness/Plex or lossy quality proof. No runtime integration.',
        timing_scope='Construction+evaluation only per arm; hashes/setup/pressure excluded. Whole wall includes everything. Repeated same synthetic shape, not real-traffic latency.',
        physical_model_io_bytes=0, pressure_sampling='arm endpoints, may miss short system pressure',
        stopped_before_larger_shapes=False)
    started = time.perf_counter()
    try:
        run_probe(document)
    except BaseException as error:
        document['failures'].append(type(error).__name__);document['passed'] = False
    document['wall_seconds'] = time.perf_counter()-started
    from tests.fixtures.qwen4_hot_boundary_http_probe import _atomic_write_private
    _atomic_write_private(args.result, document)
    print(json.dumps({k:document.get(k) for k in ('passed', 'failures', 'wall_seconds', 'true_peak_metal_bytes')}), flush=True)
    return 0 if document['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
