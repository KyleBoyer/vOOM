#!/usr/bin/env python3
"""One completed-budget HTTP generation with before/after native map diagnostics.

No model/settings/response changes. Map walks add timed kernel work to HTTP wall;
this disposable server is an ownership diagnostic, not a serving-speed A/B.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tests.fixtures.qwen4_hot_boundary_http_probe import _atomic_write_private, _token_digest


def ownership_summary(target, kv_type):
    """Read known plain owners' logical sizes, never tensor values/backing."""
    endpoint = getattr(target, 'last_kv', None)
    slots = getattr(target, '_hot_prompt_slots', None)
    if type(slots) is not list or len(slots) > 4:
        return {'available': False, 'reason': 'unknown-slot-owners'}
    def describe(kv):
        if kv is None:
            return {'present': False}
        if type(kv) is not kv_type:
            return {'present': True, 'available': False, 'reason': 'non-plain-owner'}
        return {'present': True, 'available': True, 'logical_bytes': int(kv.nbytes()),
                'positions': int(kv.offset)}
    return {'available': True, 'physical_ownership_proven': False,
            'endpoint': describe(endpoint), 'retained_slot_count': len(slots),
            'retained_slots': [dict(describe(getattr(slot, 'kv', None)),
                                    aliases_endpoint=getattr(slot, 'kv', None) is endpoint,
                                    prefix_token_count=len(getattr(slot, 'tokens', ())))
                               for slot in slots]}


class RegionHTTPProbe:
    def __init__(self, original, *, artifact, expected_tokens, output_cap,
                 engine_type, snapshot, publish=_atomic_write_private):
        if type(expected_tokens) is not int or expected_tokens <= 0:
            raise ValueError('positive expected token count required')
        if type(output_cap) is not int or not 1 <= output_cap <= 4096:
            raise ValueError('output cap must be in 1..4096')
        self.original = original
        self.artifact = Path(artifact)
        self.expected_tokens = expected_tokens
        self.output_cap = output_cap
        self.engine_type = engine_type
        self.snapshot = snapshot
        self.publish = publish
        self.claimed = False

    def _snapshot(self, target):
        started = time.perf_counter()
        try:
            value = self.snapshot(target)
        except Exception:
            value = {'available': False, 'reason': 'observation-error'}
        return {'values': value, 'wall_seconds': time.perf_counter() - started}

    def __call__(self, engine, *args, **kwargs):
        if self.claimed:
            raise RuntimeError('region probe permits exactly one generation')
        self.claimed = True
        if type(engine) is not self.engine_type:
            raise ValueError('requires the concrete Qwen4 MTP wrapper')
        prompt = args[0] if args else kwargs.get('prompt')
        cap = args[1] if len(args) > 1 else kwargs.get('max_tokens')
        prepared = _token_digest(getattr(prompt, 'token_ids', None))
        if prepared['count'] != self.expected_tokens or type(cap) is not int or cap != self.output_cap:
            raise ValueError('unexpected prepared request/output budget')
        target = engine.target
        document = {'schema': 'voom.process-region-http-probe.v1',
                    'available': False, 'prepared_tokens': prepared, 'max_output_tokens': cap,
                    'scope': 'before_and_after_one_engine_invocation_under_inference_lock',
                    'speed_benchmark': False, 'modifies_request_or_result': False,
                    'included_in_http_wall_seconds': True, 'before': self._snapshot(target)}
        started = time.perf_counter()
        try:
            result = self.original(engine, *args, **kwargs)
        except BaseException as error:
            document['generation_error_type'] = type(error).__name__
            raise
        else:
            try:
                tokens = _token_digest(result.get('tokens'))
                text = result.get('text')
                if not isinstance(text, str):
                    raise ValueError('missing raw engine text')
                document.update(available=True, generated_tokens=tokens,
                                engine_text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                                generation={key: result.get(key) for key in (
                                    'prefill_s', 'decode_s', 'total_s', 'termination_reason',
                                    'true_peak_metal_bytes')})
            except Exception:
                document['reason'] = 'generation-witness-unavailable'
            return result
        finally:
            document['generation_invocation_seconds'] = time.perf_counter() - started
            document['after'] = self._snapshot(target)
            try:
                self.publish(self.artifact, document)
            except Exception as error:
                # Missing receipt fails the outside diagnostic gate, but an
                # observer failure must not rewrite a response/generation error.
                try:
                    print('[process-region-probe] publication_failed error_type='
                          + type(error).__name__, flush=True)
                except Exception:
                    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--expected-prompt-tokens', type=int, required=True)
    parser.add_argument('--output-cap', type=int, default=1024)
    parser.add_argument('--port', type=int, default=8073)
    parser.add_argument('--profile', action='append', required=True)
    args = parser.parse_args()
    if args.artifact.exists() or args.artifact.is_symlink():
        parser.error('refusing existing artifact')
    if not 1 <= args.port <= 65535:
        parser.error('invalid port')
    # No native/MLX work on import; this CLI requires a fresh passing preflight.
    import mlx.core as mx
    from runtime import server
    from runtime.kv_cache import KVCache
    from runtime.phase_head_witness import sample_phase_head_memory
    from runtime.process_memory_witness import sample_self_memory
    from runtime.process_region_witness import sample_self_regions
    from runtime.qwen4_mtp import Qwen4MTPSpeculativeEngine

    def snapshot(target):
        return {'scalars_before_walk': sample_phase_head_memory(target, mx),
                'regions': sample_self_regions(),
                'self_after_walk': sample_self_memory(),
                'logical_owners': ownership_summary(target, KVCache)}
    original = server._engine_generate
    old_argv = sys.argv
    server._engine_generate = RegionHTTPProbe(
        original, artifact=args.artifact, expected_tokens=args.expected_prompt_tokens,
        output_cap=args.output_cap, engine_type=Qwen4MTPSpeculativeEngine, snapshot=snapshot)
    sys.argv = ['runtime.server', '--port', str(args.port)]
    for profile in args.profile:
        sys.argv.extend(('--profile', profile))
    try:
        server.main()
    finally:
        server._engine_generate = original
        sys.argv = old_argv


if __name__ == '__main__':
    main()
