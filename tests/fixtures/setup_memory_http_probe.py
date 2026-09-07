#!/usr/bin/env python3
"""Profile actual lazy HTTP setup, then intentionally stop before generation.

Disposable server only. No pre-import of MLX/engine/grammar, no tensor payload
inspection, no runtime defaults. sys/threading profiling adds diagnostic cost;
none of these timings is a serving-speed measurement or completed-model result.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MODULES = frozenset(('mlx.core', 'runtime.engine', 'runtime.model_loader',
                    'runtime.qwen4_mtp', 'runtime.structured', 'torch', 'transformers',
                    'xgrammar', 'xgrammar.compiler', 'xgrammar.tokenizer_info'))
FUNCTIONS = frozenset((
    ('runtime.server', 'EngineManager.get'),
    ('runtime.engine', 'StreamingEngine.__init__'),
    ('runtime.model_loader', 'WeightStore.__init__'),
    ('runtime.predictor', 'MarkovExpertPredictor.__init__'),
    ('runtime.server', '_prepare_chat_prompt'),
    ('runtime.server', '_chat_prompt'),
    ('runtime.server', '_configure_constraint'),
    ('runtime.structured', '_compiler'),
    ('runtime.structured', '_xgrammar'),
    ('xgrammar.tokenizer_info', 'TokenizerInfo.from_huggingface'),
    ('xgrammar.compiler', 'GrammarCompiler.__init__'),
    ('xgrammar.compiler', 'GrammarCompiler.compile_grammar'),
    ('transformers.models.auto.tokenization_auto', 'AutoTokenizer.from_pretrained'),
))
STOP_MESSAGE = 'SETUP_DIAGNOSTIC_STOP_BEFORE_GENERATION'


class SetupDiagnosticStop(RuntimeError):
    pass


def classify(frame):
    module = frame.f_globals.get('__name__', '')
    code = frame.f_code
    name = code.co_qualname
    if module == 'runtime.server' and name == '_engine_generate':
        return 'stop_before_generation'
    # dataclasses/import-time exec may reuse module globals but execute generated
    # <string> code. Do not label that as a second module import or spend its cap.
    if (name == '<module>' and module in MODULES
            and code.co_filename == frame.f_globals.get('__file__')):
        return module + '.<module>'
    if (module, name) in FUNCTIONS or (
            module == 'runtime.qwen4_mtp' and name.endswith('.__init__')):
        return module + '.' + name
    return None


def prepared_identity(frame, expected_count, expected_sha, output_cap):
    local = frame.f_locals
    args, kwargs = local.get('args', ()), local.get('kwargs', {})
    prompt = args[0] if args else kwargs.get('prompt')
    cap = args[1] if len(args) > 1 else kwargs.get('max_tokens')
    ids = getattr(prompt, 'token_ids', None)
    engine = local.get('engine')
    digest = None
    if type(ids) in (list, tuple) and all(type(t) is int and 0 <= t < 2**32 for t in ids):
        digest = hashlib.sha256(json.dumps(list(ids), separators=(',', ':')).encode('ascii')).hexdigest()
    count = len(ids) if type(ids) in (list, tuple) else None
    return {'token_count': count, 'token_sha256': digest,
            'token_digest_encoding': 'compact-json-integer-array-v1',
            'output_cap': cap if type(cap) is int else None,
            'engine_class': type(engine).__name__,
            'passed': (count == expected_count and digest == expected_sha
                       and type(cap) is int and cap == output_cap
                       and type(engine).__name__ == 'Qwen4MTPSpeculativeEngine')}


def predictor_owners(engine):
    """Constant-time, shallow owner scalars; never iterate learned entries."""
    from collections import defaultdict
    target = getattr(engine, 'target', engine)
    rc = getattr(target, 'rc', None)
    predictor = getattr(target, 'predictor', None)
    counts = getattr(predictor, 'counts', None)
    valid = counts is None or type(counts) in (dict, defaultdict)
    return {'available': valid, 'present': predictor is not None,
            'entry_count': len(counts) if valid and counts is not None else None,
            'dictionary_shallow_bytes': sys.getsizeof(counts) if valid and counts is not None else None,
            'shallow_bytes_are_total_ownership': False,
            'tracking': getattr(rc, 'expert_transition_tracking', None),
            'predictive_prefetch': getattr(rc, 'expert_predictive_prefetch', None),
            'warm_start': getattr(rc, 'warm_start', None)}


class SetupProfiler:
    def __init__(self, *, sample, publish, expected_count, expected_sha, output_cap):
        if type(expected_count) is not int or not 1 <= expected_count <= 1_000_000:
            raise ValueError('bounded positive token count required')
        if (not isinstance(expected_sha, str) or len(expected_sha) != 64
                or any(c not in '0123456789abcdef' for c in expected_sha)):
            raise ValueError('expected token SHA256 required')
        if type(output_cap) is not int or not 1 <= output_cap <= 4096:
            raise ValueError('bounded sufficient output cap required')
        self.sample, self.publish = sample, publish
        self.expected_count, self.expected_sha, self.output_cap = expected_count, expected_sha, output_cap
        self.events, self.seen = [], {}
        self.started = time.perf_counter()
        self.claimed, self.capped = False, False

    def observe(self, label, event):
        if len(self.events) >= 128:
            self.capped = True
            return
        started = time.perf_counter()
        try:
            values = self.sample()
        except Exception as error:
            values = {'available': False, 'error_type': type(error).__name__}
        self.events.append({'label': label, 'event': event,
                            'elapsed_seconds': started - self.started,
                            'observation_seconds': time.perf_counter() - started,
                            'values': values})

    def __call__(self, frame, event, arg):
        if event not in ('call', 'return') or self.claimed:
            return
        label = classify(frame)
        if label is None:
            return
        if label == 'stop_before_generation':
            if event != 'call':
                return
            self.claimed = True
            identity = prepared_identity(frame, self.expected_count, self.expected_sha, self.output_cap)
            self.observe(label, event)
            document = {'schema': 'voom.setup-memory-http-probe.v1',
                        'intentional_stop_before_generation': True,
                        'generation_executed': False, 'generated_tokens': 0,
                        'completed_model_response': False, 'speed_benchmark': False,
                        'events': self.events, 'capped': self.capped,
                        'prepared_identity': identity,
                        'predictor_owners': predictor_owners(frame.f_locals.get('engine')),
                        'diagnostic_wall_seconds': time.perf_counter() - self.started}
            profiles = sys.modules.get('runtime.profiles')
            if profiles is not None:
                document['profiles'] = profiles.active_runtime_profile_fields()
            self.publish(document)
            # A profile callback exception stops the function before its body.
            # This is the ONLY intentional behavior change and is never serving.
            raise SetupDiagnosticStop(STOP_MESSAGE)
        key = (label, event)
        count = self.seen.get(key, 0)
        self.seen[key] = count + 1
        if count < 2:  # bound repeated compiler/import/cache helper observations.
            self.observe(label, event)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--expected-prompt-tokens', type=int, required=True)
    parser.add_argument('--expected-token-sha', required=True)
    parser.add_argument('--output-cap', type=int, default=1024)
    parser.add_argument('--profile', action='append', default=[])
    args = parser.parse_args()
    if args.artifact.exists() or args.artifact.is_symlink():
        parser.error('refusing existing artifact')
    if not 1 <= args.port <= 65535:
        parser.error('invalid port')
    if sys.getprofile() is not None or threading.getprofile() is not None:
        parser.error('refusing to replace an existing profiler')
    import psutil
    from runtime.process_memory_witness import sample_self_memory
    from runtime.process_region_witness import sample_self_regions
    from tests.fixtures.qwen4_hot_boundary_http_probe import _atomic_write_private

    def sample():
        # Preserve lazy import order. Missing/partially imported MLX is null,
        # not fabricated zero. Counter reads can initialize native allocator.
        module = sys.modules.get('mlx.core')
        scalars = {}
        for name in ('get_active_memory', 'get_cache_memory', 'get_peak_memory'):
            try:
                fn = getattr(module, name, None)
                scalars[name] = int(fn()) if callable(fn) else None
            except Exception:
                scalars[name] = None
        swap = psutil.swap_memory()
        process = sample_self_memory()
        regions = sample_self_regions()
        return {'available': process['available'] and regions['coverage_complete'],
                'process': process, 'regions': regions, 'metal': scalars,
                'python_allocated_blocks': sys.getallocatedblocks(),
                'system_available_bytes': psutil.virtual_memory().available,
                'system_swap_used_bytes': swap.used, 'system_swap_out_bytes': swap.sout,
                'atomic': False}

    probe = SetupProfiler(sample=sample,
                          publish=lambda d: _atomic_write_private(args.artifact, d),
                          expected_count=args.expected_prompt_tokens,
                          expected_sha=args.expected_token_sha, output_cap=args.output_cap)
    old_argv = sys.argv
    probe.observe('before_server_import', 'boundary')
    sys.setprofile(probe)
    threading.setprofile(probe)
    try:
        from runtime import server
        probe.observe('after_server_import', 'boundary')
        sys.argv = ['runtime.server', '--port', str(args.port)]
        for profile in args.profile:
            sys.argv.extend(('--profile', profile))
        server.main()
    finally:
        threading.setprofile(None)
        sys.setprofile(None)
        sys.argv = old_argv


if __name__ == '__main__':
    main()
