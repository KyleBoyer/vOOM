#!/usr/bin/env python3
"""One bounded, profile-only arm of a captured-request transition-tracking A/B.

Preserves original Responses tools, input, streaming and reasoning fields. Only
the model, sufficient output budget and deterministic sampling are overridden.
Saves actual terminal responses privately; no tools are executed or repaired.
Small captured shapes are NOT a full harness, Plex or large-context proof.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from runtime.profiles import apply_runtime_profiles
from tests.fixtures.qwen3_large_agent_replay_gate import _post, _request_wire_metadata
from tests.fixtures.runtime_profile_http_gate import _port_is_free, _wait_ready, _stop_server, _pressure
from tests.fixtures.qwen4_hot_boundary_http_probe import _atomic_write_private


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def prepare_case(case, model):
    raw = Path(case['capture']).read_bytes()
    if len(raw) != case['bytes'] or hashlib.sha256(raw).hexdigest() != case['sha256']:
        raise ValueError('capture identity mismatch')
    source = json.loads(raw)
    if not isinstance(source.get('input'), list) or 'messages' in source or 'max_tokens' in source:
        raise ValueError('requires native Responses input, without Chat conversions')
    if bool(source.get('stream')) != bool(case.get('stream')):
        raise ValueError('case streaming expectation disagrees with captured transport')
    request = dict(source)
    request.update(model=model, max_output_tokens=1024, temperature=0.0, seed=64013)
    payload = json.dumps(request, ensure_ascii=False, separators=(',', ':')).encode()
    metadata = _request_wire_metadata(payload, raw)
    if not set(metadata['request_changed_fields']) <= {'model', 'max_output_tokens', 'temperature', 'seed'}:
        raise ValueError('unexpected request rewrite')
    assert request['input'] == source['input']
    assert request.get('tools') == source.get('tools')
    assert request.get('stream') == source.get('stream')
    assert request.get('reasoning') == source.get('reasoning')
    return request, payload, metadata


def visible_text(response):
    """Only actual message output_text, never reasoning or a prior response."""
    return '\n'.join(part['text'] for item in response.get('output', [])
        if isinstance(item, dict) and item.get('type') == 'message'
        for part in item.get('content', [])
        if isinstance(part, dict) and part.get('type') == 'output_text'
        and isinstance(part.get('text'), str))


def semantic_checks(response, case):
    output = response.get('output') or []
    calls = [item for item in output if isinstance(item, dict) and item.get('type') == 'function_call']
    text = visible_text(response).strip()
    if case['kind'] == 'weather_tool':
        arguments = None
        if len(calls) == 1:
            try:
                arguments = json.loads(calls[0].get('arguments', ''))
            except (ValueError, TypeError):
                pass
        return {'one_requested_call': len(calls) == 1 and calls[0].get('name') == 'get_weather',
                'exact_arguments': arguments == {'city': case['city']}}
    if case['kind'] == 'short_title':
        return {'no_tool_call': not calls,
                'one_to_four_words': 1 <= len(text.split()) <= 4,
                'topic_present': case['topic'].casefold() in text.casefold(),
                'plain_single_line': bool(text) and '\n' not in text
                    and not any(c in text for c in ('"', '`', '#', '*'))}
    raise ValueError('unknown semantic check')


def row_checks(row, response, case, config):
    t, usage = row.get('timing') or {}, row.get('usage') or {}
    witness = t.get('generation_witness') or {}
    before, after = row['pressure_before'], row['pressure_after']
    checks = dict(semantic_checks(response, case))
    checks.update(
        completed=row.get('http_status') == 200 and row.get('response_status') == 'completed'
            and not row.get('error'),
        sufficient_actual_output=type(usage.get('output_tokens')) is int
            and 0 < usage['output_tokens'] < 1024,
        raw_witness_available=witness.get('available') is True
            and witness.get('prepared_prompt_token_ids_sha256') is not None
            and witness.get('generated_token_count') == usage.get('output_tokens'),
        profile_identity=row.get('runtime_profiles') == config['profiles']
            and row.get('runtime_profile_digest') == config['profile_digest']
            and not row.get('runtime_profile_overrides'),
        backend=row.get('backend') == 'voom',
        no_retry=int(t.get('memory_prefill_retries') or 0) == 0,
        metal=type(t.get('true_peak_metal_bytes')) in (int, float)
            and 0 < t['true_peak_metal_bytes'] <= 8_500_000_000,
        terminal_available=after['available_bytes'] >= 5_300_000_000,
        swap_used=after['swap_used_bytes'] - before['swap_used_bytes'] <= 16_000_000,
        actual_swap_out=after['swap_out_bytes'] - before['swap_out_bytes'] <= 16_000_000)
    if case.get('stream'):
        checks['stream_final_equal'] = row.get('streamed_text_matches_final') is True
    return checks


def native_pressure_summary(log_text):
    records = [json.loads(line[len('[process-memory] '):]) for line in log_text.splitlines()
               if line.startswith('[process-memory] ')]
    if not records or not all(r.get('process', {}).get('available') is True for r in records):
        return {'available': False, 'passed': False}
    first = records[0]
    result = dict(available=True, samples=len(records),
        minimum_available_bytes=min(r['system_available_bytes'] for r in records),
        maximum_footprint_bytes=max(r['process']['physical_footprint_bytes'] for r in records),
        maximum_compressed_bytes=max(r['process']['internal_compressed_ledger_bytes'] for r in records),
        swap_used_growth_bytes=max(r['system_swap_used_bytes'] for r in records)-first['system_swap_used_bytes'],
        actual_swap_out_growth_bytes=max(r['system_swap_out_bytes'] for r in records)-first['system_swap_out_bytes'])
    result['passed'] = (result['minimum_available_bytes'] >= 5_300_000_000
        and result['swap_used_growth_bytes'] <= 16_000_000
        and result['actual_swap_out_growth_bytes'] <= 16_000_000)
    return result


def run(config):
    if not 1 <= len(config['cases']) <= 3:
        raise ValueError('one to three bounded captured cases required')
    if any(k.startswith('VMODEL_') for k in os.environ):
        raise ValueError('profile-only parent must not have VMODEL overrides')
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == config['source_commit']
    pre = json.loads(Path(config['preflight']).read_text())
    assert pre['passed'] and pre['sample_seconds'] >= 30
    assert 0 <= time.monotonic() - pre['end']['monotonic_s'] < 120
    assert pre['end']['root_free_bytes'] >= 10_000_000_000
    profiles = apply_runtime_profiles(config['profiles'], environ={})
    assert profiles.profile_digest == config['profile_digest']
    assert _port_is_free(config['port'])
    for path in [config['result'], config['server_log'], *[c['response'] for c in config['cases']]]:
        assert not Path(path).exists()
    assert sha(config['history']) == config['history_sha256']
    prepared = [prepare_case(c, config['model']) for c in config['cases']]
    reference = json.loads(Path(config['reference']).read_text()) if config.get('reference') else None
    if reference:
        assert len(reference['cases']) == len(config['cases'])
    rows, failures = [], []
    document = dict(schema='voom.captured-transition-tracking-arm.v1', passed=False,
        cases=rows, failures=failures, source_commit=config['source_commit'],
        profiles=config['profiles'], profile_digest=config['profile_digest'],
        history_sha256_before=config['history_sha256'], generated_tools_executed=False,
        scope='Small captured shapes, original tools/input/stream/reasoning; only model/max1024/temp0/seed64013 overrides. Not full harness/Plex/large-context or full-state equivalence.',
        reference_sha256=sha(config['reference']) if reference else None)
    started = time.perf_counter()
    with open(config['server_log'], 'x') as log:
        command = [sys.executable, '-m', 'runtime.server', '--port', str(config['port'])]
        for profile in config['profiles']:
            command += ['--profile', profile]
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        document['server_pid'] = server.pid
        try:
            registry = _wait_ready(server, config['port'], 120)
            assert registry['vmodel_runtime_profile_digest'] == config['profile_digest']
            assert registry['vmodel_runtime_profiles'] == config['profiles']
            assert not registry.get('vmodel_runtime_profile_overrides')
            for index, (case, (request, payload, metadata)) in enumerate(zip(config['cases'], prepared)):
                terminal = []
                def observe(response):
                    if terminal:
                        raise ValueError('more than one terminal response observed')
                    _atomic_write_private(Path(case['response']), response)
                    terminal.append(response)
                before = asdict(_pressure())
                row = _post(f'http://127.0.0.1:{config["port"]}/v1/responses', payload,
                    timeout=1800, stream=bool(request.get('stream')), print_progress=True,
                    fail_on_memory_retry=True, response_observer=observe)
                row.update(metadata, pressure_before=before, pressure_after=asdict(_pressure()))
                response = terminal[0] if terminal else {}
                checks = row_checks(row, response, case, config)
                if reference:
                    ref = reference['cases'][index]
                    checks.update(
                        same_effective_request=metadata['request_sha256'] == ref['row']['request_sha256'],
                        finite_greedy_tokens_and_text=row.get('timing', {}).get('generation_witness')
                            == ref['row'].get('timing', {}).get('generation_witness'),
                        protocol_output=row.get('output_sha256') == ref['row'].get('output_sha256'),
                        canonical_calls=row.get('function_call_canonical_sha256')
                            == ref['row'].get('function_call_canonical_sha256'))
                rows.append(dict(name=case['name'], capture_sha256=case['sha256'],
                    response_path=case['response'], response_sha256=sha(case['response']) if terminal else None,
                    checks=checks, row=row))
                failures.extend(f'{case["name"]}: {k}' for k, ok in checks.items() if not ok)
                print(json.dumps(dict(case=case['name'], checks=checks,
                                      wall_seconds=row.get('wall_seconds'))), flush=True)
                if not terminal or row.get('response_status') != 'completed':
                    break
            if len(rows) != len(config['cases']):
                failures.append('not every case completed')
        except BaseException as error:
            failures.append('driver_error:' + type(error).__name__)
            raise
        finally:
            _stop_server(server)
            document.update(server_returncode=server.returncode,
                server_log_sha256=sha(config['server_log']), wall_seconds=time.perf_counter() - started,
                history_sha256_after=sha(config['history']))
            if document['history_sha256_after'] != config['history_sha256']:
                failures.append('saved transition history changed')
            try:
                document['native_pressure'] = native_pressure_summary(Path(config['server_log']).read_text())
            except (ValueError, TypeError, KeyError):
                document['native_pressure'] = {'available': False, 'passed': False}
            if not document['native_pressure']['passed']:
                failures.append('whole-arm periodic native pressure gate')
            if rows:
                first, last = rows[0]['row']['pressure_before'], rows[-1]['row']['pressure_after']
                document['whole_arm_swap_used_growth_bytes'] = last['swap_used_bytes']-first['swap_used_bytes']
                document['whole_arm_actual_swap_out_growth_bytes'] = last['swap_out_bytes']-first['swap_out_bytes']
                if max(document['whole_arm_swap_used_growth_bytes'],
                       document['whole_arm_actual_swap_out_growth_bytes']) > 16_000_000:
                    failures.append('whole-arm HTTP swap growth gate')
            document['passed'] = not failures
            _atomic_write_private(Path(config['result']), document)
    return 0 if document['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-json', required=True)
    raise SystemExit(run(json.loads(parser.parse_args().config_json)))
