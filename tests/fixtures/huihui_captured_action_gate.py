#!/usr/bin/env python3
"""Completed initial captured action; never a final paginated Plex score.

Original input/tools/stream/reasoning; only model/max1024/temp0/seed overrides.
The selected lossy gateway profile transforms the model's prepared prompt.
All generated tools remain unexecuted. No final-answer rendering or repair.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from runtime.profiles import apply_runtime_profiles
from tests.fixtures.captured_transition_tracking_gate import (
    native_pressure_summary, prepare_case, sha)
from tests.fixtures.qwen3_large_agent_replay_gate import _post
from tests.fixtures.qwen4_hot_boundary_http_probe import _atomic_write_private
from tests.fixtures.runtime_profile_http_gate import (
    _port_is_free, _pressure, _stop_server, _wait_ready)


def action_checks(response):
    """Only initial read-only listing validity, not final-answer intelligence."""
    output = response.get('output') or []
    calls = [x for x in output if isinstance(x, dict) and x.get('type') == 'function_call']
    arguments = []
    for call in calls:
        try:
            value = json.loads(call.get('arguments', ''))
            arguments.append(value if isinstance(value, dict) else None)
        except (ValueError, TypeError):
            arguments.append(None)
    def number(value):
        return type(value) in (int, float) and math.isfinite(value) and value == int(value)
    return dict(
        listing_calls_only=1 <= len(calls) <= 2 and all(
            c.get('name') == 'plugin__plex__plex_list_library' for c in calls),
        object_arguments=bool(arguments) and all(isinstance(a, dict) for a in arguments),
        bounded_first_page=bool(arguments) and all(isinstance(a, dict)
            and (a.get('offset') is None or (number(a['offset']) and a['offset'] == 0))
            and (a.get('limit') is None or (number(a['limit']) and 0 < a['limit'] <= 100))
            for a in arguments))


def acceptance(row, response, config):
    t, usage = row.get('timing') or {}, row.get('usage') or {}
    witness = t.get('generation_witness') or {}
    before, after = row['pressure_before'], row['pressure_after']
    checks = action_checks(response)
    checks.update(
        completed=row.get('http_status') == 200 and row.get('response_status') == 'completed'
            and not row.get('error') and response.get('status') == 'completed',
        not_output_capped=type(usage.get('output_tokens')) is int
            and 0 < usage['output_tokens'] < 1024,
        raw_witness=witness.get('available') is True
            and witness.get('prepared_prompt_token_ids_sha256') is not None
            and type(witness.get('generated_token_count')) is int
            and witness['generated_token_count'] > 0,
        # Gateway workflows may generate more than once. This hash witnesses
        # the exposed engine generation, not aggregate workflow token equality.
        profile=row.get('runtime_profiles') == config['profiles']
            and row.get('runtime_profile_digest') == config['profile_digest']
            and not row.get('runtime_profile_overrides'),
        backend=row.get('backend') == 'voom',
        no_retry=int(t.get('memory_prefill_retries') or 0) == 0,
        no_prompt_reuse=usage.get('input_tokens_details', {}).get('cached_tokens') == 0,
        metal=type(t.get('true_peak_metal_bytes')) in (int, float)
            and 0 < t['true_peak_metal_bytes'] <= 8_500_000_000,
        terminal_available=after['available_bytes'] >= 5_300_000_000,
        swap_used=after['swap_used_bytes'] - before['swap_used_bytes'] <= 16_000_000,
        actual_swap_out=after['swap_out_bytes'] - before['swap_out_bytes'] <= 16_000_000,
        stream_matches=row.get('streamed_text_matches_final') is True)
    return checks


def run(config):
    assert not any(k.startswith('VMODEL_') for k in os.environ)
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == config['source_commit']
    pre = json.loads(Path(config['preflight']).read_text())
    assert pre['passed'] and pre['sample_seconds'] >= 30
    assert 0 <= time.monotonic() - pre['end']['monotonic_s'] < 120
    assert pre['known_transcoders']['passed'] is True
    assert pre['end']['root_free_bytes'] >= 10_000_000_000
    env = {}
    profile = apply_runtime_profiles(config['profiles'], environ=env)
    assert profile.profile_digest == config['profile_digest']
    assert env['VMODEL_FAST_TOOL_GATEWAY_DETERMINISTIC_POLICY'] == '0'
    assert env['VMODEL_QWEN35_HOT_KV'] == '0'
    assert env['VMODEL_QWEN35_MIXED_DEPTH_HOT_KV_PERSIST'] == '0'
    assert env['VMODEL_GENERATION_WITNESS'] == env['VMODEL_HOST_ACTIVITY_WITNESS'] == '1'
    assert _port_is_free(config['port'])
    for key in ('result', 'response', 'server_log'):
        assert not Path(config[key]).exists()
    request, wire, metadata = prepare_case(config['case'], config['model'])
    assert metadata['request_sha256'] == config['wire_sha256']
    assert len(request['tools']) == 134 and request['stream'] is True
    for path, digest in config['metadata_hashes'].items():
        assert sha(path) == digest
    started = time.perf_counter()
    document = dict(schema='voom.huihui-captured-initial-action.v1', passed=False,
        scope=__doc__, config=config, preflight_sha256=sha(config['preflight']),
        request=metadata, generated_tools_executed=False, final_plex_score=None,
        failures=[], source_commit=config['source_commit'])
    server = None
    try:
        with open(config['server_log'], 'x') as log:
            command = [sys.executable, '-m', 'runtime.server', '--port', str(config['port'])]
            for name in config['profiles']:
                command += ['--profile', name]
            server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            document['server_pid'] = server.pid
            registry = _wait_ready(server, config['port'], 120)
            assert registry['vmodel_runtime_profile_digest'] == config['profile_digest']
            assert registry['vmodel_runtime_profiles'] == config['profiles']
            assert not registry.get('vmodel_runtime_profile_overrides')
            terminal = []
            def observe(value):
                assert not terminal
                _atomic_write_private(Path(config['response']), value)
                terminal.append(value)
            before = asdict(_pressure())
            row = _post(f'http://127.0.0.1:{config["port"]}/v1/responses', wire,
                timeout=1800, stream=True, print_progress=True,
                fail_on_memory_retry=True, response_observer=observe)
            row.update(pressure_before=before, pressure_after=asdict(_pressure()))
            document['row'] = row
            document['checks'] = acceptance(row, terminal[0] if terminal else {}, config)
            document['failures'] += [k for k, v in document['checks'].items() if not v]
            if terminal:
                document['response_sha256'] = sha(config['response'])
    except BaseException as error:
        document['failures'].append('driver_error:' + type(error).__name__)
    finally:
        if server is not None:
            _stop_server(server)
            document['server_returncode'] = server.returncode
        try:
            document['server_log_sha256'] = sha(config['server_log'])
            document['native_pressure'] = native_pressure_summary(Path(config['server_log']).read_text())
        except (OSError, ValueError, TypeError, KeyError):
            document['native_pressure'] = dict(available=False, passed=False)
        if not document['native_pressure'].get('passed'):
            document['failures'].append('whole_run_pressure')
        if not document['native_pressure'].get('known_transcoders', {}).get('passed'):
            document['failures'].append('known_transcoder_isolation')
        document['metadata_unchanged'] = all(
            sha(path) == digest for path, digest in config['metadata_hashes'].items())
        if not document['metadata_unchanged']:
            document['failures'].append('model_metadata_changed')
        document['wall_seconds'] = time.perf_counter() - started
        document['passed'] = not document['failures']
        _atomic_write_private(Path(config['result']), document)
    print(json.dumps(dict(passed=document['passed'], failures=document['failures'],
        wall_seconds=document['wall_seconds'])), flush=True)
    return 0 if document['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-json', required=True)
    raise SystemExit(run(json.loads(parser.parse_args().config_json)))
