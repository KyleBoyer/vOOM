import ast
import json
from pathlib import Path

import pytest

from runtime import decode_progress as progress
from runtime.profiles import apply_runtime_profiles


def test_flag_is_explicit_and_has_no_default_observer():
    assert not progress.enabled({})
    assert progress.enabled({progress.FLAG: '1'})
    for bad in ('auto', '', 1, True, None):
        with pytest.raises(ValueError):
            progress.enabled({progress.FLAG: bad})


def test_scalar_records_rate_limit_and_terminal_record(monkeypatch, capsys):
    clock = [100.0]
    monkeypatch.setattr(progress.time, 'perf_counter', lambda: clock[0])
    p = progress.DecodeProgress(100.0, 1024)
    p.record(4, 1, 4, 2)
    clock[0] = 110.0
    p.record(10, 4, 16, 5)
    clock[0] = 130.0
    p.record(20, 7, 28, 12)
    clock[0] = 131.0
    p.record(22, 8, 32, 14, final=True)
    rows = [json.loads(line.removeprefix(progress.PREFIX))
            for line in capsys.readouterr().out.splitlines()]
    assert [r['accepted_output_tokens_including_bootstrap'] for r in rows] == [4,20,22]
    assert [r['terminal_round'] for r in rows] == [False,False,True]
    assert rows[-1]['elapsed_seconds'] == 31.0
    assert all('text' not in r and 'token_ids' not in r for r in rows)


def test_output_failure_is_not_a_generation_failure(monkeypatch):
    def fail(*a, **kw): raise BrokenPipeError()
    monkeypatch.setattr('builtins.print', fail)
    p = progress.DecodeProgress(0, 1024)
    p.record(1,1,0,0)
    assert p.failed
    p.record(2,2,0,0,final=True)


def test_costs_use_measured_plain_baseline_and_include_elapsed_overhead(monkeypatch, capsys):
    monkeypatch.setattr(progress.time, 'perf_counter', lambda: 120.0)
    p = progress.DecodeProgress(100.0, 1024)
    p.record(8, 5, 8, 4, plain_seconds=6.0, plain_sweeps=3,
             draft_seconds=4.0, verifier_seconds=7.0, speculative_rounds=2,
             rollback_seconds=1.0, adaptive_disabled=True)
    c = json.loads(capsys.readouterr().out.removeprefix(progress.PREFIX))['costs']
    assert c['measured_plain_seconds_per_token'] == 2.0
    assert c['estimated_plain_equivalent_seconds'] == 14.0
    assert c['estimated_net_seconds_including_overhead'] == -6.0
    assert c['adaptive_disabled'] is True
    assert c['rollback_seconds'] == 1.0
    assert 'not an A/B speed proof' in c['scope']


def test_costs_do_not_invent_plain_baseline(monkeypatch, capsys):
    monkeypatch.setattr(progress.time, 'perf_counter', lambda: 120.0)
    p = progress.DecodeProgress(100.0, 1024)
    p.record(4, 1, 4, 3, verifier_seconds=2.0, speculative_rounds=1)
    c = json.loads(capsys.readouterr().out.removeprefix(progress.PREFIX))['costs']
    assert c['measured_plain_seconds_per_token'] is None
    assert c['estimated_net_seconds_including_overhead'] is None


@pytest.mark.parametrize('invalid', [-1.0, float('nan'), float('inf'), 'bad'])
def test_bad_cost_telemetry_disables_observer_not_generation(invalid):
    p = progress.DecodeProgress(0, 1024)
    p.record(4, 1, 4, 2, draft_seconds=invalid)
    assert p.failed


def test_real_hook_is_after_committed_endpoint_and_before_terminal_break():
    tree = ast.parse(Path('runtime/qwen35_mtp.py').read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='generate')
    src = ast.unparse(fn)
    assert src.index('decode_progress_enabled()') < src.index('bootstrap =')
    assert src.index('catchup_tok = emitted[-1] if terminal_round') < src.index('decode_progress.record(')
    call = next(n for n in ast.walk(fn) if isinstance(n,ast.Call)
                and ast.unparse(n.func)=='decode_progress.record')
    stop = next(n for n in ast.walk(fn) if isinstance(n,ast.If)
                and ast.unparse(n.test)=='terminal_round'
                and isinstance(n.body[0],ast.Break))
    assert call.lineno < stop.lineno
    bindings = {k.arg: ast.unparse(k.value) for k in call.keywords}
    assert bindings['plain_seconds'] == 'plain_round_s'
    assert bindings['plain_sweeps'] == 'plain_timed_sweeps'
    assert bindings['draft_seconds'] == 'draft_round_s'
    assert bindings['verifier_seconds'] == 'verifier_round_s'
    assert bindings['rollback_seconds'] == 'kda_factor_restore_s'
    assert 'DecodeProgress(decode_t0, max_tokens) if report_decode_progress else None' in src


def test_profile_only_adds_observer_flag():
    before={};after={};base=['huihui-qwen38-27b-harness-preview']
    apply_runtime_profiles(base,environ=before)
    apply_runtime_profiles(base+['decode-progress-witness'],environ=after)
    assert after=={**before,progress.FLAG:'1'}
