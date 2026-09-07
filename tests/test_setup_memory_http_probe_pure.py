"""Setup profiler's content redaction, stop-before-body and boundedness gates."""

import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

PATH = Path(__file__).parent/'fixtures'/'setup_memory_http_probe.py'
spec = importlib.util.spec_from_file_location('setup_memory_http_probe', PATH)
probe_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe_module)
SHA = hashlib.sha256(b'[17]').hexdigest()


def frame(name='_engine_generate', module='runtime.server', **local):
    return SimpleNamespace(f_globals={'__name__': module, '__file__': '/module.py'},
                           f_code=SimpleNamespace(co_qualname=name, co_filename='/module.py'),
                           f_locals=local)


def make_probe(**changes):
    return probe_module.SetupProfiler(**dict(
        sample=lambda: {'available': True}, publish=lambda d: None,
        expected_count=1, expected_sha=SHA, output_cap=1024, **changes))


@pytest.mark.parametrize('count', [True, 0, -1, 1000001, 1.0])
def test_count_is_bounded(count):
    with pytest.raises(ValueError):
        probe_module.SetupProfiler(sample=None, publish=None,
            expected_count=count, expected_sha=SHA, output_cap=1024)


@pytest.mark.parametrize('cap', [True, 0, 4097, 1.0])
def test_output_cap_is_bounded(cap):
    with pytest.raises(ValueError):
        probe_module.SetupProfiler(sample=None, publish=None,
            expected_count=1, expected_sha=SHA, output_cap=cap)


@pytest.mark.parametrize('bad', ['', 'x'*64, 'A'*64, None])
def test_sha_must_be_lowercase_hex(bad):
    with pytest.raises(ValueError):
        probe_module.SetupProfiler(sample=None, publish=None,
            expected_count=1, expected_sha=bad, output_cap=1024)


def test_actual_profile_callback_stops_before_body_and_redacts_prompt():
    documents, body = [], []
    namespace = {'__name__': 'runtime.server', 'body': body}
    exec('def _engine_generate(engine, *args, **kwargs):\n body.append("executed")', namespace)
    engine = type('Qwen4MTPSpeculativeEngine', (), {})()
    prompt = SimpleNamespace(token_ids=[17], secret='DO_NOT_EXPORT')
    profiler = probe_module.SetupProfiler(sample=lambda: {'available': True},
        publish=documents.append, expected_count=1, expected_sha=SHA, output_cap=1024)
    previous = sys.getprofile()
    try:
        sys.setprofile(profiler)
        with pytest.raises(probe_module.SetupDiagnosticStop):
            namespace['_engine_generate'](engine, prompt, 1024)
    finally:
        sys.setprofile(previous)
    assert not body and len(documents) == 1
    assert documents[0]['prepared_identity']['passed']
    assert documents[0]['generated_tokens'] == 0
    assert not documents[0]['completed_model_response']
    assert 'DO_NOT_EXPORT' not in str(documents)


def test_identity_failure_still_stops_no_accidental_generation():
    profiler = make_probe()
    with pytest.raises(probe_module.SetupDiagnosticStop):
        profiler(frame(args=(SimpleNamespace(token_ids=[True]), 1024)), 'call', None)
    assert profiler.claimed


def test_repeated_helpers_and_total_records_bounded():
    profiler = make_probe()
    f = frame(name='_xgrammar', module='runtime.structured')
    for _ in range(10):
        profiler(f, 'call', None)
        profiler(f, 'return', None)
    assert len(profiler.events) == 4
    for _ in range(150):
        profiler.observe('test', 'boundary')
    assert len(profiler.events) == 128 and profiler.capped


def test_observation_failure_is_unavailable_without_breaking_setup():
    def fail():
        raise RuntimeError('DO_NOT_EXPORT')
    profiler = probe_module.SetupProfiler(sample=fail, publish=lambda d: None,
        expected_count=1, expected_sha=SHA, output_cap=1024)
    profiler(frame(name='_compiler', module='runtime.structured'), 'call', None)
    assert profiler.events[0]['values'] == {'available': False, 'error_type': 'RuntimeError'}


def test_unselected_functions_and_c_events_do_not_observe():
    profiler = make_probe()
    profiler(frame(name='unrelated'), 'call', None)
    profiler(frame(), 'c_call', None)
    assert not profiler.events and not profiler.claimed


def test_import_time_generated_exec_is_not_a_module_boundary():
    generated = frame(name='<module>', module='runtime.engine')
    generated.f_code.co_filename = '<string>'
    assert probe_module.classify(generated) is None
    real = frame(name='<module>', module='runtime.engine')
    assert probe_module.classify(real) == 'runtime.engine.<module>'
