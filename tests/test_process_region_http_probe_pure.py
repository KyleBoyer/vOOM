"""No-array diagnostic-wrapper forwarding, errors, bounds and redaction."""

import copy
from types import SimpleNamespace

import pytest

from tests.fixtures.process_region_http_probe import RegionHTTPProbe, ownership_summary


class Engine:
    target = SimpleNamespace()


def setup(original, snapshot=lambda target: {'available': True}, **extra):
    documents = []
    probe = RegionHTTPProbe(original, artifact='unused', expected_tokens=3,
                            output_cap=1024, engine_type=Engine, snapshot=snapshot,
                            publish=lambda path, doc: documents.append(doc), **extra)
    prompt = SimpleNamespace(token_ids=(4, 8, 15))
    return probe, prompt, documents


def test_forwards_exact_objects_once_and_returns_original_result():
    engine = Engine()
    result = {'tokens': [16, 23], 'text': 'PRIVATE TEXT', 'total_s': 1.5}
    expected = copy.deepcopy(result)
    calls = []
    def generate(received, *args, **kwargs):
        calls.append((received, args, kwargs))
        return result
    probe, prompt, documents = setup(generate)
    sampling = object()
    actual = probe(engine, prompt, 1024, sampling=sampling)
    assert actual is result and result == expected
    assert calls == [(engine, (prompt, 1024), {'sampling': sampling})]
    assert documents[0]['available'] and documents[0]['generated_tokens']['count'] == 2
    assert 'PRIVATE' not in str(documents)
    with pytest.raises(RuntimeError):
        probe(engine, prompt, 1024)
    assert len(calls) == 1


def test_observation_error_does_not_modify_success():
    result = {'tokens': [1], 'text': 'ok'}
    def fail(target):
        raise RuntimeError('PRIVATE')
    probe, prompt, documents = setup(lambda *a, **k: result, snapshot=fail)
    assert probe(Engine(), prompt=prompt, max_tokens=1024) is result
    assert documents[0]['before']['values']['available'] is False
    assert 'PRIVATE' not in str(documents)


def test_generation_error_is_preserved_and_published():
    error = KeyboardInterrupt()
    def fail(*args, **kwargs):
        raise error
    probe, prompt, documents = setup(fail)
    with pytest.raises(KeyboardInterrupt) as raised:
        probe(Engine(), prompt, 1024)
    assert raised.value is error
    assert documents[0]['generation_error_type'] == 'KeyboardInterrupt'
    assert not documents[0]['available'] and 'after' in documents[0]


def test_publication_and_log_failure_do_not_replace_response(monkeypatch):
    result = {'tokens': [1], 'text': 'ok'}
    probe, prompt, _ = setup(lambda *a, **k: result)
    def fail(*args, **kwargs):
        raise OSError('PRIVATE')
    probe.publish = fail
    monkeypatch.setattr('builtins.print', fail)
    assert probe(Engine(), prompt, 1024) is result


@pytest.mark.parametrize('cap', [1, True, 1024.0, None])
def test_wrong_budget_never_generates(cap):
    called = []
    probe, prompt, documents = setup(lambda *a, **k: called.append(1))
    with pytest.raises(ValueError):
        probe(Engine(), prompt, cap)
    assert not called and not documents


def test_wrong_prompt_never_generates():
    called = []
    probe, prompt, documents = setup(lambda *a, **k: called.append(1))
    prompt.token_ids = (4, 8)
    with pytest.raises(ValueError):
        probe(Engine(), prompt, 1024)
    assert not called and not documents


def test_plain_owner_logical_metadata_does_not_mutate_or_sum_aliases():
    class KV:
        offset = 12
        def nbytes(self):
            return 8192
    kv = KV()
    slots = [SimpleNamespace(kv=kv, tokens=(1, 2, 3))]
    owner = SimpleNamespace(last_kv=kv, _hot_prompt_slots=slots)
    r = ownership_summary(owner, KV)
    assert r['endpoint']['logical_bytes'] == 8192
    assert r['retained_slots'][0]['aliases_endpoint']
    assert r['retained_slots'][0]['prefix_token_count'] == 3
    assert owner.last_kv is kv and owner._hot_prompt_slots is slots
    assert not r['physical_ownership_proven']


def test_unknown_slot_owners_fail_closed():
    assert not ownership_summary(SimpleNamespace(_hot_prompt_slots=None), object)['available']
