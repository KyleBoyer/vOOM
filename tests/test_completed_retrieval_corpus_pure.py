"""CPU-only fixture/scorer/HTTP-client wiring tests, not model quality proof."""

import copy
import hashlib
import io
import json
import re
import sys
from types import SimpleNamespace

import pytest

from tests.fixtures import completed_retrieval_corpus as corpus
from tests.fixtures import qwen_large_context_output_gate as gate


class CharacterTokenizer:
    def encode(self, text):
        return SimpleNamespace(ids=list(text.encode()))

    def decode(self, ids, **kwargs):
        return bytes(ids).decode()


def fixture(seed=731908, domain='library', tokens=4096):
    return corpus.build_case(CharacterTokenizer(), tokens, fixture_seed=seed, domain=domain)


@pytest.mark.parametrize('domain', corpus.DOMAINS)
@pytest.mark.parametrize('seed', [731908, 931419, 538247])
def test_answers_only_once_in_distant_records_not_query_or_metadata(domain, seed):
    case = fixture(seed, domain)
    assert case == fixture(seed, domain)
    assert case.local_user_tokens == 4096
    suffix = case.user_text.split('End of archive.', 1)[1]
    assert set(case.metadata['requested_ids']) == case.expected.keys()
    positions = sorted(case.metadata['requested_record_token_positions'].values())
    assert 0 < positions[0] < positions[1] < case.local_user_tokens
    assert positions[1] - positions[0] > 1000
    for key, value in case.expected.items():
        assert key in suffix and value not in suffix
        assert case.user_text.count(value) == 1
        assert value not in json.dumps(case.metadata)
    assert 'VALIDATION' not in suffix and 'output limit' not in suffix


def test_seeds_change_codes_ids_order_and_positions_without_runtime_predicates():
    cases = [fixture(seed) for seed in range(20)]
    assert len({tuple(c.expected.items()) for c in cases}) == 20
    assert len({tuple(c.metadata['requested_record_token_positions'].values()) for c in cases}) > 10
    orders = {list(c.expected) == list(c.metadata['requested_record_token_positions']) for c in cases}
    assert orders == {True, False}
    assert len({fixture(domain=domain).metadata['user_text_sha256'] for domain in corpus.DOMAINS}) == 3


@pytest.mark.parametrize('seed', [None, True, -1, 2**63, 1.5, '1'])
def test_invalid_seeds_fail_before_tokenizer_use(seed):
    with pytest.raises(ValueError, match='fixture_seed'):
        corpus.build_case(None, 4096, fixture_seed=seed, domain='library')


@pytest.mark.parametrize('tokens', [True, 0, 1023, 1_000_001, float('nan')])
def test_invalid_context_requests_fail_before_tokenizer_use(tokens):
    with pytest.raises(ValueError, match='target_tokens'):
        corpus.build_case(None, tokens, fixture_seed=1, domain='library')


def response(text=None):
    text = json.dumps(fixture().expected) if text is None else text
    return dict(status='completed', usage=dict(output_tokens=52, input_tokens=4120),
        output=[dict(type='message', content=[dict(type='output_text', text=text)])])


def score(value):
    return corpus.score_response(value, fixture().expected, max_output_tokens=1024)


def test_exact_json_allows_whitespace_and_object_key_order_only():
    expected = dict(reversed(list(fixture().expected.items())))
    result = score(response('\n'+json.dumps(expected, indent=2)+'\n'))
    assert result['passed'] and all(result['checks'].values())


@pytest.mark.parametrize('mutation', ['wrong_value', 'lowercase', 'missing', 'extra', 'list',
    'nested', 'number', 'markdown', 'prose_before', 'prose_after', 'two_objects', 'duplicate'])
def test_no_repair_no_duplicate_keys_no_partial_credit(mutation):
    expected = fixture().expected
    value = dict(expected)
    first = next(iter(value))
    if mutation == 'wrong_value': value[first] = 'WRONG'
    if mutation == 'lowercase': value[first] = value[first].lower()
    if mutation == 'missing': value.pop(first)
    if mutation == 'extra': value['unrequested'] = 'CODE'
    if mutation == 'list': value = list(value.items())
    if mutation == 'nested': value = {'answer': value}
    if mutation == 'number': value[first] = 5
    text = json.dumps(value)
    if mutation == 'markdown': text = '```json\n'+text+'\n```'
    if mutation == 'prose_before': text = 'Here: '+text
    if mutation == 'prose_after': text += '\nDone.'
    if mutation == 'two_objects': text += text
    if mutation == 'duplicate': text = text[:-1]+','+json.dumps(first)+':'+json.dumps(expected[first])+'}'
    assert not score(response(text))['passed']


@pytest.mark.parametrize('mutation', ['capped', 'failed', 'cancelled', 'details', 'error',
    'tools', 'reasoning_only', 'top_only', 'inconsistent_top', 'extra_message',
    'refusal', 'invalid_output', 'invalid_part', 'missing_usage', 'bool_usage', 'cap_usage'])
def test_correct_looking_text_cannot_hide_protocol_or_termination_failure(mutation):
    value = response()
    answer = value['output'][0]['content'][0]['text']
    if mutation == 'capped': value.update(status='incomplete', incomplete_details={'reason': 'max_output_tokens'})
    if mutation == 'failed': value['status'] = 'failed'
    if mutation == 'cancelled': value['status'] = 'cancelled'
    if mutation == 'details': value['incomplete_details'] = {'reason': 'max_output_tokens'}
    if mutation == 'error': value['error'] = {'message': 'error'}
    if mutation == 'tools': value['output'].append({'type': 'function_call'})
    if mutation == 'reasoning_only': value['output'][0]['type'] = 'reasoning'
    if mutation == 'top_only': value.update(output=[], output_text=answer)
    if mutation == 'inconsistent_top': value['output_text'] = 'other'
    if mutation == 'extra_message': value['output'].append(copy.deepcopy(value['output'][0]))
    if mutation == 'refusal': value['output'][0]['content'].append({'type': 'refusal'})
    if mutation == 'invalid_output': value['output'] = 5
    if mutation == 'invalid_part': value['output'][0]['content'] = [None]
    if mutation == 'missing_usage': value.pop('usage')
    if mutation == 'bool_usage': value['usage']['output_tokens'] = True
    if mutation == 'cap_usage': value['usage']['output_tokens'] = 1024
    assert not score(value)['passed']


@pytest.mark.parametrize('budget', [True, 16, 128, 255, 1024.0])
def test_small_or_invalid_completion_budget_rejected(budget):
    with pytest.raises(ValueError, match='budget'):
        corpus.score_response(response(), fixture().expected, max_output_tokens=budget)


def task_args(completed=True, **changes):
    return SimpleNamespace(completed_retrieval=completed, min_output_tokens=None,
        min_consecutive_integers=None, max_swap_growth_mb=None,
        legacy_copy_output_diagnostic=False, fixture_seed=1 if completed else None,
        retrieval_domain='library' if completed else None, max_output_tokens=1024,
        expected_profile_digest='a'*64 if completed else None,
        max_peak_metal_gb=8.5, min_available_gb=5.3, **changes)


def test_legacy_diagnostic_defaults_and_new_completion_defaults_stay_distinct():
    import argparse
    old, new = task_args(False), task_args()
    for args in (old, new): gate._validate_task_options(argparse.ArgumentParser(), args)
    assert (old.min_output_tokens, old.min_consecutive_integers, old.max_swap_growth_mb) == (96, 8, 64)
    assert (new.min_output_tokens, new.min_consecutive_integers, new.max_swap_growth_mb) == (1, 0, 16)


@pytest.mark.parametrize('key,value', [('fixture_seed', None), ('fixture_seed', -1),
    ('retrieval_domain', None), ('max_output_tokens', 16), ('min_output_tokens', 96),
    ('min_consecutive_integers', 1), ('legacy_copy_output_diagnostic', True),
    ('expected_profile_digest', 'wrong'), ('max_swap_growth_mb', 64),
    ('max_peak_metal_gb', 9), ('min_available_gb', 5), ('max_swap_growth_mb', float('nan'))])
def test_completed_cli_cannot_weaken_acceptance_or_leak_answers(key, value):
    import argparse
    args = task_args(); setattr(args, key, value)
    with pytest.raises(SystemExit):
        gate._validate_task_options(argparse.ArgumentParser(), args)


@pytest.mark.parametrize('mutation', [None, 'incomplete', 'wrong', 'profile', 'checkpoint', 'witness', 'pressure', 'nan_peak', 'bad_usage'])
def test_completed_client_wires_hidden_fixture_and_publishes_honest_receipt(monkeypatch, tmp_path, mutation):
    tokenizer = tmp_path/'tokenizer.json'; tokenizer.touch()
    result = tmp_path/'result.json'
    monkeypatch.setattr(sys, 'argv', ['gate', '--model', 'test-model', '--tokenizer', str(tokenizer),
        '--result-json', str(result), '--completed-retrieval', '--fixture-seed', '731908',
        '--retrieval-domain', 'library', '--target-user-tokens', '4096', '--max-output-tokens', '1024',
        '--expected-profile-digest', 'a'*64])
    monkeypatch.setattr(gate, 'Tokenizer', SimpleNamespace(from_file=lambda _: CharacterTokenizer()))
    pressures = iter([gate.Pressure(7_000_000_000, 0, 0),
        gate.Pressure(4_000_000_000 if mutation == 'pressure' else 6_000_000_000, 0, 0)])
    monkeypatch.setattr(gate, '_pressure', lambda: next(pressures))
    def post(request, timeout):
        wire = json.loads(request.data)
        assert wire['model'] == 'test-model' and wire['max_output_tokens'] == 1024
        assert wire['tools'] == [] and wire['stream'] is False
        assert 'fixture_seed' not in wire and 'expected' not in wire
        user = wire['input'][1]['content'][0]['text']
        records, suffix = user.split('End of archive.', 1)
        lookup = dict(re.findall(r'accession=(accession-\d+); shelf_code=([A-Z0-9]+)', records))
        query = re.findall(r'accession-\d+', suffix)
        expected = {key: lookup[key] for key in query}
        assert len(expected) == 2 and all(value not in suffix for value in expected.values())
        out = response(json.dumps(expected))
        out.update(vmodel_backend='voom', vmodel_checkpoint='test-model', vmodel_runtime_profile_digest='a'*64,
            vmodel_timing=dict(true_peak_metal_bytes=100, memory_prefill_retries=0,
                generation_witness=dict(available=True, generated_token_count=52,
                    generated_token_ids_sha256='b'*64, prepared_prompt_token_ids_sha256='c'*64)))
        if mutation == 'incomplete': out.update(status='incomplete', incomplete_details={'reason': 'max_output_tokens'})
        if mutation == 'wrong': out['output'][0]['content'][0]['text'] = '{}'
        if mutation == 'profile': out['vmodel_runtime_profile_digest'] = 'd'*64
        if mutation == 'checkpoint': out['vmodel_checkpoint'] = 'other-model'
        if mutation == 'witness': out['vmodel_timing'].pop('generation_witness')
        if mutation == 'nan_peak': out['vmodel_timing']['true_peak_metal_bytes'] = float('nan')
        if mutation == 'bad_usage': out['usage'] = None
        return io.BytesIO(json.dumps(out).encode())
    monkeypatch.setattr(gate.urllib.request, 'urlopen', post)
    assert gate.main() == (0 if mutation is None else 1)
    report = json.loads(result.read_text())
    assert report['passed'] is (mutation is None)
    assert report['request']['completed_answer_benchmark'] is True
    assert report['request']['task'] == 'completed-synthetic-retrieval'
    assert report['request']['fixture']['answers_in_suffix'] is False
    assert report['request']['min_consecutive_integers'] == 0
    assert report['result']['canary_a_found'] is None
    assert result.stat().st_mode & 0o777 == 0o600
    receipt = tmp_path/'result.response.json'
    if mutation == 'nan_peak':
        assert report['result']['response_path'] is None and not receipt.exists()
    else:
        assert hashlib.sha256(receipt.read_bytes()).hexdigest() == report['result']['response_sha256']
        assert receipt.stat().st_mode & 0o777 == 0o600
        assert corpus.score_response(json.loads(receipt.read_text()), fixture().expected,
            max_output_tokens=1024) == report['result']['completion']
    for value in fixture().expected.values():
        assert value not in result.read_text()


def test_legacy_sustained_output_client_still_labels_cap_as_diagnostic(monkeypatch, tmp_path):
    tokenizer = tmp_path/'tokenizer.json'; tokenizer.touch()
    result = tmp_path/'diagnostic.json'
    monkeypatch.setattr(sys, 'argv', ['gate', '--model', 'test-model', '--tokenizer', str(tokenizer),
        '--result-json', str(result), '--target-user-tokens', '4096'])
    monkeypatch.setattr(gate, 'Tokenizer', SimpleNamespace(from_file=lambda _: CharacterTokenizer()))
    monkeypatch.setattr(gate, '_pressure', lambda: gate.Pressure(7_000_000_000, 0, 0))
    out = response(f'A={gate.CANARY_A} B={gate.CANARY_B} VALIDATION 001 002 003 004 005 006 007 008')
    out.update(status='incomplete', incomplete_details={'reason': 'max_output_tokens'},
               usage={'input_tokens': 4120, 'output_tokens': 128}, vmodel_timing={'true_peak_metal_bytes': 100})
    def post(request, timeout):
        wire = json.loads(request.data)
        assert wire['max_output_tokens'] == 128
        user = wire['input'][1]['content'][0]['text']
        assert user.count(gate.CANARY_A) == user.count(gate.CANARY_B) == 1
        return io.BytesIO(json.dumps(out).encode())
    monkeypatch.setattr(gate.urllib.request, 'urlopen', post)
    assert gate.main() == 0
    report = json.loads(result.read_text())
    assert report['schema'] == 'voom.qwen-large-context-output-gate.v3'
    assert report['request']['completed_answer_benchmark'] is False
    assert report['request']['task'] == 'retrieval-and-sustained-output'
    assert report['result']['response_status'] == 'incomplete'
    assert report['request']['min_output_tokens'] == 96
    assert not (tmp_path/'diagnostic.response.json').exists()
