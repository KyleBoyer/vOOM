import json

import pytest

from runtime.profiles import apply_runtime_profiles
from tests.fixtures import huihui_captured_action_gate as gate


def call(arguments, name='plugin__plex__plex_list_library'):
    return dict(type='function_call', name=name, arguments=json.dumps(arguments))


@pytest.mark.parametrize('arguments', [{}, {'limit': 50, 'offset': 0},
    {'limit': None, 'offset': None}, {'limit': 20.0, 'offset': 0.0}])
def test_initial_listing_only_is_not_final_plex_grade(arguments):
    assert all(gate.action_checks(dict(output=[call(arguments)])).values())


@pytest.mark.parametrize('output', [[], [call({}, 'wrong')],
    [call({})]*3, [call([])], [call({'limit': 101})],
    [call({'limit': True})], [call({'limit': 0})],
    [call({'limit': float('nan')})], [call({'offset': float('inf')})],
    [call({'offset': 1})], [call({'offset': False})],
    [dict(type='function_call', name='plugin__plex__plex_list_library', arguments='{bad')],
    [dict(type='message', content=[dict(type='output_text', text='Done')])]])
def test_invalid_or_non_listing_first_action_rejected(output):
    assert not all(gate.action_checks(dict(output=output)).values())


def test_evaluation_profile_only_disables_host_render_and_prompt_cache():
    base, audit = {}, {}
    apply_runtime_profiles(['huihui-qwen38-27b-fast-agent-mtpquant'], environ=base)
    apply_runtime_profiles(['huihui-qwen38-27b-fast-agent-model-only-audit'], environ=audit)
    # Profile identity fields naturally differ; settings remain otherwise exact.
    settings = lambda env: {k: v for k, v in env.items() if k.startswith('VMODEL_') and k != 'VMODEL_PROFILE'}
    assert settings(audit) == {**settings(base),
        'VMODEL_FAST_TOOL_GATEWAY_DETERMINISTIC_POLICY': '0',
        'VMODEL_QWEN35_HOT_KV': '0', 'VMODEL_QWEN35_MIXED_DEPTH_HOT_KV_PERSIST': '0'}
    assert audit['VMODEL_FAST_TOOL_GATEWAY'] == '1'
    assert audit['VMODEL_QWEN35_LOSSY_SUFFIX_PREFILL'] == '16:1024'


def row():
    return dict(http_status=200, response_status='completed', error=None,
        backend='voom', runtime_profiles=['audit'], runtime_profile_digest='digest',
        runtime_profile_overrides=[], streamed_text_matches_final=True,
        usage=dict(output_tokens=60, input_tokens_details=dict(cached_tokens=0)),
        timing=dict(generation_witness=dict(available=True, generated_token_count=55,
            prepared_prompt_token_ids_sha256='hash'), true_peak_metal_bytes=100,
            memory_prefill_retries=0),
        pressure_before=dict(available_bytes=7_000_000_000, swap_used_bytes=0, swap_out_bytes=0),
        pressure_after=dict(available_bytes=6_000_000_000, swap_used_bytes=0, swap_out_bytes=0))


def check(value, status='completed'):
    return gate.acceptance(value, dict(status=status, output=[call({})]),
        dict(profiles=['audit'], profile_digest='digest'))


def test_gateway_witness_is_single_generation_not_aggregate_token_equality():
    assert all(check(row()).values())


@pytest.mark.parametrize('count', [0, True, 1024, 1025])
def test_output_cap_not_a_completed_action(count):
    value = row(); value['usage']['output_tokens'] = count
    assert not check(value)['not_output_capped']


def test_incomplete_actual_response_not_hidden_by_summary():
    assert not check(row(), 'incomplete')['completed']


def test_prompt_cache_reuse_rejects_cold_claim():
    value = row(); value['usage']['input_tokens_details']['cached_tokens'] = 1
    assert not check(value)['no_prompt_reuse']


def test_physical_swap_out_fails_despite_unchanged_swap_usage():
    value = row(); value['pressure_after']['swap_out_bytes'] = 16_000_001
    assert not check(value)['actual_swap_out']
