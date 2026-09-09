"""Bounded completed-generation evidence that survives a later gateway failure.

No tensors, token IDs, prompts, tool arguments, answer text, or model decisions
are created or retained. These are per-generation observations, not evidence of
request completion or full failed-phase coverage. Gated by generation-witness.
"""

import json
import math
import os
import re

PREFIX = '[gateway-phase-complete] '
SCHEMA = 'voom.gateway-phase-completion.v1'
SCOPE = 'completed_single_engine_generation_not_request_completion'
INTEGER_FIELDS = ('input_tokens', 'output_tokens', 'generation_max_tokens',
                  'cached_tokens', 'cache_write_tokens', 'effective_reused_tokens',
                  'weight_store_bytes_read', 'true_peak_metal_bytes',
                  'prompt_state_approximate', 'paged_kv_budget_bytes')
SECONDS_FIELDS = ('suffix_prefill_seconds', 'decode_seconds', 'total_engine_seconds')
WITNESS_INTEGERS = ('generated_token_count', 'prepared_prompt_token_count', 'engine_text_bytes')
WITNESS_HASHES = ('generated_token_ids_sha256', 'prepared_prompt_token_ids_sha256', 'engine_text_sha256')


def _integer(value):
    return type(value) is int and 0 <= value <= 10**18


def completion_record(request_id, phase):
    """Copy only bounded scalars/hashes; unknown fields can never leak content."""
    record = dict(schema=SCHEMA, scope=SCOPE, available=False)
    if type(request_id) is not str or re.fullmatch(r'resp_[0-9a-f]{24}', request_id) is None:
        return {**record, 'reason': 'invalid-server-request-id'}
    record['request_id'] = request_id
    if type(phase) is not dict or phase.get('phase') not in ('gateway_decision', 'gateway_execution'):
        return {**record, 'reason': 'invalid-phase'}
    record['phase'] = phase['phase']
    witness = phase.get('generation_witness')
    if type(witness) is not dict or witness.get('available') is not True:
        return {**record, 'reason': 'generation-witness-unavailable'}
    constants = dict(schema='voom.generation-witness.v1',
        scope='single_engine_generation_before_protocol_parsing',
        token_hash_encoding='compact-json-integer-array-v1')
    if any(witness.get(k) != v for k,v in constants.items()):
        return {**record, 'reason': 'invalid-generation-witness'}
    if (not all(_integer(witness.get(k)) for k in WITNESS_INTEGERS)
            or not all(type(witness.get(k)) is str and re.fullmatch(r'[0-9a-f]{64}', witness[k])
                       for k in WITNESS_HASHES)):
        return {**record, 'reason': 'invalid-generation-witness'}
    required = ('input_tokens', 'output_tokens', 'generation_max_tokens')
    if (not all(k in phase for k in required)
            or any(not _integer(phase[k]) for k in INTEGER_FIELDS if k in phase)
            or any(type(phase.get(k)) not in (int, float)
                   or not 0 <= phase[k] <= 1e9 or not math.isfinite(phase[k])
                   for k in SECONDS_FIELDS)):
        return {**record, 'reason': 'invalid-phase-metrics'}
    if (phase['input_tokens'] != witness['prepared_prompt_token_count']
            or phase['output_tokens'] != witness['generated_token_count']):
        return {**record, 'reason': 'phase-token-count-mismatch'}
    if phase.get('termination_reason') not in ('eos', 'stop', 'grammar', 'max_tokens'):
        return {**record, 'reason': 'unknown-termination'}
    if ('weight_store_bytes_read' in phase and (
            phase.get('weight_store_bytes_read_source') not in ('path_stats', 'result', 'unavailable')
            or phase.get('weight_store_bytes_read_scope') != 'single_engine_phase_logical_not_physical')):
        return {**record, 'reason': 'invalid-io-scope'}
    # max_tokens is a returned generation, NOT natural/full-request completion.
    record.update(available=True, termination_reason=phase['termination_reason'],
        generation_witness={**constants, 'available': True,
            **{k:witness[k] for k in (*WITNESS_INTEGERS, *WITNESS_HASHES)}},
        **{k:phase[k] for k in (*INTEGER_FIELDS, *SECONDS_FIELDS) if k in phase})
    if 'weight_store_bytes_read' in phase:
        record.update(weight_store_bytes_read_source=phase['weight_store_bytes_read_source'],
                      weight_store_bytes_read_scope=phase['weight_store_bytes_read_scope'])
    return record


def emit_completion(request_id, phase):
    """One flushed bounded line; observation failure never changes serving."""
    if os.environ.get('VMODEL_GENERATION_WITNESS', '0').strip() != '1':
        return
    try:
        encoded = json.dumps(completion_record(request_id, phase), sort_keys=True, allow_nan=False)
        if len(encoded) > 8192:
            encoded = json.dumps(dict(schema=SCHEMA, scope=SCOPE, available=False, reason='record-limit'))
        print(PREFIX + encoded, flush=True)
    except Exception:
        # A broken sink cannot erase or retry the already completed generation.
        # Missing/malformed log records must fail any external coverage gate.
        pass
