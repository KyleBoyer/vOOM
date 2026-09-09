"""Bounded pre-generation choice/activation provenance; never a full contract.

Observe the configured constraint's plain scalar fields, not its matcher,
token mask, tensors or schema. No model evaluation or serving decisions occur.
Equal observations are necessary provenance, NOT full generation equivalence.
"""

import hashlib
import json
import os
import re

PREFIX = '[gateway-generation-start] '
SCHEMA = 'voom.gateway-generation-start.v1'
SCOPE = 'choice_and_prior_activation_only_not_full_generation_contract'
CONTROL_FIELDS = ('phase', 'tool_choice_kind', 'tool_choice_sha256',
    'prior_activation_count', 'prior_activation_names_sha256',
    'force_reason_sha256', 'allow_parallel_requested', 'constraint_profile',
    'constraint_stop_on_complete', 'constraint_completed')


def _hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=True,
        separators=(',', ':'), allow_nan=False).encode('ascii')).hexdigest()


def _name(value):
    return type(value) is str and 0 < len(value) <= 512


def start_record(request_id, phase, *, tool_choice, activation_names,
                 force_reason, constraint, allow_parallel):
    """A detached snapshot of actual serving inputs and configured scalars."""
    record = dict(schema=SCHEMA, scope=SCOPE, available=False)
    if type(request_id) is not str or re.fullmatch(r'resp_[0-9a-f]{24}', request_id) is None:
        return {**record, 'reason': 'invalid-server-request-id'}
    record['request_id'] = request_id
    if type(phase) is not str or phase not in ('gateway_decision', 'gateway_execution'):
        return {**record, 'reason': 'invalid-phase'}
    record['phase'] = phase
    if (type(tool_choice) is not str or len(tool_choice) > 521
            or (tool_choice not in ('none', 'auto', 'required')
                and not (tool_choice.startswith('specific:') and _name(tool_choice[9:])))):
        return {**record, 'reason': 'invalid-tool-choice'}
    if (type(activation_names) not in (tuple, list) or len(activation_names) > 64
            or not all(_name(name) for name in activation_names)):
        return {**record, 'reason': 'invalid-activation'}
    if type(allow_parallel) is not bool or not (force_reason is None or _name(force_reason)):
        return {**record, 'reason': 'invalid-controls'}
    if constraint is None:
        profile, stop, completed = 'none', None, None
    else:
        # vars() avoids invoking arbitrary properties or matcher/GPU methods.
        # Never copy or retain the dictionary containing its tensor/matcher.
        try:
            state = vars(constraint)
            profile = state.get('profile')
            stop = state.get('stop_on_complete')
            completed = state.get('completed')
        except (TypeError, AttributeError):
            return {**record, 'reason': 'constraint-scalars-unavailable'}
        if (type(profile) is not str
                or profile not in ('required_tool', 'auto_tool_schema', 'json', 'json_schema')
                or type(stop) is not bool or type(completed) is not bool):
            return {**record, 'reason': 'constraint-scalars-unavailable'}
    return {**record, 'available': True,
        'tool_choice_kind': 'specific' if tool_choice.startswith('specific:') else tool_choice,
        'tool_choice_sha256': _hash(tool_choice),
        'prior_activation_count': len(activation_names),
        'prior_activation_names_sha256': _hash(list(activation_names)),
        'force_reason_sha256': None if force_reason is None else _hash(force_reason),
        'allow_parallel_requested': allow_parallel,
        'constraint_profile': profile, 'constraint_stop_on_complete': stop,
        'constraint_completed': completed}


def _valid_record(value):
    if type(value) is not dict or value.get('schema') != SCHEMA or value.get('scope') != SCOPE:
        return False
    if value.get('available') is not True or set(value) != {
            'schema', 'scope', 'available', 'request_id', *CONTROL_FIELDS}:
        return False
    if (type(value['request_id']) is not str
            or re.fullmatch(r'resp_[0-9a-f]{24}', value['request_id']) is None
            or type(value['phase']) is not str
            or value['phase'] not in ('gateway_decision', 'gateway_execution')
            or type(value['tool_choice_kind']) is not str
            or value['tool_choice_kind'] not in ('none', 'auto', 'required', 'specific')
            or type(value['prior_activation_count']) is not int
            or not 0 <= value['prior_activation_count'] <= 64
            or type(value['allow_parallel_requested']) is not bool):
        return False
    for key in ('tool_choice_sha256', 'prior_activation_names_sha256', 'force_reason_sha256'):
        if key == 'force_reason_sha256' and value[key] is None:
            continue
        if type(value[key]) is not str or re.fullmatch(r'[0-9a-f]{64}', value[key]) is None:
            return False
    if (value['tool_choice_kind'] != 'specific'
            and value['tool_choice_sha256'] != _hash(value['tool_choice_kind'])):
        return False
    if value['prior_activation_count'] == 0 and value['prior_activation_names_sha256'] != _hash([]):
        return False
    profile = value['constraint_profile']
    if type(profile) is not str:
        return False
    if profile == 'none':
        return value['constraint_stop_on_complete'] is None and value['constraint_completed'] is None
    return (profile in ('required_tool', 'auto_tool_schema', 'json', 'json_schema')
        and type(value['constraint_stop_on_complete']) is bool
        and type(value['constraint_completed']) is bool)


def compare_observed_controls(left, right):
    """Fail closed on missing evidence; a match never proves full equivalence."""
    result = dict(available=False, observed_controls_match=None,
        full_contract_equivalence_proven=False, scope=SCOPE)
    if not _valid_record(left) or not _valid_record(right):
        return result
    differences = [key for key in CONTROL_FIELDS if left[key] != right[key]]
    return {**result, 'available': True, 'observed_controls_match': not differences,
        'differing_fields': differences}


def emit_start(request_id, phase, **controls):
    """Flush before generation; unavailable/broken logs cannot change serving."""
    if os.environ.get('VMODEL_GENERATION_WITNESS', '0').strip() != '1':
        return
    try:
        encoded = json.dumps(start_record(request_id, phase, **controls),
            sort_keys=True, allow_nan=False)
        if len(encoded) > 4096:
            encoded = json.dumps(dict(schema=SCHEMA, scope=SCOPE,
                available=False, reason='record-limit'))
        print(PREFIX + encoded, flush=True)
    except Exception:
        pass
