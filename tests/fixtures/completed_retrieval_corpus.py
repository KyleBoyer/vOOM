"""Seeded, answer-hidden synthetic retrieval cases and strict completion scoring.

This is a benchmark fixture, never a runtime fast path. Expected values are
checker-only: callers send only ``user_text``, not the fixture metadata/object.
No model is loaded, and no response is repaired or rendered by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random


DOMAINS = {
    'library': ('accession', 'shelf_code', 'Library catalog entry: ordinary reference volume, no special handling.'),
    'shipping': ('shipment', 'depot_code', 'Shipment ledger entry: routine package, standard handling and tracking.'),
    'builds': ('build', 'artifact_code', 'Build ledger entry: routine compilation, standard checks and packaging.'),
}


@dataclass(frozen=True)
class RetrievalCase:
    user_text: str
    local_user_tokens: int
    expected: dict[str, str]
    metadata: dict


def build_case(tokenizer, target_tokens, *, fixture_seed, domain):
    """Vary record IDs, answer codes, requested order, positions, and domain.

    Seeds are fixture identities, not inference seeds and never prompt content.
    Nominal depths refer to filler-token positions; actual token positions are
    measured after final assembly and recorded separately.
    """
    if type(fixture_seed) is not int or not 0 <= fixture_seed < 2**63:
        raise ValueError('fixture_seed must be an integer in 0..2**63-1')
    if domain not in DOMAINS:
        raise ValueError('unknown retrieval domain')
    if type(target_tokens) is not int or not 1024 <= target_tokens <= 1_000_000:
        raise ValueError('target_tokens must be an integer in 1024..1000000')
    key_field, value_field, filler_sentence = DOMAINS[domain]
    rng = random.Random(fixture_seed)
    identities = rng.sample(range(100_000, 1_000_000), 5)
    ids = [f'{key_field}-{n}' for n in identities]
    alphabet = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'
    values = []
    while len(values) < len(ids):
        value = ''.join(rng.choice(alphabet) for _ in range(12))
        if value not in values:
            values.append(value)
    records = [f'\nRecord {key_field}={key}; {value_field}={value}.\n'
               for key, value in zip(ids, values)]
    # The final query uses exact IDs, not TARGET markers near the answers.
    query_order = [0, 1] if rng.getrandbits(1) else [1, 0]
    expected = {ids[i]: values[i] for i in query_order}
    prefix = (f'Read this synthetic {domain} archive. Each Record line gives a '
              f'{key_field} and its {value_field}. Ordinary entries are filler.\n')
    suffix = (f'\nEnd of archive. Retrieve {value_field} for these exact {key_field} IDs: '
              + ', '.join(expected)
              + '. Return only one JSON object mapping each requested ID to its exact '
              'recorded code. Include both requested IDs and no others. Stop after the '
              'object. Do not invent a code or include explanations or Markdown.\n')
    fixed = len(tokenizer.encode(prefix + ''.join(records) + suffix).ids)
    if target_tokens <= fixed + 256:
        raise ValueError('target token count leaves insufficient filler')
    unit = tokenizer.encode(' ' + filler_sentence + '\n').ids
    if not unit:
        raise ValueError('tokenizer produced no filler tokens')
    count = target_tokens - fixed
    filler = (unit * (count // len(unit) + 1))[:count]
    # Two requested records lie in separate distant bands, with three decoys.
    depths = [rng.uniform(.10, .25), rng.uniform(.65, .80), .03, .46, .92]
    schedule = sorted((int(count * depth), i) for i, depth in enumerate(depths))
    chunks = [prefix]
    cursor = 0
    for position, index in schedule:
        chunks.extend((tokenizer.decode(filler[cursor:position], skip_special_tokens=False),
                       records[index]))
        cursor = position
    chunks.extend((tokenizer.decode(filler[cursor:], skip_special_tokens=False), suffix))
    text = ''.join(chunks)
    actual = len(tokenizer.encode(text).ids)
    # Detect accidental answer exposure or tokenizer-induced fixture corruption
    # before any server call. Every answer occurs once, in its own record only.
    for index in (0, 1):
        value = values[index]
        if text.count(value) != 1 or value in prefix or value in suffix or records[index] not in text:
            raise ValueError('answer isolation failed')
    if not .95 * target_tokens <= actual <= 1.05 * target_tokens:
        raise ValueError('assembled token count outside declared context tolerance')
    return RetrievalCase(text, actual, expected, {
        'schema': 'voom.completed-retrieval-fixture.v1',
        'fixture_seed': fixture_seed,
        'domain': domain,
        'target_user_tokens': target_tokens,
        'local_user_tokens': actual,
        'requested_ids': list(expected),
        'requested_record_token_positions': {
            ids[i]: len(tokenizer.encode(text[:text.index(records[i])]).ids) for i in (0, 1)},
        'nominal_filler_depths': {ids[i]: depths[i] for i in (0, 1)},
        'decoy_records': 3,
        'answers_in_suffix': False,
        'answer_occurrences_per_requested_record': 1,
        'user_text_sha256': hashlib.sha256(text.encode()).hexdigest(),
        'task': 'completed-synthetic-retrieval',
        'scope': 'Synthetic no-tool retrieval only; not captured traffic, Plex, broad intelligence or GLM DSA conformance.',
    })


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def score_response(response, expected, *, max_output_tokens):
    """Exact checker over actual completed message output, without repair.

    Output-key order/JSON whitespace are immaterial. Duplicate keys, extra keys,
    wrong types/case, Markdown/prose, tool calls, reasoning-only/top-level-only
    answers, protocol errors, and capped/cancelled responses all fail.
    """
    if (type(expected) is not dict or len(expected) != 2
            or not all(type(k) is str and type(v) is str and k and v for k, v in expected.items())):
        raise ValueError('two nonempty expected ID/code strings required')
    if type(max_output_tokens) is not int or max_output_tokens < 256:
        raise ValueError('completion gate requires output budget of at least256')
    response = response if type(response) is dict else {}
    output = response.get('output')
    valid_output = type(output) is list and bool(output)
    messages = [item for item in output if type(item) is dict and item.get('type') == 'message'] if valid_output else []
    valid_output = valid_output and all(type(item) is dict and item.get('type') in ('message', 'reasoning') for item in output)
    content = messages[0].get('content') if len(messages) == 1 else None
    valid_content = (type(content) is list and bool(content)
        and all(type(part) is dict and part.get('type') == 'output_text'
                and type(part.get('text')) is str for part in content))
    text = ''.join(part['text'] for part in content) if valid_content else ''
    parsed = None
    try:
        parsed = json.loads(text, object_pairs_hook=_unique_object)
    except (ValueError, TypeError):
        pass
    usage = response.get('usage')
    count = usage.get('output_tokens') if type(usage) is dict else None
    checks = {
        'naturally_completed': response.get('status') == 'completed' and not response.get('incomplete_details'),
        'no_protocol_error': not response.get('error'),
        'one_actual_message_no_tools': bool(valid_output and len(messages) == 1 and valid_content),
        'top_level_text_consistent': 'output_text' not in response or response['output_text'] == text,
        'positive_output_below_budget': type(count) is int and 0 < count < max_output_tokens,
        'exact_id_set': type(parsed) is dict and parsed.keys() == expected.keys(),
        'exact_recorded_values': type(parsed) is dict and parsed == expected,
    }
    return {'checks': checks, 'passed': all(checks.values()),
            'output_bytes': len(text.encode()),
            'output_sha256': hashlib.sha256(text.encode()).hexdigest()}
