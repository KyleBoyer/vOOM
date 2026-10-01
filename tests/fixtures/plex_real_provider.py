"""Pinned real provider logic over synthetic data; never a live Plex request.

Separate from the raw-result stress fixture and its unchanged legacy grader.
Coverage independently implements the captured user's eligibility contract,
not the model's requested filters. No provider result or model call is repaired.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PROVIDER = ROOT.parent / 'kai-plugin-plex'
SOURCE = PROVIDER / 'src/main/tools.ts'
RUNNER = Path(__file__).with_suffix('.mjs')
COMMIT = 'a0c33c70ebe419809aa184dee5ff9b24e58543eb'
SHA256 = 'b2a6d66db3dfe77e7ca1cc67e76f0e8620fd19ce053baf5bc80ad1a2cd58e35a'
PROFILE = 'synthetic-real-plex-provider-v1'
_KINDS = (('movie', 'movies', 'movie'), ('show', 'series', 'series'))


def provenance():
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    commit = subprocess.check_output(
        ['git', '-C', str(PROVIDER), 'rev-parse', 'HEAD'], text=True).strip()
    if digest != SHA256 or commit != COMMIT:
        raise ValueError('Pinned Plex provider identity mismatch')
    return dict(profile=PROFILE, provider_commit=commit, source_sha256=digest,
                source_path=str(SOURCE), live_tool_execution=False,
                scope='actual pinned provider function with in-memory clients only')


def _rows(source_pages, key):
    if not isinstance(source_pages, (list, tuple)) or not source_pages:
        raise ValueError('nonempty source pages required')
    rows = []
    for page in source_pages:
        if not isinstance(page, dict) or not isinstance(page.get(key), list):
            raise ValueError('invalid source catalog')
        if any(not isinstance(row, dict) for row in page[key]):
            raise ValueError('invalid source row')
        rows.extend(page[key])
    return rows


def respond(call, source_pages):
    provenance()
    if (not isinstance(call, dict)
            or call.get('name') != 'plugin__plex__plex_list_library'
            or not isinstance(call.get('arguments'), dict)):
        raise ValueError('unsupported or malformed provider call')
    node = shutil.which('node')
    if not node:
        raise RuntimeError('Node with native TypeScript stripping required')
    payload = dict(call=call, movies=_rows(source_pages, 'movies'),
                   series=_rows(source_pages, 'series'))
    result = subprocess.run([
        node, '--permission', '--allow-fs-read=' + str(SOURCE),
        '--allow-fs-read=' + str(RUNNER), str(RUNNER), str(SOURCE)],
        input=json.dumps(payload, allow_nan=False), text=True,
        capture_output=True, timeout=20, cwd=ROOT)
    if result.returncode:
        raise RuntimeError('Synthetic provider failed: ' + result.stderr[-3000:])
    response = json.loads(result.stdout)
    if not isinstance(response, dict):
        raise ValueError('invalid provider envelope')
    return response


def _rating(row):
    for field in ('contentRating', 'certification'):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _identity(kind, row):
    if not isinstance(row, dict) or not isinstance(row.get('title'), str):
        raise ValueError('row requires title')
    return (kind, row.get('id'), row['title'], row.get('rootFolderPath'), _rating(row))


def _eligible(kind, row):
    # Independent user predicate, not provider query/implementation reuse.
    rating = re.sub('[^A-Z0-9]', '', (_rating(row) or '').upper())
    allowed = {'G', 'PG', 'PG13'} if kind == 'movie' else {'TVY', 'TVY7', 'TV7'}
    root = row.get('rootFolderPath')
    return (rating in allowed and isinstance(root, str) and bool(root)
            and '/kids/' not in root.lower())


def coverage(pages, source_pages):
    """Exact eligible identities plus contiguous, exhausted per-type pages.

    This is a distinct provider-contract verdict, not the legacy Plex score.
    Both streams must be observed, including explicit empty/exhausted streams.
    """
    failures = []
    expected = {}
    observed = Counter()
    states = {}
    try:
        for kind, key, _ in _KINDS:
            rows = _rows(source_pages, key)
            identities = [_identity(kind, row) for row in rows]
            if len(set(identities)) != len(identities):
                failures.append('duplicate-source-' + kind)
            expected[kind] = {_identity(kind, row) for row in rows if _eligible(kind, row)}
            states[kind] = dict(next_offset=0, seen=False, exhausted=False)
        if not isinstance(pages, (list, tuple)) or not pages:
            raise ValueError('nonempty delivered pages required')
        for page in pages:
            if not isinstance(page, dict) or any(k.endswith('Error') or k == 'error' for k in page):
                raise ValueError('provider error/invalid page')
            offset, limit = page.get('offset'), page.get('limit')
            if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 200:
                raise ValueError('invalid pagination metadata')
            for kind, key, prefix in _KINDS:
                if key not in page:
                    continue
                rows = page[key]
                total, returned, more = (page.get(prefix + suffix)
                                          for suffix in ('Total', 'Returned', 'HasMore'))
                if (not isinstance(rows, list) or type(total) is not int
                        or total != len(expected[kind]) or type(returned) is not int
                        or returned != len(rows) or returned > limit or type(more) is not bool
                        or more != (offset + returned < total)):
                    raise ValueError('incorrect per-type totals or exhaustion')
                state = states[kind]
                # Mixed queries legitimately return an empty already-exhausted
                # stream while the other stream continues at a larger offset.
                if state['exhausted']:
                    if rows or more or offset < state['next_offset']:
                        raise ValueError('rows after exhaustion or offset rewind')
                elif offset != state['next_offset']:
                    raise ValueError('non-contiguous page offset')
                if more and returned != limit:
                    raise ValueError('short nonterminal provider page')
                observed.update(_identity(kind, row) for row in rows)
                state.update(seen=True, exhausted=not more,
                             next_offset=offset + returned)
        for kind, state in states.items():
            if not state['seen'] or not state['exhausted']:
                failures.append('missing-final-exhaustion-' + kind)
    except (ValueError, TypeError, KeyError) as error:
        failures.append(str(error))
    expected_all = set().union(*expected.values()) if expected else set()
    actual = set(observed)
    missing, unexpected = expected_all - actual, actual - expected_all
    duplicates = sum(count - 1 for count in observed.values() if count > 1)
    if missing: failures.append('missing-eligible-records')
    if unexpected: failures.append('unexpected-records')
    if duplicates: failures.append('duplicate-records')
    return dict(passed=not failures, failures=failures,
                expected_rows=len(expected_all), observed_rows=sum(observed.values()),
                missing_rows=len(missing), unexpected_rows=len(unexpected),
                duplicate_rows=duplicates,
                scope='synthetic provider-contract eligible coverage; not raw-catalog coverage or legacy rubric')
