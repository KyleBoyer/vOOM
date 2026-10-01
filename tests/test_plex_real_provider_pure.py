"""CPU-only pinned provider tests; no models, credentials or network clients."""
from copy import deepcopy
import shutil

import pytest

from tests.fixtures import plex_real_provider as provider
from tests.fixtures.plex_agent_profile import SYNTHETIC_PAGES

pytestmark = pytest.mark.skipif(
    not provider.SOURCE.is_file() or not shutil.which('node'),
    reason='pinned sibling provider and Node required; no substitute oracle')


def call(**changes):
    args = dict(mediaType='all', ratingOperator='lte', movieRatingValue='PG-13',
                showRatingValue='TV-Y7', excludeRootFolderPath='/Kids/',
                sortBy='title', sortDirection='asc')
    args.update(changes)
    return dict(name='plugin__plex__plex_list_library', arguments=args)


def test_actual_filtered_single_page_envelope_and_identity():
    before = deepcopy(SYNTHETIC_PAGES)
    page = provider.respond(call(), SYNTHETIC_PAGES)
    assert set(page) == {'movies', 'movieTotal', 'movieReturned', 'movieHasMore',
                         'series', 'seriesTotal', 'seriesReturned', 'seriesHasMore',
                         'limit', 'offset'}
    assert page['limit'] == 50 and page['offset'] == 0
    assert [r['title'] for r in page['movies']] == ['ALPHA_G', 'BRAVO_PG13']
    assert [r['title'] for r in page['series']] == ['CHARLIE_TVY', 'DELTA_TVY7']
    assert all('plexLibrarySectionName' not in r for r in page['movies'] + page['series'])
    assert provider.coverage([page], SYNTHETIC_PAGES)['passed']
    assert SYNTHETIC_PAGES == before
    assert provider.provenance()['provider_commit'] == provider.COMMIT


def test_requested_limit_one_really_pages_and_incomplete_fails():
    pages = [provider.respond(call(limit=1, offset=n), SYNTHETIC_PAGES) for n in (0, 1)]
    assert pages[0]['movieHasMore'] and not pages[1]['movieHasMore']
    assert not provider.coverage(pages[:1], SYNTHETIC_PAGES)['passed']
    assert provider.coverage(pages, SYNTHETIC_PAGES)['passed']
    assert not provider.coverage(pages[::-1], SYNTHETIC_PAGES)['passed']
    assert not provider.coverage([pages[0], pages[0], pages[1]], SYNTHETIC_PAGES)['passed']


def test_true_default_limit_fifty_split_with_asymmetric_exhaustion():
    source = [dict(movies=[dict(title=f'M{i:03}', contentRating='PG', rootFolderPath='/Movies')
                           for i in range(53)],
                   series=[dict(title='Show', contentRating='TV-Y', rootFolderPath='/TV')])]
    pages = [provider.respond(call(offset=n), source) for n in (0, 50)]
    assert pages[0]['movieReturned'] == 50 and pages[0]['movieHasMore']
    assert pages[1]['movieReturned'] == 3 and not pages[1]['movieHasMore']
    assert pages[1]['series'] == [] and not pages[1]['seriesHasMore']
    assert provider.coverage(pages, source)['passed']


def test_wrong_restrictive_query_cannot_certify_coverage():
    page = provider.respond(call(query='ALPHA'), SYNTHETIC_PAGES)
    assert len(page['movies']) == 1
    assert not provider.coverage([page], SYNTHETIC_PAGES)['passed']


def test_rating_case_unknown_upper_boundary_and_root_case():
    source = [dict(movies=[dict(title=name, contentRating=rating, rootFolderPath=root)
        for name, rating, root in [('yes', 'pg 13', '/Movies'), ('upper', 'R', '/Movies'),
                                  ('unknown', 'UNRATED', '/Movies'), ('kid', 'G', '/m/KIDS/Movies')]],
        series=[dict(title=name, contentRating=rating, rootFolderPath='/TV')
                for name, rating in [('yesTV', 'tv-y7'), ('upperTV', 'TV-Y7-FV'), ('unknownTV', None)]])]
    page = provider.respond(call(), source)
    assert [r['title'] for r in page['movies']] == ['yes']
    assert [r['title'] for r in page['series']] == ['yesTV']
    assert provider.coverage([page], source)['passed']


@pytest.mark.parametrize('mutation', ['missing', 'unexpected', 'duplicate', 'totals', 'exhaustion'])
def test_coverage_fails_closed_on_corrupted_receipts(mutation):
    page = provider.respond(call(), SYNTHETIC_PAGES)
    if mutation == 'missing': page['movies'].pop()
    if mutation == 'unexpected': page['movies'][0]['title'] = 'INVENTED'
    if mutation == 'duplicate': page['movies'][1] = deepcopy(page['movies'][0])
    if mutation == 'totals': page['movieTotal'] += 1
    if mutation == 'exhaustion': page['movieHasMore'] = True
    assert not provider.coverage([page], SYNTHETIC_PAGES)['passed']


def test_forbidden_client_methods_and_unknown_tool_fail():
    with pytest.raises(RuntimeError, match='Forbidden client access'):
        provider.respond(call(qualityProfileName='anything'), SYNTHETIC_PAGES)
    with pytest.raises(ValueError, match='unsupported'):
        provider.respond(dict(name='delete_all', arguments={}), SYNTHETIC_PAGES)


def test_provider_identity_mismatch_fails_before_execution(monkeypatch):
    monkeypatch.setattr(provider, 'SHA256', '0' * 64)
    with pytest.raises(ValueError, match='identity mismatch'):
        provider.respond(call(), SYNTHETIC_PAGES)


def test_provider_contract_pass_does_not_relabel_legacy_pagination_failure(monkeypatch):
    import json
    from tests.fixtures import plex_agent_profile as plex
    args=call(limit=50,offset=0)['arguments']
    first=dict(status='completed',output=[
        dict(type='message',role='assistant',content=[dict(type='output_text',text='Checking now.')]),
        dict(type='function_call',call_id='actual',name=plex.PLEX_TOOL,arguments=json.dumps(args))])
    last=dict(status='completed',output=[dict(type='message',role='assistant',
        content=[dict(type='output_text',text=', '.join(plex.ELIGIBLE_TITLES))])])
    replies=iter([first,last]);requests=[]
    def post(url,request,timeout):
        requests.append(deepcopy(request));return next(replies),1.0
    monkeypatch.setattr(plex,'_post',post)
    result=plex.run_profile(dict(model='test',input=[],tools=[]),'unused',10,3,
        tool_result_profile=provider.PROFILE)
    assert result['provider_contract']['passed']
    assert result['rubric']['score']==90 and not result['rubric']['passed']
    assert not result['passed']
    assert result['catalog_coverage']['expected_rows']==4
    assert requests[1]['input'][:2]==first['output']
    assert 'filtersApplied' not in json.loads(requests[1]['input'][-1]['output'])
