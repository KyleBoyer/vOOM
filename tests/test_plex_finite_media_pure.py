import copy
import json

import pytest

from tests.fixtures.plex_agent_profile import SYNTHETIC_PAGES
from tests.fixtures.plex_finite_media import respond, PAGE5_PROFILE, PLEX_PROFILE


@pytest.mark.parametrize('endpoint,step', [('plugin__plex__plex_list_library',3),
    ('plugin__plex__plex_list_library_media',5)])
def test_coverage_rejects_skipped_and_repeated_pages(endpoint, step):
    from tests.fixtures.plex_finite_media import coverage
    def page(offset):
        return respond(dict(name=endpoint, arguments=dict(offset=offset, limit=100)),
            SYNTHETIC_PAGES, profile=PLEX_PROFILE)
    assert not coverage([page(0), page(0)], SYNTHETIC_PAGES)['passed']
    assert not coverage([page(0), page(100)], SYNTHETIC_PAGES)['passed']
    assert coverage([page(0), page(step)], SYNTHETIC_PAGES)['passed']


def call(**args):
    return dict(name='plugin__plex__plex_list_library_media', arguments=args)


@pytest.mark.parametrize('media', ['all', 'movie', 'show'])
def test_complete_pages_are_finite_exact_and_query_isolated(media):
    source = copy.deepcopy(SYNTHETIC_PAGES)
    whole = respond(call(mediaType=media, limit=500), source)
    recovered = []
    for offset in range(0, whole['total'] + 2, 2):
        page = respond(call(mediaType=media, limit=2, offset=offset), source)
        assert page['returned'] == len(page['media'])
        assert page['hasMore'] == (offset + page['returned'] < whole['total'])
        assert page['media'] == whole['media'][offset:offset + 2]
        recovered.extend(page['media'])
        # An interleaved query does not advance a shared page queue.
        respond(call(mediaType='show' if media == 'movie' else 'movie'), source)
    assert recovered == whole['media']
    assert not respond(call(mediaType=media, offset=1000), source)['media']
    assert source == SYNTHETIC_PAGES
    assert whole['filtersApplied'] is False
    assert all('rootFolderPath' in row for row in whole['media'])


@pytest.mark.parametrize('args', [dict(limit=True), dict(limit=0), dict(offset=-1),
    dict(offset=0.5), dict(limit='2'), dict(mediaType='audio'), dict(query='ALPHA')])
def test_invalid_or_unsupported_requests_fail_visibly(args):
    with pytest.raises(ValueError):
        respond(call(**args), SYNTHETIC_PAGES)


def test_page_limit_clamped_and_raw_ratings_not_filtered_by_mock():
    page = respond(call(limit=999, ratingOperator='lte', movieRatingValue='PG-13'), SYNTHETIC_PAGES)
    assert page['limit'] == 500
    assert any(row['contentRating'] == 'R' for row in page['media'])


def test_explicit_server_cap_requires_real_pagination_without_changing_records():
    first=respond(call(mediaType='all',limit=500,offset=0),SYNTHETIC_PAGES,profile=PAGE5_PROFILE)
    second=respond(call(mediaType='all',limit=500,offset=first['returned']),SYNTHETIC_PAGES,profile=PAGE5_PROFILE)
    assert first['limit']==second['limit']==5
    assert first['hasMore'] is True and second['hasMore'] is False
    assert first['returned']==second['returned']==5
    whole=respond(call(mediaType='all',limit=500),SYNTHETIC_PAGES)
    assert first['media']+second['media']==whole['media']
    assert first['total']==second['total']==10
    assert not respond(call(offset=10),SYNTHETIC_PAGES,profile=PAGE5_PROFILE)['media']
    with pytest.raises(ValueError):
        respond(call(),SYNTHETIC_PAGES,profile='invented')


@pytest.mark.parametrize('kind', ['all', 'movie', 'show'])
def test_dual_endpoint_library_has_exact_finite_independent_streams(kind):
    source=copy.deepcopy(SYNTHETIC_PAGES)
    def request(offset):
        return dict(name='plugin__plex__plex_list_library',
            arguments=dict(mediaType=kind,limit=100,offset=offset,
                excludePlexLibrarySectionName='Kids',movieRatingValue='PG-13'))
    recovered={'movies':[], 'series':[]}
    for offset in (0,3,6):
        page=respond(request(offset),source,profile=PLEX_PROFILE)
        assert page['limit']==3 and page['offset']==offset
        assert page['filtersApplied'] is False
        for media_type,key,prefix in [('movie','movies','movie'),('show','series','series')]:
            expected=sorted([r for p in source for r in p[key]],key=lambda r:r['title']) if kind in ('all',media_type) else []
            assert page[key]==expected[offset:offset+3]
            assert page[prefix+'Total']==len(expected)
            assert page[prefix+'Returned']==len(page[key])
            assert page[prefix+'HasMore']==(offset+len(page[key])<len(expected))
            recovered[key].extend(page[key])
        respond(call(mediaType='all'),source,profile=PLEX_PROFILE)
    assert source==SYNTHETIC_PAGES
    if kind!='show':
        # An erroneous section predicate is not repaired or silently applied.
        assert any(r['contentRating']=='R' for r in recovered['movies'])
        assert any('/Kids/' in r['rootFolderPath'] for r in recovered['movies'])


def test_old_profiles_still_reject_library_and_new_profile_rejects_mutation():
    library=dict(name='plugin__plex__plex_list_library',arguments={})
    with pytest.raises(ValueError):
        respond(library,SYNTHETIC_PAGES,profile=PAGE5_PROFILE)
    with pytest.raises(ValueError):
        respond(dict(name='plugin__plex__plex_move_media',arguments={}),
                SYNTHETIC_PAGES,profile=PLEX_PROFILE)
    assert respond(call(),SYNTHETIC_PAGES,profile=PLEX_PROFILE)==respond(
        call(),SYNTHETIC_PAGES,profile=PAGE5_PROFILE)


def test_real_workflow_appends_library_results_and_keeps_grader(monkeypatch):
    from tests.fixtures import plex_agent_profile as fixture
    responses=[{'status':'completed','output':[dict(type='function_call',
        name=fixture.PLEX_TOOL,call_id='page'+str(offset),arguments=json.dumps(dict(
            mediaType='all',offset=offset,limit=3,excludeRootFolderPath='/Kids/',
            ratingOperator='lte',movieRatingValue='PG-13',showRatingValue='TV-Y7')))]}
        for offset in (0,3)]
    responses.append({'status':'completed','output':[dict(type='message',content=[
        dict(type='output_text',text='ALPHA_G BRAVO_PG13 CHARLIE_TVY DELTA_TVY7')])]})
    seen=[]
    def post(url,request,timeout):
        seen.append(copy.deepcopy(request));return responses[len(seen)-1],1.0
    monkeypatch.setattr(fixture,'_post',post)
    monkeypatch.setattr(fixture,'_pressure',lambda:{})
    result=fixture.run_profile(dict(model='fake',input=[],tools=[]),'unused',1,4,
                              tool_result_profile=PLEX_PROFILE)
    assert result['passed'] and result['rubric']['score']==100
    assert [t['handled_call_count'] for t in result['turns']]==[1,1,0]
    first=json.loads(seen[1]['input'][-1]['output'])
    second=json.loads(seen[2]['input'][-1]['output'])
    assert first['movieHasMore'] and first['seriesHasMore']
    assert not second['movieHasMore'] and not second['seriesHasMore']
    assert len(first['movies'])==3 and len(second['movies'])==2
