import copy

import pytest

from tests.fixtures.plex_agent_profile import SYNTHETIC_PAGES
from tests.fixtures.plex_finite_media import respond


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
