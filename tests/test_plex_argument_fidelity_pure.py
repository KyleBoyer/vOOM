import copy
import pytest
from tests.fixtures.plex_argument_fidelity import check


def call(**args):
    return dict(name='plugin__plex__plex_list_library',arguments=args)


@pytest.mark.parametrize('args',[{},dict(excludeRootFolderPath='/Kids/'),dict(
    mediaType='all',ratingOperator='lte',movieRatingValue='PG-13',showRatingValue='TV-Y7',
    excludeRootFolderPath='/Kids/',offset=3,limit=200)])
def test_exact_or_unfiltered_retrieval_allowed_without_repair(args):
    calls=[call(**args)];before=copy.deepcopy(calls)
    assert check(calls)['passed'] and calls==before


@pytest.mark.parametrize('args',[dict(excludePlexLibrarySectionName='Kids'),
    dict(sectionName='Adult'),dict(excludeRootFolderPath='Kids'),dict(rootFolderPath='/Media/'),
    dict(ratingOperator='lt',movieRatingValue='PG-13',showRatingValue='TV-Y7'),
    dict(ratingOperator='lte',ratingValue='PG-13'),dict(monitored=False),dict(year=2000)])
def test_proxy_or_extra_narrowing_cannot_certify_capture(args):
    assert not check([call(**args)])['passed']


def test_terminal_answer_has_no_new_arguments_to_validate():
    assert check([])['passed']


def test_unrelated_domain_cases_counterbalance_paths_and_labels():
    import json
    from tests.fixtures.gateway_predicate_cases import build
    cases=build()
    assert cases[0][0]['arguments']['excludePath']=='/Archive/'
    assert cases[1][0]['arguments']['excludeLabel']=='Archive'
    final=cases[2][1]
    rows=json.loads(final['input'][-1]['output'])['assets']
    expected=[r['id'] for r in rows if '/Archive/' not in r['folderPath']]
    proxy=[r['id'] for r in rows if r['collectionLabel']!='Archive']
    assert expected==cases[2][0]['expected_json']['ids'] and expected!=proxy
    assert all(request['temperature']==1 for _,request in cases)
