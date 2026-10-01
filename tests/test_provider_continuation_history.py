import copy
import json
import pytest
from tests.fixtures.plex_agent_profile import _append_full_response_and_result, response_calls


def test_provider_continuation_retains_actual_prose_and_call_without_repair():
    response=dict(output=[dict(type='message',role='assistant',content=[dict(type='output_text',text='Model-authored preface')]),
        dict(type='function_call',call_id='actual-id',name='list',arguments='{"offset": 0}')])
    before=copy.deepcopy(response); request=dict(input=[dict(role='user',content='original')])
    page=dict(movies=[],movieHasMore=False)
    _append_full_response_and_result(request,response,response_calls(response)[0],page)
    assert request['input'][1:-1]==before['output']
    assert request['input'][-1]['call_id']=='actual-id'
    assert json.loads(request['input'][-1]['output'])==page and response==before
    request['input'][1]['content'][0]['text']='edited request copy'
    assert response==before


def test_provider_continuation_rejects_identity_mismatch_before_mutation():
    request=dict(input=[])
    with pytest.raises(ValueError,match='identity'):
        _append_full_response_and_result(request,dict(output=[]),dict(call_id='bad'),{})
    assert request==dict(input=[])
