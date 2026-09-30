import json

import pytest

from runtime.toolcalls import parse_tool_calls


def call(value):
    return '<tool_call>'+json.dumps(dict(name='write_file',arguments=dict(content=value)))+'</tool_call>'


@pytest.mark.parametrize('value',['literal </tool_call>','code } </tool_call> inside data',
    'quoted " and escaped \\ } </tool_call>',{'nested':['} </tool_call>','<tool_call>']},
    'Unicode café 雪 } </tool_call>'])
def test_literal_markers_inside_valid_json_preserve_exact_argument_data(value):
    text='Before '+call(value)+' After'
    content,calls=parse_tool_calls(text,'qwen3_5',allowed_names=['write_file'])
    assert content=='Before  After' and len(calls)==1
    assert json.loads(calls[0]['function']['arguments'])=={'content':value}


@pytest.mark.parametrize('bad',['<tool_call>not JSON</tool_call>',
    '<tool_call>{"name":"write_file","arguments":{"x":1,"x":2}}</tool_call>',
    '<tool_call>{"name":"write_file","arguments":{"x":NaN}}</tool_call>',
    '<tool_call>{"name":"write_file","arguments":{}} trailing invalid</tool_call>',
    '<tool_call>broken '+call('must not execute nested')+'</tool_call>',
    '<tool_call>broken '+call('first nested')+' '+call('second nested')+'</tool_call>'])
def test_invalid_blocks_remain_visible_and_are_never_repaired_or_nested_executed(bad):
    assert parse_tool_calls(bad,'qwen3_5',allowed_names=['write_file'])==(bad,[])
    content,calls=parse_tool_calls(bad+' '+call('valid later'),'qwen3_5',allowed_names=['write_file'])
    assert content==bad+' ' and len(calls)==1
    assert json.loads(calls[0]['function']['arguments'])=={'content':'valid later'}


def test_unterminated_quoted_block_does_not_salvage_embedded_calls():
    bad='<tool_call>{"unfinished":"'+call('not an independent call')
    assert parse_tool_calls(bad,'qwen3_5')==(bad,[])


def test_parser_does_not_hide_or_repair_extra_closing_marker():
    content,calls=parse_tool_calls(call('x')+'\n</tool_call>','qwen3_5')
    assert content=='\n</tool_call>' and len(calls)==1


def test_marker_profile_explicit_and_invalid_values_fail_closed(monkeypatch):
    from runtime.profiles import apply_runtime_profiles
    from runtime.structured import GrammarConstraint
    env={};apply_runtime_profiles(['tool-strict-markers-audit'],environ=env)
    assert env=={'VMODEL_TOOL_STRICT_MARKERS':'1'}
    for bad in ('auto','true','2',''):
        monkeypatch.setenv('VMODEL_TOOL_STRICT_MARKERS',bad)
        with pytest.raises(ValueError,match='VMODEL_TOOL_STRICT_MARKERS'):
            GrammarConstraint.tools(None,[],required=False)


def test_valid_call_with_visible_protocol_leak_fails_receipt_gate():
    from tests.fixtures.captured_transition_tracking_gate import row_checks
    from tests.test_captured_transition_tracking_gate_pure import valid_row,message,weather
    response=dict(output=[weather(),message('\n</tool_call>')])
    checks=row_checks(valid_row(),response,dict(kind='weather_tool',city='Tokyo'),
        dict(profiles=['test'],profile_digest='digest'))
    assert checks['one_requested_call'] and checks['exact_arguments']
    assert not checks['no_visible_protocol_marker_leak']


def test_completion_guidance_is_explicit_preserves_catalog_and_parallel_calls(monkeypatch):
    from runtime.server import _tools_system_preamble
    tools=[dict(type='function',name='inspect',parameters=dict(type='object',properties={}))]
    monkeypatch.delenv('VMODEL_HERMES_COMPLETION_GUIDANCE',raising=False)
    original=_tools_system_preamble(tools,compact_json=True)
    monkeypatch.setenv('VMODEL_HERMES_COMPLETION_GUIDANCE','0')
    assert _tools_system_preamble(tools,compact_json=True)==original
    monkeypatch.setenv('VMODEL_HERMES_COMPLETION_GUIDANCE','1')
    candidate=_tools_system_preamble(tools,compact_json=True)
    assert candidate.split('<tools>')[1]==original.split('<tools>')[1]
    assert 'no suffix' in candidate and 'Multiple independent tool calls' in candidate
    for bad in ('true','auto','2',''):
        monkeypatch.setenv('VMODEL_HERMES_COMPLETION_GUIDANCE',bad)
        with pytest.raises(ValueError,match='VMODEL_HERMES_COMPLETION_GUIDANCE'):
            _tools_system_preamble(tools,compact_json=True)


def test_complete_turn_profile_explicit_and_invalid_values_rejected(monkeypatch):
    from runtime.profiles import apply_runtime_profiles
    from runtime.structured import GrammarConstraint
    env={};apply_runtime_profiles(['tool-complete-turn-audit'],environ=env)
    assert env=={'VMODEL_TOOL_COMPLETE_TURN':'1'}
    for bad in ('auto','true','2',''):
        monkeypatch.setenv('VMODEL_TOOL_COMPLETE_TURN',bad)
        with pytest.raises(ValueError,match='VMODEL_TOOL_COMPLETE_TURN'):
            GrammarConstraint.tools(None,[],required=False)


def test_serial_turns_require_explicit_complete_turn_and_valid_flag(monkeypatch):
    from runtime.structured import GrammarConstraint
    monkeypatch.delenv('VMODEL_TOOL_COMPLETE_TURN',raising=False)
    for bad in ('auto','true','2','', '1'):
        monkeypatch.setenv('VMODEL_TOOL_SERIAL_TURNS',bad)
        with pytest.raises(ValueError,match='VMODEL_TOOL_SERIAL_TURNS'):
            GrammarConstraint.tools(None,[],required=False)
