import ast
from pathlib import Path
from types import SimpleNamespace as NS
import io
import json

import pytest

from runtime.server import Handler, _HiddenDecisionStream, RequestValidationError, _parse_request_tool_calls, _hidden_tool_enable_pair, _private_decode_keepalive
from runtime.profiles import apply_runtime_profiles


def actual_stream_assignment(flag, callback):
    """Execute the production selection block with a fake environment only."""
    tree=ast.parse(Path('runtime/server.py').read_text())
    fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='run_hidden_gateway')
    branch=next(n for n in ast.walk(fn) if isinstance(n,ast.If)
        and ast.unparse(n.test)=='host_action is not None')
    selected=[]
    for node in branch.orelse:
        selected.append(node)
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='decision_stream' for t in node.targets):
            break
    env=dict(os=NS(environ={'VMODEL_FAST_TOOL_GATEWAY_BUFFER_DECISION':flag}),
        engine=NS(cfg=NS(model_type='qwen3')),on_token=callback,
        _HiddenDecisionStream=_HiddenDecisionStream,RequestValidationError=RequestValidationError)
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[])),
                 '<actual decision selection>', 'exec'),env)
    return env['decision_stream']


def test_buffer_matches_nonstream_and_default_retains_streaming():
    assert actual_stream_assignment('1',lambda text:pytest.fail('unexpected emission')) is None
    assert actual_stream_assignment('0',None) is None
    assert isinstance(actual_stream_assignment('0',lambda text:None),_HiddenDecisionStream)
    for bad in ('auto','',1,True,None):
        with pytest.raises(RequestValidationError):actual_stream_assignment(bad,None)


def test_actual_parser_retains_valid_private_action_after_prose():
    tool,_ = _hidden_tool_enable_pair()
    text='I will inspect the returned records.\n\n<tool_call>\n{"name":"vmodel_enable_tools","arguments":{}}\n</tool_call>'
    content,calls=_parse_request_tool_calls(text,[tool],'qwen3',allow_parallel=False)
    assert content.strip()=='I will inspect the returned records.'
    assert len(calls)==1 and calls[0]['function']['name']=='vmodel_enable_tools'
    # Buffered streaming supplies no decision stream, so it cannot trigger the
    # production late-action suppression condition.
    decision_stream=actual_stream_assignment('1',lambda text:pytest.fail('private leak'))
    assert not (calls[0] is not None and decision_stream is not None
                and decision_stream.branch=='direct')


def test_profile_only_adds_buffer_policy():
    before={};after={};base=['huihui-qwen38-27b-workflow-head-rows-audit']
    apply_runtime_profiles(base,environ=before)
    apply_runtime_profiles(base+['gateway-buffered-decision'],environ=after)
    assert after=={**before,'VMODEL_FAST_TOOL_GATEWAY_BUFFER_DECISION':'1'}


def test_deep_chain_audit_changes_only_depth():
    before={};after={};base=['huihui-qwen38-27b-workflow-head-rows-audit']
    apply_runtime_profiles(base,environ=before)
    apply_runtime_profiles(base+['qwen35-mtp-depth7-audit'],environ=after)
    assert before['VMODEL_QWEN_MTP_DEPTH']=='4'
    assert after=={**before,'VMODEL_QWEN_MTP_DEPTH':'7'}


@pytest.mark.parametrize('progress_events', [False, True])
def test_buffered_direct_answer_is_flushed_once_by_real_sse_writer(monkeypatch, progress_events):
    monkeypatch.setenv('VMODEL_FAST_TOOL_GATEWAY_BUFFER_DECISION', '1')
    handler=Handler.__new__(Handler)
    handler.wfile=io.BytesIO()
    handler.send_response=lambda *a:None
    handler.send_header=lambda *a:None
    handler.end_headers=lambda:None
    handler._sampling=NS()
    handler._constraint=None
    engine=NS(cfg=NS(model_type='qwen3'))
    text='All requested records have been checked.'
    def build(body,*args):
        return dict(status='completed',output=[dict(id='msg_test',type='message',
            role='assistant',status='completed',content=[dict(type='output_text',text=body,annotations=[])])])
    def generate(_emit,_progress):
        # The private direct-answer decision was buffered: no token callback.
        tick = _private_decode_keepalive(_progress)
        tick('PRIVATE PLANNING MUST NOT LEAK')
        tick('<tool_call>PRIVATE ACTION</tool_call>')
        return dict(text=text,tokens=[1,2],prompt_tokens=3,path_stats={})
    tool,_=_hidden_tool_enable_pair()
    # Hidden gateway decisions exist only with tools; the ordinary tool-free
    # generator keeps its normal token callback and is not buffered by this flag.
    handler._stream_responses('prompt',1024,[],engine,[tool],
        build,'resp_test','model',1,None,1,None,[], 'msg_test','auto',False,
        generate_fn=generate, progress_events=progress_events)
    wire=handler.wfile.getvalue().decode()
    assert wire.count(': private_decode\n\n') == 2
    assert 'PRIVATE' not in wire
    events=[json.loads(line[6:]) for line in wire.splitlines()
            if line.startswith('data: {')]
    assert ''.join(e['delta'] for e in events if e['type']=='response.output_text.delta')==text
    assert sum(e['type']=='response.completed' for e in events)==1


def test_private_keepalive_disabled_and_disconnect_propagates(monkeypatch):
    monkeypatch.delenv('VMODEL_FAST_TOOL_GATEWAY_BUFFER_DECISION', raising=False)
    assert _private_decode_keepalive(lambda _:pytest.fail('disabled')) is None
    monkeypatch.setenv('VMODEL_FAST_TOOL_GATEWAY_BUFFER_DECISION', '1')
    assert _private_decode_keepalive(None) is None
    def disconnected(_):
        raise BrokenPipeError('closed')
    with pytest.raises(BrokenPipeError):
        _private_decode_keepalive(disconnected)('never public')


def test_actual_hidden_generation_calls_wire_private_progress():
    tree=ast.parse(Path('runtime/server.py').read_text())
    fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='run_hidden_gateway')
    calls=[n for n in ast.walk(fn) if isinstance(n,ast.Call)
           and isinstance(n.func,ast.Name) and n.func.id=='_engine_generate']
    callbacks=[ast.unparse(k.value) for n in calls for k in n.keywords if k.arg=='on_token']
    assert sum('_private_decode_keepalive(on_progress)' in c for c in callbacks)==2
