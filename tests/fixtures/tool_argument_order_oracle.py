"""Actual installed CPU grammar and tokenizer; no target weights loaded."""
import copy
import itertools
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


@pytest.fixture(scope='module')
def engine():
    from tokenizers import Tokenizer
    from runtime.config import ModelConfig
    p=Path('models/Huihui-Qwen3.8-27B-abliterated')
    if not (p/'tokenizer.json').exists():pytest.skip('local tokenizer unavailable')
    return NS(_model_dir=p,cfg=ModelConfig.from_dir(p),tokenizer=Tokenizer.from_file(str(p/'tokenizer.json')))


def tool():
    return dict(type='function',name='list_assets',parameters=dict(type='object',
        properties={'excludePath':dict(type='string'),'maximum':dict(type='integer'),
                    'excludeLabel':dict(type='string')},required=['maximum'],additionalProperties=False))


@pytest.mark.parametrize('order',['0','1'])
def test_native_orphan_closer_exclusion_preserves_data_prose_parallel_and_eos(engine,monkeypatch,order):
    from runtime.structured import GrammarConstraint
    from runtime.toolcalls import parse_tool_calls
    monkeypatch.setenv('VMODEL_TOOL_ARGUMENT_ANY_ORDER',order)
    value='code } </tool_call> and <tool_call> inside data'
    args=dict(excludePath=value,maximum=2)
    payload='<tool_call>'+json.dumps(dict(name='list_assets',arguments=args))+'</tool_call>'
    for flag in ('0','1'):
        monkeypatch.setenv('VMODEL_TOOL_STRICT_MARKERS',flag)
        c=GrammarConstraint.tools(engine,[tool()],required=False,allow_parallel=True)
        assert not c.completed
        assert c.matcher.fork().accept_string('</tool_call>')==(flag=='0')
        assert c.matcher.accept_string('Before '+payload+' Between '+payload+' After')
        assert c.matcher.fork().accept_string('</tool_call>')==(flag=='0')
        assert c.matcher.accept_token(next(iter(engine.cfg.eos_token_ids)))
        assert c.is_terminated()
    content,calls=parse_tool_calls('Before '+payload+' After','qwen3_5',allowed_names=['list_assets'])
    assert content=='Before  After' and len(calls)==1
    assert json.loads(calls[0]['function']['arguments'])==args


@pytest.mark.parametrize('order',['0','1'])
def test_strict_closer_actual_token_mask_fork_and_rollback(engine,monkeypatch,order):
    from runtime.structured import GrammarConstraint
    from runtime.xgrammar_cpu import backend
    import numpy as np
    monkeypatch.setenv('VMODEL_TOOL_ARGUMENT_ANY_ORDER',order)
    monkeypatch.setenv('VMODEL_TOOL_STRICT_MARKERS','1')
    c=GrammarConstraint.tools(engine,[tool()],required=False,allow_parallel=True)
    text='<tool_call>'+json.dumps(dict(name='list_assets',arguments=dict(maximum=2)))+'</tool_call>'
    mask=backend().allocate_token_bitmask(1,engine.cfg.vocab_size)
    for token in engine.tokenizer.encode(text,add_special_tokens=False).ids:
        c.matcher.fill_next_token_bitmask(mask)
        assert (np.asarray(mask).view(np.uint32)[0,token//32] >> (token%32))&1
        assert c.matcher.fork().accept_token(token)
        assert c.matcher.accept_token(token)
        c.matcher.rollback(1);assert c.matcher.accept_token(token)
    probe=c.matcher.fork();blocked=False
    for token in engine.tokenizer.encode('</tool_call>',add_special_tokens=False).ids:
        probe.fill_next_token_bitmask(mask)
        if not (np.asarray(mask).view(np.uint32)[0,token//32] >> (token%32))&1:
            blocked=True;break
        assert probe.accept_token(token)
    assert blocked


def test_required_tools_keep_their_existing_termination_with_strict_markers(engine,monkeypatch):
    from runtime.structured import GrammarConstraint
    monkeypatch.setenv('VMODEL_TOOL_STRICT_MARKERS','1')
    c=GrammarConstraint.tools(engine,[tool()],required=True,allow_parallel=False)
    assert c.matcher.accept_string('<tool_call>'+json.dumps(dict(
        name='list_assets',arguments=dict(maximum=2)))+'</tool_call>')
    assert c.stop_on_complete and c.matcher.is_completed()


def test_both_released_eos_tokens_allowed_after_workspace_call(engine,monkeypatch):
    from runtime.structured import GrammarConstraint
    from runtime.xgrammar_cpu import backend
    import numpy as np
    monkeypatch.setenv('VMODEL_TOOL_STRICT_MARKERS','1')
    monkeypatch.setenv('VMODEL_TOOL_ARGUMENT_ANY_ORDER','1')
    tools=[dict(type='function',name='mastra_workspace_list_files',parameters=dict(
        type='object',properties=dict(path=dict(type='string')),required=['path']))]
    c=GrammarConstraint.tools(engine,tools,required=False,allow_parallel=True)
    assert c.matcher.accept_string('<tool_call>'+json.dumps(dict(
        name='mastra_workspace_list_files',arguments=dict(path='.')))+'</tool_call>\n')
    mask=backend().allocate_token_bitmask(1,engine.cfg.vocab_size)
    c.matcher.fill_next_token_bitmask(mask)
    for name in ('<|im_end|>','<|endoftext|>'):
        token=engine.tokenizer.token_to_id(name)
        assert token in engine.cfg.eos_token_ids
        assert (np.asarray(mask).view(np.uint32)[0,token//32] >> (token%32))&1
        branch=c.matcher.fork()
        assert branch.accept_token(token) and branch.is_terminated()


def test_actual_compiler_reproduces_and_fixes_out_of_order_field(engine,monkeypatch):
    from runtime.structured import GrammarConstraint
    for required,flag in itertools.product((True,False),('0','1')):
        monkeypatch.setenv('VMODEL_TOOL_ARGUMENT_ANY_ORDER',flag)
        c=GrammarConstraint.tools(engine,[tool()],required=required,allow_parallel=False)
        payload='<tool_call>'+json.dumps(dict(name='list_assets',arguments=dict(maximum=2,excludePath='/Archive/')))+'</tool_call>'
        assert c.matcher.accept_string(payload)==(flag=='1')


def test_all_field_permutations_token_mask_fork_and_rollback(engine,monkeypatch):
    from runtime.structured import GrammarConstraint
    from runtime.xgrammar_cpu import backend
    import numpy as np
    monkeypatch.setenv('VMODEL_TOOL_ARGUMENT_ANY_ORDER','1')
    for required,order in itertools.product((True,False),itertools.permutations(
            [('excludePath','/Archive/'),('maximum',2),('excludeLabel','private')])):
        c=GrammarConstraint.tools(engine,[tool()],required=required,allow_parallel=False)
        text='<tool_call>'+json.dumps(dict(name='list_assets',arguments=dict(order)))+'</tool_call>'
        mask=backend().allocate_token_bitmask(1,engine.cfg.vocab_size)
        for token in engine.tokenizer.encode(text,add_special_tokens=False).ids:
            c.matcher.fill_next_token_bitmask(mask)
            assert (np.asarray(mask).view(np.uint32)[0,token//32] >> (token%32))&1
            fork=c.matcher.fork();assert fork.accept_token(token)
            assert c.matcher.accept_token(token)
            c.matcher.rollback(1);assert c.matcher.accept_token(token)


@pytest.mark.parametrize('arguments',['{"maximum":2,"maximum":3}',
    '{"maximum":2,"excludePath":"a","excludePath":"b"}',
    '{"excludePath":"a"}', '{"maximum":"wrong"}', '{"maximum":2,"extra":1}'])
def test_invalid_calls_never_exposed_even_if_grammar_overapproximates(arguments):
    from runtime.toolcalls import parse_tool_calls
    from runtime.structured import tool_argument_schemas
    text='<tool_call>{"name":"list_assets","arguments":'+arguments+'}</tool_call>'
    content,calls=parse_tool_calls(text,'qwen3_5',allowed_names=['list_assets'],argument_schemas=tool_argument_schemas([tool()]))
    assert calls==[] and content==text


def test_nested_duplicates_and_duplicate_envelope_names_are_rejected():
    from runtime.toolcalls import parse_tool_calls
    for raw in ['{"name":"a","name":"b","arguments":{}}',
                '{"name":"a","arguments":{"nested":{"x":1,"x":2}}}']:
        text='<tool_call>'+raw+'</tool_call>'
        assert parse_tool_calls(text,'qwen3_5')==(text,[])


def test_original_capture_correct_arguments_in_natural_order(engine,monkeypatch):
    from runtime.structured import GrammarConstraint,tool_argument_schemas
    from runtime.toolcalls import parse_tool_calls
    path=Path('logs/captured_requests/1784574315421_94161f5f.json')
    if not path.exists():pytest.skip('private capture unavailable')
    original=json.loads(path.read_text());selected=next(t for t in original['tools'] if t.get('name')=='plugin__plex__plex_list_library')
    before=copy.deepcopy(selected)
    arguments=dict(mediaType='all',ratingOperator='lte',movieRatingValue='PG-13',showRatingValue='TV-Y7',excludeRootFolderPath='/Kids/',limit=100,offset=0)
    text='<tool_call>'+json.dumps(dict(name=selected['name'],arguments=arguments))+'</tool_call>'
    for required in (True,False):
        monkeypatch.setenv('VMODEL_TOOL_ARGUMENT_ANY_ORDER','1')
        c=GrammarConstraint.tools(engine,[selected],required=required,allow_parallel=False)
        assert c.matcher.accept_string(text)
        _,calls=parse_tool_calls(text,'qwen3_5',allowed_names=[selected['name']],argument_schemas=tool_argument_schemas([selected]))
        assert json.loads(calls[0]['function']['arguments'])==arguments
    assert selected==before
