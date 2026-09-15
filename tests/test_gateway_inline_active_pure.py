import ast
import copy
import json
from pathlib import Path

import pytest

from runtime import gateway_inline_active as policy
from runtime.profiles import apply_runtime_profiles
from runtime.server import _parse_request_tool_calls


def arguments():
    function = dict(name='read_inventory', description='Return records.',
                    parameters=dict(type='object', properties={}))
    return dict(messages=[dict(role='tool', content='untrusted data')],
        tools=[dict(type='function', function=function)],
        raw_tools=[dict(type='function', **function)], client_choice='auto',
        force_reason=None, structured_output=None, host_route=False,
        terminal_synthesis=False, buffered='1')


def test_explicit_flag_and_nonmutating_selection():
    args = arguments()
    original = copy.deepcopy(args)
    assert policy.enabled('1', **args)
    assert not policy.enabled('0', **args)
    assert args == original
    for invalid in ('auto', '', None, True, 1):
        with pytest.raises(ValueError):
            policy.enabled(invalid, **args)


@pytest.mark.parametrize('changes', [
    dict(client_choice='required'), dict(client_choice='none'),
    dict(client_choice='specific:read_inventory'),
    dict(force_reason='tool-result-pagination'), dict(force_reason='external-action'),
    dict(structured_output={}), dict(structured_output={'type':'json_schema'}),
    dict(host_route=True), dict(terminal_synthesis=True), dict(buffered='0'),
    dict(messages=[]), dict(messages=[dict(role='user', content='read_inventory')]),
    dict(tools=[]), dict(raw_tools=[]),
])
def test_unsupported_authority_and_shapes_keep_existing_path(changes):
    assert not policy.enabled('1', **{**arguments(), **changes})


@pytest.mark.parametrize('count', [1, 2, 4, 5])
def test_small_catalog_bound_and_exact_schema_alignment(count):
    args=arguments()
    function=args['tools'][0]['function']
    functions=[dict(function,name='read_'+str(i)) for i in range(count)]
    args['tools']=[dict(type='function',function=f) for f in functions]
    args['raw_tools']=[dict(type='function',**f) for f in functions]
    assert policy.enabled('1',**args) is (count<=4)
    args['raw_tools'][0]['parameters']={'type':'object','required':['other']}
    assert not policy.enabled('1',**args)


@pytest.mark.parametrize('change', ['duplicate','reserved','mismatch'])
def test_ambiguous_catalog_is_not_inlined(change):
    args=arguments()
    if change=='duplicate':
        args['tools']*=2; args['raw_tools']*=2
    elif change=='reserved':
        args['tools'][0]['function']['name']='vmodel_shadow'
        args['raw_tools'][0]['name']='vmodel_shadow'
    else:
        args['raw_tools'][0]['name']='different'
    assert not policy.enabled('1',**args)


def test_real_call_keeps_model_authored_name_and_arguments():
    from runtime.server import _hidden_gateway_virtual_pairs
    tools,_ = _hidden_gateway_virtual_pairs()
    tools += arguments()['tools']
    text='<tool_call>\n'+json.dumps(dict(name='read_inventory',arguments={}))+'\n</tool_call>'
    content,calls=_parse_request_tool_calls(text,tools,'qwen3',allow_parallel=False)
    assert not content.strip()
    assert len(calls)==1 and calls[0]['function']['name']=='read_inventory'
    assert json.loads(calls[0]['function']['arguments'])=={}


def test_production_wires_catalog_policy_prose_and_telemetry_together():
    tree=ast.parse(Path('runtime/server.py').read_text())
    src=ast.unparse(tree)
    assert 'prompt_catalog = [*gateway_virtual_tools, *gateway_initial_tools]' in src
    assert 'prompt_raw_catalog = [*gateway_virtual_raw, *gateway_initial_raw]' in src
    assert 'inline_active_policy.POLICY if gateway_inline_active' in src
    assert 'preserve_tool_parameter_prose=gateway_inline_active or' in src
    assert "'gateway_inline_active': int(gateway_inline_active)" in src
    fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='run_hidden_gateway')
    branch=next(n for n in ast.walk(fn) if isinstance(n,ast.If)
        and ast.unparse(n.test)=='gateway_call is None')
    assert ast.unparse(branch.body[-1])=='return decision'
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
        and n.func.id=='_engine_generate' for n in ast.walk(branch))


def test_profile_only_enables_explicit_catalog_experiment():
    before={};after={};base=['huihui-qwen38-27b-workflow-head-rows-audit']
    apply_runtime_profiles(base,environ=before)
    apply_runtime_profiles(base+['gateway-inline-active-audit'],environ=after)
    assert after=={**before,policy.FLAG:'1'}


@pytest.mark.parametrize('selection,passed', [({},False),
    ({'gateway_inline_active':1,'gateway_inline_active_tools':1},True),
    ({'gateway_inline_active':0,'gateway_inline_active_tools':1},False),
    ({'gateway_inline_active':True,'gateway_inline_active_tools':1},False),
    ({'gateway_inline_active':1,'gateway_inline_active_tools':0},False),
    ({'gateway_inline_active':1,'gateway_inline_active_tools':5},False)])
def test_actual_fixture_requires_observed_inline_catalog(selection,passed):
    tree=ast.parse(Path('tests/fixtures/captured_transition_tracking_gate.py').read_text())
    branch=next(n for n in ast.walk(tree) if isinstance(n,ast.If)
        and ast.unparse(n.test)=="config.get('require_inline_active') is True")
    env=dict(config={'require_inline_active':True},
             response={'vmodel_tool_selection':selection},checks={})
    exec(compile(ast.fix_missing_locations(ast.Module(body=[branch],type_ignores=[])),
                 '<actual inline gate>','exec'),env)
    assert env['checks']['inline_active_catalog_used'] is passed
