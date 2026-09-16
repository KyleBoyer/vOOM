"""Initial catalog availability never manufactures a tool call or answer."""
import ast
import copy
from pathlib import Path

import pytest

from runtime import gateway_inline_initial as policy
from runtime.profiles import apply_runtime_profiles
from tests.test_gateway_inline_active_pure import arguments


def args():
    value = arguments()
    value.update(messages=[dict(role='system', content='Respect instructions.'),
        dict(role='developer', content='Return accurate arguments.'),
        dict(role='user', content='Read inventory records.')],
        force_reason='external-action-imperative', activated_names=())
    return value


def test_default_off_strict_opt_in_and_nonmutation():
    value = args()
    original = copy.deepcopy(value)
    assert policy.candidates('0', **value) is None
    assert policy.candidates('1', **value) == (value['tools'], value['raw_tools'], 0)
    assert value == original
    for invalid in ('auto', '', None, True, 1):
        with pytest.raises(ValueError):
            policy.candidates(invalid, **value)


@pytest.mark.parametrize('change', [dict(client_choice='required'),
    dict(client_choice='none'), dict(client_choice='specific:read_inventory'),
    dict(force_reason='tool-result-pagination'), dict(force_reason='client-required'),
    dict(structured_output={}), dict(host_route=True), dict(terminal_synthesis=True),
    dict(buffered='0'), dict(activated_names=('read_inventory',)), dict(messages=[]),
    dict(messages=[dict(role='tool', content='data')]),
    dict(messages=[dict(role='tool', content='data'), dict(role='user',content='continue')]),
    dict(messages=[dict(role='assistant',tool_calls=[{}]), dict(role='user',content='continue')]),
    dict(tools=[]), dict(raw_tools=[])])
def test_authority_and_continuations_are_unchanged(change):
    assert policy.candidates('1', **{**args(), **change}) is None


def test_latest_user_only_ranking_bounded_with_no_embeddings(monkeypatch):
    value=args()
    f=value['tools'][0]['function']
    functions=[dict(f, name='lookup_'+str(i)) for i in range(9)]
    value['tools']=[dict(type='function',function=f) for f in functions]
    value['raw_tools']=[dict(type='function',**f) for f in functions]
    seen=[]
    def ranking(tools, messages, **kwargs):
        seen.append((tools,messages,kwargs)); return list(reversed(range(9)))
    monkeypatch.setattr(policy,'rank_tool_indices',ranking)
    monkeypatch.setattr(policy,'pinned_tool_indices',lambda tools,messages:[1])
    chosen,raw,pins=policy.candidates('1',**value)
    assert chosen==[value['tools'][i] for i in (1,8,7,6)]
    assert raw==[value['raw_tools'][i] for i in (1,8,7,6)] and pins==1
    assert seen==[(value['tools'],[value['messages'][-1]],{'use_embeddings':False})]
    monkeypatch.setattr(policy,'pinned_tool_indices',lambda tools,messages:list(range(5)))
    assert policy.candidates('1',**value) is None


@pytest.mark.parametrize('change', ['reserved', 'misaligned', 'duplicate'])
def test_invalid_catalog_falls_back(change):
    value=args()
    if change=='reserved':
        value['tools'][0]['function']['name']='vmodel_shadow'
        value['raw_tools'][0]['name']='vmodel_shadow'
    elif change=='misaligned':
        value['raw_tools'][0]['parameters']={'type':'array'}
    else:
        value['tools']*=2;value['raw_tools']*=2
    assert policy.candidates('1',**value) is None


def test_profile_only_opts_in_and_actual_wiring():
    before={};after={};base=['huihui-qwen38-27b-workflow-head-rows-audit']
    apply_runtime_profiles(base,environ=before)
    apply_runtime_profiles(base+['gateway-inline-initial-audit'],environ=after)
    assert after=={**before,policy.FLAG:'1'}
    source=ast.unparse(ast.parse(Path('runtime/server.py').read_text()))
    assert 'gateway_inline_active = gateway_inline_active or gateway_inline_initial' in source
    assert "'gateway_inline_initial': int(gateway_inline_initial)" in source
    assert 'prompt_catalog = [*gateway_virtual_tools, *gateway_initial_tools]' in source
    assert 'inline_initial_policy.POLICY if gateway_inline_initial' in source
    assert 'first output' in policy.POLICY and 'Do not invent external results' in policy.POLICY


@pytest.mark.parametrize('reason,expected', [(None,'auto'),
    ('external-action-imperative','required')])
def test_initial_constraint_allows_real_actions_instead_of_forcing_search(reason, expected):
    tree=ast.parse(Path('runtime/server.py').read_text())
    assignment=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='gateway_decision_choice' for t in n.targets))
    def hidden(choice,force,activated):
        return 'specific:vmodel_search_tools' if force is not None else choice
    scope=dict(gateway_terminal_pagination_synthesis=False,
        gateway_inline_conversation=False,
        gateway_deterministic_render=None,gateway_inline_initial=True,
        inline_initial_policy=policy,tool_choice='auto',gateway_force_reason=reason,
        gateway_enabled=True,gateway_activated_names=(),_hidden_gateway_decision_choice=hidden)
    code=compile(ast.fix_missing_locations(ast.Module(body=[assignment],type_ignores=[])),
                 '<actual gateway choice>','exec')
    exec(code,scope)
    assert scope['gateway_decision_choice']==expected
    scope['gateway_inline_initial']=False
    exec(code,scope)
    assert scope['gateway_decision_choice']==hidden('auto',reason,False)
    scope.update(gateway_inline_initial=True,gateway_terminal_pagination_synthesis=True)
    exec(code,scope)
    assert scope['gateway_decision_choice']=='none'


@pytest.mark.parametrize('choice,reason', [('required',None),('none',None),
    ('specific:read_inventory',None),('auto','tool-result-pagination'),
    ('auto','client-required')])
def test_initial_decision_rejects_authority_expansion(choice,reason):
    with pytest.raises(ValueError):
        policy.decision_choice(choice,reason)


@pytest.mark.parametrize('choice,expected', [('required',0),('auto',0),
    ('specific:vmodel_search_tools',1),('specific:vmodel_enable_tools',0)])
def test_search_forced_witness_reports_constraint_not_intent(choice,expected):
    tree=ast.parse(Path('runtime/server.py').read_text())
    expression=next(value for node in ast.walk(tree) if isinstance(node,ast.Dict)
        for key,value in zip(node.keys,node.values)
        if isinstance(key,ast.Constant) and key.value=='gateway_search_forced')
    actual=eval(compile(ast.Expression(expression),'<actual search witness>','eval'),
        dict(gateway_decision_choice=choice,_HIDDEN_TOOL_SEARCH_NAME='vmodel_search_tools'))
    assert actual==expected


@pytest.mark.parametrize('selection,passed', [({},False),
    ({'gateway_inline_initial':1,'gateway_inline_active':1,'gateway_inline_active_tools':4,
      'gateway_search_rounds':0,'gateway_host_routed':0},True),
    ({'gateway_inline_initial':True},False),
    ({'gateway_inline_initial':1,'gateway_inline_active':1,'gateway_inline_active_tools':4,
      'gateway_search_rounds':1,'gateway_host_routed':0},False)])
def test_actual_fixture_requires_direct_observed_path(selection,passed):
    tree=ast.parse(Path('tests/fixtures/captured_transition_tracking_gate.py').read_text())
    branch=next(n for n in ast.walk(tree) if isinstance(n,ast.If)
        and ast.unparse(n.test)=="config.get('require_inline_initial') is True")
    scope=dict(config={'require_inline_initial':True},response={'vmodel_tool_selection':selection},checks={})
    exec(compile(ast.fix_missing_locations(ast.Module(body=[branch],type_ignores=[])),
        '<actual initial acceptance>', 'exec'),scope)
    assert scope['checks']['inline_initial_direct_used'] is passed
