import ast
from pathlib import Path

import pytest

from runtime import gateway_execution_auto as policy
from runtime.profiles import apply_runtime_profiles


def allowed(value='1', **changes):
    kwargs = dict(client_choice='auto', force_reason=None,
                  structured_output=None, abstention_available=True)
    kwargs.update(changes)
    return policy.enabled(value, **kwargs)


def test_explicit_flag_and_authority_boundaries():
    assert allowed()
    assert not allowed('0')
    for choice in ('required', 'none', 'specific:call', {'type': 'function'}):
        assert not allowed(client_choice=choice)
    for reason in ('client-required', 'tool-result-pagination', 'external-action'):
        assert not allowed(force_reason=reason)
    assert not allowed(structured_output={'type': 'json_schema'})
    assert not allowed(abstention_available=False)
    for bad in ('auto', '', True, 1, None):
        with pytest.raises(ValueError):
            allowed(bad)


@pytest.mark.parametrize('answer', ['Weather is 20 C.', 'Four inventory items remain.',
    'ALPHA_G, BRAVO_PG13', '{"result": 42}', 'No matching records were returned.'])
def test_plain_answer_is_content_agnostic(answer):
    assert policy.plain_answer(answer, [])
    assert not policy.plain_answer(answer, [{'function': {'name': 'lookup'}}])


@pytest.mark.parametrize('text', ['', '  ', None, '<tool_call>{', '</tool_call>',
    '<|tool_call|>', '[TOOL_CALLS]', '<function=lookup', '<|python_tag|>',
    'vmodel_no_suitable_tool'])
def test_incomplete_call_or_empty_text_cannot_become_an_answer(text):
    assert not policy.plain_answer(text, [])


def execution_branch(**changes):
    """Execute the real server's output classification, without a model import."""
    tree = ast.parse(Path('runtime/server.py').read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
        and ast.unparse(n.test) == 'abstain_call is not None')
    env = dict(abstain_call=None, real_calls=[], execution_calls=[],
        execution_auto=True, _execution_content='Recorded answer',
        gateway_execution_auto=policy, result={'text': 'Recorded answer', 'tokens': [11, 12]},
        _HIDDEN_GATEWAY_ABSTAIN_TEXT='fixed host message')
    env.update(changes)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 '<actual server output branch>', 'exec'), env)
    return env


def test_actual_server_preserves_generated_answer_and_token_witness():
    env = execution_branch()
    assert env['execution_outcome'] == 'model_answer'
    assert env['result'] == {'text': 'Recorded answer', 'tokens': [11, 12]}
    env = execution_branch(real_calls=[{'name': 'real'}], execution_calls=[{'name': 'real'}])
    assert env['execution_outcome'] == 'real_tool'
    assert env['result']['text'] == 'Recorded answer'


def test_actual_server_keeps_old_failure_paths_and_default():
    for changes, outcome in (({'execution_auto': False}, 'invalid_or_incomplete_tool_call'),
        ({'_execution_content': '<tool_call>{'}, 'invalid_or_incomplete_tool_call'),
        ({'abstain_call': {'name': 'abstain'}}, 'no_suitable_tool')):
        env = execution_branch(**changes)
        assert env['execution_outcome'] == outcome
        assert env['result']['text'] == 'fixed host message'


def test_actual_prompt_constraint_and_telemetry_use_same_choice():
    tree = ast.parse(Path('runtime/server.py').read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
              and n.name == 'run_hidden_gateway')
    src = ast.unparse(fn)
    assert "execution_choice = 'auto' if execution_auto else 'required'" in src
    assert '_configure_constraint(engine, self._structured_output, prompt_tools, execution_choice, False)' in src
    assert "tool_choice=execution_choice" in src
    assert 'if execution_abstention_enabled and (not execution_auto):' in src
    assert "'gateway_execution_choice_required': not execution_auto" in src


def test_profile_changes_only_explicit_policy():
    before = {}; after = {}
    base = ['huihui-qwen38-27b-workflow-head-rows-audit']
    apply_runtime_profiles(base, environ=before)
    apply_runtime_profiles(base + ['gateway-execution-auto'], environ=after)
    assert after == {**before, policy.FLAG: '1'}


def test_terminal_diagnostic_profile_only_enables_existing_opt_in():
    before = {}; after = {}
    base = ['huihui-qwen38-27b-workflow-head-rows-audit']
    apply_runtime_profiles(base, environ=before)
    apply_runtime_profiles(base + ['gateway-terminal-synthesis-audit'], environ=after)
    assert after == {**before, 'VMODEL_FAST_TOOL_GATEWAY_TERMINAL_PAGINATION_SYNTHESIS': '1'}
