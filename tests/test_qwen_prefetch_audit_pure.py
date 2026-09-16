import ast
from pathlib import Path

import pytest

from runtime.profiles import apply_runtime_profiles


def test_prefetch_overlay_changes_only_bounded_cache_and_depth():
    env={}
    apply_runtime_profiles(['qwen35-prefetch1-cache512-audit'],environ=env)
    assert env=={'VMODEL_QWEN35_WEIGHT_CACHE_MB':'512','VMODEL_QWEN35_PREFETCH_DEPTH':'1'}


def test_only_actual_serial_verifier_can_submit_phase_qualified_hints():
    tree=ast.parse(Path('runtime/engine.py').read_text())
    qualified=[]
    for fn in ast.walk(tree):
        if isinstance(fn,ast.FunctionDef):
            for node in ast.walk(fn):
                if isinstance(node,ast.Call) and any(k.arg=='phase' and isinstance(k.value,ast.Constant)
                    and k.value.value=='serial_verify' for k in node.keywords):
                    qualified.append(fn.name)
    assert qualified==['forward_tokens_serial_positions']
    src=ast.unparse(tree)
    assert 'serial_verify_only=self.rc.qwen35_prefill_split_weights' in src


@pytest.mark.parametrize('timing,expected', [({},False),
    ({'weight_prefetch_useful_bytes':1,'weight_prefetch_useful_load_s':0.2},True),
    ({'weight_prefetch_useful_bytes':True,'weight_prefetch_useful_load_s':0.2},False),
    ({'weight_prefetch_useful_bytes':100,'weight_prefetch_useful_load_s':0},False)])
def test_actual_useful_prefetch_gate(timing,expected):
    tree=ast.parse(Path('tests/fixtures/captured_transition_tracking_gate.py').read_text())
    branch=next(n for n in ast.walk(tree) if isinstance(n,ast.If)
        and ast.unparse(n.test)=="config.get('require_useful_weight_prefetch') is True")
    scope=dict(config={'require_useful_weight_prefetch':True},t=timing,checks={})
    exec(compile(ast.fix_missing_locations(ast.Module(body=[branch],type_ignores=[])),
        '<actual prefetch gate>','exec'),scope)
    assert scope['checks']['useful_weight_prefetch_observed'] is expected
