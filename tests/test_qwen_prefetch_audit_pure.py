import ast
from pathlib import Path

import pytest

from runtime.profiles import apply_runtime_profiles


def test_prefetch_overlay_changes_only_bounded_cache_and_depth():
    env={}
    apply_runtime_profiles(['qwen35-prefetch1-cache512-audit'],environ=env)
    assert env=={'VMODEL_QWEN35_WEIGHT_CACHE_MB':'512','VMODEL_QWEN35_PREFETCH_DEPTH':'1'}


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
