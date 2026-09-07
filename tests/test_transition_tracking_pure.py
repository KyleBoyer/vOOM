"""No-MLX transition-history policy and real route/commit/close guard tests."""

import ast
from collections import defaultdict
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from runtime.predictor import (MarkovExpertPredictor, make_expert_predictor,
                               parse_transition_tracking, validate_transition_tracking)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('raw,expected', [(None, True), ('1', True), ('0', False)])
def test_strict_environment_default_preserves_tracking(raw, expected):
    assert parse_transition_tracking(raw) is expected


@pytest.mark.parametrize('raw', ['', 'auto', 'false', 'true', ' 0', 0, False])
def test_bad_environment_policy_fails(raw):
    with pytest.raises(ValueError):
        parse_transition_tracking(raw)


@pytest.mark.parametrize('tracking', [0, 1, None, 'false', '0'])
def test_yaml_or_direct_nonboolean_policy_fails(tracking):
    with pytest.raises(ValueError):
        validate_transition_tracking(tracking, predictive_prefetch=False, warm_start=0)


@pytest.mark.parametrize('predictive,warm', [(True, 0), (False, 1), (True, 4), (False, -1)])
def test_disabled_tracking_refuses_every_declared_consumer(predictive, warm):
    with pytest.raises(ValueError, match='conflicts'):
        make_expert_predictor(2, 4, None, tracking=False,
                              predictive_prefetch=predictive, warm_start=warm)


def test_opt_out_does_not_even_touch_unreadable_existing_history(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('history must not be read or written')
    for name in ('exists', 'read_text', 'write_text'):
        monkeypatch.setattr(Path, name, forbidden)
    assert make_expert_predictor(2, 4, '/not-read/expert_transitions.json', tracking=False) is None


def test_default_still_loads_observes_predicts_and_saves(tmp_path):
    path = tmp_path/'history.json'
    path.write_text('{"0,1,2":3,"0,1,3":1}')
    predictor = make_expert_predictor(2, 4, path)
    assert type(predictor) is MarkovExpertPredictor
    assert predictor.predict(0, [1], 2) == [2, 3]
    predictor.observe(0, [1])
    predictor.observe(1, [3])
    predictor.save()
    restored = MarkovExpertPredictor(2, 4, path)
    assert restored.counts == {(0, 1, 2): 3, (0, 1, 3): 2}


def engine_methods():
    tree = ast.parse((ROOT/'runtime/engine.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'StreamingEngine')
    methods = [copy.deepcopy(n) for n in cls.body if isinstance(n, ast.FunctionDef)
               and n.name in ('begin_provisional', 'commit_provisional', '_record_expert_route')]
    ns = {}
    exec(compile(ast.Module(body=methods, type_ignores=[]), '<real-engine-methods>', 'exec'), ns)
    close = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'close')
    guard = next(n for n in close.body if isinstance(n, ast.If)
                 and ast.unparse(n.test) == 'self.predictor is not None')
    fn = ast.parse('def close_predictor(self):\n pass').body[0]
    fn.body = [copy.deepcopy(guard)]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
                 '<real-close-guard>', 'exec'), ns)
    return ns


def fake_engine(predictor):
    return SimpleNamespace(predictor=predictor, _provisional=None,
        expert_usage={}, expert_trace=[], expert_trace_phases=[],
        prefetcher=None, rc=SimpleNamespace(expert_route_overlap_telemetry=False,
                                           expert_predictive_prefetch=False))


def test_route_and_provisional_usage_identical_history_untouched_when_off(tmp_path):
    path = tmp_path/'history.json'
    path.write_text('{"0,1,2":3}')
    original = hashlib.sha256(path.read_bytes()).hexdigest()
    enabled = fake_engine(make_expert_predictor(3, 4, path))
    disabled = fake_engine(make_expert_predictor(3, 4, path, tracking=False))
    methods = engine_methods()
    for engine in (enabled, disabled):
        methods['_record_expert_route'](engine, 0, [1, 2])
        methods['begin_provisional'](engine)
        methods['_record_expert_route'](engine, 1, [2, 3], {2: [0], 3: [1]})
        methods['commit_provisional'](engine, 1)
    assert disabled.predictor is None
    assert enabled.expert_usage == disabled.expert_usage == {(0, 1): 1, (0, 2): 1, (1, 2): 1}
    assert enabled.expert_trace == disabled.expert_trace
    assert enabled.expert_trace_phases == disabled.expert_trace_phases
    methods['close_predictor'](disabled)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original


def test_constructor_validation_precedes_cache_or_storage_and_factory_is_wired():
    tree = ast.parse((ROOT/'runtime/engine.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'StreamingEngine')
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    calls = [n for n in ast.walk(init) if isinstance(n, ast.Call)]
    named = {ast.unparse(n.func): n for n in calls}
    assert named['validate_transition_tracking'].lineno < named['mx.set_cache_limit'].lineno
    assert named['validate_transition_tracking'].lineno < named['WeightStore'].lineno
    factory = named['make_expert_predictor']
    assert {k.arg: ast.unparse(k.value) for k in factory.keywords} == {
        'tracking': 'self.rc.expert_transition_tracking',
        'predictive_prefetch': 'self.rc.expert_predictive_prefetch', 'warm_start': 'self.rc.warm_start'}


def test_server_policy_in_both_cache_identities_and_config_before_construction():
    tree = ast.parse((ROOT/'runtime/server.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'EngineManager')
    get = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'get')
    keys = [n for n in ast.walk(get) if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == 'key' for t in n.targets)]
    assert len(keys) == 2
    for key in keys:
        assert isinstance(key.value, ast.Tuple)
        assert any(isinstance(n, ast.Name) and n.id == 'expert_transition_tracking' for n in key.value.elts)
    setters = [n for n in ast.walk(get) if isinstance(n, ast.Assign)
               and any(ast.unparse(t) == 'rc.expert_transition_tracking' for t in n.targets)]
    constructors = [n for n in ast.walk(get) if isinstance(n, ast.Call)
                    and ast.unparse(n.func) == 'StreamingEngine']
    assert len(setters) == 1
    assert setters[0].lineno < min(n.lineno for n in constructors)


def test_runtime_config_and_yaml_keep_legacy_default():
    tree = ast.parse((ROOT/'runtime/engine.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RuntimeConfig')
    field = next(n for n in cls.body if isinstance(n, ast.AnnAssign)
                 and ast.unparse(n.target) == 'expert_transition_tracking')
    assert ast.literal_eval(field.value) is True
    yaml_fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'from_yaml')
    call = next(n for n in ast.walk(yaml_fn) if isinstance(n, ast.Call)
                and ast.unparse(n.func) == 'cls')
    arg = next(k.value for k in call.keywords if k.arg == 'expert_transition_tracking')
    assert ast.unparse(arg) == "run.get('expert_transition_tracking', True)"
