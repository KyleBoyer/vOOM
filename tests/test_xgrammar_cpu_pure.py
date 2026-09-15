"""Selection and dependency guards; no native grammar, model, Torch or MLX loads."""
import sys
from types import ModuleType

import pytest

from runtime import xgrammar_cpu as module
from runtime.profiles import apply_runtime_profiles


def test_profile_changes_only_explicit_grammar_backend():
    before={};after={}
    apply_runtime_profiles(['huihui-qwen38-27b-harness-preview'],environ=before)
    apply_runtime_profiles(['huihui-qwen38-27b-harness-preview','xgrammar-cpu-only'],environ=after)
    assert after=={**before,'VMODEL_XGRAMMAR_CPU_ONLY':'1'}


def test_cached_backend_does_not_reinitialize(monkeypatch):
    expected=object();monkeypatch.setattr(module,'_API',expected)
    assert module.backend() is expected


def test_unqualified_version_fails_before_native_import(monkeypatch):
    monkeypatch.setattr(module,'_API',None)
    monkeypatch.setattr(module.importlib.metadata,'version',lambda name:'wrong')
    with pytest.raises(RuntimeError,match='qualified'):module.backend()


@pytest.mark.parametrize('name',['torch','xgrammar'])
def test_preexisting_heavy_backend_is_not_overwritten(monkeypatch,name):
    monkeypatch.setattr(module,'_API',None)
    monkeypatch.setattr(module.importlib.metadata,'version',lambda n:'0.2.3' if n=='xgrammar' else '0.1.12')
    existing=ModuleType(name);monkeypatch.setitem(sys.modules,name,existing)
    with pytest.raises(RuntimeError,match='fresh process'):module.backend()
    assert sys.modules[name] is existing
