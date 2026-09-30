from types import SimpleNamespace as NS
import pytest
from runtime.server import _active_context_limit, _validate_context_budget, RequestValidationError


def engine(model=262144,runtime=0):
    return NS(cfg=NS(max_position_embeddings=model),rc=NS(context_bound=runtime))


def test_http_operator_cap_is_explicit_minimum_and_never_truncates(monkeypatch):
    e=engine();monkeypatch.delenv('VMODEL_HTTP_MAX_CONTEXT_TOKENS',raising=False)
    assert _active_context_limit(e)==262144
    monkeypatch.setenv('VMODEL_HTTP_MAX_CONTEXT_TOKENS','16384')
    assert _active_context_limit(e)==16384
    assert _active_context_limit(engine(runtime=2048))==2048
    assert _active_context_limit(engine(model=8192))==8192
    assert _validate_context_budget(e,15360,1024,prompt_label='prompt',output_label='output')==16384
    with pytest.raises(RequestValidationError,match='exceeds active context limit=16384'):
        _validate_context_budget(e,15361,1024,prompt_label='prompt',output_label='output')
    assert e.cfg.max_position_embeddings==262144 and e.rc.context_bound==0
    monkeypatch.setenv('VMODEL_HTTP_MAX_CONTEXT_TOKENS','0')
    assert _active_context_limit(e)==262144


@pytest.mark.parametrize('bad',['','-1','true','1.5',' 12','auto'])
def test_malformed_operator_cap_fails_closed(monkeypatch,bad):
    monkeypatch.setenv('VMODEL_HTTP_MAX_CONTEXT_TOKENS',bad)
    with pytest.raises(RequestValidationError,match='VMODEL_HTTP_MAX_CONTEXT_TOKENS'):
        _active_context_limit(engine())
