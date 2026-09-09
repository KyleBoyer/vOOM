"""The cache's own removed-page local must die before allocator reclamation."""
import weakref
from types import SimpleNamespace as NS

from runtime import weight_cache as module


def test_discard_drops_last_page_owner_before_clearing_allocator(monkeypatch):
    class Tensor:
        nbytes=12
    tensor=Tensor(); ref=weakref.ref(tensor)
    cache=module.WeightCache(NS(),max_bytes=100)
    cache._pages['phase']=module.WeightPage('phase',{'w':tensor},12)
    cache._total_bytes=12
    del tensor
    observations=[]
    monkeypatch.setattr(module,'_clear_device_cache',lambda:observations.append(ref() is None))
    assert cache.discard('phase',['w'])
    assert observations==[True]
    assert cache.total_bytes==0


def test_discard_does_not_drop_consumer_owned_tensors(monkeypatch):
    class Tensor:
        nbytes=12
    consumer=Tensor(); ref=weakref.ref(consumer)
    cache=module.WeightCache(NS(),max_bytes=100)
    cache._pages['phase']=module.WeightPage('phase',{'w':consumer},12)
    cache._total_bytes=12
    monkeypatch.setattr(module,'_clear_device_cache',lambda:None)
    assert cache.discard('phase',['w'])
    assert ref() is consumer


def test_pinned_and_missing_pages_preserve_existing_semantics(monkeypatch):
    cache=module.WeightCache(NS(),max_bytes=100)
    cache._pages['pin']=module.WeightPage('pin',{},0,pinned=True)
    calls=[]
    monkeypatch.setattr(module,'_clear_device_cache',lambda:calls.append(1))
    assert cache.discard('pin') is False and not calls and 'pin' in cache._pages
    assert cache.discard('missing') is False and calls==[1]
