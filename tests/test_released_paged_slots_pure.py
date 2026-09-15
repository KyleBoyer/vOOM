"""Released attention endpoints cannot retain stale recurrent hot slots."""

import gc
from types import SimpleNamespace
import weakref

from runtime.request_state import detach_released_kv_slots


class State:
    pass


def test_only_identity_matched_slots_are_detached_without_releasing_forks():
    state, fork = State(), State()
    slot = SimpleNamespace(kv=state, tokens=(1, 2))
    other = SimpleNamespace(kv=fork, tokens=(1, 2))
    owner = SimpleNamespace(_hot_prompt_slots=[slot, other, slot])
    assert detach_released_kv_slots(owner, state) == 2
    assert owner._hot_prompt_slots == [other]
    assert slot.kv is state and other.kv is fork
    assert slot.tokens == other.tokens == (1, 2)
    assert detach_released_kv_slots(owner, state) == 0


def test_no_slot_reference_retains_recurrent_state_after_endpoint_drops():
    endpoint, recurrent = State(), State()
    endpoint.kda_cache = recurrent
    witness = weakref.ref(recurrent)
    owner = SimpleNamespace(_hot_prompt_slots=[SimpleNamespace(kv=endpoint)])
    assert detach_released_kv_slots(owner, endpoint) == 1
    del endpoint, recurrent
    gc.collect()
    assert witness() is None


def test_missing_slots_is_a_noop():
    owner = SimpleNamespace()
    assert detach_released_kv_slots(owner, State()) == 0
    assert not hasattr(owner, '_hot_prompt_slots')
