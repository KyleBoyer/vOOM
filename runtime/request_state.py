"""Small ownership helpers shared by text and vision generation paths."""

from __future__ import annotations


def release_generation_state(owner) -> None:
    """Drop every engine-owned reference to a previous request's KV/state.

    This function deliberately has no MLX dependency, which gives the 16-GB
    single-owner rule a pure unit-test seam. Callers decide when to clear the
    allocator cache after these strong references are gone.
    """
    slots = list(getattr(owner, "_hot_prompt_slots", ()))
    last_kv = getattr(owner, "last_kv", None)
    states = [getattr(slot, "kv", slot) for slot in slots]
    if last_kv is not None:
        states.append(last_kv)
    # ``last_kv`` normally aliases one retained slot. Refcounted/shared caches
    # need an explicit release, but each cache object must be released only once.
    seen = set()
    for state in states:
        identity = id(state)
        if identity in seen:
            continue
        seen.add(identity)
        release = getattr(state, "release", None)
        if release is not None:
            release()
    owner._hot_prompt_slots = []
    owner.last_kv = None
    owner._h_window = None
    owner._h_last = None
    owner._provisional = None
