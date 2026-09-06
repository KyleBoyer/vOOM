"""Small ownership helpers shared by text and vision generation paths."""

from __future__ import annotations


def detach_unretained_qwen4_endpoint(owner, plain_kv_type) -> dict:
    """Remove only the serving engine's non-slot plain-RAM endpoint reference.

    Caller owns the inference lock and has finished generation/trace/report
    consumers. Never call release(), mutate a cache/slot, or clear hidden state:
    plain KV forks can share tensor storage with a retained prefix. Other owners
    remain valid. Logical bytes do not predict unique or physically freed bytes.
    Direct-engine generation does not invoke this serving-only helper.
    """
    def skipped(reason):
        return {"detached": False, "reason": reason,
                "detached_endpoint_logical_bytes": 0}

    rc = getattr(owner, "rc", None)
    if (getattr(getattr(owner, "cfg", None), "model_type", None) != "qwen4_exp"
            or getattr(rc, "hot_prompt_kv", False) is not True
            or getattr(rc, "qwen4_hot_kv_tile_aligned", False) is not True):
        return skipped("unsupported-policy")
    slots = getattr(owner, "_hot_prompt_slots", None)
    if type(slots) is not list or not slots:
        return skipped("no-known-retained-slots")
    if any(type(getattr(slot, "kv", None)) is not plain_kv_type for slot in slots):
        return skipped("non-plain-retained-slot")
    endpoint = getattr(owner, "last_kv", None)
    if type(endpoint) is not plain_kv_type:
        return skipped("no-plain-endpoint")
    if any(slot.kv is endpoint for slot in slots):
        return skipped("endpoint-is-retained-slot")
    for name in ("_hot_kv_persist", "_prompt_kv_store", "_vision_prompt_cache",
                 "_glm53_vision_prompt_cache", "_provisional",
                 "_serial_kda_endpoints", "_serial_qwen4_endpoints",
                 "_serial_kda_factors"):
        if getattr(owner, name, None) is not None:
            return skipped("other-state-owner")
    if (getattr(owner, "_vision_embedding_cache", None)
            or getattr(owner, "_glm53_vision_embedding_cache", None)):
        return skipped("other-state-owner")
    size = endpoint.nbytes()
    if type(size) is not int or size < 0:
        return skipped("invalid-logical-byte-count")
    decision = {"detached": True, "reason": "non-slot-plain-endpoint",
                "detached_endpoint_logical_bytes": size,
                "retained_slot_count": len(slots),
                "mutates_retained_slots": False,
                "calls_endpoint_release": False,
                "physical_reclamation_proven": False}
    owner.last_kv = None
    # This local reference must die on return, before the caller samples/drains
    # the allocator. No endpoint/tensor reference escapes in the decision.
    return decision


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
