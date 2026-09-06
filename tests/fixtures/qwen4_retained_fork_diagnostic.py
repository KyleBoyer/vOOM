"""Private post-generation retained-fork experiment; never serving-time proof.

Dependencies are injected so validation/sampling/ownership can be tested without
MLX. The dedicated max1 HTTP observer is the only caller. This experiment first
materializes retained PLE histories, then copies those small histories
bit-for-bit into independent storage. It never recomputes a model operator.
"""

import hashlib
import json
import time


def copy_ple_bits(value, *, array_module, numpy_module):
    """Independent host copy of the 16-bit payload, with no float conversion."""
    if value.dtype not in (array_module.bfloat16, array_module.float16):
        raise ValueError("PLE diagnostic requires 16-bit activation payloads")
    host = numpy_module.asarray(value.view(array_module.uint16)).copy(order="C")
    detached = array_module.array(host, dtype=array_module.uint16).view(value.dtype)
    array_module.eval(detached)
    return detached


def complete_state_metadata(kv, *, array_digest):
    """Supplement the original array oracle with windows/starts/pool state."""
    count = len(kv.keys)
    aux = kv.qwen4_cache
    for values in (kv.values, kv._starts, kv._windows, aux.qsa_pooled_keys):
        if len(values) != count:
            raise ValueError("incomplete state metadata")
    metadata = {
        "offset": int(kv.offset), "starts": list(kv._starts),
        "windows": list(kv._windows),
        "compressed_mla": bool(kv.compressed_mla),
        "key_present": [value is not None for value in kv.keys],
        "value_present": [value is not None for value in kv.values],
        "qsa_pool_cache_enabled": bool(aux.qsa_pool_cache_enabled),
        "qsa_pooled_keys": [array_digest(value) if value is not None else None
                            for value in aux.qsa_pooled_keys],
    }
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "layers": count,
            "pooled_arrays": sum(value is not None for value in aux.qsa_pooled_keys),
            "pooled_bytes": sum(int(value.nbytes) for value in aux.qsa_pooled_keys
                                if value is not None)}


def diagnose_retained_fork(target, prompt_ids, *, expected_prefix, kv_type,
                           state_digest, metadata_digest, copy_ple,
                           synchronize_ple, pressure, metal_memory, clear_cache,
                           clock=time.perf_counter):
    """Measure synchronization and exact PLE detachment on a dedicated fork.

    Crucially, no array hashing or host read occurs before the synchronization
    sample: doing so could materialize away the hypothesis being measured.
    """
    slots = getattr(target, "_hot_prompt_slots", None)
    if (not isinstance(slots, list) or len(slots) != 1
            or getattr(target, "_hot_kv_persist", None) is not None):
        raise ValueError("exactly one RAM-only retained slot is required")
    slot = slots[0]
    endpoint = target.last_kv
    fork = slot.kv
    if (type(endpoint) is not kv_type or type(fork) is not kv_type
            or endpoint is fork
            or tuple(slot.tokens) != tuple(prompt_ids[:expected_prefix])
            or len(slot.tokens) != expected_prefix or fork.offset != expected_prefix
            or endpoint.offset != len(prompt_ids)
            or getattr(slot, "qwen4_retention_tile", 0) <= 0
            or slot.logits is not None or slot.exact_hidden is not None
            or slot.prompt_logits is not None or slot.approximate):
        raise ValueError("a separate marked complete-prefix fork is required")
    aux = fork.qwen4_cache
    if (aux is endpoint.qwen4_cache or fork.kda_cache is endpoint.kda_cache
            or aux.ple_conv is endpoint.qwen4_cache.ple_conv
            or fork.keys is endpoint.keys or fork.values is endpoint.values):
        raise ValueError("mutable cache containers must have distinct owners")
    stages = []

    def sample(label, elapsed=0.0):
        stages.append({"stage": label, "seconds": elapsed,
                       "metal": metal_memory(), "pressure": pressure()})

    def observe(kv):
        sha, arrays, payload, components = state_digest(kv)
        return {"sha256": sha, "arrays": arrays, "bytes": payload,
                "components": components, "metadata": metadata_digest(kv)}

    logical_ple = [{"layer": layer, "bytes": int(value.nbytes),
                    "shape": list(value.shape), "dtype": str(value.dtype)}
                   for layer, value in enumerate(aux.ple_conv) if value is not None]
    sample("before_ple_synchronize")
    started = clock()
    synchronize_ple(aux.ple_conv)
    sample("after_ple_synchronize", clock() - started)
    started = clock()
    before_fork, before_endpoint = observe(fork), observe(endpoint)
    sample("after_before_hashes", clock() - started)
    started = clock()
    copied_arrays = copied_bytes = 0
    for layer in range(len(aux.ple_conv)):
        value = aux.ple_conv[layer]
        if value is None:
            continue
        replacement = copy_ple(value)
        if (replacement is value or replacement.shape != value.shape
                or replacement.dtype != value.dtype):
            raise ValueError("PLE detachment must return an independent same-format array")
        copied_arrays += 1
        copied_bytes += int(value.nbytes)
        aux.ple_conv[layer] = replacement
        del value, replacement
    sample("after_ple_detach", clock() - started)
    started = clock()
    clear_cache()
    sample("after_allocator_clear", clock() - started)
    started = clock()
    after_fork, after_endpoint = observe(fork), observe(endpoint)
    sample("after_verification_hashes", clock() - started)
    fork_equal = before_fork == after_fork
    endpoint_equal = before_endpoint == after_endpoint
    return {
        "schema": "voom.qwen4-retained-fork-memory-diagnostic.v1",
        "available": True, "serving_timing_proof": False,
        "serving_pressure_proof": False, "prefix_tokens": expected_prefix,
        "endpoint_tokens": len(prompt_ids),
        "mutation": "retained PLE arrays only; exact-bit comparison required",
        "fork_before": before_fork, "fork_after": after_fork,
        "endpoint_before": before_endpoint, "endpoint_after": after_endpoint,
        "fork_equal": fork_equal, "endpoint_equal": endpoint_equal,
        "copied_arrays": copied_arrays, "copied_bytes": copied_bytes,
        "logical_ple": logical_ple,
        "state_hash_host_read_bytes": 2 * sum(
            state["bytes"] + state["metadata"]["pooled_bytes"]
            for state in (before_fork, before_endpoint)),
        "ple_copy_host_read_bytes": copied_bytes,
        "stages": stages,
        "synchronize_active_released_bytes": (
            stages[0]["metal"]["active_bytes"] - stages[1]["metal"]["active_bytes"]),
        "detach_active_released_bytes": (
            stages[2]["metal"]["active_bytes"] - stages[3]["metal"]["active_bytes"]),
        "note": "Post-generation diagnostic mutates one retained fork, not a production optimization. Hashing and cache clearing also affect pressure; stages must not be conflated.",
    }
