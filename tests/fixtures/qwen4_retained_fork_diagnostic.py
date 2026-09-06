"""Private post-generation retained-fork experiment; never serving-time proof.

Dependencies are injected so validation/sampling/ownership can be tested without
MLX. The dedicated max1 HTTP observer is the only caller. This experiment first
materializes retained PLE histories, then copies those small histories
bit-for-bit into independent storage. An explicit extension also detaches
retained DeltaNet convolution histories, which can own large padded backings.
It never recomputes a model operator.
"""

import hashlib
import json
import time


def copy_ple_bits(value, *, array_module, numpy_module):
    """Independent PLE/KDA history bits, with no float conversion."""
    if value.dtype not in (array_module.bfloat16, array_module.float16):
        raise ValueError("PLE diagnostic requires 16-bit activation payloads")
    host = numpy_module.asarray(value.view(array_module.uint16)).copy(order="C")
    detached = array_module.array(host, dtype=array_module.uint16).view(value.dtype)
    array_module.eval(detached)
    return detached


def detach_kda_histories(kda, *, copy_bits):
    """Replace only entries of an independently owned convolution-history list.

    The history tuples and their arrays may be shared with the live endpoint:
    neither is mutated in place. Drop the old tuple before returning so it
    cannot hide the allocator effect being measured by the caller.
    """
    arrays = payload = 0
    for layer in range(len(kda._conv)):
        history = kda._conv[layer]
        if history is None:
            continue
        replacements = []
        for value in history:
            if value is None:
                replacements.append(None)
                continue
            replacement = copy_bits(value)
            if (replacement is value or replacement.shape != value.shape
                    or replacement.dtype != value.dtype):
                raise ValueError("KDA detachment requires independent same-format arrays")
            replacements.append(replacement)
            arrays += 1
            payload += int(value.nbytes)
            del value, replacement
        kda._conv[layer] = tuple(replacements)
        del history, replacements
    return arrays, payload


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
                           include_kda_conv=False, expected_kda_layers=(),
                           expected_kda_shape=None, expected_kda_dtype=None,
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
    if include_kda_conv:
        recurrent = fork.kda_cache
        histories = getattr(fork.kda_cache, "_conv", None)
        endpoint_histories = getattr(endpoint.kda_cache, "_conv", None)
        if (not isinstance(histories, list) or not isinstance(endpoint_histories, list)
                or histories is endpoint_histories
                or len(histories) != len(fork.keys)
                or len(endpoint_histories) != len(endpoint.keys)
                or any(history is not None and not isinstance(history, tuple)
                       for history in histories)):
            raise ValueError("KDA convolution histories need distinct complete list owners and immutable tuples")
        if (getattr(recurrent, "spill_enabled", False)
                or getattr(recurrent, "_spill_meta", None)
                or getattr(recurrent, "_factor_capture", None) is not None):
            raise ValueError("KDA diagnostic requires resident non-captured histories")
        expected_layers = tuple(expected_kda_layers)
        if (not expected_layers or len(set(expected_layers)) != len(expected_layers)
                or any(type(layer) is not int or not 0 <= layer < len(histories)
                       for layer in expected_layers)
                or expected_kda_shape is None or expected_kda_dtype is None
                or {layer for layer, history in enumerate(histories)
                    if history is not None} != set(expected_layers)):
            raise ValueError("KDA diagnostic needs complete expected layer geometry")
        for layer in expected_layers:
            history = histories[layer]
            if (len(history) != 1 or history[0] is None
                    or tuple(history[0].shape) != tuple(expected_kda_shape)
                    or history[0].dtype != expected_kda_dtype):
                raise ValueError("KDA history layout differs from expected 16-bit convolution")
        del histories, endpoint_histories, recurrent, history
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
    logical_kda_conv = [
        {"layer": layer, "history_index": index, "bytes": int(value.nbytes),
         "shape": list(value.shape), "dtype": str(value.dtype)}
        for layer, history in enumerate(fork.kda_cache._conv)
        if history is not None
        for index, value in enumerate(history) if value is not None
    ] if include_kda_conv else []
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
    kda_copied_arrays = kda_copied_bytes = 0
    kda_active_released = 0
    if include_kda_conv:
        started = clock()
        kda_copied_arrays, kda_copied_bytes = detach_kda_histories(
            fork.kda_cache, copy_bits=copy_ple)
        sample("after_kda_detach", clock() - started)
        kda_active_released = (
            stages[-2]["metal"]["active_bytes"] - stages[-1]["metal"]["active_bytes"])
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
        "mutation": ("retained PLE and KDA convolution arrays only; exact-bit comparison required"
                     if include_kda_conv else "retained PLE arrays only; exact-bit comparison required"),
        "fork_before": before_fork, "fork_after": after_fork,
        "endpoint_before": before_endpoint, "endpoint_after": after_endpoint,
        "fork_equal": fork_equal, "endpoint_equal": endpoint_equal,
        "copied_arrays": copied_arrays, "copied_bytes": copied_bytes,
        "logical_ple": logical_ple,
        "kda_conv_included": bool(include_kda_conv),
        "logical_kda_conv": logical_kda_conv,
        "kda_copied_arrays": kda_copied_arrays,
        "kda_copied_bytes": kda_copied_bytes,
        "kda_copy_host_read_bytes": kda_copied_bytes,
        "kda_detach_active_released_bytes": kda_active_released,
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
