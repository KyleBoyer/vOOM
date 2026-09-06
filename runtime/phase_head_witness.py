"""Non-atomic memory scalars and an opt-in default-stream allocator probe.

No MLX import, tensor traversal, request-state disposal or pressure verdicts.
"""

import psutil
import time


def sample_phase_head_memory(target, metal) -> dict:
    """Observe current ownership without synchronization or reclamation.

    Cache bytes are logical accounting, not physical memory. Metal's allocator
    peak is since its last reset, unlike the separately tracked request peak.
    Missing/failed reads remain null, never a fabricated zero or pressure pass.
    """
    probes = {
        "weight_cache_resident_bytes": lambda: target.cache.total_bytes,
        "weight_cache_pinned_bytes": lambda: target.cache.pinned_bytes,
        "weight_cache_budget_bytes": lambda: target.cache.max_bytes,
        "metal_active_bytes": lambda: metal.get_active_memory(),
        "metal_allocator_cache_bytes": lambda: metal.get_cache_memory(),
        "metal_peak_since_last_reset_bytes": lambda: metal.get_peak_memory(),
        "request_true_peak_metal_bytes": lambda: target._true_peak_metal_bytes,
        "system_available_bytes": lambda: psutil.virtual_memory().available,
        "system_swap_used_bytes": lambda: psutil.swap_memory().used,
    }
    values = {}
    unavailable = []
    for key, probe in probes.items():
        try:
            value = probe()
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("not a nonnegative byte count")
            values[key] = value
        except Exception:
            values[key] = None
            unavailable.append(key)
    return {"available": not unavailable, "unavailable_fields": unavailable,
            **values}


def post_generation_memory_witness(target, metal, *, barrier=False) -> dict:
    """Bounded serving-boundary observation, optionally draining the allocator.

    The caller must have returned from generation and still hold INFER_LOCK.
    Never dispose a request endpoint, hot slot, cache entry, RNG or tensor here.
    synchronize/clear_cache are explicit experimental operations, not passive
    observations. synchronize() waits on the default device's default stream,
    not every device/worker stream. Errors propagate: device failure is not
    telemetry loss.
    There is no sleep, collection loop, peak reset or pressure-gate replacement.
    """
    def clock():
        try:
            return time.perf_counter()
        except Exception:
            return None

    def elapsed(started):
        ended = clock()
        return ended - started if ended is not None and started is not None else None

    def sample():
        started = clock()
        try:
            values = sample_phase_head_memory(target, metal)
        except Exception:
            values = {"available": False, "reason": "observation-error"}
        values["observation_seconds"] = elapsed(started)
        return values

    started = clock()
    result = {
        "schema": "voom.post-generation-memory-witness.v1",
        "scope": "server_after_generation_return_before_protocol_completion",
        "atomic": False,
        "synchronizes_device": bool(barrier),
        "synchronization_scope": (
            "default_stream_of_default_device" if barrier else "none"),
        "clears_allocator_cache": bool(barrier),
        "disposes_request_or_reuse_state": False,
        "included_in_engine_total_s": False,
        "included_in_http_wall_s": True,
        "before": sample(),
    }
    if barrier:
        stage_started = clock()
        metal.synchronize()
        result["synchronize_seconds"] = elapsed(stage_started)
        result["after_synchronize"] = sample()
        stage_started = clock()
        metal.clear_cache()
        result["clear_cache_seconds"] = elapsed(stage_started)
        result["after_clear_cache"] = sample()
    result["wall_seconds"] = elapsed(started)
    return result
