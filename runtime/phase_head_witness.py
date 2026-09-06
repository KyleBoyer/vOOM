"""Read-only, non-atomic memory scalars; no MLX import or tensor traversal."""

import psutil


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
