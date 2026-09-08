"""Opt-in, exact KV spill recovery after a refused Qwen serial reservation.

No model/prompt-dependent sizing or allocation credit from logical eviction.
Call on the engine's owning thread, before current-layer attention begins.
"""

import json
import time

import psutil


def _snapshot(governor, metal, incoming, margin):
    active = metal.get_active_memory()
    available = psutil.virtual_memory().available
    if any(type(n) is not int or n < 0 for n in (active, available, incoming, margin)):
        raise ValueError("invalid serial KV admission byte observation")
    ceiling = governor._metal_ceiling(active, available)
    return dict(metal_active_bytes=active, system_available_bytes=available,
                ceiling_bytes=ceiling,
                deficit_bytes=max(0, active + incoming + margin - ceiling))


def recover_serial_kv_admission(target, kv, metal, *, layer, positions, offset):
    """Return True ONLY after a fresh ordinary reservation succeeds.

    The caller already attempted normal admission and retains its MemoryError.
    A disabled/inapplicable/unsuccessful recovery returns False so the caller
    re-raises that original error. Spill/device errors propagate. This is one
    bounded attempt, never a loop or a lowered safety threshold.
    """
    if getattr(getattr(target, "rc", None), "qwen35_serial_kv_reclaim", False) is not True:
        return False
    if getattr(target.cfg, "model_type", None) not in ("qwen3_5", "qwen3_5_moe"):
        return False
    from .kv_paged import PagedKVCache

    governor = target.governor
    if governor is None or not isinstance(kv, PagedKVCache):
        return False
    incoming, margin = target._layer_transient, target._layer_transient_margin
    stats = getattr(target, "_qwen35_serial_kv_reclaim_stats", None)
    if stats is None:
        stats = target._qwen35_serial_kv_reclaim_stats = {}
    started = time.perf_counter()
    spills_before, spill_s_before = kv.stats.spills, kv.stats.spill_s
    row = dict(schema="voom.qwen35-serial-kv-reclaim.v1", layer=layer,
               verifier_positions=positions, start_offset=offset,
               incoming_bytes=incoming, margin_bytes=margin,
               kv_budget_bytes=kv.max_bytes, logical_before_bytes=kv.nbytes(),
               logical_reclaimed_bytes=0, metal_active_released_bytes=None,
               outcome="error", reservation_retried=False)
    try:
        row["before"] = _snapshot(governor, metal, incoming, margin)
        requested = row["requested_bytes"] = row["before"]["deficit_bytes"]
        spill_started = time.perf_counter()
        row["logical_reclaimed_bytes"] = kv.reclaim_closed_pages(
            requested, protected_layer=layer)
        row["reclaim_seconds"] = time.perf_counter() - spill_started
        row["logical_after_bytes"] = kv.nbytes()
        row["after_reclaim"] = _snapshot(governor, metal, incoming, margin)
        # Signed, non-atomic observation, not an allocation credit or proof of
        # system-available headroom. Retained aliases may make this zero.
        row["metal_active_released_bytes"] = (
            row["before"]["metal_active_bytes"]
            - row["after_reclaim"]["metal_active_bytes"])
        if requested and not row["logical_reclaimed_bytes"]:
            row["outcome"] = "no_candidates"
            return False
        row["reservation_retried"] = True
        try:
            governor.reserve(incoming, margin=margin, reason="serial-verify-transient")
        except MemoryError:
            row["outcome"] = "refused"
            return False
        row["outcome"] = "admitted"
        return True
    except Exception as error:
        row["error_type"] = type(error).__name__
        raise
    finally:
        # A later page write can fail after earlier pages spilled successfully.
        # Preserve that partial progress in the failure witness as well.
        row["logical_after_bytes"] = kv.nbytes()
        row["logical_reclaimed_bytes"] = max(
            0, row["logical_before_bytes"] - row["logical_after_bytes"])
        row["spill_pages"] = kv.stats.spills - spills_before
        row["spill_seconds"] = kv.stats.spill_s - spill_s_before
        row["wall_seconds"] = time.perf_counter() - started
        stats["attempts"] = stats.get("attempts", 0) + 1
        outcome = row["outcome"]
        stats[outcome] = stats.get(outcome, 0) + 1
        for key in ("logical_reclaimed_bytes", "spill_pages", "spill_seconds", "wall_seconds"):
            stats[key] = stats.get(key, 0) + row[key]
        records = stats.setdefault("records", [])
        if len(records) < 64:
            records.append(row)
            try:
                print("[qwen35-serial-kv-reclaim] " + json.dumps(
                    row, sort_keys=True, allow_nan=False), flush=True)
            except (OSError, ValueError):
                stats["log_errors"] = stats.get("log_errors", 0) + 1
        else:
            stats["records_dropped"] = stats.get("records_dropped", 0) + 1
