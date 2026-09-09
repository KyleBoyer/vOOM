"""Opt-in exact head reload ordering; ordinary governor remains authoritative."""

import json
import time


def validate_policy(enabled, phase_head_enabled):
    if type(enabled) is not bool:
        raise ValueError('Qwen phase-head pre-admission must be a boolean')
    if enabled and phase_head_enabled is not True:
        raise ValueError('Qwen phase-head pre-admission requires the phase-scoped head')


def load_head(engine):
    """Trim dead LRU before fetch, reserve exact lease bytes, then restore pin.

    Cache-accounted release is never physical allocation credit. The ordinary
    governor observes actual headroom and can still refuse before any fetch.
    Pause prefetch throughout trim/reserve/fetch/promotion so it cannot refill
    the LRU in that interval. No head bytes, matmul or KV state are changed.
    """
    validate_policy(engine.rc.qwen35_phase_head_pre_admit,
        engine.rc.qwen35_serial_verify_suspend_lm_head)
    if (not engine.rc.qwen35_phase_head_pre_admit
            or not engine._qwen35_lm_head_pin_suspended
            or not engine._qwen35_lm_head_suspend_request_active
            or engine._lm_head_w is not None):
        raise ValueError('phase-head pre-admission requires an active dormant lease')
    if engine.governor is None:
        raise ValueError('phase-head pre-admission requires a live governor')
    phase_bytes = engine.cache.suspended_pin_bytes('qwen35:lm_head:persistent')
    if type(phase_bytes) is not int or phase_bytes <= 0:
        raise ValueError('phase-head pre-admission requires exact positive lease bytes')
    prefetcher = getattr(engine, 'prefetcher', None)
    was_paused = getattr(prefetcher, 'paused', False)
    stats = engine._qwen35_phase_head_admission_stats
    stats.update(schema='voom.qwen35-phase-head-admission.v1',
        scope='current_target_attempt_and_following_mtp', includes_prior_retry_attempts=False)
    stats['calls'] = stats.get('calls', 0) + 1
    row = dict(schema=stats['schema'], call=stats['calls'], requested_bytes=phase_bytes,
        logical_trimmed_bytes=0, reservation_attempted=False, fetched=False,
        pin_restored=False, outcome='error', physical_release='unmeasured')
    started = time.perf_counter()
    try:
        if prefetcher is not None:
            prefetcher.pause_and_wait_idle()
        # A suspended pin is permission to restore ownership, not extra cache
        # capacity. Evict BEFORE materializing the page, unlike ordinary get().
        target = max(0, int(engine.cache.max_bytes) - phase_bytes)
        row['logical_trimmed_bytes'] = int(engine.cache.trim_to(target))
        row['reservation_attempted'] = True
        try:
            engine.governor.reserve(phase_bytes, reason='qwen35-phase-lm-head')
        except MemoryError:
            row['outcome'] = 'reservation-refused'
            raise
        head = engine.cache.get('lm_head', ['lm_head.weight'])['lm_head.weight']
        row['fetched'] = True
        row['pin_restored'] = bool(engine._restore_qwen35_serial_verify_lm_head(head))
        row['outcome'] = 'loaded'
        return head
    finally:
        if prefetcher is not None:
            # A reservation or concurrent governor poll may have newly paused
            # speculation. Never restore an old False over that safety state.
            prefetcher.paused = bool(was_paused or getattr(
                engine.governor, 'paused_prefetch', False))
        row['seconds'] = time.perf_counter() - started
        for name, amount in (
                ('requested_bytes', phase_bytes),
                ('logical_trimmed_bytes', row['logical_trimmed_bytes']),
                ('reservation_calls', int(row['reservation_attempted'])),
                ('reservation_refusals', int(row['outcome'] == 'reservation-refused')),
                ('fetches', int(row['fetched'])),
                ('pin_restores', int(row['pin_restored'])),
                ('errors', int(row['outcome'] == 'error')),
                ('seconds', row['seconds'])):
            stats[name] = stats.get(name, 0) + amount
        try:
            print('[qwen35-phase-head-admission] '+json.dumps(row, sort_keys=True), flush=True)
        except Exception:
            pass  # logging cannot hide the original failure or change output
