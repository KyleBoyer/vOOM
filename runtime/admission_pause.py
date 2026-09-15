"""Bounded opt-in wait for an unchanged allocation predicate to become safe.

No allocation is made here. A governor-lifetime budget prevents per-layer or
generation retries from turning transient admission waits into an endless loop.
"""
from __future__ import annotations

import json
import os
import time

import psutil


class AdmissionPause:
    def __init__(self, enabled=False):
        self.remaining_s = 30.0 if enabled else 0.0
        self.events = 0

    @classmethod
    def from_environment(cls):
        mode = os.environ.get('VMODEL_ADMISSION_PAUSE', '0')
        if mode not in ('0', '1'):
            raise ValueError('VMODEL_ADMISSION_PAUSE must be 0 or 1')
        return cls(mode == '1')

    def wait(self, sample, *, critical, reason, clear_cache,
             clock=time.monotonic, sleep=time.sleep, swap=psutil.swap_memory):
        if self.remaining_s <= 0:
            return None
        started = clock()
        limit = min(5.0, self.remaining_s)
        baseline = swap()
        state = sample()
        stop = 'timeout'
        samples = 0
        try:
            while clock() - started < limit:
                active, available, ceiling, projected = state
                current = swap()
                if (available < critical or current.used - baseline.used > 16_000_000
                        or current.sout - baseline.sout > 16_000_000):
                    stop = 'pressure'
                    break
                if projected <= ceiling:
                    stop = 'admissible'
                    break
                sleep(min(0.25, max(0.0, limit - (clock() - started))))
                clear_cache()
                state = sample()
                samples += 1
            # Always return the last real sample. Caller still checks it and
            # re-samples after this method; a timeout cannot authorize a page.
            return state if stop == 'admissible' else None
        finally:
            elapsed = max(0.0, clock() - started)
            self.remaining_s = max(0.0, self.remaining_s - elapsed)
            self.events += 1
            print('[admission-pause] ' + json.dumps(dict(
                schema='voom.admission-pause.v1', event=self.events,
                reason=reason, wall_seconds=elapsed, remaining_seconds=self.remaining_s,
                samples=samples, stop=stop, system_available_bytes=state[1],
                projected_bytes=state[3], ceiling_bytes=state[2])), flush=True)
