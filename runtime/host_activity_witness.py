"""Read-only, bounded known-transcoder inventory; no ML framework or controls.

Names are a narrow allowlist, NOT proof of an idle CPU/GPU/disk or attribution
of system swap. Never inspect command lines, environment, files or user content.
No process is paused, killed, reprioritized, or otherwise modified.
"""

from __future__ import annotations

import math
import time

import psutil

FLAG = 'VMODEL_HOST_ACTIVITY_WITNESS'
MAX_PROCESSES = 8192
MAX_MATCHES = 32
MAX_WINDOW_SAMPLES = 1802
POLL_SECONDS = 2.0
NAMES = {'ffmpeg': 'ffmpeg', 'handbrakecli': 'HandBrakeCLI',
         'plex transcoder': 'Plex Transcoder'}
SCOPE = 'best-effort process-name allowlist; not general host-idle or swap-attribution proof'


def sample_known_transcoders(*, process_iter=None):
    started = time.monotonic()
    result = dict(schema='voom.known-transcoders.v1', scope=SCOPE, atomic=False,
        available=False, reason=None, monotonic_s=started, scanned=0, transcoders=[])
    try:
        iterator = (process_iter or psutil.process_iter)(attrs=['pid', 'name', 'create_time'])
        unidentified = False
        for process in iterator:
            if result['scanned'] >= MAX_PROCESSES:
                result['reason'] = 'process-limit'
                break
            result['scanned'] += 1
            info = process.info
            name = info.get('name')
            if not isinstance(name, str) or not name:
                unidentified = True
                continue
            canonical = NAMES.get(name.casefold())
            if canonical is None:
                continue
            pid, created = info.get('pid'), info.get('create_time')
            if (type(pid) is not int or pid <= 0 or type(created) not in (int, float)
                    or not math.isfinite(created) or created <= 0):
                unidentified = True
                continue
            if len(result['transcoders']) >= MAX_MATCHES:
                result['reason'] = 'match-limit'
                break
            result['transcoders'].append(dict(pid=pid, created=created, kind=canonical))
        else:
            result['available'] = not unidentified
            result['reason'] = 'unidentified-process' if unidentified else None
    except Exception as error:
        result['reason'] = 'inventory-error'
        result['error_type'] = type(error).__name__
    result['transcoders'].sort(key=lambda r: (r['pid'], r['created']))
    result['observation_seconds'] = time.monotonic() - started
    return result


def summarize_known_transcoders(samples):
    """Retain evidence of an earlier transient even if the last scan is quiet."""
    result = dict(scope=SCOPE, available=True, samples=0, transcoders=[], passed=False)
    seen = {}
    for sample in samples:
        if result['samples'] >= MAX_WINDOW_SAMPLES:
            result['available'] = False
            break
        result['samples'] += 1
        if type(sample) is not dict:
            result['available'] = False
            continue
        if sample.get('available') is not True:
            result['available'] = False
        processes = sample.get('transcoders', [])
        if type(processes) is not list:
            result['available'] = False
            continue
        if len(processes) > MAX_MATCHES:
            result['available'] = False
        for process in processes[:MAX_MATCHES]:
            if type(process) is not dict:
                result['available'] = False
                continue
            pid, created, kind = (process.get(k) for k in ('pid', 'created', 'kind'))
            if (type(pid) is not int or pid <= 0 or type(created) not in (int, float)
                    or not math.isfinite(created) or created <= 0 or kind not in NAMES.values()):
                result['available'] = False
                continue
            key = (pid, created)
            if key not in seen and len(seen) >= MAX_MATCHES:
                result['available'] = False
                continue
            seen[key] = dict(pid=pid, created=created, kind=kind)
    result['transcoders'] = [seen[k] for k in sorted(seen)]
    result['available'] = result['available'] and result['samples'] > 0
    result['passed'] = result['available'] and not result['transcoders']
    return result


def sample_transcoder_window(seconds, *, sample=sample_known_transcoders,
                             clock=time.monotonic, sleep=time.sleep):
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 0 <= seconds <= 3600:
        raise ValueError('transcoder sampling window must be finite and in 0..3600 seconds')
    started = clock()
    def snapshots():
        for _ in range(MAX_WINDOW_SAMPLES):
            yield sample()
            remaining = seconds - (clock() - started)
            if remaining <= 0:
                return
            sleep(min(POLL_SECONDS, remaining))
        # A non-advancing clock or exhausted cap cannot certify quiet coverage.
        yield {'available': False, 'transcoders': []}
    return summarize_known_transcoders(snapshots())
