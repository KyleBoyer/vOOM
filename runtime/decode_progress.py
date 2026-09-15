"""Default-off scalar-only MTP progress. No arrays, text, token IDs or RNG."""
import json
import os
import time

FLAG = 'VMODEL_DECODE_PROGRESS_WITNESS'
PREFIX = '[decode-progress] '


def enabled(environ=None):
    value = (os.environ if environ is None else environ).get(FLAG, '0')
    if value not in ('0', '1'):
        raise ValueError(FLAG + ' must be 0 or 1')
    return value == '1'


class DecodeProgress:
    def __init__(self, started, maximum):
        self.started = started
        self.last_reported = started - 30
        self.maximum = maximum
        self.failed = False

    def record(self, emitted, sweeps, proposed, accepted, *, final=False):
        if self.failed:
            return
        now = time.perf_counter()
        if not final and now - self.last_reported < 30:
            return
        row = dict(schema='voom.mtp-decode-progress.v1', pid=os.getpid(),
            decode_started_monotonic_s=self.started, monotonic_s=now,
            elapsed_seconds=round(now-self.started, 4),
            accepted_output_tokens_including_bootstrap=int(emitted),
            maximum_output_tokens=int(self.maximum),
            target_decode_sweeps=int(sweeps), draft_proposed=int(proposed),
            draft_accepted=int(accepted), terminal_round=bool(final),
            scope='accepted token count includes bootstrap and possible EOS; not request completion or quality proof')
        try:
            print(PREFIX + json.dumps(row, sort_keys=True, allow_nan=False), flush=True)
        except OSError:
            # Diagnostic output failure must not authorize a retry or alter tokens.
            self.failed = True
        self.last_reported = now
