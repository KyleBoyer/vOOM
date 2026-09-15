"""Default-off scalar-only MTP progress. No arrays, text, token IDs or RNG."""
import json
import math
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

    def record(self, emitted, sweeps, proposed, accepted, *, final=False,
               plain_seconds=0.0, plain_sweeps=0, draft_seconds=0.0,
               verifier_seconds=0.0, speculative_rounds=0,
               rollback_seconds=0.0, adaptive_disabled=False):
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
            durations = [float(x) for x in (plain_seconds, draft_seconds,
                                            verifier_seconds, rollback_seconds)]
            if any(not math.isfinite(x) or x < 0 for x in durations):
                raise ValueError('invalid diagnostic duration')
            baseline = (durations[0] / plain_sweeps if plain_sweeps > 0 else None)
            row['costs'] = dict(
                schema='voom.mtp-progress-costs.v1',
                plain_seconds=durations[0], plain_timed_sweeps=int(plain_sweeps),
                draft_seconds=durations[1], verifier_seconds=durations[2],
                rollback_seconds=durations[3],
                speculative_rounds=int(speculative_rounds),
                adaptive_disabled=bool(adaptive_disabled),
                measured_plain_seconds_per_token=baseline,
                estimated_plain_equivalent_seconds=(
                    max(0, emitted - 1) * baseline if baseline is not None else None),
                estimated_net_seconds_including_overhead=(
                    max(0, emitted - 1) * baseline - (now - self.started)
                    if baseline is not None else None),
                scope='same-request timed plain sweeps only; extrapolated counterfactual, not an A/B speed proof; verifier and rollback timers may overlap')
            print(PREFIX + json.dumps(row, sort_keys=True, allow_nan=False), flush=True)
        except (OSError, ValueError, TypeError, OverflowError):
            # Diagnostic output failure must not authorize a retry or alter tokens.
            self.failed = True
        self.last_reported = now
