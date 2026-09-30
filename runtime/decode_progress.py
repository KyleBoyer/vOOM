"""Default-off MTP progress; separately opt-in private accepted-token tails."""
import json
import math
import os
import time

FLAG = 'VMODEL_DECODE_PROGRESS_WITNESS'
PRIVATE_FLAG = 'VMODEL_PRIVATE_DECODE_TOKEN_TAIL'
PREFIX = '[decode-progress] '


def enabled(environ=None):
    value = (os.environ if environ is None else environ).get(FLAG, '0')
    if value not in ('0', '1'):
        raise ValueError(FLAG + ' must be 0 or 1')
    return value == '1'


def private_enabled(environ=None):
    env = os.environ if environ is None else environ
    value = env.get(PRIVATE_FLAG, '0')
    if value not in ('0', '1'):
        raise ValueError(PRIVATE_FLAG + ' must be 0 or 1')
    if value == '1' and not enabled(env):
        raise ValueError(PRIVATE_FLAG + ' requires ' + FLAG)
    return value == '1'


class DecodeProgress:
    def __init__(self, started, maximum, *, private_tail_enabled=False):
        self.started = started
        self.last_reported = started - 30
        self.maximum = maximum
        self.failed = False
        self.private_tail_enabled = private_tail_enabled

    def record(self, emitted, sweeps, proposed, accepted, *, final=False,
               plain_seconds=0.0, plain_sweeps=0, draft_seconds=0.0,
               verifier_seconds=0.0, speculative_rounds=0,
               rollback_seconds=0.0, adaptive_disabled=False,
               output_token_ids=None):
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
            if self.private_tail_enabled and output_token_ids is not None:
                tail = list(output_token_ids[-32:])
                if len(output_token_ids) != emitted or any(type(t) is not int or t < 0 for t in tail):
                    raise ValueError('invalid accepted output token tail')
                row['private_output_tail'] = dict(
                    schema='voom.private-accepted-output-tail.v1',
                    start_output_token_index=emitted-len(tail),
                    end_output_token_index_exclusive=emitted, token_ids=tail,
                    sensitivity='PRIVATE generated content; no prompt token IDs')
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
