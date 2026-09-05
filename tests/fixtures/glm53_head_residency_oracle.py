"""Diagnostic-only full-logit residency comparisons on real execution rows.

Extra streamed reads/evaluations deliberately invalidate speed measurements.
No prompt replacement, output truncation, or numerical tolerance is involved.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from runtime import glm_mtp
from runtime.lm_head_stream import StreamedLMHead


def _bits(value):
    mx.eval(value)
    if value.dtype != mx.bfloat16:
        raise AssertionError(f"expected released BF16 logits, got {value.dtype}")
    return np.asarray(value.view(mx.uint16))


class HeadResidencyOracle:
    def __init__(self, target, model_dir: Path):
        if (target.cfg.model_type != "glm5_next"
                or not target.rc.glm53_phase_lm_head):
            raise ValueError("head oracle requires the Flash phase-head candidate")
        self.target = target
        self.head = StreamedLMHead(
            model_dir, target.store.weight_map,
            real_name=target.store._real_name.get("lm_head.weight"))
        self.records = []
        self.checks = 0
        self.logits_compared = 0
        self.oracle_s = 0.0
        self._target_projection = target._final_logits
        self._mtp_projection = glm_mtp._project_mtp_head
        target._final_logits = self._target
        glm_mtp._project_mtp_head = self._mtp

    def _compare(self, kind, hidden, actual, expected):
        left, right = _bits(actual), _bits(expected)
        if not np.array_equal(left, right):
            raise AssertionError(
                f"{kind}: {np.count_nonzero(left != right)} unequal logits")
        self.checks += 1
        self.logits_compared += int(left.size)
        if len(self.records) < 256:
            self.records.append({
                "kind": kind,
                "hidden_shape": list(hidden.shape),
                "hidden_sha256": hashlib.sha256(_bits(hidden).tobytes()).hexdigest(),
                "logits_sha256": hashlib.sha256(left.tobytes()).hexdigest(),
                "logits": int(left.size),
                "array_equal": True,
            })

    def _target(self, hidden, head=None):
        actual = self._target_projection(hidden, head=head)
        mx.eval(actual)
        started = time.perf_counter()
        normalized = mx.fast.rms_norm(
            hidden[:, -1:, :], self.target._norm_w,
            self.target.cfg.rms_norm_eps)
        self._compare("target-streamed-rank3", normalized, actual,
                      self.head.logits(normalized)[0, 0])
        self._compare("target-serial-rank2", normalized, actual,
                      self.head.logits_serial_rows(normalized)[0, 0])
        self.oracle_s += time.perf_counter() - started
        return actual

    def _mtp(self, g, head, *, phase_resident=False):
        actual = self._mtp_projection(g, head, phase_resident=phase_resident)
        mx.eval(actual)
        started = time.perf_counter()
        self._compare("mtp", g, actual, self.head.logits(g)[0, -1])
        self.oracle_s += time.perf_counter() - started
        return actual

    def snapshot(self):
        return {
            "timing_valid_for_speed": False,
            "extra_oracle_s": self.oracle_s,
            "all_logits_equal": self.checks > 0,
            "checks": self.checks,
            "logits_compared": self.logits_compared,
            "records": self.records,
            "dropped_records": max(0, self.checks - len(self.records)),
            "head_shape": [self.head.vocab, self.head.hidden],
            "extra_streamed_head": self.head.full_scan_telemetry(),
        }

    def close(self):
        self.target._final_logits = self._target_projection
        glm_mtp._project_mtp_head = self._mtp_projection
        self.head.close()
