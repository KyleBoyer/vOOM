"""F68: peak-budget adaptive compute chunks.

The measured 4,096-token chunk optimum (docs/benchmark_results.md, "F60
chunked prefill AS a memory-transient fix") belongs to Qwen2.5-1.5B, not
GLM or any other architecture — its per-position compute-scratch cost
depends on hidden_size/intermediate_size/head count, which vary a lot
across models. Rather than hard-code an architecture-specific constant,
learn a safe chunk size ONLINE from observed peak-memory data for the
model actually being served.

Fits a conservative affine envelope from real (chunk_size, peak,
active_before) observations:

    P(C) <= B + alpha*C + beta
    C_next = floor((M_safe - B_next - margin - beta) / alpha)

`mx.get_active_memory()` already includes resident KV. Subtracting a separate
`kv_before` from both the measured delta and the future budget double-counts KV
and makes the fit increasingly wrong with context length. The argument remains in
the public method for telemetry/caller compatibility but is not subtracted.

Uses a padded upper-envelope estimate of alpha rather than the raw fitted mean.
This reduces optimism but is not a statistical confidence bound and cannot by
itself certify safety. Grows only after two consecutive safe ("green")
chunks, halves immediately after any governor event or an actual
overshoot. After 3 consecutive bad chunks it freezes at the already-halved
size; it must never restore the original fixed chunk. If size 1 itself produces
a bad event the caller aborts rather than continuing an unbounded allocation.

2026-07-14 refinement (found live on real OLMoE-1B-7B, docs/
benchmark_results.md "OLMoE follow-up"): MoE per-position memory cost
isn't purely a function of position count the way it is for a dense
model — WHICH experts a chunk's positions route to matters, and the
simple affine fit has no visibility into that, so noisy per-chunk
observations pushed the fitted slope around and the chunk size
OSCILLATED (512->1024->512->256->464->514->492->448) instead of
converging, even though safety (never overshooting) held throughout. A
proper fix would add expert-union size as a second regressor, but that
quantity isn't known for the UPCOMING chunk before it runs (routing
depends on the chunk's own tokens) — predicting it would just trade one
uncertain estimate for another. Simpler and more robust: a dead-band.
Small proposed adjustments (within `dead_band` of the current chunk size)
are noise, not signal, and are ignored — chunk size only actually changes
when the proposal clears a meaningful threshold. This directly damps the
observed oscillation (most of those swings were <20% of the chunk size)
without needing to model routing at all.

This is intended to change scheduling only, but different chunk shapes can select
different floating-point kernels/reduction shapes. Therefore every enabled shape
needs F33 block-output plus greedy-token gates; prior finite token agreement is E
evidence, not a structural L0 proof.
"""

from __future__ import annotations


class AdaptiveChunkController:
    def __init__(self, safe_bytes: int, initial_chunk: int, margin_bytes: int = int(1e9),
                dead_band: float = 0.2):
        if safe_bytes <= 0:
            raise ValueError("safe_bytes must be positive")
        self.safe_bytes = int(safe_bytes)
        self.min_safe_bytes = self.safe_bytes
        self.max_safe_bytes = self.safe_bytes
        self.margin = margin_bytes
        self.dead_band = dead_band  # ignore GREEN-path proposals within this fraction of current
        self.chunk = max(1, initial_chunk)
        self._history: list[tuple[int, int]] = []  # (C_j, P_j - active_before_j)
        self._green_streak = 0
        self._bad_streak = 0
        self.failed = False
        self.unsafe_at_minimum = False
        self.events: list[str] = []  # human-readable log of controller decisions

    def next_chunk_size(self) -> int:
        return max(1, self.chunk)

    def update_safe_bytes(self, safe_bytes: int) -> None:
        """Refresh the absolute peak boundary from a live governor sample.

        Cost history stores only incremental chunk memory, so it remains valid
        when system headroom changes.  The next observation uses the new limit
        for both overshoot detection and its following size proposal.
        """
        if safe_bytes <= 0:
            raise ValueError("safe_bytes must be positive")
        self.safe_bytes = int(safe_bytes)
        self.min_safe_bytes = min(self.min_safe_bytes, self.safe_bytes)
        self.max_safe_bytes = max(self.max_safe_bytes, self.safe_bytes)

    def _fit_alpha_beta(self) -> tuple[float, float]:
        """Conservative affine envelope for incremental Metal peak.

        The prior two-point least-squares path claimed an upper-confidence
        bound even though two samples have zero residual degrees of freedom and
        therefore produced ``se=0``. Keep the fit as a signal, but apply a 25%
        minimum slope pad and lift beta until the line covers every observation.
        This is a safety heuristic, not a statistical proof; governor events
        still force immediate halving.
        """
        n = len(self._history)
        if n < 2:
            if n == 1:
                c, y = self._history[0]
                return max(2.0 * max(y, 0) / max(c, 1), 1.0), 0.0
            return 1.0, 0.0
        xs = [c for c, _ in self._history]
        ys = [d for _, d in self._history]
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        var_x = sum((x - mean_x) ** 2 for x in xs)
        if var_x < 1e-9:
            return 1.0, mean_y
        cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        alpha = cov / var_x
        beta = mean_y - alpha * mean_x
        if n > 2:
            resid = [y - (alpha * x + beta) for x, y in zip(xs, ys)]
            resid_var = sum(r * r for r in resid) / (n - 2)
            se_alpha = (resid_var / var_x) ** 0.5
        else:
            se_alpha = 0.0
        alpha_upper = max(alpha + se_alpha, alpha * 1.25, 1e-6)
        # Recompute the intercept so the padded line is an upper envelope of
        # every observed point, including noisy non-monotonic MoE chunks.
        beta_upper = max(0.0, max(y - alpha_upper * x for x, y in self._history))
        return alpha_upper, beta_upper

    def observe(self, chunk_size: int, peak: int, active_before: int, kv_before: int,
               governor_event: bool):
        """Feed back one chunk's real measurements; updates self.chunk for
        the NEXT chunk. No-op once failed (caller should stop asking and use
        its already-reduced frozen chunk size). ``kv_before`` is intentionally
        ignored because it is included in ``active_before``."""
        delta = max(0, peak - active_before)
        self._history.append((chunk_size, delta))

        overshoot = peak > self.safe_bytes
        bad = overshoot or governor_event
        if self.failed:
            if bad and self.chunk <= 1:
                self.unsafe_at_minimum = True
                self.events.append("ABORT: size-1 chunk still caused a bad memory event")
            return
        if bad:
            self._bad_streak += 1
            self._green_streak = 0
            old = self.chunk
            self.chunk = max(1, int(self.chunk * 0.5))
            self.events.append(
                f"BAD (overshoot={overshoot}, governor_event={governor_event}): "
                f"chunk {old}->{self.chunk}")
            if old <= 1:
                self.unsafe_at_minimum = True
            if self._bad_streak >= 3:
                self.failed = True
                self.events.append(
                    f"FROZEN: 3 consecutive bad chunks; keep reduced size {self.chunk}"
                )
        else:
            self._bad_streak = 0
            self._green_streak += 1
            if self._green_streak >= 2:
                self._green_streak = 0
                alpha, beta = self._fit_alpha_beta()
                budget = self.safe_bytes - active_before - self.margin - beta
                proposed = int(budget / alpha) if alpha > 0 else self.chunk
                proposed = max(1, proposed)
                old = self.chunk
                new_chunk = min(proposed, self.chunk * 2)  # clamp growth to 2x/step
                new_chunk = max(new_chunk, int(self.chunk * 0.5))  # clamp shrink to 0.5x/step
                new_chunk = max(1, new_chunk)
                # Dead-band: a proposal within `dead_band` of the current chunk is
                # noise (routing-dependent variance on MoE models), not signal --
                # ignore it rather than churn the chunk size back and forth.
                if abs(new_chunk - old) <= old * self.dead_band:
                    self.events.append(
                        f"GREEN x2: proposal {old}->{new_chunk} within dead_band, ignored "
                        f"(alpha={alpha:.1f} beta={beta / 1e6:.1f}MB)")
                else:
                    self.chunk = new_chunk
                    self.events.append(f"GREEN x2: chunk {old}->{self.chunk} (alpha={alpha:.1f} beta={beta / 1e6:.1f}MB)")
