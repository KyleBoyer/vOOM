"""Token sampling for the local OpenAI/Anthropic-compatible runtime.

Greedy remains the default so existing correctness gates and model IDs retain
their deterministic behavior. Explicit temperature/top-p/top-k requests use a
real MLX categorical sampler; a request seed resets the MLX RNG once at the
start of generation, not once per token.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    seed: int | None = None
    repetition_penalty: float = 1.0

    def __post_init__(self) -> None:
        if (isinstance(self.temperature, bool)
                or not isinstance(self.temperature, (int, float))
                or not math.isfinite(float(self.temperature))
                or float(self.temperature) < 0):
            raise ValueError("temperature must be a finite number >= 0")
        if (isinstance(self.top_p, bool)
                or not isinstance(self.top_p, (int, float))
                or not math.isfinite(float(self.top_p))
                or not 0 <= float(self.top_p) <= 1):
            raise ValueError("top_p must be a finite number between 0 and 1")
        if (isinstance(self.top_k, bool) or not isinstance(self.top_k, int)
                or self.top_k < 0):
            raise ValueError("top_k must be a non-negative integer")
        if (self.seed is not None
                and (isinstance(self.seed, bool) or not isinstance(self.seed, int)
                     or not 0 <= self.seed < 2 ** 64)):
            raise ValueError("seed must be an unsigned 64-bit integer or null")
        if (isinstance(self.repetition_penalty, bool)
                or not isinstance(self.repetition_penalty, (int, float))
                or not math.isfinite(float(self.repetition_penalty))
                or float(self.repetition_penalty) <= 0):
            raise ValueError("repetition_penalty must be a finite number > 0")

    @property
    def is_greedy(self) -> bool:
        # A one-candidate filter is deterministically argmax regardless of the
        # requested temperature. top_p=0 similarly keeps the first sorted token.
        return self.temperature == 0 or self.top_k == 1 or self.top_p == 0

    @property
    def profile(self) -> str:
        return "greedy" if self.is_greedy else "categorical"

    def seed_rng(self) -> None:
        if self.seed is not None:
            mx.random.seed(self.seed)


def greedy(logits: mx.array) -> int:
    """Compatibility helper retained for exact speculative paths."""
    return int(mx.argmax(logits))


def _apply_repetition_penalty(
    values: mx.array, history, penalty: float) -> mx.array:
    """HF-style repetition penalty: `logit / penalty` if positive, `logit *
    penalty` if negative (an already-negative logit made MORE negative by a
    penalty > 1, since dividing a negative number would move it toward zero
    -- the wrong direction). `penalty > 1` discourages repeats; `penalty ==
    1` (the default) is a no-op and short-circuits before this is ever
    called, so existing byte-identical proofs are unaffected."""
    if not history:
        return values
    vocab_size = values.size
    unique = {int(t) for t in history}
    valid = sorted(i for i in unique if 0 <= i < vocab_size)
    if not valid:
        return values
    ids = mx.array(valid)
    selected = values[ids].astype(mx.float32)
    penalized = mx.where(selected > 0, selected / penalty, selected * penalty)
    return values.astype(mx.float32).at[ids].add(penalized - selected)


def filtered_probabilities(
    logits: mx.array,
    params: SamplingParams | None = None,
    history=None,
) -> mx.array:
    """Return the exact categorical distribution selected by ``params``.

    Unlike :func:`sample`, this keeps a full-vocabulary probability vector.
    That extra materialization is useful for exact speculative rejection:
    acceptance needs both ``p(token)`` and ``q(token)``, while a rejection
    must sample from ``normalize(max(p - q, 0))``.  Greedy requests return a
    one-hot vector so the same verifier can cover temperatures 0 through 1
    without a separate, subtly different filtering implementation.
    """
    params = params or SamplingParams()
    values = logits.reshape(-1)
    if values.size == 0:
        raise ValueError("cannot build a distribution from empty logits")
    if params.repetition_penalty != 1.0:
        values = _apply_repetition_penalty(
            values, history, float(params.repetition_penalty))
    if params.is_greedy:
        winner = mx.argmax(values)
        return mx.zeros(values.shape, dtype=mx.float32).at[winner].add(1.0)

    values = values.astype(mx.float32) / float(params.temperature)
    keep = mx.ones(values.shape, dtype=mx.bool_)
    if params.top_k and params.top_k < values.size:
        k = int(params.top_k)
        partition = mx.argpartition(values, kth=values.size - k)
        selected = partition[-k:]
        keep = mx.zeros(values.shape, dtype=mx.bool_).at[selected].add(True)

    filtered = mx.where(keep, values, float("-inf"))
    if params.top_p < 1:
        order = mx.argsort(filtered)[::-1]
        sorted_values = filtered[order]
        sorted_probabilities = mx.softmax(sorted_values)
        remove = mx.cumsum(sorted_probabilities) > float(params.top_p)
        # Keep the first token that crosses the nucleus threshold, matching
        # sample() and the serving path's established convention.
        remove = mx.concatenate([mx.array([False]), remove[:-1]])
        kept_sorted = mx.logical_not(remove)
        nucleus_keep = mx.zeros(
            values.shape, dtype=mx.bool_
        ).at[order].add(kept_sorted)
        filtered = mx.where(mx.logical_and(keep, nucleus_keep),
                            values, float("-inf"))
    return mx.softmax(filtered)


def sample_probabilities(probabilities: mx.array) -> int:
    """Sample a normalized full-vocabulary probability vector."""
    probabilities = probabilities.reshape(-1).astype(mx.float32)
    if probabilities.size == 0:
        raise ValueError("cannot sample from an empty probability vector")
    total = mx.sum(probabilities)
    mx.eval(total)
    total_value = float(total.item())
    if not math.isfinite(total_value) or total_value <= 0:
        raise ValueError("probability vector must have positive finite mass")
    normalized = probabilities / total
    return int(mx.random.categorical(mx.log(normalized)))


def speculative_residual_probabilities(
    target: mx.array, draft: mx.array
) -> mx.array:
    """Distribution used after a speculative rejection.

    The positive-part residual is the exact Leviathan rejection correction.
    A zero residual can occur only through finite-precision equality; in that
    case falling back to ``target`` remains a valid, normalized target draw.
    """
    target = target.reshape(-1).astype(mx.float32)
    draft = draft.reshape(-1).astype(mx.float32)
    if target.shape != draft.shape:
        raise ValueError("target and draft distributions must have equal shape")
    residual = mx.maximum(target - draft, 0.0)
    mass = mx.sum(residual)
    mx.eval(mass)
    value = float(mass.item())
    return target if not math.isfinite(value) or value <= 0 else residual / mass


def sample(logits: mx.array, params: SamplingParams | None = None,
          history=None) -> int:
    """Sample one token from a rank-1 (or flattenable) logits vector.

    `history`: token ids already emitted this request (prompt + generated
    so far) -- only consulted when `params.repetition_penalty != 1.0`.
    Filtering is applied before categorical sampling. When top-k is active,
    top-p sorts only those candidates rather than the whole vocabulary.
    """
    params = params or SamplingParams()
    values = logits.reshape(-1)
    if values.size == 0:
        raise ValueError("cannot sample from empty logits")
    if params.repetition_penalty != 1.0:
        values = _apply_repetition_penalty(
            values, history, float(params.repetition_penalty))
    if params.is_greedy:
        return int(mx.argmax(values))

    values = values.astype(mx.float32) / float(params.temperature)
    indices = None
    if params.top_k and params.top_k < values.size:
        k = int(params.top_k)
        partition = mx.argpartition(values, kth=values.size - k)
        indices = partition[-k:]
        values = values[indices]

    if params.top_p < 1:
        order = mx.argsort(values)[::-1]
        sorted_values = values[order]
        probabilities = mx.softmax(sorted_values)
        remove = mx.cumsum(probabilities) > float(params.top_p)
        # Keep the first token whose probability crosses top_p, matching the
        # standard nucleus-sampling convention and guaranteeing nonempty support.
        remove = mx.concatenate([mx.array([False]), remove[:-1]])
        sorted_values = mx.where(remove, float("-inf"), sorted_values)
        local_index = order[mx.random.categorical(sorted_values)]
    else:
        local_index = mx.random.categorical(values)

    # Compose filtered-index remapping on device and cross the Python boundary
    # once. The former path converted the categorical result and each remapping
    # stage separately, adding fixed MLX evaluation overhead without changing
    # the sampled distribution or RNG consumption.
    return int(indices[local_index] if indices is not None else local_index)
