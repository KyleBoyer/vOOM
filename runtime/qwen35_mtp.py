"""F94: Qwen3.5/3.6 native MTP (multi-token prediction) as a verified draft
source for speculative decoding.

Checkpoint block under the top-level ``mtp.`` prefix (DeepSeek-V3 style,
``mtp_num_hidden_layers=1``).  Depth 1 remains the safe default.  Explicit
depths 2--4 recurrently reuse the released single physical layer, feeding its
post-block hidden state and preceding proposal into the next step before one
width-(depth+1) exact target verification:

    e   = pre_fc_norm_embedding( embed(token_t) )
    hn  = pre_fc_norm_hidden( h )                  # h = trunk hidden at t-1
    x   = fc( [ e | hn ] )                          # 2*hidden -> hidden
    h'  = mtp.layers.0( x )                          # ordinary decoder layer:
                                                      # full attention (gated
                                                      # QK-RMSNorm, partial
                                                      # RoPE) + dense MLP --
                                                      # NOT DeltaNet, confirmed
                                                      # by tensor shapes
                                                      # matching qwen35.py's
                                                      # _full_attention/_swiglu
                                                      # exactly.
    logit = lm_head( mtp.norm(h') )                  # lm_head is SHARED with
                                                      # the trunk (no separate
                                                      # mtp.lm_head tensor on
                                                      # disk; mtp_use_dedicated_
                                                      # embeddings=False confirms
                                                      # the embedding table is
                                                      # shared too).
    draft = argmax(logit)

Why the recurrent-state rollback problem this blocked on is tractable here:
the TRUNK's DeltaNet/gated-linear-attention layers (KDAStateCache) update
destructively each token, so a rejected draft can't simply be "trimmed" the
way ordinary KV can (see runtime/kv_cache.py::KVCache.trim, which has no
kda_cache branch -- a real, separate, NOT-currently-reachable gap in the
generic suffix_decoding/speculative.py draft paths, guarded against
elsewhere rather than fixed here).  The serial target verifier retains every
strict recurrent prefix in a width-2 or width-3 window.  Rejection or a
terminal token therefore installs the exact accepted KDA endpoint and trims
ordinary target KV plus the MTP layer's own attention KV to the same prefix.

Round structure (matching speculative.py's own "1 + k" verify convention):
depth 1 feeds ``[catchup_token, draft_token]`` and deeper chains feed
``[catchup_token, draft1, ..., draftN]`` through one
layer-major sweep. Every position retains the ordinary one-token arithmetic
shape inside that sweep, so MoE routing never widens into a two-position union
and MXFP4 reductions remain identical to sequential decode. The verifier also
retains the recurrent KDA endpoint after the catchup position. A rejection can
therefore trim ordinary KV to that midpoint and install the retained KDA
endpoint directly; it never restores the pre-round state and re-feeds the
catchup token through another full weight sweep. This is the difference
between speculation whose misses cost an extra streamed pass and speculation
whose misses merely fail to save one.

This module mirrors runtime/glm_mtp.py's structure and safety framing:
greedy drafts require target argmax equality, while stochastic drafts use the
standard exact p/q acceptance ratio and normalized positive-part (p-q)
rejection residual, applied sequentially to every proposal.  This remains a small
dedicated adapter rather than changing the provisional GLM path.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import mlx.core as mx
from . import quant
from .kv_cache import KVCache
from .qwen35 import _full_attention, _moe, _swiglu, final_logits, qwen35_rms_norm
from .sampler import (SamplingParams, filtered_probabilities, sample,
                      sample_probabilities,
                      speculative_residual_probabilities)
from .speculative import ngram_propose


_GREEDY_DRAFT_MARGIN_THRESHOLDS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


def _reranked_head_telemetry_snapshot(target) -> dict[str, int]:
    """Return target-owned sparse-head counters without requiring the feature.

    ``StreamingEngine.generate`` snapshots these counters around an ordinary
    request.  Native MTP bootstraps through that method and then performs its
    own draft/verify projections, so retaining only bootstrap ``path_stats``
    made a real eight-token request report one call even though every later
    serial verifier position also traversed the same row-paged head.
    """
    head = getattr(target, "_lm_head_w", None)
    snapshot = getattr(head, "telemetry_snapshot", None)
    if not callable(snapshot):
        return {}
    return {key: int(value) for key, value in snapshot().items()}


def _reranked_head_telemetry_delta(
    before: Mapping[str, int], after: Mapping[str, int],
) -> dict[str, int]:
    return {
        key: max(0, int(value) - int(before.get(key, 0)))
        for key, value in after.items()
    }


def _accumulate_reranked_head_telemetry(
    totals: dict[str, int], before: Mapping[str, int],
    after: Mapping[str, int],
) -> None:
    for key, value in _reranked_head_telemetry_delta(before, after).items():
        totals[key] = totals.get(key, 0) + value


def _publish_reranked_head_telemetry(
    path_stats: dict, prefix: str, values: Mapping[str, int],
) -> None:
    if not values:
        return
    for key, value in values.items():
        path_stats[f"{prefix}{key}"] = int(value)
    probes = int(values.get("candidate_recall_probes", 0))
    hits = int(values.get("candidate_recall_hits", 0))
    path_stats[f"{prefix}candidate_recall"] = (
        hits / probes if probes else None)


def _authoritative_target_logits(
    target, logits: mx.array, constraint, position: int,
) -> mx.array:
    """Apply a sequential constraint before sparse-head candidate selection.

    A verifier window projects every row up front, but grammar state advances
    only after each preceding proposal is accepted or corrected.  Therefore a
    constrained row must be reranked lazily from its matching hidden position;
    masking the already-shortlisted unrestricted row can leave no legal token.
    Plain/fake targets retain the historical post-projection mask fallback.
    """
    if constraint is None:
        return logits
    constraint_logits = getattr(target, "_constraint_logits", None)
    hidden_window = getattr(target, "_h_window", None)
    if callable(constraint_logits) and hidden_window is not None:
        width = int(hidden_window.shape[1])
        selected = int(position)
        if selected < 0:
            selected += width
        if not 0 <= selected < width:
            raise RuntimeError(
                f"authoritative target position {position} is outside "
                f"hidden window width {width}")
        hidden = hidden_window[:, selected:selected + 1, :]
        return constraint_logits(logits, constraint, hidden=hidden)
    if callable(constraint_logits):
        from .quant import RerankedQHead

        if isinstance(getattr(target, "_lm_head_w", None), RerankedQHead):
            raise RuntimeError(
                "constrained row-paged target decision lacks its matching "
                "verifier hidden position")
    return constraint.mask_logits(logits)


def _authoritative_target_logits_from_hidden(
    target, logits: mx.array, constraint, hidden: mx.array,
) -> mx.array:
    """Apply the current grammar to one explicit tree-node hidden row."""
    if constraint is None:
        return logits
    constraint_logits = getattr(target, "_constraint_logits", None)
    if callable(constraint_logits):
        return constraint_logits(logits, constraint, hidden=hidden)
    return constraint.mask_logits(logits)


@dataclass(frozen=True)
class ProposalQPolicy:
    """A sparse proposal distribution evaluated by the offline replay tool.

    ``temperature`` reshapes the captured draft probabilities/logits on the
    selected top-k support. ``rank`` ignores their magnitudes and assigns a
    power-law mass by draft rank.  These policies affect q only; the exact
    target verifier remains authoritative.
    """

    kind: str
    top_k: int
    temperature: float = 1.0
    rank_power: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in {"flat", "temperature", "rank"}:
            raise ValueError(f"unsupported proposal-q policy kind {self.kind!r}")
        if isinstance(self.top_k, bool) or self.top_k <= 0:
            raise ValueError("proposal-q top_k must be positive")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("proposal-q temperature must be finite and positive")
        if not math.isfinite(self.rank_power) or self.rank_power < 0:
            raise ValueError("proposal-q rank_power must be finite and non-negative")

    @property
    def name(self) -> str:
        if self.kind == "flat":
            return f"flat-k{self.top_k}"
        if self.kind == "temperature":
            return f"temperature-k{self.top_k}-t{self.temperature:g}"
        return f"rank-k{self.top_k}-p{self.rank_power:g}"

    def as_dict(self) -> dict:
        result = {"name": self.name, "kind": self.kind, "top_k": self.top_k}
        if self.kind == "temperature":
            result["temperature"] = self.temperature
        elif self.kind == "rank":
            result["rank_power"] = self.rank_power
        return result


@dataclass(frozen=True)
class ProposalQReplayRow:
    """Sparse target/draft evidence for one already-verified MTP round.

    Target probability is needed only on q's support because q is zero
    elsewhere and therefore ``min(p, q)`` contributes zero there.  Capturing
    16 ranked draft tokens is sufficient to replay every default policy,
    without serializing a 248k-wide vocabulary row.
    """

    draft_token_ids: tuple[int, ...]
    draft_values: tuple[float, ...]
    target_probabilities: tuple[float, ...]
    score_kind: str = "probabilities"
    weight: float = 1.0

    def __post_init__(self) -> None:
        length = len(self.draft_values)
        if length == 0:
            raise ValueError("proposal-q replay row has empty draft support")
        if len(self.draft_token_ids) != length:
            raise ValueError("draft_token_ids and draft values differ in length")
        if len(self.target_probabilities) != length:
            raise ValueError("target probabilities and draft values differ in length")
        if len(set(self.draft_token_ids)) != length:
            raise ValueError("proposal-q replay row contains duplicate token ids")
        if self.score_kind not in {"probabilities", "logits"}:
            raise ValueError("score_kind must be 'probabilities' or 'logits'")
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("proposal-q replay row weight must be finite and positive")
        if not all(math.isfinite(value) for value in self.draft_values):
            raise ValueError("proposal-q draft values must be finite")
        if self.score_kind == "probabilities" and (
            any(value < 0 for value in self.draft_values)
            or sum(self.draft_values) <= 0
        ):
            raise ValueError("captured draft probabilities must contain positive mass")
        if any(
            not math.isfinite(value) or value < 0 or value > 1
            for value in self.target_probabilities
        ):
            raise ValueError("target probabilities must be finite values in [0, 1]")
        if sum(self.target_probabilities) > 1.0001:
            raise ValueError("captured target support mass exceeds one")

    @classmethod
    def from_mapping(cls, value: Mapping) -> "ProposalQReplayRow":
        has_probabilities = "draft_probabilities" in value
        has_logits = "draft_logits" in value
        if has_probabilities == has_logits:
            raise ValueError(
                "each proposal-q replay row needs exactly one of "
                "draft_probabilities or draft_logits")
        key = "draft_probabilities" if has_probabilities else "draft_logits"
        draft_values = tuple(float(item) for item in value[key])
        token_ids = tuple(int(item) for item in value.get(
            "draft_token_ids", range(len(draft_values))))
        return cls(
            draft_token_ids=token_ids,
            draft_values=draft_values,
            target_probabilities=tuple(
                float(item) for item in value["target_probabilities"]),
            score_kind=("probabilities" if has_probabilities else "logits"),
            weight=float(value.get("weight", 1.0)),
        )


def default_proposal_q_policies(
    top_ks: Sequence[int] = (1, 2, 4, 8, 16),
    temperatures: Sequence[float] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0),
    rank_powers: Sequence[float] = (0.5, 1.0, 2.0),
) -> tuple[ProposalQPolicy, ...]:
    """Return the content-independent policy grid used by offline replay."""
    policies: list[ProposalQPolicy] = []
    for top_k in top_ks:
        policies.append(ProposalQPolicy("flat", int(top_k)))
        if int(top_k) == 1:
            continue
        policies.extend(
            ProposalQPolicy("temperature", int(top_k), temperature=float(value))
            for value in temperatures
        )
        policies.extend(
            ProposalQPolicy("rank", int(top_k), rank_power=float(value))
            for value in rank_powers
        )
    if not policies:
        raise ValueError("proposal-q policy grid is empty")
    return tuple(policies)


def proposal_q_distribution(
    row: ProposalQReplayRow, policy: ProposalQPolicy,
) -> tuple[float, ...]:
    """Return q aligned with ``row`` while touching only its sparse support."""
    ranked = sorted(
        range(len(row.draft_values)),
        key=lambda index: (-row.draft_values[index], index),
    )
    selected = ranked[:min(policy.top_k, len(ranked))]
    masses: list[float]
    if policy.kind == "flat":
        masses = [1.0] * len(selected)
    elif policy.kind == "rank":
        masses = [
            1.0 / ((rank + 1) ** policy.rank_power)
            for rank in range(len(selected))
        ]
    elif row.score_kind == "logits":
        maximum = max(row.draft_values[index] for index in selected)
        masses = [
            math.exp((row.draft_values[index] - maximum) / policy.temperature)
            for index in selected
        ]
    else:
        exponent = 1.0 / policy.temperature
        masses = [row.draft_values[index] ** exponent for index in selected]
    total = sum(masses)
    if not math.isfinite(total) or total <= 0:
        masses = [1.0] * len(selected)
        total = float(len(selected))
    q = [0.0] * len(row.draft_values)
    for index, mass in zip(selected, masses, strict=True):
        q[index] = mass / total
    return tuple(q)


def proposal_q_overlap(
    row: ProposalQReplayRow, policy: ProposalQPolicy,
) -> float:
    """Expected exact Leviathan acceptance ``sum_i min(p_i, q_i)``."""
    q = proposal_q_distribution(row, policy)
    return sum(min(p_value, q_value) for p_value, q_value in zip(
        row.target_probabilities, q, strict=True))


def evaluate_proposal_q(
    rows: Iterable[ProposalQReplayRow],
    policy: ProposalQPolicy,
    *,
    target_sweep_bytes: int = 0,
    draft_sweep_bytes: int = 0,
) -> dict:
    """Evaluate overlap plus I/O projections for a one-draft exact verifier."""
    rows = tuple(rows)
    if not rows:
        raise ValueError("proposal-q evaluation needs at least one replay row")
    if isinstance(target_sweep_bytes, bool) or target_sweep_bytes < 0:
        raise ValueError("target_sweep_bytes must be non-negative")
    if isinstance(draft_sweep_bytes, bool) or draft_sweep_bytes < 0:
        raise ValueError("draft_sweep_bytes must be non-negative")
    total_weight = sum(row.weight for row in rows)
    overlap = sum(
        row.weight * proposal_q_overlap(row, policy) for row in rows
    ) / total_weight
    emitted_per_sweep = 1.0 + overlap
    result = {
        "policy": policy.as_dict(),
        "rows": len(rows),
        "total_weight": total_weight,
        "expected_acceptance": overlap,
        "expected_emitted_tokens_per_target_sweep": emitted_per_sweep,
        "projected_target_sweeps_per_1000_output_tokens": (
            1000.0 / emitted_per_sweep),
        "projected_target_sweep_savings_fraction": (
            1.0 - 1.0 / emitted_per_sweep),
    }
    if target_sweep_bytes or draft_sweep_bytes:
        round_bytes = int(target_sweep_bytes) + int(draft_sweep_bytes)
        bytes_per_output = round_bytes / emitted_per_sweep
        result.update({
            "target_sweep_bytes": int(target_sweep_bytes),
            "draft_sweep_bytes": int(draft_sweep_bytes),
            "projected_bytes_per_target_sweep": round_bytes,
            "projected_bytes_per_output_token": bytes_per_output,
            "projected_total_bytes_per_1000_output_tokens": (
                1000.0 * bytes_per_output),
            "projected_byte_speedup_vs_plain_target": (
                int(target_sweep_bytes) / bytes_per_output
                if target_sweep_bytes and bytes_per_output else 0.0),
        })
    return result


def calibrate_proposal_q(
    calibration_rows: Iterable[ProposalQReplayRow],
    validation_rows: Iterable[ProposalQReplayRow],
    *,
    policies: Sequence[ProposalQPolicy] | None = None,
    target_sweep_bytes: int = 0,
    draft_sweep_bytes: int = 0,
) -> dict:
    """Select q on calibration rows and report validation without leakage."""
    calibration_rows = tuple(calibration_rows)
    validation_rows = tuple(validation_rows)
    if not calibration_rows or not validation_rows:
        raise ValueError(
            "proposal-q calibration and validation sets must both be non-empty")
    policies = tuple(policies or default_proposal_q_policies())
    if not policies:
        raise ValueError("proposal-q calibration policy grid is empty")
    candidates = []
    selected_index = 0
    selected_overlap = -1.0
    for index, policy in enumerate(policies):
        calibration = evaluate_proposal_q(
            calibration_rows,
            policy,
            target_sweep_bytes=target_sweep_bytes,
            draft_sweep_bytes=draft_sweep_bytes,
        )
        validation = evaluate_proposal_q(
            validation_rows,
            policy,
            target_sweep_bytes=target_sweep_bytes,
            draft_sweep_bytes=draft_sweep_bytes,
        )
        candidates.append({
            "policy": policy.as_dict(),
            "calibration": calibration,
            "validation": validation,
        })
        candidate_overlap = calibration["expected_acceptance"]
        # Strict greater-than preserves policy-grid order on ties and never
        # consults validation evidence to make the selection.
        if candidate_overlap > selected_overlap:
            selected_index = index
            selected_overlap = candidate_overlap
    selected = candidates[selected_index]
    flat_k4 = next(
        (candidate for candidate in candidates
         if candidate["policy"]["name"] == "flat-k4"),
        None,
    )
    result = {
        "schema_version": 1,
        "selection_split": "calibration",
        "validation_used_for_selection": False,
        "exact_target_distribution": True,
        "selected": selected,
        "candidates": candidates,
    }
    if flat_k4 is not None:
        result["flat_k4_baseline"] = flat_k4
        result["selected_validation_overlap_gain_vs_flat_k4"] = (
            selected["validation"]["expected_acceptance"]
            - flat_k4["validation"]["expected_acceptance"])
    return result


def proposal_q_replay_record(
    draft_probabilities: mx.array,
    target_probabilities: mx.array,
    max_rank: int = 16,
    **metadata,
) -> dict:
    """Serialize only the ranked support needed by offline q calibration."""
    if isinstance(max_rank, bool) or max_rank <= 0:
        raise ValueError("proposal-q replay max_rank must be positive")
    draft = draft_probabilities.reshape(-1)
    target = target_probabilities.reshape(-1)
    if draft.shape != target.shape:
        raise ValueError(
            "proposal-q replay target/draft vocabulary mismatch: "
            f"target={target.shape}, draft={draft.shape}")
    ranked = mx.argsort(draft)[::-1][:min(max_rank, int(draft.size))]
    selected_draft = draft[ranked]
    selected_target = target[ranked]
    mx.eval(ranked, selected_draft, selected_target)
    record = {
        "draft_token_ids": [int(value) for value in ranked.tolist()],
        "draft_probabilities": [
            float(value) for value in selected_draft.tolist()],
        "target_probabilities": [
            float(value) for value in selected_target.tolist()],
    }
    record.update(metadata)
    # Validate the emitted schema immediately; a malformed capture is worse
    # than no capture because it can silently pick a bad q policy offline.
    ProposalQReplayRow.from_mapping(record)
    return record


def _adaptive_mtp_should_disable(
    proposed: int, accepted: int, probe_rounds: int,
) -> bool:
    """True only when a complete initial probe accepted no draft tokens."""
    return proposed >= probe_rounds and accepted == 0


def _adaptive_stochastic_mtp_should_disable(
    proposed: int,
    expected_acceptance_sum: float,
    probe_rounds: int,
    min_expected_acceptance: float = 0.05,
) -> bool:
    """Use exact p/q overlap, not random outcomes, to stop a bad draft."""
    return (
        proposed >= probe_rounds
        and expected_acceptance_sum / proposed < min_expected_acceptance
    )


def _adaptive_break_even_probe_rounds(
    draft_s: float,
    target_s: float,
    minimum_rounds: int,
    maximum_rounds: int = 16,
) -> int:
    """Return a bounded, measured all-reject probe budget.

    A successful depth-1 proposal avoids approximately one future target
    sweep. Therefore an all-reject probe can spend roughly
    ``target_s / draft_s`` rounds before one later acceptance would no longer
    repay its accumulated draft overhead. The bounds make startup timing
    noise harmless and cap regret on genuinely incompatible domains.
    """
    if minimum_rounds <= 0:
        raise ValueError("minimum adaptive probe rounds must be positive")
    if maximum_rounds < minimum_rounds:
        raise ValueError(
            "maximum adaptive probe rounds must be at least the minimum")
    if draft_s <= 0.0 or target_s <= 0.0:
        return minimum_rounds
    measured = int(math.ceil(target_s / draft_s))
    return min(maximum_rounds, max(minimum_rounds, measured))


def _verify_stochastic_mtp_token(
    proposal: int,
    draft_probabilities: mx.array,
    target_logits: mx.array,
    sampling: SamplingParams,
    history: list[int],
) -> tuple[bool, int, mx.array]:
    """Leviathan rejection correction for Qwen's one-token MTP draft.

    Returns ``(accepted, committed_token, target_distribution)``.  The
    target distribution is exposed so callers can retain the exact
    authoritative position telemetry without recomputing its filters.
    """
    target_probabilities = filtered_probabilities(
        target_logits, sampling, history=history).reshape(-1)
    draft_probabilities = draft_probabilities.reshape(-1)
    if target_probabilities.shape != draft_probabilities.shape:
        raise ValueError(
            "Qwen MTP target/draft vocabulary mismatch: "
            f"target={target_probabilities.shape}, "
            f"draft={draft_probabilities.shape}")
    mx.eval(target_probabilities, draft_probabilities)
    p_token = float(target_probabilities[proposal].item())
    q_token = float(draft_probabilities[proposal].item())
    ratio = 1.0 if q_token <= 0 else min(1.0, p_token / q_token)
    if float(mx.random.uniform().item()) <= ratio:
        return True, int(proposal), target_probabilities
    replacement = sample_probabilities(
        speculative_residual_probabilities(
            target_probabilities, draft_probabilities))
    return False, replacement, target_probabilities


def _filtered_draft_probabilities(
    draft_logits: mx.array,
    sampling: SamplingParams,
    history: list[int],
    constraint=None,
) -> mx.array:
    """Filter draft logits, tolerating disjoint head/grammar support.

    A sparse released head can have no finite value inside the current grammar
    support.  Falling back to its unconstrained ranking is still exact: any
    illegal proposal has target probability zero and is rejected by the
    authoritative verifier.  This helper is shared by serving q and optional
    offline replay capture so calibration sees the distribution that actually
    supplied the ranks.
    """
    # final_logits preserves the leading singleton batch dimension, whereas
    # the serial target verifier returns a rank-1 row. Normalize both the
    # proposal filter and q to one vocabulary axis before indexing tokens.
    values = draft_logits.reshape(-1)
    authoritative = (
        constraint.mask_logits(values)
        if constraint is not None else values
    )
    calibrated = filtered_probabilities(
        authoritative, sampling, history=history)

    calibrated_mass = mx.sum(calibrated)
    mx.eval(calibrated_mass)
    calibrated_mass_value = float(calibrated_mass.item())
    if (
        (not math.isfinite(calibrated_mass_value)
         or calibrated_mass_value <= 0)
        and constraint is not None
    ):
        calibrated = filtered_probabilities(
            values, sampling, history=history)
    return calibrated.reshape(-1)


def _flat_top_k_probabilities(
    calibrated: mx.array,
    top_k: int,
) -> mx.array:
    """Flatten q over the positive top-k ranks of a draft distribution."""
    calibrated = calibrated.reshape(-1)

    def flattened(probabilities: mx.array) -> tuple[mx.array, mx.array]:
        selected_k = min(top_k, int(probabilities.size))
        ranked = mx.argsort(probabilities)[::-1]
        support = ranked[:selected_k]
        positive = (probabilities[support] > 0).astype(mx.float32)
        return support, positive

    support, positive = flattened(calibrated)
    support_mass = mx.sum(positive)
    mx.eval(support_mass)
    support_mass_value = float(support_mass.item())
    if not math.isfinite(support_mass_value) or support_mass_value <= 0:
        # A numerically empty draft is still not a serving failure. Uniform q
        # is valid for Leviathan verification and will almost certainly be
        # rejected; the adaptive probe then disables the unhelpful drafter.
        # This keeps the target distribution exact while guaranteeing forward
        # progress for unfamiliar head/constraint combinations.
        return mx.full(
            calibrated.shape, 1.0 / int(calibrated.size), dtype=mx.float32)
    # MLX's indexed ``.at[vector].add`` currently treats this one-dimensional
    # scatter like a reduction on this platform (two distinct indices yielded
    # [1, 0, ...], not [0.5, 0.5, ...]). put_along_axis expresses the required
    # one-to-one write and is covered by the distribution test below.
    return mx.put_along_axis(
        mx.zeros(calibrated.shape, dtype=mx.float32),
        support,
        positive / support_mass,
        axis=0,
    )


def _proposal_q_probabilities(
    calibrated: mx.array, policy: ProposalQPolicy,
) -> mx.array:
    """Apply a typed serving q policy to already-filtered draft mass."""
    if policy.kind == "flat":
        return _flat_top_k_probabilities(calibrated, policy.top_k)
    calibrated = calibrated.reshape(-1).astype(mx.float32)
    support_size = min(policy.top_k, int(calibrated.size))
    support = mx.argsort(calibrated)[::-1][:support_size]
    if policy.kind == "temperature":
        selected = calibrated[support]
        masses = mx.power(
            mx.maximum(selected, mx.array(0.0, dtype=mx.float32)),
            1.0 / policy.temperature,
        )
    else:
        masses = mx.array([
            1.0 / ((rank + 1) ** policy.rank_power)
            for rank in range(support_size)
        ], dtype=mx.float32)
    total = mx.sum(masses)
    mx.eval(total)
    total_value = float(total.item())
    if not math.isfinite(total_value) or total_value <= 0:
        masses = mx.ones((support_size,), dtype=mx.float32)
        total = mx.array(float(support_size), dtype=mx.float32)
    return mx.put_along_axis(
        mx.zeros(calibrated.shape, dtype=mx.float32),
        support,
        masses / total,
        axis=0,
    )


def _flat_top_k_draft_probabilities(
    draft_logits: mx.array,
    sampling: SamplingParams,
    history: list[int],
    top_k: int,
    constraint=None,
) -> mx.array:
    """Build a flat top-k q for exact Leviathan verification."""
    calibrated = _filtered_draft_probabilities(
        draft_logits, sampling, history, constraint)
    return _flat_top_k_probabilities(calibrated, top_k)


def _native_mtp_sibling_tree(
    root_token: int,
    logits: mx.array,
    width: int,
):
    """Build one target-verified sibling level from a native MTP row.

    The released MTP layer is evaluated once. Its highest-scoring distinct
    tokens become siblings under the still-unfed target ``root_token``. The
    tree changes proposal scheduling only: the target evaluates every sibling
    and its argmax chooses the sole committed branch.
    """
    from .speculative_tree import SpeculativeTree, validate_tree

    width = int(width)
    if width < 2:
        raise ValueError("native MTP sibling width must be at least two")
    row = logits.reshape(-1)
    vocab = int(row.shape[0])
    if width > vocab:
        raise ValueError(
            f"native MTP sibling width {width} exceeds vocabulary {vocab}")
    candidates = mx.argpartition(-row, kth=width - 1)[:width]
    scores = mx.take(row, candidates)
    candidates = mx.take(candidates, mx.argsort(-scores))
    mx.eval(candidates)
    token_ids = [int(root_token), *(
        int(token) for token in candidates.tolist())]
    children = [{
        token: index for index, token in enumerate(token_ids[1:], start=1)
    }]
    children.extend({} for _ in range(width))
    tree = SpeculativeTree(
        token_ids=tuple(token_ids),
        depths=(0, *(1 for _ in range(width))),
        parents=(-1, *(0 for _ in range(width))),
        children=tuple(children),
    )
    validate_tree(tree)
    return tree


class QwenMTPDrafter:
    _RELEASED_BF16_CACHE_KEY = "qwen35_mtp:released-bf16"
    _PACKED_MXFP4_CACHE_KEY = "qwen35_mtp:proposal-mxfp4-q4-g32"
    _PACKED_MATRIX_NAMES = frozenset({
        "mtp.fc.weight",
        "mtp.layers.0.mlp.down_proj.weight",
        "mtp.layers.0.mlp.gate_proj.weight",
        "mtp.layers.0.mlp.up_proj.weight",
        "mtp.layers.0.self_attn.k_proj.weight",
        "mtp.layers.0.self_attn.o_proj.weight",
        "mtp.layers.0.self_attn.q_proj.weight",
        "mtp.layers.0.self_attn.v_proj.weight",
    })
    _PACKED_NORM_NAMES = frozenset({
        "mtp.layers.0.input_layernorm.weight",
        "mtp.layers.0.post_attention_layernorm.weight",
        "mtp.layers.0.self_attn.k_norm.weight",
        "mtp.layers.0.self_attn.q_norm.weight",
        "mtp.norm.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
    })

    def __init__(self, engine):
        self.engine = engine
        names = engine.store.names_with_prefix("mtp.")
        if not names:
            raise ValueError(
                "QwenMTPDrafter requires a checkpoint with mtp.* weights")
        # MoE checkpoints (e.g. Qwen3.6-35B-A3B) shape the MTP layer's MLP
        # exactly like a trunk MoE layer (mtp.layers.0.mlp.{gate,
        # shared_expert,shared_expert_gate,experts.N.*}); dense checkpoints
        # (Qwen3.6-27B) use a plain SwiGLU (mtp.layers.0.mlp.{gate,up,down}_proj).
        # Real tensor names confirmed directly against both released
        # checkpoints' safetensors indices, not inferred.
        self._page_names = [n for n in names if ".mlp.experts." not in n]
        self.last_cache_prepare_bytes = 0
        self.last_cache_prepare_released_bytes = 0
        if getattr(engine.store, "mtplx_mtp_sidecar", None):
            self.request_weight_representation = "released-bf16"
        else:
            self.request_weight_representation = (
                getattr(engine.store, "mtp_proposal_representation", None)
                or "demand-cache")

    def _weights(self, *, representation: str = "demand-cache") -> dict:
        # Representation is part of the cache identity.  MTPLX replaces the
        # all-MXFP4 artifact's original draft block with an explicitly indexed
        # released-BF16 sidecar; it must neither hit an earlier transformed
        # page nor be admitted under the generic key for later transformed
        # callers.  WeightStore independently makes the sidecar authoritative
        # over stale raw-fast-tier entries with the same logical names.
        if representation == "released-bf16":
            key = self._RELEASED_BF16_CACHE_KEY
            apply_transform = False
        elif representation == "mxfp4-q4-g32":
            key = self._PACKED_MXFP4_CACHE_KEY
            apply_transform = True
        elif representation == "demand-cache":
            key = "qwen35_mtp"
            apply_transform = True
        else:
            raise ValueError(
                f"unsupported Qwen MTP request representation: {representation}")
        return self.engine.cache.get(
            key,
            self._page_names,
            apply_transform=apply_transform,
        )

    def prepare_request_weights(self) -> dict | None:
        """Load an explicit BF16 MTP sidecar for one speculative round.

        WeightCache pinning is permanent and has no symmetric unpin.  A local
        strong reference keeps the dense MTP block resident only while every
        proposal depth in that round is evaluated.  The serving adapter drops
        it before target verification, whose transient reserve otherwise has
        to overlap the whole BF16 sidecar.  Raw BF16 checkpoints and ordinary
        all-quantized artifacts deliberately stay on the existing demand path.
        """
        representation = self.request_weight_representation
        if representation == "demand-cache":
            return None
        if representation == "released-bf16":
            layout = getattr(
                self.engine.store, "_mtplx_mtp_sidecar_layout", {})
            expected_resident = sum(
                int(layout[name][2])
                for name in self._page_names
                if name in layout
            )
        elif representation == "mxfp4-q4-g32":
            expected_resident = int(
                self.engine.store.mlx_quantized_resident_bytes(
                    self._page_names))
            if expected_resident <= 0:
                raise ValueError(
                    "packed Qwen MTP proposal page has incomplete physical "
                    "weight metadata")
        else:
            raise ValueError(
                f"unsupported Qwen MTP request representation: {representation}")
        self.last_cache_prepare_bytes = expected_resident
        cache_before = int(getattr(self.engine.cache, "total_bytes", 0))
        prepare_for = getattr(self.engine.cache, "prepare_for", None)
        if expected_resident and callable(prepare_for):
            # The cache otherwise discovers this 849 MB page only after it has
            # already been materialized, briefly overlapping a full target
            # LRU and delegating the excess to macOS compression/swap. The
            # sidecar header gives an exact size, so shed consumed target pages
            # before the allocation. Pinned head rows remain protected.
            prepare_for(expected_resident)
        self.last_cache_prepare_released_bytes = max(
            0,
            cache_before - int(getattr(
                self.engine.cache, "total_bytes", cache_before)),
        )
        weights = self._weights(representation=representation)
        try:
            if representation == "released-bf16":
                invalid = [
                    f"{name}:{getattr(value, 'dtype', type(value).__name__)}"
                    for name, value in weights.items()
                    if (not isinstance(value, mx.array)
                        or value.dtype != mx.bfloat16)
                ]
                if invalid:
                    raise ValueError(
                        "MTPLX request weights must be plain released BF16 "
                        f"arrays, found {invalid[:3]}")
            else:
                matrices = [weights.get(name) for name in sorted(
                    self._PACKED_MATRIX_NAMES)]
                norms = [weights.get(name) for name in sorted(
                    self._PACKED_NORM_NAMES)]
                if (
                    len(matrices) != 8
                    or any(not isinstance(value, quant.QTensor)
                           for value in matrices)
                    or len(norms) != 7
                    or any(not isinstance(value, mx.array)
                           or value.dtype != mx.bfloat16 for value in norms)
                ):
                    raise ValueError(
                        "packed Qwen MTP proposal page must contain eight "
                        "MXFP4 matrices and seven BF16 norms")
            eval_values = []
            for value in weights.values():
                if isinstance(value, quant.QTensor):
                    eval_values.extend((value.wq, value.scales))
                    if value.biases is not None:
                        eval_values.append(value.biases)
                else:
                    eval_values.append(value)
            mx.eval(*eval_values)
            return weights
        except BaseException:
            # A malformed/cancelled load must not strand the large sidecar in
            # the representation cache before the adapter owns the mapping.
            self.release_request_weights(weights)
            raise

    def release_request_weights(self, weights: dict) -> dict:
        """Drop a round-local BF16 sidecar before target verification.

        Clear the caller-owned mapping before discarding the cache page so no
        Python strong reference defeats the explicit MLX/file-mapping lifetime
        boundary.  ``discard`` also clears device cache, but the explicit
        ``mx.clear_cache`` is retained as a final barrier even if the page was
        already pressure-evicted.
        """
        resident_bytes = sum(
            int(getattr(value, "nbytes", 0)) for value in weights.values())
        weights.clear()
        discarded = False
        key = (
            self._RELEASED_BF16_CACHE_KEY
            if self.request_weight_representation == "released-bf16"
            else self._PACKED_MXFP4_CACHE_KEY)
        try:
            discarded = self.engine.cache.discard(key, self._page_names)
        finally:
            mx.clear_cache()
        return {
            "resident_bytes": resident_bytes,
            "cache_discarded": int(discarded),
        }

    def _get_experts(self, layer: int, expert_ids: list[int],
                      positions: dict[int, list[int]] | None = None) -> dict[int, dict]:
        """engine._get_experts hardcodes 'model.layers.{layer}.{prefix}.' --
        wrong location for mtp.layers.0's experts, so this is a small local
        duplicate (same reasoning as kimi_linear.py's _route_experts: a
        different weight prefix, not different math). No heat/prefetch
        tracking or governor.reserve() -- a single draft touches at most
        topk+1 tiny expert pages, not worth that machinery."""
        items = [
            (f"mtp_expert.{e}",
             self.engine.store.names_with_prefix(f"mtp.layers.0.mlp.experts.{e}."))
            for e in expert_ids
        ]
        pages = self.engine.cache.get_many(items)
        return {e: pages[f"mtp_expert.{e}"] for e in expert_ids}

    def draft_step(
        self, h_last: mx.array, last_token: int, mtp_kv, offset: int,
        weights: dict | None = None,
    ) -> tuple[mx.array, mx.array]:
        """h_last: (1, 1, hidden) trunk hidden (pre final-norm) at position
        offset-1 (i.e. the state that produced last_token). Returns the full
        draft-logit vector and the post-MTP hidden state.  The latter is the
        released recurrent input to the same physical layer at later draft depths.
        `offset` is the ABSOLUTE
        sequence position of last_token (matching the trunk's own kv.offset
        convention, not a decode-session-local counter) -- RoPE inside this
        MTP layer must see real positions or acceptance rate silently
        degrades (never correctness: every draft is exactly re-verified
        against the trunk regardless of how it was positioned). mtp_kv
        accumulates the MTP block's own ordinary attention KV.  The serving
        adapter trims this cache to the accepted input prefix after every
        recurrent proposal chain; keeping a rejected deeper input would not
        change target correctness, but would silently degrade later q."""
        eng = self.engine
        cfg = eng.cfg
        w = weights if weights is not None else self._weights()
        e = eng._embed([last_token])  # (1, 1, hidden), row-paged when enabled
        e = qwen35_rms_norm(e, w["mtp.pre_fc_norm_embedding.weight"], cfg.rms_norm_eps)
        hn = qwen35_rms_norm(h_last, w["mtp.pre_fc_norm_hidden.weight"], cfg.rms_norm_eps)
        x = quant.matmul(mx.concatenate([e, hn], axis=-1), w["mtp.fc.weight"])
        # Standard decoder-layer residual wiring (same shape as
        # run_qwen35_block's full_attention branch) -- mtp.layers.0 isn't
        # part of cfg.layer_types, so this reimplements that one block's
        # wiring directly rather than routing through run_qwen35_block's
        # layer_types[layer] dispatch.
        residual = x
        h = qwen35_rms_norm(x, w["mtp.layers.0.input_layernorm.weight"], cfg.rms_norm_eps)
        attn = _full_attention(h, w, "mtp.layers.0", cfg, mtp_kv, 0, offset)
        x = residual + attn
        residual = x
        h = qwen35_rms_norm(
            x, w["mtp.layers.0.post_attention_layernorm.weight"], cfg.rms_norm_eps)
        if not cfg.num_experts:
            x = residual + _swiglu(h, w, "mtp.layers.0.mlp")
        else:
            x = residual + _moe(
                h, w, "mtp.layers.0", cfg, 0, self._get_experts)
        shared_head = eng._lm_head_weight()
        # The same physical output head serves both target verification and
        # MTP proposals. Promotion evidence is target-only: mark this one
        # proposal projection so the ranks-only capture cannot count draft
        # hidden states toward its 1,000-position gate.
        with quant.reranked_lm_head_capture_scope(shared_head, "mtp-draft"):
            logits = final_logits(
                x, w["mtp.norm.weight"], shared_head, cfg.rms_norm_eps)
        mx.eval(logits, x)
        # qwen35.final_logits already removes batch/sequence axes and returns
        # one rank-1 vocabulary row. Indexing it again selected only the final
        # vocabulary scalar, degenerating q to a one-token distribution (and
        # making a recurrent chain fail its exact vocabulary-shape check).
        # Preserve the complete released MTP head row for every draft depth.
        if logits.ndim != 1 or int(logits.shape[0]) != int(cfg.vocab_size):
            raise ValueError(
                "Qwen MTP head must return one complete vocabulary row, "
                f"got shape {tuple(logits.shape)} for vocab {cfg.vocab_size}")
        return logits, x

    def draft_logits(
        self, h_last: mx.array, last_token: int, mtp_kv, offset: int,
        weights: dict | None = None,
    ) -> mx.array:
        logits, _hidden = self.draft_step(
            h_last, last_token, mtp_kv, offset, weights)
        return logits

    def draft_token(
        self, h_last: mx.array, last_token: int, mtp_kv, offset: int,
        weights: dict | None = None,
    ) -> int:
        return int(mx.argmax(self.draft_logits(
            h_last, last_token, mtp_kv, offset, weights)))


class QwenMTPSpeculativeEngine:
    """Serving adapter, mirroring SpeculativeEngine's shape: falls back to
    the plain target engine for any request shape the target-exact native-MTP
    verified-draft scheme doesn't cover. Attribute access delegates to the
    target so protocol rendering/telemetry see the real checkpoint,
    tokenizer, config, and execution profile."""

    def __init__(
        self, target, max_prompt_tokens: int = 32768,
        min_output_tokens: int = 32, adaptive_stop: bool = True,
        adaptive_probe_rounds: int = 3, plain_warmup_tokens: int = 3,
        stochastic_draft_top_k: int = 4,
        proposal_replay_top_k: int = 0,
        depth: int = 1,
        proposal_q_policy: ProposalQPolicy | None = None,
        adaptive_reprobe_interval: int = 4,
        ngram_first: bool = False,
        ngram_min_ngram: int = 2,
        ngram_max_ngram: int = 6,
        ngram_max_draft_tokens: int = 4,
        native_tree_width: int = 0,
        grammar_aware_draft: bool = False,
    ):
        if max_prompt_tokens <= 0:
            raise ValueError("max_prompt_tokens must be positive")
        if min_output_tokens <= 1:
            raise ValueError("min_output_tokens must be greater than one")
        if adaptive_probe_rounds <= 0:
            raise ValueError("adaptive_probe_rounds must be positive")
        if plain_warmup_tokens < 0:
            raise ValueError("plain_warmup_tokens must be non-negative")
        if adaptive_reprobe_interval <= 0:
            raise ValueError("adaptive_reprobe_interval must be positive")
        if stochastic_draft_top_k <= 0:
            raise ValueError("stochastic_draft_top_k must be positive")
        if isinstance(proposal_replay_top_k, bool) or proposal_replay_top_k < 0:
            raise ValueError("proposal_replay_top_k must be non-negative")
        if (isinstance(depth, bool) or not isinstance(depth, int)
                or not 1 <= depth <= 4):
            raise ValueError("Qwen MTP depth must be in [1, 4]")
        if not isinstance(ngram_first, bool):
            raise TypeError("Qwen MTP ngram_first must be bool")
        if (isinstance(ngram_min_ngram, bool)
                or not isinstance(ngram_min_ngram, int)
                or ngram_min_ngram <= 0):
            raise ValueError("Qwen MTP minimum n-gram must be positive")
        if (isinstance(ngram_max_ngram, bool)
                or not isinstance(ngram_max_ngram, int)
                or ngram_max_ngram < ngram_min_ngram):
            raise ValueError(
                "Qwen MTP maximum n-gram must be >= minimum n-gram")
        if (isinstance(ngram_max_draft_tokens, bool)
                or not isinstance(ngram_max_draft_tokens, int)
                or not 1 <= ngram_max_draft_tokens <= 4):
            raise ValueError(
                "Qwen MTP n-gram draft width must be in [1, 4]")
        if ngram_first and depth != 1:
            raise ValueError(
                "Qwen MTP n-gram-first cascade currently requires depth 1")
        if (isinstance(native_tree_width, bool)
                or not isinstance(native_tree_width, int)
                or native_tree_width not in (0, 2, 3, 4)):
            raise ValueError(
                "Qwen MTP native tree width must be 0 or in [2, 4]")
        if native_tree_width and depth != 1:
            raise ValueError(
                "Qwen MTP native proposal trees currently require depth 1")
        if native_tree_width and ngram_first:
            raise ValueError(
                "Qwen MTP native proposal trees cannot be combined with "
                "n-gram-first")
        if native_tree_width and (
            getattr(target.cfg, "model_type", None) != "qwen3_5"
            or getattr(target.cfg, "num_experts", 0)
        ):
            raise ValueError(
                "Qwen MTP native proposal trees require dense qwen3_5")
        if not isinstance(grammar_aware_draft, bool):
            raise TypeError("Qwen MTP grammar_aware_draft must be bool")
        self.target = target
        self.drafter = QwenMTPDrafter(target)
        self.max_prompt_tokens = max_prompt_tokens
        self.min_output_tokens = min_output_tokens
        self.adaptive_stop = bool(adaptive_stop)
        self.adaptive_probe_rounds = int(adaptive_probe_rounds)
        self.plain_warmup_tokens = int(plain_warmup_tokens)
        self.adaptive_reprobe_interval = int(adaptive_reprobe_interval)
        self.proposal_q_policy = (
            proposal_q_policy
            if proposal_q_policy is not None
            else ProposalQPolicy("flat", int(stochastic_draft_top_k))
        )
        if not isinstance(self.proposal_q_policy, ProposalQPolicy):
            raise TypeError("proposal_q_policy must be ProposalQPolicy")
        self.stochastic_draft_top_k = self.proposal_q_policy.top_k
        self.proposal_replay_top_k = int(proposal_replay_top_k)
        self.depth = int(depth)
        self.ngram_first = ngram_first
        self.ngram_min_ngram = int(ngram_min_ngram)
        self.ngram_max_ngram = int(ngram_max_ngram)
        self.ngram_max_draft_tokens = int(ngram_max_draft_tokens)
        self.native_tree_width = int(native_tree_width)
        self.grammar_aware_draft = grammar_aware_draft
        weight_identity = self.drafter.request_weight_representation
        self.mtp_engine_identity = (
            f"qwen-mtp-depth{self.depth}-{self.proposal_q_policy.name}"
            + (
                f"-weights-{weight_identity}"
                if weight_identity != "demand-cache" else ""
            )
            + (
                f"-ngram-first-k{self.ngram_max_draft_tokens}"
                if self.ngram_first else ""
            )
            + (
                f"-native-tree-w{self.native_tree_width}"
                if self.native_tree_width else ""
            )
            + ("-grammar-aware-draft" if self.grammar_aware_draft else "")
        )
        if self.depth > 1 and not callable(getattr(
            target, "forward_tokens_serial_positions", None
        )):
            raise ValueError(
                "Qwen MTP depth >1 requires serial-position target verification")
        if (
            getattr(target.cfg, "num_experts", 0)
            and not callable(getattr(
                target, "forward_tokens_serial_positions", None))
        ):
            raise ValueError(
                "MoE Qwen MTP requires serial-position target verification")

    def __getattr__(self, name):
        return getattr(self.target, name)

    def _native_tree_request_eligible(
        self, sampling: SamplingParams | None, constraint,
    ) -> bool:
        return bool(
            self.native_tree_width > 0
            and sampling is not None
            and sampling.is_greedy
            and sampling.repetition_penalty == 1.0
            and (
                constraint is None
                or (
                    callable(getattr(constraint, "mask_logits", None))
                    and callable(getattr(constraint, "accept_token", None))
                )
            )
        )

    def _target_generate(self, reason: str, prompt, max_tokens, on_token,
                          stop, on_progress, sampling, constraint) -> dict:
        kwargs = {"on_token": on_token, "stop": stop, "on_progress": on_progress}
        if sampling is not None:
            kwargs["sampling"] = sampling
        if constraint is not None:
            kwargs["constraint"] = constraint
        # Prefer the target's OWN fail-slow prefill retry (a genuine bound
        # method on the plain StreamingEngine, not a delegated wrapper
        # attribute -- see _engine_generate's docstring in server.py for the
        # bug this class used to trigger via the opposite mistake).
        target_generate = getattr(
            self.target, "generate_with_memory_retry", self.target.generate)
        result = target_generate(prompt, max_tokens, **kwargs)
        path_stats = result.setdefault("path_stats", {})
        path_stats.update({
            "qwen_mtp_enabled": 1,
            "qwen_mtp_used": 0,
            "qwen_mtp_fallback_reason": reason,
            "qwen_mtp_ngram_first_enabled": int(self.ngram_first),
            "qwen_mtp_ngram_first_eligible": int(
                self.ngram_first and sampling is not None
                and sampling.is_greedy),
            "qwen_mtp_native_tree_width": self.native_tree_width,
            "qwen_mtp_native_tree_eligible": int(
                self._native_tree_request_eligible(sampling, constraint)),
            "qwen_mtp_grammar_aware_draft_enabled": int(
                self.grammar_aware_draft),
        })
        return result

    def generate(self, prompt, max_tokens: int = 64, on_token=None,
                 stop=None, on_progress=None,
                 sampling: SamplingParams | None = None,
                 constraint=None) -> dict:
        sampling = sampling or SamplingParams()
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        tgt = self.target
        prepared_ids = getattr(prompt, "token_ids", None)
        ids = (list(prepared_ids) if prepared_ids is not None
               else list(tgt.tokenizer.encode(prompt).ids))
        if len(ids) > self.max_prompt_tokens:
            return self._target_generate(
                "prompt-limit", prompt, max_tokens, on_token, stop,
                on_progress, sampling, constraint)
        if max_tokens < self.min_output_tokens:
            return self._target_generate(
                "short-output-budget", prompt, max_tokens, on_token, stop,
                on_progress, sampling, constraint)
        if (tgt.effective_max_position_embeddings
                and len(ids) + max_tokens > tgt.effective_max_position_embeddings):
            raise ValueError(
                f"prompt({len(ids)})+max_tokens({max_tokens}) exceeds active "
                f"context limit={tgt.effective_max_position_embeddings} "
                f"({tgt.rope_profile})")

        # Let the target's ordinary generation path do exactly one thing:
        # safe prefill plus sampling the first (still-unfed) token. This keeps
        # all of StreamingEngine's chunking, layer-stationary scheduling,
        # memory retries, exact hot-prefix reuse, and request telemetry. The
        # old MTP adapter called forward_tokens(ids, kv) directly, bypassing
        # every one of those mechanisms and therefore had to fall back above
        # an arbitrary 8K prompt limit. After this bootstrap, ``last_kv`` is at
        # the same endpoint the MTP loop expects: prompt fully fed, first token
        # sampled but not fed.
        from .engine import _cache_io_snapshot, _record_cache_io_delta

        request_t0 = time.perf_counter()
        request_cache_before = _cache_io_snapshot(tgt)
        request_rerank_before = _reranked_head_telemetry_snapshot(tgt)
        draft_rerank_totals: dict[str, int] = {}
        eos = set(tgt.cfg.eos_token_ids)
        stop = stop or []
        bootstrap_generate = getattr(
            tgt, "generate_with_memory_retry", tgt.generate)
        bootstrap_kwargs = {
            "on_token": None,
            "stop": stop,
            "on_progress": on_progress,
            "sampling": sampling,
        }
        if constraint is not None:
            bootstrap_kwargs["constraint"] = constraint
        bootstrap = bootstrap_generate(prompt, 1, **bootstrap_kwargs)
        bootstrap_stats = dict(bootstrap.get("path_stats") or {})
        request_weight_representation = str(getattr(
            # Legacy/custom drafters with a prepare/release API represented
            # the released BF16 sidecar before this field existed.
            self.drafter, "request_weight_representation", "released-bf16"))
        bootstrap_stats.update({
            "qwen_mtp_enabled": 1,
            "qwen_mtp_used": 0,
            "qwen_mtp_proposal_weight_representation": (
                request_weight_representation),
            "qwen_mtp_ngram_first_enabled": int(self.ngram_first),
            "qwen_mtp_ngram_first_eligible": int(
                self.ngram_first and sampling.is_greedy),
            "qwen_mtp_native_tree_width": self.native_tree_width,
            "qwen_mtp_native_tree_eligible": int(
                self._native_tree_request_eligible(sampling, constraint)),
            "qwen_mtp_grammar_aware_draft_enabled": int(
                self.grammar_aware_draft),
        })
        if (max_tokens == 1
                or bootstrap.get("termination_reason") != "length"):
            reason = (
                "single-token-budget" if max_tokens == 1
                else f"terminal-first-token:{bootstrap.get('termination_reason')}")
            bootstrap_stats["qwen_mtp_fallback_reason"] = reason
            bootstrap["path_stats"] = bootstrap_stats
            if on_token is not None and bootstrap.get("text"):
                on_token(bootstrap["text"])
            return bootstrap

        kv = getattr(tgt, "last_kv", None)
        if kv is None:
            raise RuntimeError(
                "Qwen MTP bootstrap did not retain its exact target KV endpoint")
        decode_cache_before = _cache_io_snapshot(tgt)
        mtp_kv = KVCache(1)
        proposed = 0
        verified_proposals = 0
        accepted = 0
        speculative_rounds = 0
        full_accept_rounds = 0
        partial_accept_rounds = 0
        rejected_rounds = 0
        accepted_by_step = [0] * self.depth
        verified_by_step = [0] * self.depth
        target_decode_sweeps = 0
        plain_decode_sweeps = 0
        plain_timed_sweeps = 0
        warmup_decode_sweeps = 0
        warmup_remaining = self.plain_warmup_tokens
        adaptive_disabled = False
        adaptive_disabled_ever = False
        adaptive_probe_limit = self.adaptive_probe_rounds
        adaptive_effective_probe_limit = adaptive_probe_limit
        adaptive_window_rounds = 0
        adaptive_window_accepted = 0
        adaptive_window_expected_acceptance = 0.0
        adaptive_cooldown_remaining = 0
        adaptive_cooldown_sweeps = 0
        adaptive_disable_events = 0
        adaptive_recovery_probe = False
        adaptive_recovery_probes = 0
        adaptive_reactivations = 0
        serial_verify_rounds = 0
        kda_endpoint_restores = 0
        refeed_sweeps_saved = 0
        grammar_forced_tokens = 0
        grammar_forced_sweeps = 0
        grammar_masked_draft_tokens = 0
        grammar_masked_draft_rounds = 0
        stochastic_expected_acceptance_sum = 0.0
        stochastic_first_step_expected_acceptance_sum = 0.0
        stochastic_expected_acceptance_by_step = [0.0] * self.depth
        mtp_kv_rollbacks = 0
        target_prefix_rollbacks = 0
        round_outcomes: list[str] = []
        proposal_sources: list[str] = []
        ngram_first_attempts = 0
        ngram_first_matches = 0
        ngram_first_proposed = 0
        ngram_first_accepted = 0
        ngram_first_rejected = 0
        ngram_first_native_draft_bypasses = 0
        ngram_first_accepted_by_step = [0] * self.ngram_max_draft_tokens
        ngram_first_verified_by_step = [0] * self.ngram_max_draft_tokens
        ngram_first_max_proposed_per_round = 0
        native_mtp_proposed = 0
        native_mtp_accepted = 0
        native_mtp_rejected = 0
        native_tree_rounds = 0
        native_tree_hits = 0
        native_tree_misses = 0
        # Index zero is a complete sibling miss; indices 1..width are the
        # MTP-logit ranks selected by the authoritative target root.  This is
        # the offline decision statistic for choosing width 2/3/4 without
        # rerunning a wider tree merely to discover that its last branches are
        # never useful.
        native_tree_selected_rank_counts = [
            0 for _ in range(self.native_tree_width + 1)
        ]
        native_tree_nodes_verified = 0
        native_tree_paths_committed = 0
        native_tree_factor_bytes_peak = 0
        native_tree_factor_commit_s = 0.0
        greedy_target_rank_counts_by_step = [
            [0] * (self.proposal_replay_top_k + 1)
            for _ in range(self.depth)
        ]
        greedy_rescuable_rejections_by_step = [0] * self.depth
        greedy_margin_rank_counts_by_step = [
            [
                [0] * (self.proposal_replay_top_k + 1)
                for _ in range(len(_GREEDY_DRAFT_MARGIN_THRESHOLDS) + 1)
            ]
            for _ in range(self.depth)
        ]
        greedy_round_confidence_records: list[dict] = []
        proposal_replay_records: list[dict] = []
        sidecar_round_loads = 0
        sidecar_round_releases = 0
        sidecar_read_bytes = 0
        sidecar_loaded_resident_bytes = 0
        sidecar_released_resident_bytes = 0
        sidecar_peak_resident_bytes = 0
        sidecar_cache_discards = 0
        sidecar_cache_prepare_calls = 0
        sidecar_cache_prepare_bytes = 0
        sidecar_cache_prepare_released_bytes = 0
        sidecar_load_s = 0.0
        sidecar_release_s = 0.0
        draft_round_s = 0.0
        verifier_round_s = 0.0
        plain_round_s = 0.0
        verifier_input_positions = 0
        verifier_committed_positions = 0
        verifier_rolled_back_positions = 0
        verifier_output_tokens = 0
        verifier_accepted_draft_tokens = 0
        verifier_correction_tokens = 0
        verifier_bonus_tokens = 0
        max_verify_width_observed = 0

        # Invariant (matching speculative.py's documented one): all_tokens =
        # prompt + emitted; catchup_tok = all_tokens[-1] is sampled but not
        # yet fed to kv (kv.offset == len(all_tokens) - 1) until this
        # round's combined forward call feeds it.
        catchup_tok = int(bootstrap["tokens"][0])
        h_last = tgt._h_last
        all_tokens = list(ids) + [catchup_tok]
        emitted = [catchup_tok]
        prefill_s = float(bootstrap.get("prefill_s", 0.0))
        first_token_s = float(bootstrap.get("first_token_s", prefill_s))
        stop_text = (
            bootstrap.get("text")
            if bootstrap.get("termination_reason") == "stop_sequence"
            else None)
        matched_stop_sequence = bootstrap.get("stop_sequence")
        grammar_completed = bool(
            constraint is not None and constraint.completed)

        # StreamingEngine retained a prompt endpoint after the one-token
        # bootstrap. Transfer that slot before mutating its KV; otherwise the
        # slot's token tuple would still describe the prompt while its aliased
        # recurrent state advances through speculative decode. Stable-boundary
        # slots own a separate fork and remain untouched (they are the useful
        # next-turn cache for recurrent chat templates).
        endpoint_slot = None
        hot_slots = getattr(tgt, "_hot_prompt_slots", None)
        if hot_slots is not None:
            for index, slot in enumerate(hot_slots):
                if getattr(slot, "kv", None) is kv:
                    endpoint_slot = hot_slots.pop(index)
                    break
        last_emitted_logits = (
            getattr(endpoint_slot, "logits", None)
            if endpoint_slot is not None else None)

        def _stop_match(text: str):
            matches = [(text.find(value), index, value)
                       for index, value in enumerate(stop)
                       if value and text.find(value) != -1]
            return min(matches) if matches else None

        if stop:
            decoded = tgt.tokenizer.decode(emitted)
            match = _stop_match(decoded)
            if match is not None:
                cut, _order, matched_stop_sequence = match
                stop_text = decoded[:cut]
        stream_decoder = None
        if on_token is not None:
            from .incremental_decode import IncrementalDetokenizer

            stream_decoder = IncrementalDetokenizer(tgt.tokenizer, stop)
            if stop_text is None:
                delta = stream_decoder.push(emitted)
                if delta:
                    on_token(delta)

        def _candidate_terminal(candidate_tokens: list[int]) -> bool:
            if not candidate_tokens:
                return False
            if candidate_tokens[-1] in eos:
                return True
            if len(emitted) + len(candidate_tokens) >= max_tokens:
                return True
            if grammar_completed:
                return True
            if stop:
                return _stop_match(
                    tgt.tokenizer.decode(emitted + candidate_tokens)
                ) is not None
            return False

        decode_t0 = time.perf_counter()
        while (len(emitted) < max_tokens and catchup_tok not in eos
               and not grammar_completed
               and stop_text is None):
            speculative_round = False
            round_rejected = False
            round_verify_width = 0
            round_start_offset = None
            round_start_layer_lengths = None
            round_layer_growth = None
            round_capture_endpoint = False
            round_serial_verify = False
            round_mtp_start_lengths = None
            round_bonus_candidate = False
            round_tree_verification = None
            round_tree_selected_path = None
            round_native_tree = False
            # Grammar-deterministic spans need target state updates, but no
            # target decisions.  Preserve StreamingEngine's jump-forward win
            # inside MTP by folding catchup + the whole forced run into one
            # layer-major sweep, then sample only the first ambiguous token.
            # This is capability-driven for every schema/topic; it contains
            # no request text, tool name, or captured-shape predicate.
            forced_tokens: list[int] = []
            if (
                constraint is not None
                and getattr(
                    getattr(tgt, "rc", None),
                    "grammar_jump_forward_lossy", False)
            ):
                forced_tokens = constraint.forced_run(
                    max(0, max_tokens - len(emitted)),
                    encode=lambda text: tgt.tokenizer.encode(text).ids,
                )
                grammar_completed = bool(constraint.completed)
            if forced_tokens:
                terminal_forced = grammar_completed
                committed_forced: list[int] = []
                for token in forced_tokens:
                    committed_forced.append(int(token))
                    if stop:
                        match = _stop_match(
                            tgt.tokenizer.decode(
                                emitted + committed_forced))
                        if match is not None:
                            terminal_forced = True
                    if (
                        token in eos
                        or len(emitted) + len(committed_forced) >= max_tokens
                    ):
                        terminal_forced = True
                    if terminal_forced:
                        break

                # Retain the final-token-never-fed endpoint on terminal
                # forced spans.  Otherwise feed every forced token so the
                # last position's logits can decide the next free token.
                feed_forced = (
                    committed_forced[:-1]
                    if terminal_forced else committed_forced
                )
                verify_tokens = [catchup_tok] + feed_forced
                serial_verify = callable(getattr(
                    tgt, "forward_tokens_serial_positions", None))
                if serial_verify:
                    forced_logits = tgt.forward_tokens_serial_positions(
                        verify_tokens, kv, capture_kda_endpoints=False)
                else:
                    forced_logits = tgt.forward_tokens(verify_tokens, kv)
                target_decode_sweeps += 1
                plain_decode_sweeps += 1
                grammar_forced_sweeps += 1
                grammar_forced_tokens += len(committed_forced)
                round_outcomes.append(f"F{len(committed_forced)}")
                mx.eval(forced_logits)
                h_last = tgt._h_last
                if terminal_forced:
                    new_tokens = committed_forced
                    new_token_logits = [
                        forced_logits[index]
                        for index in range(len(committed_forced))
                    ]
                    next_catchup_tok = committed_forced[-1]
                else:
                    authoritative_logits = _authoritative_target_logits(
                        tgt, forced_logits[-1], constraint, -1)
                    next_free = sample(
                        authoritative_logits,
                        sampling,
                        history=all_tokens + committed_forced,
                    )
                    constraint.accept_token(next_free)
                    grammar_completed = bool(constraint.completed)
                    new_tokens = committed_forced + [next_free]
                    new_token_logits = [
                        forced_logits[index]
                        for index in range(len(committed_forced))
                    ] + [authoritative_logits]
                    next_catchup_tok = next_free
            elif (
                warmup_remaining
                or (adaptive_disabled and adaptive_cooldown_remaining > 0)
            ):
                plain_started = time.perf_counter()
                plain_logits = tgt.forward_tokens([catchup_tok], kv)
                target_decode_sweeps += 1
                plain_decode_sweeps += 1
                if warmup_remaining:
                    warmup_remaining -= 1
                    warmup_decode_sweeps += 1
                mx.eval(plain_logits)
                authoritative_logits = _authoritative_target_logits(
                    tgt, plain_logits[-1], constraint, -1)
                next_plain = sample(
                    authoritative_logits, sampling, history=all_tokens)
                if constraint is not None:
                    constraint.accept_token(next_plain)
                    grammar_completed = bool(constraint.completed)
                new_tokens = [next_plain]
                new_token_logits = [authoritative_logits]
                h_last = tgt._h_last
                next_catchup_tok = new_tokens[0]
                plain_round_s += time.perf_counter() - plain_started
                plain_timed_sweeps += 1
                if adaptive_disabled and not warmup_remaining:
                    adaptive_cooldown_sweeps += 1
                    adaptive_cooldown_remaining -= 1
                    if adaptive_cooldown_remaining == 0:
                        adaptive_disabled = False
                        adaptive_recovery_probe = True
            else:
                speculative_round = True
                speculative_rounds += 1
                round_accepted_before = accepted
                round_expected_acceptance_before = (
                    stochastic_first_step_expected_acceptance_sum)
                round_start_offset = kv.offset
                layer_lengths_fn = getattr(kv, "layer_lengths", None)
                round_start_layer_lengths = (
                    layer_lengths_fn() if callable(layer_lengths_fn) else None)
                round_mtp_start_lengths = mtp_kv.layer_lengths()
                # GLM's identical prefill-sync convention (glm_mtp.py:53-69,
                # "entry i covers position i") confirms the MTP entry's RoPE
                # position matches h_last's OWN position (round_start_offset-1,
                # the state that produced catchup_tok) -- not catchup_tok's
                # position. Only affects acceptance rate, never correctness.
                draft_tokens: list[int] = []
                draft_probabilities: list[mx.array | None] = []
                draft_rank_probabilities: list[mx.array | None] = []
                draft_ranked_tokens: list[tuple[int, ...] | None] = []
                draft_margin_buckets: list[int | None] = []
                draft_hidden = h_last
                draft_input_token = catchup_tok
                round_mtp_weights: dict | None = None
                ngram_tokens: list[int] = []
                round_proposal_source = "M"
                round_native_tree = self._native_tree_request_eligible(
                    sampling, constraint)
                native_tree = None
                draft_rerank_before = _reranked_head_telemetry_snapshot(tgt)
                draft_started = time.perf_counter()
                try:
                    # A deterministic prompt-lookup hit is a zero-model first
                    # choice.  It still enters the identical authoritative
                    # target verification window below, so rejection and all
                    # hybrid recurrent/KV rollback semantics stay target-owned.
                    # Stochastic requests deliberately bypass this path: an
                    # n-gram lookup does not define the full proposal q needed
                    # for exact p/q acceptance and residual correction.
                    if self.ngram_first and sampling.is_greedy:
                        ngram_first_attempts += 1
                        remaining_outputs = max_tokens - len(emitted)
                        # Leave room for the verifier's authoritative bonus
                        # when possible. With one output slot left, one draft
                        # is still useful and its overfed input is rolled back
                        # by the same final-token-never-fed endpoint rule.
                        ngram_width = min(
                            self.ngram_max_draft_tokens,
                            max(1, remaining_outputs - 1),
                        )
                        ngram_tokens = ngram_propose(
                            all_tokens,
                            ngram_width,
                            self.ngram_max_ngram,
                            self.ngram_min_ngram,
                        )
                        if ngram_tokens:
                            ngram_tokens = [int(token) for token in ngram_tokens]
                            round_proposal_source = "N"
                            ngram_first_matches += 1
                            ngram_first_native_draft_bypasses += 1

                    prepare_request_weights = getattr(
                        self.drafter, "prepare_request_weights", None)
                    read_before = int(getattr(
                        getattr(tgt.cache, "stats", None), "bytes_read", 0))
                    sidecar_load_started = time.perf_counter()
                    round_mtp_weights = (
                        prepare_request_weights()
                        if (not ngram_tokens
                            and callable(prepare_request_weights)) else None
                    )
                    prepared_bytes = (
                        int(getattr(
                            self.drafter, "last_cache_prepare_bytes", 0))
                        if not ngram_tokens else 0
                    )
                    if prepared_bytes:
                        sidecar_cache_prepare_calls += 1
                        sidecar_cache_prepare_bytes += prepared_bytes
                        sidecar_cache_prepare_released_bytes += int(getattr(
                            self.drafter,
                            "last_cache_prepare_released_bytes",
                            0,
                        ))
                    if round_mtp_weights is not None:
                        sidecar_load_s += (
                            time.perf_counter() - sidecar_load_started)
                    if round_mtp_weights is not None:
                        resident_bytes = sum(
                            int(value.nbytes)
                            for value in round_mtp_weights.values()
                        )
                        sidecar_round_loads += 1
                        sidecar_loaded_resident_bytes += resident_bytes
                        sidecar_peak_resident_bytes = max(
                            sidecar_peak_resident_bytes, resident_bytes)
                        read_after = int(getattr(
                            getattr(tgt.cache, "stats", None),
                            "bytes_read", read_before))
                        sidecar_read_bytes += max(0, read_after - read_before)

                    if ngram_tokens:
                        draft_tokens.extend(ngram_tokens)
                        draft_probabilities.extend([None] * len(ngram_tokens))
                        draft_rank_probabilities.extend(
                            [None] * len(ngram_tokens))
                        draft_ranked_tokens.extend([None] * len(ngram_tokens))
                        draft_margin_buckets.extend([None] * len(ngram_tokens))

                    draft_constraint = None
                    if (self.grammar_aware_draft
                            and constraint is not None
                            and sampling.is_greedy
                            and self.depth > 1):
                        fork_constraint = getattr(constraint, "fork", None)
                        if callable(fork_constraint):
                            try:
                                draft_constraint = fork_constraint()
                            except (RuntimeError, TypeError, ValueError):
                                # A custom constraint may expose a partial fork
                                # surface. Step zero remains safe with the
                                # authoritative read-only mask; later steps
                                # simply retain the established unmasked path.
                                draft_constraint = None

                    for step in range(0 if ngram_tokens else self.depth):
                        if self.depth == 1:
                            # Keep the established k=1 mock/API path unchanged.
                            if sampling.is_greedy:
                                if (round_native_tree or (
                                    self.grammar_aware_draft
                                    and constraint is not None
                                ) or self.proposal_replay_top_k):
                                    step_logits = self.drafter.draft_logits(
                                        draft_hidden, draft_input_token, mtp_kv,
                                        round_start_offset - 1,
                                        round_mtp_weights,
                                    )
                                    draft_tok = None
                                else:
                                    draft_tok = self.drafter.draft_token(
                                        draft_hidden, draft_input_token, mtp_kv,
                                        round_start_offset - 1,
                                        round_mtp_weights,
                                    )
                                    step_logits = None
                            else:
                                step_logits = self.drafter.draft_logits(
                                    draft_hidden, draft_input_token, mtp_kv,
                                    round_start_offset - 1,
                                    round_mtp_weights,
                                )
                        else:
                            draft_step = getattr(self.drafter, "draft_step", None)
                            if not callable(draft_step):
                                raise RuntimeError(
                                    "Qwen MTP depth >1 drafter omits draft_step")
                            step_logits, draft_hidden = draft_step(
                                draft_hidden,
                                draft_input_token,
                                mtp_kv,
                                round_start_offset - 1 + step,
                                round_mtp_weights,
                            )
                            draft_tok = None

                        step_rank_probabilities = None
                        step_probabilities = None
                        step_ranked_tokens = None
                        step_margin_bucket = None
                        if sampling.is_greedy:
                            if draft_tok is None:
                                # The target already applies this exact grammar
                                # before accepting/correcting the proposal.  Use
                                # the same current-state mask for the first MTP
                                # proposal so an obviously illegal raw argmax
                                # does not force a full target-sweep rejection.
                                # A forked grammar may advance through earlier
                                # provisional tokens and mask every later draft
                                # step without mutating the authoritative
                                # request grammar. Constraints without a fork
                                # retain the established step-zero-only mask.
                                step_constraint = (
                                    draft_constraint
                                    if draft_constraint is not None else
                                    constraint if step == 0 else None
                                )
                                if (self.grammar_aware_draft
                                        and step_constraint is not None):
                                    step_logits = step_constraint.mask_logits(
                                        step_logits)
                                    grammar_masked_draft_tokens += 1
                                    if step == 0:
                                        grammar_masked_draft_rounds += 1
                                if self.proposal_replay_top_k:
                                    rank_count = min(
                                        self.proposal_replay_top_k,
                                        int(step_logits.size),
                                    )
                                    ranked = mx.argsort(
                                        step_logits.reshape(-1))[::-1][
                                            :rank_count]
                                    mx.eval(ranked)
                                    step_ranked_tokens = tuple(
                                        int(token) for token in ranked.tolist())
                                    if rank_count >= 2:
                                        ranked_scores = mx.take(
                                            step_logits.reshape(-1), ranked[:2])
                                        mx.eval(ranked_scores)
                                        margin = float(
                                            ranked_scores[0].item()
                                            - ranked_scores[1].item())
                                        step_margin_bucket = sum(
                                            margin >= threshold
                                            for threshold in
                                            _GREEDY_DRAFT_MARGIN_THRESHOLDS
                                        )
                                draft_tok = int(mx.argmax(step_logits))
                        else:
                            if step_logits is None:
                                raise RuntimeError(
                                    "stochastic Qwen MTP draft omitted logits")
                            # Later q rows are generated before the target has
                            # accepted the preceding drafts into a mutable
                            # grammar. Conditioning them on the old grammar
                            # state would be wrong; leaving them unconstrained
                            # is still distribution-exact because each
                            # sequential target p row below is authoritative.
                            step_constraint = constraint if step == 0 else None
                            step_rank_probabilities = (
                                _filtered_draft_probabilities(
                                    step_logits,
                                    sampling,
                                    all_tokens + draft_tokens,
                                    step_constraint,
                                )
                            )
                            step_probabilities = _proposal_q_probabilities(
                                step_rank_probabilities,
                                self.proposal_q_policy,
                            )
                            draft_tok = sample_probabilities(step_probabilities)

                        draft_tokens.append(int(draft_tok))
                        draft_probabilities.append(step_probabilities)
                        draft_rank_probabilities.append(step_rank_probabilities)
                        draft_ranked_tokens.append(step_ranked_tokens)
                        draft_margin_buckets.append(step_margin_bucket)
                        draft_input_token = int(draft_tok)
                        if (sampling.is_greedy
                                and draft_constraint is not None):
                            draft_constraint.accept_token(draft_tok)

                    if round_native_tree:
                        if step_logits is None:
                            raise RuntimeError(
                                "native MTP proposal tree omitted draft logits")
                        native_tree = _native_mtp_sibling_tree(
                            catchup_tok,
                            (
                                constraint.mask_logits(step_logits)
                                if constraint is not None else step_logits
                            ),
                            self.native_tree_width,
                        )
                        draft_tokens = list(native_tree.token_ids[1:])
                finally:
                    # The target verifier's memory reserve must never overlap
                    # the released-BF16 MTP sidecar. Clear the caller mapping,
                    # then discard its representation-specific cache page.
                    # Ordinary non-sidecar MTP returns None and keeps its
                    # existing demand-cache behavior unchanged.
                    if round_mtp_weights is not None:
                        weights_to_release = round_mtp_weights
                        round_mtp_weights = None
                        release_request_weights = getattr(
                            self.drafter, "release_request_weights", None)
                        if not callable(release_request_weights):
                            weights_to_release.clear()
                            mx.clear_cache()
                            raise RuntimeError(
                                "BF16 MTP sidecar drafter omits round release")
                        sidecar_release_started = time.perf_counter()
                        release_info = release_request_weights(
                            weights_to_release) or {}
                        sidecar_release_s += (
                            time.perf_counter() - sidecar_release_started)
                        sidecar_round_releases += 1
                        sidecar_released_resident_bytes += int(
                            release_info.get("resident_bytes", 0))
                        sidecar_cache_discards += int(
                            release_info.get("cache_discarded", 0))
                    draft_round_s += time.perf_counter() - draft_started
                _accumulate_reranked_head_telemetry(
                    draft_rerank_totals,
                    draft_rerank_before,
                    _reranked_head_telemetry_snapshot(tgt),
                )

                proposed += len(draft_tokens)
                proposal_sources.append(round_proposal_source)
                if round_proposal_source == "N":
                    ngram_first_proposed += len(draft_tokens)
                    ngram_first_max_proposed_per_round = max(
                        ngram_first_max_proposed_per_round,
                        len(draft_tokens),
                    )
                else:
                    native_mtp_proposed += len(draft_tokens)
                verify_tokens = [catchup_tok] + draft_tokens
                round_verify_width = len(verify_tokens)
                max_verify_width_observed = max(
                    max_verify_width_observed, round_verify_width)
                verifier_started = time.perf_counter()
                if round_native_tree:
                    if native_tree is None:
                        raise RuntimeError(
                            "native MTP sibling verifier omitted its tree")
                    from .qwen35_tree_verify import verify_qwen35_tree

                    round_tree_verification = verify_qwen35_tree(
                        tgt, native_tree, kv)
                    spec_logits = round_tree_verification.logits
                    native_tree_rounds += 1
                    native_tree_nodes_verified += len(native_tree.token_ids)
                    native_tree_factor_bytes_peak = max(
                        native_tree_factor_bytes_peak,
                        round_tree_verification.factors.nbytes(),
                    )
                else:
                    round_serial_verify = callable(getattr(
                        tgt, "forward_tokens_serial_positions", None))
                    round_capture_endpoint = bool(
                        round_serial_verify
                        and getattr(kv, "kda_cache", None) is not None
                    )
                    if round_serial_verify:
                        spec_logits = tgt.forward_tokens_serial_positions(
                            verify_tokens,
                            kv,
                            capture_kda_endpoints=round_capture_endpoint,
                        )
                        serial_verify_rounds += 1
                    else:
                        # Compatibility fallback for old dense adapters. MoE
                        # construction fails closed above when the exact verifier
                        # is unavailable.
                        spec_logits = tgt.forward_tokens(verify_tokens, kv)
                target_decode_sweeps += 1
                if (not round_native_tree
                        and round_start_layer_lengths is not None):
                    serial_end_layer_lengths = kv.layer_lengths()
                    round_layer_growth = tuple(
                        end - start for start, end in zip(
                            round_start_layer_lengths,
                            serial_end_layer_lengths,
                            strict=True,
                        )
                    )
                    if any(
                        growth not in (0, round_verify_width)
                        for growth in round_layer_growth
                    ):
                        raise RuntimeError(
                            "serial Qwen verifier changed an attention layer "
                            "by an unexpected position count: "
                            f"{round_layer_growth}")

                new_tokens = []
                new_token_logits = []
                accepted_prefix = 0
                round_target_ranks: list[int] = []
                if round_native_tree:
                    root_logits = _authoritative_target_logits_from_hidden(
                        tgt,
                        spec_logits[0],
                        constraint,
                        round_tree_verification.hidden_nodes[0],
                    )
                    target_root = sample(
                        root_logits, sampling, history=all_tokens)
                    selected_node = native_tree.children[0].get(target_root)
                    selected_rank = (
                        int(selected_node) if selected_node is not None else 0
                    )
                    native_tree_selected_rank_counts[selected_rank] += 1
                    accepted_prefix = int(selected_node is not None)
                    round_tree_selected_path = (
                        (0, int(selected_node))
                        if selected_node is not None else (0,)
                    )
                    verified_proposals += len(draft_tokens)
                    verified_by_step[0] += len(draft_tokens)
                    round_rejected = accepted_prefix == 0
                    if accepted_prefix:
                        accepted += 1
                        accepted_by_step[0] += 1
                        native_mtp_accepted += 1
                        native_tree_hits += 1
                        if constraint is not None:
                            constraint.accept_token(target_root)
                            grammar_completed = bool(constraint.completed)
                        new_tokens = [int(target_root)]
                        new_token_logits = [root_logits]
                        if not _candidate_terminal(new_tokens):
                            bonus_logits = (
                                _authoritative_target_logits_from_hidden(
                                    tgt,
                                    spec_logits[selected_node],
                                    constraint,
                                    round_tree_verification.hidden_nodes[
                                        selected_node],
                                )
                            )
                            bonus_tok = sample(
                                bonus_logits,
                                sampling,
                                history=all_tokens + new_tokens,
                            )
                            if constraint is not None:
                                constraint.accept_token(bonus_tok)
                                grammar_completed = bool(constraint.completed)
                            new_tokens.append(int(bonus_tok))
                            new_token_logits.append(bonus_logits)
                            round_bonus_candidate = True
                        full_accept_rounds += 1
                        round_outcomes.append("A")
                    else:
                        rejected_rounds += 1
                        native_mtp_rejected += 1
                        native_tree_misses += 1
                        if constraint is not None:
                            constraint.accept_token(target_root)
                            grammar_completed = bool(constraint.completed)
                        new_tokens = [int(target_root)]
                        new_token_logits = [root_logits]
                        round_outcomes.append("R")
                    next_catchup_tok = new_tokens[-1]
                    if round_rejected:
                        refeed_sweeps_saved += 1
                else:
                    for step, draft_tok in enumerate(draft_tokens):
                        authoritative_logits = _authoritative_target_logits(
                            tgt, spec_logits[step], constraint, step)
                        verified_proposals += 1
                        if round_proposal_source == "N":
                            ngram_first_verified_by_step[step] += 1
                        else:
                            verified_by_step[step] += 1
                        if sampling.is_greedy:
                            target_winner = int(mx.argmax(authoritative_logits))
                            draft_accepted = target_winner == draft_tok
                            true_tok = (
                                draft_tok if draft_accepted
                                else target_winner
                            )
                            if (self.proposal_replay_top_k
                                    and round_proposal_source == "M"):
                                ranked_tokens = draft_ranked_tokens[step]
                                if ranked_tokens is None:
                                    raise RuntimeError(
                                        "greedy proposal-rank capture omitted "
                                        "draft candidates")
                                try:
                                    target_rank = (
                                        ranked_tokens.index(target_winner) + 1)
                                except ValueError:
                                    target_rank = 0
                                greedy_target_rank_counts_by_step[
                                    step][target_rank] += 1
                                round_target_ranks.append(target_rank)
                                if self.proposal_replay_top_k >= 2:
                                    margin_bucket = draft_margin_buckets[step]
                                    if margin_bucket is None:
                                        raise RuntimeError(
                                            "greedy proposal-rank capture "
                                            "omitted draft margin")
                                    greedy_margin_rank_counts_by_step[
                                        step][margin_bucket][target_rank] += 1
                                if not draft_accepted and target_rank > 1:
                                    greedy_rescuable_rejections_by_step[
                                        step] += 1
                        else:
                            step_probabilities = draft_probabilities[step]
                            if step_probabilities is None:
                                raise RuntimeError(
                                    "stochastic Qwen MTP proposal omitted q")
                            draft_accepted, true_tok, target_probabilities = (
                                _verify_stochastic_mtp_token(
                                    draft_tok,
                                    step_probabilities,
                                    authoritative_logits,
                                    sampling,
                                    all_tokens + new_tokens,
                                )
                            )
                            overlap = mx.sum(mx.minimum(
                                target_probabilities, step_probabilities))
                            mx.eval(overlap)
                            overlap_value = float(overlap.item())
                            stochastic_expected_acceptance_sum += overlap_value
                            stochastic_expected_acceptance_by_step[step] += (
                                overlap_value)
                            if step == 0:
                                stochastic_first_step_expected_acceptance_sum += (
                                    overlap_value)
                            if self.proposal_replay_top_k:
                                step_rank_probabilities = (
                                    draft_rank_probabilities[step])
                                if step_rank_probabilities is None:
                                    raise RuntimeError(
                                        "proposal-q replay omitted draft ranks")
                                proposal_replay_records.append(
                                    proposal_q_replay_record(
                                        step_rank_probabilities,
                                        target_probabilities,
                                        max_rank=self.proposal_replay_top_k,
                                        proposal=int(draft_tok),
                                        accepted=bool(draft_accepted),
                                        history_length=(
                                            len(all_tokens) + len(new_tokens)),
                                        round_index=speculative_rounds - 1,
                                        draft_step_index=step,
                                    )
                                )

                        if not draft_accepted:
                            round_rejected = True
                            rejected_rounds += 1
                            if accepted_prefix:
                                partial_accept_rounds += 1
                            if constraint is not None:
                                constraint.accept_token(true_tok)
                                grammar_completed = bool(constraint.completed)
                            new_tokens.append(int(true_tok))
                            new_token_logits.append(authoritative_logits)
                            break

                        accepted += 1
                        accepted_prefix += 1
                        if round_proposal_source == "N":
                            ngram_first_accepted_by_step[step] += 1
                        else:
                            accepted_by_step[step] += 1
                        if constraint is not None:
                            constraint.accept_token(draft_tok)
                            grammar_completed = bool(constraint.completed)
                        new_tokens.append(int(draft_tok))
                        new_token_logits.append(authoritative_logits)
                        if _candidate_terminal(new_tokens):
                            break

                    round_draft_width = len(draft_tokens)
                    all_drafts_accepted = (
                        accepted_prefix == round_draft_width
                        and not round_rejected)
                    if round_proposal_source == "N":
                        ngram_first_accepted += accepted_prefix
                        ngram_first_rejected += int(round_rejected)
                    else:
                        native_mtp_accepted += accepted_prefix
                        native_mtp_rejected += int(round_rejected)
                    if all_drafts_accepted:
                        full_accept_rounds += 1
                    if round_proposal_source == "N" and round_draft_width > 1:
                        if round_rejected:
                            round_outcomes.append(
                                "R" if accepted_prefix == 0
                                else f"A{accepted_prefix}R")
                        else:
                            round_outcomes.append(f"A{accepted_prefix}")
                    elif self.depth == 1:
                        round_outcomes.append(
                            "A" if all_drafts_accepted else "R")
                    elif round_rejected:
                        round_outcomes.append(
                            "R" if accepted_prefix == 0
                            else f"A{accepted_prefix}R")
                    else:
                        round_outcomes.append(f"A{accepted_prefix}")

                    if (sampling.is_greedy
                            and self.proposal_replay_top_k >= 2
                            and round_proposal_source == "M"):
                        margin_buckets = draft_margin_buckets[
                            :round_draft_width]
                        if any(bucket is None for bucket in margin_buckets):
                            raise RuntimeError(
                                "greedy round-confidence capture omitted "
                                "draft margin")
                        greedy_round_confidence_records.append({
                            # Bucket indices reference the fixed public
                            # threshold vector below.  Token IDs, prompt text,
                            # logits, and raw margins are deliberately absent.
                            "margin_buckets": [
                                int(bucket) for bucket in margin_buckets
                            ],
                            "target_ranks": list(round_target_ranks),
                            "accepted_prefix": accepted_prefix,
                            "rejected": int(round_rejected),
                        })

                    if (all_drafts_accepted
                            and not _candidate_terminal(new_tokens)):
                        bonus_logits = _authoritative_target_logits(
                            tgt, spec_logits[round_draft_width], constraint,
                            round_draft_width)
                        bonus_tok = sample(
                            bonus_logits,
                            sampling,
                            history=all_tokens + new_tokens,
                        )
                        if constraint is not None:
                            constraint.accept_token(bonus_tok)
                            grammar_completed = bool(constraint.completed)
                        new_tokens.append(int(bonus_tok))
                        new_token_logits.append(bonus_logits)
                        round_bonus_candidate = True
                    next_catchup_tok = new_tokens[-1]
                    h_last = tgt._h_last
                    if round_rejected:
                        refeed_sweeps_saved += int(round_serial_verify)
                verifier_round_s += time.perf_counter() - verifier_started

            emitted_before_round = len(emitted)
            for tok, token_logits in zip(
                    new_tokens, new_token_logits, strict=True):
                all_tokens.append(tok)
                emitted.append(tok)
                last_emitted_logits = token_logits
                if stop:
                    decoded = tgt.tokenizer.decode(emitted)
                    match = _stop_match(decoded)
                    if match is not None:
                        cut, _order, matched_stop_sequence = match
                        stop_text = decoded[:cut]
                if stream_decoder is not None and stop_text is None:
                    delta = stream_decoder.push(emitted)
                    if delta:
                        on_token(delta)
                if (stop_text is not None or tok in eos
                        or len(emitted) >= max_tokens):
                    break
            terminal_round = (
                stop_text is not None
                or emitted[-1] in eos
                or grammar_completed
                or len(emitted) >= max_tokens)
            if speculative_round:
                # The final returned token is always the next round's
                # still-unfed catchup.  Therefore a round that returned N
                # tokens commits exactly catchup + the first N-1 returned
                # positions from this round's verifier window. This one rule
                # covers reject-at-0/1, full acceptance, EOS/stop at either
                # proposal, grammar completion, and a short remaining budget.
                emitted_this_round = len(emitted) - emitted_before_round
                verifier_input_positions += round_verify_width
                verifier_output_tokens += emitted_this_round
                verifier_accepted_draft_tokens += min(
                    accepted_prefix, emitted_this_round)
                verifier_correction_tokens += int(
                    round_rejected and emitted_this_round > accepted_prefix)
                verifier_bonus_tokens += int(
                    round_bonus_candidate
                    and emitted_this_round > accepted_prefix)
                target_fed_positions = min(
                    round_verify_width,
                    1 + max(0, emitted_this_round - 1),
                )
                verifier_committed_positions += target_fed_positions
                verifier_rolled_back_positions += max(
                    0, round_verify_width - target_fed_positions)
                if round_tree_verification is not None:
                    if round_tree_selected_path is None:
                        raise RuntimeError(
                            "native MTP tree omitted its selected path")
                    commit_path = round_tree_selected_path[
                        :target_fed_positions]
                    if len(commit_path) != target_fed_positions:
                        raise RuntimeError(
                            "native MTP tree cannot commit the emitted prefix")
                    commit_started = time.perf_counter()
                    round_tree_verification.commit(
                        commit_path, target=tgt, kv=kv)
                    native_tree_factor_commit_s += (
                        time.perf_counter() - commit_started)
                    native_tree_paths_committed += 1
                    h_last = tgt._h_last
                    del round_tree_verification, spec_logits
                    mx.clear_cache()
                elif target_fed_positions < round_verify_width:
                    retained_prefix = (
                        tgt.consume_serial_kda_endpoint(target_fed_positions)
                        if round_capture_endpoint else None
                    )
                    if round_capture_endpoint and retained_prefix is None:
                        raise RuntimeError(
                            "serial Qwen verifier did not retain KDA prefix "
                            f"{target_fed_positions}/{round_verify_width}")
                    if (
                        round_start_layer_lengths is not None
                        and round_layer_growth is not None
                    ):
                        retained_lengths = tuple(
                            start + (
                                target_fed_positions if growth else 0)
                            for start, growth in zip(
                                round_start_layer_lengths,
                                round_layer_growth,
                                strict=True,
                            )
                        )
                        kv.trim_layer_lengths(retained_lengths)
                    else:
                        kv.trim(round_start_offset + target_fed_positions)
                    if round_capture_endpoint:
                        kv.kda_cache = retained_prefix
                        kda_endpoint_restores += 1
                    hidden_window = getattr(tgt, "_h_window", None)
                    if hidden_window is not None:
                        prefix_hidden = hidden_window[
                            :, target_fed_positions - 1:target_fed_positions, :]
                        tgt._h_last = prefix_hidden
                        h_last = prefix_hidden
                    elif not terminal_round:
                        raise RuntimeError(
                            "Qwen MTP target omitted verifier hidden window")
                    target_prefix_rollbacks += 1
                elif round_capture_endpoint:
                    # Drop strict-prefix snapshots; the full endpoint already
                    # lives in kv.kda_cache.
                    tgt.consume_serial_kda_endpoint(None)

                mtp_end_lengths = mtp_kv.layer_lengths()
                mtp_growth = tuple(
                    end - start for start, end in zip(
                        round_mtp_start_lengths,
                        mtp_end_lengths,
                        strict=True,
                    )
                )
                if any(growth not in (0, self.depth) for growth in mtp_growth):
                    raise RuntimeError(
                        "recurrent Qwen MTP chain changed its KV by an "
                        f"unexpected count: {mtp_growth}")
                committed_mtp_steps = min(self.depth, target_fed_positions)
                if committed_mtp_steps < self.depth:
                    mtp_lengths = tuple(
                        start + (committed_mtp_steps if growth else 0)
                        for start, growth in zip(
                            round_mtp_start_lengths,
                            mtp_growth,
                            strict=True,
                        )
                    )
                    if mtp_lengths != mtp_end_lengths:
                        mtp_kv.trim_layer_lengths(mtp_lengths)
                        mtp_kv_rollbacks += 1
            # An accepted pair may encounter EOS/stop on its first token, in
            # which case ``next_catchup_tok`` is the unused bonus token. Never
            # continue from that uncommitted value (the old behavior leaked
            # post-EOS template tokens such as ``user`` into the response).
            catchup_tok = emitted[-1] if terminal_round else next_catchup_tok
            if terminal_round:
                break
            # Retained per-layer/KDA midpoints changed the break-even math:
            # every rejected draft now emits one authoritative token in the
            # same target sweep the ordinary path required, while every
            # accepted draft emits two and saves a complete target sweep.
            # The released draft head is only one extra layer, so one accepted
            # token can repay several rejected probes. Size the initial window
            # from the measured target/draft cost instead of disabling forever
            # after three possibly unrepresentative tokens. A failed window
            # enters a bounded plain cooldown, then performs one recovery probe
            # so a later predictable region can reactivate speculation.
            if speculative_round and self.adaptive_stop:
                round_accepted = accepted - round_accepted_before
                round_expected_acceptance = (
                    stochastic_first_step_expected_acceptance_sum
                    - round_expected_acceptance_before)
                adaptive_effective_probe_limit = (
                    _adaptive_break_even_probe_rounds(
                        draft_round_s,
                        verifier_round_s,
                        adaptive_probe_limit,
                    )
                )
                if adaptive_recovery_probe:
                    adaptive_recovery_probes += 1
                    recovery_useful = (
                        round_accepted > 0 if sampling.is_greedy
                        else round_expected_acceptance >= 0.05
                    )
                    adaptive_recovery_probe = False
                    if recovery_useful:
                        adaptive_reactivations += 1
                        adaptive_window_rounds = 0
                        adaptive_window_accepted = 0
                        adaptive_window_expected_acceptance = 0.0
                    else:
                        adaptive_disabled = True
                        adaptive_disabled_ever = True
                        adaptive_disable_events += 1
                        adaptive_cooldown_remaining = (
                            self.adaptive_reprobe_interval)
                else:
                    adaptive_window_rounds += 1
                    adaptive_window_accepted += round_accepted
                    adaptive_window_expected_acceptance += (
                        round_expected_acceptance)
                    if adaptive_window_rounds >= adaptive_effective_probe_limit:
                        disable_window = (
                            _adaptive_mtp_should_disable(
                                adaptive_window_rounds,
                                adaptive_window_accepted,
                                adaptive_effective_probe_limit,
                            )
                            if sampling.is_greedy else
                            _adaptive_stochastic_mtp_should_disable(
                                adaptive_window_rounds,
                                adaptive_window_expected_acceptance,
                                adaptive_effective_probe_limit,
                            )
                        )
                        if disable_window:
                            adaptive_disabled = True
                            adaptive_disabled_ever = True
                            adaptive_disable_events += 1
                            adaptive_cooldown_remaining = (
                                self.adaptive_reprobe_interval)
                        adaptive_window_rounds = 0
                        adaptive_window_accepted = 0
                        adaptive_window_expected_acceptance = 0.0

        final_text = stop_text if stop_text is not None else tgt.tokenizer.decode(emitted)
        if stream_decoder is not None:
            delta = stream_decoder.finish(emitted, final_text=final_text)
            if delta:
                on_token(delta)
        decode_s = time.perf_counter() - decode_t0
        # A stop/EOS can land before the end of a fully-accepted round's
        # bonus token, mirroring speculative.py's own endpoint-KV contract.
        endpoint = len(ids) + len(emitted) - 1
        if kv.offset > endpoint:
            kv.trim(endpoint)
        total_s = time.perf_counter() - request_t0
        path_stats = bootstrap_stats
        plain_equivalent_sweeps = max(0, len(emitted) - 1)
        target_sweeps_avoided = max(
            0, plain_equivalent_sweeps - target_decode_sweeps)
        average_plain_target_s = (
            plain_round_s / plain_timed_sweeps
            if plain_timed_sweeps else (
                verifier_round_s / speculative_rounds
                if speculative_rounds else 0.0
            )
        )
        average_draft_s = (
            draft_round_s / speculative_rounds
            if speculative_rounds else 0.0)
        average_verifier_s = (
            verifier_round_s / speculative_rounds
            if speculative_rounds else 0.0)
        estimated_mtp_net_s = (
            target_sweeps_avoided * average_plain_target_s
            - draft_round_s
            - speculative_rounds * max(
                0.0, average_verifier_s - average_plain_target_s)
        )
        estimated_break_even_accept_rate = (
            (
                average_draft_s
                + max(0.0, average_verifier_s - average_plain_target_s)
            ) / average_plain_target_s
            if average_plain_target_s > 0.0 else 0.0
        )
        path_stats.update({
            "qwen_mtp_enabled": 1,
            "qwen_mtp_used": int(proposed > 0),
            "qwen_mtp_target_sweeps": target_decode_sweeps,
            "qwen_mtp_plain_equivalent_target_sweeps": (
                plain_equivalent_sweeps),
            "qwen_mtp_target_sweeps_avoided": target_sweeps_avoided,
            "qwen_mtp_estimated_net_saved_s": estimated_mtp_net_s,
            "qwen_mtp_estimated_break_even_accept_rate": (
                estimated_break_even_accept_rate),
            "qwen_mtp_target_tokens_per_sweep": (
                plain_equivalent_sweeps / target_decode_sweeps
                if target_decode_sweeps else 0.0),
            "qwen_mtp_depth": self.depth,
            "qwen_mtp_verify_width": max(
                self.depth + 1, max_verify_width_observed),
            "qwen_mtp_max_verify_width_observed": max_verify_width_observed,
            "qwen_mtp_speculative_rounds": speculative_rounds,
            "qwen_mtp_proposed": proposed,
            "qwen_mtp_verified_proposals": verified_proposals,
            "qwen_mtp_accepted": accepted,
            "qwen_mtp_accepted_by_step": list(accepted_by_step),
            "qwen_mtp_verified_by_step": list(verified_by_step),
            "qwen_mtp_full_accept_rounds": full_accept_rounds,
            "qwen_mtp_partial_accept_rounds": partial_accept_rounds,
            "qwen_mtp_rejected_rounds": rejected_rounds,
            "qwen_mtp_accept_rate": (
                accepted / proposed if proposed else 0.0),
            "qwen_mtp_decode_tokens": max(0, len(emitted) - 1),
            "qwen_mtp_adaptive_disabled": int(adaptive_disabled_ever),
            "qwen_mtp_adaptive_currently_disabled": int(adaptive_disabled),
            "qwen_mtp_probe_rounds": min(
                speculative_rounds, adaptive_probe_limit),
            "qwen_mtp_effective_probe_rounds": (
                adaptive_effective_probe_limit),
            "qwen_mtp_adaptive_disable_events": adaptive_disable_events,
            "qwen_mtp_adaptive_reprobe_interval": (
                self.adaptive_reprobe_interval),
            "qwen_mtp_adaptive_cooldown_sweeps": adaptive_cooldown_sweeps,
            "qwen_mtp_adaptive_recovery_probes": adaptive_recovery_probes,
            "qwen_mtp_adaptive_reactivations": adaptive_reactivations,
            "qwen_mtp_plain_decode_sweeps": plain_decode_sweeps,
            "qwen_mtp_plain_timed_sweeps": plain_timed_sweeps,
            "qwen_mtp_warmup_decode_sweeps": warmup_decode_sweeps,
            "qwen_mtp_serial_verify_rounds": serial_verify_rounds,
            "qwen_mtp_verifier_input_positions": verifier_input_positions,
            "qwen_mtp_verifier_committed_positions": (
                verifier_committed_positions),
            "qwen_mtp_verifier_rolled_back_positions": (
                verifier_rolled_back_positions),
            "qwen_mtp_verifier_output_tokens": verifier_output_tokens,
            "qwen_mtp_verifier_tokens_per_sweep": (
                verifier_output_tokens / speculative_rounds
                if speculative_rounds else 0.0),
            "qwen_mtp_verifier_accepted_draft_tokens": (
                verifier_accepted_draft_tokens),
            "qwen_mtp_verifier_correction_tokens": (
                verifier_correction_tokens),
            "qwen_mtp_verifier_bonus_tokens": verifier_bonus_tokens,
            "qwen_mtp_draft_round_s": draft_round_s,
            "qwen_mtp_verifier_round_s": verifier_round_s,
            "qwen_mtp_target_batched_mlp_layers": int(getattr(
                tgt, "_qwen35_serial_verify_batched_mlp_layers", 0)),
            "qwen_mtp_target_batched_mlp_positions": int(getattr(
                tgt, "_qwen35_serial_verify_batched_mlp_positions", 0)),
            "qwen_mtp_target_batched_mlp_s": float(getattr(
                tgt, "_qwen35_serial_verify_batched_mlp_s", 0.0)),
            "qwen_mtp_plain_round_s": plain_round_s,
            "qwen_mtp_kda_endpoint_restores": kda_endpoint_restores,
            "qwen_mtp_refeed_sweeps_saved": refeed_sweeps_saved,
            "qwen_mtp_target_prefix_rollbacks": target_prefix_rollbacks,
            "qwen_mtp_draft_kv_rollbacks": mtp_kv_rollbacks,
            "qwen_mtp_constraint_verified": int(constraint is not None),
            "qwen_mtp_stochastic": int(not sampling.is_greedy),
            "qwen_mtp_stochastic_draft_argmax": int(
                not sampling.is_greedy
                and self.stochastic_draft_top_k == 1),
            "qwen_mtp_stochastic_draft_top_k": (
                self.stochastic_draft_top_k
                if not sampling.is_greedy else 0),
            "qwen_mtp_q_policy": self.proposal_q_policy.as_dict(),
            "qwen_mtp_engine_identity": self.mtp_engine_identity,
            "qwen_mtp_native_tree_width": self.native_tree_width,
            "qwen_mtp_native_tree_eligible": int(
                self._native_tree_request_eligible(sampling, constraint)),
            "qwen_mtp_native_tree_rounds": native_tree_rounds,
            "qwen_mtp_native_tree_hits": native_tree_hits,
            "qwen_mtp_native_tree_misses": native_tree_misses,
            "qwen_mtp_native_tree_selected_rank_counts": (
                native_tree_selected_rank_counts),
            "qwen_mtp_native_tree_hit_rate": (
                native_tree_hits / native_tree_rounds
                if native_tree_rounds else 0.0),
            "qwen_mtp_native_tree_nodes_verified": (
                native_tree_nodes_verified),
            "qwen_mtp_native_tree_paths_committed": (
                native_tree_paths_committed),
            "qwen_mtp_native_tree_factor_bytes_peak": (
                native_tree_factor_bytes_peak),
            "qwen_mtp_native_tree_factor_commit_s": (
                native_tree_factor_commit_s),
            "qwen_mtp_ngram_first_enabled": int(self.ngram_first),
            "qwen_mtp_ngram_first_eligible": int(
                self.ngram_first and sampling.is_greedy),
            "qwen_mtp_ngram_first_max_draft_tokens": (
                self.ngram_max_draft_tokens),
            "qwen_mtp_ngram_first_attempts": ngram_first_attempts,
            "qwen_mtp_ngram_first_matches": ngram_first_matches,
            "qwen_mtp_ngram_first_proposed": ngram_first_proposed,
            "qwen_mtp_ngram_first_accepted": ngram_first_accepted,
            "qwen_mtp_ngram_first_rejected": ngram_first_rejected,
            "qwen_mtp_ngram_first_native_draft_bypasses": (
                ngram_first_native_draft_bypasses),
            "qwen_mtp_ngram_first_accepted_by_step": list(
                ngram_first_accepted_by_step),
            "qwen_mtp_ngram_first_verified_by_step": list(
                ngram_first_verified_by_step),
            "qwen_mtp_ngram_first_max_proposed_per_round": (
                ngram_first_max_proposed_per_round),
            "qwen_mtp_native_draft_proposed": native_mtp_proposed,
            "qwen_mtp_native_draft_accepted": native_mtp_accepted,
            "qwen_mtp_native_draft_rejected": native_mtp_rejected,
            "qwen_mtp_proposal_sources": "".join(proposal_sources),
            "qwen_mtp_stochastic_expected_acceptance": (
                stochastic_expected_acceptance_sum / verified_proposals
                if not sampling.is_greedy and verified_proposals else 0.0),
            "qwen_mtp_stochastic_expected_acceptance_by_step": [
                (
                    stochastic_expected_acceptance_by_step[index]
                    / verified_by_step[index]
                    if not sampling.is_greedy and verified_by_step[index]
                    else 0.0
                )
                for index in range(self.depth)
            ],
            "qwen_mtp_grammar_forced_tokens": grammar_forced_tokens,
            "qwen_mtp_grammar_forced_sweeps": grammar_forced_sweeps,
            "qwen_mtp_grammar_aware_draft_enabled": int(
                self.grammar_aware_draft),
            "qwen_mtp_grammar_masked_draft_tokens": (
                grammar_masked_draft_tokens),
            "qwen_mtp_grammar_masked_draft_rounds": (
                grammar_masked_draft_rounds),
            "qwen_mtp_round_outcomes": "".join(round_outcomes),
            # Representation-neutral round-page counters.  They describe the
            # actual proposal page (released BF16 or packed MXFP4) without
            # falsely labeling quantized proposal I/O as BF16 sidecar I/O.
            "qwen_mtp_proposal_weight_representation": (
                request_weight_representation),
            "qwen_mtp_proposal_page_round_loads": sidecar_round_loads,
            "qwen_mtp_proposal_page_round_releases": sidecar_round_releases,
            "qwen_mtp_proposal_page_read_bytes": sidecar_read_bytes,
            "qwen_mtp_proposal_page_loaded_resident_bytes": (
                sidecar_loaded_resident_bytes),
            "qwen_mtp_proposal_page_released_resident_bytes": (
                sidecar_released_resident_bytes),
            "qwen_mtp_proposal_page_peak_resident_bytes": (
                sidecar_peak_resident_bytes),
            "qwen_mtp_proposal_page_cache_discards": sidecar_cache_discards,
            "qwen_mtp_proposal_page_cache_prepare_calls": (
                sidecar_cache_prepare_calls),
            "qwen_mtp_proposal_page_cache_prepare_bytes": (
                sidecar_cache_prepare_bytes),
            "qwen_mtp_proposal_page_cache_prepare_released_bytes": (
                sidecar_cache_prepare_released_bytes),
            "qwen_mtp_proposal_page_load_s": sidecar_load_s,
            "qwen_mtp_proposal_page_release_s": sidecar_release_s,
            # Backward-compatible BF16-only endpoint fields: the sidecar is no
            # longer request-pinned and no round-local bytes remain at return.
            "qwen_mtp_request_local_sidecar_pin": 0,
            "qwen_mtp_request_local_sidecar_bytes": 0,
            "qwen_mtp_bf16_sidecar_round_loads": (
                sidecar_round_loads
                if request_weight_representation == "released-bf16" else 0),
            "qwen_mtp_bf16_sidecar_round_releases": (
                sidecar_round_releases
                if request_weight_representation == "released-bf16" else 0),
            "qwen_mtp_bf16_sidecar_read_bytes": (
                sidecar_read_bytes
                if request_weight_representation == "released-bf16" else 0),
            "qwen_mtp_bf16_sidecar_loaded_resident_bytes": (
                sidecar_loaded_resident_bytes
                if request_weight_representation == "released-bf16" else 0),
            "qwen_mtp_bf16_sidecar_released_resident_bytes": (
                sidecar_released_resident_bytes
                if request_weight_representation == "released-bf16" else 0),
            "qwen_mtp_bf16_sidecar_peak_resident_bytes": (
                sidecar_peak_resident_bytes
                if request_weight_representation == "released-bf16" else 0),
            "qwen_mtp_bf16_sidecar_cache_discards": (
                sidecar_cache_discards
                if request_weight_representation == "released-bf16" else 0),
            "qwen_mtp_bf16_sidecar_cache_prepare_calls": (
                sidecar_cache_prepare_calls
                if request_weight_representation == "released-bf16" else 0),
            "qwen_mtp_bf16_sidecar_cache_prepare_bytes": (
                sidecar_cache_prepare_bytes
                if request_weight_representation == "released-bf16" else 0),
            "qwen_mtp_bf16_sidecar_cache_prepare_released_bytes": (
                sidecar_cache_prepare_released_bytes
                if request_weight_representation == "released-bf16" else 0),
            "qwen_mtp_bf16_sidecar_load_s": (
                sidecar_load_s
                if request_weight_representation == "released-bf16" else 0.0),
            "qwen_mtp_bf16_sidecar_release_s": (
                sidecar_release_s
                if request_weight_representation == "released-bf16" else 0.0),
        })
        if self.proposal_replay_top_k:
            path_stats["qwen_mtp_proposal_q_replay"] = (
                proposal_replay_records)
            path_stats["qwen_mtp_greedy_target_rank_counts_by_step"] = (
                greedy_target_rank_counts_by_step)
            path_stats["qwen_mtp_greedy_rescuable_rejections_by_step"] = (
                greedy_rescuable_rejections_by_step)
            path_stats["qwen_mtp_greedy_draft_margin_thresholds"] = list(
                _GREEDY_DRAFT_MARGIN_THRESHOLDS)
            path_stats["qwen_mtp_greedy_margin_rank_counts_by_step"] = (
                greedy_margin_rank_counts_by_step)
            path_stats["qwen_mtp_greedy_round_confidence_records"] = (
                greedy_round_confidence_records)
        if not proposed:
            path_stats["qwen_mtp_fallback_reason"] = (
                "terminal-during-plain-warmup")

        # Bootstrap path_stats contain only the one-token target.generate()
        # delta. Replace that stale slice with the complete wrapper request,
        # and split shared-head traffic into authoritative target versus draft
        # projections. Target totals intentionally include bootstrap, every
        # serial verification row (including bonus/correction rows), and any
        # constraint-aware rerank from its matching retained hidden position.
        request_rerank_totals = _reranked_head_telemetry_delta(
            request_rerank_before,
            _reranked_head_telemetry_snapshot(tgt),
        )
        target_rerank_totals = {
            key: max(0, int(value) - int(draft_rerank_totals.get(key, 0)))
            for key, value in request_rerank_totals.items()
        }
        _publish_reranked_head_telemetry(
            path_stats, "reranked_lm_head_", request_rerank_totals)
        _publish_reranked_head_telemetry(
            path_stats, "qwen_mtp_target_reranked_lm_head_",
            target_rerank_totals)
        _publish_reranked_head_telemetry(
            path_stats, "qwen_mtp_draft_reranked_lm_head_",
            draft_rerank_totals)

        request_cache_after = _cache_io_snapshot(tgt)
        _record_cache_io_delta(
            tgt, request_cache_before, path_stats, after=request_cache_after)
        _record_cache_io_delta(
            tgt, request_cache_before, path_stats, prefix="prefill_",
            after=decode_cache_before)
        _record_cache_io_delta(
            tgt, decode_cache_before, path_stats, prefix="decode_",
            after=request_cache_after)

        # Restore a truthful full endpoint slot only when the bootstrap slot
        # itself owned this KV. A separately forked stable-boundary slot is
        # already the correct reusable artifact and was intentionally left in
        # the LRU above.
        if endpoint_slot is not None and last_emitted_logits is not None:
            mx.eval(last_emitted_logits)
            recurrent_state = getattr(kv, "kda_cache", None)
            if recurrent_state is not None:
                recurrent_state.synchronize()
            endpoint_slot.tokens = tuple(ids + emitted[:-1])
            endpoint_slot.kv = kv
            endpoint_slot.logits = last_emitted_logits
            endpoint_slot.prompt_length = len(ids)
            endpoint_slot.segment_chain = ()
            tgt._append_hot_prompt_slot(endpoint_slot)

        if tgt.governor is not None:
            tgt._true_peak_metal_bytes = max(
                tgt._true_peak_metal_bytes,
                tgt.governor.request_peak(),
                mx.get_active_memory(),
            )
        profiler = getattr(tgt, "_request_profiler", None)
        execution_profile = (
            profiler.result(total_s) if profiler is not None else None)
        result = {
            "text": final_text,
            "tokens": emitted,
            "prefill_s": prefill_s,
            "decode_s": decode_s,
            "first_token_s": first_token_s,
            "total_s": total_s,
            "tok_per_s": ((len(emitted) - 1) / decode_s if len(emitted) > 1 else 0.0),
            "kv_bytes": kv.nbytes(),
            "kv_positions": kv.offset,
            "stopped": stop_text is not None,
            "stop_sequence": matched_stop_sequence,
            "termination_reason": (
                "stop_sequence" if stop_text is not None else
                "grammar" if grammar_completed else
                "eos" if emitted[-1] in eos else "length"),
            "true_peak_metal_bytes": tgt._true_peak_metal_bytes,
            "path_stats": path_stats,
            "prompt_tokens": len(ids),
        }
        if execution_profile is not None:
            result["execution_profile"] = execution_profile
        return result
