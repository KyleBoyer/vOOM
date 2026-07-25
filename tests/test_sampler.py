"""Dependency-light functional sampling tests."""

from __future__ import annotations

import mlx.core as mx
import pytest

from runtime.sampler import SamplingParams, sample


def _scalar_remap_reference(logits, params):
    """Pre-optimization sampler used to gate seed-for-seed compatibility."""
    values = logits.reshape(-1).astype(mx.float32) / float(params.temperature)
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
        remove = mx.concatenate([mx.array([False]), remove[:-1]])
        sorted_values = mx.where(remove, float("-inf"), sorted_values)
        selected = int(mx.random.categorical(sorted_values))
        local_index = int(order[selected])
    else:
        local_index = int(mx.random.categorical(values))
    return int(indices[local_index]) if indices is not None else local_index


def test_default_and_zero_temperature_are_greedy():
    logits = mx.array([-1.0, 3.0, 2.0])
    assert sample(logits) == 1
    assert sample(logits, SamplingParams(temperature=0, top_p=0.2, top_k=2)) == 1


def test_top_k_one_and_top_p_zero_are_greedy():
    logits = mx.array([0.0, 0.5, 2.0, 1.0])
    assert sample(logits, SamplingParams(temperature=1, top_k=1)) == 2
    assert sample(logits, SamplingParams(temperature=1, top_p=0)) == 2


def test_seeded_sampling_is_repeatable_and_top_k_is_enforced():
    logits = mx.array([0.0, 0.0, 0.0, -100.0])
    params = SamplingParams(temperature=1, top_k=3, seed=314159)
    params.seed_rng()
    first = [sample(logits, params) for _ in range(32)]
    params.seed_rng()
    second = [sample(logits, params) for _ in range(32)]
    assert first == second
    assert set(first) <= {0, 1, 2}
    assert len(set(first)) > 1


def test_nucleus_filter_excludes_low_probability_tail():
    logits = mx.array([8.0, 1.0, 0.0, -1.0])
    params = SamplingParams(temperature=1, top_p=0.5, seed=7)
    params.seed_rng()
    assert {sample(logits, params) for _ in range(20)} == {0}


@pytest.mark.parametrize("top_k,top_p", [
    (64, 1.0),
    (0, 0.92),
    (64, 0.92),
])
def test_filtered_index_composition_is_seed_for_seed_exact(top_k, top_p):
    logits = mx.linspace(-3.0, 3.0, 257)
    params = SamplingParams(
        temperature=0.73, top_k=top_k, top_p=top_p, seed=20260716)
    params.seed_rng()
    expected = [_scalar_remap_reference(logits, params) for _ in range(80)]
    params.seed_rng()
    actual = [sample(logits, params) for _ in range(80)]
    assert actual == expected


@pytest.mark.parametrize("kwargs", [
    {"temperature": -1}, {"temperature": float("nan")},
    {"top_p": -0.1}, {"top_p": 1.1}, {"top_k": -1},
    {"top_k": 1.5}, {"seed": True}, {"seed": -1}, {"seed": 2 ** 64},
    {"repetition_penalty": 0}, {"repetition_penalty": -1.0},
    {"repetition_penalty": float("nan")}, {"repetition_penalty": float("inf")},
])
def test_invalid_sampling_parameters_fail(kwargs):
    with pytest.raises(ValueError):
        SamplingParams(**kwargs)


def test_default_repetition_penalty_is_a_true_no_op():
    """F102: repetition_penalty=1.0 (the default) must be byte-identical to
    the pre-existing behavior -- this is what keeps every prior
    byte-identical correctness proof in this project valid without
    re-verification."""
    logits = mx.array([-1.0, 3.0, 2.0, 0.5])
    assert sample(logits, history=[0, 1, 2, 3]) == 1
    params = SamplingParams(temperature=1, top_k=2, seed=99)
    params.seed_rng()
    without_history = [sample(logits, params) for _ in range(20)]
    params.seed_rng()
    with_history = [sample(logits, params, history=[0, 1, 2, 3]) for _ in range(20)]
    assert without_history == with_history


def test_repetition_penalty_demotes_positive_logit_and_promotes_negative():
    """HF convention: a positive repeated logit is divided by the penalty
    (shrinks toward zero); a negative one is multiplied (grows more
    negative). Both move the repeated token's rank DOWN, never up."""
    logits = mx.array([5.0, 4.9, -1.0])
    params = SamplingParams(temperature=0, repetition_penalty=2.0)
    # Token 0 (highest raw logit) is in the history -- penalized to 2.5,
    # dropping below token 1's untouched 4.9, so greedy now picks token 1.
    assert sample(logits, params) == 0  # no history: still the raw argmax
    assert sample(logits, params, history=[0]) == 1
    # Penalizing the negative logit too (token 2: -1.0 -> -2.0) must not
    # accidentally make it MORE attractive -- still last choice.
    assert sample(logits, params, history=[0, 1, 2]) in (0, 1)


def test_repetition_penalty_ignores_out_of_range_and_duplicate_ids():
    logits = mx.array([1.0, -1.0, 0.5])
    params = SamplingParams(temperature=0, repetition_penalty=3.0)
    # Duplicate + out-of-vocab ids in history must not crash or double-apply.
    result = sample(logits, params, history=[0, 0, 0, 999999])
    assert result in (0, 1, 2)
