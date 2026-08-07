"""F224: DSpark draft primitives, oracled against the checkpoint's own code.

``get_dspark_topk_idxs`` is pure torch, so the gather list is checked against
the released generator directly rather than against a restatement of it. The
sampling loop and acceptance rule are checked by the properties that make
speculative decoding correct, not by example.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODEL = ROOT / "models" / "DeepSeek-V4-Flash-0731"
INFERENCE = MODEL / "inference"

WINDOW, BLOCK = 8, 5


def _reference():
    stub = sys.modules.get("kernel") or types.ModuleType("kernel")
    for name in ("act_quant", "fp4_act_quant", "fp8_gemm", "fp4_gemm",
                 "sparse_attn", "hc_split_sinkhorn"):
        if not hasattr(stub, name):
            def _unavailable(*_a, __name=name, **_k):
                raise RuntimeError(f"kernel {__name!r} unavailable here")
            setattr(stub, name, _unavailable)
    sys.modules["kernel"] = stub
    sys.path.insert(0, str(INFERENCE))
    import model as reference_module

    return reference_module


released = pytest.mark.skipif(
    not (INFERENCE / "model.py").is_file(),
    reason="DeepSeek-V4-Flash-0731 inference/ not present")


@released
@pytest.mark.parametrize("start_pos", [1, 7, 8, 33, 129])
def test_gather_matches_the_released_generator(start_pos):
    import mlx.core as mx

    from runtime.dsv4_dspark import dspark_topk_idxs

    reference = _reference()
    expected = reference.get_dspark_topk_idxs(
        WINDOW, 1, BLOCK, start_pos).numpy()
    got = np.array(dspark_topk_idxs(WINDOW, BLOCK, start_pos))
    assert got.shape == expected.shape, f"{got.shape} != {expected.shape}"
    assert np.array_equal(got, expected)


def test_gather_refuses_prefill_positions():
    from runtime.dsv4_dspark import dspark_topk_idxs

    with pytest.raises(ValueError, match="start_pos"):
        dspark_topk_idxs(WINDOW, BLOCK, 0)


def test_block_positions_are_mutually_visible_not_causal():
    """A causal mask here would be a silent divergence from the released code."""
    from runtime.dsv4_dspark import dspark_topk_idxs

    idxs = np.array(dspark_topk_idxs(WINDOW, BLOCK, 64))[0]
    rows = {tuple(row.tolist()) for row in idxs}
    assert len(rows) == 1, "draft rows differ, so the gather became causal"
    tail = idxs[0][-BLOCK:]
    assert np.array_equal(tail, WINDOW + np.arange(BLOCK))


def test_draft_input_ids_layout():
    import mlx.core as mx

    from runtime.dsv4_dspark import draft_input_ids

    ids = np.array(draft_input_ids(1234, BLOCK, 999))
    assert ids.shape == (1, BLOCK)
    assert ids[0, 0] == 1234
    assert (ids[0, 1:] == 999).all(), "only slot 0 carries a real token"


def test_markov_bias_is_a_low_rank_bigram_map():
    import mlx.core as mx

    from runtime.dsv4_dspark import markov_bias

    vocab, rank = 32, 4
    rng = np.random.default_rng(0)
    w1 = mx.array(rng.normal(size=(vocab, rank)).astype(np.float32))
    w2 = mx.array(rng.normal(size=(vocab, rank)).astype(np.float32))
    bias, embed = markov_bias(mx.array([3], dtype=mx.int32), w1, w2)
    assert bias.shape == (1, vocab)
    assert embed.shape == (1, rank)
    expected = np.array(w1)[3] @ np.array(w2).T
    assert np.allclose(np.array(bias)[0], expected, atol=1e-5)


def test_sampling_is_sequential_through_the_markov_bias():
    """Position i's bias depends on the token sampled at i-1, so a vectorized
    rewrite that biases every position from the CURRENT token is wrong.

    Built rather than sampled: random logits let the chain settle on a fixed
    point, where chained and fixed-bias sampling agree and the test proves
    nothing. Here the Markov map sends token t to t+1, so the chain must count
    upward while a fixed bias repeats.
    """
    import mlx.core as mx

    from runtime.dsv4_dspark import dspark_sample_block

    vocab = 16
    # w1[t] = e_t, w2[v] = e_{v-1}  =>  bias(t)[v] is 1 exactly at v == t+1.
    w1 = mx.array(np.eye(vocab, dtype=np.float32))
    w2 = mx.array(np.roll(np.eye(vocab, dtype=np.float32), 1, axis=0))
    logits = mx.zeros((1, BLOCK, vocab))

    drafted, embeds = dspark_sample_block(logits, 5, w1, w2)
    assert len(drafted) == BLOCK
    assert embeds.shape == (1, BLOCK, vocab)
    assert drafted == [6, 7, 8, 9, 10], (
        f"chain did not follow the Markov map: {drafted}")

    fixed = []
    bias0 = np.array(w1)[5] @ np.array(w2).T
    for i in range(BLOCK):
        fixed.append(int(np.argmax(np.zeros(vocab) + bias0)))
    assert fixed == [6] * BLOCK
    assert drafted != fixed


def test_acceptance_stops_at_the_first_disagreement():
    from runtime.dsv4_dspark import accepted_prefix

    assert accepted_prefix([1, 2, 3], [1, 2, 3]) == 3
    assert accepted_prefix([1, 2, 3], [1, 9, 3]) == 1
    assert accepted_prefix([1, 2, 3], [9, 2, 3]) == 0
    # A later coincidental match must NOT be credited: it was conditioned on a
    # token the target rejected.
    assert accepted_prefix([1, 5, 7], [1, 6, 7]) == 1
    assert accepted_prefix([], [1]) == 0
