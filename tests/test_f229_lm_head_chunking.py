"""F229: the LM head projection must not materialize its transpose.

h @ w.T on a [vocab, hidden] head materializes the transposed operand.
Measured on a real 1.06GB bfloat16 head: 2.12GB of transient, versus 0.00GB
when the projection is chunked over vocabulary rows. That was the single
largest spike in a DeepSeek V4 request -- final_logits took peak from 5.87GB
to 7.99GB on a 16K prompt, a jump of exactly 2.12GB -- and it is constant, so
it applies to every model with a large untied head.

Chunking is arithmetically identical: the vocabulary axis is the output axis,
so splitting it splits the result.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_chunked_projection_matches_the_direct_one():
    import mlx.core as mx

    from runtime.layer_runner import _chunked_head_matmul, _LM_HEAD_ROW_CHUNK

    rng = np.random.default_rng(0)
    vocab = _LM_HEAD_ROW_CHUNK * 2 + 7      # forces an uneven final chunk
    weight = mx.array(rng.normal(size=(vocab, 64)).astype(np.float32))
    h = mx.array(rng.normal(size=(1, 1, 64)).astype(np.float32))

    direct = h @ weight.T
    chunked = _chunked_head_matmul(h, weight)
    mx.eval(direct, chunked)
    assert chunked.shape == direct.shape
    assert float(mx.max(mx.abs(direct - chunked))) == 0.0


def test_chunking_covers_every_vocabulary_row():
    """An off-by-one in the chunk loop silently truncates the vocabulary."""
    import mlx.core as mx

    from runtime.layer_runner import _chunked_head_matmul, _LM_HEAD_ROW_CHUNK

    for extra in (0, 1, _LM_HEAD_ROW_CHUNK - 1):
        vocab = _LM_HEAD_ROW_CHUNK + extra
        weight = mx.zeros((vocab, 8))
        out = _chunked_head_matmul(mx.zeros((1, 1, 8)), weight)
        assert out.shape[-1] == vocab, (
            f"vocab {vocab}: projected {out.shape[-1]} rows")


def test_small_heads_are_left_alone():
    """Below the threshold there is nothing to gain; keep the direct path."""
    import mlx.core as mx

    from runtime.layer_runner import _chunked_head_matmul, _LM_HEAD_ROW_CHUNK

    weight = mx.ones((_LM_HEAD_ROW_CHUNK - 1, 4))
    out = _chunked_head_matmul(mx.ones((1, 1, 4)), weight)
    assert out.shape == (1, 1, _LM_HEAD_ROW_CHUNK - 1)


def test_all_logits_and_final_logits_agree_on_the_last_row():
    import mlx.core as mx

    from runtime.layer_runner import all_logits, final_logits

    rng = np.random.default_rng(1)
    vocab, hidden, seq = 40000, 32, 5
    weight = mx.array(rng.normal(size=(vocab, hidden)).astype(np.float32))
    norm = mx.ones((hidden,), mx.float32)
    x = mx.array(rng.normal(size=(1, seq, hidden)).astype(np.float32))

    last = final_logits(x, norm, weight, 1e-6)
    every = all_logits(x, norm, weight, 1e-6)
    mx.eval(last, every)
    assert last.shape == (vocab,)
    assert every.shape == (seq, vocab)
    assert float(mx.max(mx.abs(last - every[-1]))) < 1e-4
