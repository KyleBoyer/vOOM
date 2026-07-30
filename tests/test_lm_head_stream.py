from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from runtime.lm_head_stream import StreamedLMHead


def _make_head(tmp_path, *, vocab: int = 19, hidden: int = 8, block_rows: int = 5):
    values = np.arange(vocab * hidden, dtype=np.float32).reshape(vocab, hidden)
    values = np.sin(values / 11.0).astype(np.float32)
    weight = mx.array(values).astype(mx.bfloat16)
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"),
        {"lm_head.weight": weight},
    )
    return StreamedLMHead(
        tmp_path,
        {"lm_head.weight": "model.safetensors"},
        block_rows=block_rows,
    )


def test_serial_rows_are_bit_exact_to_independent_streamed_matmuls(tmp_path):
    head = _make_head(tmp_path)
    hidden = mx.array(
        np.cos(np.arange(3 * 8, dtype=np.float32).reshape(1, 3, 8) / 7.0)
    ).astype(mx.bfloat16)
    try:
        expected = mx.concatenate(
            [head.logits(hidden[:, row : row + 1]) for row in range(3)],
            axis=1,
        )
        actual = head.logits_serial_rows(hidden)
        mx.eval(expected, actual)
        assert actual.shape == (1, 3, 19)
        assert np.array_equal(
            np.array(actual.view(mx.uint16)),
            np.array(expected.view(mx.uint16)),
        )
    finally:
        head.close()


def test_serial_rows_read_each_vocab_block_once(tmp_path, monkeypatch):
    import runtime.lm_head_stream as module

    block_rows = 5
    head = _make_head(tmp_path, block_rows=block_rows)
    hidden = mx.ones((1, 4, 8), dtype=mx.bfloat16)
    calls = 0
    original = module._pread_exact

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_pread_exact", counted)
    try:
        result = head.logits_serial_rows(hidden)
        mx.eval(result)
    finally:
        head.close()

    assert calls == math.ceil(19 / block_rows)
