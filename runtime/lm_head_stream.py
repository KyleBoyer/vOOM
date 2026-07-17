"""F02 (second half): block-streamed LM-head argmax/logits.

The LM head (GLM-5.2: 154,880 x 6144 bf16 ~= 1.9 GB) is only ever used as
`normed_hidden @ lm_head.T` — a matvec (decode) or thin matmul (verify) whose
CONTRACTION dimension is hidden, not vocab. Splitting the OUTPUT (vocab)
dimension into row blocks therefore changes nothing about summation order:
each output logit is an independent dot product over the same hidden-sized
row, computed once, whichever block it lands in. Block-streamed logits are
bit-identical to a single whole-tensor matmul — this is a real optimization,
not an approximation.

`mx.load(...)[name]` is lazy per-TENSOR but not per-SLICE: evaluating any
slice of a lazy tensor forces the whole tensor to be read (measured directly
on Qwen2.5-0.5B's embed_tokens.weight: evaluating a 1000-row slice of a
272.3 MB tensor still peaked at 272.27 MB). So this bypasses mx.load entirely
for lm_head.weight and reads raw bytes straight from the safetensors shard via
seek/pread, mirroring the row-paged technique in embed_rows.py but swept in
order across the whole vocab each call instead of cached by row index.
"""
from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import mlx.core as mx
import numpy as np

_DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4}
_MX_DTYPE = {"BF16": mx.bfloat16, "F16": mx.float16, "F32": mx.float32}
_NP_STORAGE_DTYPE = {"BF16": np.uint16, "F16": np.uint16, "F32": np.uint32}


def _pread_exact(fd: int, size: int, offset: int) -> bytes:
    """Read exactly one tensor extent or fail instead of reshaping short data."""
    parts = []
    done = 0
    while done < size:
        chunk = os.pread(fd, size - done, offset + done)
        if not chunk:
            raise IOError(f"short LM-head read at {offset}: {done}/{size} bytes")
        parts.append(chunk)
        done += len(chunk)
    return b"".join(parts)


class StreamedLMHead:
    """Drop-in replacement for a materialized lm_head.weight mx.array. Pass an
    instance of this where `layer_runner.final_logits`/`all_logits` expect the
    weight tensor; they special-case it to call `.logits(h)` in row blocks
    instead of a single `quant.matmul`."""

    def __init__(self, model_dir, weight_map: dict, name: str = "lm_head.weight",
                 block_rows: int = 16384):
        self.model_dir = Path(model_dir)
        self.weight_map = weight_map
        self.name = name
        self.block_rows = block_rows
        self._open()

    def _open(self):
        path = self.model_dir / self.weight_map[self.name]
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        meta = header[self.name]
        self.vocab, self.hidden = meta["shape"]
        self.dtype = meta["dtype"]
        self.row_bytes = self.hidden * _DTYPE_BYTES[self.dtype]
        self.data_start = 8 + n + meta["data_offsets"][0]
        self.path = path
        self._fd = os.open(path, os.O_RDONLY)

    def close(self):
        os.close(self._fd)

    def logits(self, h: mx.array) -> mx.array:
        """h: (..., hidden) already rms-normed. Returns (..., vocab). Peak
        Metal cost per block is O(block_rows * hidden), not O(vocab * hidden)."""
        mx.eval(h)
        chunks = []
        for start in range(0, self.vocab, self.block_rows):
            n_rows = min(self.block_rows, self.vocab - start)
            raw = _pread_exact(self._fd, n_rows * self.row_bytes,
                               self.data_start + start * self.row_bytes)
            block = np.frombuffer(
                raw, dtype=_NP_STORAGE_DTYPE[self.dtype]
            ).reshape(n_rows, self.hidden)
            w_block = mx.array(block).view(_MX_DTYPE[self.dtype])
            c = h @ w_block.T
            mx.eval(c)
            chunks.append(c)
        return mx.concatenate(chunks, axis=-1)
