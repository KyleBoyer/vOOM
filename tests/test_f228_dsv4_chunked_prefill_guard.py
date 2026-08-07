"""F228: DeepSeek V4 must refuse chunked prefill rather than answer wrongly.

Measured on a 1550-token prompt ending in "the capital of France is":

  prefill_chunk_size 4096 (1 chunk)  -> answers it
  prefill_chunk_size 1024 (2 chunks) -> " of early agricultural societies..."
  prefill_chunk_size  512 (4 chunks) -> " still mapping today. Tin and"

Three chunk counts, three different continuations, none answering. Positions
older than the 128-slot window are reachable only through the compressed
region, and compress_topk_idxs has never been exercised at start_pos > 0 with
seqlen > 1 -- F214 covers (0, many) and (many, 1), not (many, many).

Every other validation in this work used single-chunk prompts of at most 353
tokens, which is exactly why this went unnoticed. Failing closed converts a
silently wrong long-prompt answer into an error naming the remedy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _guard_source():
    text = (ROOT / "runtime" / "engine.py").read_text()
    start = text.index("block_decode = offset > 0 and x.shape[1] > 1")
    return text[start:start + 2000]


def test_guard_triggers_on_a_block_wider_than_the_window():
    guard = _guard_source()
    assert "if block_decode and x.shape[1] > window:" in guard
    assert "NotImplementedError" in guard


def test_guard_names_the_remedy():
    """An error that does not say what to change is only half a guard."""
    guard = _guard_source()
    assert "prefill_chunk_size" in guard
    assert "VMODEL_DSV4_CHUNKED_PREFILL" in guard


def test_speculative_blocks_stay_below_the_threshold():
    """The guard must not break DSpark verification, which is validated.

    Its blocks are dspark_block_size positions; the window is window_size.
    """
    import json

    model = ROOT / "models" / "DeepSeek-V4-Flash-0731"
    if not (model / "config.json").is_file():
        pytest.skip("DeepSeek-V4-Flash-0731 not present")
    config = json.loads((model / "config.json").read_text())
    block = int(config.get("dspark_block_size", 5))
    window = int(config.get("sliding_window", 128))
    assert block < window, (
        f"draft block {block} is not inside the {window}-slot window, so the "
        "chunked-prefill guard would also refuse speculative verification")


def test_single_chunk_prefill_is_unaffected():
    """offset == 0 is the whole-prompt path and must not be guarded."""
    guard = _guard_source()
    assert "block_decode = offset > 0" in guard, (
        "the guard must key on offset > 0, so a single-chunk prefill at "
        "offset 0 never reaches it")
