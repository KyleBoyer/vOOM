"""Convert a tiktoken-based tokenizer (Kimi family: no `tokenizer.json`, only
`tiktoken.model` + a custom `tokenization_kimi.py`) into an ordinary fast
`tokenizers.Tokenizer`, so engine.py's `Tokenizer.from_file(...)`-shaped call
sites work unchanged for these checkpoints.

F92/F93 finding (2026-07-18): neither Kimi-Linear-48B-A3B-Instruct nor
Kimi-K2.5 ship a `tokenizer.json` -- only a raw tiktoken BPE vocab file and a
`transformers`-style slow tokenizer class (`TikTokenTokenizer` in
`tokenization_kimi.py`, loaded via `auto_map`). Without a working tokenizer
the server cannot encode/decode text for these models at all, independent of
how correct the model math is.

Uses `transformers.convert_slow_tokenizer.TikTokenConverter` -- the official
HF conversion utility, not a hand-rolled one -- but does NOT hardcode Kimi's
regex split pattern or special-token list: both are read directly from the
checkpoint's own real `tokenization_kimi.py` (via `AutoTokenizer` +
`trust_remote_code=True`, so if a future Moonshot checkpoint changes either,
this keeps working without a code change) and `tokenizer_config.json`.

Verified (2026-07-18, tests/test_tiktoken_convert.py) to produce IDENTICAL
token IDs to the real slow tokenizer on English, CJK/mixed-script, code
(newlines/indentation), and special-token test strings for both
Kimi-Linear-48B-A3B-Instruct and Kimi-K2.5.
"""

from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer


def build_kimi_fast_tokenizer(model_dir: str | Path) -> Tokenizer:
    """Build a fast `tokenizers.Tokenizer` equivalent to the checkpoint's
    real slow `TikTokenTokenizer`, for a model directory that has
    `tiktoken.model` + `tokenization_kimi.py` but no `tokenizer.json`."""
    model_dir = Path(model_dir)
    from transformers import AutoTokenizer
    from transformers.convert_slow_tokenizer import TikTokenConverter

    # trust_remote_code=True runs the checkpoint's own tokenization_kimi.py --
    # this is what gives us the exact pat_str/special-token construction
    # logic without duplicating it here (see module docstring).
    real = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    pat_str = type(real).pat_str
    num_reserved = type(real).num_reserved_special_tokens

    tok_cfg = json.loads((model_dir / "tokenizer_config.json").read_text())
    added_tokens_decoder = tok_cfg.get("added_tokens_decoder", {})

    from tiktoken.load import load_tiktoken_bpe

    vocab_file = model_dir / "tiktoken.model"
    mergeable_ranks = load_tiktoken_bpe(str(vocab_file))
    num_base_tokens = len(mergeable_ranks)

    # Replicates TikTokenTokenizer.__init__'s special_tokens construction
    # exactly: IDs [num_base_tokens, num_base_tokens + num_reserved + 2) map
    # to their real content where tokenizer_config.json names them, else a
    # generic reserved-token placeholder -- same fallback, same order, so
    # the fast tokenizer's added-token IDs land in the identical positions.
    special_tokens = [
        (added_tokens_decoder[str(i)]["content"] if str(i) in added_tokens_decoder
         else f"<|reserved_token_{i}|>")
        for i in range(num_base_tokens, num_base_tokens + num_reserved + 2)
    ]

    converter = TikTokenConverter(
        vocab_file=str(vocab_file), pattern=pat_str, extra_special_tokens=special_tokens,
    )
    return converter.converted()


def has_tiktoken_tokenizer(model_dir: str | Path) -> bool:
    model_dir = Path(model_dir)
    return (not (model_dir / "tokenizer.json").exists()
            and (model_dir / "tiktoken.model").exists()
            and (model_dir / "tokenization_kimi.py").exists())
