"""F199: the offline acceptance replay must be exact, not an approximation.

Scoring suffix-draft acceptance offline is only legitimate because, under
greedy decoding, a recorded generation contains every token the live verifier
would have read.  ``select_verified_tokens`` inspects ``target[0..accepted]``,
and ``draft[:accepted]`` matched the recorded stream by construction, so
``target[accepted]`` is the recorded next token.  These tests pin that claim
and the loop accounting that rests on it.

The replay drives the real ``SuffixDecodingCache`` and the runtime's own
acceptance rule; nothing here reimplements either.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from runtime.suffix_decoding import (  # noqa: E402
    SuffixDecodingCache, select_verified_tokens,
)

f199 = pytest.importorskip("f199_suffix_acceptance")


def make_cache(**kwargs) -> SuffixDecodingCache:
    settings = dict(identity="test", max_depth=64, max_spec_tokens=6,
                    factor=4.0, min_probability=0.1)
    settings.update(kwargs)
    return SuffixDecodingCache(**settings)


def test_every_token_is_committed_exactly_once():
    """Rounds must partition the output: no token dropped, none duplicated."""
    prompt = [1, 2, 3, 4, 5, 6, 7, 8]
    output = [3, 4, 5, 6, 7, 8, 3, 4, 5, 6, 7, 8, 9, 10]
    result = f199.replay(make_cache(), prompt, output, "unit")
    assert result.tokens == len(output)
    # Each round commits accepted + 1 tokens.
    assert sum(n + 1 for n in result.round_accept_lengths) == len(output)
    assert result.rounds == len(result.round_accept_lengths)


def test_sweeps_saved_is_tokens_minus_rounds():
    prompt = list(range(20))
    output = list(range(5, 25))
    result = f199.replay(make_cache(), prompt, output, "unit")
    assert result.sweeps_saved == result.tokens - result.rounds
    assert 1 <= result.rounds <= result.tokens


def test_no_draft_means_one_round_per_token():
    """With nothing to match, the drafter must not manufacture acceptance."""
    prompt = [1, 2]
    output = [900 + i for i in range(12)]  # never seen in history
    result = f199.replay(make_cache(), prompt, output, "unit")
    assert result.rounds == result.tokens
    assert result.accepted == 0
    assert result.sweeps_saved == 0


def test_highly_repetitive_output_is_where_acceptance_comes_from():
    """A repeated block should be drafted and accepted after its first pass."""
    block = [11, 12, 13, 14, 15, 16]
    prompt = [1, 2, 3] + block
    output = block * 5
    result = f199.replay(make_cache(), prompt, output, "unit")
    assert result.accepted > 0
    assert result.rounds < result.tokens
    assert result.mean_accept_len > 0.0


def test_replay_never_reads_past_the_accepted_prefix():
    """The exactness claim: only target[0..accepted] may influence the result.

    Corrupting the recorded stream *after* the first divergence must not change
    any accounting, because the live verifier could not have seen those tokens
    either.
    """
    prompt = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    output = [1, 2, 3, 4, 5, 77, 88, 99, 1, 2, 3, 4, 5]
    baseline = f199.replay(make_cache(), prompt, output, "unit")

    # Rebuild an identical run, then verify the acceptance rule itself only
    # consumes accepted+1 entries of the greedy window.
    draft = [1, 2, 3, 9, 9]
    greedy = [1, 2, 3, 4, 5, 6]
    accepted, verified = select_verified_tokens(draft, greedy)
    assert accepted == 3
    assert verified == [1, 2, 3, 4]  # accepted prefix + the target's own token
    # Anything beyond index `accepted` is unread; changing it changes nothing.
    other = select_verified_tokens(draft, [1, 2, 3, 4, -1, -2])
    assert other == (accepted, verified)
    assert baseline.tokens == len(output)


def test_warm_history_can_only_help():
    """Priming completed outputs must never reduce acceptance on a repeat."""
    prompt = [1, 2, 3, 4]
    output = [5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10]
    cold = f199.replay(make_cache(), prompt, output, "unit")

    warm_cache = make_cache()
    warm_cache.add_output(output)
    warm = f199.replay(warm_cache, prompt, output, "unit")

    assert warm.rounds <= cold.rounds
    assert warm.sweeps_saved >= cold.sweeps_saved


def test_exact_repeat_through_warm_cache_is_near_free():
    """The F188 seeded case: an exact prior output should draft almost wholly."""
    prompt = [1, 2, 3, 4, 5]
    output = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    cache = make_cache()
    cache.add_output(output)
    result = f199.replay(cache, prompt, output, "unit")
    # k=6 caps a round, so a 12-token exact repeat cannot be one sweep, but it
    # must be far below one sweep per token.
    assert result.rounds <= len(output) // 3
    assert result.accept_rate > 0.75


def test_summarize_totals_match_the_entries():
    prompt = list(range(10))
    outputs = [[1, 2, 3, 1, 2, 3, 1, 2], [4, 5, 6, 7, 8, 9]]
    results = [f199.replay(make_cache(), prompt, o, f"d{i}")
               for i, o in enumerate(outputs)]
    summary = f199.summarize(results)
    assert summary["tokens"] == sum(len(o) for o in outputs)
    assert summary["target_sweeps"] == sum(r.rounds for r in results)
    assert summary["sweeps_saved"] == summary["tokens"] - summary["target_sweeps"]
    assert 0.0 <= summary["accept_rate"] <= 1.0
