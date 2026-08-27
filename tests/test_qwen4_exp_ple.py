from __future__ import annotations

import random

import pytest

from runtime.qwen4_exp_ple import Qwen4ExpPLELayout


def _layout() -> Qwen4ExpPLELayout:
    return Qwen4ExpPLELayout.from_text_config({
        "vocab_size": 248_320,
        "hidden_size": 2_560,
        "eos_token_id": 248_044,
        "ngram_size": 3,
        "heads_per_ngram": 8,
        "ngram_vocab_size_base": 20_000_000,
        "make_ngram_vocab_size_divisible_by": 128,
    })


def test_released_layout_geometry_and_known_ids():
    layout = _layout()

    assert layout.context_len == 2
    assert layout.ngram_heads == 16
    assert layout.row_width == 160
    assert layout.row_bytes_bf16 == 320
    assert layout.bytes_per_token_bf16 == 5_120
    assert layout.padded_vocab_size % 128 == 0
    assert len(layout.head_vocab_sizes) == 16
    assert all(size >= 20_000_000 for size in layout.head_vocab_sizes)
    assert layout.row_ids([17, 23, 29]) == (
        (
            7961118, 35595600, 53154041, 65925962,
            86478454, 100794091, 133015820, 144041389,
            176216895, 183261499, 211010573, 235637741,
            252030661, 269649351, 289128081, 308564037,
        ),
        (
            11680906, 32824225, 53396086, 66196875,
            82883471, 115112459, 123455857, 147913839,
            174923579, 181978945, 218694756, 231526020,
            258829176, 266491735, 295792702, 302895670,
        ),
        (
            12667906, 33401711, 53769445, 78285190,
            90728089, 114875957, 131097864, 159394075,
            171380410, 193186418, 211961015, 231322423,
            242227407, 273132736, 290101276, 314943813,
        ),
    )


def test_chunked_continuation_matches_one_shot_across_eos_boundaries():
    layout = _layout()
    tokens = [9, 10, layout.eos_token_id, 11, 12, 13, 14]
    expected = layout.row_ids(tokens)

    first = layout.row_ids(tokens[:4])
    context = layout.next_context(tokens[:4])
    second = layout.row_ids(tokens[4:], previous_context=context)

    assert first + second == expected
    assert context == (layout.eos_token_id, 11)
    assert layout.next_context(tokens[4:], previous_context=context) == (13, 14)


def test_ids_match_independent_torch_integer_oracle():
    torch = pytest.importorskip("torch")
    layout = _layout()
    tokens = [17, 23, layout.eos_token_id, 29, 31]
    history = torch.tensor(
        [[layout.eos_token_id, layout.eos_token_id, *tokens]],
        dtype=torch.long,
    )
    positions = torch.arange(history.shape[1], dtype=torch.long)
    eos_positions = torch.where(history == layout.eos_token_id, positions, -1)
    previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
    previous_eos = torch.cat([
        eos_positions.new_full((1, 1), -1),
        previous_eos_inclusive[:, :-1],
    ], dim=1)
    position_in_segment = positions.unsqueeze(0) - (previous_eos + 1)
    shifted = []
    for shift in range(layout.ngram_size):
        source_positions = positions - shift
        gathered = history.gather(
            1, source_positions.clamp_min(0).unsqueeze(0))
        valid = (
            (position_in_segment >= shift)
            & (source_positions.unsqueeze(0) >= 0)
        )
        shifted.append(torch.where(
            valid, gathered, history.new_full((), layout.eos_token_id)))
    # Pinned Qwen/Qwen3.8-Flash-Next config (seed default 1234, PLE index 0)
    # evaluated by the released Transformers SplitMix64/prime construction.
    multipliers = torch.tensor(
        [23703573157769, 20109073645365, 8052911324071],
        dtype=torch.long,
    )
    sizes = torch.tensor((
        20000003, 20000023, 20000033, 20000047,
        20000059, 20000063, 20000069, 20000077,
        20000081, 20000093, 20000107, 20000147,
        20000153, 20000159, 20000161, 20000171,
    ), dtype=torch.long)
    offsets = torch.cat([
        torch.zeros(1, dtype=torch.long), sizes.cumsum(0)[:-1]])
    blocks = []
    for ngram in range(2, layout.ngram_size + 1):
        mixed = shifted[0] * multipliers[0]
        for prior in range(1, ngram):
            mixed = torch.bitwise_xor(
                mixed, shifted[prior] * multipliers[prior])
        start = (ngram - 2) * layout.heads_per_ngram
        end = start + layout.heads_per_ngram
        blocks.append(
            torch.remainder(mixed.unsqueeze(-1), sizes[start:end])
            + offsets[start:end]
        )
    expected = torch.cat(blocks, dim=-1)[:, -len(tokens):]

    assert layout.row_ids(tokens) == tuple(
        tuple(int(value) for value in row) for row in expected.tolist()[0])


def test_random_arbitrary_splits_match_one_shot():
    layout = _layout()
    rng = random.Random(3848)
    for length in (1, 2, 3, 17, 64):
        tokens = [rng.randrange(0, layout.unigram_vocab_size - 1)
                  for _ in range(length)]
        if length >= 3:
            tokens[length // 2] = layout.eos_token_id
        expected = layout.row_ids(tokens)
        for split in range(length + 1):
            left = layout.row_ids(tokens[:split])
            context = layout.next_context(tokens[:split])
            right = layout.row_ids(tokens[split:], previous_context=context)
            assert left + right == expected


def test_invalid_config_and_context_fail_closed():
    with pytest.raises(ValueError, match="eos_token_id"):
        Qwen4ExpPLELayout.from_text_config({"eos_token_id": []})
    with pytest.raises(ValueError, match="divide"):
        Qwen4ExpPLELayout(
            unigram_vocab_size=100, eos_token_id=99, embedding_dim=17)
    layout = _layout()
    with pytest.raises(ValueError, match="previous_context"):
        layout.row_ids([1], previous_context=[2])
    with pytest.raises(ValueError, match="outside"):
        layout.row_ids([layout.unigram_vocab_size])
