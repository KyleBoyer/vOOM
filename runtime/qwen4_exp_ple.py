"""Pure Qwen4-Exp PLE n-gram row addressing.

This module isolates the released integer-only address calculation from the
95GiB embedding table so storage planning and row paging never need to import
Torch or MLX. The algorithm is adapted from Hugging Face Transformers'
``modeling_qwen4_exp.py`` (copyright 2026 Qwen Team and Hugging Face Inc.,
Apache-2.0). No model tensor is loaded here.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import math
from typing import Mapping, Sequence


_MASK64 = (1 << 64) - 1
_SIGN64 = 1 << 63
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def _signed_i64(value: int) -> int:
    value &= _MASK64
    return value - (1 << 64) if value & _SIGN64 else value


def _multiply_i64(left: int, right: int) -> int:
    return _signed_i64((left & _MASK64) * (right & _MASK64))


def _xor_i64(left: int, right: int) -> int:
    return _signed_i64((left & _MASK64) ^ (right & _MASK64))


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, math.isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def _find_nth_prime_after(start: int, count: int) -> int:
    if count <= 0:
        raise ValueError("prime count must be positive")
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


def _build_layer_multipliers(
    unigram_vocab_size: int,
    ngram_size: int,
    ple_layer_index: int,
    seed: int,
) -> tuple[int, ...]:
    max_long = (1 << 63) - 1
    multiplier_max = max_long // max(unigram_vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + _PRIME_1 * ple_layer_index
    values = []
    for index in range(ngram_size):
        value = (base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64
        values.append(2 * (_splitmix64(value) % half_bound) + 1)
    return tuple(values)


def _shift_right_ignore_eos(
    token_ids: Sequence[int], shift: int, eos_token_id: int,
) -> list[int]:
    if shift == 0:
        return list(token_ids)
    shifted: list[int] = []
    previous_eos = -1
    for position, token in enumerate(token_ids):
        source = position - shift
        valid = position - (previous_eos + 1) >= shift and source >= 0
        shifted.append(int(token_ids[source]) if valid else eos_token_id)
        if token == eos_token_id:
            previous_eos = position
    return shifted


@dataclass(frozen=True)
class Qwen4ExpPLELayout:
    """Released PLE vocabulary layout and exact row-ID generator."""

    unigram_vocab_size: int
    eos_token_id: int
    embedding_dim: int = 2560
    ngram_size: int = 3
    heads_per_ngram: int = 8
    ngram_vocab_size_base: int = 20_000_000
    make_vocab_divisible_by: int = 128
    ple_layer_index: int = 0
    seed: int = 1234

    def __post_init__(self) -> None:
        if self.unigram_vocab_size <= 0:
            raise ValueError("unigram vocabulary size must be positive")
        if self.eos_token_id < 0:
            raise ValueError("EOS token ID must be non-negative")
        if self.ngram_size < 2 or self.heads_per_ngram <= 0:
            raise ValueError("PLE requires ngram_size >= 2 and positive heads")
        if self.ngram_vocab_size_base < 2:
            raise ValueError("n-gram vocabulary base must be at least two")
        if self.make_vocab_divisible_by <= 0:
            raise ValueError("vocabulary divisor must be positive")
        if self.embedding_dim <= 0 or self.embedding_dim % self.ngram_heads:
            raise ValueError("embedding dimension must divide across n-gram heads")

    @classmethod
    def from_text_config(
        cls, config: Mapping[str, object], *, ple_layer_index: int = 0,
    ) -> "Qwen4ExpPLELayout":
        eos = config.get("eos_token_id")
        if isinstance(eos, list):
            if not eos:
                raise ValueError("eos_token_id list is empty")
            eos = eos[0]
        if isinstance(eos, bool) or not isinstance(eos, int):
            raise ValueError("eos_token_id must be an integer")
        return cls(
            unigram_vocab_size=int(config.get("vocab_size", 248_320)),
            eos_token_id=eos,
            embedding_dim=int(
                config.get("ple_embed_dim")
                or config.get("hidden_size", 2_048)),
            ngram_size=int(config.get("ngram_size", 3)),
            heads_per_ngram=int(config.get("heads_per_ngram", 8)),
            ngram_vocab_size_base=int(
                config.get("ngram_vocab_size_base", 20_000_000)),
            make_vocab_divisible_by=int(
                config.get("make_ngram_vocab_size_divisible_by", 128)),
            ple_layer_index=ple_layer_index,
            seed=int(config.get("seed", 1234)),
        )

    @property
    def context_len(self) -> int:
        return self.ngram_size - 1

    @property
    def ngram_heads(self) -> int:
        return self.context_len * self.heads_per_ngram

    @property
    def row_width(self) -> int:
        return self.embedding_dim // self.ngram_heads

    @property
    def row_bytes_bf16(self) -> int:
        return self.row_width * 2

    @property
    def bytes_per_token_bf16(self) -> int:
        return self.ngram_heads * self.row_bytes_bf16

    @cached_property
    def head_vocab_sizes(self) -> tuple[int, ...]:
        return tuple(
            _find_nth_prime_after(
                self.ngram_vocab_size_base - 1,
                self.ple_layer_index * self.ngram_heads + head + 1,
            )
            for head in range(self.ngram_heads)
        )

    @cached_property
    def head_offsets(self) -> tuple[int, ...]:
        offsets = []
        total = 0
        for size in self.head_vocab_sizes:
            offsets.append(total)
            total += size
        return tuple(offsets)

    @cached_property
    def total_vocab_size(self) -> int:
        return sum(self.head_vocab_sizes)

    @cached_property
    def padded_vocab_size(self) -> int:
        divisor = self.make_vocab_divisible_by
        return math.ceil(self.total_vocab_size / divisor) * divisor

    @cached_property
    def multipliers(self) -> tuple[int, ...]:
        return _build_layer_multipliers(
            self.unigram_vocab_size,
            self.ngram_size,
            self.ple_layer_index,
            self.seed,
        )

    def row_ids(
        self,
        input_ids: Sequence[int],
        *,
        previous_context: Sequence[int] | None = None,
    ) -> tuple[tuple[int, ...], ...]:
        """Return the released global PLE row IDs for each new token."""
        tokens = tuple(int(token) for token in input_ids)
        if previous_context is None:
            context = (self.eos_token_id,) * self.context_len
        else:
            context = tuple(int(token) for token in previous_context)
            if len(context) != self.context_len:
                raise ValueError(
                    f"previous_context must contain {self.context_len} tokens")
        if not tokens:
            return ()
        for token in (*context, *tokens):
            if not 0 <= token < self.unigram_vocab_size:
                raise ValueError(f"token ID {token} is outside the vocabulary")

        history = context + tokens
        shifted = tuple(
            _shift_right_ignore_eos(history, shift, self.eos_token_id)
            for shift in range(self.ngram_size)
        )
        sizes = self.head_vocab_sizes
        offsets = self.head_offsets
        multipliers = self.multipliers
        result: list[tuple[int, ...]] = []
        for position in range(len(context), len(history)):
            row: list[int] = []
            for ngram in range(2, self.ngram_size + 1):
                mixed = _multiply_i64(shifted[0][position], multipliers[0])
                for prior in range(1, ngram):
                    mixed = _xor_i64(
                        mixed,
                        _multiply_i64(
                            shifted[prior][position], multipliers[prior]),
                    )
                head_start = (ngram - 2) * self.heads_per_ngram
                for head in range(head_start, head_start + self.heads_per_ngram):
                    row.append(mixed % sizes[head] + offsets[head])
            result.append(tuple(row))
        return tuple(result)

    def next_context(
        self,
        input_ids: Sequence[int],
        *,
        previous_context: Sequence[int] | None = None,
    ) -> tuple[int, ...]:
        context = (
            (self.eos_token_id,) * self.context_len
            if previous_context is None
            else tuple(int(token) for token in previous_context)
        )
        if len(context) != self.context_len:
            raise ValueError(
                f"previous_context must contain {self.context_len} tokens")
        return (context + tuple(int(token) for token in input_ids))[
            -self.context_len:]
