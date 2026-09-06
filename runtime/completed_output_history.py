"""Bounded CPU-only proposals from earlier completed generated outputs.

The owning engine must serialize access and supply only actually generated token
IDs, never prompts or tool schemas. Completion is an explicit caller assertion;
this helper cannot determine whether an HTTP response was delivered or whether
an answer is correct. Returned proposals are NOT verified or safe to emit.

History is engine-local, nonserialized, and intended for an explicit single-
tenant opt-in. Namespace filtering is not authentication: sharing an engine can
still expose prior workload membership through proposal-dependent timing.

Accounting deliberately overcharges retained Python structures, namespace text,
and each bounded-size integer. It is a structural bound, not measured RSS. Call
temporaries are separately bounded; no model arrays, logits, or files are owned.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


_MAX_NAMESPACE_BYTES = 256
_MAX_VOCAB_SIZE = 1 << 31
_ACCOUNTED_BASE_BYTES = 4096
_ACCOUNTED_ENTRY_BYTES = 512
# Includes an integer object (at most 31 value bits) and its tuple reference.
_ACCOUNTED_TOKEN_BYTES = 64
# Conservatively allow a four-byte Unicode representation per UTF-8 byte.
_ACCOUNTED_NAMESPACE_MULTIPLIER = 4


@dataclass(frozen=True, slots=True)
class CompletedOutputProposal:
    tokens: tuple[int, ...] = ()
    match_length: int = 0


@dataclass(frozen=True, slots=True)
class _Output:
    namespace: str
    namespace_bytes: int
    tokens: tuple[int, ...]


_MISS = CompletedOutputProposal()


def _positive_int(value) -> bool:
    return type(value) is int and value > 0


def _namespace_bytes(namespace) -> int | None:
    # Bound encoding work before making even a temporary byte copy. Exact
    # builtin types avoid user-defined __len__/encode/iteration callbacks.
    if type(namespace) is not str or not 0 < len(namespace) <= _MAX_NAMESPACE_BYTES:
        return None
    try:
        length = len(namespace.encode("utf-8"))
    except UnicodeEncodeError:
        return None
    return length if length <= _MAX_NAMESPACE_BYTES else None


class CompletedOutputHistory:
    """Globally FIFO-bounded histories, searched only within one namespace.

    Each output remains a separate sequence. Selection prefers the longest
    emitted suffix, then the newest completed output, then its latest eligible
    occurrence. A candidate must have at least two following tokens.
    """

    def __init__(self, *, vocab_size: int, max_requests: int = 8,
                 max_tokens: int = 4096, max_request_tokens: int = 1024):
        for name, value in (("max_requests", max_requests),
                            ("max_tokens", max_tokens),
                            ("max_request_tokens", max_request_tokens)):
            if not _positive_int(value):
                raise ValueError(f"{name} must be a positive integer")
        if max_request_tokens > max_tokens:
            raise ValueError("max_request_tokens must not exceed max_tokens")
        if not _positive_int(vocab_size) or vocab_size >= _MAX_VOCAB_SIZE:
            raise ValueError("vocab_size must be an integer in [1, 2**31)")
        self.vocab_size = vocab_size
        self.max_requests = max_requests
        self.max_tokens = max_tokens
        self.max_request_tokens = max_request_tokens
        self._outputs: deque[_Output] = deque()
        self.clear()

    def _valid_tokens(self, tokens) -> bool:
        if type(tokens) not in (list, tuple):
            return False
        if not 0 < len(tokens) <= self.max_request_tokens:
            return False
        return all(type(token) is int and 0 <= token < self.vocab_size
                   for token in tokens)

    def add_output(self, namespace: str, tokens: list | tuple, *,
                   completed: bool) -> bool:
        """Remember a completed output; reject invalid input before copying.

        The caller must not mutate a sequence during this serialized call.
        Mutation after return cannot change the stored immutable sequence.
        Rejected or incomplete submissions do not evict valid history.
        """
        namespace_bytes = _namespace_bytes(namespace)
        if (type(completed) is not bool or not completed
                or namespace_bytes is None or not self._valid_tokens(tokens)):
            self._rejected_outputs += 1
            return False
        # A generator forces an independent tuple even when the input is one.
        sequence = tuple(token for token in tokens)
        entry = _Output(namespace, namespace_bytes, sequence)
        while self._outputs and (
                len(self._outputs) >= self.max_requests
                or self._retained_tokens + len(sequence) > self.max_tokens):
            removed = self._outputs.popleft()
            self._retained_tokens -= len(removed.tokens)
            self._retained_namespace_bytes -= removed.namespace_bytes
            self._evicted_outputs += 1
        self._outputs.append(entry)
        self._retained_tokens += len(sequence)
        self._retained_namespace_bytes += namespace_bytes
        self._added_outputs += 1
        return True

    def propose(self, namespace: str, emitted: list | tuple, *,
                max_tokens: int, min_match: int = 2,
                max_match: int = 6) -> CompletedOutputProposal:
        """Return an unverified proposal without inserting current output.

        Invalid input or a budget below two is a miss. Oversized numeric search
        bounds are clamped by the already-bounded emitted/stored sequences.
        No prompt or current-output prefix is a searchable proposal source.
        """
        self._proposal_calls += 1
        if (_namespace_bytes(namespace) is None or not self._valid_tokens(emitted)
                or not _positive_int(max_tokens) or max_tokens < 2
                or not _positive_int(min_match) or not _positive_int(max_match)
                or min_match > max_match):
            return _MISS
        longest = min(max_match, len(emitted), self.max_request_tokens - 2)
        for matched in range(longest, min_match - 1, -1):
            emitted_start = len(emitted) - matched
            for entry in reversed(self._outputs):
                if entry.namespace != namespace:
                    continue
                # Starting at len-matched-2 omits matches without two following
                # tokens; an older eligible occurrence can still be selected.
                for start in range(len(entry.tokens) - matched - 2, -1, -1):
                    if all(entry.tokens[start + index] == emitted[emitted_start + index]
                           for index in range(matched)):
                        continuation = start + matched
                        width = min(max_tokens, len(entry.tokens) - continuation)
                        self._proposal_hits += 1
                        return CompletedOutputProposal(
                            entry.tokens[continuation:continuation + width], matched)
        return _MISS

    def clear(self) -> None:
        """Drop helper-owned history and reset counts, including on close.

        The engine should call this when it closes or changes identity. Already
        returned immutable proposals remain owned by their callers.
        """
        self._outputs.clear()
        self._retained_tokens = 0
        self._retained_namespace_bytes = 0
        self._added_outputs = 0
        self._rejected_outputs = 0
        self._evicted_outputs = 0
        self._proposal_calls = 0
        self._proposal_hits = 0

    def telemetry(self) -> dict[str, int]:
        """Return fresh aggregate counts only; never text, tokens, or hashes."""
        entry_namespace_limit = (
            _ACCOUNTED_ENTRY_BYTES
            + _ACCOUNTED_NAMESPACE_MULTIPLIER * _MAX_NAMESPACE_BYTES)
        return {
            "retained_requests": len(self._outputs),
            "retained_tokens": self._retained_tokens,
            "retained_namespace_bytes": self._retained_namespace_bytes,
            "accounted_bytes": (
                _ACCOUNTED_BASE_BYTES
                + len(self._outputs) * _ACCOUNTED_ENTRY_BYTES
                + self._retained_namespace_bytes * _ACCOUNTED_NAMESPACE_MULTIPLIER
                + self._retained_tokens * _ACCOUNTED_TOKEN_BYTES),
            "accounted_byte_limit": (
                _ACCOUNTED_BASE_BYTES + self.max_requests * entry_namespace_limit
                + self.max_tokens * _ACCOUNTED_TOKEN_BYTES),
            # One staged insertion or emitted proposal, plus bounded namespace
            # encoding. This is separate from the retained-structure count.
            "operation_scratch_byte_limit": (
                entry_namespace_limit
                + self.max_request_tokens * _ACCOUNTED_TOKEN_BYTES
                + _MAX_NAMESPACE_BYTES * 4),
            "added_outputs": self._added_outputs,
            "rejected_outputs": self._rejected_outputs,
            "evicted_outputs": self._evicted_outputs,
            "proposal_calls": self._proposal_calls,
            "proposal_hits": self._proposal_hits,
        }
