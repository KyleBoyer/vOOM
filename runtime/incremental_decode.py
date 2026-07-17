"""Stateful streaming detokenization shared by text and vision generation."""

from __future__ import annotations

import codecs
from functools import lru_cache


@lru_cache(maxsize=1)
def _byte_level_inverse() -> dict[str, int]:
    """GPT-2/ByteLevel unicode alphabet -> original byte value."""
    byte_values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    codepoints = list(byte_values)
    extra = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            codepoints.append(256 + extra)
            extra += 1
    return {
        chr(codepoint): value
        for value, codepoint in zip(byte_values, codepoints)
    }


class IncrementalDetokenizer:
    """Emit only decoded text that cannot be revised by a later token.

    Byte-level tokenizers can decode an incomplete byte sequence as U+FFFD and
    replace it when the next token arrives, so concatenating ``decode([id])`` is
    incorrect. ``push_token`` carries UTF-8 decoder state for raw ByteLevel
    tokenizers; ``push`` retains the complete-prefix reference behavior. Both
    retain text that could still become a configured stop string.
    """

    def __init__(self, tokenizer, stop_sequences=()):
        self.tokenizer = tokenizer
        self.stop_sequences = tuple(value for value in stop_sequences if value)
        self.emitted = ""
        self._stream_ids: list[int] = []
        self._stream_emitted: list[str] = []
        self._stream_pending = ""
        self._stream_stop_text: str | None = None
        self._stream_stop_sequence: str | None = None
        self._stream_finished = False
        decoder = getattr(tokenizer, "decoder", None)
        self._byte_level = bool(
            decoder is not None
            and decoder.__class__.__name__ == "ByteLevel"
            and callable(getattr(tokenizer, "id_to_token", None)))
        self._byte_cache: dict[int, bytes | None] = {}
        self._special_ids = set()
        if self._byte_level:
            added = getattr(tokenizer, "get_added_tokens_decoder", lambda: {})()
            self._special_ids = {
                int(token_id) for token_id, token in added.items()
                if getattr(token, "special", False)
            }
            self._utf8 = codecs.getincrementaldecoder("utf-8")(
                errors="replace")

    def _advance(self, text: str) -> str:
        if not text.startswith(self.emitted):
            raise RuntimeError("incremental decoded text revised an emitted prefix")
        delta = text[len(self.emitted):]
        self.emitted = text
        return delta

    def push(self, token_ids: list[int]) -> str:
        text = self.tokenizer.decode(token_ids)
        # U+FFFD at the end may represent one or more incomplete byte-fallback
        # tokens. A genuine final replacement character is flushed by finish().
        safe_end = len(text.rstrip("\ufffd"))
        safe_text = text[:safe_end]
        if self.stop_sequences:
            max_stop = max(map(len, self.stop_sequences))
            start = max(len(self.emitted), len(safe_text) - max_stop + 1)
            for index in range(start, len(safe_text)):
                suffix = safe_text[index:]
                if any(stop.startswith(suffix) for stop in self.stop_sequences):
                    safe_text = safe_text[:index]
                    break
        return self._advance(safe_text)

    def finish(self, token_ids: list[int], *, final_text: str | None = None) -> str:
        text = self.tokenizer.decode(token_ids) if final_text is None else final_text
        return self._advance(text)

    @property
    def matched_stop_sequence(self) -> str | None:
        return self._stream_stop_sequence

    @property
    def stop_text(self) -> str | None:
        return self._stream_stop_text

    def _find_stop(self, text: str):
        matches = []
        for index, value in enumerate(self.stop_sequences):
            position = text.find(value)
            if position >= 0:
                matches.append((position, index, value))
        return min(matches) if matches else None

    def _safe_stream_prefix(self, text: str) -> tuple[str, str]:
        """Split callback-safe text from a possible stop/UTF-8 suffix."""
        safe_end = len(text.rstrip("\ufffd"))
        safe_text = text[:safe_end]
        hold_at = len(safe_text)
        if self.stop_sequences:
            max_stop = max(map(len, self.stop_sequences))
            for index in range(max(0, len(safe_text) - max_stop + 1),
                               len(safe_text)):
                suffix = safe_text[index:]
                if any(stop.startswith(suffix)
                       for stop in self.stop_sequences):
                    hold_at = index
                    break
        return text[:hold_at], text[hold_at:]

    def _accept_stream_delta(self, delta: str, preview: str = "") -> str:
        self._stream_pending += delta
        match = self._find_stop(self._stream_pending + preview)
        if match is not None:
            cut, _order, self._stream_stop_sequence = match
            self._stream_stop_text = (
                "".join(self._stream_emitted)
                + (self._stream_pending + preview)[:cut])
            return ""
        safe, self._stream_pending = self._safe_stream_prefix(
            self._stream_pending)
        if safe:
            self._stream_emitted.append(safe)
        return safe

    def _push_full_stream(self) -> str:
        """Exact fallback for non-ByteLevel or unexpected added tokens."""
        text = self.tokenizer.decode(self._stream_ids)
        match = self._find_stop(text)
        if match is not None:
            cut, _order, self._stream_stop_sequence = match
            self._stream_stop_text = text[:cut]
            return ""
        emitted = "".join(self._stream_emitted)
        if not text.startswith(emitted):
            raise RuntimeError(
                "incremental decoded text revised an emitted prefix")
        safe_end = len(text.rstrip("\ufffd"))
        safe_text = text[:safe_end]
        if self.stop_sequences:
            max_stop = max(map(len, self.stop_sequences))
            start = max(len(emitted), len(safe_text) - max_stop + 1)
            for index in range(start, len(safe_text)):
                suffix = safe_text[index:]
                if any(stop.startswith(suffix)
                       for stop in self.stop_sequences):
                    safe_text = safe_text[:index]
                    break
        delta = safe_text[len(emitted):]
        if delta:
            self._stream_emitted.append(delta)
        self._stream_pending = ""
        return delta

    def _token_bytes(self, token_id: int) -> bytes | None:
        cached = self._byte_cache.get(token_id, ...)
        if cached is not ...:
            return cached
        if token_id in self._special_ids:
            value = b""
        else:
            token = self.tokenizer.id_to_token(token_id)
            if token is None:
                value = None
            else:
                inverse = _byte_level_inverse()
                try:
                    value = bytes(inverse[character] for character in token)
                except KeyError:
                    # Added non-special tokens can bypass ByteLevel's alphabet.
                    # Falling back to the released tokenizer is exact and rare.
                    value = None
        self._byte_cache[token_id] = value
        return value

    def push_token(self, token_id: int) -> str:
        """Consume one new token without re-decoding the generated prefix.

        Raw ``tokenizers.Tokenizer`` ByteLevel models (including Qwen3-VL)
        follow their exact byte decoder incrementally. Other tokenizer shapes
        retain the full-prefix reference behavior.
        """
        if self._stream_finished:
            raise RuntimeError("cannot push after incremental decoding finished")
        if self._stream_stop_sequence is not None:
            raise RuntimeError("cannot push after a stop sequence matched")
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError("token id must be an integer")
        self._stream_ids.append(token_id)
        if not self._byte_level:
            return self._push_full_stream()

        token_bytes = self._token_bytes(token_id)
        if token_bytes is None:
            self._byte_level = False
            return self._push_full_stream()
        delta = self._utf8.decode(token_bytes, final=False)
        buffered = self._utf8.getstate()[0]
        preview = buffered.decode("utf-8", errors="replace")
        return self._accept_stream_delta(delta, preview)

    def finish_token_stream(
        self, *, final_text: str | None = None,
    ) -> tuple[str, str]:
        """Finalize ``push_token`` state as ``(callback_delta, full_text)``."""
        if self._stream_finished:
            target = self.emitted if final_text is None else final_text
            return "", target
        self._stream_finished = True
        if final_text is not None:
            target = final_text
        elif self._stream_stop_text is not None:
            target = self._stream_stop_text
        elif self._byte_level:
            tail = self._utf8.decode(b"", final=True)
            target = (
                "".join(self._stream_emitted)
                + self._stream_pending + tail)
        else:
            target = self.tokenizer.decode(self._stream_ids)
        streamed = "".join(self._stream_emitted)
        if not target.startswith(streamed):
            raise RuntimeError(
                "final decoded text revised an emitted prefix")
        delta = target[len(streamed):]
        self.emitted = target
        return delta, target
