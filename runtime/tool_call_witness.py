"""Content-free observations of engine-authored Hermes frames, not a parser.

This diagnostic never changes output, executes a tool, or validates a schema.
Unlike the serving regex it uses JSONDecoder boundaries, so a closing-tag
literal inside an argument string cannot terminate a frame. Lexical marker
counts are labeled separately: quoted examples are not proof of an invocation.
"""

from __future__ import annotations

import hashlib
import json
import math


class _ObservationLimit(Exception):
    pass


class _NonFiniteNumber(ValueError):
    pass


def _bounded_integer(value: str) -> int:
    if len(value.lstrip("-")) > 4096:
        raise _ObservationLimit("integer-limit")
    try:
        return int(value)
    except ValueError:
        # JSONDecoder supplies syntactically valid integer text; a stricter
        # interpreter digit ceiling is an observer limit, not malformed JSON.
        raise _ObservationLimit("integer-limit") from None


def _finite_number(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise _NonFiniteNumber("nonfinite JSON number")
    return number


def hermes_text_witness(text: str, *, max_text_chars: int = 1_048_576,
                        max_frames: int = 256) -> dict:
    """Observe the unchanged raw engine text with bounded, no-content output.

    Canonical digests hash ASCII-escaped sorted compact JSON containing only
    name and argument object, matching the private HTTP replay fixture. They
    omit random protocol call IDs. A well-framed JSON call is NOT necessarily
    allowed, schema-valid, outside reasoning/examples, or semantically correct.
    """
    base = {
        "schema": "voom.hermes-text-witness.v1",
        "scope": "raw_engine_text_before_protocol_parsing",
        "formats_observed": ["hermes-json"],
        "canonical_encoding": "ascii-escaped-sorted-compact-name-arguments-json",
        "semantic_validation": False,
    }
    if (type(max_text_chars) is not int or max_text_chars <= 0
            or type(max_frames) is not int or max_frames <= 0):
        raise ValueError("diagnostic limits must be positive integers")
    if not isinstance(text, str):
        return {**base, "available": False, "reason": "not-text"}
    if len(text) > max_text_chars:
        return {**base, "available": False, "reason": "text-limit"}
    try:
        encoded = text.encode("utf-8")
    except UnicodeEncodeError:
        return {**base, "available": False, "reason": "invalid-utf8"}
    decoder = json.JSONDecoder(
        parse_float=_finite_number, parse_constant=_finite_number,
        parse_int=_bounded_integer)
    opening, closing = "<tool_call>", "</tool_call>"
    cursor = 0
    frames = []
    while (start := text.find(opening, cursor)) >= 0:
        if len(frames) >= max_frames:
            return {**base, "available": False, "reason": "frame-limit"}
        cursor = start + len(opening)
        value_start = cursor
        while value_start < len(text) and text[value_start] in " \t\r\n":
            value_start += 1
        frame = {"start_char": start, "canonical_call_sha256": None}
        try:
            value, end = decoder.raw_decode(text, value_start)
        except RecursionError:
            return {**base, "available": False, "reason": "nesting-limit"}
        except _ObservationLimit as error:
            return {**base, "available": False, "reason": str(error)}
        except _NonFiniteNumber:
            frame["status"] = "nonfinite-number"
            frames.append(frame)
            continue
        except ValueError:
            frame["status"] = "invalid-json"
            frames.append(frame)
            continue
        while end < len(text) and text[end] in " \t\r\n":
            end += 1
        if not text.startswith(closing, end):
            frame["status"] = "missing-close-after-json"
            frames.append(frame)
            continue
        cursor = end + len(closing)
        frame["end_char"] = cursor
        frame["span_sha256"] = hashlib.sha256(text[start:cursor].encode("utf-8")).hexdigest()
        if (isinstance(value, dict) and isinstance(value.get("name"), str)
                and value["name"] and isinstance(value.get("arguments", {}), dict)):
            try:
                canonical = json.dumps(
                    {"name": value["name"], "arguments": value.get("arguments", {})},
                    ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False,
                ).encode("ascii")
            except RecursionError:
                return {**base, "available": False, "reason": "nesting-limit"}
            frame["canonical_call_sha256"] = hashlib.sha256(canonical).hexdigest()
            frame["status"] = "framed-call-object"
        else:
            frame["status"] = "framed-noncall-json"
        frames.append(frame)
    digests = [frame["canonical_call_sha256"] for frame in frames
               if frame["canonical_call_sha256"] is not None]
    return {
        **base, "available": True,
        "raw_text_bytes": len(encoded),
        "raw_text_sha256": hashlib.sha256(encoded).hexdigest(),
        "lexical_open_markers": text.count(opening),
        "lexical_close_markers": text.count(closing),
        "observed_frame_starts": len(frames),
        "framed_call_objects": len(digests),
        "canonical_call_sha256": digests,
        "canonical_duplicate_count": len(digests) - len(set(digests)),
        "frames": frames,
    }
