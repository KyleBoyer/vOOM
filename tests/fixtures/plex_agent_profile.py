#!/usr/bin/env python3
"""Score real vOOM models on the captured Plex planning failure mode.

The private capture is read locally and never copied into the result.  The
profile records only model output, synthetic tool results, telemetry, and a
rubric that separates tool selection, filter planning, pagination, and final
answer precision/recall.

Two prompt profiles are useful:

``focused``
    The captured user request plus only the real Plex tool schema.  This
    isolates model comprehension from large-catalog retrieval and is cheap
    enough for broad model sweeps.

``captured``
    The complete 130+ tool request. By default sampling, streaming, storage,
    parallel calls and output budget are overridden; use
    ``--preserve-capture-shape`` to change only the model alias. Effective
    request metadata records the actual changes rather than just source size.

``captured-adapted``
    Preserve the 134-tool catalog and original messages, but replace only the
    Plex list function's schema with the selected planner/policy contract. This
    models a plugin schema upgrade without pretending it is the untouched
    capture.

Example (against an already-running server)::

    .venv/bin/python tests/fixtures/plex_agent_profile.py \
      logs/captured_requests/1784574315421_94161f5f.json \
      --model lossy-Qwen3.6-35B-A3B --profile captured \
      --result-json logs/plex_profile_qwen36.json
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path

import psutil


PLEX_TOOL = "plugin__plex__plex_list_library"
# The captured catalog also exposes a narrower media-listing endpoint. Larger
# Qwen models sometimes select it instead of the planner-adapted list endpoint.
# It is still a real paginatable Plex call and must receive its tool result;
# historically the profiler recorded it in ``call_names`` but silently stopped
# after turn one because the continuation loop matched only ``PLEX_TOOL``.
PLEX_MEDIA_TOOL = "plugin__plex__plex_list_library_media"
PLEX_PAGINATION_TOOLS = frozenset((PLEX_TOOL, PLEX_MEDIA_TOOL))
EXPECTED_PROMPT_TERMS = ("plex", "movies/tv", "/Kids/")

SYNTHETIC_PAGES = (
    {
        "filtersApplied": False,
        "notice": (
            "Raw mixed rows are returned for this evaluation. Verify every "
            "rating and either rootFolderPath or plexLibrarySectionName "
            "against the user's criteria."
        ),
        "movies": [
            {"title": "ALPHA_G", "contentRating": "G",
             "rootFolderPath": "/Media/Movies",
             "plexLibrarySectionName": "Movies"},
            {"title": "ECHO_R", "contentRating": "R",
             "rootFolderPath": "/Media/Movies",
             "plexLibrarySectionName": "Movies"},
            {"title": "FOXTROT_KIDS_PG", "contentRating": "PG",
             "rootFolderPath": "/Media/Kids/Movies",
             "plexLibrarySectionName": "Kids"},
        ],
        "series": [
            {"title": "CHARLIE_TVY", "contentRating": "TV-Y",
             "rootFolderPath": "/Media/TV",
             "plexLibrarySectionName": "TV Shows"},
            {"title": "GOLF_TV14", "contentRating": "TV-14",
             "rootFolderPath": "/Media/TV",
             "plexLibrarySectionName": "TV Shows"},
        ],
        "movieHasMore": True,
        "seriesHasMore": True,
    },
    {
        "filtersApplied": False,
        "notice": (
            "Final raw page. Verify every rating and either rootFolderPath or "
            "plexLibrarySectionName before producing the list."
        ),
        "movies": [
            {"title": "BRAVO_PG13", "contentRating": "PG-13",
             "rootFolderPath": "/Media/Movies",
             "plexLibrarySectionName": "Movies"},
            {"title": "INDIA_UNRATED", "contentRating": None,
             "rootFolderPath": "/Media/Movies",
             "plexLibrarySectionName": "Movies"},
        ],
        "series": [
            {"title": "DELTA_TVY7", "contentRating": "TV-Y7",
             "rootFolderPath": "/Media/TV",
             "plexLibrarySectionName": "TV Shows"},
            {"title": "HOTEL_KIDS_TVY", "contentRating": "TV-Y",
             "rootFolderPath": "/Media/Kids/TV",
             "plexLibrarySectionName": "Kids"},
            {"title": "JULIET_TVPG", "contentRating": "TV-PG",
             "rootFolderPath": "/Media/TV",
             "plexLibrarySectionName": "TV Shows"},
        ],
        "movieHasMore": False,
        "seriesHasMore": False,
    },
)

ELIGIBLE_TITLES = (
    "ALPHA_G", "BRAVO_PG13", "CHARLIE_TVY", "DELTA_TVY7")
INELIGIBLE_TITLES = (
    "ECHO_R", "FOXTROT_KIDS_PG", "GOLF_TV14", "HOTEL_KIDS_TVY",
    "INDIA_UNRATED", "JULIET_TVPG")


def load_kai_tool_result_export(path: Path) -> tuple[list[dict], dict]:
    """Load exact paginated Plex calls/results from a private Kai export.

    The export stays local and unmodified.  The returned identity contains
    only hashes/counts/contract witnesses; callers append the exact call and
    result objects to the captured request in memory.  This closes the old
    profiler's most important realism gap: its ten invented rows did not have
    the same payload shape or answer cardinality as live Plex traffic.
    """
    raw = path.read_bytes()
    value = json.loads(raw)
    messages = value.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Kai export has no messages array")
    calls = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if (not isinstance(part, dict)
                    or part.get("type") != "tool-call"
                    or part.get("toolName") != PLEX_MEDIA_TOOL):
                continue
            arguments = part.get("args")
            result = part.get("result")
            if not isinstance(arguments, dict) or not isinstance(result, dict):
                raise ValueError("Kai Plex call is missing object args/result")
            media = result.get("media")
            if not isinstance(media, list) or any(
                    not isinstance(row, dict) for row in media):
                raise ValueError("Kai Plex result media must be object rows")
            try:
                offset = int(arguments.get("offset", result.get("offset", 0)))
                limit = int(arguments.get("limit", result.get("limit", 0)))
                returned = int(result.get("returned"))
                total = int(result.get("total"))
            except (TypeError, ValueError) as error:
                raise ValueError("Kai Plex pagination fields must be integers") from error
            if (offset != int(result.get("offset", offset))
                    or limit != int(result.get("limit", limit))
                    or returned != len(media)
                    or limit <= 0 or returned < 0 or total < returned
                    or not isinstance(result.get("hasMore"), bool)):
                raise ValueError("Kai Plex pagination contract is inconsistent")
            calls.append({
                "name": PLEX_MEDIA_TOOL,
                "call_id": str(part.get("toolCallId") or f"call_export_{offset}"),
                "arguments": copy.deepcopy(arguments),
                "arguments_raw": json.dumps(
                    arguments, ensure_ascii=False, separators=(",", ":")),
                "result": copy.deepcopy(result),
                "offset": offset,
                "limit": limit,
                "returned": returned,
                "total": total,
            })
    if not calls:
        raise ValueError("Kai export contains no Plex media calls")
    calls.sort(key=lambda call: call["offset"])
    totals = {call["total"] for call in calls}
    if (len(totals) != 1
            or calls[0]["offset"] != 0
            or any(right["offset"] <= left["offset"]
                   for left, right in zip(calls, calls[1:]))
            or any(call["result"]["hasMore"] is not True for call in calls[:-1])
            or calls[-1]["result"]["hasMore"] is not False
            or sum(call["returned"] for call in calls) != calls[-1]["total"]):
        raise ValueError("Kai export is not one complete monotonic page stream")
    rows = [row for call in calls for row in call["result"]["media"]]
    identity = {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "plex_calls": len(calls),
        "offsets": [call["offset"] for call in calls],
        "rows": len(rows),
        "total": calls[-1]["total"],
        "final_has_more": calls[-1]["result"]["hasMore"],
        "root_path_evidence_available": any(
            isinstance(row.get("rootFolderPath"), str)
            and bool(row.get("rootFolderPath")) for row in rows),
    }
    return calls, identity


def _actual_export_oracle(export_calls: list[dict]) -> dict:
    """Derive the catalog answer supported by a broad media-list export.

    The live endpoint returns section names but not root folder paths.  A
    section whose name contains the standalone word Kid/Kids is therefore a
    useful local proxy, not strict proof of the user's root-path predicate.
    Keep that evidence limitation explicit so an inferred title match cannot
    be mislabeled as a fully released-plugin-correct result.
    """
    movie_ladder = ("G", "PG", "PG-13", "R", "NC-17")
    show_ladder = (
        "TV-Y", "TV-Y7", "TV-Y7-FV", "TV-G", "TV-PG", "TV-14", "TV-MA")
    rows = [row for call in export_calls for row in call["result"]["media"]]

    def root_path(row: dict) -> str:
        value = row.get("rootFolderPath")
        return value.strip() if isinstance(value, str) else ""

    def eligible(row: dict) -> bool:
        root = root_path(row)
        excluded_by_location = (
            "/kids/" in root.casefold()
            if root else
            bool(re.search(
                r"\bkids?\b", str(row.get("sectionName") or ""), re.I)))
        if excluded_by_location:
            return False
        media_type = row.get("type")
        rating = row.get("contentRating")
        if media_type == "movie" and rating in movie_ladder:
            return movie_ladder.index(rating) <= movie_ladder.index("PG-13")
        if media_type == "show" and rating in show_ladder:
            return show_ladder.index(rating) <= show_ladder.index("TV-Y7")
        return False

    expected = tuple(str(row.get("title")) for row in rows
                     if row.get("title") and eligible(row))
    all_titles = tuple(str(row.get("title")) for row in rows if row.get("title"))
    root_evidence_rows = sum(bool(root_path(row)) for row in rows)
    root_evidence = bool(root_evidence_rows)
    # A rooted row cannot certify the location predicate for different rows
    # whose inclusion still depends on the section-name proxy. Keep strict
    # evidence conservative: every catalog row must carry a usable root path.
    root_evidence_complete = bool(rows) and root_evidence_rows == len(rows)
    return {
        "expected_titles": expected,
        "all_titles": all_titles,
        "expected_count": len(expected),
        "catalog_count": len(rows),
        "expected_titles_sha256": hashlib.sha256(
            "\n".join(sorted(expected)).encode("utf-8")).hexdigest(),
        "root_path_evidence_available": root_evidence,
        "root_path_evidence_complete": root_evidence_complete,
        "root_path_evidence_rows": root_evidence_rows,
        "root_path_evidence_missing_rows": len(rows) - root_evidence_rows,
        "exclusion_basis": (
            "rootFolderPath" if root_evidence_complete else
            "mixed rootFolderPath and sectionName Kid/Kids proxy; "
            "strict root predicate incomplete" if root_evidence else
            "sectionName Kid/Kids proxy; strict root predicate unavailable"),
    }


# Models render catalog titles with typographic characters that carry no
# semantic difference: U+2011 non-breaking hyphens ("Spider-Man"), U+2019
# apostrophes ("God's"), U+202F narrow no-break spaces before a year, and
# hyphen/space interchange ("The A-Team" written "A Team").  Matching the raw
# title with re.escape misses every one of those, so a correct answer scored
# as a miss.  Measured on the 2026-07-31 real-export run: 7 titles matched
# strictly versus 42 after folding.  Folding is applied identically to the
# answer text and to the catalog titles, so this changes precision, not what
# counts as correct.
# Horizontal separators only: folding newlines away would merge the whole
# answer into one line and let a later "[Excluded: ...]" marker suppress
# every earlier title.
_TITLE_SEPARATORS_RE = re.compile(r"(?:[^\S\r\n]|[\u2010-\u2015\-_])+")
_TITLE_APOSTROPHES_RE = re.compile(r"['‘’ʼ´`]")
_LEADING_ARTICLE_RE = re.compile(r"(?i)^(?:the|a|an)\s+")
# A title named with an explicit rejection on the same line is the model
# REPORTING an exclusion, not claiming the item.  Counting those mentions as
# leaks penalizes exactly the behavior the task asks for.
_EXCLUSION_MARKER_RE = re.compile(
    r"(?i)\b(?:exclud|reject|omit|filtered\s+out|not\s+included|"
    r"does\s+not\s+(?:meet|qualify)|too\s+high)")


def _fold_title_text(value: str) -> str:
    """Canonicalize typographic variation without changing word content."""
    folded = unicodedata.normalize("NFKC", value or "")
    folded = _TITLE_APOSTROPHES_RE.sub("", folded)
    return _TITLE_SEPARATORS_RE.sub(" ", folded)


def _mention_is_excluded(text: str, end: int) -> bool:
    """True when the remainder of the mention's line marks it rejected."""
    line_end = text.find("\n", end)
    tail = text[end:] if line_end < 0 else text[end:line_end]
    return bool(_EXCLUSION_MARKER_RE.search(tail))


def _mentioned_catalog_titles(text: str, titles: tuple[str, ...]) -> set[str]:
    """Find non-overlapping known titles the answer ASSERTS, longest first.

    A title the answer names only to reject is not asserted, so it is neither
    a leak when ineligible nor a hit when eligible.
    """
    folded_text = _fold_title_text(text)
    occupied: list[tuple[int, int]] = []
    found: set[str] = set()
    candidates = []
    for title in set(titles):
        folded = _fold_title_text(title).strip()
        if not folded:
            continue
        # The full form is always preferred; the article-less form is only a
        # fallback, and longest-first ordering plus the occupied-span check
        # still prevent a shorter catalog title from stealing a longer match.
        variants = [folded]
        stripped = _LEADING_ARTICLE_RE.sub("", folded).strip()
        if stripped and stripped != folded:
            variants.append(stripped)
        candidates.append((title, variants))
    candidates.sort(key=lambda item: (-len(item[1][0]), item[0]))
    for title, variants in candidates:
        for variant in variants:
            pattern = re.compile(
                r"(?<![\w])" + re.escape(variant) + r"(?![\w])", re.I)
            for match in pattern.finditer(folded_text):
                span = match.span()
                if any(span[0] < end and start < span[1]
                       for start, end in occupied):
                    continue
                occupied.append(span)
                if not _mention_is_excluded(folded_text, span[1]):
                    found.add(title)
            if title in found:
                break
    return found


def score_actual_export(final_text: str, export_calls: list[dict]) -> dict:
    """Score final synthesis against every row in an exact Kai result stream."""
    oracle = _actual_export_oracle(export_calls)
    expected = set(oracle["expected_titles"])
    mentioned = _mentioned_catalog_titles(
        _harmony_final_channel(final_text), oracle["all_titles"])
    missing = expected - mentioned
    unexpected = mentioned - expected
    inferred_match = not missing and not unexpected
    return {
        "inferred_catalog_match": inferred_match,
        "strict_evidence_passed": bool(
            inferred_match and oracle["root_path_evidence_complete"]),
        "expected_count": len(expected),
        "mentioned_expected_count": len(expected & mentioned),
        "mentioned_ineligible_count": len(unexpected),
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
        "missing_examples": sorted(missing)[:10],
        "unexpected_examples": sorted(unexpected)[:10],
        "catalog_count": oracle["catalog_count"],
        "expected_titles_sha256": oracle["expected_titles_sha256"],
        "root_path_evidence_available": oracle["root_path_evidence_available"],
        "root_path_evidence_complete": oracle["root_path_evidence_complete"],
        "root_path_evidence_rows": oracle["root_path_evidence_rows"],
        "root_path_evidence_missing_rows": oracle["root_path_evidence_missing_rows"],
        "exclusion_basis": oracle["exclusion_basis"],
    }


def _pressure() -> dict[str, int]:
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {
        "available_bytes": int(memory.available),
        "swap_used_bytes": int(swap.used),
        "swap_out_bytes": int(swap.sout),
    }


def compact_plex_planner_tool(tool: dict, *, bind_policy: bool = False) -> dict:
    """Task-specialized view of Plex without changing the executable API.

    The full plugin exposes 25 mostly unrelated filters. A small specialist
    should plan the eight fields relevant to this request, while the host keeps
    the original tool object for execution. Core semantic fields are required
    in this planner view; paging fields remain model-selected so pagination is
    still measured rather than filled in by the evaluator.
    """
    selected = (
        "mediaType", "ratingOperator", "movieRatingValue", "showRatingValue",
        "excludeRootFolderPath", "excludePlexLibrarySectionName", "limit",
        "offset",
    )
    copied = copy.deepcopy(tool)
    schema = copied.get("parameters") or {}
    properties = schema.get("properties") or {}
    schema["properties"] = {
        name: copy.deepcopy(properties[name]) for name in selected
        if name in properties
    }
    # Required planner fields must not retain the provider adapter's nullable
    # wrapper. More importantly, enumerate the two distinct rating vocabularies
    # instead of asking a small model to invent a code. This is the contract the
    # Plex plugin itself should expose: models can still choose a threshold, but
    # invalid hybrids such as "TV-7" are unrepresentable under constrained
    # decoding.
    rating_enums = {
        "movieRatingValue": ["G", "PG", "PG-13", "R", "NC-17"],
        "showRatingValue": [
            "TV-Y", "TV-Y7", "TV-Y7-FV", "TV-G", "TV-PG", "TV-14",
            "TV-MA",
        ],
    }
    for name in (*rating_enums, "excludeRootFolderPath",
                 "excludePlexLibrarySectionName", "limit", "offset"):
        value = schema["properties"].get(name)
        variants = value.get("anyOf") if isinstance(value, dict) else None
        nonnull = next((copy.deepcopy(candidate) for candidate in variants or []
                        if candidate.get("type") != "null"), None)
        if nonnull is not None:
            schema["properties"][name] = nonnull
    for name, values in rating_enums.items():
        schema["properties"][name]["enum"] = values
    if bind_policy:
        # A higher-level Plex adapter can extract explicit thresholds from the
        # user request and bind them into the per-turn grammar. This removes a
        # classification decision the caller already made; it is not a silent
        # conversion between the incompatible MPAA and TV ladders.
        schema["properties"]["mediaType"] = {"const": "all"}
        schema["properties"]["ratingOperator"] = {"const": "lte"}
        schema["properties"]["movieRatingValue"] = {"const": "PG-13"}
        schema["properties"]["showRatingValue"] = {"const": "TV-Y7"}
    schema["required"] = [
        "mediaType", "ratingOperator", "movieRatingValue", "showRatingValue",
        "limit", "offset",
    ]
    schema["anyOf"] = [{"required": ["excludeRootFolderPath"]}, {
        "required": ["excludePlexLibrarySectionName"]}]
    schema.pop("x-optional", None)
    copied["parameters"] = schema
    copied["description"] = (
        "List both Plex movies and TV shows with independent rating ladders. "
        "For this request set mediaType=all, ratingOperator=lte, the movie "
        "threshold in movieRatingValue, and the TV threshold in "
        "showRatingValue. Excluding root /Kids/ OR Plex section Kids are "
        "equivalent. Start offset at 0 and increase it while HasMore is true."
    )
    return copied


def _canonical_rating(value, media_type: str) -> tuple[str | None, str | None]:
    raw = "" if value is None else str(value).strip().upper()
    aliases = {
        "movie": {"PG13": "PG-13"},
        "show": {"TV-7": "TV-Y7", "TV7": "TV-Y7", "TVY-7": "TV-Y7"},
    }[media_type]
    canonical = aliases.get(raw, raw)
    ladder = (
        ("G", "PG", "PG-13", "R", "NC-17") if media_type == "movie"
        else ("TV-Y", "TV-Y7", "TV-Y7-FV", "TV-G", "TV-PG", "TV-14", "TV-MA")
    )
    if canonical not in ladder:
        return None, None
    return canonical, (f"{raw}->{canonical}" if canonical != raw else None)


def evaluate_plex_policy_adapter(calls: list[dict]) -> dict:
    """Prototype plugin-side correctness boundary for a specialist proposal.

    One small model proposes the filter. The adapter canonicalizes only an
    explicit, auditable rating alias set, owns pagination, and deterministically
    applies rating/root-or-section policy to raw rows. It never guesses an
    unknown rating or invents a missing Kids exclusion.
    """
    plex = [call for call in calls if call.get("name") in PLEX_PAGINATION_TOOLS
            and isinstance(call.get("arguments"), dict)]
    if not plex:
        return {"passed": False, "reason": "no_valid_plex_proposal"}
    proposal = copy.deepcopy(plex[0]["arguments"])
    movie, movie_repair = _canonical_rating(
        proposal.get("movieRatingValue"), "movie")
    show, show_repair = _canonical_rating(
        proposal.get("showRatingValue"), "show")
    root = str(proposal.get("excludeRootFolderPath") or "").lower()
    section = str(proposal.get("excludePlexLibrarySectionName") or "").lower()
    if (proposal.get("mediaType") != "all"
            or proposal.get("ratingOperator") != "lte"
            or movie is None or show is None
            or ("/kids/" not in root and "kids" not in section)):
        return {"passed": False, "reason": "proposal_failed_policy_validation",
                "proposal": proposal}
    proposal["movieRatingValue"] = movie
    proposal["showRatingValue"] = show
    try:
        limit = max(1, min(200, int(proposal.get("limit") or 50)))
    except (TypeError, ValueError):
        limit = 50

    movie_ladder = ("G", "PG", "PG-13", "R", "NC-17")
    show_ladder = (
        "TV-Y", "TV-Y7", "TV-Y7-FV", "TV-G", "TV-PG", "TV-14", "TV-MA")
    selected = []
    pages_fetched = 0
    for page in SYNTHETIC_PAGES:
        pages_fetched += 1
        for kind, rows, threshold, ladder in (
            ("movie", page["movies"], movie, movie_ladder),
            ("show", page["series"], show, show_ladder),
        ):
            ceiling = ladder.index(threshold)
            for row in rows:
                row_root = str(row.get("rootFolderPath") or "").lower()
                row_section = str(
                    row.get("plexLibrarySectionName") or "").lower()
                if "/kids/" in row_root or "kids" in row_section:
                    continue
                rating = row.get("contentRating")
                if rating in ladder and ladder.index(rating) <= ceiling:
                    selected.append(row["title"])
        if not page.get("movieHasMore") and not page.get("seriesHasMore"):
            break

    normalized_calls = []
    for offset in range(0, pages_fetched * limit, limit):
        arguments = copy.deepcopy(proposal)
        arguments.update(limit=limit, offset=offset)
        normalized_calls.append(_call_dict(arguments))
    final_text = ", ".join(selected)
    rubric = score_profile(normalized_calls, final_text)
    repairs = [value for value in (movie_repair, show_repair) if value]
    return {
        "passed": rubric["passed"],
        "profile": "specialist-plan+deterministic-plex-policy-v1",
        "normalized_arguments": proposal,
        "rating_repairs": repairs,
        "pages_fetched": pages_fetched,
        "final_titles": selected,
        "rubric": rubric,
    }


def _call_dict(arguments: dict) -> dict:
    return {"name": PLEX_TOOL, "arguments": arguments}


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        str(part.get("text", "")) for part in content
        if isinstance(part, dict) and part.get("text"))


def request_shape(request: dict) -> dict:
    """Content-free identity of the effective request, not its source capture."""
    encoded = json.dumps(
        request, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")
    items = request.get("input") or []
    return {
        "canonical_sha256": hashlib.sha256(encoded).hexdigest(),
        "canonical_bytes": len(encoded),
        "tool_count": len(request.get("tools") or []),
        "input_items": len(items),
        "input_text_chars": sum(
            len(_content_text(item.get("content")))
            for item in items if isinstance(item, dict)),
        "developer_messages": sum(
            isinstance(item, dict) and item.get("role") == "developer"
            for item in items),
        "stream": request.get("stream", False),
        "temperature": request.get("temperature"),
        "output_cap": request.get(
            "max_output_tokens", request.get("max_tokens")),
    }


def load_profile_request(capture: Path, model: str, profile: str,
                         max_output_tokens: int,
                         reasoning_effort: str | None = None,
                         temperature: float = 0.0,
                         tool_choice: str = "capture",
                         tool_schema_profile: str = "full", *,
                         preserve_capture_shape: bool = False) -> tuple[dict, dict]:
    """Return a runnable request and non-sensitive capture identity."""
    if preserve_capture_shape and profile != "captured":
        raise ValueError(
            "preserve_capture_shape requires the unadapted captured profile")
    if (tool_schema_profile != "full"
            and profile not in ("focused", "captured-adapted")):
        raise ValueError(
            "the planner schema is a focused-profile adapter; the captured "
            "profile must preserve its original 130+ tool request")
    raw = capture.read_bytes()
    value = json.loads(raw)
    tools = value.get("tools") or []
    plex = [tool for tool in tools if tool.get("name") == PLEX_TOOL]
    user_items = [item for item in value.get("input") or []
                  if item.get("role") == "user"]
    user_text = _content_text(user_items[-1].get("content")) if user_items else ""
    if len(tools) < 130:
        raise ValueError(f"capture has only {len(tools)} tools; expected 130+")
    if len(plex) != 1:
        raise ValueError(f"capture has {len(plex)} Plex list tools; expected one")
    if not all(term.lower() in user_text.lower() for term in EXPECTED_PROMPT_TERMS):
        raise ValueError("capture does not contain the expected Plex user request")

    request = copy.deepcopy(value)
    if profile == "focused":
        request["input"] = [{
            "role": "system",
            "content": [{
                "type": "input_text",
                "text": (
                    "Use the supplied tool to satisfy the request. Preserve the "
                    "distinction between movie and TV rating systems, paginate "
                    "until both media types report no more results, then verify "
                    "the returned rows and list only matching titles. Excluding "
                    "root paths containing /Kids/ or excluding the authoritative "
                    "Plex library section named Kids are equivalent for this "
                    "request; either plan is valid."
                ),
            }],
        }, {
            "role": "user",
            "content": [{"type": "input_text", "text": user_text}],
        }]
        request["tools"] = [
            (compact_plex_planner_tool(
                plex[0], bind_policy=tool_schema_profile == "policy")
             if tool_schema_profile in ("planner", "policy")
             else copy.deepcopy(plex[0]))]
    elif profile == "captured-adapted":
        if tool_schema_profile == "full":
            raise ValueError(
                "captured-adapted requires planner or policy tool schema")
        request["tools"] = [
            (compact_plex_planner_tool(
                tool, bind_policy=tool_schema_profile == "policy")
             if tool.get("name") == PLEX_TOOL else copy.deepcopy(tool))
            for tool in tools
        ]
    elif profile != "captured":
        raise ValueError(f"unknown profile {profile!r}")

    request["model"] = model
    if not preserve_capture_shape:
        request["stream"] = False
        request["store"] = False
        request["temperature"] = float(temperature)
        # Each synthetic page must inform the next offset. Speculative parallel
        # calls cannot prove pagination comprehension, so keep the legacy
        # profiler sequential. Export-terminal replay may instead preserve the
        # original streaming/temperature/absent-cap request verbatim.
        request["parallel_tool_calls"] = False
        request["max_output_tokens"] = max_output_tokens
        request.pop("max_tokens", None)
        if tool_choice == "specific":
            request["tool_choice"] = {"type": "function", "name": PLEX_TOOL}
        elif tool_choice != "capture":
            request["tool_choice"] = tool_choice
    elif tool_choice != "capture":
        raise ValueError(
            "preserve_capture_shape cannot override the captured tool choice")
    if reasoning_effort is not None:
        request["reasoning"] = {"effort": reasoning_effort}
    missing = object()
    changed_fields = sorted(
        key for key in value.keys() | request.keys()
        if value.get(key, missing) != request.get(key, missing))
    if preserve_capture_shape and set(changed_fields) - {"model"}:
        raise ValueError(
            "preserve_capture_shape may change only the model alias")
    identity = {
        "schema": "plex-capture-identity-v2",
        "scope": "base_request_before_any_tool_result_history",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "tools": len(tools),
        "profile": profile,
        "reasoning_effort": reasoning_effort or "unspecified",
        "temperature": float(temperature),
        "tool_choice": tool_choice,
        "tool_schema_profile": tool_schema_profile,
        "preserve_capture_shape": bool(preserve_capture_shape),
        "effective_stream": request.get("stream", False),
        "effective_temperature": request.get("temperature"),
        "effective_output_cap": request.get(
            "max_output_tokens", request.get("max_tokens")),
        "effective_request": request_shape(request),
        "changed_top_level_fields": changed_fields,
        "unchanged_except_model": not bool(set(changed_fields) - {"model"}),
    }
    return request, identity


def _post(url: str, request: dict, timeout: float) -> tuple[dict, float]:
    payload = json.dumps(
        request, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    http_request = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"},
        method="POST")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(http_request, timeout=timeout) as response:
            body = response.read()
            content_type = response.headers.get("Content-Type", "")
            if "text/event-stream" not in content_type:
                return json.loads(body), time.perf_counter() - started
            terminal = None
            for line in body.decode("utf-8", errors="replace").splitlines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if not payload or payload == "[DONE]":
                    continue
                try:
                    event = json.loads(payload)
                except ValueError:
                    continue
                if event.get("type") in (
                        "response.completed", "response.incomplete",
                        "response.failed") and isinstance(
                            event.get("response"), dict):
                    terminal = event["response"]
            if terminal is None:
                raise ValueError("Responses SSE stream had no terminal response")
            return terminal, time.perf_counter() - started
    except urllib.error.HTTPError as error:
        body = error.read()
        try:
            detail = json.loads(body)
        except ValueError:
            detail = {"error": body.decode("utf-8", errors="replace")[:2000]}
        detail.setdefault("http_status", error.code)
        return detail, time.perf_counter() - started


def response_calls(response: dict) -> list[dict]:
    calls = []
    for item in response.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "function_call":
            continue
        raw_args = item.get("arguments", "{}")
        try:
            arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except (TypeError, ValueError):
            arguments = None
        calls.append({
            "name": item.get("name"),
            "call_id": item.get("call_id") or item.get("id"),
            "arguments": arguments,
            "arguments_raw": raw_args,
        })
    return calls


def response_text(response: dict) -> str:
    direct = response.get("output_text")
    if isinstance(direct, str) and direct:
        return direct
    pieces = []
    for item in response.get("output") or []:
        if not isinstance(item, dict):
            continue
        for part in item.get("content") or []:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                pieces.append(part["text"])
    return "\n".join(pieces)


def _norm(value) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _offset(arguments) -> int | None:
    if not isinstance(arguments, dict):
        return None
    # Both accepted Plex pagination endpoints declare offset=0 as their
    # executable schema default. An omitted optional offset therefore means the
    # first page, not an unknown/null page. Score the wire contract the same way
    # the tool adapter executes it.
    if "offset" not in arguments:
        return 0
    try:
        return int(arguments.get("offset"))
    except (TypeError, ValueError):
        return None


# gpt-oss emits harmony channels: an `analysis` chain-of-thought channel and
# the user-visible `final` channel. A correct answer routinely REJECTS the
# ineligible titles by name inside analysis, so scoring the concatenated text
# counts those rejections as if the model had listed them -- inverting the
# exclusion rubric and rewarding runs that merely truncated before naming
# them. Slice to the final channel first. Kept here as well as server-side so
# archived raw artifacts re-score correctly.
_HARMONY_FINAL_RE = re.compile(
    r"(?:<\|channel\|>final(?:<\|message\|>)?|assistantfinal)")
_HARMONY_END_RE = re.compile(r"<\|(?:return|end|start)\|>")


def _harmony_final_channel(text: str) -> str:
    """Return harmony's final channel alone, or the text unchanged."""
    markers = list(_HARMONY_FINAL_RE.finditer(text or ""))
    if not markers:
        return text or ""
    tail = (text or "")[markers[-1].end():]
    end = _HARMONY_END_RE.search(tail)
    if end:
        tail = tail[:end.start()]
    return tail.strip() or (text or "")


def _explicit_kids_root_verification(text: str) -> bool:
    """Witness that both adversarial Kids rows were inspected and rejected."""
    lowered = text.lower()
    for title in ("FOXTROT_KIDS_PG", "HOTEL_KIDS_TVY"):
        found = False
        for match in re.finditer(re.escape(title.lower()), lowered):
            window = lowered[match.start():match.end() + 320]
            if (("/kids/" in window
                 or "kids section" in window
                 or "section named kids" in window
                 or "kids library" in window)
                    and any(term in window for term in (
                        "exclude", "excluded", "reject", "contains"))):
                found = True
                break
        if not found:
            return False
    return True


def score_profile(calls: list[dict], final_text: str,
                  reasoning: str = "") -> dict:
    """Score the four independent behaviors on a stable 100-point rubric."""
    plex_calls = [
        call for call in calls
        if call.get("name") in PLEX_PAGINATION_TOOLS
    ]
    first = plex_calls[0].get("arguments") if plex_calls else None
    first = first if isinstance(first, dict) else {}
    offsets = [_offset(call.get("arguments")) for call in plex_calls]
    numeric_offsets = [value for value in offsets if value is not None]
    increasing_offsets = (
        len(numeric_offsets) >= 2
        and all(right > left for left, right in zip(
            numeric_offsets, numeric_offsets[1:])))
    # Score every character the user can see.  Harmony analysis is a separate
    # protocol channel and is removed, but an ordinary visible "Final list:"
    # heading is not a trust boundary: a bad prefix followed by a clean suffix
    # is still a bad answer.  Previous marker slicing let exactly that output
    # masquerade as 100/100.
    answer_text = _harmony_final_channel(final_text).strip()
    # A model may implement the Kids exclusion by post-filtering in its
    # reasoning instead of via a tool argument, and that reasoning is the only
    # witness of it. Harmony now delivers reasoning out-of-band (it is no
    # longer glued to the answer), so the witness must consider both -- while
    # the answer itself is still scored from the final channel alone.
    verification_text = "\n".join(part for part in (final_text, reasoning)
                                  if part)
    root_or_section_filter_or_verification = (
        "/kids/" in str(first.get("excludeRootFolderPath") or "").lower()
        or "kids" in str(
            first.get("excludePlexLibrarySectionName") or "").lower()
        or _explicit_kids_root_verification(verification_text))
    checks = {
        "selected_plex_tool": (bool(plex_calls), 10),
        "media_type_all": (_norm(first.get("mediaType")) == "all", 8),
        # Keep the historical key so archived result tooling remains readable.
        # Its semantics now match the user's actual contract: either the
        # filesystem root or the authoritative Plex section may implement the
        # Kids exclusion.
        "excluded_kids_root": (
            root_or_section_filter_or_verification, 12),
        "rating_operator_lte": (
            _norm(first.get("ratingOperator")) in {"lte", "lessthanorequal"}, 6),
        "movie_rating_pg13": (
            _norm(first.get("movieRatingValue")) == "pg13", 8),
        "show_rating_tvy7": (
            _norm(first.get("showRatingValue")) == "tvy7", 8),
        "initial_offset_zero": (bool(plex_calls) and offsets[0] == 0, 4),
        "bounded_page_limit": (
            isinstance(first.get("limit"), int) and 1 <= first["limit"] <= 1000, 4),
        "paginated_after_has_more": (len(plex_calls) >= 2, 5),
        "pagination_offset_increased": (increasing_offsets, 5),
    }
    eligible_found = {
        title: bool(re.search(re.escape(title), answer_text, flags=re.IGNORECASE))
        for title in ELIGIBLE_TITLES
    }
    ineligible_absent = {
        title: not bool(re.search(re.escape(title), answer_text, flags=re.IGNORECASE))
        for title in INELIGIBLE_TITLES
    }
    eligible_points = 15 * sum(eligible_found.values()) / len(eligible_found)
    exclusion_points = 15 * sum(ineligible_absent.values()) / len(ineligible_absent)
    plan_and_pagination = sum(
        points for passed, points in checks.values() if passed)
    total = plan_and_pagination + eligible_points + exclusion_points
    critical = (
        checks["selected_plex_tool"][0]
        and checks["excluded_kids_root"][0]
        and checks["movie_rating_pg13"][0]
        and checks["show_rating_tvy7"][0]
        and checks["paginated_after_has_more"][0]
        and all(eligible_found.values())
        and all(ineligible_absent.values())
    )
    return {
        "score": round(total, 2),
        "passed": bool(total >= 85 and critical),
        "checks": {
            name: {"passed": passed, "points": points}
            for name, (passed, points) in checks.items()
        },
        "eligible_titles_found": eligible_found,
        "ineligible_titles_absent": ineligible_absent,
        "eligible_points": round(eligible_points, 2),
        "exclusion_points": round(exclusion_points, 2),
        "plex_call_count": len(plex_calls),
        "offsets": offsets,
        "answer_slice_used": answer_text != (final_text or "").strip(),
        "whole_visible_output_scanned": True,
    }


def _append_call_and_result(request: dict, call: dict, page: dict) -> None:
    call_id = str(call.get("call_id") or f"call_profile_{len(request['input'])}")
    arguments = call.get("arguments_raw")
    if not isinstance(arguments, str):
        arguments = json.dumps(call.get("arguments") or {}, separators=(",", ":"))
    request["input"].append({
        "type": "function_call", "call_id": call_id,
        "name": str(call.get("name") or ""), "arguments": arguments,
    })
    request["input"].append({
        "type": "function_call_output", "call_id": call_id,
        "output": json.dumps(page, ensure_ascii=False, separators=(",", ":")),
    })


def _append_export_history(request: dict, export_calls: list[dict]) -> None:
    for call in export_calls:
        _append_call_and_result(request, call, call["result"])


def _response_turn(response: dict, wall: float, turn: int,
                   request: dict) -> dict:
    calls = response_calls(response)
    return {
        "turn": turn,
        "wall_seconds": round(wall, 4),
        "usage": response.get("usage"),
        "timing": response.get("vmodel_timing"),
        "cache_phases": response.get("vmodel_cache_phases"),
        "tool_selection": response.get("vmodel_tool_selection"),
        "constraint": response.get("vmodel_constraint"),
        "call_names": [call.get("name") for call in calls],
        "handled_call_count": 0,
        "visible_text": response_text(response),
        "error": response.get("error"),
        "response_status": response.get("status"),
        "incomplete_details": response.get("incomplete_details"),
        "effective_request": request_shape(request),
        "checkpoint": response.get("vmodel_checkpoint"),
        "backend": response.get("vmodel_backend"),
        "execution_profile": response.get("vmodel_execution_profile"),
        "runtime_profiles": response.get("vmodel_runtime_profiles"),
        "runtime_profile_digest": response.get("vmodel_runtime_profile_digest"),
        "runtime_effective_digest": response.get("vmodel_runtime_effective_digest"),
    }


def completion_gate(turns: list[dict]) -> dict:
    """A good rubric alone cannot certify truncated or unfinished work."""
    incomplete = [turn["turn"] for turn in turns
                  if turn.get("response_status") != "completed"]
    errors = [turn["turn"] for turn in turns if turn.get("error")]
    unhandled = [turn["turn"] for turn in turns
                 if len(turn.get("call_names") or [])
                 > turn.get("handled_call_count", 0)]
    terminal_text_present = bool(
        turns and str(turns[-1].get("visible_text") or "").strip())
    pending_calls = bool(turns and turns[-1].get("call_names"))
    return {
        "passed": bool(turns and not incomplete and not errors
                       and not unhandled and not pending_calls
                       and terminal_text_present),
        "non_completed_turns": incomplete,
        "error_turns": errors,
        "terminal_has_unhandled_calls": pending_calls,
        "unhandled_call_turns": unhandled,
        "terminal_text_present": terminal_text_present,
    }


def run_export_terminal_profile(request: dict, export_calls: list[dict],
                                url: str, timeout: float,
                                repeats: int = 1) -> dict:
    """Replay the exact completed Kai page stream into final synthesis."""
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    replay = copy.deepcopy(request)
    _append_export_history(replay, export_calls)
    if replay.get("tool_choice") == "required" or isinstance(
            replay.get("tool_choice"), dict):
        replay["tool_choice"] = "auto"
    pressure_before = _pressure()
    turns = []
    scores = []
    final_texts = []
    started = time.perf_counter()
    for repeat in range(repeats):
        response, wall = _post(url, replay, timeout)
        turns.append(_response_turn(response, wall, repeat + 1, replay))
        text = response_text(response)
        final_texts.append(text)
        scores.append(score_actual_export(text, export_calls))
        if response.get("error"):
            break
    pressure_after = _pressure()
    completion = completion_gate(turns)
    # Each export replay is an independent terminal-answer request, unlike
    # the successive tool turns of run_profile.
    completion["empty_terminal_turns"] = [
        turn["turn"] for turn in turns
        if not str(turn.get("visible_text") or "").strip()]
    if completion["empty_terminal_turns"]:
        completion["passed"] = False
    if any(turn.get("call_names") for turn in turns):
        completion["terminal_has_unhandled_calls"] = True
        completion["passed"] = False
    return {
        "gate": "plex-agent-kai-export-terminal-v2",
        "model": request.get("model"),
        "wire_request": request_shape(replay),
        "tool_results_source": "private_export_appended_to_base_request",
        # The export itself lacks rootFolderPath, so this conservative top-level
        # pass means exact catalog inference only; strict evidence is reported
        # separately and cannot silently become true from section-name proxying.
        "passed": bool(completion["passed"] and scores and all(
            score["inferred_catalog_match"] for score in scores)),
        "completion": completion,
        "strict_evidence_passed": bool(completion["passed"] and scores and all(
            score["strict_evidence_passed"] for score in scores)),
        "scores": scores,
        "turns": turns,
        "final_text": final_texts[-1] if final_texts else "",
        "repeat_output_sha256": [
            hashlib.sha256(text.encode("utf-8")).hexdigest()
            for text in final_texts
        ],
        "wall_seconds": round(time.perf_counter() - started, 4),
        "pressure_before": pressure_before,
        "pressure_after": pressure_after,
    }


def run_profile(request: dict, url: str, timeout: float,
                max_tool_rounds: int) -> dict:
    pressure_before = _pressure()
    turns = []
    all_calls: list[dict] = []
    final_text = ""
    final_reasoning = ""
    page_index = 0
    protocol_failures = []
    started = time.perf_counter()
    for turn_index in range(max_tool_rounds + 1):
        response, wall = _post(url, request, timeout)
        calls = response_calls(response)
        text = response_text(response)
        reasoning = response.get("vmodel_reasoning")
        turns.append(_response_turn(response, wall, turn_index + 1, request))
        if response.get("error"):
            break
        all_calls.extend(calls)
        plex_calls = [
            call for call in calls
            if call.get("name") in PLEX_PAGINATION_TOOLS
        ]
        if not calls:
            # Only the actual terminal turn owns the final answer; an empty
            # terminal response must never inherit earlier planning prose.
            final_text = text
            final_reasoning = reasoning if isinstance(reasoning, str) else ""
            break
        if len(calls) != 1 or len(plex_calls) != 1:
            protocol_failures.append({
                "turn": turn_index + 1,
                "reason": "expected_one_supported_plex_call_per_tool_turn",
            })
            break
        if turn_index == max_tool_rounds:
            protocol_failures.append({"turn": turn_index + 1,
                                      "reason": "tool_round_limit"})
            break
        call = plex_calls[0]
        page = SYNTHETIC_PAGES[min(page_index, len(SYNTHETIC_PAGES) - 1)]
        _append_call_and_result(request, call, page)
        turns[-1]["handled_call_count"] = 1
        # A forced choice governs the planning turn only. Keeping it on every
        # follow-up request would make a compliant model call forever even
        # after both HasMore flags are false, which tests the harness rather
        # than the model's pagination/termination behavior.
        if request.get("tool_choice") == "required" or isinstance(
                request.get("tool_choice"), dict):
            request["tool_choice"] = "auto"
        page_index += 1

    rubric = score_profile(all_calls, final_text, final_reasoning)
    pressure_after = _pressure()
    completion = completion_gate(turns)
    visible_text = "\n".join(turn["visible_text"] for turn in turns)
    visible_ineligible_absent = {
        title: not bool(re.search(re.escape(title), visible_text,
                                 flags=re.IGNORECASE))
        for title in INELIGIBLE_TITLES
    }
    visible_passed = all(visible_ineligible_absent.values())
    rubric["scope"] = "terminal_answer"
    return {
        "gate": "plex-agent-profile-v2",
        "model": request.get("model"),
        "passed": (rubric["passed"] and completion["passed"]
                   and visible_passed and not protocol_failures),
        "completion": completion,
        "protocol_failures": protocol_failures,
        "tool_results_source": "synthetic_two_page_fixture",
        "visible_output_gate": {
            "passed": visible_passed,
            "ineligible_titles_absent": visible_ineligible_absent,
            "all_response_text_scanned": True,
        },
        "rubric": rubric,
        "turns": turns,
        "calls": [{
            "name": call.get("name"),
            "arguments": call.get("arguments"),
        } for call in all_calls],
        "final_text": final_text,
        "final_reasoning": final_reasoning,
        "wall_seconds": round(time.perf_counter() - started, 4),
        "pressure_before": pressure_before,
        "pressure_after": pressure_after,
    }


def _wait_for_server(url: str, process: subprocess.Popen, timeout: float) -> None:
    models_url = url.rsplit("/v1/responses", 1)[0] + "/v1/models"
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        if process.poll() is not None:
            raise RuntimeError(f"server exited early with code {process.returncode}")
        try:
            with urllib.request.urlopen(models_url, timeout=2):
                return
        except Exception:
            time.sleep(0.5)
    raise TimeoutError("server did not become ready")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--profile",
                        choices=("focused", "captured", "captured-adapted"),
                        default="focused")
    parser.add_argument("--url", default="http://127.0.0.1:8077/v1/responses")
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "minimal", "low", "medium", "high", "xhigh"),
        help="Explicit Responses API reasoning effort; omitted preserves capture",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature; deterministic greedy profiling defaults to 0",
    )
    parser.add_argument("--max-tool-rounds", type=int, default=4)
    parser.add_argument(
        "--tool-choice", choices=("capture", "auto", "required", "specific"),
        default="capture",
        help="Override tool choice; specific forces the captured Plex function",
    )
    parser.add_argument(
        "--tool-schema-profile", choices=("full", "planner", "policy"),
        default="full",
        help="Use the full schema, compact planner, or request-bound policy view",
    )
    parser.add_argument(
        "--policy-adapter", action="store_true",
        help="also score the explicit specialist+deterministic Plex pipeline",
    )
    parser.add_argument(
        "--kai-tool-export", type=Path,
        help=(
            "append the exact completed Plex call/result history from a "
            "private Kai conversation export and benchmark final synthesis"),
    )
    parser.add_argument(
        "--repeat-requests", type=int, default=1,
        help="repeat an export-backed terminal request in one server process",
    )
    parser.add_argument(
        "--preserve-capture-shape", action="store_true",
        help=(
            "preserve captured streaming, temperature, tool choice, and absent "
            "output cap; only the requested model is changed"),
    )
    parser.add_argument("--start-server", action="store_true")
    parser.add_argument("--server-ready-timeout", type=float, default=120)
    parser.add_argument("--server-log", type=Path)
    parser.add_argument("--result-json", type=Path)
    args = parser.parse_args()
    if (args.timeout <= 0 or args.max_output_tokens <= 0
            or args.max_tool_rounds < 0 or args.temperature < 0
            or args.repeat_requests <= 0):
        parser.error(
            "timeouts/output must be positive; tool rounds and temperature "
            "must be nonnegative")
    if args.kai_tool_export is not None and args.policy_adapter:
        parser.error("--policy-adapter cannot be combined with --kai-tool-export")

    request, identity = load_profile_request(
        args.capture, args.model, args.profile, args.max_output_tokens,
        args.reasoning_effort, args.temperature, args.tool_choice,
        args.tool_schema_profile,
        preserve_capture_shape=args.preserve_capture_shape)
    export_calls = None
    export_identity = None
    if args.kai_tool_export is not None:
        export_calls, export_identity = load_kai_tool_result_export(
            args.kai_tool_export)
    process = None
    log_file = None
    try:
        if args.start_server:
            port_match = re.search(r":(\d+)(?:/|$)", args.url)
            port = int(port_match.group(1)) if port_match else 8077
            server_log = args.server_log or Path("logs") / (
                f"plex_profile_server_{_norm(args.model)}_{args.profile}.log")
            server_log.parent.mkdir(parents=True, exist_ok=True)
            log_file = open(server_log, "w")
            process = subprocess.Popen(
                [sys.executable, "-m", "runtime.server", "--port", str(port)],
                stdout=log_file, stderr=subprocess.STDOUT)
            _wait_for_server(args.url, process, args.server_ready_timeout)
        result = (
            run_export_terminal_profile(
                request, export_calls, args.url, args.timeout,
                repeats=args.repeat_requests)
            if export_calls is not None else
            run_profile(request, args.url, args.timeout, args.max_tool_rounds)
        )
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=20)
        if log_file is not None:
            log_file.close()
    result["capture"] = identity
    if export_identity is not None:
        result["kai_tool_export"] = export_identity
    if args.policy_adapter:
        result["policy_adapter"] = evaluate_plex_policy_adapter(result["calls"])
        if result.get("turns"):
            result["policy_adapter"]["model_planning_wall_seconds"] = (
                result["turns"][0]["wall_seconds"])
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(encoded, end="")
    if args.result_json is not None:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(encoded)
    effective_pass = (result["policy_adapter"]["passed"]
                      if args.policy_adapter else result["passed"])
    return 0 if effective_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
