"""Fail-closed deterministic rendering over completed typed tool results.

The model remains responsible for selecting a tool and proposing arguments.
This module owns the narrower, auditable step that models are bad at: applying
ordered predicates and exclusions to raw paginated rows.  A renderer is
returned only when the request intent, tool contract, page sequence, and every
row needed for the answer are understood.  Unknown shapes fall back to normal
model synthesis instead of being guessed.

The generic predicate types are deliberately independent of Plex.  The Plex
adapter below is one production-connected policy compiler for the captured
movies/TV workflow; future adapters can compile other verified tool contracts
to the same executor.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Mapping, Sequence


PLEX_LIBRARY_TOOL = "plugin__plex__plex_list_library"


@dataclass(frozen=True)
class OrderedPredicate:
    """Accept known values at or below ``ceiling`` in an explicit ladder."""

    field: str
    ceiling: str
    ladder: tuple[str, ...]
    aliases: Mapping[str, str]

    def canonicalize(self, value: object) -> str | None:
        if value is None:
            return None
        raw = str(value).strip().upper()
        return self.aliases.get(raw, raw)

    def accepts(self, row: Mapping[str, object]) -> tuple[bool, str]:
        canonical = self.canonicalize(row.get(self.field))
        if canonical is None:
            return False, "missing-ordered-value"
        if canonical not in self.ladder:
            raise ValueError(
                f"unknown {self.field} value {row.get(self.field)!r}")
        if self.ceiling not in self.ladder:
            raise ValueError(f"unknown ordered ceiling {self.ceiling!r}")
        return (
            self.ladder.index(canonical) <= self.ladder.index(self.ceiling),
            canonical,
        )


@dataclass(frozen=True)
class AnySubstringExclusion:
    """Reject a row when any typed field contains a normalized component."""

    fields: tuple[str, ...]
    needle: str

    def excludes(self, row: Mapping[str, object]) -> bool:
        needle = self.needle.casefold()
        for field in self.fields:
            value = str(row.get(field) or "").casefold()
            if field.lower().endswith("path"):
                components = tuple(
                    component for component in re.split(r"[/\\]+", value)
                    if component)
                if needle in components:
                    return True
            elif re.search(rf"(?<![\w]){re.escape(needle)}(?![\w])", value):
                return True
        return False


@dataclass(frozen=True)
class CollectionPolicy:
    """One named row collection and the predicates that govern it."""

    key: str
    label: str
    ordered: OrderedPredicate
    exclusions: tuple[AnySubstringExclusion, ...]


@dataclass(frozen=True)
class AcceptedRow:
    collection: str
    title: str
    ordered_value: str


@dataclass(frozen=True)
class PolicyExecution:
    accepted: tuple[AcceptedRow, ...]
    rejected_rows: int
    input_rows: int


@dataclass(frozen=True)
class DeterministicRender:
    text: str
    profile: str
    pages: int
    input_rows: int
    accepted_rows: int
    rejected_rows: int
    output_titles: tuple[str, ...]


@dataclass(frozen=True)
class PolicyRenderAttempt:
    render: DeterministicRender | None
    reason: str


def execute_typed_predicates(
    collections: Mapping[str, Sequence[Mapping[str, object]]],
    policies: Sequence[CollectionPolicy],
) -> PolicyExecution:
    """Apply typed predicates without guessing unknown rows or values.

    Missing ratings are unambiguously outside an age-rating allowlist and are
    rejected.  A non-empty rating that is absent from the declared ladder is
    ambiguous, so it raises and makes the enclosing renderer fail closed.
    Titles and row objects must be typed because silently stringifying a
    malformed payload could create an apparently valid but wrong answer.
    """

    accepted: list[AcceptedRow] = []
    rejected = 0
    seen: set[tuple[str, str]] = set()
    input_rows = 0
    for policy in policies:
        rows = collections.get(policy.key)
        if rows is None:
            raise ValueError(f"missing collection {policy.key!r}")
        for row in rows:
            input_rows += 1
            if not isinstance(row, Mapping):
                raise ValueError(f"non-object row in {policy.key!r}")
            title = row.get("title")
            if not isinstance(title, str) or not title.strip():
                raise ValueError(f"row in {policy.key!r} has no string title")
            if len(title) > 512 or any(ord(character) < 32 for character in title):
                raise ValueError(
                    f"row in {policy.key!r} has an unsafe title")
            identity = (policy.key, title.casefold())
            if identity in seen:
                raise ValueError(
                    f"duplicate paginated row {title!r} in {policy.key!r}")
            seen.add(identity)
            if any(exclusion.excludes(row) for exclusion in policy.exclusions):
                rejected += 1
                continue
            allowed, ordered_value = policy.ordered.accepts(row)
            if not allowed:
                rejected += 1
                continue
            accepted.append(AcceptedRow(
                collection=policy.label,
                title=title.strip(),
                ordered_value=ordered_value,
            ))
    return PolicyExecution(tuple(accepted), rejected, input_rows)


_MOVIE_LADDER = ("G", "PG", "PG-13", "R", "NC-17")
_SHOW_LADDER = (
    "TV-Y", "TV-Y7", "TV-Y7-FV", "TV-G", "TV-PG", "TV-14", "TV-MA")
_MOVIE_ALIASES = {"PG13": "PG-13"}
_SHOW_ALIASES = {
    "TV-7": "TV-Y7", "TV7": "TV-Y7", "TVY-7": "TV-Y7",
}
_UNSUPPORTED_FORMAT_RE = re.compile(
    r"\b(?:csv|json|xml|yaml|table|spreadsheet|count only|titles only|"
    r"group(?:ed)? by|sort(?:ed)? by)\b",
    re.IGNORECASE,
)


def _message_text(message: Mapping[str, object]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(part.get("text", part.get("output", "")))
            for part in content if isinstance(part, Mapping))
    return ""


def _supports_captured_plex_intent(messages: Sequence[Mapping[str, object]]) -> bool:
    user_text = next((
        _message_text(message)
        for message in reversed(messages)
        if message.get("role") == "user"
    ), "")
    normalized = re.sub(r"[^a-z0-9]+", "", user_text.casefold())
    required = (
        "plex", "list", "movie", "tv", "pg13", "kids", "paginate")
    if not all(term in normalized for term in required):
        return False
    if not any(alias in normalized for alias in ("tv7", "tvy7")):
        return False
    return not _UNSUPPORTED_FORMAT_RE.search(user_text)


def _parse_object(value: object) -> dict | None:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _canonical_threshold(value: object, *, media: str) -> str | None:
    raw = str(value or "").strip().upper()
    aliases = _MOVIE_ALIASES if media == "movie" else _SHOW_ALIASES
    canonical = aliases.get(raw, raw)
    ladder = _MOVIE_LADDER if media == "movie" else _SHOW_LADDER
    return canonical if canonical in ladder else None


def _valid_kids_exclusion(arguments: Mapping[str, object]) -> bool:
    root = str(arguments.get("excludeRootFolderPath") or "")
    root_components = tuple(
        component.casefold() for component in re.split(r"[/\\]+", root)
        if component)
    section = str(arguments.get("excludePlexLibrarySectionName") or "")
    return (
        "kids" in root_components
        or bool(re.search(r"(?<![\w])kids(?![\w])", section, re.IGNORECASE))
    )


def _tool_pages(
    messages: Sequence[Mapping[str, object]],
) -> tuple[list[tuple[dict, dict]], str]:
    if not messages or messages[-1].get("role") != "tool":
        return [], "latest-message-is-not-tool-result"

    calls: dict[str, tuple[str, object]] = {}
    for message in messages:
        if message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls")
        if tool_calls is None:
            continue
        if not isinstance(tool_calls, list):
            return [], "malformed-assistant-tool-calls"
        for call in tool_calls:
            if not isinstance(call, Mapping):
                return [], "malformed-assistant-tool-call"
            call_id = str(call.get("id") or "")
            function = call.get("function")
            if (not call_id or not isinstance(function, Mapping)
                    or call_id in calls):
                return [], "malformed-or-duplicate-tool-call-id"
            calls[call_id] = (
                str(function.get("name") or ""),
                function.get("arguments", "{}"),
            )

    pages: list[tuple[dict, dict]] = []
    latest_is_plex = False
    for index, message in enumerate(messages):
        if message.get("role") != "tool":
            continue
        call_id = str(message.get("tool_call_id") or "")
        call = calls.get(call_id)
        if call is None:
            return [], "tool-result-without-matching-call"
        name, raw_arguments = call
        if name != PLEX_LIBRARY_TOOL:
            latest_is_plex = index != len(messages) - 1 and latest_is_plex
            continue
        arguments = _parse_object(raw_arguments)
        output = _parse_object(_message_text(message))
        if arguments is None or output is None:
            return [], "non-json-plex-call-or-result"
        pages.append((arguments, output))
        latest_is_plex = index == len(messages) - 1
    if not pages:
        return [], "no-completed-plex-pages"
    if not latest_is_plex:
        return [], "latest-tool-result-is-not-plex"
    return pages, "candidate"


def attempt_completed_plex_policy(
    messages: Sequence[Mapping[str, object]],
) -> PolicyRenderAttempt:
    """Render the captured Plex workflow only from a complete verified suffix.

    The recognition rules are intentionally semantic but narrow: the latest
    user request must ask for the known list/pagination policy, every call must
    carry the same typed predicate arguments, offsets must form a complete
    conventional sequence, and the last page must explicitly terminate both
    movie and series streams.  Any mismatch returns a reason and no answer.
    """

    if not _supports_captured_plex_intent(messages):
        return PolicyRenderAttempt(None, "unsupported-user-intent")
    pages, page_reason = _tool_pages(messages)
    if not pages:
        return PolicyRenderAttempt(None, page_reason)

    first_arguments = pages[0][0]
    movie_threshold = _canonical_threshold(
        first_arguments.get("movieRatingValue"), media="movie")
    show_threshold = _canonical_threshold(
        first_arguments.get("showRatingValue"), media="show")
    limit = first_arguments.get("limit")
    if (first_arguments.get("mediaType") != "all"
            or first_arguments.get("ratingOperator") != "lte"
            or movie_threshold != "PG-13" or show_threshold != "TV-Y7"
            or not _valid_kids_exclusion(first_arguments)
            or isinstance(limit, bool) or not isinstance(limit, int)
            or not 1 <= limit <= 1000):
        return PolicyRenderAttempt(None, "unsupported-or-invalid-plex-policy")

    semantic_keys = (
        "mediaType", "ratingOperator", "movieRatingValue", "showRatingValue",
        "excludeRootFolderPath", "excludePlexLibrarySectionName", "limit",
    )
    collections: dict[str, list[Mapping[str, object]]] = {
        "movies": [], "series": [],
    }
    for page_index, (arguments, page) in enumerate(pages):
        if arguments.get("offset") != page_index * limit:
            return PolicyRenderAttempt(None, "non-contiguous-plex-offsets")
        if any(arguments.get(key) != first_arguments.get(key)
               for key in semantic_keys):
            return PolicyRenderAttempt(None, "plex-policy-changed-between-pages")
        movie_has_more = page.get("movieHasMore")
        series_has_more = page.get("seriesHasMore")
        if not isinstance(movie_has_more, bool) or not isinstance(
                series_has_more, bool):
            return PolicyRenderAttempt(None, "missing-typed-pagination-state")
        if page_index < len(pages) - 1:
            if not (movie_has_more or series_has_more):
                return PolicyRenderAttempt(None, "page-after-terminal-state")
        elif movie_has_more or series_has_more:
            return PolicyRenderAttempt(None, "incomplete-pagination")
        for key in ("movies", "series"):
            rows = page.get(key)
            if not isinstance(rows, list):
                return PolicyRenderAttempt(None, f"missing-{key}-rows")
            collections[key].extend(rows)

    exclusion = AnySubstringExclusion(
        ("rootFolderPath", "plexLibrarySectionName"), "kids")
    policies = (
        CollectionPolicy(
            "movies", "Movies",
            OrderedPredicate(
                "contentRating", movie_threshold, _MOVIE_LADDER,
                _MOVIE_ALIASES),
            (exclusion,),
        ),
        CollectionPolicy(
            "series", "TV Shows",
            OrderedPredicate(
                "contentRating", show_threshold, _SHOW_LADDER,
                _SHOW_ALIASES),
            (exclusion,),
        ),
    )
    try:
        execution = execute_typed_predicates(collections, policies)
    except ValueError as error:
        return PolicyRenderAttempt(None, f"unverified-row:{error}")

    grouped = {policy.label: [] for policy in policies}
    for row in execution.accepted:
        grouped[row.collection].append(f"{row.title} ({row.ordered_value})")
    lines = [
        f"{policy.label}: " + (
            ", ".join(grouped[policy.label]) if grouped[policy.label] else "None")
        for policy in policies
    ]
    text = "\n".join(lines)
    titles = tuple(row.title for row in execution.accepted)
    return PolicyRenderAttempt(DeterministicRender(
        text=text,
        profile="typed-paginated-plex-policy-v1",
        pages=len(pages),
        input_rows=execution.input_rows,
        accepted_rows=len(execution.accepted),
        rejected_rows=execution.rejected_rows,
        output_titles=titles,
    ), "rendered")
