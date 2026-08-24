"""CPU-only gates for the fail-closed typed policy renderer."""

from __future__ import annotations

import json

import pytest

from runtime.policy_executor import (
    AnySubstringExclusion,
    CollectionPolicy,
    OrderedPredicate,
    PLEX_LIBRARY_TOOL,
    attempt_completed_plex_policy,
    execute_typed_predicates,
)


USER = (
    'list the plex movies/tv shows that are age rating PG13 or TV-7 or less '
    '(for younger kids) and whose root folder does NOT contain "/Kids/"\n'
    "Make sure to paginate the plex listing"
)


PAGES = ({
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
    ],
    "movieHasMore": True,
    "seriesHasMore": True,
}, {
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
        {"title": "JULIET_TVPG", "contentRating": "TV-PG",
         "rootFolderPath": "/Media/TV",
         "plexLibrarySectionName": "TV Shows"},
    ],
    "movieHasMore": False,
    "seriesHasMore": False,
})


def _arguments(offset: int) -> dict:
    return {
        "mediaType": "all",
        "ratingOperator": "lte",
        "movieRatingValue": "PG-13",
        "showRatingValue": "TV-Y7",
        "excludeRootFolderPath": "/Kids/",
        "limit": 50,
        "offset": offset,
    }


def _messages(pages=PAGES, *, offsets=(0, 50), user=USER,
              mutate_arguments=None) -> list[dict]:
    messages = [{"role": "user", "content": user}]
    for index, (page, offset) in enumerate(zip(pages, offsets)):
        arguments = _arguments(offset)
        if mutate_arguments is not None:
            mutate_arguments(index, arguments)
        call_id = f"call_{index}"
        messages.extend(({
            "role": "assistant", "content": "", "tool_calls": [{
                "id": call_id, "type": "function", "function": {
                    "name": PLEX_LIBRARY_TOOL,
                    "arguments": json.dumps(arguments),
                },
            }],
        }, {
            "role": "tool", "tool_call_id": call_id,
            "content": json.dumps(page),
        }))
    return messages


def test_typed_executor_applies_ladder_and_component_exclusion():
    policies = (CollectionPolicy(
        "items", "Items",
        OrderedPredicate("rating", "PG", ("G", "PG", "R"), {}),
        (AnySubstringExclusion(("path", "section"), "kids"),),
    ),)
    execution = execute_typed_predicates({"items": [
        {"title": "A", "rating": "G", "path": "/Media/Movies"},
        {"title": "B", "rating": "PG", "path": "/Media/Kids/Movies"},
        {"title": "C", "rating": "R", "path": "/Media/Movies"},
    ]}, policies)
    assert [row.title for row in execution.accepted] == ["A"]
    assert execution.input_rows == 3
    assert execution.rejected_rows == 2


def test_typed_executor_rejects_unknown_ordered_value_instead_of_guessing():
    policy = CollectionPolicy(
        "items", "Items",
        OrderedPredicate("rating", "PG", ("G", "PG", "R"), {}), ())
    with pytest.raises(ValueError, match="unknown rating value"):
        execute_typed_predicates({"items": [
            {"title": "Ambiguous", "rating": "PG-Custom"},
        ]}, (policy,))


def test_completed_plex_pages_render_only_eligible_titles():
    attempt = attempt_completed_plex_policy(_messages())
    assert attempt.reason == "rendered"
    assert attempt.render is not None
    assert attempt.render.text == (
        "Movies: ALPHA_G (G), BRAVO_PG13 (PG-13)\n"
        "TV Shows: CHARLIE_TVY (TV-Y), DELTA_TVY7 (TV-Y7)")
    assert attempt.render.pages == 2
    assert attempt.render.input_rows == 8
    assert attempt.render.accepted_rows == 4
    assert attempt.render.rejected_rows == 4
    assert "FOXTROT" not in attempt.render.text
    assert "JULIET" not in attempt.render.text


@pytest.mark.parametrize(("messages", "reason"), [
    (_messages(pages=(PAGES[0],), offsets=(0,)), "incomplete-pagination"),
    (_messages(offsets=(0, 75)), "non-contiguous-plex-offsets"),
    (_messages(user="List Plex movies as a CSV."), "unsupported-user-intent"),
])
def test_plex_renderer_fails_closed_for_incomplete_or_unsupported_history(
        messages, reason):
    attempt = attempt_completed_plex_policy(messages)
    assert attempt.render is None
    assert attempt.reason == reason


def test_plex_renderer_fails_closed_when_policy_changes_between_pages():
    attempt = attempt_completed_plex_policy(_messages(
        mutate_arguments=lambda index, arguments: arguments.update(
            showRatingValue="TV-PG") if index == 1 else None))
    assert attempt.render is None
    assert attempt.reason == "plex-policy-changed-between-pages"


def test_plex_renderer_fails_closed_on_unknown_nonempty_rating():
    pages = [dict(page) for page in PAGES]
    pages[1] = {**pages[1], "movies": [
        *pages[1]["movies"],
        {"title": "UNKNOWN", "contentRating": "FAMILY-7",
         "rootFolderPath": "/Media/Movies",
         "plexLibrarySectionName": "Movies"},
    ]}
    attempt = attempt_completed_plex_policy(_messages(tuple(pages)))
    assert attempt.render is None
    assert attempt.reason.startswith("unverified-row:unknown contentRating")
