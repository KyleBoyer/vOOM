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
    The complete 130+ tool request, with only ``model``, streaming, and output
    budget overridden.  This measures the actual large-agent path.

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
import urllib.error
import urllib.request
from pathlib import Path

import psutil


PLEX_TOOL = "plugin__plex__plex_list_library"
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
    plex = [call for call in calls if call.get("name") == PLEX_TOOL
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


def load_profile_request(capture: Path, model: str, profile: str,
                         max_output_tokens: int,
                         reasoning_effort: str | None = None,
                         temperature: float = 0.0,
                         tool_choice: str = "capture",
                         tool_schema_profile: str = "full") -> tuple[dict, dict]:
    """Return a runnable request and non-sensitive capture identity."""
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
    request["stream"] = False
    request["store"] = False
    request["temperature"] = float(temperature)
    # Each synthetic page must inform the next offset.  Speculative parallel
    # page calls would be executed against no prior result and cannot prove
    # pagination comprehension, so make the replay deliberately sequential.
    request["parallel_tool_calls"] = False
    request["max_output_tokens"] = max_output_tokens
    request.pop("max_tokens", None)
    if tool_choice == "specific":
        request["tool_choice"] = {"type": "function", "name": PLEX_TOOL}
    elif tool_choice != "capture":
        request["tool_choice"] = tool_choice
    if reasoning_effort is not None:
        request["reasoning"] = {"effort": reasoning_effort}
    identity = {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "tools": len(tools),
        "profile": profile,
        "reasoning_effort": reasoning_effort or "unspecified",
        "temperature": float(temperature),
        "tool_choice": tool_choice,
        "tool_schema_profile": tool_schema_profile,
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
            return json.loads(response.read()), time.perf_counter() - started
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
    try:
        return int(arguments.get("offset"))
    except (TypeError, ValueError):
        return None


def _final_answer_slice(text: str) -> str:
    """Prefer an explicitly labelled final selection over analysis prose."""
    markers = list(re.finditer(
        r"(?im)^\s*(?:#{1,6}\s*)?(?:final\s+(?:list|answer)|"
        r"titles\s+meeting\s+criteria|(?:let\s+me\s+)?re-verify)\s*:?\s*$",
        text or ""))
    return (text[markers[-1].end():] if markers else text).strip()


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


def score_profile(calls: list[dict], final_text: str) -> dict:
    """Score the four independent behaviors on a stable 100-point rubric."""
    plex_calls = [call for call in calls if call.get("name") == PLEX_TOOL]
    first = plex_calls[0].get("arguments") if plex_calls else None
    first = first if isinstance(first, dict) else {}
    offsets = [_offset(call.get("arguments")) for call in plex_calls]
    numeric_offsets = [value for value in offsets if value is not None]
    increasing_offsets = (
        len(numeric_offsets) >= 2
        and all(right > left for left, right in zip(
            numeric_offsets, numeric_offsets[1:])))
    answer_text = _final_answer_slice(final_text)
    root_or_section_filter_or_verification = (
        "/kids/" in str(first.get("excludeRootFolderPath") or "").lower()
        or "kids" in str(
            first.get("excludePlexLibrarySectionName") or "").lower()
        or _explicit_kids_root_verification(final_text))
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


def run_profile(request: dict, url: str, timeout: float,
                max_tool_rounds: int) -> dict:
    pressure_before = _pressure()
    turns = []
    all_calls: list[dict] = []
    final_text = ""
    page_index = 0
    started = time.perf_counter()
    for turn_index in range(max_tool_rounds + 1):
        response, wall = _post(url, request, timeout)
        calls = response_calls(response)
        text = response_text(response)
        turns.append({
            "turn": turn_index + 1,
            "wall_seconds": round(wall, 4),
            "usage": response.get("usage"),
            "timing": response.get("vmodel_timing"),
            "tool_selection": response.get("vmodel_tool_selection"),
            "constraint": response.get("vmodel_constraint"),
            "call_names": [call.get("name") for call in calls],
            "error": response.get("error"),
            "response_status": response.get("status"),
        })
        if response.get("error"):
            break
        if text:
            final_text = text
        all_calls.extend(calls)
        plex_calls = [call for call in calls if call.get("name") == PLEX_TOOL]
        if not plex_calls:
            break
        call = plex_calls[0]
        page = SYNTHETIC_PAGES[min(page_index, len(SYNTHETIC_PAGES) - 1)]
        _append_call_and_result(request, call, page)
        # A forced choice governs the planning turn only. Keeping it on every
        # follow-up request would make a compliant model call forever even
        # after both HasMore flags are false, which tests the harness rather
        # than the model's pagination/termination behavior.
        if request.get("tool_choice") == "required" or isinstance(
                request.get("tool_choice"), dict):
            request["tool_choice"] = "auto"
        page_index += 1

    rubric = score_profile(all_calls, final_text)
    pressure_after = _pressure()
    return {
        "gate": "plex-agent-profile-v1",
        "model": request.get("model"),
        "passed": rubric["passed"],
        "rubric": rubric,
        "turns": turns,
        "calls": [{
            "name": call.get("name"),
            "arguments": call.get("arguments"),
        } for call in all_calls],
        "final_text": final_text,
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
    parser.add_argument("--start-server", action="store_true")
    parser.add_argument("--server-ready-timeout", type=float, default=120)
    parser.add_argument("--server-log", type=Path)
    parser.add_argument("--result-json", type=Path)
    args = parser.parse_args()
    if (args.timeout <= 0 or args.max_output_tokens <= 0
            or args.max_tool_rounds < 0 or args.temperature < 0):
        parser.error(
            "timeouts/output must be positive; tool rounds and temperature "
            "must be nonnegative")

    request, identity = load_profile_request(
        args.capture, args.model, args.profile, args.max_output_tokens,
        args.reasoning_effort, args.temperature, args.tool_choice,
        args.tool_schema_profile)
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
        result = run_profile(request, args.url, args.timeout, args.max_tool_rounds)
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
