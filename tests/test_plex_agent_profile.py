import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))

from plex_agent_profile import (PLEX_MEDIA_TOOL, PLEX_TOOL,
                                compact_plex_planner_tool,
                                evaluate_plex_policy_adapter,
                                load_kai_tool_result_export,
                                load_profile_request, score_actual_export,
                                score_profile)


def _call(**arguments):
    return {"name": PLEX_TOOL, "arguments": arguments}


def test_perfect_plex_plan_and_answer_scores_100():
    calls = [
        _call(mediaType="all", excludeRootFolderPath="/Kids/",
              ratingOperator="lte", movieRatingValue="PG-13",
              showRatingValue="TV-Y7", limit=32, offset=0),
        _call(mediaType="all", excludeRootFolderPath="/Kids/",
              ratingOperator="lte", movieRatingValue="PG-13",
              showRatingValue="TV-Y7", limit=32, offset=32),
    ]
    text = "ALPHA_G, BRAVO_PG13, CHARLIE_TVY, DELTA_TVY7"
    result = score_profile(calls, text)
    assert result["score"] == 100
    assert result["passed"] is True


def test_kids_library_section_is_equivalent_to_kids_root_filter():
    calls = [
        _call(mediaType="all", excludePlexLibrarySectionName="Kids",
              ratingOperator="lte", movieRatingValue="PG-13",
              showRatingValue="TV-Y7", limit=32, offset=0),
        _call(mediaType="all", excludePlexLibrarySectionName="Kids",
              ratingOperator="lte", movieRatingValue="PG-13",
              showRatingValue="TV-Y7", limit=32, offset=32),
    ]
    text = "ALPHA_G, BRAVO_PG13, CHARLIE_TVY, DELTA_TVY7"
    result = score_profile(calls, text)
    assert result["checks"]["excluded_kids_root"]["passed"]
    assert result["score"] == 100
    assert result["passed"]


def test_generic_rating_and_missing_path_fail_critical_checks():
    calls = [
        _call(mediaType="all", ratingOperator="lte", ratingValue="PG-13",
              limit=32, offset=0),
        _call(mediaType="all", ratingOperator="lte", ratingValue="PG-13",
              limit=32, offset=32),
    ]
    text = "ALPHA_G, BRAVO_PG13, CHARLIE_TVY, DELTA_TVY7"
    result = score_profile(calls, text)
    assert not result["checks"]["excluded_kids_root"]["passed"]
    assert not result["checks"]["movie_rating_pg13"]["passed"]
    assert not result["checks"]["show_rating_tvy7"]["passed"]
    assert result["passed"] is False


def test_no_pagination_and_leaked_bad_titles_are_scored_separately():
    calls = [
        _call(mediaType="all", excludeRootFolderPath="/Kids/",
              ratingOperator="lte", movieRatingValue="PG13",
              showRatingValue="TV-7", limit=32, offset=0),
    ]
    text = "ALPHA_G, ECHO_R, CHARLIE_TVY"
    result = score_profile(calls, text)
    assert result["checks"]["movie_rating_pg13"]["passed"]
    assert not result["checks"]["show_rating_tvy7"]["passed"]
    assert not result["checks"]["paginated_after_has_more"]["passed"]
    assert not result["eligible_titles_found"]["BRAVO_PG13"]
    assert not result["ineligible_titles_absent"]["ECHO_R"]
    assert result["passed"] is False


def test_omitted_first_offset_uses_tool_schema_default_zero():
    calls = [_call(), _call(limit=100, offset=100)]
    result = score_profile(calls, "")
    assert result["offsets"] == [0, 100]
    assert result["checks"]["initial_offset_zero"]["passed"]
    assert result["checks"]["pagination_offset_increased"]["passed"]


def test_visible_bad_prefix_cannot_be_hidden_behind_final_list_heading():
    calls = [
        _call(mediaType="all", ratingOperator="lte",
              movieRatingValue="PG-13", showRatingValue="TV-Y7",
              limit=50, offset=0),
        _call(mediaType="all", ratingOperator="lte",
              movieRatingValue="PG-13", showRatingValue="TV-Y7",
              limit=50, offset=50),
    ]
    text = """Analysis:
FOXTROT_KIDS_PG has /Media/Kids/Movies, which contains /Kids/: EXCLUDE.
HOTEL_KIDS_TVY has /Media/Kids/TV, which contains /Kids/: EXCLUDE.
ECHO_R, GOLF_TV14, INDIA_UNRATED, and JULIET_TVPG are also excluded.

### Final List:
ALPHA_G
BRAVO_PG13
CHARLIE_TVY
DELTA_TVY7
"""
    result = score_profile(calls, text)
    assert result["checks"]["excluded_kids_root"]["passed"]
    assert not any(result["ineligible_titles_absent"].values())
    assert not result["answer_slice_used"]
    assert result["whole_visible_output_scanned"]
    assert result["score"] == 85
    assert not result["passed"]


def test_visible_reverification_heading_does_not_amnesty_bad_titles():
    calls = [
        _call(mediaType="all", excludeRootFolderPath="/Kids/",
              ratingOperator="lte", movieRatingValue="PG-13",
              showRatingValue="TV-Y7", limit=200, offset=0),
        _call(mediaType="all", excludeRootFolderPath="/Kids/",
              ratingOperator="lte", movieRatingValue="PG-13",
              showRatingValue="TV-Y7", limit=200, offset=200),
    ]
    text = """Analysis mentions ECHO_R and JULIET_TVPG as rejects.

Let me re-verify:
ALPHA_G
BRAVO_PG13
CHARLIE_TVY
DELTA_TVY7
"""
    result = score_profile(calls, text)
    assert not result["answer_slice_used"]
    assert not result["ineligible_titles_absent"]["ECHO_R"]
    assert not result["ineligible_titles_absent"]["JULIET_TVPG"]
    assert result["score"] == 95
    assert not result["passed"]


def test_profile_can_explicitly_enable_reasoning_without_mutating_capture(tmp_path):
    capture = tmp_path / "capture.json"
    tools = [{"type": "function", "name": f"tool_{index}",
              "description": "fixture", "parameters": {"type": "object"}}
             for index in range(130)]
    tools.append({
        "type": "function", "name": PLEX_TOOL,
        "description": "list Plex movies and TV",
        "parameters": {"type": "object"},
    })
    capture.write_text(json.dumps({
        "model": "old", "tools": tools,
        "input": [{"role": "user", "content": [{
            "type": "input_text",
            "text": "Use Plex for movies/TV and exclude /Kids/.",
        }]}],
    }))

    request, identity = load_profile_request(
        capture, "new", "focused", 512, "high")
    assert request["reasoning"] == {"effort": "high"}
    assert request["temperature"] == 0.0
    assert request["max_output_tokens"] == 512
    assert identity["reasoning_effort"] == "high"
    assert identity["temperature"] == 0.0
    assert identity["tool_choice"] == "capture"
    assert identity["tool_schema_profile"] == "full"
    assert "reasoning" not in json.loads(capture.read_text())


def test_profile_can_force_the_specific_plex_tool(tmp_path):
    capture = tmp_path / "capture.json"
    tools = [{"type": "function", "name": f"tool_{index}",
              "description": "fixture", "parameters": {"type": "object"}}
             for index in range(130)]
    tools.append({
        "type": "function", "name": PLEX_TOOL,
        "description": "list Plex movies and TV",
        "parameters": {"type": "object"},
    })
    capture.write_text(json.dumps({
        "model": "old", "tools": tools, "tool_choice": "auto",
        "input": [{"role": "user", "content": [{
            "type": "input_text",
            "text": "Use Plex for movies/TV and exclude /Kids/.",
        }]}],
    }))
    request, identity = load_profile_request(
        capture, "new", "focused", 128, tool_choice="specific")
    assert request["tool_choice"] == {"type": "function", "name": PLEX_TOOL}
    assert identity["tool_choice"] == "specific"


def test_compact_planner_schema_keeps_only_relevant_plex_arguments():
    properties = {
        name: {"type": "string"} for name in (
            "mediaType", "ratingOperator", "movieRatingValue",
            "showRatingValue", "excludeRootFolderPath",
            "excludePlexLibrarySectionName", "limit", "offset", "query")
    }
    tool = {"type": "function", "name": PLEX_TOOL,
            "description": "full", "parameters": {
                "type": "object", "properties": properties,
                "required": list(properties),
                "x-optional": list(properties),
            }}
    compact = compact_plex_planner_tool(tool)
    schema = compact["parameters"]
    assert "query" not in schema["properties"]
    assert set(schema["properties"]) == {
        "mediaType", "ratingOperator", "movieRatingValue", "showRatingValue",
        "excludeRootFolderPath", "excludePlexLibrarySectionName", "limit",
        "offset",
    }
    assert set(schema["required"]) == {
        "mediaType", "ratingOperator", "movieRatingValue", "showRatingValue",
        "limit", "offset",
    }
    assert "x-optional" not in schema
    assert len(schema["anyOf"]) == 2
    assert schema["properties"]["showRatingValue"]["enum"] == [
        "TV-Y", "TV-Y7", "TV-Y7-FV", "TV-G", "TV-PG", "TV-14", "TV-MA"]
    assert schema["properties"]["showRatingValue"]["type"] == "string"
    assert "query" in tool["parameters"]["properties"]


def test_policy_bound_schema_removes_already_explicit_rating_choice():
    properties = {name: {"anyOf": [{"type": "string"}, {"type": "null"}]}
                  for name in (
                      "mediaType", "ratingOperator", "movieRatingValue",
                      "showRatingValue", "excludeRootFolderPath",
                      "excludePlexLibrarySectionName", "limit", "offset")}
    tool = {"type": "function", "name": PLEX_TOOL,
            "parameters": {"type": "object", "properties": properties}}
    bound = compact_plex_planner_tool(tool, bind_policy=True)
    schema = bound["parameters"]["properties"]
    assert schema["mediaType"] == {"const": "all"}
    assert schema["ratingOperator"] == {"const": "lte"}
    assert schema["movieRatingValue"] == {"const": "PG-13"}
    assert schema["showRatingValue"] == {"const": "TV-Y7"}


def test_policy_adapter_repairs_explicit_tv_alias_and_owns_pagination():
    adapted = evaluate_plex_policy_adapter([_call(
        mediaType="all", ratingOperator="lte",
        movieRatingValue="PG-13", showRatingValue="TV-7",
        excludePlexLibrarySectionName="Kids Movies", limit=50, offset=0)])
    assert adapted["passed"]
    assert adapted["rating_repairs"] == ["TV-7->TV-Y7"]
    assert adapted["pages_fetched"] == 2
    assert adapted["final_titles"] == [
        "ALPHA_G", "CHARLIE_TVY", "BRAVO_PG13", "DELTA_TVY7"]


def test_policy_adapter_fails_closed_without_kids_scope():
    adapted = evaluate_plex_policy_adapter([_call(
        mediaType="all", ratingOperator="lte",
        movieRatingValue="PG-13", showRatingValue="TV-Y7",
        limit=50, offset=0)])
    assert not adapted["passed"]
    assert adapted["reason"] == "proposal_failed_policy_validation"


def test_planner_schema_cannot_masquerade_as_unchanged_capture(tmp_path):
    capture = tmp_path / "capture.json"
    tools = [{
        "type": "function", "name": f"fixture_{index}",
        "parameters": {"type": "object"},
    } for index in range(130)]
    tools.append({
        "type": "function", "name": PLEX_TOOL,
        "parameters": {"type": "object"},
    })
    capture.write_text(json.dumps({
        "tools": tools,
        "input": [{"role": "user", "content": [{
            "type": "input_text",
            "text": "Use Plex for movies/TV and exclude /Kids/.",
        }]}],
    }))
    try:
        load_profile_request(
            capture, "test", "captured", 64,
            tool_schema_profile="planner")
    except ValueError as error:
        assert "must preserve" in str(error)
    else:
        raise AssertionError("captured profile accepted a rewritten schema")


def test_captured_adapted_preserves_catalog_but_rewrites_only_plex(tmp_path):
    capture = tmp_path / "capture.json"
    tools = [{
        "type": "function", "name": f"fixture_{index}",
        "parameters": {"type": "object"},
    } for index in range(130)]
    plex_properties = {name: {"anyOf": [
        {"type": "string"}, {"type": "null"}]}
        for name in (
            "mediaType", "ratingOperator", "movieRatingValue",
            "showRatingValue", "excludeRootFolderPath",
            "excludePlexLibrarySectionName", "limit", "offset")}
    tools.insert(17, {"type": "function", "name": PLEX_TOOL,
                      "parameters": {"type": "object",
                                     "properties": plex_properties}})
    capture.write_text(json.dumps({
        "tools": tools,
        "input": [{"role": "user", "content": [{
            "type": "input_text",
            "text": "Use Plex for movies/TV and exclude /Kids/.",
        }]}],
    }))
    request, identity = load_profile_request(
        capture, "test", "captured-adapted", 64,
        tool_schema_profile="policy")
    assert len(request["tools"]) == len(tools)
    assert [tool["name"] for tool in request["tools"]] == [
        tool["name"] for tool in tools]
    adapted = request["tools"][17]["parameters"]["properties"]
    assert adapted["showRatingValue"] == {"const": "TV-Y7"}
    assert request["tools"][0] == tools[0]
    assert identity["profile"] == "captured-adapted"


def test_harmony_analysis_channel_does_not_count_as_the_answer():
    """A correct gpt-oss reply rejects the ineligible titles BY NAME inside
    harmony's analysis channel. Scoring the concatenated channels counted
    those rejections as leaks and zeroed the exclusion points -- the exact
    reason the 2026-07-31 gate read 85.0/fail while its final channel was
    perfect. Verbatim shape from that run."""
    calls = [
        _call(mediaType="all", excludeRootFolderPath="/Kids/",
              ratingOperator="lte", movieRatingValue="PG-13",
              showRatingValue="TV-Y7", limit=100, offset=0),
        _call(mediaType="all", excludeRootFolderPath="/Kids/",
              ratingOperator="lte", movieRatingValue="PG-13",
              showRatingValue="TV-Y7", limit=100, offset=100),
    ]
    text = (
        "analysisFirst page movies: ALPHA_G G root /Media/Movies OK include. "
        "ECHO_R R excluded. FOXTROT_KIDS_PG root /Media/Kids/Movies contains "
        "/Kids/ so exclude.\n\nSeries: CHARLIE_TVY include. GOLF_TV14 TV-14 "
        "excluded. BRAVO_PG13 include. INDIA_UNRATED unknown exclude. "
        "DELTA_TVY7 include. HOTEL_KIDS_TVY root /Media/Kids/TV contains "
        "/Kids/ exclude. JULIET_TVPG TV-PG exclude.\n\n"
        "Thus final list: ALPHA_G, CHARLIE_TVY, BRAVO_PG13, DELTA_TVY7.\n\n"
        "Return plain list.assistantfinalALPHA_G\nCHARLIE_TVY\nBRAVO_PG13\n"
        "DELTA_TVY7")
    result = score_profile(calls, text)
    assert result["exclusion_points"] == 15
    assert all(result["ineligible_titles_absent"].values())
    assert all(result["eligible_titles_found"].values())
    assert result["score"] == 100
    assert result["passed"] is True


def test_truncated_harmony_analysis_still_fails_without_a_final_channel():
    """The safety direction: a run cut off inside analysis has produced no
    answer, and must not be rescued by slicing."""
    calls = [
        _call(mediaType="all", excludeRootFolderPath="/Kids/",
              ratingOperator="lte", movieRatingValue="PG-13",
              showRatingValue="TV-Y7", limit=100, offset=0),
        _call(mediaType="all", excludeRootFolderPath="/Kids/",
              ratingOperator="lte", movieRatingValue="PG-13",
              showRatingValue="TV-Y7", limit=100, offset=100),
    ]
    text = "analysisALPHA_G include. ECHO_R excluded. Still verifying"
    result = score_profile(calls, text)
    assert result["passed"] is False


def test_exact_kai_export_pages_drive_full_catalog_oracle(tmp_path):
    rows = [
        {"title": "Allowed Movie", "type": "movie", "contentRating": "PG-13",
         "sectionName": "Movies"},
        {"title": "Too Mature", "type": "movie", "contentRating": "R",
         "sectionName": "Movies"},
        {"title": "Allowed Show", "type": "show", "contentRating": "TV-Y7",
         "sectionName": "TV Shows"},
        {"title": "Kid Root Proxy", "type": "movie", "contentRating": "G",
         "sectionName": "Kid Movies"},
    ]
    parts = []
    for offset in (0, 2):
        page = rows[offset:offset + 2]
        parts.append({
            "type": "tool-call", "toolName": PLEX_MEDIA_TOOL,
            "toolCallId": f"call_{offset}", "args": {"limit": 2, "offset": offset},
            "result": {"offset": offset, "limit": 2, "returned": len(page),
                       "total": 4, "hasMore": offset == 0, "media": page},
        })
    export = tmp_path / "kai.json"
    export.write_text(json.dumps({
        "messages": [{"role": "assistant", "content": parts}],
    }))

    calls, identity = load_kai_tool_result_export(export)
    assert identity["offsets"] == [0, 2]
    assert identity["rows"] == 4
    assert not identity["root_path_evidence_available"]

    score = score_actual_export("Allowed Movie\nAllowed Show", calls)
    assert score["inferred_catalog_match"]
    assert not score["strict_evidence_passed"]
    assert score["expected_count"] == 2
    assert score["mentioned_ineligible_count"] == 0


def test_kai_export_oracle_rejects_unrelated_live_answer(tmp_path):
    export = tmp_path / "kai.json"
    export.write_text(json.dumps({
        "messages": [{"role": "assistant", "content": [{
            "type": "tool-call", "toolName": PLEX_MEDIA_TOOL,
            "toolCallId": "call_0", "args": {"limit": 100, "offset": 0},
            "result": {"offset": 0, "limit": 100, "returned": 1,
                       "total": 1, "hasMore": False, "media": [{
                           "title": "Expected", "type": "movie",
                           "contentRating": "PG", "sectionName": "Movies",
                       }]},
        }]}],
    }))
    calls, _identity = load_kai_tool_result_export(export)
    score = score_actual_export(
        "Who is mentioned in the show Young Sheldon?", calls)
    assert not score["inferred_catalog_match"]
    assert score["missing_count"] == 1


def test_kids_exclusion_done_in_reasoning_still_scores_when_delivered_out_of_band():
    """The model may implement the Kids exclusion by post-filtering in its
    reasoning rather than via a tool argument. Harmony now delivers that
    reasoning out-of-band, so the witness must still find it there."""
    calls = [
        _call(mediaType="all", ratingOperator="lte", movieRatingValue="PG-13",
              showRatingValue="TV-Y7", limit=100, offset=0),
        _call(mediaType="all", ratingOperator="lte", movieRatingValue="PG-13",
              showRatingValue="TV-Y7", limit=100, offset=100),
    ]
    reasoning = (
        "FOXTROT_KIDS_PG has rootFolderPath /Media/Kids/Movies which contains "
        "/Kids/ so exclude it. HOTEL_KIDS_TVY rootFolderPath /Media/Kids/TV "
        "contains /Kids/ so exclude it too.")
    answer = "ALPHA_G\nCHARLIE_TVY\nBRAVO_PG13\nDELTA_TVY7"
    without = score_profile(calls, answer)
    with_reasoning = score_profile(calls, answer, reasoning)
    assert without["checks"]["excluded_kids_root"]["passed"] is False
    assert with_reasoning["checks"]["excluded_kids_root"]["passed"] is True
    assert with_reasoning["passed"] is True


from plex_agent_profile import _mentioned_catalog_titles


def test_typographic_title_variants_are_matched():
    """Models render titles with non-breaking hyphens, curly apostrophes,
    narrow no-break spaces, and dropped leading articles. None of those are
    semantic differences, but re.escape matching missed all of them: the
    2026-07-31 real-export run scored 7 matches where 45 were present."""
    catalog = ("The Amazing Spider-Man", "The A-Team", "God's Not Dead",
               "Arrival")
    answer = (
        "Amazing Spider‑Man (2012) – PG‑13\n"
        "A Team (2010) – PG‑13\n"
        "God’s Not Dead (2014) – PG\n"
        "Arrival (2016) - PG-13\n")
    assert _mentioned_catalog_titles(answer, catalog) == set(catalog)


def test_explicitly_rejected_titles_are_not_counted_as_leaks():
    """Naming a row in order to reject it is the requested behavior, not a
    leak. Only titles the answer actually asserts should count."""
    answer = (
        "Anastasia (1997) - G [Excluded: Kid Movies]\n"
        "Balto (1995) - G [Excluded: Kid Movies]\n"
        "Arrival (2016) - PG-13\n")
    found = _mentioned_catalog_titles(answer, ("Anastasia", "Balto", "Arrival"))
    assert found == {"Arrival"}


def test_an_unlabelled_ineligible_title_is_still_a_leak():
    """The exclusion allowance must not become a blanket amnesty: a title
    listed with no rejection marker is still claimed."""
    answer = "Arrival (2016) - PG-13\nAnastasia (1997) - G\n"
    found = _mentioned_catalog_titles(answer, ("Arrival", "Anastasia"))
    assert found == {"Arrival", "Anastasia"}


def test_an_eligible_title_marked_excluded_is_not_credited():
    """Symmetry: wrongly rejecting a required title is a miss, not a hit."""
    answer = "Arrival (2016) - PG-13 [Excluded: rating too high]\n"
    assert _mentioned_catalog_titles(answer, ("Arrival",)) == set()


def test_a_later_exclusion_marker_does_not_suppress_earlier_titles():
    """Regression: folding separators must preserve line breaks. Collapsing
    newlines merged the whole answer into one line, so a single later
    '[Excluded: ...]' marker suppressed every title above it."""
    answer = (
        "Arrival (2016) - PG-13\n"
        "Avatar (2009) - PG-13\n"
        "Anastasia (1997) - G [Excluded: Kid Movies]\n")
    found = _mentioned_catalog_titles(answer, ("Arrival", "Avatar", "Anastasia"))
    assert found == {"Arrival", "Avatar"}
