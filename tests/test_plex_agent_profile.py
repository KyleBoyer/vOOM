import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))

from plex_agent_profile import (PLEX_TOOL, compact_plex_planner_tool,
                                evaluate_plex_policy_adapter,
                                load_profile_request, score_profile)


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


def test_explicit_post_filtering_and_final_section_are_scored_semantically():
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
    assert all(result["ineligible_titles_absent"].values())
    assert result["answer_slice_used"]
    assert result["score"] == 100
    assert result["passed"]


def test_reverification_heading_delimits_clean_selection_from_analysis():
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
    assert result["answer_slice_used"]
    assert result["passed"]


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
