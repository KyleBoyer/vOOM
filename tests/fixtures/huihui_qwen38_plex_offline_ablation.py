#!/usr/bin/env python3
"""CPU-only Plex quality ablations with a strict 64-token output gate.

This driver never loads model weights and never starts Metal inference.  It
re-scores the complete user-visible output from an archived result, truncates
that same output at a real tokenizer boundary, and compares it with the
production typed-policy renderer and a minimal structured representation.
The deliberately adversarial suffix-marker case proves that a clean
``Final list:`` suffix can no longer hide bad visible text above it.

Example::

    .venv/bin/python tests/fixtures/huihui_qwen38_plex_offline_ablation.py \
      --assert-gate --output logs/huihui_qwen38_plex_offline_ablation.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "fixtures"))

from runtime.policy_executor import attempt_completed_plex_policy
from plex_agent_profile import (  # noqa: E402
    ELIGIBLE_TITLES,
    INELIGIBLE_TITLES,
    PLEX_TOOL,
    SYNTHETIC_PAGES,
    score_profile,
)


DEFAULT_RESULT = ROOT / "logs" / "huihui_qwen38_fast_plex_focused_specific.json"
DEFAULT_TOKENIZER = (
    ROOT / "models" / "Huihui-Qwen3.8-27B-abliterated-mlx-all-mxfp4"
    / "tokenizer.json"
)
CAPTURED_USER = (
    "list the plex movies/tv shows that are age rating PG13 or TV-7 or less"
    '(for younger kids) and whose root folder does NOT contain "/Kids/"\n'
    "Make sure to paginate the plex listing"
)


def _canonical_messages(calls: list[dict]) -> list[dict]:
    messages = [{"role": "user", "content": CAPTURED_USER}]
    for index, page in enumerate(SYNTHETIC_PAGES):
        if index >= len(calls):
            break
        call = calls[index]
        call_id = f"offline_call_{index}"
        messages.extend(({
            "role": "assistant", "content": "", "tool_calls": [{
                "id": call_id, "type": "function", "function": {
                    "name": call.get("name", PLEX_TOOL),
                    "arguments": json.dumps(
                        call.get("arguments") or {},
                        ensure_ascii=False, separators=(",", ":")),
                },
            }],
        }, {
            "role": "tool", "tool_call_id": call_id,
            "content": json.dumps(
                page, ensure_ascii=False, separators=(",", ":")),
        }))
    return messages


def _title_scan(text: str) -> dict[str, list[str]]:
    folded = text.casefold()
    return {
        "eligible_mentions": [
            title for title in ELIGIBLE_TITLES
            if title.casefold() in folded
        ],
        "ineligible_mentions": [
            title for title in INELIGIBLE_TITLES
            if title.casefold() in folded
        ],
    }


def _case(name: str, text: str, calls: list[dict], tokenizer: Tokenizer,
          cap: int) -> dict:
    token_ids = tokenizer.encode(text).ids
    rubric = score_profile(calls, text)
    return {
        "name": name,
        "text": text,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "tokens": len(token_ids),
        "within_output_budget": len(token_ids) <= cap,
        "whole_visible_scan": _title_scan(text),
        "rubric": rubric,
    }


def evaluate(result_path: Path, tokenizer_path: Path, cap: int,
             extra_candidates: list[tuple[str, Path]]) -> dict:
    raw_bytes = result_path.read_bytes()
    artifact = json.loads(raw_bytes)
    calls = artifact.get("calls")
    if not isinstance(calls, list):
        raise ValueError("result artifact has no calls list")
    raw_text = artifact.get("final_text")
    if not isinstance(raw_text, str):
        raise ValueError("result artifact has no string final_text")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    raw_ids = tokenizer.encode(raw_text).ids
    capped_text = tokenizer.decode(raw_ids[:cap], skip_special_tokens=False)

    attempt = attempt_completed_plex_policy(_canonical_messages(calls))
    if attempt.render is None:
        raise ValueError(
            f"production typed renderer did not verify fixture: {attempt.reason}")
    deterministic = attempt.render.text
    structured = json.dumps({
        "movies": ["ALPHA_G", "BRAVO_PG13"],
        "tvShows": ["CHARLIE_TVY", "DELTA_TVY7"],
    }, ensure_ascii=False, separators=(",", ":"))
    suffix_attack = (
        "Incorrect visible selection: ECHO_R, JULIET_TVPG\n\n"
        "Final list:\nALPHA_G\nBRAVO_PG13\nCHARLIE_TVY\nDELTA_TVY7"
    )
    cases = [
        _case("archived_raw_full", raw_text, calls, tokenizer, cap),
        _case("archived_raw_first_64_tokens", capped_text, calls, tokenizer, cap),
        _case("typed_deterministic_render", deterministic, calls, tokenizer, cap),
        _case("strict_structured_json", structured, calls, tokenizer, cap),
        _case("suffix_marker_attack", suffix_attack, calls, tokenizer, cap),
    ]
    for name, path in extra_candidates:
        cases.append(_case(
            f"candidate:{name}", path.read_text(), calls, tokenizer, cap))
    by_name = {case["name"]: case for case in cases}
    strict_names = ("typed_deterministic_render", "strict_structured_json")
    gate = (
        all(by_name[name]["within_output_budget"]
            and by_name[name]["rubric"]["score"] == 100
            and by_name[name]["rubric"]["passed"]
            for name in strict_names)
        and not by_name["suffix_marker_attack"]["rubric"]["passed"]
        and by_name["suffix_marker_attack"]["rubric"]["score"] < 100
    )
    return {
        "schema": "voom.huihui-qwen38-plex-offline-ablation.v1",
        "model_inference_run": False,
        "source": {
            "path": str(result_path),
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "calls": len(calls),
            "synthetic_page_fixture": (
                "tests/fixtures/plex_agent_profile.py:SYNTHETIC_PAGES"),
        },
        "tokenizer": str(tokenizer_path),
        "max_output_tokens": cap,
        "scoring_contract": "whole-user-visible-output-v1",
        "typed_renderer": {
            "profile": attempt.render.profile,
            "reason": attempt.reason,
            "pages": attempt.render.pages,
            "input_rows": attempt.render.input_rows,
            "accepted_rows": attempt.render.accepted_rows,
            "rejected_rows": attempt.render.rejected_rows,
        },
        "cases": cases,
        "passed": gate,
    }


def _candidate(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("candidate must be NAME=TEXT_FILE")
    return name, Path(path).expanduser().resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--max-output-tokens", type=int, default=64)
    parser.add_argument("--candidate", type=_candidate, action="append", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--assert-gate", action="store_true")
    args = parser.parse_args()
    if args.max_output_tokens <= 0:
        parser.error("--max-output-tokens must be positive")
    report = evaluate(
        args.result.expanduser().resolve(),
        args.tokenizer.expanduser().resolve(),
        args.max_output_tokens,
        args.candidate,
    )
    encoded = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")
    return int(args.assert_gate and not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
