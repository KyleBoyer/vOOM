"""Strict root-path evidence must cover the complete exported catalog."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))

from plex_agent_profile import score_actual_export


def _row(title, section="Movies", **fields):
    return {"title": title, "type": "movie", "contentRating": "PG",
            "sectionName": section, **fields}


def _score(answer, *rows):
    return score_actual_export(answer, [{"result": {"media": list(rows)}}])


def test_partial_root_evidence_does_not_certify_proxy_rows():
    result = _score(
        "Rooted Movie\nProxy Movie",
        _row("Rooted Movie", rootFolderPath="/Media/Movies"),
        _row("Proxy Movie"),
        _row("Proxy Kids Movie", "Kids Movies"),
    )
    assert result["inferred_catalog_match"]
    assert result["root_path_evidence_available"]
    assert not result["root_path_evidence_complete"]
    assert not result["strict_evidence_passed"]
    assert result["root_path_evidence_rows"] == 1
    assert result["root_path_evidence_missing_rows"] == 2
    assert result["exclusion_basis"].startswith("mixed rootFolderPath")


def test_complete_root_evidence_uses_paths_not_section_proxy():
    rows = (
        _row("Allowed Movie", "Kids Movies", rootFolderPath="/Media/Movies"),
        _row("Excluded Movie", rootFolderPath="/Media/Kids/Movies"),
    )
    result = _score("Allowed Movie", *rows)
    assert result["inferred_catalog_match"]
    assert result["strict_evidence_passed"]
    assert result["root_path_evidence_complete"]
    assert result["root_path_evidence_rows"] == 2
    assert result["root_path_evidence_missing_rows"] == 0
    assert result["exclusion_basis"] == "rootFolderPath"
    assert not _score("Excluded Movie", *rows)["strict_evidence_passed"]


def test_section_only_export_remains_explicitly_inferred():
    result = _score("Allowed Movie", _row("Allowed Movie"),
                    _row("Excluded Movie", "Kids Movies"))
    assert result["inferred_catalog_match"]
    assert not result["strict_evidence_passed"]
    assert not result["root_path_evidence_available"]
    assert not result["root_path_evidence_complete"]
    assert result["root_path_evidence_rows"] == 0
    assert result["root_path_evidence_missing_rows"] == 2
    assert result["exclusion_basis"].startswith("sectionName Kid/Kids proxy")


@pytest.mark.parametrize("unusable_root", [None, "", "  \t", 17, False])
def test_unusable_root_is_missing_evidence_and_uses_section_proxy(unusable_root):
    result = _score(
        "Allowed Movie",
        _row("Allowed Movie", rootFolderPath=" /Media/Movies "),
        _row("Excluded Movie", "Kids Movies", rootFolderPath=unusable_root),
    )
    assert result["inferred_catalog_match"]
    assert not result["strict_evidence_passed"]
    assert result["root_path_evidence_rows"] == 1
    assert result["root_path_evidence_missing_rows"] == 1


def test_empty_export_does_not_claim_root_evidence():
    result = _score("")
    assert not result["root_path_evidence_available"]
    assert not result["root_path_evidence_complete"]
    assert not result["strict_evidence_passed"]
    assert result["root_path_evidence_rows"] == 0
    assert result["root_path_evidence_missing_rows"] == 0
