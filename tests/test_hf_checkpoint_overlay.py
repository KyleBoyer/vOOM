import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from runtime.hf_checkpoint_overlay import (
    PLAN_NAME,
    _assert_model_metadata,
    _hub_record,
    _safe_relative,
    build_plan,
    plan_command,
)


def _record(path: str, payload: bytes):
    return {
        "path": path,
        "size": len(payload),
        "hash_kind": "sha256",
        "hash": hashlib.sha256(payload).hexdigest(),
    }


def test_build_plan_links_identical_and_downloads_only_changed(tmp_path):
    shard_a = _record("model-00001-of-00002.safetensors", b"same")
    shard_b = _record("model-00002-of-00002.safetensors", b"base")
    changed_b = _record("model-00002-of-00002.safetensors", b"edited")
    config = _record("config.json", b"{}")
    candidate_only = _record("ABLIT_META.json", b"metadata")
    plan = build_plan(
        base_repo="base/repo", base_revision="a" * 40,
        base_dir=tmp_path / "base",
        base_records={item["path"]: item for item in (
            shard_a, shard_b, config)},
        candidate_repo="candidate/repo", candidate_revision="b" * 40,
        candidate_records={item["path"]: item for item in (
            shard_a, changed_b, config, candidate_only)},
        destination=tmp_path / "candidate",
    )
    assert plan["files"]["changed_safetensor_shards"] == 1
    assert [item["path"] for item in plan["files"]["link"]] == [
        "config.json", "model-00001-of-00002.safetensors"]
    assert [item["path"] for item in plan["files"]["download"]] == [
        "ABLIT_META.json", "model-00002-of-00002.safetensors"]


def test_build_plan_rejects_changed_shard_layout(tmp_path):
    with pytest.raises(ValueError, match="layout differs"):
        build_plan(
            base_repo="base/repo", base_revision="a" * 40,
            base_dir=tmp_path / "base",
            base_records={
                "a.safetensors": _record("a.safetensors", b"a")},
            candidate_repo="candidate/repo", candidate_revision="b" * 40,
            candidate_records={
                "b.safetensors": _record("b.safetensors", b"b")},
            destination=tmp_path / "candidate",
        )


@pytest.mark.parametrize("name", ["../x", "/tmp/x", "a/../x", "./x"])
def test_safe_relative_rejects_path_escape(name):
    with pytest.raises(ValueError, match="unsafe"):
        _safe_relative(name)


def test_hub_record_prefers_lfs_sha256():
    sibling = SimpleNamespace(
        rfilename="model.safetensors", size=9, blob_id="a" * 40,
        lfs=SimpleNamespace(sha256="b" * 64))
    assert _hub_record(sibling) == {
        "path": "model.safetensors", "size": 9,
        "hash_kind": "sha256", "hash": "b" * 64,
    }


def test_model_metadata_requires_identical_config_and_tensor_map(tmp_path):
    base = tmp_path / "base"
    candidate = tmp_path / "candidate"
    base.mkdir()
    candidate.mkdir()
    config = {"model_type": "qwen"}
    index = {"weight_map": {"a": "model-1.safetensors"}}
    for directory in (base, candidate):
        (directory / "config.json").write_text(json.dumps(config))
        (directory / "model.safetensors.index.json").write_text(
            json.dumps(index))
    assert _assert_model_metadata(base, candidate) == 1
    (candidate / "config.json").write_text(
        json.dumps({"model_type": "other"}))
    with pytest.raises(ValueError, match="config.json differs"):
        _assert_model_metadata(base, candidate)


def test_plan_name_is_hidden_from_model_loader():
    assert Path(PLAN_NAME).name.startswith(".")


@pytest.mark.parametrize("destination_kind", ["inside", "contains"])
def test_plan_rejects_destination_base_containment(tmp_path, destination_kind):
    base = tmp_path / "base"
    base.mkdir()
    destination = base / "candidate" if destination_kind == "inside" else tmp_path
    args = SimpleNamespace(base_dir=base, destination=destination)
    with pytest.raises(ValueError, match="must not contain"):
        plan_command(args)


def test_plan_rejects_nonempty_new_destination(tmp_path):
    base = tmp_path / "base"
    destination = tmp_path / "candidate"
    base.mkdir()
    destination.mkdir()
    (destination / "unrelated").write_text("preserve me")
    args = SimpleNamespace(base_dir=base, destination=destination)
    with pytest.raises(ValueError, match="must be empty"):
        plan_command(args)
