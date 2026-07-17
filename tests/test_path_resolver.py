#!/usr/bin/env python3
"""Pure SMB mountpoint resolution tests; imports no ML framework."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.path_resolver import find_healthy_model_dir, resolve_model_dir


def _valid_config() -> dict:
    return {
        "model_type": "glm_moe_dsa",
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "vocab_size": 32,
    }


def _model(root: Path, mount: str, config: str | dict) -> Path:
    path = root / mount / "vmodel-models" / "GLM-5.2"
    path.mkdir(parents=True)
    (path / "config.json").write_text(
        config if isinstance(config, str) else json.dumps(config)
    )
    return path


def test_rejects_existing_but_corrupt_old_mount() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        old = _model(root, "Plex", "{")
        new = _model(root, "Plex-1", _valid_config())
        # find_healthy gets a synthetic /Volumes-like root; construct a path whose
        # suffix identifies the model, then force the candidate scan directly.
        found = find_healthy_model_dir(
            old, volumes_root=root, store_name="Plex", models_subdir="vmodel-models")
        assert found == new


def test_rejects_parseable_but_incomplete_old_mount() -> None:
    """A partial SMB metadata read may close braces and still parse as JSON."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        old = _model(root, "Plex", {"model_type": "glm_moe_dsa"})
        new = _model(root, "Plex-1", _valid_config())
        found = find_healthy_model_dir(
            old, volumes_root=root, store_name="Plex", models_subdir="vmodel-models")
        assert found == new


def test_resolver_retries_then_finds_new_mount() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        requested = root / "Plex" / "vmodel-models" / "GLM-5.2"
        calls = []

        def remount():
            calls.append(1)
            _model(root, "Plex-2", _valid_config())

        found = resolve_model_dir(
            requested, attempts=2, remount=remount, sleep=lambda _: None,
            volumes_root=root, store_name="Plex", models_subdir="vmodel-models",
        )
        assert found.parent.parent.name == "Plex-2"
        assert len(calls) == 1


def test_local_path_is_returned_without_remount() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "local-model"
        called = []
        found = resolve_model_dir(
            path, remount=lambda: called.append(1), sleep=lambda _: None,
            store_name="Plex", models_subdir="vmodel-models",
        )
        assert found == path
        assert not called


def test_unconfigured_store_name_never_matches() -> None:
    """With no store_name configured at all (the out-of-the-box default for
    anyone without a voom.local.yaml), every path is treated as local."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "Plex" / "vmodel-models" / "GLM-5.2"
        called = []
        found = resolve_model_dir(path, remount=lambda: called.append(1), sleep=lambda _: None)
        assert found == path
        assert not called


def _run_all() -> None:
    tests = [
        test_rejects_existing_but_corrupt_old_mount,
        test_rejects_parseable_but_incomplete_old_mount,
        test_resolver_retries_then_finds_new_mount,
        test_local_path_is_returned_without_remount,
        test_unconfigured_store_name_never_matches,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    assert "mlx" not in sys.modules and "torch" not in sys.modules
    print(f"PASS {len(tests)}/{len(tests)}; no MLX/Torch import")


if __name__ == "__main__":
    try:
        _run_all()
    except Exception as exc:
        print(f"FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
