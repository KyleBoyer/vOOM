"""runtime/local_config.py: pure dataclass logic, plus file-loading behavior
with the module-level cache reset between cases so tests don't leak state."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import runtime.local_config as local_config
from runtime.local_config import ModelStore, StorageConfig, get_storage_config


def test_model_store_matches_base_and_cycled_names():
    store = ModelStore(name="Plex", models_subdir="vmodel-models")
    assert store.matches("/Volumes/Plex/vmodel-models/GLM-5.2")
    assert store.matches("/Volumes/Plex-1/vmodel-models/GLM-5.2")
    assert not store.matches("/Volumes/OtherShare/models/GLM-5.2")
    assert not store.matches("/local/models/GLM-5.2")


def test_model_store_model_dir_joins_subdir():
    store = ModelStore(name="Plex", models_subdir="vmodel-models")
    assert store.model_dir("/Volumes/Plex", "GLM-5.2") == Path("/Volumes/Plex/vmodel-models/GLM-5.2")
    bare = ModelStore(name="Plex")
    assert bare.model_dir("/Volumes/Plex", "GLM-5.2") == Path("/Volumes/Plex/GLM-5.2")


def test_storage_config_empty_treats_everything_as_local():
    cfg = StorageConfig()
    assert cfg.store_for_path("/Volumes/Plex/vmodel-models/GLM-5.2") is None
    assert not cfg.is_configured_path("/Volumes/Plex/vmodel-models/GLM-5.2")
    assert cfg.remount_command_for("/Volumes/Plex/vmodel-models/GLM-5.2") == ""


def test_storage_config_finds_matching_store():
    store = ModelStore(name="Plex", models_subdir="vmodel-models", remount_command="echo remount")
    cfg = StorageConfig(stores=(store,))
    found = cfg.store_for_path("/Volumes/Plex-2/vmodel-models/GLM-5.2")
    assert found is store
    assert cfg.is_configured_path("/Volumes/Plex-2/vmodel-models/GLM-5.2")
    assert cfg.remount_command_for("/Volumes/Plex-2/vmodel-models/GLM-5.2") == "echo remount"


def test_get_storage_config_defaults_when_no_file_present():
    local_config._cached = None
    old_path = local_config._CONFIG_PATH
    try:
        local_config._CONFIG_PATH = Path("/nonexistent/voom.local.yaml")
        cfg = get_storage_config()
        assert cfg.stores == ()
        assert cfg.volumes_root == "/Volumes"
    finally:
        local_config._CONFIG_PATH = old_path
        local_config._cached = None


def test_get_storage_config_parses_real_file():
    local_config._cached = None
    old_path = local_config._CONFIG_PATH
    try:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "voom.local.yaml"
            cfg_path.write_text(
                "volumes_root: /Volumes\n"
                "model_stores:\n"
                "  - name: TestShare\n"
                "    models_subdir: models\n"
                "    remount_command: echo hi\n"
            )
            local_config._CONFIG_PATH = cfg_path
            cfg = get_storage_config()
            assert cfg.volumes_root == "/Volumes"
            assert len(cfg.stores) == 1
            assert cfg.stores[0].name == "TestShare"
            assert cfg.stores[0].models_subdir == "models"
            assert cfg.stores[0].remount_command == "echo hi"
    finally:
        local_config._CONFIG_PATH = old_path
        local_config._cached = None


def _run_all() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"PASS {len(tests)}/{len(tests)}")


if __name__ == "__main__":
    try:
        _run_all()
    except Exception as exc:
        print(f"FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
