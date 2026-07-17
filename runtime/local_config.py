"""Machine-local configuration: where model directories can be found besides
the repo's own ``models/`` folder, and how to recover a dropped mount.

Intentionally kept out of version control: this project runs standalone on a
base Apple Silicon Mac with no extra storage configured at all. Real values
live in a gitignored ``voom.local.yaml`` at the repo root (see
``voom.local.example.yaml`` for the shape); with no such file present, every
model path is just treated as an ordinary local path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "voom.local.yaml"


@dataclass(frozen=True)
class ModelStore:
    """One configured storage location (a NAS share, a second SSD, a USB
    drive, ...) that may appear under `volumes_root` and may need
    re-mounting after it drops (e.g. macOS cycling an SMB share's mountpoint
    name Plex -> Plex-1 -> Plex-2 ...)."""

    name: str  # the mountpoint's base name under volumes_root
    models_subdir: str = ""  # subdirectory under the mount where model dirs live
    remount_command: str = ""  # shell command that attempts to (re)mount it; "" = no-op

    def matches(self, path: str | Path) -> bool:
        parts = Path(path).parts
        return any(part == self.name or part.startswith(f"{self.name}-") for part in parts)

    def model_dir(self, mount: str | Path, model_name: str) -> Path:
        p = Path(mount)
        if self.models_subdir:
            p = p / self.models_subdir
        return p / model_name


@dataclass(frozen=True)
class StorageConfig:
    volumes_root: str = "/Volumes"
    stores: tuple[ModelStore, ...] = field(default_factory=tuple)

    def store_for_path(self, path: str | Path) -> ModelStore | None:
        for store in self.stores:
            if store.matches(path):
                return store
        return None

    def is_configured_path(self, path: str | Path) -> bool:
        return self.store_for_path(path) is not None

    def remount_command_for(self, path: str | Path) -> str:
        store = self.store_for_path(path)
        return store.remount_command if store else ""

    def resolve(self, model_dir, **kwargs):
        """`path_resolver.resolve_model_dir`, pre-filled with this config's
        details for whichever store (if any) `model_dir` belongs to."""
        from .path_resolver import resolve_model_dir

        store = self.store_for_path(model_dir)
        return resolve_model_dir(
            model_dir,
            volumes_root=self.volumes_root,
            store_name=store.name if store else "",
            models_subdir=store.models_subdir if store else "",
            remount_command=store.remount_command if store else "",
            **kwargs,
        )


_cached: StorageConfig | None = None


def get_storage_config() -> StorageConfig:
    """Load (and cache) the local storage config. Missing file => no extra
    storage locations configured, everything is treated as a local path."""
    global _cached
    if _cached is not None:
        return _cached
    if _CONFIG_PATH.exists():
        raw = yaml.safe_load(_CONFIG_PATH.read_text()) or {}
        stores = tuple(
            ModelStore(
                name=str(s.get("name", "")),
                models_subdir=str(s.get("models_subdir", "")),
                remount_command=str(s.get("remount_command", "")),
            )
            for s in (raw.get("model_stores") or [])
            if s.get("name")
        )
        _cached = StorageConfig(
            volumes_root=str(raw.get("volumes_root", "/Volumes")),
            stores=stores,
        )
    else:
        _cached = StorageConfig()
    return _cached
