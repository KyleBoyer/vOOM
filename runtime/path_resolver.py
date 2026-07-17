"""Pure path health/re-resolution helpers for cycling mountpoints (e.g. macOS
re-mounting an SMB share under a new name: Plex -> Plex-1 -> Plex-2 ...).

Deliberately takes the storage-location details (`store_name`, `models_subdir`)
as plain parameters rather than reading any global config, so this module has
no hidden dependencies and its tests are fully self-contained. Real callers
look those values up via `runtime.local_config.get_storage_config()` and pass
them in explicitly.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable


def _matches_store(path: Path, store_name: str) -> bool:
    if not store_name:
        return False
    return any(
        part == store_name or part.startswith(f"{store_name}-")
        for part in Path(path).parts
    )


def _readable_config(model_dir: Path) -> bool:
    try:
        value = json.loads((model_dir / "config.json").read_text())
        if not isinstance(value, dict):
            return False
        # A transient/partial SMB read can still be syntactically valid JSON.
        # Do not select that mount merely because it parsed; require the fields
        # ModelConfig needs before this path is allowed to outrank a cycled mount.
        text = value.get("text_config", value)
        required = {
            "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "vocab_size",
        }
        return isinstance(text, dict) and required.issubset(text)
    except (OSError, ValueError, TypeError):
        return False


def find_healthy_model_dir(model_dir: str | Path, volumes_root: str | Path = "/Volumes",
                           *, store_name: str = "", models_subdir: str = "") -> Path | None:
    """Return the same model on a mount whose config can actually be parsed."""
    model_dir = Path(model_dir)
    candidates = [model_dir]
    if _matches_store(model_dir, store_name):
        candidates.extend(
            (mount / models_subdir if models_subdir else mount) / model_dir.name
            for mount in sorted(Path(volumes_root).glob(f"{store_name}*"))
        )
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if _readable_config(candidate):
            return candidate
    return None


def resolve_model_dir(model_dir: str | Path, attempts: int = 4, *,
                      remount: Callable[[], object] | None = None,
                      sleep: Callable[[float], None] = time.sleep,
                      volumes_root: str | Path = "/Volumes",
                      store_name: str = "", models_subdir: str = "",
                      remount_command: str = "") -> Path:
    """Resolve a readable model directory, remounting the configured storage
    location between attempts.

    Paths that don't match `store_name` are returned immediately (no retry).
    Paths that DO match get bounded retry/backoff, since a remount may cycle
    the mountpoint name (e.g. Plex -> Plex-1).
    """
    model_dir = Path(model_dir)
    if not _matches_store(model_dir, store_name):
        return model_dir
    if remount is None:
        remount = (lambda: os.system(remount_command)) if remount_command else (lambda: None)
    for attempt in range(attempts):
        healthy = find_healthy_model_dir(
            model_dir, volumes_root, store_name=store_name, models_subdir=models_subdir)
        if healthy is not None:
            return healthy
        if attempt == attempts - 1:
            break
        remount()
        sleep(5 * (2 ** attempt))
    raise OSError(
        f"no readable config for {model_dir.name} on any {store_name} mount "
        f"after {attempts} attempts"
    )
