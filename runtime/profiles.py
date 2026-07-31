"""Named, composable runtime configuration profiles.

Profiles are deliberately a thin layer over vOOM's existing ``VMODEL_*``
configuration surface.  They provide saved values, inheritance, and notes
without creating a second set of runtime knobs.  Explicit process environment
values always win over profile values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, MutableMapping, Sequence

import yaml


ROOT = Path(__file__).resolve().parent.parent
PROFILE_SCHEMA = "voom.runtime-profile.v1"
DEFAULT_PROFILE_DIRS = (ROOT / "profiles", ROOT / "profiles.local")
_PROFILE_NAME = re.compile(r"[a-z0-9][a-z0-9._-]*\Z")
_SETTING_NAME = re.compile(r"VMODEL_[A-Z0-9_]+\Z")
_RESERVED_SETTINGS = frozenset(("VMODEL_PROFILE", "VMODEL_PROFILE_DIR"))
_PROFILE_KEYS = frozenset((
    "schema", "name", "description", "notes", "extends", "settings",
))


class RuntimeProfileError(ValueError):
    """A profile file or selection that an operator can correct."""


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    description: str
    notes: tuple[str, ...]
    extends: tuple[str, ...]
    settings: Mapping[str, str]
    source: Path


@dataclass(frozen=True)
class RuntimeProfileApplication:
    """The immutable identity of one applied profile selection."""

    selected: tuple[str, ...]
    resolution_order: tuple[str, ...]
    profile_digest: str
    effective_digest: str
    overridden_keys: tuple[str, ...]
    setting_keys: tuple[str, ...]


_active_application: RuntimeProfileApplication | None = None


def _string_setting(value, *, source: Path, key: str) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeProfileError(
            f"{source}: setting {key} must not be NaN or infinite")
    if isinstance(value, (str, int, float)) and not isinstance(value, complex):
        rendered = str(value)
        if "\x00" in rendered:
            raise RuntimeProfileError(
                f"{source}: setting {key} must not contain a NUL byte")
        return rendered
    raise RuntimeProfileError(
        f"{source}: setting {key} must be a string, number, or boolean")


def _string_list(value, *, source: Path, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, list):
        values = tuple(value)
    else:
        raise RuntimeProfileError(f"{source}: {field} must be a string or list")
    if any(not isinstance(item, str) or not item.strip() for item in values):
        raise RuntimeProfileError(
            f"{source}: every {field} entry must be a non-empty string")
    return tuple(item.strip() for item in values)


def load_runtime_profile(path: str | Path) -> RuntimeProfile:
    """Load and strictly validate one profile YAML file."""
    source = Path(path).resolve()
    try:
        raw = yaml.safe_load(source.read_text())
    except (OSError, yaml.YAMLError) as error:
        raise RuntimeProfileError(f"cannot read profile {source}: {error}") from error
    if not isinstance(raw, dict):
        raise RuntimeProfileError(f"{source}: profile must be a YAML mapping")
    unknown = sorted(set(raw) - _PROFILE_KEYS)
    if unknown:
        raise RuntimeProfileError(
            f"{source}: unknown profile fields: {', '.join(unknown)}")
    if raw.get("schema") != PROFILE_SCHEMA:
        raise RuntimeProfileError(
            f"{source}: schema must be exactly {PROFILE_SCHEMA!r}")

    name = raw.get("name")
    if not isinstance(name, str) or not _PROFILE_NAME.fullmatch(name):
        raise RuntimeProfileError(
            f"{source}: name must match {_PROFILE_NAME.pattern!r}")
    if source.stem != name:
        raise RuntimeProfileError(
            f"{source}: filename stem must match profile name {name!r}")
    description = raw.get("description")
    if not isinstance(description, str) or not description.strip():
        raise RuntimeProfileError(f"{source}: description must be a non-empty string")
    notes = _string_list(raw.get("notes"), source=source, field="notes")
    extends = _string_list(raw.get("extends"), source=source, field="extends")
    for parent in extends:
        if not _PROFILE_NAME.fullmatch(parent):
            raise RuntimeProfileError(
                f"{source}: invalid parent profile name {parent!r}")

    raw_settings = raw.get("settings", {})
    if not isinstance(raw_settings, dict):
        raise RuntimeProfileError(f"{source}: settings must be a mapping")
    settings: dict[str, str] = {}
    for key, value in raw_settings.items():
        if not isinstance(key, str) or not _SETTING_NAME.fullmatch(key):
            raise RuntimeProfileError(
                f"{source}: setting names must match {_SETTING_NAME.pattern!r}; "
                f"got {key!r}")
        if key in _RESERVED_SETTINGS:
            raise RuntimeProfileError(
                f"{source}: {key} controls profile discovery and cannot be "
                "set by a profile")
        settings[key] = _string_setting(value, source=source, key=key)
    if not settings and not extends:
        raise RuntimeProfileError(
            f"{source}: profile must define settings or extend another profile")
    return RuntimeProfile(
        name=name,
        description=description.strip(),
        notes=notes,
        extends=extends,
        settings=settings,
        source=source,
    )


def runtime_profile_dirs(
    extra_dirs: Sequence[str | Path] = (),
    *,
    environ: Mapping[str, str] | None = None,
) -> tuple[Path, ...]:
    """Return default, environment, then explicit profile search directories."""
    env = os.environ if environ is None else environ
    candidates: list[str | Path] = list(DEFAULT_PROFILE_DIRS)
    configured = env.get("VMODEL_PROFILE_DIR", "")
    if configured:
        candidates.extend(part for part in configured.split(os.pathsep) if part)
    candidates.extend(extra_dirs)
    result: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = Path(candidate).expanduser().resolve()
        if resolved not in seen:
            result.append(resolved)
            seen.add(resolved)
    return tuple(result)


def discover_runtime_profiles(
    search_dirs: Sequence[str | Path] | None = None,
) -> dict[str, RuntimeProfile]:
    """Discover profile YAML files, rejecting ambiguous duplicate names."""
    directories = (
        tuple(Path(path).expanduser().resolve() for path in search_dirs)
        if search_dirs is not None else DEFAULT_PROFILE_DIRS)
    catalog: dict[str, RuntimeProfile] = {}
    for directory in directories:
        if not directory.exists():
            continue
        if not directory.is_dir():
            raise RuntimeProfileError(
                f"runtime profile search path is not a directory: {directory}")
        paths = sorted((*directory.glob("*.yaml"), *directory.glob("*.yml")))
        for path in paths:
            profile = load_runtime_profile(path)
            previous = catalog.get(profile.name)
            if previous is not None:
                raise RuntimeProfileError(
                    f"duplicate runtime profile {profile.name!r}: "
                    f"{previous.source} and {profile.source}")
            catalog[profile.name] = profile
    return catalog


def parse_runtime_profile_names(value: str | Sequence[str] | None) -> tuple[str, ...]:
    """Parse comma-separated environment or repeated CLI profile selections."""
    if value is None:
        return ()
    values = (value,) if isinstance(value, str) else tuple(value)
    names = tuple(
        name.strip()
        for entry in values
        for name in entry.split(",")
        if name.strip()
    )
    for name in names:
        if not _PROFILE_NAME.fullmatch(name):
            raise RuntimeProfileError(f"invalid runtime profile name {name!r}")
    return names


def resolve_runtime_profiles(
    selected: Sequence[str],
    catalog: Mapping[str, RuntimeProfile],
) -> tuple[tuple[str, ...], dict[str, str]]:
    """Flatten inheritance and ordered selections into effective defaults."""
    order: list[str] = []
    complete: set[str] = set()
    visiting: list[str] = []

    def visit(name: str) -> None:
        profile = catalog.get(name)
        if profile is None:
            available = ", ".join(sorted(catalog)) or "<none>"
            raise RuntimeProfileError(
                f"unknown runtime profile {name!r}; available: {available}")
        if name in complete:
            return
        if name in visiting:
            start = visiting.index(name)
            cycle = visiting[start:] + [name]
            raise RuntimeProfileError(
                "runtime profile inheritance cycle: " + " -> ".join(cycle))
        visiting.append(name)
        for parent in profile.extends:
            visit(parent)
        visiting.pop()
        complete.add(name)
        order.append(name)

    for name in selected:
        # A later top-level selection is a later layer even when it also
        # appeared as an earlier selection's ancestor. Deduplicate only
        # within one inheritance graph, not across ordered selections.
        complete.clear()
        visit(name)
    settings: dict[str, str] = {}
    for name in order:
        settings.update(catalog[name].settings)
    return tuple(order), settings


def _digest(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def apply_runtime_profiles(
    selected: Sequence[str],
    *,
    search_dirs: Sequence[str | Path] | None = None,
    environ: MutableMapping[str, str] | None = None,
    activate: bool = False,
) -> RuntimeProfileApplication | None:
    """Apply profiles as defaults and optionally publish process telemetry.

    Explicit entries already present in ``environ`` have the highest
    precedence.  Profiles selected later have precedence over earlier ones;
    child profiles have precedence over their parents.
    """
    global _active_application
    env = os.environ if environ is None else environ
    names = parse_runtime_profile_names(selected)
    if not names:
        if activate:
            _active_application = None
        return None
    directories = (
        runtime_profile_dirs(environ=env)
        if search_dirs is None else tuple(
            Path(path).expanduser().resolve() for path in search_dirs))
    catalog = discover_runtime_profiles(directories)
    order, configured = resolve_runtime_profiles(names, catalog)
    explicit_keys = set(env)
    overridden = tuple(sorted(key for key in configured if key in explicit_keys))
    for key, value in configured.items():
        if key not in explicit_keys:
            env[key] = value
    effective = {key: env[key] for key in sorted(configured)}
    identity = {
        "schema": PROFILE_SCHEMA,
        "selected": list(names),
        "resolution_order": list(order),
    }
    application = RuntimeProfileApplication(
        selected=names,
        resolution_order=order,
        profile_digest=_digest({**identity, "settings": configured}),
        effective_digest=_digest({**identity, "settings": effective}),
        overridden_keys=overridden,
        setting_keys=tuple(sorted(configured)),
    )
    if activate:
        _active_application = application
    return application


def clear_active_runtime_profiles() -> None:
    """Clear process profile telemetry (primarily for isolated tests)."""
    global _active_application
    _active_application = None


def active_runtime_profile_fields() -> dict[str, object]:
    """Return non-sensitive response fields for the active selection."""
    application = _active_application
    if application is None:
        return {}
    fields: dict[str, object] = {
        "vmodel_runtime_profiles": list(application.selected),
        "vmodel_runtime_profile_groups": list(application.resolution_order),
        "vmodel_runtime_profile_digest": application.profile_digest,
        "vmodel_runtime_effective_digest": application.effective_digest,
    }
    if application.overridden_keys:
        fields["vmodel_runtime_profile_overrides"] = list(
            application.overridden_keys)
    return fields


def _catalog_for_cli(extra_dirs: Sequence[str | Path]) -> dict[str, RuntimeProfile]:
    return discover_runtime_profiles(runtime_profile_dirs(extra_dirs))


def _print_catalog(catalog: Mapping[str, RuntimeProfile]) -> None:
    for name in sorted(catalog):
        profile = catalog[name]
        parents = f" (extends {', '.join(profile.extends)})" if profile.extends else ""
        print(f"{name}{parents}\n  {profile.description}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile-dir", action="append", default=[], metavar="PATH",
        help="add a profile search directory after the built-in/local directories",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="list discovered runtime profiles")
    show = commands.add_parser("show", help="show a profile and its resolved settings")
    show.add_argument("name")
    commands.add_parser("validate", help="validate every discovered profile")
    args = parser.parse_args(argv)
    try:
        catalog = _catalog_for_cli(args.profile_dir)
        if args.command == "list":
            _print_catalog(catalog)
        elif args.command == "validate":
            # Discovery already performs strict schema and duplicate checks;
            # resolving each entry additionally proves all parents and cycles.
            for name in catalog:
                resolve_runtime_profiles((name,), catalog)
            print(f"validated {len(catalog)} runtime profiles")
        else:
            names = parse_runtime_profile_names(args.name)
            order, settings = resolve_runtime_profiles(names, catalog)
            profile = catalog[names[0]]
            print(yaml.safe_dump({
                "schema": PROFILE_SCHEMA,
                "name": profile.name,
                "description": profile.description,
                "resolution_order": list(order),
                "groups": [{
                    "name": catalog[name].name,
                    "description": catalog[name].description,
                    "notes": list(catalog[name].notes),
                    "source": str(catalog[name].source),
                } for name in order],
                "resolved_settings": settings,
            }, sort_keys=False).rstrip())
    except RuntimeProfileError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
