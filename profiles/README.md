# Runtime profiles

Each YAML file is one named, composable group of existing `VMODEL_*` runtime
settings. Profiles add reproducibility and operator notes; they do not bypass
the runtime's existing validation or make an experimental setting automatic.

Use `python -m runtime.profiles list`, `show NAME`, or `validate` to inspect
the catalog. Start the server with `python -m runtime.server --profile NAME`.
Repeat `--profile` to layer groups; later groups win. A profile's own settings
win over its inherited parents, and explicit process environment variables win
over every profile.

Machine-specific or experimental profiles can live in the gitignored
`profiles.local/` directory. Additional directories may be supplied through
`VMODEL_PROFILE_DIR` (using the platform path separator) or repeated server
`--profile-dir PATH` arguments. Profile names must be unique across all search
directories so a run cannot silently resolve to a different file.

The schema is intentionally narrow:

```yaml
schema: voom.runtime-profile.v1
name: my-profile
description: A short purpose statement.
notes:
  - Correctness scope, hardware assumptions, and rollback details belong here.
extends:
  - an-existing-group
settings:
  VMODEL_SOME_FEATURE: true  # booleans become 1/0
  VMODEL_SOME_VALUE: 512
  VMODEL_SOME_MODE: "off"   # quote textual on/off/yes/no modes in YAML
```

Only scalar `VMODEL_*` settings are accepted. `VMODEL_PROFILE` and
`VMODEL_PROFILE_DIR` cannot be set recursively inside a profile. API responses
report the selected names, resolved group order, configured digest, effective
digest, and names of explicit environment overrides, but never setting values.
