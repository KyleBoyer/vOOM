from types import SimpleNamespace

import pytest

from runtime.free_token_cpu_probe import run


def test_probe_rejects_invalid_fraction_before_allocating(tmp_path):
    preflight = tmp_path / "preflight.json"
    # A current monotonic value is injected by using an intentionally invalid
    # fraction: input validation after freshness must fail before MLX allocation.
    import json
    import time

    preflight.write_text(json.dumps({
        "passed": True,
        "end": {"monotonic_s": time.monotonic()},
    }))
    args = SimpleNamespace(
        preflight=preflight, max_preflight_age=300.0,
        hidden=8, output_rows=8, repeats=1,
        cpu_fractions="0,1.5", seed=1,
    )
    with pytest.raises(ValueError, match="CPU fractions"):
        run(args)
