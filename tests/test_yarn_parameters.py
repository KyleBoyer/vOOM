"""Pure YaRN math checks; does not import MLX or load a model."""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from runtime.rope import supported_qwen_rope_type, yarn_parameters


def test_factor_one_is_exact_base_rope():
    freqs, scale = yarn_parameters(128, 1_000_000.0, 1.0, 32_768)
    expected = tuple(1_000_000.0 ** (i / 128) for i in range(0, 128, 2))
    assert freqs == expected
    assert scale == 1.0


def test_yarn_endpoints_and_harmonic_midpoint():
    dims, base, factor, original = 128, 1_000_000.0, 2.0, 32_768
    freqs, scale = yarn_parameters(dims, base, factor, original)

    def correction(rotations):
        return dims * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(base))

    low = max(math.floor(correction(32.0)), 0)
    high = min(math.ceil(correction(1.0)), dims - 1)
    assert freqs[0] == 1.0  # high-frequency endpoint is extrapolated unchanged
    assert math.isclose(freqs[-1], factor * base ** (126 / dims), rel_tol=1e-15)

    pair = (low + high) // 2
    ramp = min(max((pair - low) / (high - low), 0.0), 1.0)
    base_denom = base ** ((2 * pair) / dims)
    expected = factor * base_denom / (factor * (1 - ramp) + ramp)
    assert math.isclose(freqs[pair], expected, rel_tol=1e-15)
    # Catch the tempting but wrong linear-denominator interpolation.
    wrong = base_denom * (1 - ramp) + factor * base_denom * ramp
    assert not math.isclose(freqs[pair], wrong, rel_tol=1e-3)
    assert math.isclose(scale, 0.1 * math.log(2.0) + 1.0, rel_tol=1e-15)


def test_yarn_rejects_invalid_profiles():
    bad = [
        (127, 1_000_000.0, 2.0, 32_768),
        (128, 1.0, 2.0, 32_768),
        (128, 1_000_000.0, 0.5, 32_768),
        (128, 1_000_000.0, 2.0, 0),
        (128, float("nan"), 2.0, 32_768),
        (128, 1_000_000.0, float("inf"), 32_768),
    ]
    for args in bad:
        try:
            yarn_parameters(*args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid YaRN profile accepted: {args}")


def test_qwen_rope_profile_fails_closed_on_unimplemented_scaling():
    assert supported_qwen_rope_type(None) is None
    assert supported_qwen_rope_type({"rope_type": "default"}) == "default"
    assert supported_qwen_rope_type({"type": "yarn", "factor": 2}) == "yarn"
    for scaling in ({"type": "linear", "factor": 2}, {"factor": 2}):
        try:
            supported_qwen_rope_type(scaling)
        except ValueError as error:
            assert "unsupported Qwen2" in str(error)
        else:
            raise AssertionError(f"unsupported scaling was accepted: {scaling}")


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  {test.__name__}: PASS")
    print(f"\n{len(tests)}/{len(tests)} tests passed")


if __name__ == "__main__":
    _run_all()
