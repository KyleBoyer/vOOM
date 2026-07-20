from types import SimpleNamespace

import mlx.core as mx
import pytest

from runtime.engine import _gptoss_rope_state


def _config():
    return SimpleNamespace(
        head_dim=64,
        rope_theta=150_000.0,
        rope_scaling={
            "rope_type": "yarn",
            "factor": 32.0,
            "original_max_position_embeddings": 4096,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
        },
    )


def test_packed_gptoss_always_initializes_rope_frequencies():
    frequencies, scale = _gptoss_rope_state(_config(), packed=True)
    mx.eval(frequencies)
    assert frequencies.shape == (32,)
    assert bool(mx.all(mx.isfinite(frequencies)).item())
    assert scale > 1.0


def test_raw_gptoss_still_fails_with_actionable_pack_error():
    with pytest.raises(RuntimeError, match="requires a packed store"):
        _gptoss_rope_state(_config(), packed=False)
