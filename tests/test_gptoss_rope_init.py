from types import SimpleNamespace
import math

import mlx.core as mx
import numpy as np
import pytest

from runtime.engine import _gptoss_rope_state
from runtime.gptoss import yarn_params


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
            "truncate": False,
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


def _released_yarn_reference(cfg):
    """Independent inverse-frequency form used by the released HF model."""
    rs = cfg.rope_scaling
    dim = cfg.head_dim
    base = cfg.rope_theta
    factor = rs["factor"]
    pos_freqs = base ** (np.arange(0, dim, 2, dtype=np.float32) / dim)

    def correction_dim(num_rotations):
        return dim * math.log(
            rs["original_max_position_embeddings"]
            / (num_rotations * 2 * math.pi)
        ) / (2 * math.log(base))

    low = max(correction_dim(rs["beta_fast"]), 0)
    high = min(correction_dim(rs["beta_slow"]), dim - 1)
    ramp = np.clip((np.arange(dim // 2) - low) / (high - low), 0, 1)
    inv_extra = 1 / pos_freqs
    inv_inter = 1 / (factor * pos_freqs)
    inv_freq = inv_inter * ramp + inv_extra * (1 - ramp)
    return 1 / inv_freq


def test_released_gptoss_yarn_matches_untruncated_inverse_frequency_reference():
    cfg = _config()
    frequencies, _ = yarn_params(cfg)
    actual = np.asarray(frequencies)
    expected = _released_yarn_reference(cfg)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=0)

    # The transition has non-integer bounds for the released checkpoint, so
    # this also guards against silently reverting to floor/ceil truncation.
    truncated_cfg = _config()
    truncated_cfg.rope_scaling = dict(truncated_cfg.rope_scaling, truncate=True)
    truncated_frequencies, _ = yarn_params(truncated_cfg)
    assert not np.allclose(actual, np.asarray(truncated_frequencies), rtol=1e-6)
