"""Small, dependency-free RoPE parameter helpers.

The runtime deliberately does not import ``mlx_lm``.  These equations mirror
MLX-LM's canonical ``YarnRoPE`` construction, but return ordinary Python values
so they can be exhaustively unit-tested without creating a Metal context.
"""

from __future__ import annotations

import math


def supported_qwen_rope_type(scaling: dict | None) -> str | None:
    """Return a supported Qwen RoPE type or fail closed on unknown math."""
    if not scaling:
        return None
    rope_type = scaling.get("type") or scaling.get("rope_type")
    if rope_type in ("default", "yarn"):
        return rope_type
    raise ValueError(
        f"unsupported Qwen2 rope_scaling type {rope_type!r}; "
        "refusing to serve it as released RoPE"
    )


def yarn_parameters(
    dims: int,
    base: float,
    factor: float,
    original_max_position_embeddings: int,
    *,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    mscale: float = 1.0,
    mscale_all_dim: float = 0.0,
) -> tuple[tuple[float, ...], float]:
    """Return ``(RoPE denominators, attention scale)`` for static YaRN.

    ``mlx.core.fast.rope(..., freqs=...)`` calls its argument *frequencies*,
    but the API actually expects the per-pair RoPE denominators
    ``base ** (2*i/dims)``.  Canonical YaRN linearly blends inverse frequency;
    expressed as denominators this is a harmonic, not linear, interpolation.
    """
    numeric = {
        "base": base, "factor": factor,
        "beta_fast": beta_fast, "beta_slow": beta_slow,
        "mscale": mscale, "mscale_all_dim": mscale_all_dim,
    }
    for name, value in numeric.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"YaRN {name} must be finite")
    if dims <= 0 or dims % 2:
        raise ValueError("YaRN dims must be a positive even integer")
    if base <= 1:
        raise ValueError("YaRN base must be greater than 1")
    if factor < 1:
        raise ValueError("YaRN factor must be at least 1")
    if original_max_position_embeddings <= 0:
        raise ValueError("YaRN original context must be positive")
    if beta_fast <= 0 or beta_slow <= 0:
        raise ValueError("YaRN beta values must be positive")

    def correction_dim(rotations: float) -> float:
        return (
            dims * math.log(
                original_max_position_embeddings / (rotations * 2 * math.pi)
            ) / (2 * math.log(base))
        )

    low = max(math.floor(correction_dim(beta_fast)), 0)
    high = min(math.ceil(correction_dim(beta_slow)), dims - 1)
    width = high - low
    if width == 0:
        width = 0.001

    denominators = []
    for pair in range(dims // 2):
        ramp = min(max((pair - low) / width, 0.0), 1.0)
        # MLX-LM calls (1-ramp) ``freq_mask``.  Algebraically this is:
        # factor*D / (factor*(1-ramp) + ramp), where D is the base denominator.
        base_denominator = base ** ((2 * pair) / dims)
        denominator = (
            factor * base_denominator
            / (factor * (1.0 - ramp) + ramp)
        )
        if not math.isfinite(denominator):
            raise ValueError("YaRN produced a non-finite RoPE denominator")
        denominators.append(denominator)

    def yarn_mscale(multiplier: float) -> float:
        if factor <= 1:
            return 1.0
        return 0.1 * multiplier * math.log(factor) + 1.0

    attention_scale = yarn_mscale(mscale) / yarn_mscale(mscale_all_dim)
    if not math.isfinite(attention_scale):
        raise ValueError("YaRN produced a non-finite attention scale")
    return tuple(denominators), attention_scale
