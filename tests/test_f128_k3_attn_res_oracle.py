"""F128 numerical oracle: Kimi K3's "Attention Residuals" (AttnRes).

Kimi K3's real config.json sets `attn_res_block_size=12` (confirmed active
on the real downloaded checkpoint, not a no-op default) -- this is AttnRes
(arXiv 2603.15031, cited in Moonshot's own K3 release announcement),
which replaces the ordinary `x = x + sublayer_out` residual with a
softmax-attention-weighted readout over residual-stream snapshots taken
every `attn_res_block_size` layers. runtime.kimi_linear.attn_res_wrap_layer
(bookkeeping) + runtime.kimi_linear._apply_attn_res (the softmax readout
itself) implement this; this test validates both against verbatim
transcriptions of the real modeling_kimi_linear_k3.py source (module-level
`_apply_attn_res`, `KimiDecoderLayer._forward_attn_residual`,
`KimiLinearModel.forward`'s block_residual loop, and
`KimiLinearModel._apply_output_attn_res`) -- same methodology as
test_f92_kda_oracle.py: real reference source, tiny random weights, compare
MLX vs. torch on identical inputs.

attn_res_wrap_layer's `attn_fn`/`mlp_fn` are deliberately generic (see its
own docstring) so this test can use trivial deterministic stand-ins instead
of real KDA/MLA/MoE math -- that math is already independently
oracle-verified elsewhere (F92 for KDA/MLA, F93/this file's own
test_f128_k3_mxfp4_dequant.py-adjacent tests for MoE routing/experts), so
this isolates the genuinely new risk: getting AttnRes's own block-boundary
reset/snapshot bookkeeping right.
"""

from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx

from runtime.kimi_linear import _apply_attn_res, _situ_and_mul, attn_res_wrap_layer

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
_torch_skip = pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed in this venv")


# Verbatim from the real, downloaded modeling_kimi_linear.py (K3's own copy,
# checked 2026-07-27 hours after weights landed) -- module-level function,
# used here only as a numerical oracle, never imported into the runtime.
def _real_apply_attn_res(prefix_sum, block_residual, proj_weight, norm_weight, eps):
    """proj_weight: (1, hidden) -- nn.Linear(hidden, 1, bias=False).weight.
    norm_weight: (hidden,) -- KimiRMSNorm.weight."""
    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    v_float = v.float()
    variance = v_float.pow(2).mean(-1, keepdim=True)
    k = v_float * torch.rsqrt(variance + eps)
    score_weight = norm_weight.float() * proj_weight.squeeze(0).float()
    scores = (k * score_weight).sum(-1)
    probs = scores.softmax(-1).unsqueeze(1)
    hidden_states = torch.matmul(probs, v_float).squeeze(1)
    return hidden_states.to(v.dtype)


# Verbatim from the real modeling_kimi_linear_k3.py's SituAndMul.forward.
def _real_situ_and_mul(x, beta, linear_beta):
    d = x.shape[-1] // 2
    gate = x[..., :d].to(torch.float32)
    up = x[..., d:].to(torch.float32)
    situ_a = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (situ_a * up).to(x.dtype)


# Verbatim from the real KimiRMSNorm.forward.
def _real_rms_norm(x, weight, eps):
    dtype = x.dtype
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (weight * xf).to(dtype)


# Verbatim (control-flow only) from the real KimiDecoderLayer's
# _forward_attn_residual + KimiLinearModel.forward's block_residual loop +
# _apply_output_attn_res, INCLUDING the real input_layernorm/
# post_attention_layernorm calls (KimiRMSNorm, always present in the real
# code, not omittable). attn_fn/mlp_fn are the real self_attn/mlp forward
# calls in the reference; here they are the test's own trivial stand-ins,
# identical to what is fed to the MLX side.
def _real_attn_res_stack(
    hidden_states, num_layers, attn_res_block_size, res_weights, eps,
    attn_fn, mlp_fn, output_proj_weight, output_norm_weight,
):
    B, L, H = hidden_states.shape
    block_residual = hidden_states.new_zeros(B * L, 0, H)
    for layer_idx in range(num_layers):
        prefix_sum = hidden_states.reshape(-1, H)
        hidden = hidden_states
        if block_residual.shape[1] > 0:
            hidden = _real_apply_attn_res(
                prefix_sum, block_residual,
                res_weights[layer_idx]["self_proj"],
                res_weights[layer_idx]["self_norm"], eps,
            ).reshape(B, L, H)

        if layer_idx % attn_res_block_size == 0:
            block_residual = torch.cat(
                [block_residual, prefix_sum.unsqueeze(1)], dim=1)
            prefix_sum = None

        normed = _real_rms_norm(hidden, res_weights[layer_idx]["input_norm"], eps)
        attn_out = attn_fn(normed, layer_idx).reshape(-1, H)
        prefix_sum = (prefix_sum + attn_out) if prefix_sum is not None else attn_out

        hidden = _real_apply_attn_res(
            prefix_sum, block_residual,
            res_weights[layer_idx]["mlp_proj"],
            res_weights[layer_idx]["mlp_norm"], eps,
        ).reshape(B, L, H)

        normed2 = _real_rms_norm(hidden, res_weights[layer_idx]["post_norm"], eps)
        mlp_out = mlp_fn(normed2, layer_idx).reshape(-1, H)
        prefix_sum = prefix_sum + mlp_out
        hidden_states = prefix_sum.reshape(B, L, H)

    hidden_states = _real_apply_attn_res(
        hidden_states.reshape(-1, H), block_residual,
        output_proj_weight, output_norm_weight, eps,
    ).reshape(B, L, H)
    return hidden_states


@_torch_skip
def test_apply_attn_res_matches_real_reference():
    rng = np.random.default_rng(0)
    hidden, num_blocks, n_tokens = 8, 4, 5
    prefix_sum = rng.standard_normal((n_tokens, hidden)).astype(np.float32)
    block_residual = rng.standard_normal((n_tokens, num_blocks, hidden)).astype(np.float32)
    proj_weight = rng.standard_normal((1, hidden)).astype(np.float32)
    norm_weight = rng.standard_normal((hidden,)).astype(np.float32)
    eps = 1e-5

    ref = _real_apply_attn_res(
        torch.from_numpy(prefix_sum), torch.from_numpy(block_residual),
        torch.from_numpy(proj_weight), torch.from_numpy(norm_weight), eps,
    ).numpy()

    mine = _apply_attn_res(
        mx.array(prefix_sum), mx.array(block_residual),
        mx.array(proj_weight), mx.array(norm_weight), eps)
    mx.eval(mine)

    np.testing.assert_allclose(np.array(mine), ref, atol=1e-5, rtol=1e-5)


@_torch_skip
def test_apply_attn_res_matches_real_reference_zero_blocks_edge_case():
    """block_residual.shape[1] == 0 -- the ONLY call site with an actual
    empty block_residual (real code guards this call away entirely, but
    the function itself must still degenerate to a pure single-column
    softmax, i.e. probs==1, if ever called this way)."""
    rng = np.random.default_rng(1)
    hidden, n_tokens = 6, 3
    prefix_sum = rng.standard_normal((n_tokens, hidden)).astype(np.float32)
    block_residual = np.zeros((n_tokens, 0, hidden), dtype=np.float32)
    proj_weight = rng.standard_normal((1, hidden)).astype(np.float32)
    norm_weight = rng.standard_normal((hidden,)).astype(np.float32)
    eps = 1e-5

    ref = _real_apply_attn_res(
        torch.from_numpy(prefix_sum), torch.from_numpy(block_residual),
        torch.from_numpy(proj_weight), torch.from_numpy(norm_weight), eps,
    ).numpy()
    mine = _apply_attn_res(
        mx.array(prefix_sum), mx.array(block_residual),
        mx.array(proj_weight), mx.array(norm_weight), eps)
    mx.eval(mine)

    np.testing.assert_allclose(np.array(mine), prefix_sum, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(np.array(mine), ref, atol=1e-5, rtol=1e-5)


@_torch_skip
def test_situ_and_mul_matches_real_reference():
    rng = np.random.default_rng(2)
    n_tokens, d = 5, 8
    x = rng.standard_normal((n_tokens, 2 * d)).astype(np.float32)
    beta, linear_beta = 4.0, 25.0

    ref = _real_situ_and_mul(torch.from_numpy(x), beta, linear_beta).numpy()
    gate = mx.array(x[:, :d])
    up = mx.array(x[:, d:])
    mine = _situ_and_mul(gate, up, beta, linear_beta)
    mx.eval(mine)

    np.testing.assert_allclose(np.array(mine), ref, atol=1e-5, rtol=1e-5)


@_torch_skip
def test_situ_and_mul_matches_real_reference_no_linear_beta():
    """linear_beta unset (0.0 in this runtime's convention, None in the
    real config) -- `up` must pass through untransformed."""
    rng = np.random.default_rng(3)
    n_tokens, d = 4, 6
    x = rng.standard_normal((n_tokens, 2 * d)).astype(np.float32)
    beta = 1.0

    ref = _real_situ_and_mul(torch.from_numpy(x), beta, None).numpy()
    gate = mx.array(x[:, :d])
    up = mx.array(x[:, d:])
    mine = _situ_and_mul(gate, up, beta, 0.0)
    mx.eval(mine)

    np.testing.assert_allclose(np.array(mine), ref, atol=1e-5, rtol=1e-5)


@_torch_skip
def test_attn_res_block_bookkeeping_matches_real_forward_loop():
    """The genuinely new risk: block-boundary reset/snapshot bookkeeping
    across many layers, several block boundaries, and the final output
    readout -- isolated from real attention/MoE math via trivial
    deterministic stand-ins fed identically to both sides."""
    rng = np.random.default_rng(4)
    B, L, H = 1, 3, 8
    num_layers = 10
    attn_res_block_size = 3
    eps = 1e-5

    x_np = rng.standard_normal((B, L, H)).astype(np.float32)

    res_weights_np = [
        {
            "self_proj": rng.standard_normal((1, H)).astype(np.float32),
            "self_norm": rng.standard_normal((H,)).astype(np.float32),
            "mlp_proj": rng.standard_normal((1, H)).astype(np.float32),
            "mlp_norm": rng.standard_normal((H,)).astype(np.float32),
            "input_norm": rng.standard_normal((H,)).astype(np.float32),
            "post_norm": rng.standard_normal((H,)).astype(np.float32),
            "attn_w": rng.standard_normal((H, H)).astype(np.float32) * 0.1,
            "mlp_w": rng.standard_normal((H, H)).astype(np.float32) * 0.1,
        }
        for _ in range(num_layers)
    ]
    output_proj_np = rng.standard_normal((1, H)).astype(np.float32)
    output_norm_np = rng.standard_normal((H,)).astype(np.float32)

    def torch_attn_fn(h, layer_idx):
        return torch.tanh(h @ torch.from_numpy(res_weights_np[layer_idx]["attn_w"]))

    def torch_mlp_fn(h, layer_idx):
        return torch.sigmoid(h @ torch.from_numpy(res_weights_np[layer_idx]["mlp_w"]))

    res_weights_torch = [
        {k: torch.from_numpy(v) for k, v in layer.items()} for layer in res_weights_np
    ]
    ref = _real_attn_res_stack(
        torch.from_numpy(x_np), num_layers, attn_res_block_size,
        res_weights_torch, eps, torch_attn_fn, torch_mlp_fn,
        torch.from_numpy(output_proj_np), torch.from_numpy(output_norm_np),
    ).numpy()

    def mx_attn_fn_factory(layer_idx):
        w = mx.array(res_weights_np[layer_idx]["attn_w"])
        return lambda h: mx.tanh(h @ w)

    def mx_mlp_fn_factory(layer_idx):
        w = mx.array(res_weights_np[layer_idx]["mlp_w"])
        return lambda h: mx.sigmoid(h @ w)

    x = mx.array(x_np)
    block_residual = mx.zeros((B * L, 0, H))

    block_size = attn_res_block_size

    class _Cfg:
        rms_norm_eps = eps
        attn_res_block_size = block_size

    cfg = _Cfg()
    for layer_idx in range(num_layers):
        w = {
            f"layers.{layer_idx}.self_attention_res_proj.weight": mx.array(
                res_weights_np[layer_idx]["self_proj"]),
            f"layers.{layer_idx}.self_attention_res_norm.weight": mx.array(
                res_weights_np[layer_idx]["self_norm"]),
            f"layers.{layer_idx}.mlp_res_proj.weight": mx.array(
                res_weights_np[layer_idx]["mlp_proj"]),
            f"layers.{layer_idx}.mlp_res_norm.weight": mx.array(
                res_weights_np[layer_idx]["mlp_norm"]),
            f"layers.{layer_idx}.input_layernorm.weight": mx.array(
                res_weights_np[layer_idx]["input_norm"]),
            f"layers.{layer_idx}.post_attention_layernorm.weight": mx.array(
                res_weights_np[layer_idx]["post_norm"]),
        }
        x, block_residual = attn_res_wrap_layer(
            x, block_residual, w, f"layers.{layer_idx}", cfg, layer_idx,
            mx_attn_fn_factory(layer_idx), mx_mlp_fn_factory(layer_idx),
        )
        mx.eval(x, block_residual)

    from runtime.kimi_linear import apply_output_attn_res

    class _CfgOut:
        rms_norm_eps = eps

    out = apply_output_attn_res(
        x, {
            "model.output_attn_res_proj.weight": mx.array(output_proj_np),
            "model.output_attn_res_norm.weight": mx.array(output_norm_np),
        }, block_residual, _CfgOut(),
    )
    mx.eval(out)

    np.testing.assert_allclose(np.array(out), ref, atol=1e-4, rtol=1e-4)
