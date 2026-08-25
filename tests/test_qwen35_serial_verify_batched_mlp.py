from types import SimpleNamespace

import mlx.core as mx
from runtime.quant import QTensor
from runtime.qwen35 import _qwen35_mlp_residual


def _mxfp4(weight: mx.array) -> QTensor:
    wq, scales = mx.quantize(
        weight.astype(mx.bfloat16),
        group_size=32,
        bits=4,
        mode="mxfp4",
    )
    return QTensor(wq, scales, None, 4, 32, "mxfp4")


def test_dense_mxfp4_mlp_batch_matches_concatenated_serial_rows():
    hidden_size = 128
    intermediate_size = 256
    width = 5
    prefix = "model.layers.0"
    hidden = mx.sin(
        mx.arange(width * hidden_size, dtype=mx.float32) * 0.013
    ).reshape(1, width, hidden_size).astype(mx.bfloat16)
    gate = mx.cos(
        mx.arange(intermediate_size * hidden_size, dtype=mx.float32) * 0.007
    ).reshape(intermediate_size, hidden_size)
    up = mx.sin(
        mx.arange(intermediate_size * hidden_size, dtype=mx.float32) * 0.011
    ).reshape(intermediate_size, hidden_size)
    down = mx.cos(
        mx.arange(hidden_size * intermediate_size, dtype=mx.float32) * 0.017
    ).reshape(hidden_size, intermediate_size)
    weights = {
        f"{prefix}.post_attention_layernorm.weight": mx.ones(
            (hidden_size,), dtype=mx.bfloat16),
        f"{prefix}.mlp.gate_proj.weight": _mxfp4(gate),
        f"{prefix}.mlp.up_proj.weight": _mxfp4(up),
        f"{prefix}.mlp.down_proj.weight": _mxfp4(down),
    }
    config = SimpleNamespace(num_experts=0, rms_norm_eps=1e-6)

    serial = mx.concatenate([
        _qwen35_mlp_residual(
            hidden[:, position:position + 1, :],
            weights,
            prefix,
            config,
            0,
            None,
        )
        for position in range(width)
    ], axis=1)
    batched = _qwen35_mlp_residual(
        hidden, weights, prefix, config, 0, None)
    mx.eval(serial, batched)

    assert bool(mx.array_equal(batched, serial))
