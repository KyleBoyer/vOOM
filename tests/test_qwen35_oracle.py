"""Tiny identical-weight oracles for Qwen3.5/Qwen3.6 released text math."""

from __future__ import annotations

import json

import mlx.core as mx
import numpy as np
import torch

from runtime.config import ModelConfig
from runtime.kda_state import KDAStateCache
from runtime.kv_cache import KVCache
from runtime.qwen35 import (
    _full_attention,
    _gated_delta_net,
    _moe,
    qwen35_rms_norm,
)

from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeAttention,
    Qwen3_5MoeGatedDeltaNet,
    Qwen3_5MoeRMSNorm,
    Qwen3_5MoeSparseMoeBlock,
    Qwen3_5MoeTextRotaryEmbedding,
)


HIDDEN = 32
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 8
LINEAR_K_HEADS = 2
LINEAR_V_HEADS = 4
LINEAR_DIM = 8
EXPERTS = 6
TOP_K = 2
MOE_DIM = 12
LENGTH = 7


def _hf_config() -> Qwen3_5MoeTextConfig:
    config = Qwen3_5MoeTextConfig(
        vocab_size=64,
        hidden_size=HIDDEN,
        num_hidden_layers=2,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        layer_types=["linear_attention", "full_attention"],
        linear_conv_kernel_dim=4,
        linear_key_head_dim=LINEAR_DIM,
        linear_value_head_dim=LINEAR_DIM,
        linear_num_key_heads=LINEAR_K_HEADS,
        linear_num_value_heads=LINEAR_V_HEADS,
        moe_intermediate_size=MOE_DIM,
        shared_expert_intermediate_size=MOE_DIM,
        num_experts=EXPERTS,
        num_experts_per_tok=TOP_K,
        rms_norm_eps=1e-6,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
            "mrope_section": [1, 1, 0],
            "mrope_interleaved": True,
        },
        partial_rotary_factor=0.5,
        attention_bias=False,
    )
    config._attn_implementation = "eager"
    return config


def _runtime_config() -> ModelConfig:
    return ModelConfig(
        model_type="qwen3_5_moe",
        hidden_size=HIDDEN,
        intermediate_size=MOE_DIM,
        num_hidden_layers=2,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        vocab_size=64,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        max_position_embeddings=4096,
        tie_word_embeddings=False,
        attention_bias=False,
        head_dim=HEAD_DIM,
        eos_token_ids=(),
        torch_dtype="float32",
        num_experts=EXPERTS,
        num_experts_per_tok=TOP_K,
        moe_intermediate_size=MOE_DIM,
        layer_types=("linear_attention", "full_attention"),
        linear_num_key_heads=LINEAR_K_HEADS,
        linear_num_value_heads=LINEAR_V_HEADS,
        linear_key_head_dim=LINEAR_DIM,
        linear_value_head_dim=LINEAR_DIM,
        linear_conv_kernel_dim=4,
        shared_expert_intermediate_size=MOE_DIM,
        partial_rotary_factor=0.5,
        attn_output_gate=True,
    )


def _randomize(module: torch.nn.Module, seed: int) -> None:
    torch.manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(mean=0.0, std=0.25)


def _mx_state(module: torch.nn.Module, prefix: str) -> dict:
    return {
        f"{prefix}.{name}": mx.array(value.detach().numpy())
        for name, value in module.state_dict().items()
    }


def _assert_close(actual: mx.array, expected: torch.Tensor, tolerance=1e-4):
    mx.eval(actual)
    actual_np = np.array(actual)
    expected_np = expected.detach().numpy()
    assert actual_np.shape == expected_np.shape
    difference = float(np.max(np.abs(actual_np - expected_np)))
    assert difference < tolerance, f"oracle mismatch: max abs diff {difference}"


def test_zero_centered_rmsnorm_matches_reference():
    real = Qwen3_5MoeRMSNorm(HIDDEN, eps=1e-6)
    _randomize(real, 1)
    torch.manual_seed(2)
    hidden = torch.randn(1, LENGTH, HIDDEN)
    with torch.no_grad():
        expected = real(hidden)
    actual = qwen35_rms_norm(
        mx.array(hidden.numpy()),
        mx.array(real.weight.detach().numpy()),
        1e-6,
    )
    _assert_close(actual, expected, tolerance=2e-6)


def test_gated_delta_net_matches_reference():
    config = _hf_config()
    real = Qwen3_5MoeGatedDeltaNet(config, layer_idx=0)
    _randomize(real, 3)
    with torch.no_grad():
        real.A_log.copy_(torch.log(
            torch.empty_like(real.A_log).uniform_(1.0, 8.0)))
    torch.manual_seed(4)
    hidden = torch.randn(1, LENGTH, HIDDEN)
    with torch.no_grad():
        expected = real(hidden, cache_params=None, attention_mask=None)
    prefix = "model.layers.0"
    weights = _mx_state(real, f"{prefix}.linear_attn")
    actual = _gated_delta_net(
        mx.array(hidden.numpy()), weights, prefix, _runtime_config(),
        KDAStateCache(2), 0)
    _assert_close(actual, expected, tolerance=2e-4)


def test_gated_full_attention_matches_reference():
    config = _hf_config()
    real = Qwen3_5MoeAttention(config, layer_idx=1)
    _randomize(real, 5)
    torch.manual_seed(6)
    hidden = torch.randn(1, LENGTH, HIDDEN)
    positions = torch.arange(LENGTH).unsqueeze(0)
    rope = Qwen3_5MoeTextRotaryEmbedding(config)
    embeddings = rope(hidden, positions)
    causal = torch.where(
        torch.arange(LENGTH)[None, :] <= torch.arange(LENGTH)[:, None],
        0.0, float("-inf"),
    )[None, None, :, :]
    with torch.no_grad():
        expected, _ = real(
            hidden, position_embeddings=embeddings,
            attention_mask=causal, past_key_values=None)
    prefix = "model.layers.1"
    weights = _mx_state(real, f"{prefix}.self_attn")
    actual = _full_attention(
        mx.array(hidden.numpy()), weights, prefix, _runtime_config(),
        KVCache(2), 1, 0)
    _assert_close(actual, expected, tolerance=1e-3)


def test_routed_and_shared_moe_matches_reference():
    config = _hf_config()
    real = Qwen3_5MoeSparseMoeBlock(config)
    _randomize(real, 7)
    torch.manual_seed(8)
    hidden = torch.randn(1, LENGTH, HIDDEN)
    with torch.no_grad():
        expected = real(hidden)
    prefix = "model.layers.0"
    state = real.state_dict()
    weights = {
        f"{prefix}.mlp.{name}": mx.array(value.detach().numpy())
        for name, value in state.items()
        if not name.startswith("experts.")
    }
    experts = {}
    fused_gate_up = state["experts.gate_up_proj"]
    fused_down = state["experts.down_proj"]
    for expert in range(EXPERTS):
        gate, up = fused_gate_up[expert].chunk(2, dim=0)
        expert_prefix = f"{prefix}.mlp.experts.{expert}"
        experts[expert] = {
            f"{expert_prefix}.gate_proj.weight": mx.array(gate.numpy()),
            f"{expert_prefix}.up_proj.weight": mx.array(up.numpy()),
            f"{expert_prefix}.down_proj.weight": mx.array(
                fused_down[expert].numpy()),
        }

    def get_experts(_layer, ids, positions=None):
        return {expert: experts[expert] for expert in ids}

    actual = _moe(
        mx.array(hidden.numpy()), weights, prefix, _runtime_config(), 0,
        get_experts)
    _assert_close(actual, expected, tolerance=1e-3)


def test_released_wrapper_config_is_lifted(tmp_path):
    config = {
        "model_type": "qwen3_5_moe",
        "tie_word_embeddings": False,
        "image_token_id": 56,
        "video_token_id": 57,
        "vision_start_token_id": 53,
        "vision_end_token_id": 54,
        "vision_config": {"depth": 2},
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_size": 32,
            "moe_intermediate_size": 12,
            "shared_expert_intermediate_size": 12,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "vocab_size": 64,
            "eos_token_id": 44,
            "num_experts": 6,
            "num_experts_per_tok": 2,
            "layer_types": ["linear_attention", "full_attention"],
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_conv_kernel_dim": 4,
            "partial_rotary_factor": 0.5,
            "rope_parameters": {"rope_theta": 10000.0},
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    parsed = ModelConfig.from_dir(tmp_path)
    assert parsed.model_type == "qwen3_5_moe"
    assert parsed.intermediate_size == 12
    assert parsed.layer_types == ("linear_attention", "full_attention")
    assert parsed.linear_num_value_heads == 4
    assert parsed.shared_expert_intermediate_size == 12
    assert parsed.image_token_id == 56
    assert parsed.vision_config == {"depth": 2}
