"""Tiny identical-weight oracles for Qwen3.5/Qwen3.6 released text math."""

from __future__ import annotations

import json

import mlx.core as mx
import numpy as np
import torch

from runtime.config import ModelConfig
from runtime.kda_state import KDAFactorWindow, KDAStateCache
from runtime.kv_cache import Fp8KVCache, KVCache
from runtime.qwen35 import (
    _full_attention,
    _gated_delta_net,
    _moe,
    qwen35_rms_norm,
)
from runtime.qwen35_tree_verify import QwenTreeKVProxy
from runtime.speculative_tree import SpeculativeTree

from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
    Qwen3_5MoeVisionConfig,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeAttention,
    Qwen3_5MoeGatedDeltaNet,
    Qwen3_5MoeRMSNorm,
    Qwen3_5MoeSparseMoeBlock,
    Qwen3_5MoeTextRotaryEmbedding,
    Qwen3_5MoeVisionModel,
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
        rope_scaling={
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
            "mrope_section": [1, 1, 0],
            "mrope_interleaved": True,
        },
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


def test_gated_delta_net_exact_endpoint_continuation_matches_one_shot():
    config = _hf_config()
    real = Qwen3_5MoeGatedDeltaNet(config, layer_idx=0)
    _randomize(real, 31)
    with torch.no_grad():
        real.A_log.copy_(torch.log(
            torch.empty_like(real.A_log).uniform_(1.0, 8.0)))
    torch.manual_seed(32)
    hidden = torch.randn(1, LENGTH, HIDDEN)
    prefix = "model.layers.0"
    weights = _mx_state(real, f"{prefix}.linear_attn")

    full_cache = KDAStateCache(2)
    full = _gated_delta_net(
        mx.array(hidden.numpy()), weights, prefix, _runtime_config(),
        full_cache, 0)

    compiled_cache = KDAStateCache(2)
    compiled = _gated_delta_net(
        mx.array(hidden.numpy()), weights, prefix, _runtime_config(),
        compiled_cache, 0, compiled_delta_prefill=True)

    split_cache = KDAStateCache(2)
    left = _gated_delta_net(
        mx.array(hidden[:, :4].numpy()), weights, prefix, _runtime_config(),
        split_cache, 0, compiled_delta_prefill=True)
    retained = split_cache.fork()
    right = _gated_delta_net(
        mx.array(hidden[:, 4:].numpy()), weights, prefix, _runtime_config(),
        retained, 0, compiled_delta_prefill=True)
    split = mx.concatenate((left, right), axis=1)
    mx.eval(
        split, compiled, full,
        retained.state(0), compiled_cache.state(0), full_cache.state(0))
    assert np.array_equal(np.array(compiled), np.array(full))
    assert np.array_equal(np.array(split), np.array(full))
    assert bool(mx.array_equal(compiled_cache.state(0), full_cache.state(0)))
    assert bool(mx.array_equal(retained.state(0), full_cache.state(0)))
    assert retained.nbytes() > 0


def test_qwen_scalar_delta_factors_commit_tree_path_array_equal():
    config = _hf_config()
    real = Qwen3_5MoeGatedDeltaNet(config, layer_idx=0)
    _randomize(real, 33)
    with torch.no_grad():
        real.A_log.copy_(torch.log(
            torch.empty_like(real.A_log).uniform_(1.0, 8.0)))
    torch.manual_seed(34)
    hidden = torch.randn(1, 4, HIDDEN)
    prefix = "model.layers.0"
    weights = _mx_state(real, f"{prefix}.linear_attn")
    runtime = _runtime_config()

    # Capture the selected path exactly, then insert an unrelated sibling
    # factor between its final two nodes. Tree factors include causal-conv
    # history, so the final selected factor must itself have been evaluated
    # from the selected parent rather than from the sibling.
    selected_capture = KDAStateCache(2)
    selected_capture.begin_factor_capture()
    for position in (0, 1, 3):
        _gated_delta_net(
            mx.array(hidden[:, position:position + 1].numpy()),
            weights, prefix, runtime, selected_capture, 0)
    selected = selected_capture.finish_factor_capture(3)
    assert selected is not None

    sibling_capture = KDAStateCache(2)
    sibling_capture.begin_factor_capture()
    _gated_delta_net(
        mx.array(hidden[:, 2:3].numpy()),
        weights, prefix, runtime, sibling_capture, 0)
    sibling = sibling_capture.finish_factor_capture(1)
    assert sibling is not None
    factors = KDAFactorWindow([
        [selected.steps[0][0], selected.steps[0][1],
         sibling.steps[0][0], selected.steps[0][2]],
        [],
    ], positions=4)
    committed = factors.commit_indices(KDAStateCache(2), (0, 1, 3))

    assert bool(mx.array_equal(
        committed.state(0), selected_capture.state(0)))
    actual_history = committed.conv_history(0)[0]
    expected_history = selected_capture.conv_history(0)[0]
    assert bool(mx.array_equal(actual_history, expected_history))


def test_qwen_tree_attention_nodes_and_commit_match_sequential_paths():
    config = _hf_config()
    real = Qwen3_5MoeAttention(config, layer_idx=1)
    _randomize(real, 35)
    torch.manual_seed(36)
    hidden = torch.randn(1, 6, HIDDEN)
    prefix = "model.layers.1"
    weights = _mx_state(real, f"{prefix}.self_attn")
    runtime = _runtime_config()

    base = KVCache(2)
    _full_attention(
        mx.array(hidden[:, :2].numpy()), weights, prefix, runtime,
        base, 1, 0)
    tree = SpeculativeTree(
        token_ids=(1, 2, 3, 4),
        depths=(0, 1, 1, 2),
        parents=(-1, 0, 0, 1),
        children=({2: 1, 3: 2}, {4: 3}, {}, {}),
    )
    proxy = QwenTreeKVProxy(base, tree, 2)
    tree_outputs = []
    for node in range(4):
        proxy.current_node = node
        tree_outputs.append(_full_attention(
            mx.array(hidden[:, 2 + node:3 + node].numpy()),
            weights, prefix, runtime, proxy, 1,
            base.offset + tree.depths[node],
        ))

    for node, actual in enumerate(tree_outputs):
        reference = KVCache(2)
        reference.keys = list(base.keys)
        reference.values = list(base.values)
        expected = None
        for relative, index in enumerate(tree.path(node)):
            expected = _full_attention(
                mx.array(hidden[:, 2 + index:3 + index].numpy()),
                weights, prefix, runtime, reference, 1,
                base.offset + relative,
            )
        mx.eval(actual, expected)
        assert bool(mx.array_equal(actual, expected))

    committed = KVCache(2)
    committed.keys = list(base.keys)
    committed.values = list(base.values)
    proxy.commit_attention_path((0, 1, 3), committed)
    sequential = KVCache(2)
    sequential.keys = list(base.keys)
    sequential.values = list(base.values)
    for relative, index in enumerate((0, 1, 3)):
        _full_attention(
            mx.array(hidden[:, 2 + index:3 + index].numpy()),
            weights, prefix, runtime, sequential, 1,
            base.offset + relative,
        )
    mx.eval(committed.keys[1], committed.values[1],
            sequential.keys[1], sequential.values[1])
    assert bool(mx.array_equal(committed.keys[1], sequential.keys[1]))
    assert bool(mx.array_equal(committed.values[1], sequential.values[1]))


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


def test_fp8_kv_cache_full_attention_precision_impact_is_bounded():
    """Fp8KVCache is explicitly lossy -- this does not assert it matches the
    real reference at the same 1e-3 tolerance as the lossless KVCache case
    above. It measures the ACTUAL precision cost (both against the real HF
    reference and, more directly, against vOOM's own full-precision output)
    and asserts it stays within a bounded, expected-for-fp8-e4m3 range,
    across an incremental multi-call sequence (prefill-then-decode shaped),
    not just one one-shot call."""
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
    rcfg = _runtime_config()

    # Incremental: a 4-token "prefill" call followed by 3 one-at-a-time
    # "decode" calls into the SAME cache, mirroring real production use --
    # a one-shot-only test would miss bugs in the incremental concatenate
    # path specifically.
    lossless_cache = KVCache(2)
    lossless_chunks = [
        _full_attention(mx.array(hidden[:, :4].numpy()), weights, prefix,
                         rcfg, lossless_cache, 1, 0)]
    for position in range(4, LENGTH):
        lossless_chunks.append(_full_attention(
            mx.array(hidden[:, position:position + 1].numpy()), weights,
            prefix, rcfg, lossless_cache, 1, position))
    lossless = mx.concatenate(lossless_chunks, axis=1)

    fp8_cache = Fp8KVCache(2)
    fp8_chunks = [
        _full_attention(mx.array(hidden[:, :4].numpy()), weights, prefix,
                         rcfg, fp8_cache, 1, 0)]
    for position in range(4, LENGTH):
        fp8_chunks.append(_full_attention(
            mx.array(hidden[:, position:position + 1].numpy()), weights,
            prefix, rcfg, fp8_cache, 1, position))
    fp8 = mx.concatenate(fp8_chunks, axis=1)
    mx.eval(lossless, fp8)

    lossless_np = np.array(lossless)
    fp8_np = np.array(fp8)
    expected_np = expected.detach().numpy()
    assert lossless_np.shape == fp8_np.shape == expected_np.shape

    vs_lossless = float(np.max(np.abs(fp8_np - lossless_np)))
    vs_reference = float(np.max(np.abs(fp8_np - expected_np)))
    lossless_vs_reference = float(np.max(np.abs(lossless_np - expected_np)))
    # e4m3 has 3 mantissa bits (~12.5% max relative step); bound generously
    # above that on this tiny random-weight fixture rather than pin an exact
    # number, but a real bug (wrong axis, wrong dtype, stale cache) would
    # blow through this by orders of magnitude, not stay within it.
    assert vs_lossless < 0.35, (
        f"fp8 cache diverged from vOOM's own full-precision output by "
        f"{vs_lossless} -- larger than e4m3 precision alone should explain")
    assert vs_reference < lossless_vs_reference + 0.35, (
        f"fp8 cache's gap to the real reference ({vs_reference}) grew far "
        f"beyond the lossless implementation's own baseline gap "
        f"({lossless_vs_reference})")


def test_multimodal_partial_interleaved_attention_matches_reference():
    config = _hf_config()
    real = Qwen3_5MoeAttention(config, layer_idx=1)
    _randomize(real, 51)
    torch.manual_seed(52)
    hidden = torch.randn(1, LENGTH, HIDDEN)
    positions3 = torch.stack((
        torch.arange(LENGTH),
        torch.tensor([0, 1, 2, 2, 3, 4, 5]),
        torch.tensor([0, 1, 1, 2, 3, 4, 5]),
    ))[:, None, :]
    rope = Qwen3_5MoeTextRotaryEmbedding(config)
    embeddings = rope(hidden, positions3)
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
        KVCache(2), 1, 0, positions3[:, 0].numpy())
    _assert_close(actual, expected, tolerance=1e-3)


def test_qwen36_vision_tower_matches_official_reference():
    from runtime.qwen3vl import vision_forward

    config = Qwen3_5MoeVisionConfig(
        depth=2,
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        in_channels=3,
        patch_size=4,
        temporal_patch_size=2,
        spatial_merge_size=2,
        num_position_embeddings=16,
        out_hidden_size=32,
        deepstack_visual_indexes=[],
    )
    config._attn_implementation = "eager"
    real = Qwen3_5MoeVisionModel(config).to(torch.bfloat16).eval()
    _randomize(real, 61)
    real = real.to(torch.bfloat16)
    torch.manual_seed(62)
    pixels = torch.randn(16, 3 * 2 * 4 * 4, dtype=torch.bfloat16)
    grid = torch.tensor([[1, 4, 4]], dtype=torch.long)
    with torch.no_grad():
        expected = real(pixels, grid_thw=grid).pooler_output.float()
    weights = {
        f"model.visual.{name}": mx.array(value.float().detach().numpy()).astype(
            mx.bfloat16)
        for name, value in real.state_dict().items()
    }
    actual, deepstack = vision_forward(
        None, pixels.float().numpy(), 4, 4, {
            "depth": 2,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "patch_size": 4,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "num_position_embeddings": 16,
            "out_hidden_size": 32,
            "deepstack_visual_indexes": [],
        }, weights=weights)
    assert deepstack == []
    mx.eval(actual)
    actual_np = np.array(actual.astype(mx.float32))
    expected_np = expected.numpy()
    max_abs = float(np.max(np.abs(actual_np - expected_np)))
    cosine = float(
        np.dot(actual_np.ravel(), expected_np.ravel())
        / (np.linalg.norm(actual_np) * np.linalg.norm(expected_np)))
    assert max_abs < 6e-2
    assert cosine > 0.9999


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
    assert parsed.rope_scaling == {"rope_theta": 10000.0}
