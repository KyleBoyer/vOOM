"""Small exact gates for GPT-OSS split and bounded layer-stationary math."""

from types import SimpleNamespace

import mlx.core as mx

from runtime.kv_cache import KVCache
import runtime.gptoss as gptoss


def _fake_mxfp4_linear(x, blocks, scales, bias):
    del scales
    return x @ blocks.T + bias


def _fixture():
    hidden = 8
    intermediate = 3
    experts = 4
    cfg = SimpleNamespace(
        rms_norm_eps=1e-5,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        layer_types=("full_attention",),
        sliding_window=4,
        num_experts_per_tok=2,
        swiglu_limit=7.0,
    )
    prefix = "model.layers.0"
    generator = mx.random.key(91)

    def normal(shape, scale=0.15):
        nonlocal generator
        generator, key = mx.random.split(generator)
        return (mx.random.normal(shape, key=key) * scale).astype(mx.float32)

    weights = {
        f"{prefix}.input_layernorm.weight": mx.ones((hidden,)),
        f"{prefix}.post_attention_layernorm.weight": mx.ones((hidden,)),
        f"{prefix}.self_attn.q_proj.weight": normal((hidden, hidden)),
        f"{prefix}.self_attn.q_proj.bias": normal((hidden,), 0.03),
        f"{prefix}.self_attn.k_proj.weight": normal((4, hidden)),
        f"{prefix}.self_attn.k_proj.bias": normal((4,), 0.03),
        f"{prefix}.self_attn.v_proj.weight": normal((4, hidden)),
        f"{prefix}.self_attn.v_proj.bias": normal((4,), 0.03),
        f"{prefix}.self_attn.o_proj.weight": normal((hidden, hidden)),
        f"{prefix}.self_attn.o_proj.bias": normal((hidden,), 0.03),
        f"{prefix}.self_attn.sinks": normal((2,), 0.03),
        f"{prefix}.mlp.router.weight": normal((experts, hidden)),
        f"{prefix}.mlp.router.bias": normal((experts,), 0.03),
    }
    pages = {}
    for expert in range(experts):
        p = f"{prefix}.mlp.experts.{expert}"
        pages[expert] = {
            f"{p}.gate_up_blocks": normal((2 * intermediate, hidden)),
            f"{p}.gate_up_scales": mx.ones((2 * intermediate, 1)),
            f"{p}.gate_up_bias": normal((2 * intermediate,), 0.03),
            f"{p}.down_blocks": normal((hidden, intermediate)),
            f"{p}.down_scales": mx.ones((hidden, 1)),
            f"{p}.down_bias": normal((hidden,), 0.03),
        }
    x = normal((1, 7, hidden))
    freqs = mx.array([1.0, 2.0], dtype=mx.float32)
    return cfg, prefix, weights, pages, x, freqs


def test_bounded_expert_batches_match_historical_union(monkeypatch):
    monkeypatch.setattr(gptoss, "_mxfp4_linear", _fake_mxfp4_linear)
    cfg, prefix, weights, pages, x, _freqs = _fixture()

    def get_experts(_layer, expert_ids, positions=None):
        assert positions is not None
        return {expert: pages[expert] for expert in expert_ids}

    batch_widths = []

    def iter_batches(_layer, expert_ids, positions=None):
        assert positions is not None
        for start in range(0, len(expert_ids), 2):
            batch = expert_ids[start:start + 2]
            batch_widths.append(len(batch))
            yield batch, {expert: pages[expert] for expert in batch}

    historical = gptoss._gptoss_mlp_residual(
        x, weights, prefix, cfg, 0, get_experts)
    bounded = gptoss._gptoss_mlp_residual(
        x, weights, prefix, cfg, 0, get_experts,
        iter_expert_batches=iter_batches)
    mx.eval(historical, bounded)

    assert mx.array_equal(historical, bounded).item()
    assert batch_widths and max(batch_widths) <= 2


def test_tiled_attention_then_full_moe_matches_monolithic_block(monkeypatch):
    monkeypatch.setattr(gptoss, "_mxfp4_linear", _fake_mxfp4_linear)
    cfg, prefix, weights, pages, x, freqs = _fixture()

    def get_experts(_layer, expert_ids, positions=None):
        return {expert: pages[expert] for expert in expert_ids}

    baseline_kv = KVCache(1)
    baseline_tiles = []
    for start in range(0, x.shape[1], 2):
        baseline_tiles.append(gptoss.run_gptoss_block(
            x[:, start:start + 2, :], weights, prefix, cfg,
            baseline_kv, 0, start, get_experts, freqs, 1.0))
    baseline = mx.concatenate(baseline_tiles, axis=1)

    tiled_kv = KVCache(1)
    tiles = []
    for start in range(0, x.shape[1], 2):
        tiles.append(gptoss._gptoss_attention_residual(
            x[:, start:start + 2, :], weights, prefix, cfg,
            tiled_kv, 0, start, freqs, 1.0))
    candidate = gptoss._gptoss_tiled_mlp_residual(
        tiles, weights, prefix, cfg, 0, get_experts)
    mx.eval(baseline, candidate)

    assert mx.array_equal(baseline, candidate).item()
    assert tiled_kv.offset == x.shape[1]
