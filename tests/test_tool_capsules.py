from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from runtime import layer_runner
from runtime.kv_cache import KVCache, PositionFreeKVCache, PositionFreePagePool
from runtime.tool_capsules import (
    ToolCapsuleSpan,
    ToolPICPlan,
    build_pic_plan,
    prefill_with_tool_capsules,
)


def test_pic_plan_reuses_exact_prefix_and_repairs_capsule_boundaries():
    source = tuple(range(30))
    # The current prompt inserted four uncached tokens before capsule B and
    # shifted B/C. Their token bodies remain byte/token identical.
    current = source[:10] + (90, 91, 92, 93) + source[10:]
    source_spans = (
        ToolCapsuleSpan("a", 4, 10),
        ToolCapsuleSpan("b", 10, 18),
        ToolCapsuleSpan("c", 18, 26),
    )
    current_spans = (
        ToolCapsuleSpan("a", 4, 10),
        ToolCapsuleSpan("new", 10, 14),
        ToolCapsuleSpan("b", 14, 22),
        ToolCapsuleSpan("c", 22, 30),
    )

    plan = build_pic_plan(
        current, current_spans, source, source_spans,
        exact_prefix_tokens=8, repair_tokens=2)

    assert plan is not None
    assert [(value.start, value.end, value.source_start, value.kind)
            for value in plan.reused] == [
        (0, 8, 0, "exact_prefix"),
        (8, 10, 8, "tool_capsule"),
        (16, 22, 12, "tool_capsule"),
        (24, 30, 20, "tool_capsule"),
    ]
    assert plan.capsule_tokens_reused == 14
    assert plan.capsule_tokens_repaired == 4
    assert plan.selected_positions[-1] == len(current) - 1


def test_pic_plan_fails_closed_on_token_or_identity_mismatch():
    source = tuple(range(12))
    current = list(source)
    current[6] = 99
    assert build_pic_plan(
        current, [ToolCapsuleSpan("same", 4, 8)],
        source, [ToolCapsuleSpan("same", 4, 8)],
        repair_tokens=1) is None
    assert build_pic_plan(
        source, [ToolCapsuleSpan("new", 4, 8)],
        source, [ToolCapsuleSpan("old", 4, 8)],
        repair_tokens=1) is None


def test_pic_plan_reuses_survivors_across_reorder_add_and_remove():
    # The source has A/B/C.  The edited catalog reorders C ahead of A, removes
    # B, and adds D.  Matching is by content identity and exact token body, not
    # catalog ordinal, so the unchanged tails of C and A remain reusable.
    source = (0, 1, 10, 11, 12, 20, 21, 22, 30, 31, 32, 99)
    current = (0, 1, 30, 31, 32, 10, 11, 12, 40, 41, 42, 99)
    source_spans = (
        ToolCapsuleSpan("a", 2, 5),
        ToolCapsuleSpan("b", 5, 8),
        ToolCapsuleSpan("c", 8, 11),
    )
    current_spans = (
        ToolCapsuleSpan("c", 2, 5),
        ToolCapsuleSpan("a", 5, 8),
        ToolCapsuleSpan("d", 8, 11),
    )

    plan = build_pic_plan(
        current, current_spans, source, source_spans,
        exact_prefix_tokens=2, repair_tokens=1)

    assert plan is not None
    assert [(value.start, value.end, value.source_start, value.kind)
            for value in plan.reused] == [
        (0, 2, 0, "exact_prefix"),
        (3, 5, 9, "tool_capsule"),
        (6, 8, 3, "tool_capsule"),
    ]
    assert plan.capsule_tokens_reused == 4
    assert plan.capsule_tokens_repaired == 2
    assert plan.selected_positions == (2, 5, 8, 9, 10, 11)


class _Cache:
    def __init__(self, weights):
        self.weights = weights

    def get(self, _key, _names):
        return self.weights


class _TinyDenseEngine:
    def __init__(self):
        self.cfg = SimpleNamespace(
            num_experts=0,
            model_type="qwen2",
            vision_config=None,
            num_hidden_layers=1,
            hidden_size=8,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            rope_theta=10_000.0,
            rms_norm_eps=1e-6,
        )
        self.rc = SimpleNamespace(fused_swiglu=False)
        self._rope_freqs = None
        self._mscale = 1.0
        self.embedding = mx.arange(20 * 8, dtype=mx.float32).reshape(20, 8) / 100
        prefix = "model.layers.0"
        weights = {
            f"{prefix}.input_layernorm.weight": mx.ones((8,)),
            f"{prefix}.post_attention_layernorm.weight": mx.ones((8,)),
            f"{prefix}.self_attn.q_proj.weight": mx.arange(
                8 * 8, dtype=mx.float32).reshape(8, 8) / 200,
            f"{prefix}.self_attn.k_proj.weight": mx.arange(
                4 * 8, dtype=mx.float32).reshape(4, 8) / 170,
            f"{prefix}.self_attn.v_proj.weight": mx.arange(
                4 * 8, dtype=mx.float32).reshape(4, 8) / 190,
            f"{prefix}.self_attn.o_proj.weight": mx.arange(
                8 * 8, dtype=mx.float32).reshape(8, 8) / 210,
            f"{prefix}.mlp.gate_proj.weight": mx.arange(
                16 * 8, dtype=mx.float32).reshape(16, 8) / 230,
            f"{prefix}.mlp.up_proj.weight": mx.arange(
                16 * 8, dtype=mx.float32).reshape(16, 8) / 250,
            f"{prefix}.mlp.down_proj.weight": mx.arange(
                8 * 16, dtype=mx.float32).reshape(8, 16) / 270,
        }
        self.cache = _Cache(weights)
        self._norm_w = mx.ones((8,))
        self._head = mx.arange(20 * 8, dtype=mx.float32).reshape(20, 8) / 290

    def _embed(self, tokens):
        return self.embedding[mx.array(tokens)][None]

    @staticmethod
    def _layer_key(layer):
        return f"layer.{layer}"

    @staticmethod
    def _layer_names(_layer):
        return []

    def _lm_head_weight(self):
        return self._head


def test_selective_sweep_matches_normal_prefill_when_every_token_recomputed():
    engine = _TinyDenseEngine()
    tokens = [1, 2, 3, 4, 5]
    baseline_kv = KVCache(1)
    x = layer_runner.run_block(
        engine._embed(tokens), engine.cache.weights, "model.layers.0",
        engine.cfg, baseline_kv, 0, 0)
    baseline_logits = layer_runner.final_logits(
        x, engine._norm_w, engine._head, engine.cfg.rms_norm_eps)

    source = KVCache(1)
    source.keys[0] = mx.zeros((1, 1, len(tokens), 4))
    source.values[0] = mx.zeros((1, 1, len(tokens), 4))
    plan = ToolPICPlan(
        reused=(), selected_positions=tuple(range(len(tokens))),
        exact_prefix_tokens=0, capsule_tokens_reused=0,
        capsule_tokens_repaired=0)
    candidate_kv, candidate_logits = prefill_with_tool_capsules(
        engine, tokens, source, plan)
    mx.eval(baseline_logits, candidate_logits)

    assert mx.allclose(candidate_logits, baseline_logits, rtol=2e-5, atol=2e-5)
    assert mx.allclose(candidate_kv.keys[0], baseline_kv.keys[0],
                       rtol=2e-5, atol=2e-5)
    assert mx.allclose(candidate_kv.values[0], baseline_kv.values[0],
                       rtol=2e-5, atol=2e-5)


def test_shared_pic_matches_private_relocation_and_reuses_physical_pages():
    engine = _TinyDenseEngine()
    source_tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    current_tokens = [1, 2, 3, 4, 5, 11, 6, 7, 8, 9, 10]
    source_spans = (
        ToolCapsuleSpan("a", 1, 5),
        ToolCapsuleSpan("b", 5, 9),
    )
    current_spans = (
        ToolCapsuleSpan("a", 1, 5),
        ToolCapsuleSpan("new", 5, 6),
        ToolCapsuleSpan("b", 6, 10),
    )
    plan = build_pic_plan(
        current_tokens, current_spans, source_tokens, source_spans,
        repair_tokens=1)
    assert plan is not None

    private_source = KVCache(1)
    layer_runner.run_block(
        engine._embed(source_tokens), engine.cache.weights, "model.layers.0",
        engine.cfg, private_source, 0, 0)
    pool = PositionFreePagePool(1, 1, 4, min_capacity=16)
    shared_source = PositionFreeKVCache(pool)
    layer_runner.run_block(
        engine._embed(source_tokens), engine.cache.weights, "model.layers.0",
        engine.cfg, shared_source, 0, 0)

    private_kv, private_logits = prefill_with_tool_capsules(
        engine, current_tokens, private_source, plan)
    shared_kv, shared_logits = prefill_with_tool_capsules(
        engine, current_tokens, shared_source, plan)
    mx.eval(private_logits, shared_logits)

    assert mx.allclose(shared_logits, private_logits, rtol=2e-5, atol=2e-5)
    for reused in plan.reused:
        assert (
            shared_kv.page_ids[reused.start:reused.end]
            == shared_source.page_ids[
                reused.source_start:reused.source_start + reused.length]
        )
    # Source has ten pages; destination allocates only its five selected pages.
    assert pool.live_pages == len(source_tokens) + plan.selected_tokens
    shared_source.release()
    assert pool.live_pages == len(current_tokens)
    shared_kv.release()
    assert pool.live_pages == 0


class _TinyOLMoEEngine(_TinyDenseEngine):
    def __init__(self):
        super().__init__()
        self.cfg.model_type = "olmoe"
        self.cfg.num_experts = 2
        self.cfg.num_experts_per_tok = 2
        self.cfg.expert_top_k_by_layer = ()
        self.cfg.norm_topk_prob = True
        self._resident_moe_layers = None
        prefix = "model.layers.0"
        weights = self.cache.weights
        for name in (
                f"{prefix}.mlp.gate_proj.weight",
                f"{prefix}.mlp.up_proj.weight",
                f"{prefix}.mlp.down_proj.weight"):
            weights.pop(name)
        weights[f"{prefix}.self_attn.q_norm.weight"] = mx.ones((8,))
        weights[f"{prefix}.self_attn.k_norm.weight"] = mx.ones((4,))
        weights[f"{prefix}.mlp.gate.weight"] = mx.array([
            [0.2] * 8,
            [-0.15] * 8,
        ])
        self.experts = {}
        for expert in range(2):
            ep = f"{prefix}.mlp.experts.{expert}"
            scale = expert + 1
            self.experts[expert] = {
                f"{ep}.gate_proj.weight": mx.arange(
                    16 * 8, dtype=mx.float32).reshape(16, 8)
                    / (230 * scale),
                f"{ep}.up_proj.weight": mx.arange(
                    16 * 8, dtype=mx.float32).reshape(16, 8)
                    / (250 * scale),
                f"{ep}.down_proj.weight": mx.arange(
                    8 * 16, dtype=mx.float32).reshape(8, 16)
                    / (270 * scale),
            }

    def _get_experts(self, _layer, expert_ids, positions=None):
        return {expert: self.experts[expert] for expert in expert_ids}


def test_olmoe_selective_sweep_matches_normal_when_every_token_recomputed():
    engine = _TinyOLMoEEngine()
    engine.cfg.expert_top_k_by_layer = (1,)
    tokens = [1, 2, 3, 4, 5]
    baseline_kv = KVCache(1)
    x = layer_runner.run_moe_block(
        engine._embed(tokens), engine.cache.weights, "model.layers.0",
        engine.cfg, baseline_kv, 0, 0, engine._get_experts)
    baseline_logits = layer_runner.final_logits(
        x, engine._norm_w, engine._head, engine.cfg.rms_norm_eps)

    source = KVCache(1)
    source.keys[0] = mx.zeros((1, 1, len(tokens), 4))
    source.values[0] = mx.zeros((1, 1, len(tokens), 4))
    plan = ToolPICPlan(
        reused=(), selected_positions=tuple(range(len(tokens))),
        exact_prefix_tokens=0, capsule_tokens_reused=0,
        capsule_tokens_repaired=0)
    candidate_kv, candidate_logits = prefill_with_tool_capsules(
        engine, tokens, source, plan)
    mx.eval(baseline_logits, candidate_logits)

    assert mx.allclose(candidate_logits, baseline_logits, rtol=2e-5, atol=2e-5)
    assert mx.allclose(candidate_kv.keys[0], baseline_kv.keys[0],
                       rtol=2e-5, atol=2e-5)
    assert mx.allclose(candidate_kv.values[0], baseline_kv.values[0],
                       rtol=2e-5, atol=2e-5)

    prefix = "model.layers.0"
    fused = {
        projection: layer_runner.stack_expert_weights([
            engine.experts[expert][
                f"{prefix}.mlp.experts.{expert}.{projection}.weight"]
            for expert in range(engine.cfg.num_experts)
        ])
        for projection in ("gate_proj", "up_proj", "down_proj")
    }
    engine._resident_moe_layers = ((engine.cache.weights, fused),)
    resident_baseline_kv = KVCache(1)
    resident_x = layer_runner.run_fused_moe_block(
        engine._embed(tokens), engine.cache.weights, fused, prefix,
        engine.cfg, resident_baseline_kv, 0, 0, mlx_router_semantics=True)
    resident_baseline_logits = layer_runner.final_logits(
        resident_x, engine._norm_w, engine._head, engine.cfg.rms_norm_eps)
    resident_candidate_kv, resident_candidate_logits = prefill_with_tool_capsules(
        engine, tokens, source, plan)
    mx.eval(resident_baseline_logits, resident_candidate_logits)

    assert mx.allclose(
        resident_candidate_logits, resident_baseline_logits,
        rtol=2e-5, atol=2e-5)
    assert mx.allclose(
        resident_candidate_kv.keys[0], resident_baseline_kv.keys[0],
        rtol=2e-5, atol=2e-5)
    assert mx.allclose(
        resident_candidate_kv.values[0], resident_baseline_kv.values[0],
        rtol=2e-5, atol=2e-5)


def test_qwen3vl_selective_sweep_matches_full_mrope_and_deepstack_prefill():
    from runtime.qwen3vl import vl_prefill, vl_prefill_with_tool_capsules

    engine = _TinyDenseEngine()
    engine.cfg.model_type = "qwen3_vl"
    engine.cfg.vision_config = {"spatial_merge_size": 2}
    engine.cfg.image_token_id = 9
    engine.cfg.video_token_id = 0
    engine.cfg.rope_scaling = {"mrope_section": [2, 1, 1]}
    prefix = "model.layers.0"
    engine.cache.weights[f"{prefix}.self_attn.q_norm.weight"] = mx.ones((4,))
    engine.cache.weights[f"{prefix}.self_attn.k_norm.weight"] = mx.ones((4,))
    tokens = [1, 9, 9, 2, 3]
    pos3 = np.tile(np.arange(len(tokens), dtype=np.int64), (3, 1))
    image_embeds = mx.arange(16, dtype=mx.float32).reshape(2, 8) / 50
    deepstack = [mx.arange(16, dtype=mx.float32).reshape(2, 8) / 70]

    baseline_kv = KVCache(1)
    baseline_logits = vl_prefill(
        engine, tokens, image_embeds, deepstack, pos3, baseline_kv)
    source = KVCache(1)
    source.keys[0] = mx.zeros((1, 1, len(tokens), 4))
    source.values[0] = mx.zeros((1, 1, len(tokens), 4))
    plan = ToolPICPlan(
        reused=(), selected_positions=tuple(range(len(tokens))),
        exact_prefix_tokens=0, capsule_tokens_reused=0,
        capsule_tokens_repaired=0)

    candidate_kv, candidate_logits = vl_prefill_with_tool_capsules(
        engine, tokens, image_embeds, deepstack, pos3, source, pos3, plan)
    mx.eval(baseline_logits, candidate_logits)

    assert mx.allclose(candidate_logits, baseline_logits, rtol=2e-5, atol=2e-5)
    assert mx.allclose(candidate_kv.keys[0], baseline_kv.keys[0],
                       rtol=2e-5, atol=2e-5)
    assert mx.allclose(candidate_kv.values[0], baseline_kv.values[0],
                       rtol=2e-5, atol=2e-5)
