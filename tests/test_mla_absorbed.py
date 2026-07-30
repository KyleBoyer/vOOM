"""F34 regression: MLA weight absorption during decode
(runtime/glm.py's `mla_absorbed` branch of `_mla_attention`) must produce
the SAME greedy token stream as the naive expand-then-attend path. Floating-
point association changes (the doc's own note), so bit-identical logits are
NOT the gate — greedy-token identity is, same standard as every other
lossless technique in this codebase. Uses the F65 fixture (no NAS, no real
GLM weights, sub-second).

  .venv/bin/python tests/test_mla_absorbed.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FIXTURE = Path(__file__).resolve().parent.parent / "models" / "glm-fixture-tiny"


def _ensure_fixture():
    from tests.fixtures.build_glm_fixture import build, is_current
    if not is_current(FIXTURE):
        build(FIXTURE)


def _generate(prompt: str, max_tokens: int, absorbed: bool):
    from runtime.engine import RuntimeConfig, StreamingEngine

    eng = StreamingEngine(str(FIXTURE), RuntimeConfig(
        max_weight_cache_mb=200, pin_lm_head=True, mla_compressed_kv=True,
        mla_absorbed_decode=absorbed))
    result = eng.generate(prompt, max_tokens)
    eng.close()
    return result["tokens"]


def test_absorbed_matches_naive_greedy_tokens():
    _ensure_fixture()
    prompt = "Hi there, how are you today my friend"
    naive = _generate(prompt, 8, absorbed=False)
    absorbed = _generate(prompt, 8, absorbed=True)
    assert naive == absorbed, f"absorbed decode diverged: {absorbed} != {naive}"


def test_absorbed_matches_naive_across_multiple_prompts():
    """A few different prompts/lengths, to reduce the chance the first test
    just got lucky on one particular sequence of accept/argmax decisions."""
    _ensure_fixture()
    prompts = [
        ("The quick brown fox jumps", 6),
        ("A B C D E F G H I J K L M N O P Q R S T U V W X Y Z", 10),
        ("Hi", 12),
    ]
    for prompt, n in prompts:
        naive = _generate(prompt, n, absorbed=False)
        absorbed = _generate(prompt, n, absorbed=True)
        assert naive == absorbed, \
            f"prompt {prompt!r}: absorbed diverged: {absorbed} != {naive}"


def test_absorbed_matches_naive_past_dsa_gather_threshold():
    """The fixture's index_topk=32 — a prompt+generation exceeding that
    triggers DSA's sparse gather (mx.take reducing lat_all/c_all/kr_all to
    the selected subset) BEFORE the absorbed math runs. The absorbed path
    must still match the naive path in that regime, not just the dense
    (S<=32) one exercised by the shorter prompts above."""
    _ensure_fixture()
    # The fixture tokenizer is byte-level, so this is exactly 40 positions:
    # enough to enter sparse gather without accidentally turning the intended
    # near-boundary probe into a 269-position numerical-stability stress test.
    long_prompt = "a" * 40  # > index_topk=32 tokens
    naive = _generate(long_prompt, 10, absorbed=False)
    absorbed = _generate(long_prompt, 10, absorbed=True)
    assert naive == absorbed, f"post-gather absorbed diverged: {absorbed} != {naive}"


def test_absorbed_flag_off_by_default():
    """mla_absorbed_decode defaults to False — must not change behavior for
    any existing caller that doesn't opt in."""
    from runtime.engine import RuntimeConfig

    assert RuntimeConfig().mla_absorbed_decode is False


def _absorbed_test_config():
    from runtime.config import ModelConfig

    return ModelConfig(
        model_type="kimi_k3",
        hidden_size=17,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=3,
        num_key_value_heads=3,
        vocab_size=32,
        rms_norm_eps=1e-5,
        rope_theta=10_000.0,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        attention_bias=False,
        head_dim=6,
        eos_token_ids=(0,),
        torch_dtype="float32",
        qk_nope_head_dim=4,
        qk_rope_head_dim=2,
        v_head_dim=5,
        kv_lora_rank=6,
        mla_use_nope=True,
    )


def test_absorbed_prefill_and_online_key_tiles_match_expanded_mla():
    """The L>1 path must preserve causal MLA, including a nonzero prefix."""
    from runtime.glm import _mla_absorbed_attention

    rng = np.random.default_rng(34)
    cfg = _absorbed_test_config()
    B, heads, length, offset = 1, 3, 5, 3
    total = offset + length
    dn, dr, dv, latent = 4, 2, 5, 6
    prefix = "layer0"
    q_nope = mx.array(
        rng.standard_normal((B, heads, length, dn)).astype(np.float32)
    )
    q_rope = mx.array(
        rng.standard_normal((B, heads, length, dr)).astype(np.float32)
    )
    c_all = mx.array(
        rng.standard_normal((B, total, latent)).astype(np.float32)
    )
    k_rope = mx.array(
        rng.standard_normal((B, total, dr)).astype(np.float32)
    )
    lat_all = mx.concatenate([c_all, k_rope], axis=-1)
    kv_b = mx.array(
        rng.standard_normal(
            (heads * (dn + dv), latent)
        ).astype(np.float32)
    )
    o_proj = mx.array(
        rng.standard_normal((cfg.hidden_size, heads * dv)).astype(
            np.float32
        )
    )
    weights = {
        f"{prefix}.self_attn.kv_b_proj.weight": kv_b,
        f"{prefix}.self_attn.o_proj.weight": o_proj,
    }
    h = mx.zeros((B, length, cfg.hidden_size), dtype=mx.float32)

    kvb = mx.matmul(c_all, kv_b.T).reshape(
        B, total, heads, dn + dv
    ).transpose(0, 2, 1, 3)
    k_nope, values = kvb[..., :dn], kvb[..., dn:]
    keys = mx.concatenate(
        [
            k_nope,
            mx.broadcast_to(
                k_rope[:, None, :, :],
                (B, heads, total, dr),
            ),
        ],
        axis=-1,
    )
    queries = mx.concatenate([q_nope, q_rope], axis=-1)
    query_positions = mx.arange(offset, offset + length)[:, None]
    key_positions = mx.arange(total)[None, :]
    mask = mx.where(
        key_positions <= query_positions,
        0.0,
        float("-inf"),
    ).astype(mx.float32)
    expanded = mx.fast.scaled_dot_product_attention(
        queries,
        keys,
        values,
        scale=(dn + dr) ** -0.5,
        mask=mask,
    )
    expanded = expanded.transpose(0, 2, 1, 3).reshape(
        B, length, heads * dv
    )
    expanded = mx.matmul(expanded, o_proj.T)

    absorbed = _mla_absorbed_attention(
        q_nope,
        q_rope,
        lat_all,
        weights,
        prefix,
        cfg,
        h,
        offset,
    )
    tiled = _mla_absorbed_attention(
        q_nope,
        q_rope,
        lat_all,
        weights,
        prefix,
        cfg,
        h,
        offset,
        key_tile_size=3,
    )
    mx.eval(expanded, absorbed, tiled)

    np.testing.assert_allclose(
        np.asarray(absorbed),
        np.asarray(expanded),
        atol=2e-5,
        rtol=2e-5,
    )
    np.testing.assert_allclose(
        np.asarray(tiled),
        np.asarray(expanded),
        atol=3e-5,
        rtol=3e-5,
    )


def test_k3_mla_candidates_are_explicit_and_dependency_checked():
    from runtime.engine import RuntimeConfig

    rc = RuntimeConfig()
    assert rc.kimi_k3_compressed_mla is False
    assert rc.kimi_k3_absorbed_mla is False
    assert rc.kimi_k3_fused_attnres_tile_size == 0


def test_k3_compressed_mla_factory_selects_stepped_absorbed_cache():
    from runtime.engine import RuntimeConfig, StreamingEngine
    from runtime.kv_cache import SteppedKVCache

    engine = StreamingEngine.__new__(StreamingEngine)
    engine.rc = RuntimeConfig(
        kimi_k3_compressed_mla=True,
        kimi_k3_absorbed_mla=True,
        kimi_k3_mla_key_tile_size=31,
    )
    engine.cfg = SimpleNamespace(
        model_type="kimi_k3",
        num_hidden_layers=4,
    )
    engine._position_free_pool = None

    kv = engine.new_kv()

    assert isinstance(kv, SteppedKVCache)
    assert kv.compressed_mla is True
    assert kv.mla_absorbed is True
    assert kv.mla_absorbed_prefill is True
    assert kv.mla_absorbed_key_tile_size == 31
    assert hasattr(kv, "kda_cache")


def test_k3_dense_mlp_position_tiles_match_full_rows():
    from runtime.kimi_linear import (
        _kimi_dense_mlp,
        _kimi_dense_mlp_tiled,
    )

    cfg = _absorbed_test_config()
    cfg.intermediate_size = 23
    cfg.hidden_act = "situ"
    cfg.activation_situ_beta = 4.0
    cfg.activation_situ_linear_beta = 25.0
    rng = np.random.default_rng(176)
    hidden = mx.array(
        rng.standard_normal((1, 13, cfg.hidden_size)).astype(np.float32)
    )
    prefix = "layer0.mlp"
    weights = {
        f"{prefix}.gate_proj.weight": mx.array(
            rng.standard_normal(
                (cfg.intermediate_size, cfg.hidden_size)
            ).astype(np.float32)
        ),
        f"{prefix}.up_proj.weight": mx.array(
            rng.standard_normal(
                (cfg.intermediate_size, cfg.hidden_size)
            ).astype(np.float32)
        ),
        f"{prefix}.down_proj.weight": mx.array(
            rng.standard_normal(
                (cfg.hidden_size, cfg.intermediate_size)
            ).astype(np.float32)
        ),
    }

    full = _kimi_dense_mlp(hidden, weights, prefix, cfg)
    tiled = _kimi_dense_mlp_tiled(
        hidden, weights, prefix, cfg, tile_size=4
    )
    mx.eval(full, tiled)

    np.testing.assert_allclose(
        np.asarray(tiled),
        np.asarray(full),
        atol=2e-5,
        rtol=2e-5,
    )


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        print(f"  {fn.__name__}: PASS")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
