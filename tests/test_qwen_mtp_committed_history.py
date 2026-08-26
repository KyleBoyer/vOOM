"""Exact operator gates for native-Qwen committed MTP attention history."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx

from runtime import quant
from runtime.kv_cache import KVCache
from runtime.qwen35 import _full_attention
from runtime.qwen35_mtp import QwenMTPDrafter
from runtime.qwen35 import qwen35_rms_norm


def _matrix(rows: int, columns: int, offset: int) -> mx.array:
    values = mx.arange(offset, offset + rows * columns, dtype=mx.float32)
    return ((values.reshape(rows, columns) % 29) / 31.0 - 0.45).astype(
        mx.bfloat16)


def _fixture():
    hidden_size = 8
    head_dim = 4
    heads = 2
    kv_heads = 1
    cfg = SimpleNamespace(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        partial_rotary_factor=0.5,
        rope_theta=10_000.0,
        rope_scaling=None,
        rms_norm_eps=1e-6,
        num_experts=0,
        vocab_size=16,
    )
    prefix = "mtp.layers.0"
    weights = {
        "mtp.pre_fc_norm_embedding.weight": mx.zeros(
            (hidden_size,), dtype=mx.bfloat16),
        "mtp.pre_fc_norm_hidden.weight": mx.zeros(
            (hidden_size,), dtype=mx.bfloat16),
        "mtp.fc.weight": _matrix(hidden_size, hidden_size * 2, 1),
        f"{prefix}.input_layernorm.weight": mx.zeros(
            (hidden_size,), dtype=mx.bfloat16),
        f"{prefix}.self_attn.q_proj.weight": _matrix(
            heads * 2 * head_dim, hidden_size, 101),
        f"{prefix}.self_attn.k_proj.weight": _matrix(
            kv_heads * head_dim, hidden_size, 251),
        f"{prefix}.self_attn.v_proj.weight": _matrix(
            kv_heads * head_dim, hidden_size, 331),
        f"{prefix}.self_attn.o_proj.weight": _matrix(
            hidden_size, heads * head_dim, 401),
        f"{prefix}.self_attn.q_norm.weight": mx.zeros(
            (head_dim,), dtype=mx.bfloat16),
        f"{prefix}.self_attn.k_norm.weight": mx.zeros(
            (head_dim,), dtype=mx.bfloat16),
    }
    embedding_table = _matrix(cfg.vocab_size, hidden_size, 701)

    class Store:
        mtplx_mtp_sidecar = None

        @staticmethod
        def names_with_prefix(prefix_):
            return list(weights) if prefix_ == "mtp." else []

    engine = SimpleNamespace(
        cfg=cfg,
        store=Store(),
        cache=SimpleNamespace(get=lambda *_args, **_kwargs: weights),
        _embed=lambda token_ids: mx.take(
            embedding_table, mx.array(token_ids), axis=0)[None, :, :],
    )
    return engine, weights


def test_committed_history_kv_matches_full_attention_operator_exactly():
    engine, weights = _fixture()
    drafter = QwenMTPDrafter(engine)
    hidden = _matrix(5, engine.cfg.hidden_size, 997).reshape(
        1, 5, engine.cfg.hidden_size)
    tokens = [1, 3, 5, 7, 9]
    offset = 17

    candidate = KVCache(1)
    info = drafter.append_committed_history(
        hidden, tokens, candidate, offset, weights, tile_size=2)

    embedding = engine._embed(tokens)
    embedding = qwen35_rms_norm(
        embedding,
        weights["mtp.pre_fc_norm_embedding.weight"],
        engine.cfg.rms_norm_eps,
    )
    hidden_norm = qwen35_rms_norm(
        hidden,
        weights["mtp.pre_fc_norm_hidden.weight"],
        engine.cfg.rms_norm_eps,
    )
    fused = quant.matmul(
        mx.concatenate([embedding, hidden_norm], axis=-1),
        weights["mtp.fc.weight"],
    )
    attn_input = qwen35_rms_norm(
        fused,
        weights["mtp.layers.0.input_layernorm.weight"],
        engine.cfg.rms_norm_eps,
    )
    oracle = KVCache(1)
    oracle_output = _full_attention(
        attn_input,
        weights,
        "mtp.layers.0",
        engine.cfg,
        oracle,
        0,
        offset,
    )
    mx.eval(oracle_output, candidate.keys[0], candidate.values[0],
            oracle.keys[0], oracle.values[0])

    assert bool(mx.array_equal(candidate.keys[0], oracle.keys[0]).item())
    assert bool(mx.array_equal(candidate.values[0], oracle.values[0]).item())
    assert candidate.layer_lengths() == (5,)
    assert info["rows"] == 5
    assert info["tiles"] == 3
    assert info["kv_bytes"] == candidate.nbytes()


def test_committed_history_validation_fails_closed():
    engine, weights = _fixture()
    drafter = QwenMTPDrafter(engine)
    hidden = mx.zeros((1, 2, engine.cfg.hidden_size), dtype=mx.bfloat16)

    for kwargs, message in (
        ({"next_tokens": [1], "start_offset": 0}, "row count mismatch"),
        ({"next_tokens": [1, 2], "start_offset": -1}, "non-negative"),
        ({"next_tokens": [1, 2], "start_offset": 0, "tile_size": 0},
         "positive"),
    ):
        try:
            drafter.append_committed_history(
                hidden,
                kwargs["next_tokens"],
                KVCache(1),
                kwargs["start_offset"],
                weights,
                tile_size=kwargs.get("tile_size", 128),
            )
        except ValueError as error:
            assert message in str(error)
        else:
            raise AssertionError("invalid committed history was accepted")
