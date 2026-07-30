"""Real Kimi K3 gate for compressed and weight-absorbed MLA.

This isolates the first released MLA layer (layer 3) on identical multi-token
hidden states.  It validates the architecture transformation itself without
mixing in MoE routing or a prompt-specific output expectation.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from runtime.kda_state import KDAStateCache
from runtime.kv_cache import KVCache, SteppedKVCache


ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "Kimi-K3"
_MODEL_AVAILABLE = (
    MODEL_DIR / "model.safetensors.index.json"
).exists()
_model_skip = pytest.mark.skipif(
    not _MODEL_AVAILABLE,
    reason="Kimi-K3 real checkpoint is unavailable",
)


def _stepped_mla_cache(num_layers: int, *, absorbed: bool):
    cache = SteppedKVCache(num_layers)
    cache.compressed_mla = True
    cache.mla_absorbed = absorbed
    cache.mla_absorbed_prefill = absorbed
    # Force multiple online-softmax tiles even in this tiny gate.
    cache.mla_absorbed_key_tile_size = 2
    cache.kda_cache = KDAStateCache(num_layers)
    return cache


@_model_skip
def test_real_k3_mla_compressed_and_absorbed_prefill_match_expanded():
    from runtime.engine import RuntimeConfig, StreamingEngine
    from runtime.glm import _mla_attention

    engine = StreamingEngine(
        str(MODEL_DIR),
        RuntimeConfig(
            min_weight_cache_mb=150,
            max_weight_cache_mb=2000,
            embed_rows=True,
            stream_lm_head=True,
            native_ct_mxfp4=True,
        ),
    )
    try:
        layer = 3
        assert layer in engine.cfg.full_attn_layers
        prefix = f"model.layers.{layer}"
        weights = engine.cache.get(
            engine._layer_key(layer), engine._layer_names(layer)
        )
        rng = np.random.default_rng(176)
        hidden = mx.array(
            rng.standard_normal(
                (1, 5, engine.cfg.hidden_size)
            ).astype(np.float32)
        ).astype(mx.bfloat16)

        expanded_cache = KVCache(engine.cfg.num_hidden_layers)
        expanded = _mla_attention(
            hidden,
            weights,
            prefix,
            engine.cfg,
            expanded_cache,
            layer,
            0,
        )

        compressed_cache = _stepped_mla_cache(
            engine.cfg.num_hidden_layers, absorbed=False
        )
        compressed = _mla_attention(
            hidden,
            weights,
            prefix,
            engine.cfg,
            compressed_cache,
            layer,
            0,
        )

        absorbed_cache = _stepped_mla_cache(
            engine.cfg.num_hidden_layers, absorbed=True
        )
        absorbed = _mla_attention(
            hidden,
            weights,
            prefix,
            engine.cfg,
            absorbed_cache,
            layer,
            0,
        )
        mx.eval(expanded, compressed, absorbed)

        expanded_np = np.asarray(expanded.astype(mx.float32))
        compressed_np = np.asarray(compressed.astype(mx.float32))
        absorbed_np = np.asarray(absorbed.astype(mx.float32))
        compressed_diff = float(
            np.max(np.abs(compressed_np - expanded_np))
        )
        absorbed_diff = float(
            np.max(np.abs(absorbed_np - expanded_np))
        )

        assert compressed_diff < 2e-3, compressed_diff
        assert absorbed_diff < 5e-2, absorbed_diff
        assert compressed_cache.offset == hidden.shape[1]
        assert absorbed_cache.offset == hidden.shape[1]
        assert compressed_cache.nbytes() * 40 < expanded_cache.nbytes()
        assert absorbed_cache.nbytes() == compressed_cache.nbytes()
    finally:
        engine.close()
