"""F94: layer-stationary (layer-major) dense prefill oracle.

Proves two separate things, mirroring this doc's own established gate order
(docs/future_lossless_techniques.md, F94): (1) numerical equivalence against
a REAL HF reference AND against the existing chunk-major sweep, across
several tile widths; (2) the actual point of this technique -- each layer's
weights are fetched from the paging callback exactly once regardless of how
many tiles the prompt is split into, unlike chunk-major which re-fetches
every layer once per chunk.

Uses a tiny synthetic dense (ordinary Qwen2-style attention, no MoE, no
hybrid DeltaNet) HF config + random weights -- no real checkpoint or NAS
needed, same pattern as tests/test_qwen35_oracle.py.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import torch

from runtime.config import ModelConfig
from runtime.kv_cache import KVCache
from runtime.layer_stationary import run_layer_stationary_sweep
from runtime import layer_runner

from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model


HIDDEN = 32
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 8
INTERMEDIATE = 48
LAYERS = 3
VOCAB = 64
LENGTH = 16


def _hf_config() -> Qwen2Config:
    config = Qwen2Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=LAYERS,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    return config


def _runtime_config() -> ModelConfig:
    return ModelConfig(
        model_type="qwen2",
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=LAYERS,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        vocab_size=VOCAB,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        max_position_embeddings=4096,
        tie_word_embeddings=False,
        attention_bias=False,
        head_dim=HEAD_DIM,
        eos_token_ids=(),
        torch_dtype="float32",
    )


def _randomize(module: torch.nn.Module, seed: int) -> None:
    torch.manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(mean=0.0, std=0.25)


def _weights_by_layer(real: Qwen2Model) -> list[dict]:
    """One weight dict per layer, keyed exactly as layer_runner.run_block
    expects (``model.layers.{i}.*``)."""
    layers = []
    for i, layer in enumerate(real.layers):
        prefix = f"model.layers.{i}"
        state = layer.state_dict()
        w = {
            f"{prefix}.{name}": mx.array(value.detach().numpy())
            for name, value in state.items()
        }
        layers.append(w)
    return layers


def _final_norm(x: mx.array, real: Qwen2Model, cfg: ModelConfig) -> mx.array:
    """Qwen2Model.last_hidden_state is POST its own final norm; run_block's
    raw sweep output is pre-norm (that's layer_runner.final_logits's job,
    applied separately in the real engine). Apply it here so comparisons
    against last_hidden_state are apples-to-apples."""
    norm_weight = mx.array(real.norm.weight.detach().numpy())
    return mx.fast.rms_norm(x, norm_weight, cfg.rms_norm_eps)


def _assert_close(actual: mx.array, expected: torch.Tensor, tolerance: float = 2e-4):
    mx.eval(actual)
    actual_np = np.array(actual)
    expected_np = expected.detach().numpy()
    assert actual_np.shape == expected_np.shape
    difference = float(np.max(np.abs(actual_np - expected_np)))
    assert difference < tolerance, f"oracle mismatch: max abs diff {difference}"


def _chunk_major_sweep(x: mx.array, weights_by_layer: list[dict],
                        cfg: ModelConfig, kv: KVCache, chunk: int) -> mx.array:
    """The EXISTING schedule, reimplemented minimally here (not calling
    StreamingEngine, which needs a real weight store/engine instance) --
    chunk-major: for each chunk of positions, iterate every layer."""
    total = int(x.shape[1])
    outputs = []
    pos = 0
    while pos < total:
        end = min(pos + chunk, total)
        xc = x[:, pos:end, :]
        for layer in range(cfg.num_hidden_layers):
            xc = layer_runner.run_block(
                xc, weights_by_layer[layer], f"model.layers.{layer}", cfg,
                kv, layer, pos)
        outputs.append(xc)
        pos = end
    return outputs[0] if len(outputs) == 1 else mx.concatenate(outputs, axis=1)


def test_chunk_major_baseline_matches_real_hf_reference():
    """Sanity check on the EXISTING path before trusting it as a comparison
    baseline for the new layer-stationary schedule."""
    config = _hf_config()
    real = Qwen2Model(config)
    _randomize(real, 1)
    real.eval()
    torch.manual_seed(2)
    input_ids = torch.randint(0, VOCAB, (1, LENGTH))
    with torch.no_grad():
        expected = real(input_ids).last_hidden_state

    weights_by_layer = _weights_by_layer(real)
    embed_weight = mx.array(real.embed_tokens.weight.detach().numpy())
    x = layer_runner.embed(mx.array(input_ids.numpy()[0]), embed_weight)

    rc = _runtime_config()
    for chunk in (1, 4, 8, 16):
        kv = KVCache(LAYERS)
        actual = _chunk_major_sweep(x, weights_by_layer, rc, kv, chunk)
        _assert_close(_final_norm(actual, real, rc), expected, tolerance=5e-4)


def test_layer_stationary_matches_chunk_major_and_hf_reference():
    config = _hf_config()
    real = Qwen2Model(config)
    _randomize(real, 3)
    real.eval()
    torch.manual_seed(4)
    input_ids = torch.randint(0, VOCAB, (1, LENGTH))
    with torch.no_grad():
        expected = real(input_ids).last_hidden_state

    weights_by_layer = _weights_by_layer(real)
    embed_weight = mx.array(real.embed_tokens.weight.detach().numpy())
    x = layer_runner.embed(mx.array(input_ids.numpy()[0]), embed_weight)
    rc = _runtime_config()

    kv_reference = KVCache(LAYERS)
    reference = _chunk_major_sweep(x, weights_by_layer, rc, kv_reference, chunk=16)
    _assert_close(_final_norm(reference, real, rc), expected, tolerance=5e-4)

    for tile_width in (1, 4, 8, 16):
        kv = KVCache(LAYERS)
        actual = run_layer_stationary_sweep(
            x, rc, kv, offset=0, tile_width=tile_width,
            get_layer_weights=lambda layer: weights_by_layer[layer])
        _assert_close(_final_norm(actual, real, rc), expected, tolerance=5e-4)
        # Byte-identical (well within float noise) against the existing
        # chunk-major schedule too, not just the HF reference -- the actual
        # claim F94 makes is "same computation, different order."
        _assert_close(actual, torch.from_numpy(np.array(reference)),
                       tolerance=2e-6)


def test_layer_stationary_fetches_each_layers_weights_exactly_once():
    """The actual point of F94: unlike chunk-major (one weight fetch per
    chunk per layer), layer-stationary fetches each layer's weights exactly
    once regardless of how many tiles the prompt is split into."""
    config = _hf_config()
    real = Qwen2Model(config)
    _randomize(real, 5)
    weights_by_layer = _weights_by_layer(real)
    embed_weight = mx.array(real.embed_tokens.weight.detach().numpy())
    torch.manual_seed(6)
    input_ids = torch.randint(0, VOCAB, (1, LENGTH))
    x = layer_runner.embed(mx.array(input_ids.numpy()[0]), embed_weight)
    rc = _runtime_config()

    for tile_width in (1, 4, 8, 16):
        fetch_counts: dict[int, int] = {}

        def counting_get_layer_weights(layer: int, _fetch_counts=fetch_counts):
            _fetch_counts[layer] = _fetch_counts.get(layer, 0) + 1
            return weights_by_layer[layer]

        kv = KVCache(LAYERS)
        run_layer_stationary_sweep(
            x, rc, kv, offset=0, tile_width=tile_width,
            get_layer_weights=counting_get_layer_weights)
        assert fetch_counts == {i: 1 for i in range(LAYERS)}, (
            f"tile_width={tile_width}: expected exactly one fetch per layer, "
            f"got {fetch_counts}")


def test_layer_stationary_mlp_last_only_matches_final_position():
    """F36's dead-position elimination composes with F94 exactly as it
    already does with chunk-major: only the tile containing the true final
    position needs a meaningful mlp_last_only result, and that result must
    match the un-sliced full sweep's own final position."""
    config = _hf_config()
    real = Qwen2Model(config)
    _randomize(real, 7)
    weights_by_layer = _weights_by_layer(real)
    embed_weight = mx.array(real.embed_tokens.weight.detach().numpy())
    torch.manual_seed(8)
    input_ids = torch.randint(0, VOCAB, (1, LENGTH))
    x = layer_runner.embed(mx.array(input_ids.numpy()[0]), embed_weight)
    rc = _runtime_config()

    kv_full = KVCache(LAYERS)
    full = run_layer_stationary_sweep(
        x, rc, kv_full, offset=0, tile_width=8,
        get_layer_weights=lambda layer: weights_by_layer[layer])

    for tile_width in (1, 4, 8, 16):
        kv = KVCache(LAYERS)
        sliced = run_layer_stationary_sweep(
            x, rc, kv, offset=0, tile_width=tile_width,
            get_layer_weights=lambda layer: weights_by_layer[layer],
            mlp_last_only=True)
        assert sliced.shape == (1, 1, HIDDEN)
        _assert_close(sliced, torch.from_numpy(np.array(full[:, -1:, :])),
                       tolerance=1e-6)
