"""Small gates for the Qwen3.8 SpecForge DSpark sidecar contract."""

from __future__ import annotations

import json
import hashlib
from types import SimpleNamespace

import mlx.core as mx
import pytest

from runtime.dspark import (
    CtxCache,
    DSparkConfig,
    DSparkDrafter,
    DSparkSpeculativeDecoder,
    DSparkTapCollector,
)
from runtime.speculative_tree import SpeculativeTree, TreeDraft
from runtime.dspark_sidecar import build_sidecar


def _qwen38_config():
    return {
        "architectures": ["Qwen3DSparkModel"],
        "model_type": "qwen3",
        "hidden_size": 5120,
        "vocab_size": 248320,
        "num_hidden_layers": 5,
        "num_target_layers": 64,
        "intermediate_size": 10240,
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "block_size": 7,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 262144,
        "enable_confidence_head": True,
        "confidence_head_with_markov": True,
        "markov_rank": 256,
        "rope_parameters": {
            "rope_type": "yarn",
            "rope_theta": 10_000_000,
            "factor": 32,
            "original_max_position_embeddings": 8192,
            "beta_fast": 32,
            "beta_slow": 1,
        },
        "dflash_config": {
            "projector_type": "dspark",
            "mask_token_id": 248077,
            "target_layer_ids": [4, 16, 28, 40, 52],
        },
    }


@pytest.mark.parametrize(
    "architecture",
    ["DSparkDraftModel", "Qwen3DSparkModel"],
)
def test_qwen38_specforge_config_is_target_reuse_sidecar(tmp_path, architecture):
    raw = _qwen38_config()
    raw["architectures"] = [architecture]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw))

    cfg = DSparkConfig.from_json(path)

    assert cfg.model_type == "qwen3_specforge"
    assert cfg.target_model_type == "qwen3_5"
    assert cfg.share_target_embed
    assert cfg.share_target_lm_head
    assert cfg.logits_start == 0
    assert cfg.target_layer_ids == [4, 16, 28, 40, 52]
    assert cfg.rope_parameters["rope_type"] == "yarn"
    assert cfg.rope_theta == 10_000_000


class _RecordingDrafter:
    def __init__(self):
        self.config = SimpleNamespace(target_layer_ids=[4, 16, 28, 40, 52])
        self.values = []

    def update_context(self, hidden, offset, caches):
        self.values.append((int(offset), hidden))
        width = int(hidden.shape[1])
        k = mx.zeros((1, 1, width, 1), dtype=mx.bfloat16)
        for cache in caches:
            cache.append(k, k, position_start=offset)


def test_mixed_depth_taps_align_to_deepest_contiguous_suffix():
    drafter = _RecordingDrafter()
    caches = [CtxCache()]
    collector = DSparkTapCollector(drafter, caches)
    collector.begin_attempt()

    full = mx.arange(8, dtype=mx.float32).reshape(1, 8, 1)
    suffix = mx.arange(6, 8, dtype=mx.float32).reshape(1, 2, 1)
    collector.observe(4, full, position_start=0)
    for layer in (16, 28, 40, 52):
        collector.observe(layer, suffix + layer, position_start=6)
    collector.finish(8)

    offset, fused = drafter.values[0]
    mx.eval(fused)
    assert offset == 6
    assert fused.shape == (1, 2, 5)
    assert fused[0, :, 0].tolist() == [6.0, 7.0]
    assert caches[0].position_start == 6
    assert caches[0].position_end == 8
    assert caches[0].length == 2


def test_tap_collector_bounds_proposal_context_to_recent_window():
    drafter = _RecordingDrafter()
    caches = [CtxCache()]
    collector = DSparkTapCollector(
        drafter, caches, position_floor=6)
    collector.begin_attempt()

    for start in (0, 4):
        hidden = mx.arange(start, start + 4, dtype=mx.float32).reshape(1, 4, 1)
        for layer in collector.tap_layers:
            collector.observe(layer, hidden + layer, position_start=start)
    collector.finish(8)

    assert len(drafter.values) == 1
    assert drafter.values[0][0] == 6
    assert drafter.values[0][1].shape == (1, 2, 5)
    assert caches[0].position_start == 6
    assert caches[0].position_end == 8


def test_tap_collector_drops_old_rows_before_all_taps_are_retained():
    drafter = _RecordingDrafter()
    collector = DSparkTapCollector(
        drafter, [CtxCache()], position_floor=6)
    collector.begin_attempt()
    hidden = mx.arange(8, dtype=mx.float32).reshape(1, 8, 1)

    for layer in collector.tap_layers[:-1]:
        collector.observe(layer, hidden + layer, position_start=0)

    assert set(collector._seen) == set(collector.tap_layers[:-1])
    assert all(value[0] == 6 for value in collector._seen.values())
    assert all(value[2].shape[1] == 2 for value in collector._seen.values())


def test_tap_retry_clears_partial_context_and_noncontiguous_fails_closed():
    drafter = _RecordingDrafter()
    cache = CtxCache()
    cache.append(
        mx.zeros((1, 1, 2, 1)), mx.zeros((1, 1, 2, 1)),
        position_start=6)
    collector = DSparkTapCollector(drafter, [cache])
    collector.begin_attempt()
    assert cache.k is None
    assert cache.position_start is None
    assert cache.position_end == 0

    positions = mx.array([7, 9], dtype=mx.int32)
    with pytest.raises(ValueError, match="contiguous"):
        collector.observe(
            4, mx.zeros((1, 2, 1)), positions=positions)


def test_quantized_sidecar_build_is_hash_pinned_and_reloadable(tmp_path):
    from mlx.utils import tree_flatten

    source = tmp_path / "source"
    output = tmp_path / "sidecar"
    source.mkdir()
    raw = {
        "architectures": ["DSparkDraftModel"],
        "model_type": "qwen3",
        "hidden_size": 32,
        "vocab_size": 64,
        "num_hidden_layers": 1,
        "num_target_layers": 3,
        "intermediate_size": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "block_size": 3,
        "rms_norm_eps": 1e-6,
        "enable_confidence_head": True,
        "confidence_head_with_markov": True,
        "markov_rank": 32,
        "dflash_config": {
            "projector_type": "dspark",
            "mask_token_id": 63,
            "target_layer_ids": [0, 1],
        },
    }
    config_path = source / "config.json"
    config_path.write_text(json.dumps(raw))
    model = DSparkDrafter(DSparkConfig.from_json(config_path))
    source_weights = dict(tree_flatten(model.parameters()))
    mx.eval(*source_weights.values())
    weights_path = source / "model.safetensors"
    mx.save_safetensors(str(weights_path), source_weights)
    source_hash = hashlib.sha256(weights_path.read_bytes()).hexdigest()

    manifest = build_sidecar(
        source,
        output,
        expected_source_sha256=source_hash,
        source_repo="test/tiny",
        source_revision="deadbeef",
        group_size=32,
    )

    assert manifest["target_verified"]
    assert manifest["output_bytes"] < manifest["source_bytes"]
    assert json.loads((output / "config.json").read_text())["quantization"] == {
        "bits": 4, "group_size": 32, "mode": "affine"}
    loaded = DSparkDrafter.load(output)
    assert loaded.config.share_target_embed
    assert loaded.config.share_target_lm_head
    loaded.bind_target_embed(
        lambda token_ids: mx.zeros((1, len(token_ids), 32)))
    assert loaded._embed_block([1, 2, 3]).shape == (1, 3, 32)
    assert any(name.endswith(".scales") for name in dict(
        tree_flatten(loaded.parameters())))

    with pytest.raises(ValueError, match="hash mismatch"):
        build_sidecar(
            source,
            tmp_path / "bad", expected_source_sha256="0" * 64)


class _Endpoint:
    def __init__(self, fed):
        self.fed = fed

    def fork(self):
        return self


class _HybridKV:
    def __init__(self):
        self.offset = 3
        self.lengths = [3, 0]
        self.kda_cache = _Endpoint(0)

    def layer_lengths(self):
        return tuple(self.lengths)

    def trim_layer_lengths(self, lengths):
        self.lengths = list(lengths)
        self.offset = self.lengths[0]

    def nbytes(self):
        return 0


class _HybridTarget:
    def __init__(self, accepted):
        self.accepted = accepted
        self.cfg = SimpleNamespace(
            model_type="qwen3_5", hidden_size=2, vocab_size=16,
            num_hidden_layers=2, eos_token_ids=((6,) if accepted == 0 else ()))
        self.rc = SimpleNamespace(stepped_kv_threshold=0, context_bound=0)
        self.effective_max_position_embeddings = 262144
        self.rope_profile = "native"
        self.governor = None
        self._hot_prompt_slots = []
        self._tap_hidden = {}
        self._h_last = mx.array([[[30.0, 31.0]]])
        self._collector = None
        self._true_peak_metal_bytes = 0
        self.last_kv = None
        self.endpoints = {i: _Endpoint(i) for i in (1, 2, 3)}
        self.endpoint_requests = []
        self.tokenizer = SimpleNamespace(
            encode=lambda _text: SimpleNamespace(ids=[1, 2, 3]),
            decode=lambda ids: ",".join(str(v) for v in ids),
        )

    def release_request_state(self):
        return None

    def begin_dspark_tap_capture(self, collector):
        self._collector = collector

    def end_dspark_tap_capture(self, collector):
        assert self._collector is collector
        self._collector = None

    def generate_with_memory_retry(self, _prompt, max_tokens, **_kwargs):
        assert max_tokens == 1
        self.bootstrap_prompt = _prompt
        self._collector.begin_attempt()
        for tap in self._collector.tap_layers:
            self._collector.observe(
                tap, mx.full((1, 3, 2), tap + 1), position_start=0)
        self.last_kv = _HybridKV()
        return {
            "text": "4", "tokens": [4], "prefill_s": 0.0,
            "first_token_s": 0.0, "path_stats": {},
        }

    def _final_logits(self, _hidden):
        return mx.full((16,), -100.0).at[4].add(200.0)

    def forward_tokens_serial_positions(
        self, tokens, kv, *, tap_layers, capture_kda_endpoints=False,
    ):
        verify_hook = getattr(self, "verify_hook", None)
        if verify_hook is not None:
            verify_hook()
        width = len(tokens)
        assert tokens == [4, 10, 11][:width]
        assert capture_kda_endpoints == (width > 1)
        kv.offset += width
        kv.lengths[0] += width
        kv.kda_cache = self.endpoints[width]
        self._h_window = mx.array([[
            [40.0, 41.0], [100.0, 101.0], [110.0, 111.0]][:width]])
        self._h_last = self._h_window[:, -1:, :]
        self._tap_hidden = {
            layer: mx.full((1, width, 2), layer + 10)
            for layer in tap_layers}
        predictions = [
            10 if self.accepted >= 1 else 6,
            11 if self.accepted >= 2 else 7,
            8,
        ]
        logits = mx.full((width, 16), -100.0)
        for row, token in enumerate(predictions[:width]):
            logits[row, token] = 100.0
        return logits

    def consume_serial_kda_endpoint(self, fed):
        self.endpoint_requests.append(fed)
        return None if fed is None else self.endpoints[fed]

    def _note_true_peak(self):
        return None


class _HybridDrafter:
    block_size = 3
    confidence_head = None

    def __init__(self):
        self.config = SimpleNamespace(
            model_type="qwen3_specforge", target_model_type="qwen3_5",
            hidden_size=2, vocab_size=16, block_size=3, logits_start=0,
            target_layer_ids=[0, 1], share_target_embed=True,
            share_target_lm_head=True)

    def make_ctx_cache(self):
        return [CtxCache()]

    def update_context(self, hidden, offset, caches):
        width = int(hidden.shape[1])
        value = mx.zeros((1, 1, width, 1))
        caches[0].append(value, value, position_start=offset)


@pytest.mark.parametrize(
    "accepted,max_tokens,expected,fed,width,endpoint_request",
    [
        (0, 3, [4, 6], 1, 2, 1),
        (1, 3, [4, 10, 7], 2, 2, None),
        (2, 4, [4, 10, 11, 8], 3, 3, None),
    ],
)
def test_qwen_hybrid_dspark_accept_prefix_restores_exact_recurrent_endpoint(
    accepted, max_tokens, expected, fed, width, endpoint_request,
):
    target = _HybridTarget(accepted)
    decoder = DSparkSpeculativeDecoder(
        target, _HybridDrafter(), max_draft_tokens=2,
        prompt_cache_min_tokens=0)
    decoder._propose = lambda *_args: [10, 11][:_args[3]]

    result = decoder.generate("x", max_tokens=max_tokens)

    assert result["tokens"] == expected
    assert target.bootstrap_prompt.disable_hot_prompt_kv is True
    assert target.bootstrap_prompt.retain_paged_kv_after_generate is True
    assert target.bootstrap_prompt.stable_boundary_tokens == 0
    assert target.last_kv.offset == 3 + fed
    assert target.last_kv.lengths == [3 + fed, 0]
    assert target.last_kv.kda_cache is target.endpoints[fed]
    assert target.endpoint_requests == [endpoint_request]
    assert result["path_stats"]["dspark_qwen_kda_endpoint_restores"] == int(
        fed < width)


def test_qwen_tree_round_commits_only_authoritative_branch(monkeypatch):
    target = _HybridTarget(accepted=0)
    decoder = DSparkSpeculativeDecoder(
        target, _HybridDrafter(), max_draft_tokens=2,
        prompt_cache_min_tokens=0)
    tree = SpeculativeTree(
        token_ids=(4, 10, 12, 11),
        depths=(0, 1, 1, 2),
        parents=(-1, 0, 0, 1),
        children=({10: 1, 12: 2}, {11: 3}, {}, {}),
    )
    decoder._propose = lambda *_args: TreeDraft(tree)

    class Factors:
        @staticmethod
        def nbytes():
            return 123

    class Verification:
        def __init__(self):
            self.tree = tree
            self.factors = Factors()
            self.logits = mx.full((4, 16), -100.0)
            for row, token in enumerate((10, 11, 7, 8)):
                self.logits[row, token] = 100.0

        def commit(self, path, *, target, kv):
            assert tuple(path) == (0, 1, 3)
            kv.offset += len(path)
            kv.lengths[0] += len(path)
            kv.kda_cache = target.endpoints[len(path)]
            target._h_window = mx.zeros((1, len(path), 2))
            target._h_last = target._h_window[:, -1:, :]
            target._tap_hidden = {
                layer: mx.full((1, len(path), 2), layer + 10)
                for layer in (0, 1)}

    monkeypatch.setattr(
        "runtime.qwen35_tree_verify.verify_qwen35_tree",
        lambda *_args, **_kwargs: Verification(),
    )
    result = decoder.generate("x", max_tokens=4)

    assert result["tokens"] == [4, 10, 11, 8]
    assert target.last_kv.offset == 6
    assert target.endpoint_requests == []
    assert result["path_stats"]["speculative_proposed"] == 3
    assert result["path_stats"]["speculative_accepted"] == 2
    assert result["path_stats"]["dspark_tree_nodes_verified"] == 4
    assert result["path_stats"]["dspark_tree_paths_committed"] == 1
    assert result["path_stats"]["dspark_tree_factor_bytes_peak"] == 123


def test_qwen_sidecar_is_released_during_target_sweep_then_reloaded():
    target = _HybridTarget(accepted=0)
    initial = _HybridDrafter()
    reloads = []

    def loader():
        replacement = _HybridDrafter()
        reloads.append(replacement)
        return replacement

    decoder = DSparkSpeculativeDecoder(
        target,
        initial,
        max_draft_tokens=2,
        prompt_cache_min_tokens=0,
        drafter_loader=loader,
        release_between_sweeps=True,
        drafter_storage_bytes=1234,
    )
    def assert_released():
        assert decoder.drafter is None

    target.verify_hook = assert_released
    decoder._propose = lambda *_args: [10]

    result = decoder.generate("x", max_tokens=3)

    assert len(reloads) == 1
    assert decoder.drafter is reloads[0]
    assert result["tokens"] == [4, 6]
    assert result["path_stats"]["dspark_sidecar_release_between_sweeps"] == 1
    assert result["path_stats"]["dspark_sidecar_round_releases"] == 1
    assert result["path_stats"]["dspark_sidecar_round_loads"] == 1
    assert result["path_stats"]["dspark_sidecar_loaded_bytes"] == 1234


def test_forced_sidecar_release_is_available_for_prompt_lifetime_isolation():
    decoder = DSparkSpeculativeDecoder(
        _HybridTarget(accepted=0),
        _HybridDrafter(),
        max_draft_tokens=2,
        prompt_cache_min_tokens=0,
        release_between_sweeps=False,
    )
    resident = decoder.drafter

    assert decoder._release_drafter() == 0
    assert decoder.drafter is resident
    decoder._release_drafter(force=True)
    assert decoder.drafter is None
