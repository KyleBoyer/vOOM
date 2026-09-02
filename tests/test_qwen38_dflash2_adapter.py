"""Small exhaustive gates for the opt-in Qwen3.8 DFlash2 adapter.

No test in this file opens the released draft or target checkpoint.  The MLX
arrays are deliberately tiny and cover the complete supported Q4 width ladder:
one anchor plus one through four proposal positions.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from runtime.dflash2 import CandidateSelector
from runtime.dflash2_adapter import (
    DFlash2DecoderLayer,
    DFlash2Drafter,
    DFlash2RuntimeConfig,
    DFlash2SpeculativeDecoder,
    MAX_QUANTIZED_PROPOSALS,
    build_proposal_block,
    dflash2_sliding_mask,
    expand_sparse_candidate_probabilities,
    greedy_candidate_recall,
    validate_target_compatibility,
)
from runtime.dflash2_schema import (
    DFlash2Config,
    GLM53_FLASH_CONFIG,
    OFFICIAL_CONFIG,
)
from runtime.dspark import (
    CtxCache,
    DSparkSpeculativeDecoder,
    DSparkStats,
    DSparkTapCollector,
)
from runtime.sampler import SamplingParams


TAPS = (5, 19, 33, 47, 61)


def _runtime_config() -> DFlash2RuntimeConfig:
    return DFlash2RuntimeConfig(DFlash2Config.from_mapping(OFFICIAL_CONFIG))


def _glm_runtime_config() -> DFlash2RuntimeConfig:
    return DFlash2RuntimeConfig(
        DFlash2Config.from_mapping(GLM53_FLASH_CONFIG),
        target_model_type="glm5_next",
    )


def test_fused_dynamic_convolution_is_an_explicit_runtime_opt_in():
    default = _runtime_config()
    assert default.fused_dynamic_conv is False
    fused = DFlash2RuntimeConfig(
        DFlash2Config.from_mapping(OFFICIAL_CONFIG),
        fused_dynamic_conv=True,
    )
    assert fused.fused_dynamic_conv is True


def test_ablation_projects_residual_writers_not_the_accumulated_stream():
    identity_conv = SimpleNamespace(
        prepare=lambda hidden: (hidden, None),
        finish=lambda hidden, _kernel, **_kwargs: hidden,
    )
    layer = SimpleNamespace(
        input_layernorm=lambda hidden: hidden,
        post_attention_layernorm=lambda hidden: hidden,
        attention_conv=identity_conv,
        mlp_conv=identity_conv,
        self_attn=SimpleNamespace(
            attend=lambda hidden, _offset, _cache: hidden * 2),
        mlp=lambda hidden: hidden * 3,
    )
    calls = []

    def project(branch):
        calls.append(branch)
        return branch * mx.array([0.0, 1.0])

    output = DFlash2DecoderLayer.__call__(
        layer, mx.array([[[1.0, 2.0]]]), 0, None,
        project_residual=project)
    mx.eval(output)
    # Attention branch [2,4] -> [0,4], leaving the residual [1,2].  The MLP
    # then sees [1,6], and its [3,18] branch becomes [0,18].  Projecting the
    # accumulated hidden stream instead would incorrectly erase coordinate 0.
    np.testing.assert_array_equal(np.array(output), [[[1.0, 24.0]]])
    assert len(calls) == 2


def _compatible_target(**changes):
    values = {
        "model_type": "qwen3_5",
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "vocab_size": 248320,
        "num_hidden_layers": 64,
        "rope_theta": 10_000_000,
        "max_position_embeddings": 262144,
        "tie_word_embeddings": False,
        "attention_bias": False,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "full_attention_interval": 4,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "layer_types": tuple(
            "full_attention" if (index + 1) % 4 == 0
            else "linear_attention"
            for index in range(64)
        ),
    }
    values.update(changes)
    return SimpleNamespace(**values)


def _compatible_glm_target(**changes):
    values = {
        "model_type": "glm5_next",
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "vocab_size": 154880,
        "num_hidden_layers": 45,
        "rope_theta": 10000.0,
        "max_position_embeddings": 1048576,
        "tie_word_embeddings": False,
        "attention_bias": False,
        "layer_types": tuple(
            "deepseek_sparse_attention" if (index + 1) % 4 == 0
            else "linear_attention"
            for index in range(45)
        ),
    }
    values.update(changes)
    return SimpleNamespace(**values)


def test_official_taps_and_huihui_target_geometry_are_compatible():
    config = _runtime_config()
    assert tuple(config.target_layer_ids) == TAPS
    validate_target_compatibility(config, _compatible_target())


def test_glm53_flash_dflash_taps_and_target_geometry_are_compatible():
    config = _glm_runtime_config()
    assert tuple(config.target_layer_ids) == (5, 14, 24, 33, 42)
    validate_target_compatibility(config, _compatible_glm_target())

    with pytest.raises(ValueError, match="compatibility failure"):
        validate_target_compatibility(
            config,
            _compatible_glm_target(
                layer_types=("linear_attention",) * 45),
        )


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("model_type", "qwen3"),
        ("hidden_size", 5119),
        ("intermediate_size", 17407),
        ("vocab_size", 248319),
        ("num_hidden_layers", 63),
        ("rope_theta", 1_000_000),
        ("max_position_embeddings", 131072),
        ("tie_word_embeddings", True),
        ("attention_bias", True),
        ("num_attention_heads", 16),
        ("num_key_value_heads", 8),
        ("head_dim", 128),
        ("full_attention_interval", 8),
        ("linear_num_key_heads", 8),
        ("linear_num_value_heads", 32),
        ("linear_key_head_dim", 64),
        ("linear_value_head_dim", 64),
        ("linear_conv_kernel_dim", 3),
        ("layer_types", ("full_attention",) * 64),
    ],
)
def test_every_target_compatibility_field_fails_closed(field, bad_value):
    with pytest.raises(ValueError, match="compatibility failure"):
        validate_target_compatibility(
            _runtime_config(), _compatible_target(**{field: bad_value}))


@pytest.mark.parametrize("proposal_count", range(1, 5))
def test_q4_proposal_block_is_anchor_plus_one_to_four_masks(proposal_count):
    block = build_proposal_block(11, 31, 8, proposal_count)
    assert block == [11] + [31] * proposal_count
    assert len(block) == proposal_count + 1 <= 5


@pytest.mark.parametrize("proposal_count", [0, 5])
def test_q4_proposal_block_rejects_out_of_range_width(proposal_count):
    with pytest.raises(ValueError, match="proposal count"):
        build_proposal_block(11, 31, 8, proposal_count)


def test_q4_proposal_block_cannot_exceed_checkpoint_width():
    with pytest.raises(ValueError, match="checkpoint block"):
        build_proposal_block(11, 31, 4, 4)


@pytest.mark.parametrize("proposal_count", range(1, 5))
@pytest.mark.parametrize("context_length", [0, 1, 4, 7])
def test_every_supported_block_has_exact_noncausal_sliding_mask(
    proposal_count,
    context_length,
):
    query_length = proposal_count + 1
    window = 5
    actual = dflash2_sliding_mask(context_length, query_length, window)
    mx.eval(actual)
    expected = np.zeros(
        (query_length, context_length + query_length), dtype=np.bool_)
    for query_slot in range(query_length):
        query = context_length + query_slot
        for key in range(context_length + query_length):
            expected[query_slot, key] = (
                (key < context_length and query - key < window)
                or key >= context_length
            )
    np.testing.assert_array_equal(np.array(actual), expected)
    # Later proposal slots are visible to earlier slots: the block is not causal.
    assert np.all(np.array(actual)[:, context_length:])


@pytest.mark.parametrize("proposal_count", range(1, 5))
def test_candidate_selector_q_matches_independent_conditional_oracle(
    proposal_count,
):
    hidden_size, vocab_size, rank, top_k = 4, 8, 3, 4
    temperature = 0.7
    selector = CandidateSelector(hidden_size, vocab_size, rank, top_k)
    selector.hidden_projection.weight = mx.array([
        [0.5, -0.25, 0.75, 0.0],
        [-0.5, 0.5, 0.0, 0.25],
        [0.25, 0.0, -0.5, 0.75],
    ])
    predecessor = mx.array([
        [0.5, -0.1, 0.2],
        [-0.2, 0.7, 0.3],
        [0.8, 0.1, -0.4],
        [0.0, -0.6, 0.9],
        [0.4, 0.3, 0.2],
        [-0.7, 0.2, 0.5],
        [0.1, 0.8, -0.3],
        [0.6, -0.4, 0.1],
    ])
    successor = mx.array([
        [-0.1, 0.3, 0.7],
        [0.6, -0.2, 0.1],
        [0.2, 0.9, -0.5],
        [-0.4, 0.1, 0.8],
        [0.7, 0.2, -0.1],
        [0.3, -0.8, 0.4],
        [-0.6, 0.5, 0.2],
        [0.9, -0.3, 0.0],
    ])
    selector.predecessor_codebook.weight = predecessor
    selector.successor_codebook.weight = successor
    hidden = mx.array([[
        [1.0 + position, 0.5, -0.25 * position, 0.75]
        for position in range(proposal_count)
    ]])
    logits = mx.array([[
        [
            -1.1 + 0.17 * token + 0.09 * position
            + (0.4 if token == (position + 3) % vocab_size else 0.0)
            for token in range(vocab_size)
        ]
        for position in range(proposal_count)
    ]])

    mx.random.seed(100 + proposal_count)
    path, candidates, sparse_q = selector.select(
        hidden, logits, mx.array([1]), temperature=temperature)
    mx.eval(path, candidates, sparse_q)
    full_q = expand_sparse_candidate_probabilities(
        candidates[0], sparse_q[0], vocab_size)
    mx.eval(*full_q)

    projected = hidden @ selector.hidden_projection.weight.T
    parent = 1
    for position in range(proposal_count):
        ids = candidates[0, position]
        scores = (
            logits[0, position, ids]
            + mx.sum(
                predecessor[parent][None]
                * projected[0, position][None]
                * successor[ids],
                axis=-1,
            )
        )
        expected = mx.softmax(scores.astype(mx.float32) / temperature)
        mx.eval(expected)
        np.testing.assert_allclose(
            np.array(sparse_q[0, position]), np.array(expected),
            rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(
            float(mx.sum(full_q[position]).item()), 1.0,
            rtol=0, atol=1e-6)
        selected = int(path[0, position].item())
        selected_slot = int(mx.argmax(ids == selected).item())
        assert float(full_q[position][selected].item()) == pytest.approx(
            float(sparse_q[0, position, selected_slot].item()))
        support = set(int(value) for value in ids.tolist())
        assert all(
            float(full_q[position][token].item()) == 0.0
            for token in range(vocab_size) if token not in support)
        parent = selected


@pytest.mark.parametrize("proposal_count", range(1, 5))
def test_sparse_q_exact_correction_covers_every_accept_prefix(proposal_count):
    vocab_size = 12
    proposals = list(range(proposal_count))
    candidates = mx.array([
        [proposal, proposal + 6] for proposal in proposals
    ])
    sparse = mx.array([[1.0, 0.0]] * proposal_count)
    q_rows = expand_sparse_candidate_probabilities(
        candidates, sparse, vocab_size)
    sampling = SamplingParams(temperature=1.0)

    # reject_after==proposal_count is the all-accepted/bonus case.  Every
    # shorter prefix forces rejection and a deterministic (p-q)+ correction.
    for reject_after in range(proposal_count + 1):
        verified_rows = []
        for position, proposal in enumerate(proposals):
            winner = proposal if position < reject_after else 10
            row = [float("-inf")] * vocab_size
            row[winner] = 0.0
            verified_rows.append(row)
        bonus = [float("-inf")] * vocab_size
        bonus[11] = 0.0
        verified_rows.append(bonus)
        verified = mx.array(verified_rows, dtype=mx.float32)
        mx.random.seed(200 + proposal_count * 10 + reject_after)
        accepted, committed = DFlash2SpeculativeDecoder._verify_stochastic(
            proposals, q_rows, verified, sampling, history=[5])
        if reject_after < proposal_count:
            assert accepted == reject_after
            assert committed == proposals[:reject_after] + [10]
        else:
            assert accepted == proposal_count
            assert committed == proposals + [11]

    assert issubclass(
        DFlash2SpeculativeDecoder, DSparkSpeculativeDecoder)


class _RecordingTapDrafter:
    def __init__(self):
        self.config = SimpleNamespace(target_layer_ids=list(TAPS))
        self.updates = []

    def update_context(self, hidden, offset, caches):
        self.updates.append((int(offset), hidden))
        width = int(hidden.shape[1])
        values = mx.arange(width, dtype=mx.float32).reshape(1, 1, width, 1)
        for cache in caches:
            cache.append(values, values + 100, position_start=offset)


def test_exact_qwen38_taps_are_fused_in_declared_order_across_chunks():
    drafter = _RecordingTapDrafter()
    cache = CtxCache()
    collector = DSparkTapCollector(drafter, [cache])
    collector.begin_attempt()

    for start, width in ((0, 3), (3, 4)):
        positions = mx.arange(start, start + width, dtype=mx.float32)
        # Observation order is irrelevant; fusion order must remain the
        # checkpoint's exact [5, 19, 33, 47, 61] declaration.
        for layer in reversed(TAPS):
            hidden = (positions + layer * 1000).reshape(1, width, 1)
            collector.observe(layer, hidden, position_start=start)
    collector.finish(7)

    assert [offset for offset, _hidden in drafter.updates] == [0, 3]
    for offset, fused in drafter.updates:
        mx.eval(fused)
        assert fused.shape[-1] == len(TAPS)
        for column, layer in enumerate(TAPS):
            expected = mx.arange(
                offset, offset + fused.shape[1], dtype=mx.float32
            ) + layer * 1000
            assert fused[0, :, column].tolist() == expected.tolist()
    assert cache.position_start == 0
    assert cache.position_end == 7
    assert cache.length == 7


def test_dflash_taps_defer_weighted_projection_until_after_target_prefill():
    drafter = _RecordingTapDrafter()
    cache = CtxCache()
    collector = DSparkTapCollector(
        None,
        [cache],
        tap_layers=TAPS,
        position_floor=2,
        defer_updates=True,
    )
    collector.begin_attempt()

    for start, width in ((0, 3), (3, 4)):
        positions = mx.arange(start, start + width, dtype=mx.float32)
        for layer in reversed(TAPS):
            hidden = (positions + layer * 1000).reshape(1, width, 1)
            collector.observe(layer, hidden, position_start=start)

    collector.finish(7)
    assert drafter.updates == []
    assert cache.length == 0
    assert collector.positions == 5
    assert collector.deferred_bytes_peak == 5 * len(TAPS) * 4

    collector.materialize(drafter)
    assert [offset for offset, _hidden in drafter.updates] == [2, 3]
    assert cache.position_start == 2
    assert cache.position_end == 7
    assert cache.length == 5
    assert collector.updates == 2


def test_deferred_taps_fail_closed_on_gaps_and_incomplete_endpoint():
    collector = DSparkTapCollector(
        None, [CtxCache()], tap_layers=TAPS, defer_updates=True)
    collector.begin_attempt()
    for layer in TAPS:
        collector.observe(
            layer,
            mx.zeros((1, 1, 1), dtype=mx.float32),
            position_start=0,
        )
    with pytest.raises(RuntimeError, match="did not reach"):
        collector.finish(3)

    for layer in TAPS[:-1]:
        collector.observe(
            layer,
            mx.zeros((1, 1, 1), dtype=mx.float32),
            position_start=2,
        )
    # The second chunk begins at 2, leaving position 1 absent. The mismatch is
    # detected as soon as the final tap completes that set.
    with pytest.raises(ValueError, match="not contiguous"):
        collector.observe(
            TAPS[-1],
            mx.zeros((1, 1, 1), dtype=mx.float32),
            position_start=2,
        )


def test_dflash_context_left_trim_preserves_absolute_endpoint_and_recent_values():
    cache = CtxCache()
    keys = mx.arange(7, dtype=mx.float32).reshape(1, 1, 7, 1)
    cache.append(keys, keys + 100, position_start=20)

    DFlash2Drafter._trim_context_left(cache, max_length=4)
    mx.eval(cache.k, cache.v)

    assert cache.position_start == 23
    assert cache.position_end == 27
    assert cache.length == 4
    assert cache.k.reshape(-1).tolist() == [3.0, 4.0, 5.0, 6.0]
    assert cache.v.reshape(-1).tolist() == [103.0, 104.0, 105.0, 106.0]


def test_sparse_q_shape_mismatch_fails_closed():
    with pytest.raises(ValueError, match="shapes differ"):
        expand_sparse_candidate_probabilities(
            mx.zeros((2, 2), dtype=mx.int32),
            mx.zeros((2, 3), dtype=mx.float32),
            8,
        )


def test_runtime_quantized_cap_constant_is_four():
    assert MAX_QUANTIZED_PROPOSALS == 4


def _adaptive_policy_decoder(*, minimum=4, threshold=1.0):
    decoder = DFlash2SpeculativeDecoder.__new__(
        DFlash2SpeculativeDecoder)
    decoder.native_mtp_fallback = True
    decoder.fallback_min_dflash_rounds = minimum
    decoder.fallback_min_accepted_per_round = threshold
    decoder._proposal_mode = "dflash2"
    decoder._round_source = "D"
    decoder._proposal_sources = []
    decoder._dflash_rounds = 0
    decoder._dflash_proposed = 0
    decoder._dflash_accepted = 0
    decoder._native_mtp_rounds = 0
    decoder._native_mtp_proposed = 0
    decoder._native_mtp_accepted = 0
    decoder._fallback_switch_round = 0
    return decoder


def test_adaptive_fallback_switches_only_from_verified_productivity():
    decoder = _adaptive_policy_decoder()
    for accepted in (1, 1, 0):
        decoder._proposal_sources.append("D")
        decoder._note_verified_round(4, accepted)
        assert decoder._proposal_mode == "dflash2"

    decoder._proposal_sources.append("D")
    decoder._note_verified_round(4, 0)
    assert decoder._proposal_mode == "native-mtp"
    assert decoder._fallback_switch_round == 4
    assert decoder._dflash_proposed == 16
    assert decoder._dflash_accepted == 2

    decoder._round_source = "M"
    decoder._proposal_sources.append("M")
    decoder._note_verified_round(1, 1)
    assert decoder._native_mtp_rounds == 1
    assert decoder._native_mtp_proposed == 1
    assert decoder._native_mtp_accepted == 1


def test_adaptive_fallback_does_not_switch_at_threshold_or_without_opt_in():
    decoder = _adaptive_policy_decoder()
    for _ in range(4):
        decoder._proposal_sources.append("D")
        decoder._note_verified_round(4, 1)
    assert decoder._proposal_mode == "dflash2"

    decoder = _adaptive_policy_decoder(threshold=4.0)
    decoder.native_mtp_fallback = False
    for _ in range(8):
        decoder._proposal_sources.append("D")
        decoder._note_verified_round(4, 0)
    assert decoder._proposal_mode == "dflash2"


def test_native_mtp_fallback_releases_dflash_context_and_owns_q():
    class FakeMTP:
        def prepare_request_weights(self):
            return {"plain": mx.ones((2,), dtype=mx.bfloat16)}

        def draft_step(self, hidden, pending, _kv, offset, weights):
            assert hidden.shape == (1, 1, 2)
            assert pending == 7 and offset == 12
            assert weights["plain"].dtype == mx.bfloat16
            return mx.array([0.0, 1.0, 4.0, 2.0]), hidden

        def release_request_weights(self, weights):
            resident = sum(value.nbytes for value in weights.values())
            weights.clear()
            return {"resident_bytes": resident}

    decoder = DFlash2SpeculativeDecoder.__new__(
        DFlash2SpeculativeDecoder)
    decoder._native_mtp_drafter = FakeMTP()
    decoder._mtp_kv = object()
    decoder.target = SimpleNamespace(
        _h_last=mx.zeros((1, 1, 2)),
        cache=SimpleNamespace(stats=SimpleNamespace(bytes_read=19)),
    )
    decoder._draft_context_released = False
    decoder._native_mtp_load_s = 0.0
    decoder._native_mtp_release_s = 0.0
    decoder._native_mtp_read_bytes = 0
    decoder._native_mtp_loaded_bytes = 0
    decoder._native_mtp_released_bytes = 0
    decoder._proposal_sources = []
    decoder._candidate_rounds = []
    decoder._unary_rounds = []
    decoder._round_source = ""
    caches = [CtxCache()]
    values = mx.ones((1, 1, 3, 1))
    caches[0].append(values, values, position_start=10)

    proposals, distributions = decoder._propose_native_mtp(
        7, 13, caches, SamplingParams(temperature=0.0), [1, 7])

    assert proposals == [2]
    assert distributions is None
    assert caches[0].k is None and caches[0].v is None
    assert decoder._proposal_sources == ["M"]
    assert decoder._round_source == "M"
    assert decoder._native_mtp_loaded_bytes == 4
    assert decoder._native_mtp_released_bytes == 4


def test_dflash_context_cpu_suspend_restore_is_bit_exact():
    decoder = DFlash2SpeculativeDecoder.__new__(
        DFlash2SpeculativeDecoder)
    decoder._draft_context_released = False
    caches = [CtxCache(), CtxCache()]
    original = []
    for layer, cache in enumerate(caches):
        bits = mx.array(
            np.arange(16, dtype=np.uint16).reshape(1, 2, 4, 2)
            + 100 * layer,
        )
        k = bits.view(mx.bfloat16)
        v = (bits + 17).view(mx.bfloat16)
        cache.append(k, v, position_start=23)
        original.append((
            np.array(k.view(mx.uint16), copy=True),
            np.array(v.view(mx.uint16), copy=True),
        ))
    stats = DSparkStats()

    snapshot = decoder._suspend_draft_context(caches, stats)

    assert all(cache.k is None and cache.v is None for cache in caches)
    assert [cache.position_start for cache in caches] == [23, 23]
    assert [cache.position_end for cache in caches] == [27, 27]
    assert stats.draft_context_suspend_rounds == 1
    assert stats.draft_context_suspended_bytes == 128

    decoder._restore_draft_context(caches, snapshot, stats)

    assert stats.draft_context_restore_rounds == 1
    for cache, (expected_k, expected_v) in zip(
        caches, original, strict=True,
    ):
        np.testing.assert_array_equal(
            np.array(cache.k.view(mx.uint16)), expected_k)
        np.testing.assert_array_equal(
            np.array(cache.v.view(mx.uint16)), expected_v)
        assert cache.position_start == 23
        assert cache.position_end == 27


def test_dflash_context_cpu_snapshot_is_discarded_after_source_switch():
    decoder = DFlash2SpeculativeDecoder.__new__(
        DFlash2SpeculativeDecoder)
    decoder._draft_context_released = False
    cache = CtxCache()
    value = mx.ones((1, 1, 3, 2), dtype=mx.bfloat16)
    cache.append(value, value, position_start=10)
    stats = DSparkStats()

    snapshot = decoder._suspend_draft_context([cache], stats)
    decoder._discard_suspended_draft_context([cache], snapshot)

    assert cache.k is None and cache.v is None
    assert cache.position_start is None and cache.position_end == 0
    assert decoder._draft_context_released is True
    assert stats.draft_context_restore_rounds == 0


def test_greedy_candidate_recall_counts_only_progress_decisions():
    # Bootstrap token 90, then: one accepted + mismatch, all accepted + bonus,
    # and an immediate mismatch. Rejected tails are not observable progress.
    emitted = [90, 11, 12, 21, 22, 23, 31]
    rounds = [(4, 1), (2, 2), (3, 0)]
    candidates = [
        [[11, 41], [12, 42], [99], [98]],
        [[21, 51], [22, 52]],
        [[77, 78], [79], [80]],
    ]
    positions, hits = greedy_candidate_recall(
        emitted, rounds, candidates)
    assert positions == 5
    assert hits == 4
