"""Focused exactness gates for the explicit recurrent Qwen MTP depth-2 path."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import pytest

from runtime.kv_cache import KVCache
from runtime.qwen35_mtp import (
    ProposalQPolicy, QwenMTPDrafter, QwenMTPSpeculativeEngine)
from runtime.sampler import SamplingParams


class _Encoding:
    def __init__(self, ids):
        self.ids = list(ids)


class _Tokenizer:
    def encode(self, _text):
        return _Encoding((1, 2, 3))

    def decode(self, ids):
        return ",".join(str(value) for value in ids)


class _Endpoint:
    def __init__(self, fed_positions):
        self.fed_positions = fed_positions

    def synchronize(self):
        return None


class _TargetKV:
    def __init__(self):
        self.offset = 3
        self.lengths = [3, 1]
        self.kda_cache = _Endpoint(0)

    def layer_lengths(self):
        return tuple(self.lengths)

    def trim_layer_lengths(self, lengths):
        self.lengths = list(lengths)
        self.offset = self.lengths[0]

    def trim(self, offset):
        self.offset = int(offset)
        self.lengths[0] = int(offset)

    def nbytes(self):
        return 0


class _Store:
    mtplx_mtp_sidecar = None

    def __init__(self):
        for name in (
            "fast_tier_bytes", "archive_bytes", "parallel_tier_fetches",
            "parallel_tier_fast_bytes", "parallel_tier_archive_bytes",
        ):
            setattr(self, name, 0)

    def names_with_prefix(self, prefix):
        return ["mtp.fc.weight"] if prefix == "mtp." else []


class _Target:
    """Serial verifier whose endpoint labels are its ordinary-step oracle."""

    def __init__(self, accepted_prefix, *, eos=()):
        self.accepted_prefix = accepted_prefix
        self.tokenizer = _Tokenizer()
        self.cfg = SimpleNamespace(num_experts=0, eos_token_ids=tuple(eos))
        self.store = _Store()
        self.cache = SimpleNamespace(
            stats=SimpleNamespace(
                hits=0, misses=0, evictions=0, bytes_read=0),
            total_bytes=0,
            max_bytes=1,
        )
        self.governor = None
        self.expert_hits = self.expert_misses = 0
        self._layer_transient = 0
        self._prefill_layer_transient = 0
        self._decode_layer_transient = 0
        self._layer_transient_margin = 0
        self._token_transient = 0
        self._true_peak_metal_bytes = 0
        self._request_profiler = None
        self._hot_prompt_slots = []
        self._h_last = mx.array([[[30.0]]])
        self._model_dir = "/models/fake-qwen"
        self.effective_max_position_embeddings = 0
        self.rope_profile = "test"
        self.last_kv = None
        self.endpoint_requests = []
        self.serial_calls = []
        self.endpoints = {index: _Endpoint(index) for index in (1, 2, 3)}

    def generate(self, _prompt, max_tokens, **kwargs):
        assert max_tokens == 1
        if kwargs.get("constraint") is not None:
            kwargs["constraint"].accept_token(4)
        self.last_kv = _TargetKV()
        return {
            "text": "4", "tokens": [4], "prefill_s": 0.0,
            "first_token_s": 0.0, "decode_s": 0.0, "total_s": 0.0,
            "termination_reason": "length", "stop_sequence": None,
            "path_stats": {}, "prompt_tokens": 3,
        }

    def forward_tokens_serial_positions(
        self, tokens, kv, *, capture_kda_endpoints=False,
    ):
        assert tokens == [4, 10, 11]
        assert capture_kda_endpoints
        self.serial_calls.append(list(tokens))
        kv.offset += 3
        kv.lengths[0] += 3
        kv.kda_cache = self.endpoints[3]
        self._h_window = mx.array([[[40.0], [100.0], [110.0]]])
        self._h_last = self._h_window[:, -1:, :]
        target_tokens = [
            10 if self.accepted_prefix >= 1 else 6,
            11 if self.accepted_prefix >= 2 else 7,
            8,
        ]
        logits = mx.full((3, 16), -100.0)
        for row, token in enumerate(target_tokens):
            logits = logits.at[row, token].add(200.0)
        return logits

    def consume_serial_kda_endpoint(self, fed_positions):
        self.endpoint_requests.append(fed_positions)
        return None if fed_positions is None else self.endpoints[fed_positions]


class _RecurrentDrafter:
    def __init__(self):
        self.calls = []
        self.mtp_kv = None

    def draft_step(self, hidden, token, mtp_kv, offset, _weights=None):
        step = len(self.calls)
        self.mtp_kv = mtp_kv
        self.calls.append({
            "hidden": float(hidden.reshape(-1)[0].item()),
            "token": int(token),
            "offset": int(offset),
        })
        key = mx.full((1, 1, 1, 1), float(step + 1))
        mtp_kv.update(0, key, key)
        proposal = (10, 11)[step]
        logits = mx.full((16,), -100.0).at[proposal].add(200.0)
        next_hidden = mx.array([[[1000.0 + step]]])
        return logits, next_hidden


def test_released_drafter_step_returns_post_block_hidden(monkeypatch):
    weights = {
        "mtp.pre_fc_norm_embedding.weight": mx.ones((1,)),
        "mtp.pre_fc_norm_hidden.weight": mx.ones((1,)),
        "mtp.fc.weight": mx.ones((1, 2)),
        "mtp.layers.0.input_layernorm.weight": mx.ones((1,)),
        "mtp.layers.0.post_attention_layernorm.weight": mx.ones((1,)),
        "mtp.norm.weight": mx.ones((1,)),
    }

    def fake_attention(h, _w, _prefix, _cfg, kv, _layer, _offset):
        item = mx.ones((1, 1, 1, 1))
        kv.update(0, item, item)
        return mx.full(h.shape, 2.0)

    monkeypatch.setattr("runtime.qwen35_mtp.qwen35_rms_norm",
                        lambda value, _weight, _eps: value)
    monkeypatch.setattr("runtime.qwen35_mtp._full_attention", fake_attention)
    monkeypatch.setattr("runtime.qwen35_mtp._swiglu",
                        lambda h, _w, _prefix: mx.full(h.shape, 3.0))
    monkeypatch.setattr("runtime.qwen35_mtp.final_logits",
                        lambda hidden, _norm, _head, _eps: hidden.reshape(-1))

    engine = SimpleNamespace(
        store=SimpleNamespace(names_with_prefix=lambda prefix: (
            list(weights) if prefix == "mtp." else [])),
        cache=SimpleNamespace(get=lambda _key, _names: weights),
        cfg=SimpleNamespace(num_experts=0, rms_norm_eps=0.0, vocab_size=1),
        _embed=lambda tokens: mx.array(tokens).reshape(1, 1, 1),
        _lm_head_weight=lambda: mx.ones((1, 1)),
    )
    drafter = QwenMTPDrafter(engine)
    mtp_kv = KVCache(1)

    logits, hidden = drafter.draft_step(
        mx.array([[[5.0]]]), 7, mtp_kv, 2, weights)

    # fc([embed=7, previous_hidden=5])=12; attention residual adds 2;
    # MLP residual adds 3.  The returned recurrent hidden is exactly 17.
    assert hidden.tolist() == [[[17.0]]]
    assert logits.shape == (1,)
    assert logits.tolist() == [17.0]
    assert mtp_kv.layer_lengths() == (1,)


@pytest.mark.parametrize(
    "accepted_prefix,max_tokens,expected_tokens,expected_fed,mtp_length,outcome",
    [
        (0, 2, [4, 6], 1, 1, "R"),
        (1, 3, [4, 10, 7], 2, 2, "A1R"),
        (2, 4, [4, 10, 11, 8], 3, 2, "A2"),
    ],
)
def test_k2_accept_prefixes_match_ordinary_state_oracle(
    accepted_prefix, max_tokens, expected_tokens, expected_fed,
    mtp_length, outcome,
):
    target = _Target(accepted_prefix)
    engine = QwenMTPSpeculativeEngine(
        target,
        max_prompt_tokens=8,
        min_output_tokens=2,
        plain_warmup_tokens=0,
        adaptive_stop=False,
        depth=2,
    )
    drafter = _RecurrentDrafter()
    engine.drafter = drafter

    result = engine.generate("x", max_tokens)

    assert result["tokens"] == expected_tokens
    assert target.last_kv.offset == 3 + expected_fed
    assert target.last_kv.lengths == [3 + expected_fed, 1]
    assert target.last_kv.kda_cache is target.endpoints[expected_fed]
    assert float(target._h_last.reshape(-1)[0].item()) == {
        1: 40.0, 2: 100.0, 3: 110.0,
    }[expected_fed]
    assert target.endpoint_requests == [
        None if expected_fed == 3 else expected_fed]
    assert drafter.mtp_kv.layer_lengths() == (mtp_length,)
    assert drafter.calls == [
        {"hidden": 30.0, "token": 4, "offset": 2},
        {"hidden": 1000.0, "token": 10, "offset": 3},
    ]
    stats = result["path_stats"]
    assert stats["qwen_mtp_depth"] == 2
    assert stats["qwen_mtp_verify_width"] == 3
    assert stats["qwen_mtp_round_outcomes"] == outcome
    assert stats["qwen_mtp_accepted"] == accepted_prefix
    assert stats["qwen_mtp_verified_proposals"] == min(
        2, accepted_prefix + 1)
    assert stats["qwen_mtp_target_prefix_rollbacks"] == int(expected_fed < 3)
    assert stats["qwen_mtp_draft_kv_rollbacks"] == int(mtp_length < 2)


def test_k2_grammar_fork_masks_both_provisional_steps_without_mutating_target():
    class _Constraint:
        def __init__(self, accepted=()):
            self.accepted = list(accepted)
            self.completed = False
            self.forks = []

        def fork(self):
            forked = _Constraint(self.accepted)
            self.forks.append(forked)
            return forked

        def mask_logits(self, logits):
            legal = {
                (4,): 10,
                (4, 10): 11,
                (4, 10, 11): 8,
            }[tuple(self.accepted)]
            masked = mx.full(logits.shape, -1000.0)
            return masked.at[..., legal].add(2000.0)

        def accept_token(self, token):
            self.accepted.append(int(token))

    class _RawIllegalDrafter(_RecurrentDrafter):
        def draft_step(self, hidden, token, mtp_kv, offset, _weights=None):
            step = len(self.calls)
            self.mtp_kv = mtp_kv
            self.calls.append({
                "hidden": float(hidden.reshape(-1)[0].item()),
                "token": int(token),
                "offset": int(offset),
            })
            key = mx.full((1, 1, 1, 1), float(step + 1))
            mtp_kv.update(0, key, key)
            raw_illegal, legal = ((12, 10), (13, 11))[step]
            logits = mx.full((16,), -100.0)
            logits = logits.at[legal].add(200.0)
            logits = logits.at[raw_illegal].add(300.0)
            return logits, mx.array([[[1000.0 + step]]])

    target = _Target(2)
    engine = QwenMTPSpeculativeEngine(
        target,
        max_prompt_tokens=8,
        min_output_tokens=2,
        plain_warmup_tokens=0,
        adaptive_stop=False,
        depth=2,
        grammar_aware_draft=True,
    )
    drafter = _RawIllegalDrafter()
    engine.drafter = drafter
    constraint = _Constraint()

    result = engine.generate("x", 4, constraint=constraint)

    assert result["tokens"] == [4, 10, 11, 8]
    assert constraint.accepted == [4, 10, 11, 8]
    assert len(constraint.forks) == 1
    assert constraint.forks[0].accepted == [4, 10, 11]
    assert drafter.calls[1]["token"] == 10
    stats = result["path_stats"]
    assert stats["qwen_mtp_accepted_by_step"] == [1, 1]
    assert stats["qwen_mtp_grammar_masked_draft_tokens"] == 2
    assert stats["qwen_mtp_grammar_masked_draft_rounds"] == 1


@pytest.mark.parametrize(
    "eos,expected_tokens,expected_fed,verified",
    [
        ((10,), [4, 10], 1, 1),
        ((11,), [4, 10, 11], 2, 2),
    ],
)
def test_k2_terminal_proposal_rolls_back_every_unfed_suffix(
    eos, expected_tokens, expected_fed, verified,
):
    target = _Target(2, eos=eos)
    engine = QwenMTPSpeculativeEngine(
        target, max_prompt_tokens=8, min_output_tokens=2,
        plain_warmup_tokens=0, adaptive_stop=False, depth=2)
    drafter = _RecurrentDrafter()
    engine.drafter = drafter

    result = engine.generate("x", 8)

    assert result["tokens"] == expected_tokens
    assert result["termination_reason"] == "eos"
    assert target.last_kv.offset == 3 + expected_fed
    assert target.last_kv.kda_cache is target.endpoints[expected_fed]
    assert drafter.mtp_kv.layer_lengths() == (expected_fed,)
    assert result["path_stats"]["qwen_mtp_verified_proposals"] == verified


def test_k2_stochastic_uses_separate_sequential_q_and_rejection_correction():
    target = _Target(1)
    engine = QwenMTPSpeculativeEngine(
        target,
        max_prompt_tokens=8,
        min_output_tokens=2,
        plain_warmup_tokens=0,
        adaptive_stop=False,
        depth=2,
        proposal_replay_top_k=1,
        proposal_q_policy=ProposalQPolicy("flat", 1),
    )
    engine.drafter = _RecurrentDrafter()

    result = engine.generate(
        "x", 3, sampling=SamplingParams(temperature=1.0, seed=17))

    # q1=p1=delta(10), then q2=delta(11) is rejected against p2=delta(7).
    # The positive-part residual must emit 7, not leak q2's proposal.
    assert result["tokens"] == [4, 10, 7]
    stats = result["path_stats"]
    assert stats["qwen_mtp_accepted_by_step"] == [1, 0]
    assert stats["qwen_mtp_verified_by_step"] == [1, 1]
    assert stats["qwen_mtp_stochastic_expected_acceptance_by_step"] == [1.0, 0.0]
    assert stats["qwen_mtp_q_policy"]["name"] == "flat-k1"
    records = stats["qwen_mtp_proposal_q_replay"]
    assert [record["draft_step_index"] for record in records] == [0, 1]
    assert [record["accepted"] for record in records] == [True, False]


def test_k2_constructor_is_strict_and_opt_in():
    target = _Target(2)
    assert QwenMTPSpeculativeEngine(target).depth == 1
    for invalid in (0, 3, True, "2"):
        with pytest.raises(ValueError, match="depth must be 1 or 2"):
            QwenMTPSpeculativeEngine(target, depth=invalid)
    with pytest.raises(ValueError, match="requires depth 1"):
        QwenMTPSpeculativeEngine(target, depth=2, ngram_first=True)
    for invalid in (1, 5, True, "4"):
        with pytest.raises(ValueError, match="tree width must be 0 or"):
            QwenMTPSpeculativeEngine(target, native_tree_width=invalid)
    with pytest.raises(ValueError, match="trees currently require depth 1"):
        QwenMTPSpeculativeEngine(
            target, depth=2, native_tree_width=2)
    with pytest.raises(ValueError, match="cannot be combined"):
        QwenMTPSpeculativeEngine(
            target, ngram_first=True, native_tree_width=2)
    for invalid in (0, 5, True, "4"):
        with pytest.raises(ValueError, match="draft width must be in"):
            QwenMTPSpeculativeEngine(
                target, ngram_first=True,
                ngram_max_draft_tokens=invalid)
    with pytest.raises(TypeError, match="grammar_aware_draft must be bool"):
        QwenMTPSpeculativeEngine(target, grammar_aware_draft=1)


def test_server_q_policy_is_strict_and_part_of_engine_cache_identity():
    from runtime.server import EngineManager, RequestValidationError

    with patch.dict(os.environ, {"VMODEL_QWEN_MTP_Q_POLICY": "mystery"}):
        with pytest.raises(RequestValidationError, match="flat, temperature"):
            EngineManager().get(Path("/tmp/not-opened"), "fast")
    with patch.dict(os.environ, {"VMODEL_QWEN_MTP_NGRAM_FIRST": "auto"}):
        with pytest.raises(RequestValidationError, match="must be 0 or 1"):
            EngineManager().get(Path("/tmp/not-opened"), "fast")
    with patch.dict(os.environ, {
        "VMODEL_QWEN_MTP_GRAMMAR_AWARE_DRAFT": "auto",
    }):
        with pytest.raises(RequestValidationError, match="must be 0 or 1"):
            EngineManager().get(Path("/tmp/not-opened"), "fast")
    with patch.dict(os.environ, {
        "VMODEL_QWEN_MTP_NGRAM_FIRST": "1",
        "VMODEL_QWEN_MTP_DEPTH": "2",
    }):
        with pytest.raises(RequestValidationError, match="requires.*DEPTH=1"):
            EngineManager().get(Path("/tmp/not-opened"), "fast")
    with patch.dict(os.environ, {"VMODEL_QWEN_MTP_TREE_WIDTH": "1"}):
        with pytest.raises(RequestValidationError, match="must be 0 or"):
            EngineManager().get(Path("/tmp/not-opened"), "fast")
    with patch.dict(os.environ, {
        "VMODEL_QWEN35_SERIAL_VERIFY_EXACT_PAGE_ADMISSION": "yes",
    }):
        with pytest.raises(RequestValidationError, match="must be 0 or 1"):
            EngineManager().get(Path("/tmp/not-opened"), "fast")
    with patch.dict(os.environ, {
        "VMODEL_QWEN_MTP_TREE_WIDTH": "2",
        "VMODEL_QWEN_MTP_DEPTH": "2",
    }):
        with pytest.raises(RequestValidationError, match="requires.*DEPTH=1"):
            EngineManager().get(Path("/tmp/not-opened"), "fast")
    with patch.dict(os.environ, {
        "VMODEL_QWEN_MTP_TREE_WIDTH": "2",
        "VMODEL_QWEN_MTP_NGRAM_FIRST": "1",
    }):
        with pytest.raises(RequestValidationError, match="cannot be combined"):
            EngineManager().get(Path("/tmp/not-opened"), "fast")

    made = []

    class FakeEngine:
        def __init__(self, _path, rc):
            self.rc = rc
            self.closes = 0
            self.store = SimpleNamespace(names_with_prefix=lambda _prefix: [])
            made.append(self)

        def close(self):
            self.closes += 1

    cfg = SimpleNamespace(
        model_type="qwen3_5", tie_word_embeddings=False,
        index_topk=0, vision_config=None, num_experts=0,
        hidden_size=5120, intermediate_size=17408,
        num_hidden_layers=64, num_attention_heads=24,
        num_key_value_heads=4, head_dim=256, vocab_size=248320,
        attention_bias=False, layer_types=(
            "linear_attention", "linear_attention", "linear_attention",
            "full_attention") * 16,
    )
    manager = EngineManager()
    env = {
        "VMODEL_QWEN_MTP_SPECULATIVE": "0",
        "VMODEL_QWEN_MTP_Q_POLICY": "rank",
        "VMODEL_QWEN_MTP_Q_PARAMETER": "1",
        "VMODEL_QWEN_MTP_STOCHASTIC_DRAFT_TOP_K": "8",
        "VMODEL_QWEN_MTP_DEPTH": "1",
        "VMODEL_QWEN_MTP_NGRAM_FIRST": "0",
        "VMODEL_QWEN_MTP_TREE_WIDTH": "0",
        "VMODEL_QWEN_MTP_GRAMMAR_AWARE_DRAFT": "0",
        "VMODEL_QWEN35_SERIAL_VERIFY_EXACT_PAGE_ADMISSION": "0",
    }
    with patch.dict(os.environ, env), \
         patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir",
               side_effect=lambda path: path), \
         patch("runtime.engine.StreamingEngine", FakeEngine), \
         patch("runtime.server.psutil.virtual_memory",
               return_value=SimpleNamespace(available=8_000_000_000)):
        first = manager.get(Path("/tmp/fake-qwen-q-policy"), "fast")
        os.environ["VMODEL_QWEN_MTP_Q_PARAMETER"] = "2"
        second = manager.get(Path("/tmp/fake-qwen-q-policy"), "fast")
        os.environ["VMODEL_QWEN_MTP_NGRAM_FIRST"] = "1"
        third = manager.get(Path("/tmp/fake-qwen-q-policy"), "fast")
        os.environ["VMODEL_QWEN_MTP_NGRAM_FIRST"] = "0"
        os.environ["VMODEL_QWEN_MTP_TREE_WIDTH"] = "4"
        fourth = manager.get(Path("/tmp/fake-qwen-q-policy"), "fast")
        os.environ[
            "VMODEL_QWEN35_SERIAL_VERIFY_EXACT_PAGE_ADMISSION"
        ] = "1"
        fifth = manager.get(Path("/tmp/fake-qwen-q-policy"), "fast")
        os.environ["VMODEL_QWEN_MTP_GRAMMAR_AWARE_DRAFT"] = "1"
        sixth = manager.get(Path("/tmp/fake-qwen-q-policy"), "fast")

    assert first is made[0]
    assert second is made[1]
    assert third is made[2]
    assert fourth is made[3]
    assert fifth is made[4]
    assert sixth is made[5]
    assert first.closes == 1
    assert second.closes == 1
    assert third.closes == 1
    assert fourth.closes == 1
    assert fifth.closes == 1


def test_server_wires_typed_q_policy_and_explicit_depth_two():
    from runtime.server import EngineManager

    captured = []

    class FakeStore(_Store):
        def names_with_prefix(self, prefix):
            return ["mtp.fc.weight"] if prefix == "mtp." else []

    class FakeEngine:
        def __init__(self, _path, rc):
            self.rc = rc
            self.store = FakeStore()
            self.cache = SimpleNamespace(max_bytes=2_000_000_000)
            self.governor = None

        def close(self):
            return None

    class FakeMTP:
        def __init__(self, target, **kwargs):
            self.target = target
            self.kwargs = kwargs
            captured.append(self)

        def close(self):
            self.target.close()

    cfg = SimpleNamespace(
        model_type="qwen3_5", tie_word_embeddings=False,
        index_topk=0, vision_config=None, num_experts=0,
        hidden_size=5120, intermediate_size=17408,
        num_hidden_layers=64, num_attention_heads=24,
        num_key_value_heads=4, head_dim=256, vocab_size=248320,
        attention_bias=False, layer_types=(
            "linear_attention", "linear_attention", "linear_attention",
            "full_attention") * 16,
    )
    env = {
        "VMODEL_QWEN_MTP_SPECULATIVE": "1",
        "VMODEL_QWEN_MTP_Q_POLICY": "temperature",
        "VMODEL_QWEN_MTP_Q_PARAMETER": "0.75",
        "VMODEL_QWEN_MTP_STOCHASTIC_DRAFT_TOP_K": "8",
        "VMODEL_QWEN_MTP_DEPTH": "2",
        "VMODEL_QWEN_MTP_TREE_WIDTH": "0",
        "VMODEL_QWEN_MTP_GRAMMAR_AWARE_DRAFT": "1",
    }
    with patch.dict(os.environ, env), \
         patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir",
               side_effect=lambda path: path), \
         patch("runtime.engine.StreamingEngine", FakeEngine), \
         patch("runtime.qwen35_mtp.QwenMTPSpeculativeEngine", FakeMTP), \
         patch("runtime.server._checkpoint_payload_bytes",
               return_value=54_000_000_000), \
         patch("runtime.server.psutil.virtual_memory",
               return_value=SimpleNamespace(available=8_000_000_000)):
        wrapped = EngineManager().get(
            Path("/tmp/fake-qwen-q-policy-wiring"), "fast")

    assert wrapped is captured[0]
    assert wrapped.kwargs["depth"] == 2
    assert wrapped.kwargs["ngram_first"] is False
    assert wrapped.kwargs["grammar_aware_draft"] is True
    policy = wrapped.kwargs["proposal_q_policy"]
    assert policy.name == "temperature-k8-t0.75"

    env.update({
        "VMODEL_QWEN_MTP_DEPTH": "1",
        "VMODEL_QWEN_MTP_NGRAM_FIRST": "1",
    })
    with patch.dict(os.environ, env), \
         patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir",
               side_effect=lambda path: path), \
         patch("runtime.engine.StreamingEngine", FakeEngine), \
         patch("runtime.qwen35_mtp.QwenMTPSpeculativeEngine", FakeMTP), \
         patch("runtime.server._checkpoint_payload_bytes",
               return_value=54_000_000_000), \
         patch("runtime.server.psutil.virtual_memory",
               return_value=SimpleNamespace(available=8_000_000_000)):
        cascaded = EngineManager().get(
            Path("/tmp/fake-qwen-ngram-mtp-wiring"), "fast")

    assert cascaded is captured[1]
    assert cascaded.kwargs["depth"] == 1
    assert cascaded.kwargs["ngram_first"] is True
    assert cascaded.kwargs["grammar_aware_draft"] is True

    env.update({
        "VMODEL_QWEN_MTP_NGRAM_FIRST": "0",
        "VMODEL_QWEN_MTP_TREE_WIDTH": "4",
    })
    with patch.dict(os.environ, env), \
         patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir",
               side_effect=lambda path: path), \
         patch("runtime.engine.StreamingEngine", FakeEngine), \
         patch("runtime.qwen35_mtp.QwenMTPSpeculativeEngine", FakeMTP), \
         patch("runtime.server._checkpoint_payload_bytes",
               return_value=54_000_000_000), \
         patch("runtime.server.psutil.virtual_memory",
               return_value=SimpleNamespace(available=8_000_000_000)):
        tree = EngineManager().get(
            Path("/tmp/fake-qwen-native-mtp-tree-wiring"), "fast")

    assert tree is captured[2]
    assert tree.kwargs["depth"] == 1
    assert tree.kwargs["ngram_first"] is False
    assert tree.kwargs["native_tree_width"] == 4
    assert tree.kwargs["grammar_aware_draft"] is True
