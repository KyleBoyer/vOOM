"""Focused exactness gates for explicit recurrent Qwen MTP chain depths."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import pytest

from runtime.kv_cache import KVCache
from runtime.qwen35_mtp import (
    _QwenMTPBootstrapPrompt,
    ProposalQPolicy, QwenMTPDrafter, QwenMTPSpeculativeEngine)
from runtime.sampler import SamplingParams


def test_mtp_bootstrap_prompt_retains_paged_endpoint_and_identity():
    class Prompt(str):
        pass

    prompt = Prompt("rendered")
    prompt.tool_capsules = (("tool", 1, 2),)
    prompt.cache_namespace = "execution"
    prompt.force_paged_kv = True
    prompt.stable_boundary_tokens = 17
    prompt.rerank_capture_shape = {"tool_count": 2}
    prompt.disable_hot_prompt_kv = False

    bootstrap = _QwenMTPBootstrapPrompt(prompt, (3, 4, 5))

    assert str(bootstrap) == "rendered"
    assert bootstrap.token_ids == (3, 4, 5)
    assert bootstrap.tool_capsules == (("tool", 1, 2),)
    assert bootstrap.cache_namespace == "execution"
    assert bootstrap.force_paged_kv is True
    assert bootstrap.stable_boundary_tokens == 17
    assert bootstrap.rerank_capture_shape == {"tool_count": 2}
    assert bootstrap.disable_hot_prompt_kv is False
    assert bootstrap.retain_paged_kv_after_generate is True


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

    def fork(self):
        return self


class _TargetKV:
    def __init__(self):
        self.offset = 3
        self.lengths = [3, 1]
        self.kda_cache = _Endpoint(0)
        self.released = False

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

    def release(self):
        self.released = True


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
        self.bootstrap_prompt = None
        self.bootstrap_kv = None
        self.endpoint_requests = []
        self.serial_calls = []
        self.endpoints = {index: _Endpoint(index) for index in (1, 2, 3)}

    def generate(self, _prompt, max_tokens, **kwargs):
        assert max_tokens == 1
        self.bootstrap_prompt = _prompt
        if kwargs.get("constraint") is not None:
            kwargs["constraint"].accept_token(4)
        self.last_kv = _TargetKV()
        self.bootstrap_kv = self.last_kv
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


class _WideTarget(_Target):
    """Width-five serial oracle for every accepted prefix at depth four."""

    draft_tokens = (10, 11, 12, 13)
    correction_tokens = (5, 6, 7, 9)

    def __init__(self, accepted_prefix, *, eos=()):
        super().__init__(accepted_prefix, eos=eos)
        self.endpoints = {index: _Endpoint(index) for index in range(1, 6)}

    def forward_tokens_serial_positions(
        self, tokens, kv, *, capture_kda_endpoints=False,
    ):
        expected = [4, *self.draft_tokens]
        assert tokens == expected
        assert capture_kda_endpoints
        self.serial_calls.append(list(tokens))
        width = len(expected)
        kv.offset += width
        kv.lengths[0] += width
        kv.kda_cache = self.endpoints[width]
        hidden_rows = (40.0, 100.0, 110.0, 120.0, 130.0)
        self._h_window = mx.array(
            [[[value] for value in hidden_rows]])
        self._h_last = self._h_window[:, -1:, :]
        target_tokens = [
            (
                proposal if self.accepted_prefix > index
                else self.correction_tokens[index]
            )
            for index, proposal in enumerate(self.draft_tokens)
        ]
        target_tokens.append(8)
        logits = mx.full((width, 16), -100.0)
        for row, token in enumerate(target_tokens):
            logits = logits.at[row, token].add(200.0)
        return logits


class _WideRecurrentDrafter(_RecurrentDrafter):
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
        proposal = _WideTarget.draft_tokens[step]
        logits = mx.full((16,), -100.0).at[proposal].add(200.0)
        return logits, mx.array([[[1000.0 + step]]])


class _ExternalARDrafter:
    proposal_source = "A"
    request_weight_representation = "resident-ar-mxfp4"
    identity = "fake-qwen-ar"

    def __init__(self):
        self.calls = []
        self.begin_requests = []
        self.begin_rounds = []
        self.commits = []
        self.ended = 0

    def begin_request(self, ids):
        self.begin_requests.append(list(ids))

    def begin_round(self, all_tokens):
        self.begin_rounds.append(list(all_tokens))

    def draft_step(self, hidden, token, _mtp_kv, offset, _weights=None):
        step = len(self.calls)
        self.calls.append({"token": int(token), "offset": int(offset)})
        proposal = _WideTarget.draft_tokens[step]
        logits = mx.full((16,), -100.0).at[proposal].add(200.0)
        return logits, hidden

    def commit_target_inputs(self, tokens):
        self.commits.append(list(tokens))

    def telemetry_snapshot(self):
        return {
            "identity": self.identity,
            "proposal_steps": len(self.calls),
            "commit_replay_steps": sum(map(len, self.commits)),
            "round_sync_steps": 0,
            "peak_cache_bytes": 1234,
        }

    def end_request(self):
        self.ended += 1


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


def test_prompt_history_is_primed_once_and_provisional_rows_are_rolled_back():
    class HistoryTarget(_Target):
        def generate(self, prompt, max_tokens, **kwargs):
            result = super().generate(prompt, max_tokens, **kwargs)
            self._h_window = mx.array([[[10.0], [20.0], [30.0]]])
            self._h_last = self._h_window[:, -1:, :]
            return result

    class HistoryDrafter(_RecurrentDrafter):
        proposal_source = "M"
        request_weight_representation = "demand-cache"

        def __init__(self):
            super().__init__()
            self.history_appends = []

        def append_committed_history(
            self, hidden, tokens, mtp_kv, offset, _weights=None,
            *, tile_size=128,
        ):
            self.history_appends.append({
                "hidden": hidden.tolist(),
                "tokens": list(tokens),
                "offset": int(offset),
            })
            rows = len(tokens)
            key = mx.arange(rows, dtype=mx.float32).reshape(1, 1, rows, 1)
            mtp_kv.update(0, key, key)
            return {
                "rows": rows,
                "tiles": 1,
                "seconds": 0.001,
                "kv_bytes": mtp_kv.nbytes(),
            }

    target = HistoryTarget(accepted_prefix=2)
    drafter = HistoryDrafter()
    engine = QwenMTPSpeculativeEngine(
        target,
        max_prompt_tokens=8,
        min_output_tokens=2,
        adaptive_stop=False,
        plain_warmup_tokens=0,
        depth=2,
        prompt_history_tokens=2,
        drafter=drafter,
    )

    result = engine.generate("x", 4)
    stats = result["path_stats"]

    assert result["tokens"] == [4, 10, 11, 8]
    assert drafter.history_appends == [{
        "hidden": [[[10.0], [20.0]]],
        "tokens": [2, 3],
        "offset": 0,
    }]
    # The two prompt rows remain. Both provisional draft rows were trimmed;
    # no target row is queued because this request ended in the same round.
    assert drafter.mtp_kv.layer_lengths() == (2,)
    assert stats["qwen_mtp_prompt_history_captured_rows"] == 2
    assert stats["qwen_mtp_prompt_endpoint_detach_calls"] == 1
    assert stats["qwen_mtp_prompt_endpoint_source_rows"] == 3
    assert stats["qwen_mtp_prompt_endpoint_source_bytes"] == 12
    assert stats["qwen_mtp_prompt_endpoint_retained_bytes"] == 4
    assert stats["qwen_mtp_committed_history_flushed_rows"] == 2
    assert stats["qwen_mtp_committed_history_pending_rows"] == 0
    assert stats["qwen_mtp_draft_kv_rollbacks"] == 1
    assert "committed-history-k2" in stats["qwen_mtp_engine_identity"]

    short_target = HistoryTarget(accepted_prefix=2)
    short_drafter = HistoryDrafter()
    short_engine = QwenMTPSpeculativeEngine(
        short_target,
        max_prompt_tokens=8,
        min_output_tokens=2,
        adaptive_stop=False,
        plain_warmup_tokens=0,
        depth=2,
        prompt_history_tokens=2,
        prompt_history_min_prompt_tokens=4,
        drafter=short_drafter,
    )
    short_stats = short_engine.generate("x", 4)["path_stats"]
    assert short_drafter.history_appends == []
    assert short_stats["qwen_mtp_prompt_history_enabled"] == 1
    assert short_stats["qwen_mtp_prompt_history_request_active"] == 0
    assert short_stats["qwen_mtp_prompt_history_min_prompt_tokens"] == 4
    assert short_stats["qwen_mtp_prompt_history_skip_reason"] == (
        "below-min-prompt")
    assert short_stats["qwen_mtp_prompt_endpoint_detach_calls"] == 0


def test_prompt_history_flushes_authoritative_target_rows_on_next_round():
    class TwoRoundTarget(_WideTarget):
        def __init__(self):
            super().__init__(accepted_prefix=2)
            self.round = 0

        def generate(self, prompt, max_tokens, **kwargs):
            result = super().generate(prompt, max_tokens, **kwargs)
            self._h_window = mx.array([[[10.0], [20.0], [30.0]]])
            self._h_last = self._h_window[:, -1:, :]
            return result

        def forward_tokens_serial_positions(
            self, tokens, kv, *, capture_kda_endpoints=False,
        ):
            assert capture_kda_endpoints
            expected = ([4, 10, 11], [8, 10, 11])[self.round]
            assert tokens == expected
            hidden = (
                (40.0, 100.0, 110.0),
                (80.0, 200.0, 210.0),
            )[self.round]
            winners = ((10, 11, 8), (6, 7, 8))[self.round]
            self.round += 1
            kv.offset += 3
            kv.lengths[0] += 3
            kv.kda_cache = self.endpoints[3]
            self._h_window = mx.array([[[value] for value in hidden]])
            self._h_last = self._h_window[:, -1:, :]
            logits = mx.full((3, 16), -100.0)
            for row, token in enumerate(winners):
                logits = logits.at[row, token].add(200.0)
            return logits

    class HistoryDrafter(_WideRecurrentDrafter):
        proposal_source = "M"
        request_weight_representation = "demand-cache"

        def __init__(self):
            super().__init__()
            self.history_appends = []

        def append_committed_history(
            self, hidden, tokens, mtp_kv, offset, _weights=None,
            *, tile_size=128,
        ):
            self.history_appends.append(
                (hidden.tolist(), list(tokens), int(offset)))
            rows = len(tokens)
            key = mx.zeros((1, 1, rows, 1))
            mtp_kv.update(0, key, key)
            return {
                "rows": rows, "tiles": 1, "seconds": 0.001,
                "kv_bytes": mtp_kv.nbytes(),
            }

        def draft_step(self, hidden, token, mtp_kv, offset, _weights=None):
            step = len(self.calls) % 2
            self.mtp_kv = mtp_kv
            self.calls.append({
                "hidden": float(hidden.reshape(-1)[0].item()),
                "token": int(token),
                "offset": int(offset),
            })
            key = mx.ones((1, 1, 1, 1))
            mtp_kv.update(0, key, key)
            proposal = (10, 11)[step]
            logits = mx.full((16,), -100.0).at[proposal].add(200.0)
            return logits, mx.array([[[1000.0 + step]]])

    target = TwoRoundTarget()
    drafter = HistoryDrafter()
    engine = QwenMTPSpeculativeEngine(
        target,
        max_prompt_tokens=8,
        min_output_tokens=2,
        adaptive_stop=False,
        plain_warmup_tokens=0,
        depth=2,
        prompt_history_tokens=2,
        drafter=drafter,
    )

    result = engine.generate("x", 5)
    stats = result["path_stats"]

    assert result["tokens"] == [4, 10, 11, 8, 6]
    assert drafter.history_appends == [
        ([[[10.0], [20.0]]], [2, 3], 0),
        ([[[30.0], [40.0], [100.0]]], [4, 10, 11], 2),
    ]
    assert drafter.mtp_kv.layer_lengths() == (5,)
    assert stats["qwen_mtp_committed_history_rows_queued"] == 3
    assert stats["qwen_mtp_committed_history_flushed_rows"] == 5
    assert stats["qwen_mtp_committed_history_flush_calls"] == 2
    assert stats["qwen_mtp_draft_kv_rollbacks"] == 2


def test_native_mtp_phase_head_detaches_only_vocab_row_through_host():
    engine = SimpleNamespace(
        store=SimpleNamespace(
            names_with_prefix=lambda prefix: (
                ["mtp.fc.weight"] if prefix == "mtp." else []),
            mtplx_mtp_sidecar=None,
        ),
        rc=SimpleNamespace(qwen35_serial_verify_suspend_lm_head=True),
        _qwen35_lm_head_suspend_request_active=True,
    )
    drafter = QwenMTPDrafter(engine)
    source = mx.array([1.5, -2.0, 3.25], dtype=mx.bfloat16)

    detached = drafter._detach_head_logits_for_verification(source)

    assert detached.dtype == mx.float32
    assert detached.tolist() == [1.5, -2.0, 3.25]
    assert drafter._head_host_detach_calls == 1
    assert drafter._head_host_detach_bytes == 12
    assert drafter._head_host_detach_s >= 0.0


def test_native_mtp_phase_head_detach_is_neutral_when_disabled():
    engine = SimpleNamespace(
        store=SimpleNamespace(
            names_with_prefix=lambda prefix: (
                ["mtp.fc.weight"] if prefix == "mtp." else []),
            mtplx_mtp_sidecar=None,
        ),
        rc=SimpleNamespace(qwen35_serial_verify_suspend_lm_head=False),
        _qwen35_lm_head_suspend_request_active=False,
    )
    drafter = QwenMTPDrafter(engine)
    source = mx.array([1.0], dtype=mx.bfloat16)

    assert drafter._detach_head_logits_for_verification(source) is source
    assert drafter._head_host_detach_calls == 0


def test_native_mtp_ablation_projects_branches_not_accumulated_residual(
    monkeypatch,
):
    weights = {
        "mtp.pre_fc_norm_embedding.weight": mx.ones((2,)),
        "mtp.pre_fc_norm_hidden.weight": mx.ones((2,)),
        "mtp.fc.weight": mx.ones((2, 4)),
        "mtp.layers.0.input_layernorm.weight": mx.ones((2,)),
        "mtp.layers.0.post_attention_layernorm.weight": mx.ones((2,)),
        "mtp.norm.weight": mx.ones((2,)),
    }

    def fake_attention(_h, _w, _prefix, _cfg, kv, _layer, _offset):
        item = mx.ones((1, 1, 1, 1))
        kv.update(0, item, item)
        return mx.array([[[3.0, 4.0]]])

    monkeypatch.setattr(
        "runtime.qwen35_mtp.qwen35_rms_norm", lambda value, _w, _eps: value)
    monkeypatch.setattr(
        "runtime.qwen35_mtp.quant.matmul",
        lambda _value, _weight: mx.array([[[1.0, 2.0]]]),
    )
    monkeypatch.setattr("runtime.qwen35_mtp._full_attention", fake_attention)
    monkeypatch.setattr(
        "runtime.qwen35_mtp._swiglu",
        lambda _h, _w, _prefix: mx.array([[[5.0, 6.0]]]),
    )
    monkeypatch.setattr(
        "runtime.qwen35_mtp.final_logits",
        lambda hidden, _norm, _head, _eps: hidden.reshape(-1),
    )
    engine = SimpleNamespace(
        store=SimpleNamespace(names_with_prefix=lambda prefix: (
            list(weights) if prefix == "mtp." else [])),
        cache=SimpleNamespace(get=lambda _key, _names: weights),
        cfg=SimpleNamespace(
            num_experts=0, rms_norm_eps=0.0, vocab_size=2, hidden_size=2),
        _embed=lambda _tokens: mx.zeros((1, 1, 2)),
        _lm_head_weight=lambda: mx.ones((2, 2)),
    )
    drafter = QwenMTPDrafter(engine)
    drafter.set_ablation(
        [1.0, 0.0], strength=1.0, fingerprint="a" * 64)

    logits, hidden = drafter.draft_step(
        mx.zeros((1, 1, 2)), 7, KVCache(1), 2, weights)

    # Attention [3,4] -> [0,4], residual [1,2] -> [1,6].
    # MLP [5,6] -> [0,6], final hidden -> [1,12]. The accumulated
    # residual's first component survives, proving only writers are projected.
    assert hidden.tolist() == [[[1.0, 12.0]]]
    assert logits.tolist() == [1.0, 12.0]


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


def test_paged_bootstrap_restores_phase_budget_then_releases_endpoint():
    class Governor:
        shrinks = 0

        def __init__(self):
            self.calls = []

        def restore_phase_budget(
            self, target_max, *, starting_pressure_shrinks, reason,
        ):
            self.calls.append(
                (target_max, starting_pressure_shrinks, reason))
            return 2_100_000_000

        def request_peak(self):
            return 0

    target = _Target(0)
    target.cache.max_bytes = 2_200_000_000
    target.rc = SimpleNamespace(
        max_kv_mb=768, release_paged_kv_after_generate=True)
    target.governor = Governor()
    target.released_kv = None

    def release_kv(kv):
        target.released_kv = kv
        kv.release()

    target._release_kv = release_kv
    engine = QwenMTPSpeculativeEngine(
        target,
        max_prompt_tokens=8,
        min_output_tokens=2,
        plain_warmup_tokens=0,
        adaptive_stop=False,
        depth=2,
    )
    engine.drafter = _RecurrentDrafter()

    result = engine.generate("x", 2)

    assert target.governor.calls == [(
        2_200_000_000, 0, "qwen-mtp-paged-bootstrap")]
    assert target.bootstrap_prompt.retain_paged_kv_after_generate is True
    assert target.last_kv is None
    assert target.released_kv is target.bootstrap_kv
    assert target.bootstrap_kv.released is True
    assert result["path_stats"][
        "qwen_mtp_paged_bootstrap_budget_restored_bytes"] == 2_100_000_000


def test_paged_wrapper_publishes_complete_reuse_and_spill_telemetry():
    target = _Target(0)
    target.rc = SimpleNamespace(
        max_kv_mb=768, release_paged_kv_after_generate=False)
    engine = QwenMTPSpeculativeEngine(
        target,
        max_prompt_tokens=8,
        min_output_tokens=2,
        plain_warmup_tokens=0,
        adaptive_stop=False,
        depth=2,
    )
    engine.drafter = _RecurrentDrafter()
    original_generate = target.generate

    def generate(*args, **kwargs):
        result = original_generate(*args, **kwargs)
        target.last_kv.stats = SimpleNamespace(
            spills=7,
            reloads=5,
            spill_s=0.7,
            reload_s=0.5,
            spill_bytes_raw=700,
            spill_bytes_compressed=350,
        )
        return result

    target.generate = generate
    stats = engine.generate("x", 2)["path_stats"]

    assert stats["paged_kv_spills"] == 7
    assert stats["paged_kv_reloads"] == 5


def test_k2_greedy_rank_capture_identifies_rescuable_second_choice():
    class _RankedDrafter(_RecurrentDrafter):
        def draft_step(self, hidden, token, mtp_kv, offset, _weights=None):
            step = len(self.calls)
            self.mtp_kv = mtp_kv
            self.calls.append({
                "hidden": float(hidden.reshape(-1)[0].item()),
                "token": int(token),
                "offset": int(offset),
            })
            item = mx.full((1, 1, 1, 1), float(step + 1))
            mtp_kv.update(0, item, item)
            primary, secondary = ((10, 6), (11, 7))[step]
            logits = mx.full((16,), -100.0)
            logits = logits.at[primary].add(300.0)
            logits = logits.at[secondary].add(200.0)
            return logits, mx.array([[[1000.0 + step]]])

    target = _Target(1)
    engine = QwenMTPSpeculativeEngine(
        target,
        max_prompt_tokens=8,
        min_output_tokens=2,
        plain_warmup_tokens=0,
        adaptive_stop=False,
        depth=2,
        proposal_replay_top_k=3,
    )
    engine.drafter = _RankedDrafter()

    result = engine.generate("x", 3)

    assert result["tokens"] == [4, 10, 7]
    stats = result["path_stats"]
    assert stats["qwen_mtp_greedy_target_rank_counts_by_step"] == [
        [0, 1, 0, 0],
        [0, 0, 1, 0],
    ]
    assert stats["qwen_mtp_greedy_rescuable_rejections_by_step"] == [0, 1]
    assert stats["qwen_mtp_greedy_draft_margin_thresholds"] == [
        0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0,
    ]
    margin_counts = stats["qwen_mtp_greedy_margin_rank_counts_by_step"]
    assert margin_counts[0][-1] == [0, 1, 0, 0]
    assert margin_counts[1][-1] == [0, 0, 1, 0]
    assert sum(
        sum(bucket) for step in margin_counts for bucket in step
    ) == 2
    assert stats["qwen_mtp_greedy_round_confidence_records"] == [{
        "margin_buckets": [7, 7],
        "target_ranks": [1, 2],
        "accepted_prefix": 1,
        "rejected": 1,
    }]
    assert stats["qwen_mtp_proposal_q_replay"] == []


@pytest.mark.parametrize("accepted_prefix", range(5))
def test_k4_every_accepted_prefix_matches_ordinary_state_oracle(
    accepted_prefix,
):
    target = _WideTarget(accepted_prefix)
    engine = QwenMTPSpeculativeEngine(
        target,
        max_prompt_tokens=8,
        min_output_tokens=2,
        plain_warmup_tokens=0,
        adaptive_stop=False,
        depth=4,
        ngram_first=True,
    )
    drafter = _WideRecurrentDrafter()
    engine.drafter = drafter

    result = engine.generate("x", accepted_prefix + 2)

    accepted_tokens = list(_WideTarget.draft_tokens[:accepted_prefix])
    final_token = (
        8 if accepted_prefix == 4
        else _WideTarget.correction_tokens[accepted_prefix]
    )
    expected_tokens = [4, *accepted_tokens, final_token]
    expected_fed = accepted_prefix + 1
    expected_mtp_length = min(4, expected_fed)
    assert result["tokens"] == expected_tokens
    assert target.last_kv.offset == 3 + expected_fed
    assert target.last_kv.lengths == [3 + expected_fed, 1]
    assert target.last_kv.kda_cache is target.endpoints[expected_fed]
    assert float(target._h_last.reshape(-1)[0].item()) == {
        1: 40.0,
        2: 100.0,
        3: 110.0,
        4: 120.0,
        5: 130.0,
    }[expected_fed]
    assert target.endpoint_requests == [
        None if expected_fed == 5 else expected_fed]
    assert drafter.mtp_kv.layer_lengths() == (expected_mtp_length,)
    assert [call["token"] for call in drafter.calls] == [4, 10, 11, 12]
    assert [call["offset"] for call in drafter.calls] == [2, 3, 4, 5]
    stats = result["path_stats"]
    assert stats["qwen_mtp_depth"] == 4
    assert stats["qwen_mtp_ngram_first_attempts"] == 1
    assert stats["qwen_mtp_ngram_first_matches"] == 0
    assert stats["qwen_mtp_proposal_sources"] == "M"
    assert stats["qwen_mtp_verify_width"] == 5
    assert stats["qwen_mtp_accepted"] == accepted_prefix
    assert stats["qwen_mtp_accepted_by_step"] == [
        int(index < accepted_prefix) for index in range(4)]
    assert stats["qwen_mtp_verified_by_step"] == [
        int(index <= accepted_prefix) for index in range(4)]
    assert stats["qwen_mtp_target_prefix_rollbacks"] == int(
        expected_fed < 5)
    assert stats["qwen_mtp_draft_kv_rollbacks"] == int(
        expected_mtp_length < 4)


@pytest.mark.parametrize("accepted_prefix", range(5))
def test_external_ar_draft_commits_only_authoritative_target_prefix(
    accepted_prefix,
):
    target = _WideTarget(accepted_prefix)
    drafter = _ExternalARDrafter()
    engine = QwenMTPSpeculativeEngine(
        target,
        max_prompt_tokens=8,
        min_output_tokens=2,
        plain_warmup_tokens=0,
        adaptive_stop=False,
        depth=4,
        drafter=drafter,
    )

    result = engine.generate("x", accepted_prefix + 2)

    accepted_tokens = list(_WideTarget.draft_tokens[:accepted_prefix])
    final_token = (
        8 if accepted_prefix == 4
        else _WideTarget.correction_tokens[accepted_prefix]
    )
    assert result["tokens"] == [4, *accepted_tokens, final_token]
    assert drafter.begin_requests == [[1, 2, 3]]
    assert drafter.begin_rounds == [[1, 2, 3, 4]]
    assert drafter.commits == [[4, *accepted_tokens]]
    assert drafter.ended == 1
    assert [call["token"] for call in drafter.calls] == [4, 10, 11, 12]
    # A resident AR draft owns no native-MTP attention cache.  Target recurrent
    # rollback remains identical and the empty compatibility cache never grows.
    assert target.last_kv.offset == 3 + accepted_prefix + 1
    stats = result["path_stats"]
    assert stats["qwen_mtp_proposal_sources"] == "A"
    assert stats["qwen_mtp_proposal_weight_representation"] == (
        "resident-ar-mxfp4")
    assert stats["qwen_mtp_native_draft_proposed"] == 0
    assert stats["qwen_mtp_native_draft_accepted"] == 0
    assert stats["qwen_mtp_native_draft_rejected"] == 0
    assert stats["qwen_mtp_ar_draft_proposed"] == 4
    assert stats["qwen_mtp_ar_draft_accepted"] == accepted_prefix
    assert stats["qwen_mtp_ar_draft_rejected"] == int(accepted_prefix < 4)
    assert stats["qwen_mtp_ar_draft_proposal_steps"] == 4
    assert stats["qwen_mtp_ar_draft_commit_replay_steps"] == (
        accepted_prefix + 1)
    assert stats["qwen_mtp_ar_draft_peak_cache_bytes"] == 1234
    assert stats["qwen_mtp_ar_draft_identity"] == "fake-qwen-ar"


def test_staged_ar_weights_are_suspended_during_authoritative_target_sweep():
    events = []

    class _Cache:
        pinned_bytes = 17
        total_bytes = 0
        max_bytes = 1
        stats = SimpleNamespace(
            hits=0, misses=0, evictions=0, bytes_read=0)

        def trim_to(self, target):
            events.append(("trim", target))
            return 0

    class _Prefetcher:
        paused = False

        def pause_and_wait_idle(self):
            events.append("prefetch-idle")
            self.paused = True

    class _StagedDrafter(_ExternalARDrafter):
        staged_load = True

        def __init__(self):
            super().__init__()
            self.loaded = False
            self.suspends = 0

        def ensure_loaded(self):
            events.append("draft-load")
            self.loaded = True

        def begin_request(self, ids):
            assert self.loaded
            super().begin_request(ids)

        def draft_step(self, *args, **kwargs):
            assert self.loaded
            return super().draft_step(*args, **kwargs)

        def suspend_for_target_verification(self):
            assert self.loaded
            events.append("draft-suspend")
            self.loaded = False
            self.suspends += 1
            return {"suspended": 1}

        def telemetry_snapshot(self):
            result = super().telemetry_snapshot()
            result.update({
                "verification_suspends": self.suspends,
                "verification_suspend_s": 0.25,
                "verification_released_active_bytes": 123,
            })
            return result

    class _TargetWithSuspensionOracle(_WideTarget):
        def forward_tokens_serial_positions(self, *args, **kwargs):
            assert not drafter.loaded
            events.append("target-verify")
            return super().forward_tokens_serial_positions(*args, **kwargs)

    drafter = _StagedDrafter()
    target = _TargetWithSuspensionOracle(1)
    target.cache = _Cache()
    target.prefetcher = _Prefetcher()
    engine = QwenMTPSpeculativeEngine(
        target,
        max_prompt_tokens=8,
        min_output_tokens=2,
        plain_warmup_tokens=0,
        adaptive_stop=False,
        depth=4,
        drafter=drafter,
    )

    result = engine.generate("x", 3)

    assert result["tokens"] == [4, 10, 6]
    assert events.index("draft-suspend") < events.index("target-verify")
    # Terminal completion intentionally avoids reloading proposal weights just
    # to construct state that end_request immediately discards.
    assert events.count("draft-load") == 2
    assert drafter.commits == []
    stats = result["path_stats"]
    assert stats["qwen_mtp_ar_draft_verification_suspends"] == 1
    assert stats["qwen_mtp_ar_draft_verification_suspend_s"] == 0.25
    assert (
        stats["qwen_mtp_ar_draft_verification_released_active_bytes"] == 123
    )


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


def test_k2_grammar_fork_termination_shortens_target_verification():
    class _TerminalConstraint:
        def __init__(self, accepted=()):
            self.accepted = list(accepted)
            self.completed = False
            self.forks = []

        def fork(self):
            forked = _TerminalConstraint(self.accepted)
            self.forks.append(forked)
            return forked

        def mask_logits(self, logits):
            if self.completed:
                raise RuntimeError("mask requested after terminal token")
            legal = 10 if self.accepted == [4] else 8
            masked = mx.full(logits.shape, -1000.0)
            return masked.at[..., legal].add(2000.0)

        def accept_token(self, token):
            self.accepted.append(int(token))
            if int(token) == 10:
                self.completed = True

    class _ShortTarget(_Target):
        def forward_tokens_serial_positions(
            self, tokens, kv, *, capture_kda_endpoints=False,
        ):
            assert tokens == [4, 10]
            assert capture_kda_endpoints
            self.serial_calls.append(list(tokens))
            kv.offset += 2
            kv.lengths[0] += 2
            kv.kda_cache = self.endpoints[2]
            self._h_window = mx.array([[[40.0], [100.0]]])
            self._h_last = self._h_window[:, -1:, :]
            logits = mx.full((2, 16), -100.0)
            logits = logits.at[0, 10].add(200.0)
            logits = logits.at[1, 8].add(200.0)
            return logits

    target = _ShortTarget(1)
    engine = QwenMTPSpeculativeEngine(
        target,
        max_prompt_tokens=8,
        min_output_tokens=2,
        plain_warmup_tokens=0,
        adaptive_stop=False,
        depth=2,
        grammar_aware_draft=True,
    )
    drafter = _RecurrentDrafter()
    engine.drafter = drafter
    constraint = _TerminalConstraint()

    result = engine.generate("x", 4, constraint=constraint)

    assert result["tokens"] == [4, 10]
    assert result["termination_reason"] == "grammar"
    assert target.serial_calls == [[4, 10]]
    assert len(drafter.calls) == 1
    assert constraint.forks[0].accepted == [4, 10]
    stats = result["path_stats"]
    assert stats["qwen_mtp_verify_width"] == 3
    assert stats["qwen_mtp_max_verify_width_observed"] == 2
    assert stats["qwen_mtp_grammar_masked_draft_tokens"] == 1


def test_k4_grammar_fork_conditions_every_provisional_step():
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
                (4, 10, 11): 12,
                (4, 10, 11, 12): 13,
                (4, 10, 11, 12, 13): 8,
            }[tuple(self.accepted)]
            masked = mx.full(logits.shape, -1000.0)
            return masked.at[..., legal].add(2000.0)

        def accept_token(self, token):
            self.accepted.append(int(token))

    class _RawIllegalWideDrafter(_WideRecurrentDrafter):
        def draft_step(self, hidden, token, mtp_kv, offset, _weights=None):
            logits, next_hidden = super().draft_step(
                hidden, token, mtp_kv, offset, _weights)
            illegal = (15, 14, 15, 14)[len(self.calls) - 1]
            logits = logits.at[illegal].add(300.0)
            return logits, next_hidden

    target = _WideTarget(4)
    engine = QwenMTPSpeculativeEngine(
        target,
        max_prompt_tokens=8,
        min_output_tokens=2,
        plain_warmup_tokens=0,
        adaptive_stop=False,
        depth=4,
        grammar_aware_draft=True,
    )
    drafter = _RawIllegalWideDrafter()
    engine.drafter = drafter
    constraint = _Constraint()

    result = engine.generate("x", 6, constraint=constraint)

    assert result["tokens"] == [4, 10, 11, 12, 13, 8]
    assert constraint.accepted == [4, 10, 11, 12, 13, 8]
    assert len(constraint.forks) == 1
    assert constraint.forks[0].accepted == [4, 10, 11, 12, 13]
    stats = result["path_stats"]
    assert stats["qwen_mtp_accepted_by_step"] == [1, 1, 1, 1]
    assert stats["qwen_mtp_grammar_masked_draft_tokens"] == 4
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
    assert QwenMTPSpeculativeEngine(target, depth=4).depth == 4
    for invalid in (0, 5, True, "2"):
        with pytest.raises(ValueError, match=r"depth must be in \[1, 4\]"):
            QwenMTPSpeculativeEngine(target, depth=invalid)
    cascade = QwenMTPSpeculativeEngine(
        target, depth=4, ngram_first=True)
    assert cascade.depth == 4
    assert cascade.ngram_first is True
    for invalid in (1, 5, True, "4"):
        with pytest.raises(ValueError, match="tree width must be 0 or"):
            QwenMTPSpeculativeEngine(target, native_tree_width=invalid)
    target.cfg.model_type = "qwen3_5"
    assert QwenMTPSpeculativeEngine(
        target, depth=4, native_tree_width=2).native_tree_width == 2
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
    with pytest.raises(TypeError, match="compact_kda_factors must be bool"):
        QwenMTPSpeculativeEngine(target, compact_kda_factors=1)
    for invalid in (-1, 4097, True, "128"):
        with pytest.raises(ValueError, match="history tokens must be in"):
            QwenMTPSpeculativeEngine(
                target, prompt_history_tokens=invalid)
    for invalid in (-1, 1_048_577, True, "4096"):
        with pytest.raises(ValueError, match="minimum prompt tokens"):
            QwenMTPSpeculativeEngine(
                target, prompt_history_min_prompt_tokens=invalid)
    with pytest.raises(ValueError, match="not yet supported with native trees"):
        QwenMTPSpeculativeEngine(
            target, prompt_history_tokens=128, native_tree_width=2)


def test_server_q_policy_is_strict_and_part_of_engine_cache_identity():
    from runtime.server import EngineManager, RequestValidationError

    with patch.dict(os.environ, {"VMODEL_QWEN_MTP_Q_POLICY": "mystery"}):
        with pytest.raises(RequestValidationError, match="flat, temperature"):
            EngineManager().get(Path("/tmp/not-opened"), "fast")
    with patch.dict(os.environ, {"VMODEL_QWEN_MTP_DEPTH": "5"}):
        with pytest.raises(RequestValidationError, match=r"in \[1, 4\]"):
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
        "VMODEL_QWEN_MTP_COMPACT_KDA_FACTORS": "auto",
    }):
        with pytest.raises(RequestValidationError, match="must be 0 or 1"):
            EngineManager().get(Path("/tmp/not-opened"), "fast")
    with patch.dict(os.environ, {"VMODEL_QWEN_MTP_TREE_WIDTH": "1"}):
        with pytest.raises(RequestValidationError, match="must be 0 or"):
            EngineManager().get(Path("/tmp/not-opened"), "fast")
    with patch.dict(os.environ, {
        "VMODEL_QWEN_MTP_PROMPT_HISTORY_TOKENS": "4097",
    }):
        with pytest.raises(RequestValidationError, match=r"in \[0, 4096\]"):
            EngineManager().get(Path("/tmp/not-opened"), "fast")
    with patch.dict(os.environ, {
        "VMODEL_QWEN_MTP_PROMPT_HISTORY_MIN_PROMPT_TOKENS": "1048577",
    }):
        with pytest.raises(RequestValidationError, match=r"in \[0, 1048576\]"):
            EngineManager().get(Path("/tmp/not-opened"), "fast")
    with patch.dict(os.environ, {
        "VMODEL_QWEN35_SERIAL_VERIFY_EXACT_PAGE_ADMISSION": "yes",
    }):
        with pytest.raises(RequestValidationError, match="must be 0 or 1"):
            EngineManager().get(Path("/tmp/not-opened"), "fast")
    with patch.dict(os.environ, {
        "VMODEL_QWEN35_SERIAL_VERIFY_BATCHED_MLP": "yes",
    }):
        with pytest.raises(RequestValidationError, match="must be 0 or 1"):
            EngineManager().get(Path("/tmp/not-opened"), "fast")
    with patch.dict(os.environ, {
        "VMODEL_QWEN35_SERIAL_VERIFY_SUSPEND_LM_HEAD": "yes",
    }):
        with pytest.raises(RequestValidationError, match="must be 0 or 1"):
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
        "VMODEL_QWEN_MTP_PROMPT_HISTORY_TOKENS": "0",
        "VMODEL_QWEN_MTP_PROMPT_HISTORY_MIN_PROMPT_TOKENS": "0",
        "VMODEL_QWEN_MTP_GRAMMAR_AWARE_DRAFT": "0",
        "VMODEL_QWEN35_SERIAL_VERIFY_EXACT_PAGE_ADMISSION": "0",
        "VMODEL_QWEN35_SERIAL_VERIFY_BATCHED_MLP": "0",
        "VMODEL_QWEN35_SERIAL_VERIFY_SUSPEND_LM_HEAD": "0",
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
        os.environ["VMODEL_QWEN35_SERIAL_VERIFY_BATCHED_MLP"] = "1"
        sixth = manager.get(Path("/tmp/fake-qwen-q-policy"), "fast")
        os.environ["VMODEL_QWEN_MTP_GRAMMAR_AWARE_DRAFT"] = "1"
        seventh = manager.get(Path("/tmp/fake-qwen-q-policy"), "fast")
        os.environ[
            "VMODEL_QWEN35_SERIAL_VERIFY_SUSPEND_LM_HEAD"
        ] = "1"
        eighth = manager.get(Path("/tmp/fake-qwen-q-policy"), "fast")
        os.environ["VMODEL_QWEN_MTP_TREE_WIDTH"] = "0"
        os.environ["VMODEL_QWEN_MTP_PROMPT_HISTORY_TOKENS"] = "128"
        ninth = manager.get(Path("/tmp/fake-qwen-q-policy"), "fast")
        os.environ[
            "VMODEL_QWEN_MTP_PROMPT_HISTORY_MIN_PROMPT_TOKENS"
        ] = "4096"
        tenth = manager.get(Path("/tmp/fake-qwen-q-policy"), "fast")

    assert first is made[0]
    assert second is made[1]
    assert third is made[2]
    assert fourth is made[3]
    assert fifth is made[4]
    assert sixth is made[5]
    assert seventh is made[6]
    assert eighth is made[7]
    assert ninth is made[8]
    assert tenth is made[9]
    assert first.closes == 1
    assert second.closes == 1
    assert third.closes == 1
    assert fourth.closes == 1
    assert fifth.closes == 1
    assert sixth.closes == 1
    assert seventh.closes == 1
    assert eighth.closes == 1
    assert ninth.closes == 1


def test_server_wires_typed_q_policy_and_explicit_deep_chain():
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
        "VMODEL_QWEN_MTP_DEPTH": "4",
        "VMODEL_QWEN_MTP_TREE_WIDTH": "0",
        "VMODEL_QWEN_MTP_PROMPT_HISTORY_TOKENS": "128",
        "VMODEL_QWEN_MTP_PROMPT_HISTORY_MIN_PROMPT_TOKENS": "4096",
        "VMODEL_QWEN_MTP_GRAMMAR_AWARE_DRAFT": "1",
        "VMODEL_QWEN_MTP_COMPACT_KDA_FACTORS": "1",
        "VMODEL_QWEN35_SERIAL_VERIFY_BATCHED_MLP": "1",
        "VMODEL_QWEN35_SERIAL_VERIFY_SUSPEND_LM_HEAD": "1",
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
    assert wrapped.kwargs["depth"] == 4
    assert wrapped.kwargs["ngram_first"] is False
    assert wrapped.kwargs["grammar_aware_draft"] is True
    assert wrapped.kwargs["compact_kda_factors"] is True
    assert wrapped.kwargs["prompt_history_tokens"] == 128
    assert wrapped.kwargs["prompt_history_min_prompt_tokens"] == 4096
    assert wrapped.target.rc.qwen35_serial_verify_batched_mlp is True
    assert wrapped.target.rc.qwen35_serial_verify_suspend_lm_head is True
    policy = wrapped.kwargs["proposal_q_policy"]
    assert policy.name == "temperature-k8-t0.75"

    env.update({
        "VMODEL_QWEN_MTP_DEPTH": "4",
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
    assert cascaded.kwargs["depth"] == 4
    assert cascaded.kwargs["ngram_first"] is True
    assert cascaded.kwargs["grammar_aware_draft"] is True

    env.update({
        "VMODEL_QWEN_MTP_DEPTH": "4",
        "VMODEL_QWEN_MTP_NGRAM_FIRST": "0",
        "VMODEL_QWEN_MTP_TREE_WIDTH": "4",
        "VMODEL_QWEN_MTP_PROMPT_HISTORY_TOKENS": "0",
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
    assert tree.kwargs["depth"] == 4
    assert tree.kwargs["ngram_first"] is False
    assert tree.kwargs["native_tree_width"] == 4
    assert tree.kwargs["grammar_aware_draft"] is True
