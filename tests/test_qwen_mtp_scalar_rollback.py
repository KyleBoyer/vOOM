"""Flat MTP ownership/rollback gates with real scalar recurrent arrays.

Logits, target weights and proposal selection are mocked; the actual Qwen
scalar recurrence, capture and prefix replay are not. No model-quality claim.
"""
from __future__ import annotations

import gc
import os
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import pytest

from runtime.kda_state import KDAStateCache
from runtime.qwen35 import _sequential_gated_delta_rule
from runtime.qwen35_mtp import QwenMTPSpeculativeEngine, _capture_qwen_serial_factors
from runtime.sampler import SamplingParams
from tests.test_qwen35_mtp_k2 import _WideTarget, _WideRecurrentDrafter


class _FactorTarget(_WideTarget):
    def __init__(self, accepted_prefix, *, eos=()):
        super().__init__(accepted_prefix, eos=eos)
        self.cfg.model_type = "qwen3_5"
        self.rc = SimpleNamespace(native_fused_deltanet_decode=False)
        self.factors = None
        self.factor_refs = []
        self.fail_verify = False
        self.omit_layer = False
        self.factor_peak_observations = 0

    def _note_true_peak(self):
        self.factor_peak_observations += 1
        self._true_peak_metal_bytes = max(
            self._true_peak_metal_bytes, mx.get_peak_memory())

    def generate(self, *args, **kwargs):
        result = super().generate(*args, **kwargs)
        state = KDAStateCache(2)
        state.set_state(0, mx.full((1, 2, 3, 3), 0.125))
        state.set_conv_history(0, (mx.zeros((1, 3, 8), dtype=mx.bfloat16),))
        self.last_kv.kda_cache = state
        return result

    def forward_tokens_serial_positions(self, tokens, kv, *,
                                       capture_kda_endpoints=False,
                                       capture_kda_factors=False):
        assert capture_kda_endpoints != capture_kda_factors
        source = kv.kda_cache
        # Reuse the independent deterministic logit/hidden/KV-length oracle.
        logits = super().forward_tokens_serial_positions(
            tokens, kv, capture_kda_endpoints=True)
        kv.kda_cache = source
        self.endpoints = {}
        if capture_kda_factors:
            source.begin_factor_capture()
        for index, token in enumerate(tokens):
            key = mx.full((1, 1, 2, 3), (token + 1) / 32)
            value = mx.full(key.shape, (index + 1) / 7)
            gate = mx.full((1, 1, 2), -0.07 * (index + 1))
            beta = mx.full(gate.shape, 0.31)
            _, state = _sequential_gated_delta_rule(
                key, key, value, beta, gate, source.state(0))
            history = (mx.full((1, 3, 8), token, dtype=mx.bfloat16),)
            mx.eval(state, *history)
            if capture_kda_factors and not self.omit_layer:
                source.capture_factor_step(
                    0, gate=gate[:, 0], key=key[:, 0], value=value[:, 0],
                    beta=beta[:, 0], conv_history=history)
            source.set_state(0, state)
            source.set_conv_history(0, history)
            if capture_kda_endpoints:
                self.endpoints[index + 1] = source.fork()
            if self.fail_verify:
                raise MemoryError("injected-verifier-admission")
        if capture_kda_factors:
            self.factors = source.finish_factor_capture(len(tokens))
            self.factor_refs.append(weakref.ref(self.factors))
        return logits

    def consume_serial_kda_factors(self):
        result, self.factors = self.factors, None
        return result


def _engine(target, compact):
    engine = QwenMTPSpeculativeEngine(
        target, max_prompt_tokens=8, min_output_tokens=2,
        plain_warmup_tokens=0, adaptive_stop=False, depth=4,
        compact_kda_rollback=compact)
    engine.drafter = _WideRecurrentDrafter()
    return engine


@pytest.mark.parametrize("accepted", range(5))
@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_compact_matches_dense_every_rejection_and_full_accept(accepted, temperature):
    targets = [_FactorTarget(accepted), _FactorTarget(accepted)]
    results = [_engine(target, compact).generate(
        "x", accepted + 2, sampling=SamplingParams(temperature=temperature, seed=17))
        for target, compact in zip(targets, [False, True])]
    _assert_equal(targets, results)
    stats = results[1]["path_stats"]
    assert stats["qwen_mtp_kda_factor_rounds"] == 1
    assert stats["qwen_mtp_kda_factor_restores"] == int(accepted < 4)


def _assert_equal(targets, results):
    assert results[0]["tokens"] == results[1]["tokens"]
    assert results[0]["text"] == results[1]["text"]
    assert results[0]["termination_reason"] == results[1]["termination_reason"]
    left, right = [target.last_kv for target in targets]
    assert left.layer_lengths() == right.layer_lengths()
    assert bool(mx.array_equal(left.kda_cache.state(0), right.kda_cache.state(0)).item())
    assert bool(mx.array_equal(left.kda_cache.conv_history(0)[0],
                               right.kda_cache.conv_history(0)[0]).item())
    assert right.kda_cache.state(1) is None
    assert bool(mx.array_equal(targets[0]._h_last, targets[1]._h_last).item())
    assert not right.kda_cache.factor_capture_active
    assert targets[1].factors is None
    assert targets[1].endpoint_requests == []
    gc.collect()
    assert all(ref() is None for ref in targets[1].factor_refs)
    stats = results[1]["path_stats"]
    assert stats["qwen_mtp_compact_kda_rollback_enabled"] == 1
    assert stats["qwen_mtp_kda_factor_bytes_peak"] > 0
    assert stats["qwen_mtp_kda_factor_base_bytes_peak"] > 0
    assert stats["qwen_mtp_kda_endpoint_restores"] == 0
    assert targets[1].factor_peak_observations == stats["qwen_mtp_kda_factor_restores"]


@pytest.mark.parametrize("stop_kind", ["eos", "stop", "budget", "grammar"])
@pytest.mark.parametrize("position", [1, 2, 3, 4])
def test_compact_matches_dense_terminal_prefix(stop_kind, position):
    class Constraint:
        completed = False

        def __init__(self):
            self.accepted = []

        def mask_logits(self, logits):
            return logits

        def accept_token(self, token):
            self.accepted.append(token)
            self.completed = len(self.accepted) >= position + 1

    token = _WideTarget.draft_tokens[position - 1]
    targets = [_FactorTarget(4, eos=(token,) if stop_kind == "eos" else ())
               for _ in range(2)]
    results = [_engine(target, compact).generate(
        "x", position + 1 if stop_kind == "budget" else 8,
        stop=[str(token)] if stop_kind == "stop" else None,
        constraint=Constraint() if stop_kind == "grammar" else None)
        for target, compact in zip(targets, [False, True])]
    _assert_equal(targets, results)
    assert results[1]["path_stats"]["qwen_mtp_kda_factor_restores"] == 1


def test_failed_verifier_cancels_capture_without_retry(capsys):
    target = _FactorTarget(4)
    target.fail_verify = True
    with pytest.raises(MemoryError, match="injected-verifier-admission"):
        _engine(target, True).generate("x", 6)
    assert not target.last_kv.kda_cache.factor_capture_active
    assert target.factors is None
    assert len(target.serial_calls) == 1
    diagnostic = capsys.readouterr().out
    assert '[qwen-mtp-scalar-factor-failure]' in diagnostic
    assert 'factor_layers=1 factor_steps=1' in diagnostic


def test_missing_active_layer_factors_fail_closed():
    target = _FactorTarget(4)
    target.omit_layer = True
    with pytest.raises(RuntimeError, match="omitted KDA factors for layer"):
        _engine(target, True).generate("x", 6)
    assert not target.last_kv.kda_cache.factor_capture_active
    assert target.factors is None


def test_failed_reconstruction_still_records_true_transient_peak():
    from runtime.kda_state import KDAFactorWindow

    target = _FactorTarget(0)
    error = MemoryError('injected-reconstruction')
    with patch.object(KDAFactorWindow, 'commit_prefix', side_effect=error):
        with pytest.raises(MemoryError) as raised:
            _engine(target, True).generate('x', 2)
    assert raised.value is error
    assert target.factor_peak_observations == 1
    assert target._true_peak_metal_bytes > 0
    assert not target.last_kv.kda_cache.factor_capture_active
    assert target.factors is None


def test_compact_mode_strict_default_and_identity():
    target = _FactorTarget(4)
    assert not _engine(target, False).compact_kda_rollback
    assert "compact-kda-scalar" in _engine(target, True).mtp_engine_identity
    with pytest.raises(TypeError, match="must be bool"):
        _engine(target, 1)
    target.rc.native_fused_deltanet_decode = True
    with pytest.raises(ValueError, match="plain scalar"):
        _engine(target, True)
    target.rc.native_fused_deltanet_decode = False
    for model_type in ("qwen4_exp", "qwen3_5_moe"):
        target.cfg.model_type = model_type
        with pytest.raises(ValueError, match="dense flat"):
            _engine(target, True)
    target.cfg.model_type = "qwen3_5"
    for options in ({"native_tree_width": 2},
                    {"selective_tree_margin": 1, "prompt_history_tokens": 128}):
        with pytest.raises(ValueError, match="dense flat"):
            QwenMTPSpeculativeEngine(target, compact_kda_rollback=True, **options)


def test_server_compact_switch_is_strict():
    from runtime.server import EngineManager, RequestValidationError

    with patch.dict(os.environ, {"VMODEL_QWEN_MTP_COMPACT_KDA_ROLLBACK": "auto"}):
        with pytest.raises(RequestValidationError, match="must be 0 or 1"):
            EngineManager().get(Path("/tmp/not-opened"), "fast")


def test_preexisting_capture_is_not_cancelled_or_overwritten():
    target = _FactorTarget(4)
    target.generate("x", max_tokens=1)
    target.last_kv.kda_cache.begin_factor_capture()
    with pytest.raises(ValueError, match="idle resident"):
        _capture_qwen_serial_factors(target, [4, 10], target.last_kv)
    assert target.last_kv.kda_cache.factor_capture_active
    target.last_kv.kda_cache.cancel_factor_capture()
