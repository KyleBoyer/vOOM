"""Pure serving-contract tests for the Qwen native-MTP speculative adapter
(mirrors tests/test_speculative_engine.py's fake-engine pattern)."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest


class _Encoding:
    def __init__(self, ids):
        self.ids = list(ids)


class _Tokenizer:
    def __init__(self, ids=(1, 2, 3)):
        self.ids = list(ids)

    def encode(self, _text):
        return _Encoding(self.ids)

    def decode(self, ids):
        return ",".join(str(value) for value in ids)


class _Store:
    def names_with_prefix(self, prefix):
        return [f"{prefix}fc.weight"] if prefix == "mtp." else []


class _Engine:
    def __init__(self, path: str, *, ids=(1, 2, 3)):
        self._model_dir = path
        self.tokenizer = _Tokenizer(ids)
        self.cfg = SimpleNamespace(num_experts=0, eos_token_ids=())
        self.store = _Store()
        self.effective_max_position_embeddings = 0
        self.rope_profile = "test"
        self.calls = []
        self.releases = 0

    def generate(self, prompt, max_tokens, on_token=None, stop=None,
                 on_progress=None, sampling=None, constraint=None):
        self.calls.append((prompt, max_tokens, on_token, stop, on_progress,
                            sampling, constraint))
        return {"text": "target", "tokens": [4], "path_stats": {}}

    def release_request_state(self):
        self.releases += 1


def test_qwen_mtp_adaptive_break_even_uses_refeed_free_rejection_math():
    from runtime.qwen35_mtp import (
        _adaptive_break_even_probe_rounds,
        _adaptive_mtp_should_disable,
        _adaptive_stochastic_mtp_should_disable,
    )

    assert not _adaptive_mtp_should_disable(2, 0, 3)
    assert _adaptive_mtp_should_disable(3, 0, 3)
    assert not _adaptive_mtp_should_disable(3, 1, 3)
    assert not _adaptive_mtp_should_disable(6, 3, 3)
    assert _adaptive_stochastic_mtp_should_disable(3, 0.12, 3)
    assert not _adaptive_stochastic_mtp_should_disable(3, 0.18, 3)
    assert not _adaptive_stochastic_mtp_should_disable(2, 0.0, 3)
    # The observed Huihui cost shape is about 0.54s of draft work for a
    # 7.2s target sweep: three misses are much too little evidence. The
    # measured break-even window is 14 rounds, bounded to 16 under even
    # cheaper/noisier drafts.
    assert _adaptive_break_even_probe_rounds(0.54, 7.2, 3) == 14
    assert _adaptive_break_even_probe_rounds(0.1, 7.2, 3) == 16
    assert _adaptive_break_even_probe_rounds(4.0, 7.2, 3) == 3
    assert _adaptive_break_even_probe_rounds(0.0, 0.0, 3) == 3
    with pytest.raises(ValueError, match="maximum adaptive probe"):
        _adaptive_break_even_probe_rounds(1.0, 1.0, 4, 3)


def test_qwen_mtp_adaptive_cooldown_reprobes_and_reactivates(monkeypatch):
    """A bad opening region must not permanently hide a later useful draft."""
    import runtime.qwen35_mtp as mtp_module
    from runtime.qwen35_mtp import QwenMTPSpeculativeEngine

    monkeypatch.setattr(
        mtp_module, "_adaptive_break_even_probe_rounds",
        lambda *_args, **_kwargs: 3)

    class KV:
        def __init__(self):
            self.offset = 3
            self.kda_cache = None

        def trim(self, offset):
            self.offset = int(offset)

        def nbytes(self):
            return 0

    class Target(_Engine):
        def __init__(self):
            super().__init__("/models/target")
            self.cfg = SimpleNamespace(num_experts=0, eos_token_ids=())
            self.cache = SimpleNamespace(
                stats=SimpleNamespace(
                    hits=0, misses=0, evictions=0, bytes_read=0),
                total_bytes=0, max_bytes=1)
            for name in (
                "fast_tier_bytes", "archive_bytes", "parallel_tier_fetches",
                "parallel_tier_fast_bytes", "parallel_tier_archive_bytes",
            ):
                setattr(self.store, name, 0)
            self.expert_hits = self.expert_misses = 0
            self.governor = None
            self._layer_transient = self._prefill_layer_transient = 0
            self._decode_layer_transient = self._layer_transient_margin = 0
            self._token_transient = self._true_peak_metal_bytes = 0
            self._request_profiler = None
            self._hot_prompt_slots = []
            self._h_last = mx.zeros((1, 1, 1))
            self._h_window = self._h_last
            self.last_kv = None
            self.accepts = iter((False, False, False, True, False, False))
            self.plain_calls = 0

        def generate(self, _prompt, max_tokens, **_kwargs):
            assert max_tokens == 1
            self.last_kv = KV()
            return {
                "text": "4", "tokens": [4], "prefill_s": 0.0,
                "first_token_s": 0.0, "decode_s": 0.0, "total_s": 0.0,
                "termination_reason": "length", "stop_sequence": None,
                "path_stats": {}, "prompt_tokens": 3,
            }

        def forward_tokens_serial_positions(
            self, tokens, kv, *, capture_kda_endpoints=False,
        ):
            assert tokens == [tokens[0], 9]
            assert not capture_kda_endpoints
            accepted = next(self.accepts)
            kv.offset += 2
            self._h_window = mx.zeros((1, 2, 1))
            self._h_last = self._h_window[:, -1:, :]
            logits = mx.zeros((2, 12))
            logits = logits.at[0, 9 if accepted else 6].add(2)
            return logits.at[1, 7].add(2)

        def forward_tokens(self, tokens, kv):
            assert len(tokens) == 1
            self.plain_calls += 1
            kv.offset += 1
            self._h_window = self._h_last = mx.zeros((1, 1, 1))
            return mx.zeros((1, 12)).at[0, 6].add(2)

    class Drafter:
        @staticmethod
        def draft_token(*_args, **_kwargs):
            return 9

    target = Target()
    engine = QwenMTPSpeculativeEngine(
        target, max_prompt_tokens=8, min_output_tokens=2,
        plain_warmup_tokens=0, adaptive_probe_rounds=3,
        adaptive_reprobe_interval=2)
    engine.drafter = Drafter()
    result = engine.generate("x", 10)
    stats = result["path_stats"]

    assert len(result["tokens"]) == 10
    assert target.plain_calls == 2
    assert stats["qwen_mtp_round_outcomes"] == "RRRARR"
    assert stats["qwen_mtp_adaptive_disable_events"] == 1
    assert stats["qwen_mtp_adaptive_cooldown_sweeps"] == 2
    assert stats["qwen_mtp_adaptive_recovery_probes"] == 1
    assert stats["qwen_mtp_adaptive_reactivations"] == 1
    assert stats["qwen_mtp_target_sweeps_avoided"] == 1


def test_qwen_mtp_adapter_preserves_constraint_on_short_budget_fallback():
    from runtime.qwen35_mtp import QwenMTPSpeculativeEngine

    target = _Engine("/models/target")
    engine = QwenMTPSpeculativeEngine(
        target, max_prompt_tokens=8, min_output_tokens=8)
    constraint = object()

    result = engine.generate("x", 4, constraint=constraint)
    assert result["path_stats"]["qwen_mtp_fallback_reason"] == (
        "short-output-budget")
    assert len(target.calls) == 1
    assert target.calls[0][-1] is constraint


def test_qwen_mtp_stochastic_rejection_uses_target_minus_draft_residual():
    from runtime.qwen35_mtp import _verify_stochastic_mtp_token
    from runtime.sampler import SamplingParams

    for temperature in (0.3, 0.5, 0.7, 1.0):
        sampling = SamplingParams(temperature=temperature, seed=17)
        sampling.seed_rng()
        accepted, token, probabilities = _verify_stochastic_mtp_token(
            0,
            mx.array([1.0, 0.0, 0.0]),
            mx.array([-100.0, 100.0, -100.0]),
            sampling,
            history=[2],
        )
        mx.eval(probabilities)
        assert not accepted
        assert token == 1
        assert probabilities.tolist() == [0.0, 1.0, 0.0]

        accepted, token, probabilities = _verify_stochastic_mtp_token(
            0,
            mx.array([0.5, 0.5, 0.0]),
            mx.array([-100.0, -100.0, 100.0]),
            sampling,
            history=[2],
        )
        assert not accepted
        assert token == 2


def test_qwen_mtp_sparse_draft_survives_disjoint_grammar_support():
    from runtime.qwen35_mtp import _flat_top_k_draft_probabilities
    from runtime.sampler import SamplingParams

    class Constraint:
        def mask_logits(self, logits):
            # Only token 2 is grammar-legal, while the candidate-reranked
            # draft has finite scores only for tokens 0 and 1.
            return mx.array([
                float("-inf"), float("-inf"), logits[2], float("-inf")])

    # Drafter final_logits returns a singleton batch axis; q must still be a
    # flat vocabulary distribution matching the target verifier's row.
    logits = mx.array([[4.0, 3.0, float("-inf"), float("-inf")]])
    q = _flat_top_k_draft_probabilities(
        logits, SamplingParams(temperature=0.7, seed=17), [], 2,
        Constraint())
    mx.eval(q)

    assert bool(mx.allclose(q, mx.array([0.5, 0.5, 0.0, 0.0])))

    uniform = _flat_top_k_draft_probabilities(
        mx.full((4,), float("-inf")),
        SamplingParams(temperature=0.7, seed=17), [], 2)
    mx.eval(uniform)
    assert bool(mx.allclose(uniform, mx.full((4,), 0.25)))


def test_authoritative_target_logits_uses_matching_serial_hidden_position():
    from runtime.qwen35_mtp import _authoritative_target_logits

    calls = []

    class Target:
        _h_window = mx.array([[[10.0], [20.0], [30.0]]])

        def _constraint_logits(self, logits, constraint, hidden=None):
            calls.append(float(hidden.item()))
            return constraint.mask_logits(logits)

    class Constraint:
        def mask_logits(self, logits):
            return logits + 1

    constraint = Constraint()
    logits = mx.array([1.0, 2.0])
    first = _authoritative_target_logits(
        Target(), logits, constraint, 0)
    bonus = _authoritative_target_logits(
        Target(), logits, constraint, 1)
    last = _authoritative_target_logits(
        Target(), logits, constraint, -1)
    mx.eval(first, bonus, last)

    assert calls == [10.0, 20.0, 30.0]
    assert first.tolist() == [2.0, 3.0]
    assert bonus.tolist() == [2.0, 3.0]
    assert last.tolist() == [2.0, 3.0]
    with pytest.raises(RuntimeError, match="outside hidden window"):
        _authoritative_target_logits(Target(), logits, constraint, 3)


def test_qwen_mtp_adapter_falls_back_for_prompt_limit():
    from runtime.qwen35_mtp import QwenMTPSpeculativeEngine

    target = _Engine("/models/target", ids=(1, 2, 3, 4, 5))
    engine = QwenMTPSpeculativeEngine(target, max_prompt_tokens=2)

    result = engine.generate("x", 4)
    assert result["path_stats"]["qwen_mtp_fallback_reason"] == "prompt-limit"
    assert len(target.calls) == 1


def test_qwen_mtp_adapter_falls_back_for_short_output_budget():
    from runtime.qwen35_mtp import QwenMTPSpeculativeEngine

    target = _Engine("/models/target")
    engine = QwenMTPSpeculativeEngine(
        target, max_prompt_tokens=8, min_output_tokens=8)

    result = engine.generate("x", 4)
    assert result["path_stats"]["qwen_mtp_fallback_reason"] == (
        "short-output-budget")
    assert len(target.calls) == 1


def test_qwen_mtp_adapter_rejects_invalid_max_tokens():
    from runtime.qwen35_mtp import QwenMTPSpeculativeEngine

    target = _Engine("/models/target")
    engine = QwenMTPSpeculativeEngine(target, max_prompt_tokens=8)

    for bad in (0, -1, True, 1.5):
        try:
            engine.generate("x", bad)
            raised = False
        except ValueError:
            raised = True
        assert raised, f"max_tokens={bad!r} should have raised"


def test_qwen_mtp_adapter_delegates_target_attributes():
    from runtime.qwen35_mtp import QwenMTPSpeculativeEngine

    target = _Engine("/models/target")
    target.marker = "target-owned"
    engine = QwenMTPSpeculativeEngine(target, max_prompt_tokens=8)

    assert engine.marker == "target-owned"


def test_qwen_mtp_drafter_construction_requires_mtp_weights():
    from runtime.qwen35_mtp import QwenMTPDrafter

    class _NoMTPStore:
        def names_with_prefix(self, prefix):
            return []

    target = _Engine("/models/target")
    target.store = _NoMTPStore()
    try:
        QwenMTPDrafter(target)
        raised = False
    except ValueError:
        raised = True
    assert raised, "QwenMTPDrafter must refuse a checkpoint with no mtp.* weights"


def test_qwen_mtp_accepted_pair_stops_on_first_token_eos():
    """An accepted pair is [verified draft, bonus]. If the draft itself is
    EOS, the unused bonus must never become the next catchup token. The same
    round also gates BF16-sidecar release ordering and exception cleanup."""
    from runtime.qwen35_mtp import QwenMTPSpeculativeEngine

    events = []

    class _KV:
        def __init__(self):
            self.offset = 3
            self.kda_cache = None

        def trim(self, offset):
            self.offset = offset

        def nbytes(self):
            return 0

    class _Target(_Engine):
        def __init__(self):
            super().__init__("/models/target")
            self.cfg.eos_token_ids = (9,)
            self.cache = SimpleNamespace(
                stats=SimpleNamespace(
                    hits=0, misses=0, evictions=0, bytes_read=0),
                total_bytes=0, max_bytes=1)
            self.expert_hits = self.expert_misses = 0
            for name in (
                    "fast_tier_bytes", "archive_bytes",
                    "parallel_tier_fetches", "parallel_tier_fast_bytes",
                    "parallel_tier_archive_bytes"):
                setattr(self.store, name, 0)
            self.governor = None
            self._layer_transient = 0
            self._prefill_layer_transient = 0
            self._decode_layer_transient = 0
            self._layer_transient_margin = 0
            self._token_transient = 0
            self._true_peak_metal_bytes = 0
            self._request_profiler = None
            self._hot_prompt_slots = []
            self._h_last = mx.zeros((1, 1, 1))
            self.last_kv = None

        def generate(self, prompt, max_tokens, **_kwargs):
            assert max_tokens == 1
            self.last_kv = _KV()
            return {
                "text": "4", "tokens": [4], "prefill_s": 0.0,
                "first_token_s": 0.0, "decode_s": 0.0, "total_s": 0.0,
                "termination_reason": "length", "stop_sequence": None,
                "path_stats": {}, "prompt_tokens": 3,
            }

        def forward_tokens(self, tokens, kv):
            assert events[-1] == "release"
            events.append("target")
            assert tokens[0] in (4, 7)
            assert tokens[1] == 9
            kv.offset += 2
            logits = mx.zeros((2, 10))
            logits = logits.at[0, 9].add(1)
            logits = logits.at[1, 7].add(1)
            self._h_last = mx.zeros((1, 1, 1))
            return logits

    class _RoundSidecarDrafter:
        def __init__(self, target, *, fail=False):
            self.target = target
            self.fail = fail
            self.mappings = []

        def prepare_request_weights(self):
            events.append("load")
            self.target.cache.stats.bytes_read += 33
            weights = {"mtp.test": mx.ones((4,), dtype=mx.bfloat16)}
            self.mappings.append(weights)
            return weights

        def draft_token(self, *_args):
            weights = _args[-1]
            assert weights is self.mappings[-1]
            assert weights
            events.append("draft")
            if self.fail:
                raise RuntimeError("synthetic draft failure")
            return 9

        def release_request_weights(self, weights):
            assert weights is self.mappings[-1]
            resident_bytes = sum(value.nbytes for value in weights.values())
            weights.clear()
            events.append("release")
            return {"resident_bytes": resident_bytes, "cache_discarded": 1}

    target = _Target()
    engine = QwenMTPSpeculativeEngine(
        target, max_prompt_tokens=8, min_output_tokens=2,
        plain_warmup_tokens=0, ngram_first=True)
    drafter = _RoundSidecarDrafter(target)
    engine.drafter = drafter

    result = engine.generate("x", 32)

    assert events == ["load", "draft", "release", "target"]
    assert all(mapping == {} for mapping in drafter.mappings)
    assert result["tokens"] == [4, 9]
    assert result["termination_reason"] == "eos"
    assert result["kv_positions"] == 4
    stats = result["path_stats"]
    assert stats["qwen_mtp_bf16_sidecar_round_loads"] == 1
    assert stats["qwen_mtp_bf16_sidecar_round_releases"] == 1
    assert stats["qwen_mtp_bf16_sidecar_read_bytes"] == 33
    assert stats["qwen_mtp_bf16_sidecar_loaded_resident_bytes"] == 8
    assert stats["qwen_mtp_bf16_sidecar_released_resident_bytes"] == 8
    assert stats["qwen_mtp_bf16_sidecar_peak_resident_bytes"] == 8
    assert stats["qwen_mtp_bf16_sidecar_cache_discards"] == 1
    assert stats["qwen_mtp_request_local_sidecar_pin"] == 0
    assert stats["qwen_mtp_request_local_sidecar_bytes"] == 0
    assert stats["qwen_mtp_ngram_first_enabled"] == 1
    assert stats["qwen_mtp_ngram_first_eligible"] == 1
    assert stats["qwen_mtp_ngram_first_attempts"] == 1
    assert stats["qwen_mtp_ngram_first_matches"] == 0
    assert stats["qwen_mtp_ngram_first_native_draft_bypasses"] == 0
    assert stats["qwen_mtp_ngram_first_proposed"] == 0
    assert stats["qwen_mtp_native_draft_proposed"] == 1
    assert stats["qwen_mtp_native_draft_accepted"] == 1
    assert stats["qwen_mtp_native_draft_rejected"] == 0
    assert stats["qwen_mtp_proposal_sources"] == "M"

    # A continuing request reloads a fresh mapping in the next round rather
    # than retaining the released first-round page across verification.
    events.clear()
    reload_target = _Target()
    reload_target.cfg.eos_token_ids = ()
    reload_engine = QwenMTPSpeculativeEngine(
        reload_target, max_prompt_tokens=8, min_output_tokens=2,
        plain_warmup_tokens=0, adaptive_stop=False)
    reload_drafter = _RoundSidecarDrafter(reload_target)
    reload_engine.drafter = reload_drafter
    reload_result = reload_engine.generate("x", 4)
    assert events == [
        "load", "draft", "release", "target",
        "load", "draft", "release", "target",
    ]
    assert len(reload_drafter.mappings) == 2
    assert all(mapping == {} for mapping in reload_drafter.mappings)
    reload_stats = reload_result["path_stats"]
    assert reload_stats["qwen_mtp_bf16_sidecar_round_loads"] == 2
    assert reload_stats["qwen_mtp_bf16_sidecar_round_releases"] == 2
    assert reload_stats["qwen_mtp_bf16_sidecar_read_bytes"] == 66

    # Proposal failure must take the same finally path without ever entering
    # target verification or retaining a Python reference to the sidecar.
    events.clear()
    failing_target = _Target()
    failing_engine = QwenMTPSpeculativeEngine(
        failing_target, max_prompt_tokens=8, min_output_tokens=2,
        plain_warmup_tokens=0)
    failing_drafter = _RoundSidecarDrafter(failing_target, fail=True)
    failing_engine.drafter = failing_drafter
    with pytest.raises(RuntimeError, match="synthetic draft failure"):
        failing_engine.generate("x", 32)
    assert events == ["load", "draft", "release"]
    assert all(mapping == {} for mapping in failing_drafter.mappings)


@pytest.mark.parametrize("accepted", [True, False])
def test_qwen_mtp_ngram_first_keeps_exact_target_state_and_acceptance_telemetry(
    accepted,
):
    """A lookup hit bypasses the sidecar, never the authoritative target.

    Both acceptance and correction retain the exact one-position serial KDA,
    layer-length, hidden-state, and final-token-never-fed endpoint required by
    ordinary Qwen hybrid decode.
    """
    from runtime.qwen35_mtp import QwenMTPSpeculativeEngine

    midpoint = object()

    class _KV:
        def __init__(self):
            self.offset = 4
            self.kda_cache = object()
            self.lengths = [4, 2]

        def layer_lengths(self):
            return tuple(self.lengths)

        def trim_layer_lengths(self, lengths):
            self.lengths = list(lengths)
            self.offset = self.lengths[0]

        def trim(self, offset):
            self.offset = int(offset)

        def nbytes(self):
            return 0

    class _Target(_Engine):
        def __init__(self):
            # [3, 4] is both an earlier prompt bigram and the final suffix
            # after bootstrap emits 4, so real ngram_propose returns 3.
            super().__init__("/models/target", ids=(1, 3, 4, 3))
            self.cfg = SimpleNamespace(num_experts=8, eos_token_ids=())
            self.cache = SimpleNamespace(
                stats=SimpleNamespace(
                    hits=0, misses=0, evictions=0, bytes_read=0),
                total_bytes=0, max_bytes=1)
            self.expert_hits = self.expert_misses = 0
            for name in (
                    "fast_tier_bytes", "archive_bytes",
                    "parallel_tier_fetches", "parallel_tier_fast_bytes",
                    "parallel_tier_archive_bytes"):
                setattr(self.store, name, 0)
            self.governor = None
            self._layer_transient = self._prefill_layer_transient = 0
            self._decode_layer_transient = self._layer_transient_margin = 0
            self._token_transient = self._true_peak_metal_bytes = 0
            self._request_profiler = None
            self._hot_prompt_slots = []
            self._h_last = mx.array([[[30.0]]])
            self._h_window = self._h_last
            self.last_kv = None
            self.endpoint_requests = []

        def generate(self, _prompt, max_tokens, **_kwargs):
            assert max_tokens == 1
            self.last_kv = _KV()
            return {
                "text": "4", "tokens": [4], "prefill_s": 0.0,
                "first_token_s": 0.0, "decode_s": 0.0, "total_s": 0.0,
                "termination_reason": "length", "stop_sequence": None,
                "path_stats": {}, "prompt_tokens": 4,
            }

        def forward_tokens_serial_positions(
            self, tokens, kv, *, capture_kda_endpoints=False,
        ):
            assert tokens == [4, 3]
            assert capture_kda_endpoints
            kv.offset += 2
            kv.lengths = [value + 2 for value in kv.lengths]
            self._h_window = mx.array([[[40.0], [90.0]]])
            self._h_last = self._h_window[:, -1:, :]
            logits = mx.zeros((2, 10))
            logits = logits.at[0, 3 if accepted else 6].add(2)
            return logits.at[1, 7].add(2)

        def consume_serial_kda_endpoint(self, fed_positions):
            self.endpoint_requests.append(fed_positions)
            return midpoint if fed_positions == 1 else None

    class _ForbiddenSidecar:
        @staticmethod
        def prepare_request_weights():
            raise AssertionError("n-gram hit must not load the BF16 sidecar")

        @staticmethod
        def draft_token(*_args, **_kwargs):
            raise AssertionError("n-gram hit must not call native MTP")

    target = _Target()
    engine = QwenMTPSpeculativeEngine(
        target, max_prompt_tokens=8, min_output_tokens=2,
        plain_warmup_tokens=0, adaptive_stop=False, ngram_first=True)
    engine.drafter = _ForbiddenSidecar()

    result = engine.generate("x", 2)
    expected_token = 3 if accepted else 6
    stats = result["path_stats"]

    assert result["tokens"] == [4, expected_token]
    assert result["kv_positions"] == 5
    assert target.last_kv.lengths == [5, 3]
    assert target.last_kv.kda_cache is midpoint
    assert float(target._h_last.item()) == 40.0
    assert target.endpoint_requests == [1]
    assert stats["qwen_mtp_round_outcomes"] == ("A" if accepted else "R")
    assert stats["qwen_mtp_proposal_sources"] == "N"
    assert stats["qwen_mtp_ngram_first_attempts"] == 1
    assert stats["qwen_mtp_ngram_first_matches"] == 1
    assert stats["qwen_mtp_ngram_first_proposed"] == 1
    assert stats["qwen_mtp_ngram_first_accepted"] == int(accepted)
    assert stats["qwen_mtp_ngram_first_rejected"] == int(not accepted)
    assert stats["qwen_mtp_ngram_first_native_draft_bypasses"] == 1
    assert stats["qwen_mtp_native_draft_proposed"] == 0
    assert stats["qwen_mtp_native_draft_accepted"] == 0
    assert stats["qwen_mtp_native_draft_rejected"] == 0
    assert stats["qwen_mtp_bf16_sidecar_round_loads"] == 0
    assert stats["qwen_mtp_bf16_sidecar_round_releases"] == 0


def test_qwen_mtp_ngram_first_never_proposes_without_exact_stochastic_q(
    monkeypatch,
):
    """Lookup proposals have no full q, so categorical decode must use MTP."""
    import runtime.qwen35_mtp as mtp_module
    from runtime.qwen35_mtp import QwenMTPSpeculativeEngine
    from runtime.sampler import SamplingParams

    def forbidden_lookup(*_args, **_kwargs):
        raise AssertionError("stochastic decode must not query n-gram proposals")

    monkeypatch.setattr(mtp_module, "ngram_propose", forbidden_lookup)

    class _KV:
        def __init__(self):
            self.offset = 3
            self.kda_cache = None

        def trim(self, offset):
            self.offset = int(offset)

        def nbytes(self):
            return 0

    class _Target(_Engine):
        def __init__(self):
            super().__init__("/models/target")
            self.cache = SimpleNamespace(
                stats=SimpleNamespace(
                    hits=0, misses=0, evictions=0, bytes_read=0),
                total_bytes=0, max_bytes=1)
            self.expert_hits = self.expert_misses = 0
            for name in (
                    "fast_tier_bytes", "archive_bytes",
                    "parallel_tier_fetches", "parallel_tier_fast_bytes",
                    "parallel_tier_archive_bytes"):
                setattr(self.store, name, 0)
            self.governor = None
            self._layer_transient = self._prefill_layer_transient = 0
            self._decode_layer_transient = self._layer_transient_margin = 0
            self._token_transient = self._true_peak_metal_bytes = 0
            self._request_profiler = None
            self._hot_prompt_slots = []
            self._h_last = mx.zeros((1, 1, 1))
            self._h_window = self._h_last
            self.last_kv = None

        def generate(self, _prompt, max_tokens, **_kwargs):
            assert max_tokens == 1
            self.last_kv = _KV()
            return {
                "text": "4", "tokens": [4], "prefill_s": 0.0,
                "first_token_s": 0.0, "decode_s": 0.0, "total_s": 0.0,
                "termination_reason": "length", "stop_sequence": None,
                "path_stats": {}, "prompt_tokens": 3,
            }

        def forward_tokens_serial_positions(
            self, tokens, kv, *, capture_kda_endpoints=False,
        ):
            assert tokens == [4, 9]
            assert not capture_kda_endpoints
            kv.offset += 2
            self._h_window = mx.zeros((1, 2, 1))
            self._h_last = self._h_window[:, -1:, :]
            logits = mx.full((2, 12), -1000.0)
            logits = logits.at[0, 9].add(1000.0)
            return logits.at[1, 7].add(1000.0)

    class _NativeDrafter:
        @staticmethod
        def draft_logits(*_args, **_kwargs):
            return mx.full((12,), -1000.0).at[9].add(1000.0)

    target = _Target()
    engine = QwenMTPSpeculativeEngine(
        target, max_prompt_tokens=8, min_output_tokens=2,
        plain_warmup_tokens=0, adaptive_stop=False, ngram_first=True)
    engine.drafter = _NativeDrafter()
    result = engine.generate(
        "x", 2, sampling=SamplingParams(temperature=1.0, seed=17))
    stats = result["path_stats"]

    assert result["tokens"] == [4, 9]
    assert stats["qwen_mtp_ngram_first_enabled"] == 1
    assert stats["qwen_mtp_ngram_first_eligible"] == 0
    assert stats["qwen_mtp_ngram_first_attempts"] == 0
    assert stats["qwen_mtp_ngram_first_matches"] == 0
    assert stats["qwen_mtp_ngram_first_proposed"] == 0
    assert stats["qwen_mtp_native_draft_proposed"] == 1
    assert stats["qwen_mtp_native_draft_accepted"] == 1
    assert stats["qwen_mtp_native_draft_rejected"] == 0
    assert stats["qwen_mtp_proposal_sources"] == "M"


def test_qwen_mtp_batches_grammar_forced_span_before_next_decision():
    """Jump-forward and MTP compose: deterministic grammar tokens share one
    target sweep and the draft head is reserved for genuinely free choices."""
    from runtime.qwen35_mtp import QwenMTPSpeculativeEngine

    class _Constraint:
        profile = "required_tool"

        def __init__(self):
            self.completed = False
            self.accepted = []
            self.forced_calls = 0

        def forced_run(self, _limit, encode=None):
            assert encode is not None
            self.forced_calls += 1
            if self.forced_calls > 1:
                return []
            self.accepted.extend((8, 9))
            return [8, 9]

        def mask_logits(self, logits):
            masked = mx.full(logits.shape, -1e9)
            return masked.at[..., 5].add(1e9 + 1)

        def accept_token(self, token):
            self.accepted.append(int(token))

    class _KV:
        def __init__(self):
            self.offset = 3
            self.kda_cache = None

        def trim(self, offset):
            self.offset = offset

        def nbytes(self):
            return 0

    class _Target(_Engine):
        def __init__(self):
            super().__init__("/models/target")
            self.cfg = SimpleNamespace(num_experts=8, eos_token_ids=())
            self.rc = SimpleNamespace(grammar_jump_forward_lossy=True)
            self.cache = SimpleNamespace(
                stats=SimpleNamespace(
                    hits=0, misses=0, evictions=0, bytes_read=0),
                total_bytes=0, max_bytes=1)
            self.expert_hits = self.expert_misses = 0
            for name in (
                    "fast_tier_bytes", "archive_bytes",
                    "parallel_tier_fetches", "parallel_tier_fast_bytes",
                    "parallel_tier_archive_bytes"):
                setattr(self.store, name, 0)
            self.governor = None
            self._layer_transient = 0
            self._prefill_layer_transient = 0
            self._decode_layer_transient = 0
            self._layer_transient_margin = 0
            self._token_transient = 0
            self._true_peak_metal_bytes = 0
            self._request_profiler = None
            self._hot_prompt_slots = []
            self._h_last = mx.zeros((1, 1, 1))
            self.last_kv = None
            self.serial_calls = []

        def generate(self, prompt, max_tokens, **kwargs):
            assert max_tokens == 1
            kwargs["constraint"].accept_token(4)
            self.last_kv = _KV()
            return {
                "text": "4", "tokens": [4], "prefill_s": 0.0,
                "first_token_s": 0.0, "decode_s": 0.0, "total_s": 0.0,
                "termination_reason": "length", "stop_sequence": None,
                "path_stats": {}, "prompt_tokens": 3,
            }

        def forward_tokens_serial_positions(
                self, tokens, kv, *, capture_kda_endpoints=False):
            assert not capture_kda_endpoints
            self.serial_calls.append(list(tokens))
            kv.offset += len(tokens)
            self._h_last = mx.zeros((1, 1, 1))
            logits = mx.zeros((len(tokens), 10))
            return logits.at[-1, 5].add(1)

    target = _Target()
    engine = QwenMTPSpeculativeEngine(
        target, max_prompt_tokens=8, min_output_tokens=2,
        plain_warmup_tokens=0)
    engine.drafter = SimpleNamespace(
        draft_token=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("forced span must bypass the draft head")))
    constraint = _Constraint()

    result = engine.generate("x", 4, constraint=constraint)

    assert result["tokens"] == [4, 8, 9, 5]
    assert result["kv_positions"] == 6
    assert target.serial_calls == [[4, 8, 9]]
    assert constraint.accepted == [4, 8, 9, 5]
    assert result["path_stats"]["qwen_mtp_proposed"] == 0
    assert result["path_stats"]["qwen_mtp_grammar_forced_tokens"] == 2
    assert result["path_stats"]["qwen_mtp_grammar_forced_sweeps"] == 1
    assert result["path_stats"]["qwen_mtp_round_outcomes"] == "F2"


def test_qwen_mtp_rejection_restores_serial_kda_midpoint_without_refeed():
    from runtime.qwen35_mtp import QwenMTPSpeculativeEngine

    midpoint = object()

    class _Constraint:
        profile = "test-grammar"

        def __init__(self):
            self.completed = False
            self.accepted = []

        def mask_logits(self, logits):
            masked = mx.full(logits.shape, -1e9)
            return masked.at[..., 5].add(1e9 + 1)

        def accept_token(self, token):
            self.accepted.append(int(token))

    class _KV:
        def __init__(self):
            self.offset = 3
            self.kda_cache = object()
            self.lengths = [3, 1]

        def trim(self, offset):
            self.offset = offset

        def layer_lengths(self):
            return tuple(self.lengths)

        def trim_layer_lengths(self, lengths):
            self.lengths = list(lengths)
            self.offset = self.lengths[0]

        def nbytes(self):
            return 0

    class _Target(_Engine):
        def __init__(self):
            super().__init__("/models/target")
            self.cfg = SimpleNamespace(num_experts=8, eos_token_ids=())
            self.cache = SimpleNamespace(
                stats=SimpleNamespace(
                    hits=0, misses=0, evictions=0, bytes_read=0),
                total_bytes=0, max_bytes=1)
            self.expert_hits = self.expert_misses = 0
            for name in (
                    "fast_tier_bytes", "archive_bytes",
                    "parallel_tier_fetches", "parallel_tier_fast_bytes",
                    "parallel_tier_archive_bytes"):
                setattr(self.store, name, 0)
            self.governor = None
            self._layer_transient = 0
            self._prefill_layer_transient = 0
            self._decode_layer_transient = 0
            self._layer_transient_margin = 0
            self._token_transient = 0
            self._true_peak_metal_bytes = 0
            self._request_profiler = None
            self._hot_prompt_slots = []
            self._h_last = mx.zeros((1, 1, 1))
            self.last_kv = None
            self.forward_calls = 0
            self.endpoint_requests = []
            self.constraint_hidden_positions = []

        def generate(self, prompt, max_tokens, **_kwargs):
            assert max_tokens == 1
            _kwargs["constraint"].accept_token(4)
            self.last_kv = _KV()
            return {
                "text": "4", "tokens": [4], "prefill_s": 0.0,
                "first_token_s": 0.0, "decode_s": 0.0, "total_s": 0.0,
                "termination_reason": "length", "stop_sequence": None,
                "path_stats": {}, "prompt_tokens": 3,
            }

        def forward_tokens(self, _tokens, _kv):
            self.forward_calls += 1
            raise AssertionError("rejection must not refeed the catchup token")

        def forward_tokens_serial_positions(
                self, tokens, kv, *, capture_kda_endpoints=False):
            assert tokens == [4, 9]
            assert capture_kda_endpoints
            kv.offset += 2
            kv.lengths = [value + 2 for value in kv.lengths]
            self._h_window = mx.array([[[40.0], [90.0]]])
            self._h_last = self._h_window[:, -1:, :]
            logits = mx.zeros((2, 10))
            logits = logits.at[0, 6].add(1)
            logits = logits.at[1, 7].add(1)
            return logits

        def consume_serial_kda_endpoint(self, fed_positions):
            self.endpoint_requests.append(fed_positions)
            return midpoint if fed_positions == 1 else None

        def _constraint_logits(self, logits, constraint, hidden=None):
            self.constraint_hidden_positions.append(float(hidden.item()))
            return constraint.mask_logits(logits)

    target = _Target()
    engine = QwenMTPSpeculativeEngine(
        target, max_prompt_tokens=8, min_output_tokens=2,
        plain_warmup_tokens=0)
    engine.drafter = SimpleNamespace(
        draft_token=lambda *_args, **_kwargs: 9)

    constraint = _Constraint()
    result = engine.generate("x", 2, constraint=constraint)

    assert result["tokens"] == [4, 5]
    assert result["kv_positions"] == 4
    assert target.last_kv.kda_cache is midpoint
    assert target.last_kv.lengths == [4, 2]
    assert target.forward_calls == 0
    assert target.endpoint_requests == [1]
    assert target.constraint_hidden_positions == [40.0]
    assert constraint.accepted == [4, 5]
    assert result["path_stats"]["qwen_mtp_serial_verify_rounds"] == 1
    assert result["path_stats"]["qwen_mtp_kda_endpoint_restores"] == 1
    assert result["path_stats"]["qwen_mtp_refeed_sweeps_saved"] == 1
    assert result["path_stats"]["qwen_mtp_constraint_verified"] == 1


def test_qwen_mtp_reports_complete_target_and_draft_rerank_telemetry():
    """Reproduce the real short8 depth-1 ARRRA round shape without weights.

    One bootstrap target row, five shared-head draft rows, and five width-2
    target verifier windows must report 16 total positions, split into 11
    authoritative target positions and five non-authoritative draft positions.
    """
    from runtime.qwen35_mtp import QwenMTPSpeculativeEngine

    class _TelemetryHead:
        def __init__(self):
            self.values = {
                "calls": 0,
                "positions": 0,
                "candidate_winner_changes": 0,
                "candidate_recall_probes": 0,
                "candidate_recall_hits": 0,
                "candidate_read_calls": 0,
                "candidate_read_extents": 0,
                "candidate_rows_requested": 0,
                "candidate_unique_rows_read": 0,
                "candidate_bytes_read": 0,
                "candidate_recall_full_scan_calls": 0,
                "candidate_recall_full_scan_bytes": 0,
            }

        def record(self, positions):
            positions = int(positions)
            self.values["calls"] += positions
            self.values["positions"] += positions
            self.values["candidate_winner_changes"] += positions
            self.values["candidate_recall_probes"] += positions
            self.values["candidate_recall_hits"] += positions
            self.values["candidate_read_calls"] += positions
            self.values["candidate_read_extents"] += 64 * positions
            self.values["candidate_rows_requested"] += 64 * positions
            self.values["candidate_unique_rows_read"] += 64 * positions
            self.values["candidate_bytes_read"] += 655_360 * positions
            self.values["candidate_recall_full_scan_calls"] += positions
            self.values[
                "candidate_recall_full_scan_bytes"] += 2_542_796_800 * positions

        def telemetry_snapshot(self):
            return dict(self.values)

    class _KV:
        def __init__(self):
            self.offset = 3
            self.kda_cache = None

        def trim(self, offset):
            self.offset = int(offset)

        def nbytes(self):
            return 0

    class _Target(_Engine):
        def __init__(self):
            super().__init__("/models/target")
            self._lm_head_w = _TelemetryHead()
            self.cfg = SimpleNamespace(num_experts=0, eos_token_ids=())
            self.cache = SimpleNamespace(
                stats=SimpleNamespace(
                    hits=0, misses=0, evictions=0, bytes_read=0),
                total_bytes=0, max_bytes=1)
            self.expert_hits = self.expert_misses = 0
            for name in (
                    "fast_tier_bytes", "archive_bytes",
                    "parallel_tier_fetches", "parallel_tier_fast_bytes",
                    "parallel_tier_archive_bytes"):
                setattr(self.store, name, 0)
            self.governor = None
            self._layer_transient = 0
            self._prefill_layer_transient = 0
            self._decode_layer_transient = 0
            self._layer_transient_margin = 0
            self._token_transient = 0
            self._true_peak_metal_bytes = 0
            self._request_profiler = None
            self._hot_prompt_slots = []
            self._h_last = mx.zeros((1, 1, 1))
            self._h_window = self._h_last
            self.last_kv = None
            self.round = 0
            self.accepts = (True, False, False, False, True)

        def generate(self, prompt, max_tokens, **_kwargs):
            assert max_tokens == 1
            self._lm_head_w.record(1)
            self.last_kv = _KV()
            return {
                "text": "4", "tokens": [4], "prefill_s": 0.0,
                "first_token_s": 0.0, "decode_s": 0.0, "total_s": 0.0,
                "termination_reason": "length", "stop_sequence": None,
                # This is intentionally the stale bootstrap-only shape that
                # the production wrapper must replace at return.
                "path_stats": {
                    "reranked_lm_head_calls": 1,
                    "reranked_lm_head_positions": 1,
                },
                "prompt_tokens": 3,
            }

        def forward_tokens_serial_positions(
                self, tokens, kv, *, capture_kda_endpoints=False):
            assert tokens[1] == 9
            assert not capture_kda_endpoints
            assert self.round < len(self.accepts)
            self._lm_head_w.record(len(tokens))
            kv.offset += len(tokens)
            self._h_window = mx.array([[[40.0], [90.0]]])
            self._h_last = self._h_window[:, -1:, :]
            logits = mx.zeros((2, 12))
            winner = 9 if self.accepts[self.round] else 6
            logits = logits.at[0, winner].add(2)
            logits = logits.at[1, 7].add(2)
            self.round += 1
            return logits

    class _Drafter:
        def __init__(self, target):
            self.target = target

        def draft_token(self, *_args, **_kwargs):
            self.target._lm_head_w.record(1)
            return 9

    target = _Target()
    engine = QwenMTPSpeculativeEngine(
        target, max_prompt_tokens=8, min_output_tokens=2,
        plain_warmup_tokens=0, adaptive_stop=False)
    engine.drafter = _Drafter(target)

    result = engine.generate("x", 8)
    stats = result["path_stats"]

    assert len(result["tokens"]) == 8
    assert stats["qwen_mtp_round_outcomes"] == "ARRRA"
    assert stats["qwen_mtp_serial_verify_rounds"] == 5
    assert stats["qwen_mtp_plain_equivalent_target_sweeps"] == 7
    assert stats["qwen_mtp_target_sweeps"] == 5
    assert stats["qwen_mtp_target_sweeps_avoided"] == 2
    assert stats["qwen_mtp_target_tokens_per_sweep"] == 1.4
    assert stats["qwen_mtp_verifier_input_positions"] == 10
    assert stats["qwen_mtp_verifier_committed_positions"] == 7
    assert stats["qwen_mtp_verifier_rolled_back_positions"] == 3
    assert stats["qwen_mtp_verifier_output_tokens"] == 7
    assert stats["qwen_mtp_verifier_tokens_per_sweep"] == 1.4
    assert stats["qwen_mtp_verifier_accepted_draft_tokens"] == 2
    assert stats["qwen_mtp_verifier_correction_tokens"] == 3
    assert stats["qwen_mtp_verifier_bonus_tokens"] == 2
    assert stats["reranked_lm_head_calls"] == 16
    assert stats["reranked_lm_head_positions"] == 16
    assert stats["reranked_lm_head_candidate_recall_probes"] == 16
    assert stats["reranked_lm_head_candidate_recall_hits"] == 16
    assert stats["reranked_lm_head_candidate_recall"] == 1.0
    assert stats["qwen_mtp_target_reranked_lm_head_calls"] == 11
    assert stats["qwen_mtp_target_reranked_lm_head_positions"] == 11
    assert stats[
        "qwen_mtp_target_reranked_lm_head_candidate_recall_probes"] == 11
    assert stats["qwen_mtp_draft_reranked_lm_head_calls"] == 5
    assert stats["qwen_mtp_draft_reranked_lm_head_positions"] == 5
    assert stats[
        "qwen_mtp_draft_reranked_lm_head_candidate_recall_probes"] == 5
    assert stats["reranked_lm_head_candidate_bytes_read"] == 10_485_760
    assert stats[
        "reranked_lm_head_candidate_recall_full_scan_bytes"] == 40_684_748_800


def test_forward_tokens_serial_positions_excludes_hybrid_model_types():
    """F94-discovered gap: layer_runner.run_block (called by
    forward_tokens_serial_positions) is a plain dense-transformer block with
    no awareness of qwen3_5/kimi_linear's hybrid DeltaNet/full-attention
    layer_types -- it would KeyError on 'model.layers.N.self_attn.*' tensor
    names that don't exist on a linear_attention layer. This reproduced live
    against a real Qwen3.6-27B checkpoint (qwen35_mtp_gate.py).

    F113 follow-on (2026-07-25/26): qwen3_5/qwen3_5_moe and kimi_linear
    were each given a real per-position dispatch (_qwen35_attention_residual/
    _qwen35_mlp_residual and _kimi_linear_attention_residual/
    _kimi_linear_mlp_residual respectively, both verified byte-identical
    against real checkpoints -- Qwen3.5-9B and Kimi-Linear-48B-A3B-
    Instruct) and are no longer refused by this guard. gpt_oss (plain
    MoE, no hybrid/recurrent layers, never given an equivalent dispatch)
    still is."""
    from runtime.engine import StreamingEngine

    engine = object.__new__(StreamingEngine)
    engine.cfg = SimpleNamespace(num_experts=8, model_type="gpt_oss")
    try:
        engine.forward_tokens_serial_positions([1, 2], kv=None)
        raised = False
    except ValueError:
        raised = True
    assert raised, "forward_tokens_serial_positions must refuse gpt_oss"

    for model_type in ("qwen3_5", "qwen3_5_moe", "kimi_linear"):
        engine = object.__new__(StreamingEngine)
        engine.cfg = SimpleNamespace(num_experts=0, model_type=model_type)
        try:
            engine.forward_tokens_serial_positions([1, 2], kv=None)
        except ValueError:
            raise AssertionError(
                f"forward_tokens_serial_positions must no longer refuse "
                f"model_type={model_type!r} via the guard clause"
            )
        except AttributeError:
            # Expected: this fake engine has no real weights/kv, so it
            # proceeds past the guard into the per-position dispatch and
            # fails there instead -- proving the guard itself let it
            # through, which is what this test is checking.
            pass
