"""Pure serving-contract tests for the Qwen native-MTP speculative adapter
(mirrors tests/test_speculative_engine.py's fake-engine pattern)."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx


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
    EOS, the unused bonus must never become the next catchup token."""
    from runtime.qwen35_mtp import QwenMTPSpeculativeEngine

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
            assert tokens == [4, 9]
            kv.offset += 2
            logits = mx.zeros((2, 10))
            logits = logits.at[0, 9].add(1)
            logits = logits.at[1, 7].add(1)
            self._h_last = mx.zeros((1, 1, 1))
            return logits

    target = _Target()
    engine = QwenMTPSpeculativeEngine(
        target, max_prompt_tokens=8, min_output_tokens=2,
        plain_warmup_tokens=0)
    engine.drafter = SimpleNamespace(
        draft_token=lambda *_args, **_kwargs: 9)

    result = engine.generate("x", 32)

    assert result["tokens"] == [4, 9]
    assert result["termination_reason"] == "eos"
    assert result["kv_positions"] == 4


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
    assert constraint.accepted == [4, 5]
    assert result["path_stats"]["qwen_mtp_serial_verify_rounds"] == 1
    assert result["path_stats"]["qwen_mtp_kda_endpoint_restores"] == 1
    assert result["path_stats"]["qwen_mtp_refeed_sweeps_saved"] == 1
    assert result["path_stats"]["qwen_mtp_constraint_verified"] == 1


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
