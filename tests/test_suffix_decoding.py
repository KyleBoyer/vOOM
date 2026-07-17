"""Focused contracts for engine-local shared-prefill SuffixDecoding."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest


def _cache(**overrides):
    from runtime.suffix_decoding import SuffixDecodingCache

    settings = {
        "identity": "fixture-identity",
        "max_depth": 4,
        "max_spec_tokens": 6,
        "factor": 4.0,
        "min_probability": 0.1,
        "max_cached_requests": 8,
        "max_cached_tokens": 64,
        "max_nodes": 256,
        "max_bytes": 128_000,
        "max_local_tokens": 8,
    }
    settings.update(overrides)
    return SuffixDecodingCache(**settings)


@pytest.mark.parametrize(
    ("proposal", "target", "accepted", "committed"),
    [
        ([5, 6], [7, 8, 9], 0, [7]),
        ([5, 6], [5, 7, 9], 1, [5, 7]),
        ([5, 6], [5, 6, 7], 2, [5, 6, 7]),
    ],
)
def test_select_verified_tokens_accept_none_partial_all(
        proposal, target, accepted, committed):
    from runtime.suffix_decoding import select_verified_tokens

    assert select_verified_tokens(proposal, target) == (accepted, committed)


class _FakeKV:
    def __init__(self, offset):
        self.offset = offset
        self.trim_calls = []

    def trim(self, length):
        self.trim_calls.append(length)
        self.offset = min(self.offset, length)


class _FakeTokenizer:
    def __init__(self, pieces=None):
        self.pieces = pieces or {}

    def decode(self, tokens):
        return "".join(self.pieces.get(int(token), str(int(token)))
                       for token in tokens)


class _FakeEngine:
    def __init__(self, target_tokens, *, eos=(), pieces=None):
        self.target_tokens = list(target_tokens)
        self.cfg = SimpleNamespace(eos_token_ids=tuple(eos))
        self.tokenizer = _FakeTokenizer(pieces)
        self.governor = None
        self._token_transient = 0

    def forward_tokens_serial_positions(self, tokens, kv):
        assert len(tokens) == len(self.target_tokens)
        kv.offset += len(tokens)
        vocab = max(32, max(self.target_tokens) + 1)
        logits = mx.full((len(tokens), vocab), -10.0)
        rows = mx.arange(len(tokens))
        logits[rows, mx.array(self.target_tokens)] = 10.0
        return logits

    def _note_true_peak(self):
        pass


class _FakeProposalCache:
    def __init__(self, proposal):
        from runtime.suffix_decoding import SuffixProposal

        self.proposal = SuffixProposal(
            tuple(proposal), score=1.0, match_length=2, source="global")

    def propose(self, _state, max_tokens=None):
        del max_tokens
        return self.proposal


class _FakeState:
    def __init__(self):
        self.committed = []

    def append_committed(self, tokens):
        self.committed.extend(tokens)


def _one_suffix_round(*, proposal, target, max_tokens, eos=(), stop=(), pieces=None):
    from runtime.suffix_decoding import run_shared_prefill_suffix_decode

    prompt = [10, 11, 12]
    generated = [3]
    kv = _FakeKV(len(prompt))
    state = _FakeState()

    def stop_match(text):
        matches = [(text.find(value), index, value)
                   for index, value in enumerate(stop)
                   if value and text.find(value) != -1]
        return min(matches) if matches else None

    result = run_shared_prefill_suffix_decode(
        _FakeEngine(target, eos=eos, pieces=pieces),
        _FakeProposalCache(proposal),
        state,
        prompt_tokens=prompt,
        generated=generated,
        kv=kv,
        logits=mx.zeros((1, 32)),
        max_tokens=max_tokens,
        stop=list(stop),
        stream_decoder=None,
        on_token=lambda _delta: None,
        stop_match=stop_match,
    )
    return generated, kv, state, result


@pytest.mark.parametrize(
    ("target", "expected", "accepted"),
    [
        ([7, 8, 9], [3, 7], 0),
        ([5, 7, 9], [3, 5, 7], 1),
        ([5, 6, 7], [3, 5, 6, 7], 2),
    ],
)
def test_shared_prefill_round_rolls_back_rejected_tail(
        target, expected, accepted):
    generated, kv, state, result = _one_suffix_round(
        proposal=[5, 6], target=target, max_tokens=len(expected))

    assert generated == expected
    assert state.committed == expected[1:]
    assert result.stats.accepted == accepted
    assert kv.offset == 3 + len(expected) - 1
    assert kv.trim_calls[-1] == kv.offset


@pytest.mark.parametrize(("stop", "eos", "expected_stop"), [
    (("STOP",), (), "3A"),
    ((), (6,), None),
])
def test_stop_or_eos_inside_accepted_window_trims_and_commits_only_emitted(
        stop, eos, expected_stop):
    generated, kv, state, result = _one_suffix_round(
        proposal=[5, 6],
        target=[5, 6, 7],
        max_tokens=4,
        eos=eos,
        stop=stop,
        pieces={5: "A", 6: "STOP"},
    )

    assert generated == [3, 5, 6]
    assert state.committed == [5, 6]
    assert result.stop_text == expected_stop
    assert result.stop_sequence == ("STOP" if stop else None)
    assert kv.offset == 3 + len(generated) - 1
    assert kv.trim_calls[-1] == kv.offset


def test_completed_history_enforces_fifo_token_node_and_byte_bounds():
    fifo = _cache(max_cached_requests=2, max_cached_tokens=4)
    assert fifo.add_output([1, 2])
    assert fifo.add_output([3, 4])
    assert fifo.add_output([5, 6])
    assert list(fifo._history) == [(3, 4), (5, 6)]
    assert fifo.cached_tokens == 4
    assert fifo.evicted_requests == 1
    assert fifo.accounted_bytes <= fifo._global_byte_limit

    node_limited = _cache(
        max_nodes=8, max_bytes=128_000, max_local_tokens=1)
    assert not node_limited.add_output([1, 2, 3])
    assert node_limited.rejected_requests == 1

    byte_limited = _cache(
        max_nodes=128, max_bytes=1_280, max_local_tokens=1)
    assert not byte_limited.add_output([1])
    assert byte_limited.rejected_requests == 1


def test_model_tokenizer_fingerprint_changes_with_tokenizer(tmp_path):
    from runtime.suffix_decoding import model_tokenizer_fingerprint

    (tmp_path / "config.json").write_text('{"model_type":"fixture"}')
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text('{"version":1}')
    (tmp_path / "model.safetensors").write_bytes(b"fixture")
    first = model_tokenizer_fingerprint(tmp_path)
    tokenizer.write_text('{"version":2}')
    second = model_tokenizer_fingerprint(tmp_path)

    assert first != second


def _fallback_engine(*, enabled=True):
    return SimpleNamespace(
        rc=SimpleNamespace(
            suffix_decoding=enabled,
            resident_fast_decode=False,
            resident_moe_decode=False,
            rerank_lm_head=False,
        ),
        cfg=SimpleNamespace(
            num_experts=0,
            model_type="qwen2",
            vision_config=None,
        ),
        _embed_rows=None,
        _streamed_lm_head=None,
    )


def test_suffix_fallbacks_are_explicit_and_fail_closed():
    from runtime.kv_cache import KVCache
    from runtime.sampler import SamplingParams
    from runtime.suffix_decoding import fallback_reason

    kv = KVCache(1)
    greedy = SamplingParams()
    engine = _fallback_engine(enabled=False)
    assert fallback_reason(
        engine, kv, greedy, None, terminal=False) == "disabled"

    engine.rc.suffix_decoding = True
    assert fallback_reason(
        engine, kv, SamplingParams(temperature=1.0), None,
        terminal=False) == "stochastic-sampling"
    assert fallback_reason(
        engine, kv, greedy, object(), terminal=False) == "constrained-decoding"

    engine.rc.resident_fast_decode = True
    assert fallback_reason(
        engine, kv, greedy, None, terminal=False) == "resident-decode"
    engine.rc.resident_fast_decode = False
    assert fallback_reason(
        engine, object(), greedy, None,
        terminal=False) == "unproven-kv-layout"


def test_runtime_config_yaml_is_explicit_and_default_off(tmp_path):
    from runtime.engine import RuntimeConfig

    assert RuntimeConfig().suffix_decoding is False
    config = tmp_path / "suffix.yaml"
    config.write_text("""
runtime:
  suffix_decoding: true
  suffix_decoding_k: 3
  suffix_decoding_factor: 2.5
  suffix_decoding_max_cached_tokens: 123
""")
    parsed = RuntimeConfig.from_yaml(config)
    assert parsed.suffix_decoding is True
    assert parsed.suffix_decoding_k == 3
    assert parsed.suffix_decoding_factor == 2.5
    assert parsed.suffix_decoding_max_cached_tokens == 123


def test_engine_shared_prefill_is_feature_off_exact_and_keeps_kv_endpoint(
        tmp_path):
    from runtime.engine import RuntimeConfig, StreamingEngine
    from tests.test_resident_fast_decode import _build_dense_fixture

    _build_dense_fixture(tmp_path)
    common = dict(
        max_weight_cache_mb=100,
        prefill_chunk_size=3,
        governor=False,
    )
    prompt = "shared prefill suffix chunk endpoint proof"
    baseline_engine = StreamingEngine(tmp_path, RuntimeConfig(**common))
    try:
        baseline_engine.cfg.eos_token_ids = ()
        baseline = baseline_engine.generate(prompt, max_tokens=12, stop=[])
        assert baseline_engine._suffix_cache is None
        assert baseline["path_stats"]["suffix_decoding_used"] == 0
        assert baseline["path_stats"]["suffix_decoding_fallback_reason"] == "disabled"
    finally:
        baseline_engine.close()

    suffix_engine = StreamingEngine(tmp_path, RuntimeConfig(
        **common,
        suffix_decoding=True,
        suffix_decoding_max_cached_requests=8,
        suffix_decoding_max_cached_tokens=256,
        suffix_decoding_max_nodes=4_096,
        suffix_decoding_max_bytes=2_000_000,
        suffix_decoding_max_local_tokens=32,
    ))
    try:
        suffix_engine.cfg.eos_token_ids = ()
        assert suffix_engine._suffix_cache.add_output(baseline["tokens"])
        candidate = suffix_engine.generate(prompt, max_tokens=12, stop=[])

        assert candidate["tokens"] == baseline["tokens"]
        assert candidate["path_stats"]["suffix_decoding_used"] == 1
        assert candidate["path_stats"]["suffix_decoding_proposed"] > 0
        assert candidate["path_stats"]["suffix_decoding_target_sweeps"] > 0
        assert candidate["path_stats"]["prompt_state_approximate"] == 0
        assert candidate["path_stats"]["prefill_chunks"] > 0
        assert candidate["kv_positions"] == (
            candidate["prompt_tokens"] + len(candidate["tokens"]) - 1)
        assert suffix_engine.last_kv.offset == candidate["kv_positions"]
    finally:
        suffix_engine.close()
