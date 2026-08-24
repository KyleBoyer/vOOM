"""DSpark block verification and rollback invariants with forced outcomes."""

import sys
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from runtime.dspark import (
    CtxCache,
    DSparkConfig,
    DSparkDrafter,
    DSparkSpeculativeDecoder,
)
from runtime.sampler import SamplingParams


class _Tokenizer:
    def encode(self, _prompt):
        return SimpleNamespace(ids=[0, 0])

    def decode(self, tokens):
        return " ".join(str(token) for token in tokens)


class _KV:
    def __init__(self):
        self.offset = 0

    def trim(self, length):
        self.offset = length

    def nbytes(self):
        return self.offset * 8


class _Target:
    def __init__(self):
        self.cfg = SimpleNamespace(
            model_type="qwen3", hidden_size=2, vocab_size=8,
            num_hidden_layers=3, eos_token_ids=(),
        )
        self.rc = SimpleNamespace(stepped_kv_threshold=0, context_bound=0)
        self.effective_max_position_embeddings = 128
        self.rope_profile = "native"
        self.governor = None
        self.tokenizer = _Tokenizer()
        self._tap_hidden = {}
        self._provisional = None
        self._true_peak_metal_bytes = 0
        self.verify_predictions = [
            [2, 3, 4],  # forced accept-none
            [3, 4, 5],  # accept first proposal only
            [5, 6, 0],  # accept both proposals
        ]
        self.verify_index = 0
        self.prefill_calls = 0

    def release_request_state(self):
        pass

    def new_kv(self, stepped=False):
        return _KV()

    def _set_taps(self, width, tap_layers):
        self._tap_hidden = {
            layer: mx.full((1, width, 2), layer + 1, dtype=mx.bfloat16)
            for layer in tap_layers
        }

    @staticmethod
    def _logits(tokens):
        out = mx.full((len(tokens), 8), -10.0)
        rows = mx.arange(len(tokens))
        out[rows, mx.array(tokens)] = 10.0
        return out

    def forward_tokens(self, tokens, kv, tap_layers=None):
        self.prefill_calls += 1
        kv.offset += len(tokens)
        self._set_taps(len(tokens), tap_layers or ())
        # Only the final prompt row is consumed; it predicts pending token 1.
        return self._logits([0] * (len(tokens) - 1) + [1])

    def forward_tokens_serial_positions(self, tokens, kv, tap_layers=None):
        predictions = self.verify_predictions[self.verify_index]
        self.verify_index += 1
        assert len(predictions) == len(tokens)
        kv.offset += len(tokens)
        self._set_taps(len(tokens), tap_layers or ())
        return self._logits(predictions)

    def _note_true_peak(self):
        pass


class _Drafter:
    block_size = 7
    confidence_head = None

    def __init__(self):
        self.config = SimpleNamespace(
            hidden_size=2, vocab_size=8, block_size=7,
            target_layer_ids=[0, 1],
        )
        self.last_caches = None

    def make_ctx_cache(self):
        self.last_caches = [CtxCache(), CtxCache()]
        return self.last_caches

    def update_context(self, target_hidden_cat, ctx_offset, ctx_caches):
        width = target_hidden_cat.shape[1]
        assert all(cache.length == ctx_offset for cache in ctx_caches)
        k = mx.zeros((1, 1, width, 1), dtype=mx.bfloat16)
        v = mx.zeros((1, 1, width, 1), dtype=mx.bfloat16)
        for cache in ctx_caches:
            cache.append(k, v)


def test_forced_accept_none_partial_and_all_keep_exact_state():
    target = _Target()
    drafter = _Drafter()
    decoder = DSparkSpeculativeDecoder(target, drafter, max_draft_tokens=2)
    proposals = iter(([7, 7], [3, 7], [5, 6]))
    decoder._propose = lambda *_args: list(next(proposals))

    result = decoder.generate("ignored", max_tokens=7, stop=[])

    assert result["tokens"] == [1, 2, 3, 4, 5, 6, 0]
    assert result["stats"].rounds == [(2, 0), (2, 1), (2, 2)]
    assert result["kv_positions"] == result["prompt_tokens"] + 7 - 1
    assert all(cache.length == result["kv_positions"]
               for cache in drafter.last_caches)


def test_ctx_cache_trim_rejects_negative_and_keeps_prefix():
    cache = CtxCache()
    cache.append(mx.zeros((1, 1, 4, 2)), mx.zeros((1, 1, 4, 2)))
    cache.trim_to(2)
    assert cache.length == 2
    try:
        cache.trim_to(-1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative DSpark context trim should fail")


def test_exact_repeat_reuses_target_and_drafter_prompt_state():
    target = _Target()
    drafter = _Drafter()
    decoder = DSparkSpeculativeDecoder(
        target, drafter, max_draft_tokens=2,
        prompt_cache_min_tokens=2)

    cold = decoder.generate("ignored", max_tokens=1, stop=[])
    warm = decoder.generate("ignored", max_tokens=1, stop=[])

    assert cold["tokens"] == warm["tokens"] == [1]
    assert target.prefill_calls == 1
    assert not cold["path_stats"]["prompt_cache_exact_hit"]
    assert warm["path_stats"]["prompt_cache_exact_hit"]
    assert warm["path_stats"]["prompt_cache_prefix_tokens"] == 2
    assert warm["path_stats"]["prompt_cache_source"] == "dspark-memory"


def test_constrained_dspark_keeps_target_grammar_authoritative():
    class Constraint:
        def __init__(self):
            self.allowed = [1, 6]
            self.accepted = []

        @property
        def completed(self):
            return len(self.accepted) == len(self.allowed)

        def mask_logits(self, logits):
            token = self.allowed[len(self.accepted)]
            masked = mx.full(logits.shape, -mx.inf)
            masked[token] = logits[token]
            return masked

        def accept_token(self, token):
            assert token == self.allowed[len(self.accepted)]
            self.accepted.append(int(token))

    target = _Target()
    drafter = _Drafter()
    decoder = DSparkSpeculativeDecoder(target, drafter, max_draft_tokens=2)
    decoder._propose = lambda *_args: [7, 7]
    constraint = Constraint()

    result = decoder.generate(
        "ignored", max_tokens=4, stop=[], constraint=constraint)

    assert result["tokens"] == [1, 6]
    assert constraint.completed
    assert result["termination_reason"] == "grammar"
    assert result["path_stats"]["dspark_constraint_verified"] == 1
    assert result["stats"].accepted == 0


def test_k3_mla_draft_uses_compressed_context_and_shared_target_head():
    cfg = DSparkConfig(
        hidden_size=8,
        vocab_size=16,
        num_hidden_layers=2,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        block_size=3,
        mask_token_id=15,
        target_layer_ids=[0, 1],
        model_type="k3_dspark",
        target_hidden_size=8,
        q_lora_rank=4,
        kv_lora_rank=4,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        v_head_dim=2,
        markov_rank=0,
        enable_confidence_head=False,
    )
    draft = DSparkDrafter(cfg)
    head = mx.zeros((16, 8), dtype=mx.float32).at[3, 0].add(10)
    draft.bind_target_lm_head(lambda: head)
    caches = draft.make_ctx_cache()
    draft.update_context(mx.zeros((1, 5, 16)), 0, caches)
    logits, hidden = draft.draft_block(1, 5, caches, cap=2)
    mx.eval(logits, hidden)

    assert logits.shape == (2, 16)
    assert hidden.shape == (2, 8)
    assert all(cache.length == 5 for cache in caches)
    # Content latent and shared positional key stay compressed.
    assert caches[0].k.shape == (1, 1, 5, 4)
    assert caches[0].v.shape == (1, 1, 5, 2)


@pytest.mark.parametrize("temperature", [0.3, 0.5, 0.7, 1.0])
def test_stochastic_rejection_samples_target_minus_draft_residual(
    temperature,
):
    sampling = SamplingParams(temperature=temperature, seed=17)
    sampling.seed_rng()
    # Proposal 0 has q=1 but p=0, so it always rejects.  The residual puts
    # all mass on token 1; the replacement is therefore deterministic.
    q = [mx.array([1.0, 0.0, 0.0])]
    verified = mx.array([
        [-100.0, 100.0, -100.0],
        [0.0, 0.0, 0.0],
    ])
    accepted, committed = DSparkSpeculativeDecoder._verify_stochastic(
        [0], q, verified, sampling, history=[2])

    assert accepted == 0
    assert committed == [1]
