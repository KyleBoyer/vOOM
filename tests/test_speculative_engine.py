"""Pure serving-contract tests for the exact speculative adapter."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


class _Encoding:
    def __init__(self, ids):
        self.ids = list(ids)


class _Tokenizer:
    def __init__(self, ids=(1, 2, 3)):
        self.ids = list(ids)
        self.encode_calls = 0

    def encode(self, _text):
        self.encode_calls += 1
        return _Encoding(self.ids)

    def decode(self, ids):
        return ",".join(str(value) for value in ids)


class _Engine:
    def __init__(self, path: str, *, ids=(1, 2, 3), vocab=16):
        self._model_dir = Path(path)
        self.tokenizer = _Tokenizer(ids)
        self.cfg = SimpleNamespace(num_experts=0, vocab_size=vocab)
        self.calls = []
        self.closes = 0
        self.releases = 0

    def generate(self, prompt, max_tokens, on_token=None, stop=None,
                 on_progress=None):
        self.calls.append((prompt, max_tokens, on_token, stop, on_progress))
        return {"text": "target", "tokens": [4], "path_stats": {}}

    def release_request_state(self):
        self.releases += 1

    def close(self):
        self.closes += 1


def test_speculative_adapter_falls_back_for_limits_and_vocab():
    from runtime.speculative import SpeculativeEngine

    target = _Engine("/models/target")
    draft = _Engine("/models/draft", vocab=4)
    engine = SpeculativeEngine(target, draft, k=3, max_prompt_tokens=2)

    result = engine.generate("x", 4)
    assert result["path_stats"]["speculative_fallback_reason"] == "prompt-limit"

    engine.max_prompt_tokens = 8
    target.tokenizer.ids = draft.tokenizer.ids = [1, 5]
    result = engine.generate("x", 4)
    assert result["path_stats"]["speculative_fallback_reason"] == "draft-vocab"
    assert len(target.calls) == 2


def test_speculative_adapter_forwards_callbacks_and_stops_to_decoder():
    from runtime.speculative import SpeculativeEngine

    target = _Engine("/models/target")
    draft = _Engine("/models/draft")
    engine = SpeculativeEngine(target, draft, k=3, max_prompt_tokens=8)
    captured = {}
    expected = {
        "text": "speculative", "tokens": [7], "first_token_s": 1.0,
        "total_s": 1.0, "path_stats": {
            "speculative_used": 1, "prompt_tokenize_s": 0.0},
    }

    def generate(*args, **kwargs):
        captured.update(kwargs)
        return expected

    engine.decoder = SimpleNamespace(generate=generate)
    on_token = lambda _text: None
    on_progress = lambda _progress: None
    assert engine.generate(
        "x", 1, on_token=on_token, stop=["done"],
        on_progress=on_progress) is expected
    assert captured["on_token"] is on_token
    assert captured["on_progress"] is on_progress
    assert captured["stop"] == ["done"]
    assert not target.calls


def test_speculative_adapter_reuses_prepared_target_ids():
    from runtime.server import PreparedPrompt
    from runtime.speculative import SpeculativeEngine

    target = _Engine("/models/target")
    draft = _Engine("/models/draft")
    engine = SpeculativeEngine(target, draft, k=3, max_prompt_tokens=8)
    target_calls_before = target.tokenizer.encode_calls
    draft_calls_before = draft.tokenizer.encode_calls
    captured = {}
    expected = {
        "text": "speculative", "tokens": [7], "first_token_s": 0.0,
        "total_s": 0.0, "path_stats": {"prompt_tokenize_s": 0.0},
    }

    def generate(*args, **kwargs):
        captured.update(kwargs)
        return expected

    engine.decoder = SimpleNamespace(generate=generate)
    engine.generate(PreparedPrompt("ignored carrier text", [1, 2, 3]), 1)

    assert target.tokenizer.encode_calls == target_calls_before
    assert draft.tokenizer.encode_calls == draft_calls_before + 1
    assert captured["encoded_ids"] == [1, 2, 3]


def test_speculative_adapter_delegates_target_attributes_and_closes_both_once():
    from runtime.speculative import SpeculativeEngine

    target = _Engine("/models/target")
    draft = _Engine("/models/draft")
    target.marker = "target-owned"
    engine = SpeculativeEngine(target, draft, k=6, max_prompt_tokens=8)

    assert engine.marker == "target-owned"
    assert engine._speculative_draft_dir == Path("/models/draft")
    engine.release_request_state()
    assert (target.releases, draft.releases) == (1, 1)

    engine.close()
    engine.close()
    assert (target.closes, draft.closes) == (1, 1)


def test_speculative_adapter_preserves_success_result_schema():
    from runtime.speculative import SpeculativeEngine

    target = _Engine("/models/target")
    draft = _Engine("/models/draft")
    engine = SpeculativeEngine(target, draft, k=3, max_prompt_tokens=8)
    expected = {
        "text": "speculative", "tokens": [7, 8],
        "prefill_s": 1.0, "decode_s": 2.0, "first_token_s": 1.0,
        "total_s": 3.0, "tok_per_s": 0.5, "kv_bytes": 10,
        "kv_positions": 4, "stopped": False, "stop_sequence": None,
        "termination_reason": "length", "true_peak_metal_bytes": 20,
        "prompt_tokens": 3, "path_stats": {"speculative_used": 1},
    }
    engine.decoder = SimpleNamespace(generate=lambda *args, **kwargs: expected)

    result = engine.generate("x", 2)
    assert result is expected
    assert result["path_stats"]["speculative_used"] == 1
    assert result["path_stats"]["speculative_enabled"] == 1
    assert result["path_stats"]["prompt_tokenize_s"] >= 0
