"""Pure serving-contract tests for the Qwen native-MTP speculative adapter
(mirrors tests/test_speculative_engine.py's fake-engine pattern)."""

from __future__ import annotations

from types import SimpleNamespace


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


def test_qwen_mtp_adapter_falls_back_for_constrained_decoding():
    from runtime.qwen35_mtp import QwenMTPSpeculativeEngine

    target = _Engine("/models/target")
    engine = QwenMTPSpeculativeEngine(target, max_prompt_tokens=8)

    result = engine.generate("x", 4, constraint=object())
    assert result["path_stats"]["qwen_mtp_fallback_reason"] == "constrained-decoding"
    assert len(target.calls) == 1


def test_qwen_mtp_adapter_falls_back_for_stochastic_sampling():
    from runtime.qwen35_mtp import QwenMTPSpeculativeEngine
    from runtime.sampler import SamplingParams

    target = _Engine("/models/target")
    engine = QwenMTPSpeculativeEngine(target, max_prompt_tokens=8)
    sampling = SamplingParams(temperature=0.8)
    assert not sampling.is_greedy

    result = engine.generate("x", 4, sampling=sampling)
    assert result["path_stats"]["qwen_mtp_fallback_reason"] == "stochastic-sampling"
    assert len(target.calls) == 1


def test_qwen_mtp_adapter_falls_back_for_prompt_limit():
    from runtime.qwen35_mtp import QwenMTPSpeculativeEngine

    target = _Engine("/models/target", ids=(1, 2, 3, 4, 5))
    engine = QwenMTPSpeculativeEngine(target, max_prompt_tokens=2)

    result = engine.generate("x", 4)
    assert result["path_stats"]["qwen_mtp_fallback_reason"] == "prompt-limit"
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


def test_forward_tokens_serial_positions_excludes_hybrid_model_types():
    """F94-discovered gap: layer_runner.run_block (called by
    forward_tokens_serial_positions) is a plain dense-transformer block with
    no awareness of qwen3_5/kimi_linear's hybrid DeltaNet/full-attention
    layer_types -- it would KeyError on 'model.layers.N.self_attn.*' tensor
    names that don't exist on a linear_attention layer. num_experts already
    excludes qwen3_5_moe (MoE); dense qwen3_5 (Qwen3.5-4B/9B, Qwen3.6-27B)
    was NOT excluded before this fix, and this reproduced live against a
    real Qwen3.6-27B checkpoint (qwen35_mtp_gate.py)."""
    from runtime.engine import StreamingEngine

    for model_type in ("qwen3_5", "qwen3_5_moe", "kimi_linear"):
        engine = object.__new__(StreamingEngine)
        engine.cfg = SimpleNamespace(num_experts=0, model_type=model_type)
        try:
            engine.forward_tokens_serial_positions([1, 2], kv=None)
            raised = False
        except ValueError:
            raised = True
        assert raised, (
            f"forward_tokens_serial_positions must refuse model_type="
            f"{model_type!r}")
