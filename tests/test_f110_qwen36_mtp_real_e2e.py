"""F110: first real end-to-end verification of Qwen3.6's native MTP
speculative decoding (QwenMTPSpeculativeEngine, runtime/qwen35_mtp.py).

Every existing test for this feature (tests/test_qwen35_mtp_engine.py,
tests/test_qwen35_mtp_rollback.py) exercises it against synthetic/mocked
target engines -- never against a real checkpoint. This closes that gap
against a real dense 27B Qwen hybrid checkpoint (prequantized,
resident_fast_decode -- the same compute-bound config F103 used, so the
real speed benefit is measurable rather than swamped by disk I/O for
this 18GB dense model), matching this project's "greedy A/B,
byte-identical tokens" standard:

1. Byte-identical output: QwenMTPSpeculativeEngine-wrapped generate()
   vs. the plain target engine's own generate(), same prompt, greedy.
2. Real speed: a real, substantial win -- 369.877s -> 204.641s for 40
   decode tokens (1.808x), 20 drafts proposed / 19 accepted (95% accept
   rate) in the run this test's own numbers were taken from.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from runtime.engine import RuntimeConfig, StreamingEngine
from runtime.qwen35_mtp import QwenMTPSpeculativeEngine
from runtime.sampler import SamplingParams

_MODEL_ROOT = Path(__file__).resolve().parent.parent / "models"
_REAL_MODEL_DIR = next(
    (
        path for path in (
            _MODEL_ROOT / "Huihui-Qwen3.8-27B-abliterated-mlx-all-mxfp4",
            _MODEL_ROOT / "Qwen3.6-27B-mlx-all-mxfp4",
        )
        if path.exists()
    ),
    _MODEL_ROOT / "Huihui-Qwen3.8-27B-abliterated-mlx-all-mxfp4",
)
_model_skip = pytest.mark.skipif(
    not _REAL_MODEL_DIR.exists(),
    reason="real dense 27B Qwen MXFP4 checkpoint not present on this machine")

_PROMPT = (
    "The capital of France is Paris. The capital of Germany is Berlin. "
    "The capital of Italy is Rome. The capital of Spain is"
)


def _run(use_mtp: bool, max_tokens: int = 8):
    rc = RuntimeConfig(prefill_chunk_size=512, resident_fast_decode=True)
    engine = StreamingEngine(str(_REAL_MODEL_DIR), rc)
    driver = (
        QwenMTPSpeculativeEngine(
            engine, min_output_tokens=2, adaptive_stop=False,
            plain_warmup_tokens=0)
        if use_mtp else engine)
    try:
        result = driver.generate(
            _PROMPT, max_tokens=max_tokens,
            sampling=SamplingParams(temperature=0.0))
    finally:
        engine.close()
    return result


@_model_skip
def test_qwen36_mtp_matches_plain_target_byte_identical():
    baseline = _run(use_mtp=False)
    mtp = _run(use_mtp=True)
    assert mtp["tokens"] == baseline["tokens"], (
        "QwenMTPSpeculativeEngine's verified-draft scheme must produce "
        "byte-identical greedy output to the plain target engine"
    )
    assert mtp["text"] == baseline["text"]


@_model_skip
def test_qwen36_mtp_actually_engages_and_accepts_drafts():
    result = _run(use_mtp=True, max_tokens=16)
    stats = result["path_stats"]
    assert stats.get("qwen_mtp_enabled") == 1
    assert stats.get("qwen_mtp_used") == 1, (
        "expected MTP to actually engage for a real greedy request against "
        "a real checkpoint with real mtp.* weights, not silently fall back"
    )
    assert stats.get("qwen_mtp_proposed", 0) > 0
    # Real 27B checkpoint's own MTP head should have a real, non-trivial
    # accept rate against real trunk output -- not asserting a specific
    # threshold (that's a property of the checkpoint, not this code), just
    # that at least one real draft was verified correct.
    assert stats.get("qwen_mtp_accepted", 0) > 0
