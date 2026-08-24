from __future__ import annotations

import math
import json
import hashlib
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from runtime.lm_head_stream import (
    StreamedLMHead, lm_head_source_identity, open_verified_exact_lm_head)
from runtime.quant import QTensor, make_row_paged_reranked_q_head, matmul


def _make_head(tmp_path, *, vocab: int = 19, hidden: int = 8, block_rows: int = 5):
    values = np.arange(vocab * hidden, dtype=np.float32).reshape(vocab, hidden)
    values = np.sin(values / 11.0).astype(np.float32)
    weight = mx.array(values).astype(mx.bfloat16)
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"),
        {"lm_head.weight": weight},
    )
    return StreamedLMHead(
        tmp_path,
        {"lm_head.weight": "model.safetensors"},
        block_rows=block_rows,
    )


def test_serial_rows_are_bit_exact_to_independent_streamed_matmuls(tmp_path):
    head = _make_head(tmp_path)
    hidden = mx.array(
        np.cos(np.arange(3 * 8, dtype=np.float32).reshape(1, 3, 8) / 7.0)
    ).astype(mx.bfloat16)
    try:
        expected = mx.concatenate(
            [head.logits(hidden[:, row : row + 1]) for row in range(3)],
            axis=1,
        )
        actual = head.logits_serial_rows(hidden)
        mx.eval(expected, actual)
        assert actual.shape == (1, 3, 19)
        assert np.array_equal(
            np.array(actual.view(mx.uint16)),
            np.array(expected.view(mx.uint16)),
        )
    finally:
        head.close()


def test_serial_rows_read_each_vocab_block_once(tmp_path, monkeypatch):
    import runtime.lm_head_stream as module

    block_rows = 5
    head = _make_head(tmp_path, block_rows=block_rows)
    hidden = mx.ones((1, 4, 8), dtype=mx.bfloat16)
    calls = 0
    original = module._pread_exact

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_pread_exact", counted)
    try:
        result = head.logits_serial_rows(hidden)
        mx.eval(result)
    finally:
        head.close()

    assert calls == math.ceil(19 / block_rows)


def test_candidate_rows_are_sorted_coalesced_and_exact(tmp_path, monkeypatch):
    import runtime.lm_head_stream as module

    head = _make_head(tmp_path)
    hidden = mx.array(
        np.cos(np.arange(2 * 8, dtype=np.float32).reshape(1, 2, 8) / 7.0)
    ).astype(mx.bfloat16)
    candidates = mx.array([[[8, 1, 2, 7], [9, 2, 8, 1]]])
    reads = []
    original = module._pread_exact

    def counted(fd, size, offset):
        reads.append((size, offset))
        return original(fd, size, offset)

    monkeypatch.setattr(module, "_pread_exact", counted)
    try:
        actual = head.candidate_logits(hidden, candidates)
        full = head.logits(hidden)
        expected = mx.take_along_axis(full, candidates, axis=-1)
        mx.eval(actual, expected)
        assert np.array_equal(
            np.asarray(actual.view(mx.uint16)),
            np.asarray(expected.view(mx.uint16)))
        # Union {1,2,7,8,9} becomes two contiguous physical reads.
        assert len(reads) == 2 + math.ceil(19 / head.block_rows)
        stats = head.candidate_telemetry()
        assert stats == {
            "candidate_read_calls": 1,
            "candidate_read_extents": 2,
            "candidate_rows_requested": 8,
            "candidate_unique_rows_read": 5,
            "candidate_bytes_read": 5 * 8 * 2,
            "candidate_recall_full_scan_calls": 0,
            "candidate_recall_full_scan_bytes": 0,
        }
    finally:
        head.close()


def test_row_paged_reranker_uses_quantized_target_and_streamed_exact_rows(
        tmp_path):
    head = _make_head(tmp_path, vocab=19, hidden=32)
    exact = mx.load(str(tmp_path / "model.safetensors"))["lm_head.weight"]
    packed = mx.quantize(exact, group_size=32, bits=4, mode="mxfp4")
    approx = QTensor(
        packed[0], packed[1], packed[2] if len(packed) > 2 else None,
        4, 32, "mxfp4")
    reranked = make_row_paged_reranked_q_head(
        approx, head, candidates=19, recall_probe_every=1)
    hidden = mx.array(np.cos(np.arange(32, dtype=np.float32) / 7.0)).reshape(
        1, 1, 32).astype(mx.bfloat16)
    try:
        got = matmul(hidden, reranked)
        expected = hidden @ exact.T
        mx.eval(got, expected)
        assert np.array_equal(
            np.asarray(got.view(mx.uint16)),
            np.asarray(expected.view(mx.uint16)))
        telemetry = reranked.telemetry_snapshot()
        assert telemetry["calls"] == 1
        assert telemetry["positions"] == 1
        assert telemetry["candidate_read_calls"] == 1
        assert telemetry["candidate_recall_probes"] == 1
        assert telemetry["candidate_recall_hits"] == 1
        assert telemetry["candidate_recall_full_scan_calls"] == 1
        assert telemetry["candidate_recall_full_scan_bytes"] == 19 * 32 * 2
    finally:
        head.close()


def test_qwen_serial_verifier_reranks_every_target_position(tmp_path):
    """The depth-1 MTP verifier feeds catchup+draft in one layer sweep.

    Its LM-head tail must nevertheless call the row-paged exact-candidate
    projection once for each authoritative position, rather than falling
    through a batched quantized-only projection.
    """
    from runtime.engine import StreamingEngine

    provider = _make_head(tmp_path, vocab=19, hidden=32)
    exact = mx.load(str(tmp_path / "model.safetensors"))["lm_head.weight"]
    packed = mx.quantize(exact, group_size=32, bits=4, mode="mxfp4")
    approx = QTensor(
        packed[0], packed[1], packed[2] if len(packed) > 2 else None,
        4, 32, "mxfp4")
    reranked = make_row_paged_reranked_q_head(
        approx, provider, candidates=19)
    hidden = mx.array(np.cos(
        np.arange(2 * 32, dtype=np.float32).reshape(1, 2, 32) / 7.0
    )).astype(mx.bfloat16)

    engine = StreamingEngine.__new__(StreamingEngine)
    engine.cfg = SimpleNamespace(
        model_type="qwen3_5", num_experts=0, num_hidden_layers=0,
        rms_norm_eps=1e-6, tie_word_embeddings=False)
    engine._serial_kda_endpoints = None
    engine._serial_kda_endpoint_retained_bytes = 0
    engine._serial_kda_factors = None
    engine._serial_kda_factor_retained_bytes = 0
    engine._prefill_layer_transient = 0
    engine._decode_layer_transient = 0
    engine._serial_verify_layer_transient = {}
    engine._embed = lambda _tokens: hidden
    engine._norm_w = mx.ones((32,), dtype=mx.bfloat16)
    engine._lm_head_w = reranked
    engine._streamed_lm_head = None
    engine._tied_lm_head_w = None
    kv = SimpleNamespace(offset=0, kda_cache=None)

    try:
        logits = engine.forward_tokens_serial_positions([4, 9], kv)
        mx.eval(logits)
        telemetry = reranked.telemetry_snapshot()
        assert logits.shape == (2, 19)
        assert telemetry["calls"] == 2
        assert telemetry["positions"] == 2
        assert telemetry["candidate_read_calls"] == 2
        assert telemetry["candidate_rows_requested"] == 38
    finally:
        provider.close()


def _write_release_identity(source, target):
    for path in (source, target):
        (path / "tokenizer.json").write_text("same-tokenizer")
        (path / "tokenizer_config.json").write_text("same-config")
    shard = source / "model.safetensors"
    release = source / ".cache" / "huggingface" / "trees"
    release.mkdir(parents=True)
    (release / "revision.json").write_text(json.dumps({
        "files": {shard.name: {
            "lfs_sha256": hashlib.sha256(b"release-fixture").hexdigest(),
            "lfs_size": shard.stat().st_size,
        }},
    }))
    metadata = source / ".cache" / "huggingface" / "download"
    metadata.mkdir(parents=True)
    lfs_sha = hashlib.sha256(b"release-fixture").hexdigest()
    (metadata / f"{shard.name}.metadata").write_text(
        f"revision\n{lfs_sha}\n0\n")


def test_verified_exact_source_fails_closed_on_fingerprint_mismatch(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    weight = mx.ones((19, 32), dtype=mx.bfloat16)
    mx.save_safetensors(str(source / "model.safetensors"), {
        "lm_head.weight": weight})
    (source / "config.json").write_text('{"model_type":"fixture"}')
    _write_release_identity(source, target)
    (target / "config.json").write_text(json.dumps({
        "voom_quantization": {"source": str(source.resolve())},
    }))
    identity = lm_head_source_identity(source)
    assert identity.verified_release_hash
    with np.testing.assert_raises_regex(ValueError, "fingerprint mismatch"):
        open_verified_exact_lm_head(target, source, "0" * 64)

    head = open_verified_exact_lm_head(
        target, source, identity.fingerprint, block_rows=4)
    try:
        assert (head.vocab, head.hidden, head.dtype) == (19, 32, "BF16")
    finally:
        head.close()
