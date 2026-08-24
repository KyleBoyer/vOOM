"""Tiny exact gates for disk-only paged Qwen hybrid prefix persistence."""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from runtime.engine import RuntimeConfig, StreamingEngine
from runtime.hot_kv_persist import HotPromptKVPersistence
from runtime.kda_state import KDAStateCache
from runtime.kv_paged import PagedKVCache


_LAYERS = (
    "linear_attention",
    "full_attention",
    "linear_attention",
    "full_attention",
)


def test_paged_persist_architecture_validation_runs_after_config_load(tmp_path):
    non_qwen = Path(__file__).resolve().parent.parent / "models/glm-fixture-tiny"
    rc = RuntimeConfig(
        prefill_chunk_size=128,
        max_kv_mb=1,
        kv_spill_dir=str(tmp_path / "spill"),
        hot_prompt_kv=True,
        hot_prompt_kv_chunk_size=128,
        hot_prompt_kv_persist_dir=str(tmp_path / "journal"),
        paged_kv_persist=True,
        release_paged_kv_after_generate=True,
    )
    with pytest.raises(ValueError, match="requires a qwen3_5/qwen3_5_moe"):
        StreamingEngine(non_qwen, rc)


def _config():
    return SimpleNamespace(
        model_type="qwen3_5",
        layer_types=_LAYERS,
        kda_layers=(),
    )


def _cache(path, length: int) -> PagedKVCache:
    kv = PagedKVCache(
        len(_LAYERS), max_bytes=1, spill_dir=path,
        page_positions=4, resident_pages=0)
    for layer in (1, 3):
        base = mx.arange(length * 6, dtype=mx.float32).reshape(
            1, 2, length, 3)
        kv.update(layer, base + layer * 100, base + layer * 100 + 50)
    recurrent = KDAStateCache(len(_LAYERS))
    for layer in (0, 2):
        recurrent.set_state(
            layer,
            mx.arange(24, dtype=mx.float32).reshape(1, 2, 3, 4)
            + length * 10 + layer,
        )
        recurrent.set_conv_history(
            layer,
            (mx.arange(12, dtype=mx.float32).reshape(1, 2, 2, 3)
             + length + layer,),
        )
    kv.kda_cache = recurrent
    recurrent.synchronize()
    return kv


def _journal(path, spill, fingerprint="paged-qwen-test"):
    return HotPromptKVPersistence(
        path, fingerprint, 4, config=_config(), require_recurrent=True,
        paged_cache_factory=lambda layers: PagedKVCache(
            layers, max_bytes=1, spill_dir=spill,
            page_positions=4, resident_pages=0),
    )


def _save_stable(journal, tokens, kv):
    return journal.save(
        (), 0, tokens, kv, None, None,
        prompt_length=len(tokens), reusable_prefix=len(tokens),
        checkpoint_kind="stable_prefix")


def _assert_exact(actual, expected):
    assert actual.offset == expected.offset
    for layer in (1, 3):
        actual_k, actual_v = actual.materialize_layer(layer)
        expected_k, expected_v = expected.materialize_layer(layer)
        assert np.array_equal(np.array(actual_k), np.array(expected_k))
        assert np.array_equal(np.array(actual_v), np.array(expected_v))
    for layer in (0, 2):
        assert np.array_equal(
            np.array(actual.kda_cache.state(layer)),
            np.array(expected.kda_cache.state(layer)))
        assert np.array_equal(
            np.array(actual.kda_cache.conv_history(layer)[0]),
            np.array(expected.kda_cache.conv_history(layer)[0]))


def test_paged_stable_prefix_restores_bounded_kv_and_all_recurrent_state(
        tmp_path):
    tokens = list(range(8))
    expected = _cache(tmp_path / "source-spill", len(tokens))
    journal = _journal(tmp_path / "journal", tmp_path / "restore-spill")
    _save_stable(journal, tokens, expected)

    # A state-only boundary cannot answer an identical request and cannot be
    # rewound for an edited branch. Only a strict extension is exact.
    assert journal.find_best_match(tokens, 4) is None
    assert journal.find_best_match(tokens[:6] + [999, 1000], 4) is None
    match = journal.find_best_match(tokens + [1000], 4)
    assert match is not None
    assert match["case"] == "extension"
    assert match["matched"] == len(tokens)

    loaded = journal.load_matched_chain(match, len(_LAYERS))
    assert loaded is not None
    loaded_tokens, actual, exact_logits = loaded
    assert loaded_tokens == tuple(tokens)
    assert exact_logits is None
    assert isinstance(actual, PagedKVCache)
    assert actual.nbytes() == 0
    _assert_exact(actual, expected)

    # Extension starts exactly at the restored endpoint. Both conventional KV
    # and recurrent state remain owned by the same request-local cache.
    for layer in (1, 3):
        k = mx.full((1, 2, 1, 3), 700 + layer, dtype=mx.float32)
        actual.update(layer, k, k + 1)
    assert actual.offset == 9
    assert actual.layer_positions(1) == actual.layer_positions(3) == 9


def test_two_paged_restores_use_disjoint_spills_and_release_independently(
        tmp_path):
    tokens = list(range(8))
    journal = _journal(tmp_path / "journal", tmp_path / "restore-spill")
    _save_stable(journal, tokens, _cache(tmp_path / "source", len(tokens)))
    match = journal.find_best_match(tokens + [90], 4)
    first = journal.load_matched_chain(match, len(_LAYERS))[1]
    second = journal.load_matched_chain(match, len(_LAYERS))[1]

    first_paths = {
        page.path for pages in first._pages for page in pages if page.path}
    second_paths = {
        page.path for pages in second._pages for page in pages if page.path}
    assert first_paths and second_paths and first_paths.isdisjoint(second_paths)
    first.release()
    assert all(path.exists() for path in second_paths)
    second.materialize_layer(1)


def test_corruption_and_fingerprint_mismatch_fail_before_paged_restore(tmp_path):
    tokens = list(range(8))
    journal_dir = tmp_path / "journal"
    writer = _journal(journal_dir, tmp_path / "spill-a")
    _save_stable(writer, tokens, _cache(tmp_path / "source", len(tokens)))

    segment = next(journal_dir.glob("*.seg.safetensors"))
    damaged = bytearray(segment.read_bytes())
    damaged[len(damaged) // 2] ^= 1
    segment.write_bytes(damaged)
    assert _journal(
        journal_dir, tmp_path / "spill-b",
    ).find_best_match(tokens + [90], 4) is None
    assert _journal(
        journal_dir, tmp_path / "spill-c", fingerprint="other-runtime",
    ).find_best_match(tokens + [90], 4) is None
    assert not list((tmp_path / "spill-b").glob("kv_*.safetensors"))


def test_checkpoint_corruption_after_match_releases_request_local_pages(tmp_path):
    tokens = list(range(8))
    journal_dir = tmp_path / "journal"
    restore_spill = tmp_path / "restore-spill"
    journal = _journal(journal_dir, restore_spill)
    _save_stable(journal, tokens, _cache(tmp_path / "source", len(tokens)))
    match = journal.find_best_match(tokens + [90], 4)
    assert match is not None

    checkpoint = match["ckpt_payload"]
    damaged = bytearray(checkpoint.read_bytes())
    damaged[len(damaged) // 2] ^= 1
    checkpoint.write_bytes(damaged)
    assert journal.load_matched_chain(match, len(_LAYERS)) is None
    # Segment tensors were restored before checkpoint recurrent state was
    # checked. Failure must clean those request-local copies, not leak a full
    # prefix into the shared spill tier.
    assert not list(restore_spill.glob("kv_*.safetensors"))
