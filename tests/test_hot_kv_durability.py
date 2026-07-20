"""Crash, corruption, immutability, and scaling proofs for the v3 KV journal."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import pytest

import runtime.hot_kv_persist as persist_module
from runtime.hot_kv_persist import HotPromptKVPersistence
from runtime.kv_cache import KVCache


def _state(tokens: list[int]):
    kv = KVCache(1)
    kv.compressed_mla = True
    values = mx.arange(len(tokens) * 2, dtype=mx.float32).reshape(
        1, len(tokens), 2)
    kv.keys[0] = values
    logits = mx.array([[float(len(tokens)), 1.0, -1.0]], dtype=mx.float32)
    mx.eval(values, logits)
    return kv, logits


def _journal(tmp_path: Path, *, chunk_size: int = 2,
             max_checkpoints: int = 64, max_bytes: int = 0):
    return HotPromptKVPersistence(
        tmp_path, "durability-test", chunk_size,
        max_checkpoints=max_checkpoints, max_bytes=max_bytes)


def _save(journal, tokens, *, parent=(), covered=0,
          cache_namespace="default"):
    kv, logits = _state(tokens)
    return journal.save(
        tuple(parent), covered, tokens, kv, logits, logits,
        prompt_length=len(tokens), reusable_prefix=len(tokens),
        cache_namespace=cache_namespace)


def _flip_one_byte_same_size(path: Path) -> None:
    payload = bytearray(path.read_bytes())
    assert payload
    payload[len(payload) // 2] ^= 0x01
    path.write_bytes(payload)


def _checkpoint_ids(path: Path) -> set[str]:
    return {item.name[:-len(".ckpt.json")]
            for item in path.glob("*.ckpt.json")}


def test_same_size_segment_corruption_is_rejected_and_repaired(tmp_path):
    journal = _journal(tmp_path)
    tokens = list(range(4))
    original_chain = _save(journal, tokens)
    corrupt_id = original_chain[0]
    _flip_one_byte_same_size(journal._segment_payload_path(corrupt_id))

    assert journal.find_best_match(tokens, 2) is None
    repaired_chain = _save(journal, tokens)

    assert repaired_chain[0] != corrupt_id
    repaired_meta = journal._read_segment_meta(
        repaired_chain[0], verify_payload=True)
    assert repaired_meta is not None
    assert repaired_meta["recovery_of"] == corrupt_id
    match = journal.find_best_match(tokens, 2)
    assert match is not None
    assert journal.load_matched_chain(match, 1) is not None


@pytest.mark.parametrize("corrupt_manifest", [False, True])
def test_corrupt_checkpoint_collision_publishes_repair_generation(
        tmp_path, corrupt_manifest):
    journal = _journal(tmp_path)
    tokens = list(range(4))
    _save(journal, tokens)
    original_id = next(iter(_checkpoint_ids(tmp_path)))
    target = (journal._checkpoint_meta_path(original_id) if corrupt_manifest
              else journal._checkpoint_payload_path(original_id))
    if corrupt_manifest:
        target.write_text("{")
    else:
        _flip_one_byte_same_size(target)

    _save(journal, tokens)

    ids = _checkpoint_ids(tmp_path)
    assert original_id in ids
    assert len(ids) == 2
    repair_id = next(item for item in ids if item != original_id)
    repair = journal._read_checkpoint_meta(repair_id, verify_payload=True)
    assert repair is not None
    assert repair["recovery_of"] == original_id
    match = journal.find_best_match(tokens, 2)
    assert match is not None and match["checkpoint_id"] == repair_id


def test_corrupt_newest_generation_falls_back_to_older_prefix(tmp_path):
    journal = _journal(tmp_path)
    older_tokens = list(range(4))
    older_chain = _save(journal, older_tokens)
    newer_tokens = list(range(6))
    newer_chain = _save(
        journal, newer_tokens, parent=older_chain, covered=len(older_tokens))
    _flip_one_byte_same_size(
        journal._segment_payload_path(newer_chain[-1]))

    match = journal.find_best_match(list(range(8)), 2)

    assert match is not None
    assert match["matched"] == len(older_tokens)
    assert match["chain"] == list(older_chain)


def test_match_hash_proof_is_reused_only_while_files_are_unchanged(
        tmp_path, monkeypatch):
    journal = _journal(tmp_path)
    tokens = list(range(6))
    _save(journal, tokens)
    sha_calls = 0
    original_sha = persist_module._sha256_file

    def counted(path, chunk_bytes=8 * 1024 * 1024):
        nonlocal sha_calls
        sha_calls += 1
        return original_sha(path, chunk_bytes)

    monkeypatch.setattr(persist_module, "_sha256_file", counted)
    match = journal.find_best_match(tokens, 2)
    assert match is not None
    verified_calls = sha_calls
    assert verified_calls > 0

    assert journal.load_matched_chain(match, 1) is not None
    assert sha_calls == verified_calls, (
        "unchanged payloads were hashed again between match and load")


def test_match_hash_proof_fails_closed_after_same_size_mutation(
        tmp_path, monkeypatch):
    journal = _journal(tmp_path)
    tokens = list(range(6))
    chain = _save(journal, tokens)
    match = journal.find_best_match(tokens, 2)
    assert match is not None
    _flip_one_byte_same_size(journal._segment_payload_path(chain[-1]))
    sha_calls = 0
    original_sha = persist_module._sha256_file

    def counted(path, chunk_bytes=8 * 1024 * 1024):
        nonlocal sha_calls
        sha_calls += 1
        return original_sha(path, chunk_bytes)

    monkeypatch.setattr(persist_module, "_sha256_file", counted)
    assert journal.load_matched_chain(match, 1) is None
    assert sha_calls > 0, "changed signature did not force fresh SHA validation"


def test_committed_objects_are_never_replaced_on_exact_resave(tmp_path):
    journal = _journal(tmp_path)
    tokens = list(range(4))
    chain = _save(journal, tokens)
    committed = [
        *tmp_path.glob("*.seg.json"),
        *tmp_path.glob("*.seg.safetensors"),
        *tmp_path.glob("*.ckpt.json"),
        *tmp_path.glob("*.ckpt.safetensors"),
    ]
    before = {
        path.name: (path.stat().st_ino, path.read_bytes()) for path in committed
    }

    same_chain = _save(journal, tokens, parent=chain, covered=len(tokens))

    assert same_chain == chain
    assert set(before) == {
        path.name for path in (
            *tmp_path.glob("*.seg.json"),
            *tmp_path.glob("*.seg.safetensors"),
            *tmp_path.glob("*.ckpt.json"),
            *tmp_path.glob("*.ckpt.safetensors"),
        )
    }
    for name, (inode, payload) in before.items():
        path = tmp_path / name
        assert path.stat().st_ino == inode
        assert path.read_bytes() == payload


def test_reader_lease_prevents_gc_from_deleting_checkpoint(tmp_path):
    journal = _journal(tmp_path, max_checkpoints=1)
    first = list(range(2))
    _save(journal, first)
    first_id = next(iter(_checkpoint_ids(tmp_path)))
    _save(journal, [10, 11])
    assert len(_checkpoint_ids(tmp_path)) == 2
    paths = (
        journal._checkpoint_meta_path(first_id),
        journal._checkpoint_payload_path(first_id),
    )

    with journal._lease(paths):
        journal.gc()
        assert all(path.exists() for path in paths)

    assert all(path.exists() for path in paths)
    assert len(_checkpoint_ids(tmp_path)) == 1


def test_gc_cleans_dead_reader_lease(tmp_path, monkeypatch):
    journal = _journal(tmp_path)
    lease = journal._leases_dir / "dead.lease.json"
    lease.write_text(json.dumps({
        "format": "hot-kv-reader-lease-v1",
        "pid": 999_999_999,
        "files": ["never-created"],
    }))
    monkeypatch.setattr(journal, "_pid_alive", lambda _pid: False)

    journal.gc()

    assert not lease.exists()


def test_crash_after_segment_payload_publication_leaves_no_generation(
        tmp_path, monkeypatch):
    journal = _journal(tmp_path)
    original = persist_module._publish_json_immutable

    def crash(path, value):
        if path.name.endswith(".seg.json"):
            raise OSError("simulated crash before segment commit")
        return original(path, value)

    monkeypatch.setattr(persist_module, "_publish_json_immutable", crash)
    with pytest.raises(OSError, match="simulated crash"):
        _save(journal, list(range(4)))
    monkeypatch.setattr(persist_module, "_publish_json_immutable", original)

    assert not list(tmp_path.glob("*.seg.json"))
    assert list(tmp_path.glob("*.seg.safetensors"))
    journal.gc()
    assert not list(tmp_path.glob("*.seg.safetensors"))
    assert not list(tmp_path.glob(".*.tmp.*"))


def test_crash_before_checkpoint_commit_preserves_older_generation(
        tmp_path, monkeypatch):
    journal = _journal(tmp_path)
    older_tokens = list(range(4))
    older_chain = _save(journal, older_tokens)
    older_id = next(iter(_checkpoint_ids(tmp_path)))
    original = persist_module._publish_json_immutable

    def crash(path, value):
        if path.name.endswith(".ckpt.json"):
            raise OSError("simulated crash before checkpoint commit")
        return original(path, value)

    monkeypatch.setattr(persist_module, "_publish_json_immutable", crash)
    with pytest.raises(OSError, match="simulated crash"):
        _save(journal, list(range(6)), parent=older_chain, covered=4)
    monkeypatch.setattr(persist_module, "_publish_json_immutable", original)

    journal.gc()
    assert _checkpoint_ids(tmp_path) == {older_id}
    match = journal.find_best_match(list(range(8)), 2)
    assert match is not None and match["matched"] == 4
    assert journal.load_matched_chain(match, 1) is not None


def test_extension_writes_only_delta_and_reconstruction_concatenates_once(
        tmp_path, monkeypatch):
    journal = _journal(tmp_path)
    initial = list(range(4))
    initial_chain = _save(journal, initial)
    before = {path.name for path in tmp_path.glob("*.seg.json")}
    extended = list(range(8))
    extended_chain = _save(
        journal, extended, parent=initial_chain, covered=len(initial))
    new_manifests = [path for path in tmp_path.glob("*.seg.json")
                     if path.name not in before]

    assert len(new_manifests) == 2
    assert [json.loads(path.read_text())["tokens"]
            for path in sorted(new_manifests)] != []
    appended = []
    for seg_id in extended_chain[len(initial_chain):]:
        appended.extend(journal._read_segment_meta(seg_id)["tokens"])
    assert appended == extended[len(initial):]

    calls = 0
    concatenate = persist_module.mx.concatenate

    def counted(values, *, axis=0):
        nonlocal calls
        calls += 1
        return concatenate(values, axis=axis)

    monkeypatch.setattr(persist_module.mx, "concatenate", counted)
    loaded = journal._load_chain_prefix(list(extended_chain), 1)

    assert loaded is not None
    assert loaded[0] == extended
    assert calls == 1  # one compressed key tensor, independent of chain depth


def test_gc_enforces_reachable_byte_budget(tmp_path):
    journal = _journal(tmp_path, max_checkpoints=0)
    _save(journal, [0, 1])
    first_id = next(iter(_checkpoint_ids(tmp_path)))
    _save(journal, [10, 11])
    entries = []
    for checkpoint_id in _checkpoint_ids(tmp_path):
        meta = journal._read_checkpoint_meta(checkpoint_id)
        chain = journal._walk_chain(meta["leaf"])
        paths = {
            journal._checkpoint_meta_path(checkpoint_id),
            journal._checkpoint_payload_path(checkpoint_id),
        }
        for seg_id in chain:
            paths.update((journal._segment_meta_path(seg_id),
                          journal._segment_payload_path(seg_id)))
        entries.append(sum(path.stat().st_size for path in paths))
    journal.max_bytes = max(entries)

    journal.gc()

    assert len(_checkpoint_ids(tmp_path)) == 1
    assert first_id not in _checkpoint_ids(tmp_path)


def test_noop_gc_reads_each_segment_once_and_skips_durability_rebuild(
        tmp_path, monkeypatch):
    journal = _journal(tmp_path, max_checkpoints=64)
    chain = ()
    for end in range(2, 14, 2):
        tokens = list(range(end))
        chain = _save(
            journal, tokens, parent=chain, covered=max(0, end - 2))

    segment_count = len(list(tmp_path.glob("*.seg.json")))
    reads = 0
    original_read = journal._read_segment_meta

    def counted(seg_id, *, verify_payload=False):
        nonlocal reads
        reads += 1
        return original_read(seg_id, verify_payload=verify_payload)

    rebuilds = 0

    def counted_rebuild():
        nonlocal rebuilds
        rebuilds += 1

    fsyncs = 0

    def counted_fsync(_path):
        nonlocal fsyncs
        fsyncs += 1

    monkeypatch.setattr(journal, "_read_segment_meta", counted)
    monkeypatch.setattr(journal, "_rebuild_segment_index", counted_rebuild)
    monkeypatch.setattr(persist_module, "_fsync_dir", counted_fsync)

    assert journal.gc() == 0
    assert reads == segment_count
    assert rebuilds == 0
    assert fsyncs == 0


def test_approximate_endpoint_label_survives_restart_load(tmp_path):
    journal = _journal(tmp_path)
    tokens = list(range(4))
    kv, logits = _state(tokens)
    journal.save(
        (), 0, tokens, kv, logits, logits,
        prompt_length=len(tokens), reusable_prefix=0, approximate=True)

    loaded = _journal(tmp_path).load_all(num_layers=1, limit=1)

    assert len(loaded) == 1
    assert loaded[0][6] is True


def test_tool_capsule_spans_survive_restart_load(tmp_path):
    journal = _journal(tmp_path)
    tokens = list(range(8))
    kv, logits = _state(tokens)
    capsules = (("weather-v1", 1, 4), ("search-v2", 5, 8))
    journal.save(
        (), 0, tokens, kv, logits, logits,
        prompt_length=len(tokens), reusable_prefix=len(tokens),
        tool_capsules=capsules)

    loaded = _journal(tmp_path).load_all(num_layers=1, limit=1)

    assert len(loaded) == 1
    assert loaded[0][7] == capsules


def test_decision_and_execution_namespaces_both_survive_one_slot_disk_tier(
        tmp_path):
    """RAM capacity must not decide which hidden phase remains durable."""
    journal = _journal(tmp_path, max_checkpoints=8)
    decision_tokens = list(range(8))
    execution_tokens = list(range(20, 30))
    _save(journal, decision_tokens, cache_namespace="gateway_decision")
    _save(journal, execution_tokens, cache_namespace="gateway_execution")

    # Both checkpoint generations coexist even though an engine may load only
    # one of them into a slots=1 memory tier.
    checkpoint_metas = [
        json.loads(path.read_text()) for path in tmp_path.glob("*.ckpt.json")]
    assert {meta["cache_namespace"] for meta in checkpoint_metas} == {
        "gateway_decision", "gateway_execution"}

    assert journal.find_best_match(
        decision_tokens, 2, cache_namespace="gateway_execution") is None
    decision_match = journal.find_best_match(
        decision_tokens, 2, cache_namespace="gateway_decision")
    execution_match = journal.find_best_match(
        execution_tokens, 2, cache_namespace="gateway_execution")
    assert decision_match is not None
    assert execution_match is not None
    assert journal.load_matched_chain(decision_match, 1) is not None
    assert journal.load_matched_chain(execution_match, 1) is not None

    loaded = _journal(tmp_path).load_all(num_layers=1, limit=2)
    assert {entry[9] for entry in loaded} == {
        "gateway_decision", "gateway_execution"}


def test_invalid_persisted_tool_capsule_spans_fail_closed(tmp_path):
    journal = _journal(tmp_path)
    tokens = list(range(4))
    kv, logits = _state(tokens)

    with pytest.raises(ValueError, match="tool capsule"):
        journal.save(
            (), 0, tokens, kv, logits, logits,
            prompt_length=len(tokens), reusable_prefix=len(tokens),
            tool_capsules=(("outside", 2, 5),))
