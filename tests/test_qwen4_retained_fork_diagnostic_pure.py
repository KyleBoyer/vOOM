"""No-MLX ownership, sampling-order and metadata regression tests."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from tests.fixtures.qwen4_retained_fork_diagnostic import (
    complete_state_metadata, diagnose_retained_fork)


class Array:
    def __init__(self, payload=b"\x80\x00\xff\x7f"):
        self.payload = payload
        self.shape = (1, 1, 2)
        self.dtype = "bf16"
        self.nbytes = len(payload)


class KV:
    def __init__(self, offset, shared):
        self.offset = offset
        self.keys = [Array()]
        self.values = [Array()]
        self._starts = [0]
        self._windows = [None]
        self.compressed_mla = False
        self.kda_cache = object()
        self.qwen4_cache = SimpleNamespace(
            ple_conv=[shared], qsa_pooled_keys=[None], qsa_pool_cache_enabled=False)


def setup():
    shared = Array()
    endpoint, fork = KV(6, shared), KV(4, shared)
    slot = SimpleNamespace(kv=fork, tokens=(1, 2, 3, 4), qwen4_retention_tile=4,
                           logits=None, prompt_logits=None, exact_hidden=None,
                           approximate=False)
    target = SimpleNamespace(_hot_prompt_slots=[slot], _hot_kv_persist=None,
                             last_kv=endpoint)
    return target, endpoint, fork, shared


def metadata(kv):
    return complete_state_metadata(kv, array_digest=lambda a: {
        "sha256": hashlib.sha256(a.payload).hexdigest(), "shape": a.shape,
        "dtype": a.dtype})


def run(target, *, bad_copy=False, same_copy=False):
    events = []
    live = [1000]

    def synchronize(values):
        events.append("synchronize")
        live[0] = 900

    def digest(kv):
        events.append("hash")
        raw = b"".join(a.payload for a in kv.qwen4_cache.ple_conv)
        return hashlib.sha256(raw).hexdigest(), 1, len(raw), {"ple": raw.hex()}

    def copy(value):
        events.append("copy")
        live[0] = 700
        return value if same_copy else Array(b"bad!" if bad_copy else value.payload)

    def sample_memory():
        events.append("sample")
        return {"active_bytes": live[0], "cache_bytes": 123, "allocator_peak_bytes": 2000}

    report = diagnose_retained_fork(
        target, (1, 2, 3, 4, 5, 6), expected_prefix=4, kv_type=KV,
        state_digest=digest, metadata_digest=metadata, copy_ple=copy,
        synchronize_ple=synchronize, pressure=lambda: {"available_bytes": 10_000},
        metal_memory=sample_memory, clear_cache=lambda: events.append("clear"))
    return report, events


def test_materialization_precedes_any_hash_and_only_fork_ple_is_replaced():
    target, endpoint, fork, shared = setup()
    report, events = run(target)
    assert events[:4] == ["sample", "synchronize", "sample", "hash"]
    assert report["fork_equal"] and report["endpoint_equal"]
    assert endpoint.qwen4_cache.ple_conv[0] is shared
    assert fork.qwen4_cache.ple_conv[0] is not shared
    assert fork.qwen4_cache.ple_conv[0].payload == shared.payload
    assert report["copied_arrays"] == 1 and report["copied_bytes"] == 4
    assert report["synchronize_active_released_bytes"] == 100
    assert report["detach_active_released_bytes"] == 200
    assert [s["stage"] for s in report["stages"]] == [
        "before_ple_synchronize", "after_ple_synchronize", "after_before_hashes",
        "after_ple_detach", "after_allocator_clear", "after_verification_hashes"]
    assert report["serving_timing_proof"] is False
    assert report["serving_pressure_proof"] is False


def test_corrupted_payload_is_reported_as_failed_equality():
    report, _ = run(setup()[0], bad_copy=True)
    assert report["fork_equal"] is False
    assert report["endpoint_equal"] is True


def test_aliasing_replacement_is_rejected():
    with pytest.raises(ValueError, match="independent"):
        run(setup()[0], same_copy=True)


@pytest.mark.parametrize("change", [
    "no_slot", "two_slots", "persisted", "same_kv", "shared_aux",
    "shared_ple_list", "shared_keys", "shared_kda", "wrong_prefix",
    "wrong_offset", "unmarked", "raw_logits", "approximate",
])
def test_invalid_ownership_or_prefix_fails_before_sampling(change):
    target, endpoint, fork, _ = setup()
    slot = target._hot_prompt_slots[0]
    if change == "no_slot": target._hot_prompt_slots = []
    elif change == "two_slots": target._hot_prompt_slots.append(slot)
    elif change == "persisted": target._hot_kv_persist = object()
    elif change == "same_kv": slot.kv = endpoint
    elif change == "shared_aux": fork.qwen4_cache = endpoint.qwen4_cache
    elif change == "shared_ple_list": fork.qwen4_cache.ple_conv = endpoint.qwen4_cache.ple_conv
    elif change == "shared_keys": fork.keys = endpoint.keys
    elif change == "shared_kda": fork.kda_cache = endpoint.kda_cache
    elif change == "wrong_prefix": slot.tokens = (9, 2, 3, 4)
    elif change == "wrong_offset": fork.offset = 3
    elif change == "unmarked": slot.qwen4_retention_tile = 0
    elif change == "raw_logits": slot.logits = object()
    elif change == "approximate": slot.approximate = True
    with pytest.raises(ValueError):
        run(target)


@pytest.mark.parametrize("change", ["start", "window", "pool_policy", "pool_array"])
def test_additional_metadata_changes_are_detected(change):
    _, _, fork, _ = setup()
    before = metadata(fork)
    if change == "start": fork._starts[0] = 1
    elif change == "window": fork._windows[0] = 128
    elif change == "pool_policy": fork.qwen4_cache.qsa_pool_cache_enabled = True
    elif change == "pool_array": fork.qwen4_cache.qsa_pooled_keys[0] = Array()
    assert metadata(fork)["sha256"] != before["sha256"]


def test_incomplete_metadata_is_rejected():
    _, _, fork, _ = setup()
    fork._starts.clear()
    with pytest.raises(ValueError, match="incomplete"):
        metadata(fork)


def test_report_contains_hashes_not_metadata_payloads():
    report, _ = run(setup()[0])
    encoded = json.dumps(report)
    assert '"starts"' not in encoded and '"windows"' not in encoded
