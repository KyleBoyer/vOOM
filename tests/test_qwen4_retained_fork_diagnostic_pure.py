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


def run(target, *, bad_copy=False, same_copy=False, include_kda=False,
        bad_kda=False, kda_geometry=None):
    events = []
    live = [1000]

    def synchronize(values):
        events.append("synchronize")
        live[0] = 900

    def digest(kv):
        events.append("hash")
        raw = b"".join(a.payload for a in kv.qwen4_cache.ple_conv)
        for history in getattr(kv.kda_cache, "_conv", ()):
            if history is not None:
                raw += b"".join(a.payload for a in history if a is not None)
        return hashlib.sha256(raw).hexdigest(), 1, len(raw), {"ple": raw.hex()}

    def copy(value):
        events.append("copy")
        live[0] -= 200
        corrupt = bad_copy or (bad_kda and value.payload == b"kda!")
        return value if same_copy else Array(b"bad!" if corrupt else value.payload)

    def sample_memory():
        events.append("sample")
        return {"active_bytes": live[0], "cache_bytes": 123, "allocator_peak_bytes": 2000}

    report = diagnose_retained_fork(
        target, (1, 2, 3, 4, 5, 6), expected_prefix=4, kv_type=KV,
        state_digest=digest, metadata_digest=metadata, copy_ple=copy,
        synchronize_ple=synchronize, pressure=lambda: {"available_bytes": 10_000},
        metal_memory=sample_memory, clear_cache=lambda: events.append("clear"),
        include_kda_conv=include_kda,
        **({"expected_kda_layers": (0,), "expected_kda_shape": (1, 1, 2),
            "expected_kda_dtype": "bf16", **(kda_geometry or {})}))
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


def setup_kda():
    target, endpoint, fork, shared = setup()
    conv = Array(b"kda!")
    recurrent = object()  # Deliberately not an array: never sent to bit copier.
    for kv in (endpoint, fork):
        kv.kda_cache = SimpleNamespace(_conv=[(conv,)], _state=[recurrent],
                                       _spill_meta={}, _factor_capture=None,
                                       spill_enabled=False)
    return target, endpoint, fork, shared, conv, recurrent


def test_all_conv_detach_preserves_authoritative_owner_and_recurrent_matrix():
    target, endpoint, fork, shared, conv, recurrent = setup_kda()
    old_tuple = fork.kda_cache._conv[0]
    report, events = run(target, include_kda=True)
    assert report["fork_equal"] and report["endpoint_equal"]
    assert events.index("hash") < events.index("copy")
    assert report["kda_conv_included"] is True
    assert report["copied_arrays"] == report["kda_copied_arrays"] == 1
    assert report["copied_bytes"] == report["kda_copied_bytes"] == 4
    assert report["kda_copy_host_read_bytes"] == 4
    assert report["detach_active_released_bytes"] == 200
    assert report["kda_detach_active_released_bytes"] == 200
    assert endpoint.kda_cache._conv[0][0] is old_tuple[0] is conv
    assert fork.kda_cache._conv[0] is not old_tuple
    assert fork.kda_cache._conv[0][0] is not conv
    assert fork.kda_cache._conv[0][0].payload == conv.payload
    assert fork.kda_cache._state[0] is endpoint.kda_cache._state[0] is recurrent
    assert endpoint.qwen4_cache.ple_conv[0] is shared
    assert [s["stage"] for s in report["stages"]][-4:] == [
        "after_ple_detach", "after_kda_detach", "after_allocator_clear",
        "after_verification_hashes"]


def test_kda_corruption_fails_full_fork_equality_without_changing_endpoint():
    report, _ = run(setup_kda()[0], include_kda=True, bad_kda=True)
    assert report["fork_equal"] is False and report["endpoint_equal"] is True


def test_default_mode_does_not_detach_kda_or_add_a_kda_stage():
    target, endpoint, fork, _, conv, _ = setup_kda()
    report, _ = run(target)
    assert report["kda_conv_included"] is False
    assert report["kda_copied_arrays"] == report["kda_copied_bytes"] == 0
    assert not report["logical_kda_conv"]
    assert fork.kda_cache._conv[0][0] is endpoint.kda_cache._conv[0][0] is conv
    assert "after_kda_detach" not in [s["stage"] for s in report["stages"]]


@pytest.mark.parametrize("change", [
    "shared_list", "missing_list", "short_list", "mutable_history",
    "spilled", "spill_metadata", "factor_capture", "wrong_shape",
    "wrong_dtype", "missing_history", "two_arrays", "null_array",
])
def test_all_conv_rejects_unsupported_history_before_any_ple_mutation(change):
    target, endpoint, fork, shared, _, _ = setup_kda()
    kda = fork.kda_cache
    if change == "shared_list": kda._conv = endpoint.kda_cache._conv
    elif change == "missing_list": del kda._conv
    elif change == "short_list": kda._conv = []
    elif change == "mutable_history": kda._conv[0] = list(kda._conv[0])
    elif change == "spilled": kda.spill_enabled = True
    elif change == "spill_metadata": kda._spill_meta = {0: {}}
    elif change == "factor_capture": kda._factor_capture = []
    elif change == "wrong_shape": kda._conv[0][0].shape = (2, 1, 1)
    elif change == "wrong_dtype": kda._conv[0][0].dtype = "fp32"
    elif change == "missing_history": kda._conv[0] = None
    elif change == "two_arrays": kda._conv[0] = (Array(), Array())
    elif change == "null_array": kda._conv[0] = (None,)
    with pytest.raises(ValueError, match="KDA"):
        run(target, include_kda=True)
    assert fork.qwen4_cache.ple_conv[0] is shared


@pytest.mark.parametrize("geometry", [
    {"expected_kda_layers": ()}, {"expected_kda_layers": (0, 0)},
    {"expected_kda_layers": (1,)}, {"expected_kda_layers": (True,)},
    {"expected_kda_shape": None}, {"expected_kda_dtype": None},
])
def test_all_conv_requires_expected_geometry_not_just_nonempty_copy(geometry):
    with pytest.raises(ValueError, match="KDA"):
        run(setup_kda()[0], include_kda=True, kda_geometry=geometry)


def test_cli_requires_prefix_when_kda_mode_is_selected(tmp_path):
    from tests.fixtures.qwen4_hot_boundary_http_probe import main
    with pytest.raises(SystemExit) as error:
        main(["--artifact", str(tmp_path / "never.json"), "--label", "guard",
              "--expected-prompt-tokens", "6", "--profile", "unused",
              "--diagnose-retained-kda-conv"])
    assert error.value.code == 2
