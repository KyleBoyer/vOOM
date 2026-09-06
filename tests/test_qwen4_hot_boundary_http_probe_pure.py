"""No-MLX first-token HTTP diagnostic schema/ownership regressions."""

from __future__ import annotations

import builtins
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import sys
from types import ModuleType, SimpleNamespace

import pytest


FIXTURE = Path(__file__).resolve().parent / "fixtures/qwen4_hot_boundary_http_probe.py"
SPEC = importlib.util.spec_from_file_location("qwen4_hot_boundary_probe_pure", FIXTURE)
probe_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe_module)


def test_import_does_not_load_array_runtime_or_model_modules(monkeypatch):
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert not name.startswith(("mlx", "numpy", "psutil", "runtime"))
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    module = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(module)


@pytest.mark.parametrize("invalid", [None, "1", [True], [1.0], [-1], {1}])
def test_token_hash_requires_actual_integer_ids(invalid):
    with pytest.raises(ValueError):
        probe_module._token_digest(invalid)


def test_token_hash_matches_server_generation_witness_encoding():
    result = probe_module._token_digest((0, 248068, 11))
    assert result == {
        "count": 3, "encoding": "compact-json-integer-array-v1",
        "sha256": hashlib.sha256(b"[0,248068,11]").hexdigest(),
    }


def test_hidden_hash_observes_bf16_bits_without_casting_or_evaluation():
    raw = b"\x00\x80\x01\x7f\x80\xff\x01\x00"
    views = []

    class Array:
        dtype = "bfloat16"
        shape = (1, 1, 4)

        def view(self, dtype):
            views.append(dtype)
            assert dtype == "uint16"
            return self

        def astype(self, dtype):
            raise AssertionError("no value conversion permitted")

    value = Array()

    def asarray(array):
        assert array is value
        return SimpleNamespace(shape=value.shape,
                               tobytes=lambda order: raw if order == "C" else None)

    witness = probe_module._hidden_digest(
        value, array_module=SimpleNamespace(bfloat16="bfloat16", uint16="uint16"),
        numpy_module=SimpleNamespace(asarray=asarray))
    assert views == ["uint16"]
    assert witness["sha256"] == hashlib.sha256(raw).hexdigest()
    assert witness["bytes"] == 8
    assert witness["shape"] == [1, 1, 4]


@pytest.mark.parametrize("value", [None, SimpleNamespace(dtype="float32"),
                                   SimpleNamespace(dtype="bfloat16", shape=(1, 5, 4)),
                                   SimpleNamespace(dtype="bfloat16", shape=(1, 1, 0))])
def test_missing_or_nonendpoint_hidden_fails_closed(value):
    with pytest.raises(ValueError):
        probe_module._hidden_digest(
            value, array_module=SimpleNamespace(bfloat16="bfloat16"),
            numpy_module=None)


def test_private_atomic_artifact_refuses_overwrite_and_cleans_temporary(tmp_path):
    artifact = tmp_path / "nested" / "probe.json"
    probe_module._atomic_write_private(artifact, {"available": True})
    assert json.loads(artifact.read_text()) == {"available": True}
    assert stat.S_IMODE(artifact.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        probe_module._atomic_write_private(artifact, {"available": False})
    assert json.loads(artifact.read_text()) == {"available": True}
    assert list(artifact.parent.iterdir()) == [artifact]


def test_failed_serialization_never_publishes_partial_artifact(tmp_path):
    artifact = tmp_path / "probe.json"
    with pytest.raises(ValueError):
        probe_module._atomic_write_private(artifact, {"invalid": float("nan")})
    assert not artifact.exists()
    assert not list(tmp_path.iterdir())


class Prompt(str):
    token_ids = tuple(range(1611))
    stable_boundary_tokens = 1606


def retained_setup():
    from tests.test_qwen4_retained_history_pure import setup, KV
    _, endpoint, fork = setup()
    endpoint.offset = 7
    slot = SimpleNamespace(kv=fork, tokens=(0, 1, 2, 3), qwen4_retention_tile=4,
                           approximate=False, logits=None, prompt_logits=None, exact_hidden=None)
    target = SimpleNamespace(_hot_prompt_slots=[slot], last_kv=endpoint, _hot_kv_persist=None)
    return target, slot, endpoint, fork, KV


def test_retained_witness_only_reads_existing_prefix_and_both_metadata_sets():
    target, slot, endpoint, fork, kind = retained_setup()
    old_histories, old_state = list(fork.kda_cache._conv), list(fork.kda_cache._state)
    calls = []
    result = probe_module._retained_prefix_witness(
        target, tuple(range(7)), expected_prefix=4, kv_type=kind,
        state_digest=lambda kv: calls.append(kv) or ("a" * 64, 12, 4096, {"kda": "b" * 64}),
        metadata_digest=lambda kv: calls.append(kv) or {"positions": kv.offset})
    assert calls == [fork, fork, endpoint]
    assert result["positions"] == 4 and result["read_only"]
    assert result["metadata"] == {"positions": 4}
    assert result["authoritative_endpoint_metadata"] == {"positions": 7}
    assert all(a is b for a, b in zip(fork.kda_cache._conv, old_histories))
    assert all(a is b for a, b in zip(fork.kda_cache._state, old_state))
    assert target.last_kv is endpoint and slot.kv is fork


@pytest.mark.parametrize("invalid", ["two_slots", "persist", "source_alias", "wrong_tokens",
                                    "wrong_offset", "tile", "approx", "logits", "owner_alias", "list_alias"])
def test_retained_witness_rejects_invalid_ownership_before_hashing(invalid):
    target, slot, endpoint, fork, kind = retained_setup()
    if invalid == "two_slots": target._hot_prompt_slots.append(slot)
    elif invalid == "persist": target._hot_kv_persist = object()
    elif invalid == "source_alias": slot.kv = endpoint
    elif invalid == "wrong_tokens": slot.tokens = (0, 1, 2, 9)
    elif invalid == "wrong_offset": fork.offset = 3
    elif invalid == "tile": slot.qwen4_retention_tile = 3
    elif invalid == "approx": slot.approximate = True
    elif invalid == "logits": slot.logits = object()
    elif invalid == "owner_alias": fork.kda_cache = endpoint.kda_cache
    else: fork.qwen4_cache.ple_conv = endpoint.qwen4_cache.ple_conv
    with pytest.raises(ValueError):
        probe_module._retained_prefix_witness(
            target, tuple(range(7)), expected_prefix=4, kv_type=kind,
            state_digest=lambda _: pytest.fail("hash before validation"),
            metadata_digest=lambda _: pytest.fail("hash before validation"))


class Wrapper:
    def __init__(self, target):
        self.target = target


def _setup(tmp_path, **overrides):
    calls = []
    target = SimpleNamespace(
        cfg=SimpleNamespace(model_type="qwen4_exp"), _model_dir=tmp_path,
        last_kv=SimpleNamespace(offset=1611, kda_cache=object(), qwen4_cache=object()),
        _h_last=object())
    result = {
        "tokens": [248069], "text": "PRIVATE_RESPONSE_NOT_FOR_ARTIFACT",
        "kv_positions": 1611, "kv_bytes": 160260624,
        "true_peak_metal_bytes": 4494260014, "termination_reason": "length",
        "path_stats": {"prompt_cache_source": "cold", "prefill_step_size": 1024},
    }

    def original(engine, *args, **kwargs):
        calls.append((engine, args, kwargs))
        return result

    ticks = iter(range(100))
    pressure_values = iter((
        {"available_bytes": 700, "swap_used_bytes": 10, "swap_out_bytes": 30},
        {"available_bytes": 650, "swap_used_bytes": 10, "swap_out_bytes": 31},
        {"available_bytes": 600, "swap_used_bytes": 10, "swap_out_bytes": 33},
    ))
    metal_values = iter((
        {"active_bytes": 100, "allocator_peak_bytes": 200},
        {"active_bytes": 150, "allocator_peak_bytes": 300},
        {"active_bytes": 155, "allocator_peak_bytes": 350},
    ))

    def state_digest(kv):
        assert kv is target.last_kv
        calls.append("state-digest")
        return "a" * 64, 121, 160260624, {key: "b" * 64 for key in ("kv", "kda", "qsa", "ple")}

    def hidden_digest(hidden):
        assert hidden is target._h_last
        calls.append("hidden-digest")
        return {"sha256": "c" * 64, "bytes": 20480, "shape": [1, 1, 4, 2560],
                "dtype": "bfloat16", "encoding": "bf16-bit-view-uint16-c-order-v1"}

    params = {
        "original": original, "artifact": tmp_path / "diagnostic.json",
        "expected_prompt_tokens": 1611, "label": "cold-control",
        "engine_type": Wrapper, "state_digest": state_digest,
        "hidden_digest": hidden_digest, "pressure": lambda: next(pressure_values),
        "metal_memory": lambda: next(metal_values),
        "profile_identity": lambda: {"vmodel_runtime_profiles": ["test-profile"],
                                     "vmodel_runtime_profile_digest": "d" * 64},
        "model_revision": lambda path: "e" * 40, "clock": lambda: next(ticks),
    }
    params.update(overrides)
    return (probe_module.FirstTokenHTTPProbe(**params), Wrapper(target), result, calls)


def test_observer_preserves_arguments_result_and_owners_and_separates_costs(tmp_path):
    probe, engine, result, calls = _setup(tmp_path)
    prompt = Prompt("PRIVATE_PROMPT_NOT_FOR_ARTIFACT")
    callback = object()
    endpoint, hidden = engine.target.last_kv, engine.target._h_last
    returned = probe(engine, prompt, 1, on_token=callback, sampling=callback)
    assert returned is result
    assert calls[0] == (engine, (prompt, 1), {"on_token": callback, "sampling": callback})
    assert calls[0][1][0] is prompt
    assert calls[1:] == ["state-digest", "hidden-digest"]
    assert engine.target.last_kv is endpoint and engine.target._h_last is hidden
    assert "state" not in vars(probe) and "result" not in vars(probe)
    document = json.loads(probe.artifact.read_text())
    assert document["available"] is True
    assert document["schema"] == probe_module.SCHEMA
    assert document["first_token_only"] is True
    assert not any(document[key] for key in ("timing_benchmark", "quality_benchmark", "full_harness_proof"))
    assert document["prepared_tokens"] == probe_module._token_digest(prompt.token_ids)
    assert document["generated_tokens"] == probe_module._token_digest(result["tokens"])
    assert document["stable_boundary_tokens"] == 1606
    assert document["state"]["positions"] == 1611
    assert document["generation"]["true_peak_metal_bytes"] == 4494260014
    assert document["pressure_after_generation"]["swap_out_bytes"] == 31
    assert document["pressure_after_instrumentation"]["swap_out_bytes"] == 33
    assert document["metal_after_generation"]["allocator_peak_bytes"] == 300
    assert document["metal_after_instrumentation"]["allocator_peak_bytes"] == 350
    assert document["instrumentation"]["endpoint_capture_seconds"] == 1
    assert document["instrumentation"]["state_host_read_bytes"] == 160260624
    serialized = probe.artifact.read_text()
    assert str(prompt) not in serialized and result["text"] not in serialized
    assert "248069" not in serialized


def test_read_only_prefix_observer_costs_are_explicit_and_result_is_unchanged(tmp_path):
    calls = []
    witness = lambda target, ids: calls.append((target, ids)) or {"bytes": 144003072, "read_only": True}
    probe, engine, result, _ = _setup(tmp_path, retained_witness=witness)
    prompt = Prompt("private")
    assert probe(engine, prompt, 1) is result
    document = json.loads(probe.artifact.read_text())
    assert calls == [(engine.target, prompt.token_ids)]
    assert document["retained_prefix"]["read_only"] is True
    assert document["instrumentation"]["retained_state_host_read_bytes"] == 144003072
    assert document["timing_benchmark"] is False
    assert "retained_fork_diagnostic" not in document


def test_read_only_and_mutating_observer_modes_cannot_mix(tmp_path):
    with pytest.raises(ValueError, match="exclusive"):
        _setup(tmp_path, retained_witness=lambda *a: {}, retained_diagnostic=lambda *a: {})


def test_retained_experiment_is_explicit_and_precedes_endpoint_host_hashes(tmp_path):
    probe, engine, result, calls = _setup(tmp_path)
    def retained(target, token_ids):
        assert target is engine.target and token_ids == Prompt.token_ids
        calls.append("retained-experiment")
        return {"available": True, "fork_equal": True, "serving_timing_proof": False}
    probe.retained_diagnostic = retained
    returned = probe(engine, Prompt("private"), 1)
    assert returned is result
    assert calls[1:] == ["retained-experiment", "state-digest", "hidden-digest"]
    document = json.loads(probe.artifact.read_text())
    assert document["retained_fork_diagnostic"]["fork_equal"] is True


def test_retained_experiment_failure_does_not_replace_generation_result(tmp_path):
    def retained(*args):
        raise ValueError("private metadata must not escape")
    probe, engine, result, calls = _setup(tmp_path, retained_diagnostic=retained)
    assert probe(engine, Prompt("private"), 1) is result
    document = json.loads(probe.artifact.read_text())
    assert document["available"] is False
    assert document["error_type"] == "ValueError"
    assert "private metadata" not in json.dumps(document)
    assert document["instrumentation"]["state_host_read_bytes"] is None


@pytest.mark.parametrize("bad", ["plain-engine", "wrong-model", "no-prepared-ids",
                                "wrong-input-count", "invalid-boundary", "max2", "float1", "bool1"])
def test_request_guards_fail_before_generation(tmp_path, bad):
    probe, engine, _result, calls = _setup(tmp_path)
    prompt, max_tokens = Prompt("private"), 1
    if bad == "plain-engine":
        engine = engine.target
    elif bad == "wrong-model":
        engine.target.cfg.model_type = "glm5_next"
    elif bad == "no-prepared-ids":
        prompt = "private"
    elif bad == "wrong-input-count":
        prompt.token_ids = (1, 2)
    elif bad == "invalid-boundary":
        prompt.stable_boundary_tokens = 1611
    else:
        max_tokens = {"max2": 2, "float1": 1.0, "bool1": True}[bad]
    with pytest.raises(ValueError):
        probe(engine, prompt, max_tokens)
    assert not calls
    document = json.loads(probe.artifact.read_text())
    assert document["available"] is False
    assert document["failure_phase"] == "request_validation"


@pytest.mark.parametrize("bad", ["missing", "kda", "qwen4", "offset", "result-offset", "two-output"])
def test_absent_or_misaligned_endpoint_never_passes(tmp_path, bad):
    probe, engine, result, calls = _setup(tmp_path)
    if bad == "missing":
        engine.target.last_kv = None
    elif bad == "kda":
        engine.target.last_kv.kda_cache = None
    elif bad == "qwen4":
        engine.target.last_kv.qwen4_cache = None
    elif bad == "offset":
        engine.target.last_kv.offset = 1606
    elif bad == "result-offset":
        result["kv_positions"] = 1606
    else:
        result["tokens"] = [1, 2]
    assert probe(engine, Prompt("private"), 1) is result
    assert len(calls) == 1
    document = json.loads(probe.artifact.read_text())
    assert document["available"] is False
    assert document["failure_phase"] == "endpoint_capture"


def test_empty_state_digest_is_unavailable(tmp_path):
    probe, engine, result, _calls = _setup(
        tmp_path, state_digest=lambda kv: ("a" * 64, 0, 0, {}))
    assert probe(engine, Prompt("private"), 1) is result
    assert json.loads(probe.artifact.read_text())["available"] is False


def test_generation_failure_is_preserved_and_published_without_error_text(tmp_path):
    original_error = RuntimeError("PRIVATE_FAILURE_DETAIL")

    def fail(*args, **kwargs):
        raise original_error

    probe, engine, _result, _calls = _setup(tmp_path, original=fail)
    with pytest.raises(RuntimeError) as caught:
        probe(engine, Prompt("private"), 1)
    assert caught.value is original_error
    raw = probe.artifact.read_text()
    assert "PRIVATE_FAILURE_DETAIL" not in raw
    assert json.loads(raw)["failure_phase"] == "generation"


def test_second_request_is_rejected_without_model_work_or_artifact_overwrite(tmp_path):
    probe, engine, _result, calls = _setup(tmp_path)
    probe(engine, Prompt("private"), 1)
    before = probe.artifact.read_bytes()
    with pytest.raises(RuntimeError, match="exactly one"):
        probe(engine, Prompt("private"), 1)
    assert len(calls) == 3
    assert probe.artifact.read_bytes() == before


def test_publication_failure_does_not_change_returned_generation(tmp_path, monkeypatch):
    def fail_publish(*args):
        raise OSError("disk full")

    probe, engine, result, _calls = _setup(tmp_path, publish=fail_publish)
    monkeypatch.setattr(probe_module, "print", lambda *args, **kwargs: (
        (_ for _ in ()).throw(BrokenPipeError())), raising=False)
    assert probe(engine, Prompt("private"), 1) is result
    assert not probe.artifact.exists()


@pytest.mark.parametrize("fail_server", [False, True])
def test_cli_runs_normal_server_main_and_restores_wrapper_and_argv(
        tmp_path, monkeypatch, fail_server):
    modules = {name: ModuleType(name) for name in (
        "mlx", "mlx.core", "numpy", "runtime", "runtime.server",
        "runtime.profiles", "runtime.qwen4_mtp", "tests", "tests.fixtures",
        "tests.fixtures.qwen4_flash_next_real_oracle")}
    for package in ("mlx", "runtime", "tests", "tests.fixtures"):
        modules[package].__path__ = []
    modules["mlx"].core = modules["mlx.core"]
    modules["runtime"].server = server = modules["runtime.server"]
    modules["runtime.profiles"].active_runtime_profile_fields = lambda: {}
    modules["runtime.qwen4_mtp"].Qwen4MTPSpeculativeEngine = Wrapper
    oracle = modules["tests.fixtures.qwen4_flash_next_real_oracle"]
    oracle._model_revision = lambda path: ""
    oracle._pressure = lambda: {}
    oracle._state_digest = lambda kv: None
    original = server._engine_generate = object()
    original_argv = sys.argv
    calls = []

    def server_main():
        assert isinstance(server._engine_generate, probe_module.FirstTokenHTTPProbe)
        assert server._engine_generate.original is original
        assert server._engine_generate.expected_prompt_tokens == 1611
        calls.append(list(sys.argv))
        if fail_server:
            raise RuntimeError("server failure")

    server.main = server_main
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    argv = ["--artifact", str(tmp_path / "probe.json"), "--label", "cold",
            "--expected-prompt-tokens", "1611", "--port", "8088",
            "--profile", "base", "--profile", "generation-witness",
            "--profile-dir", str(tmp_path / "profiles")]
    if fail_server:
        with pytest.raises(RuntimeError, match="server failure"):
            probe_module.main(argv)
    else:
        probe_module.main(argv)
    assert calls == [["runtime.server", "--port", "8088", "--profile", "base",
                      "--profile", "generation-witness", "--profile-dir",
                      str(tmp_path / "profiles")]]
    assert server._engine_generate is original
    assert sys.argv is original_argv
