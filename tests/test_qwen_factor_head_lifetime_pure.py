"""Head-release integration/order without importing MLX or allocating arrays."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest


ROOT = Path(__file__).resolve().parents[1]


def helper():
    tree = ast.parse((ROOT/'runtime/qwen35_mtp.py').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
        and n.name == '_suspend_qwen_head_for_factor_restore')
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(
        body=[copy.deepcopy(fn)], type_ignores=[])), '<real-head-release>', 'exec'), namespace)
    return namespace[fn.name], tree


def test_missing_adapter_is_noop():
    run, _ = helper()
    assert run(object()) == (0, 0)
    assert run(SimpleNamespace(_suspend_qwen35_serial_verify_lm_head=None)) == (0, 0)


@pytest.mark.parametrize('enabled', [False, True])
def test_existing_engine_guard_owns_release_and_byte_kinds_stay_separate(enabled):
    run, _ = helper()
    class Head:
        pass
    target = SimpleNamespace(_lm_head_w=Head(),
        _qwen35_serial_verify_head_suspend_active_released_bytes=500)
    ref = weakref.ref(target._lm_head_w)
    calls = []
    def suspend():
        calls.append('existing-engine-policy')
        if not enabled:
            return 0
        target._lm_head_w = None
        target._qwen35_serial_verify_head_suspend_active_released_bytes += 321
        return 456
    target._suspend_qwen35_serial_verify_lm_head = suspend
    assert run(target) == ((456, 321) if enabled else (0, 0))
    assert calls == ['existing-engine-policy']
    assert (ref() is None) is enabled


def test_release_failure_propagates_before_reconstruction():
    run, _ = helper()
    error = MemoryError('existing lease failure')
    def fail():
        raise error
    with pytest.raises(MemoryError) as raised:
        run(SimpleNamespace(_suspend_qwen35_serial_verify_lm_head=fail))
    assert raised.value is error


def test_real_call_is_after_emission_and_before_unchanged_reserve_and_commit():
    _, tree = helper()
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
        and n.name == 'QwenMTPSpeculativeEngine')
    generate = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'generate')
    text = ast.unparse(generate)
    release = text.index('_suspend_qwen_head_for_factor_restore(tgt)')
    assert text.index('emitted.append(tok)') < release
    assert text.index('elif target_fed_positions < round_verify_width:') < release
    assert release < text.index("reason='qwen-mtp-factor-restore'")
    assert release < text.index('round_factors.commit_prefix(')
    assert text.count('_suspend_qwen_head_for_factor_restore(tgt)') == 1
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
        and n.name == '_suspend_qwen_head_for_factor_restore')
    for forbidden in ('mx.', 'synchronize(', 'clear_cache(', 'eval(', 'reserve(', 'sleep('):
        assert forbidden not in ast.unparse(fn)
