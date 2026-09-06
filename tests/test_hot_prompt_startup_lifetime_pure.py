"""Execute the real hot-startup prefix in a live frame, without importing MLX.

Weak references test Python ownership, not physical GPU reclamation. Keeping the
observer inside that frame catches obsolete locals that survive until return.
"""

from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path
import time
from types import SimpleNamespace
import weakref

import pytest


ENGINE = Path(__file__).resolve().parents[1] / "runtime/engine.py"


@pytest.fixture
def startup():
    tree = ast.parse(ENGINE.read_text())
    owner = next(node for node in tree.body
                 if isinstance(node, ast.ClassDef) and node.name == "StreamingEngine")
    generate = next(node for node in owner.body
                    if isinstance(node, ast.FunctionDef) and node.name == "generate")
    blocks = [node for node in generate.body if isinstance(node, ast.If)
              and isinstance(node.test, ast.Name) and node.test.id == "hot_eligible"]
    assert len(blocks) == 1
    block = blocks[0]
    boundary = next(i for i, node in enumerate(block.body)
                    if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "best_idx"
                            for target in node.targets))
    wrapper = ast.parse("def startup(self, observe):\n    observe()\n").body[0]
    wrapper.body = deepcopy(block.body[:boundary]) + wrapper.body
    release = deepcopy(next(node for node in owner.body
                            if isinstance(node, ast.FunctionDef)
                            and node.name == "_release_kv"))
    release.decorator_list = []
    namespace = {"time": time}
    exec(compile(ast.fix_missing_locations(ast.Module(
        body=[release, wrapper], type_ignores=[])), str(ENGINE), "exec"), namespace)
    return namespace["startup"], namespace["_release_kv"]


class PlainState:
    pass


def make_owner(startup, endpoint, slots=()):
    return SimpleNamespace(last_kv=endpoint, _hot_prompt_slots=list(slots),
                           _h_window=endpoint, _h_last=endpoint,
                           _release_kv=startup[1])


def test_orphan_is_dead_before_slot_scan_in_same_generate_frame(startup):
    owner = make_owner(startup, PlainState())
    old = weakref.ref(owner.last_kv)

    def before_scan():
        assert owner.last_kv is owner._h_window is owner._h_last is None
        assert old() is None, "obsolete previous_last_kv still owns the old request"

    startup[0](owner, before_scan)


@pytest.mark.parametrize("slot_index", [0, 1])
def test_aliased_slot_keeps_endpoint_and_metadata_without_release(startup, slot_index):
    class RetainedState:
        def release(self):
            pytest.fail("a retained slot must not be released")

    endpoint = RetainedState()
    slots = [SimpleNamespace(kv=PlainState(), tokens=(7, 9), logits=object()),
             SimpleNamespace(kv=PlainState(), tokens=(4, 6), logits=object())]
    slots[slot_index].kv = endpoint
    owner = make_owner(startup, endpoint, slots)
    before = [dict(vars(slot)) for slot in slots]

    def before_scan():
        assert owner.last_kv is owner._h_window is owner._h_last is None
        assert owner._hot_prompt_slots[slot_index].kv is endpoint
        assert [vars(slot) for slot in owner._hot_prompt_slots] == before

    startup[0](owner, before_scan)


def test_external_owner_and_shared_prefix_payload_are_not_mutated(startup):
    payload = PlainState()
    retained = SimpleNamespace(kv=PlainState(), tokens=(1, 2))
    retained.kv.payload = payload
    external = PlainState()
    external.payload = payload
    owner = make_owner(startup, external, [retained])

    def before_scan():
        assert owner.last_kv is None
        assert external.payload is retained.kv.payload is payload
        assert owner._hot_prompt_slots == [retained]

    startup[0](owner, before_scan)


def test_unaliased_explicit_release_occurs_once_and_local_is_dropped(startup):
    events = []

    class Releasable:
        def release(self):
            events.append("release")

    owner = make_owner(startup, Releasable())
    old = weakref.ref(owner.last_kv)

    def before_scan():
        assert events == ["release"]
        assert old() is None

    startup[0](owner, before_scan)
    startup[0](owner, before_scan)


def test_release_failure_does_not_clear_engine_owner_or_continue(startup):
    class Releasable:
        def release(self):
            raise RuntimeError("cannot release")

    owner = make_owner(startup, Releasable())
    old = weakref.ref(owner.last_kv)
    with pytest.raises(RuntimeError, match="cannot release"):
        startup[0](owner, lambda: pytest.fail("must stop on release failure"))
    assert owner.last_kv is old()


def test_empty_first_request_remains_empty(startup):
    owner = make_owner(startup, None)
    startup[0](owner, lambda: None)
    assert owner.last_kv is owner._h_window is owner._h_last is None
    assert owner._hot_prompt_slots == []
