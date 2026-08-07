"""F226: DeepSeek V4's unimplemented Indexer must not be paged.

deepseek_v4_attention never calls index_scores or index_topk_idxs -- the
ratio-4 layers use the plain compressed gather -- so the Indexer's weights are
read and dequantized once per layer per token purely to be discarded. They are
7.8% of a trunk layer (13.11MB of 167.29MB measured on layer 20).

The elision is by non-implementation, not by proof, so the safety property is
that it fails LOUDLY if the Indexer is ever wired up: the names are absent
from the page rather than zeroed, so a consumer raises rather than silently
attending over a subset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_engine_never_calls_the_indexer_for_deepseek_v4():
    """The premise of the elision. If this fails, the elision is unsafe.

    Checked by parsing rather than substring search: the comment explaining
    the elision names both functions, so a text match reports itself.
    """
    import ast

    tree = ast.parse((ROOT / "runtime" / "engine.py").read_text())
    called, imported = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            target = node.func
            name = getattr(target, "id", None) or getattr(target, "attr", None)
            if name:
                called.add(name)
        elif isinstance(node, ast.ImportFrom):
            imported.update(alias.name for alias in node.names)
    for symbol in ("index_scores", "index_topk_idxs"):
        assert symbol not in called, f"engine calls {symbol}"
        assert symbol not in imported, f"engine imports {symbol}"


def test_indexer_primitives_still_exist_and_are_tested():
    """Elision is a paging decision, not a deletion: the math stays."""
    from runtime import deepseek_v4

    assert hasattr(deepseek_v4, "index_scores")
    assert hasattr(deepseek_v4, "index_topk_idxs")


def test_layer_names_drop_indexer_tensors_only_for_deepseek_v4():
    engine = (ROOT / "runtime" / "engine.py").read_text()
    assert '.attn.indexer.' in engine
    assert 'if self.cfg.model_type == "deepseek_v4":' in engine
    # GLM's own elision must remain independent of this one.
    assert '".self_attn.indexer." not in n' in engine


def test_elision_is_absence_not_substitution():
    """A zero-filled stand-in would attend over a subset silently."""
    engine = (ROOT / "runtime" / "engine.py").read_text()
    block = engine[engine.index('if self.cfg.model_type == "deepseek_v4":'):]
    block = block[:block.index("return names")]
    assert "zeros" not in block and "mx.zeros" not in block
    assert 'names = [n for n in names if ".attn.indexer." not in n]' in block
