import numpy as np
import pytest

from runtime.speculative_tree import (
    SpeculativeTree,
    build_tree_from_topk,
    greedy_tree_walk,
    validate_tree,
)


def test_best_first_tree_build_and_walk():
    tree = build_tree_from_topk(
        root_token=9,
        top_token_ids=np.array([[10, 11], [20, 21], [30, 31]]),
        top_log_probabilities=np.log(np.array([
            [0.8, 0.2], [0.7, 0.3], [0.6, 0.4],
        ], dtype=np.float32)),
        budget=5,
    )

    assert tree.token_ids == (9, 10, 20, 30, 21, 31)
    assert tree.parents == (-1, 0, 1, 2, 1, 2)
    assert tree.depths == (0, 1, 2, 3, 2, 3)
    assert tree.path(5) == (0, 1, 2, 5)

    accepted, bonus = greedy_tree_walk(
        tree, [10, 20, 31, 99, 7, 42])
    assert accepted == (0, 1, 2, 5)
    assert bonus == 42


def test_tree_builder_is_bounded_and_empty_safe():
    empty = build_tree_from_topk(
        root_token=3,
        top_token_ids=np.empty((0, 0), dtype=np.int64),
        top_log_probabilities=np.empty((0, 0), dtype=np.float32),
        budget=0,
    )
    assert empty.node_count == 0
    assert greedy_tree_walk(empty, [17]) == ((0,), 17)

    bounded = build_tree_from_topk(
        root_token=3,
        top_token_ids=np.array([[4, 5, 6], [7, 8, 9]]),
        top_log_probabilities=np.log(np.array([
            [0.5, 0.3, 0.2], [0.7, 0.2, 0.1],
        ], dtype=np.float32)),
        budget=2,
    )
    assert bounded.node_count == 2


def test_tree_validation_rejects_non_topological_or_ambiguous_nodes():
    with pytest.raises(ValueError, match="precede"):
        validate_tree(SpeculativeTree(
            token_ids=(1, 2), depths=(0, 1), parents=(-1, 1),
            children=({}, {}),
        ))
    with pytest.raises(ValueError, match="disagree"):
        validate_tree(SpeculativeTree(
            token_ids=(1, 2), depths=(0, 1), parents=(-1, 0),
            children=({}, {}),
        ))
