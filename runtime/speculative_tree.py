"""Small target-verification tree primitives.

The best-first construction and greedy walk are adapted from DDTree-MLX
(humanrouter/ddtree-mlx, commit 4b12590abc9909fb03bfdf7dd736e76cef7ebdb0,
MIT).  This module deliberately contains no model math: a proposal tree is
only useful after the released target has evaluated every node against its
actual ancestors.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpeculativeTree:
    """Topologically ordered proposal nodes (root is implicit index zero)."""

    token_ids: tuple[int, ...]
    depths: tuple[int, ...]
    parents: tuple[int, ...]
    children: tuple[dict[int, int], ...]

    @property
    def node_count(self) -> int:
        return len(self.token_ids) - 1

    def path(self, node: int) -> tuple[int, ...]:
        node = int(node)
        if not 0 <= node < len(self.token_ids):
            raise ValueError("tree node is outside the proposal")
        result = []
        while node >= 0:
            result.append(node)
            node = self.parents[node]
        return tuple(reversed(result))


@dataclass(frozen=True)
class TreeDraft:
    """Explicit marker returned by an opt-in proposal-tree source."""

    tree: SpeculativeTree


def validate_tree(tree: SpeculativeTree) -> None:
    size = len(tree.token_ids)
    if not size or len(tree.depths) != size or len(tree.parents) != size:
        raise ValueError("speculative tree arrays have inconsistent lengths")
    if len(tree.children) != size:
        raise ValueError("speculative tree child maps have inconsistent length")
    if tree.parents[0] != -1 or tree.depths[0] != 0:
        raise ValueError("speculative tree root must have parent -1/depth 0")
    rebuilt = [dict() for _ in range(size)]
    for node in range(1, size):
        parent = int(tree.parents[node])
        if not 0 <= parent < node:
            raise ValueError("speculative tree parents must precede children")
        if tree.depths[node] != tree.depths[parent] + 1:
            raise ValueError("speculative tree depth does not follow its parent")
        token = int(tree.token_ids[node])
        if token in rebuilt[parent]:
            raise ValueError("one tree parent has duplicate token children")
        rebuilt[parent][token] = node
    if tuple(rebuilt) != tree.children:
        raise ValueError("speculative tree child maps disagree with parents")


def build_tree_from_topk(
    *,
    root_token: int,
    top_token_ids: np.ndarray,
    top_log_probabilities: np.ndarray,
    budget: int,
) -> SpeculativeTree:
    """Best-first factorized tree under a fixed non-root node budget.

    Each draft depth supplies marginal top-k probabilities.  A heap expands
    the highest-probability prefix, its next-ranked sibling, and its rank-zero
    child.  This is proposal scheduling only; it does not make marginals
    conditional and never substitutes for target verification.
    """
    ids = np.asarray(top_token_ids, dtype=np.int64)
    logp = np.asarray(top_log_probabilities, dtype=np.float32)
    budget = int(budget)
    if budget < 0:
        raise ValueError("speculative tree budget must be non-negative")
    if ids.ndim != 2 or logp.ndim != 2 or ids.shape != logp.shape:
        raise ValueError("tree top-k IDs/probabilities must be paired matrices")
    if ids.size and (not np.all(np.isfinite(logp)) or np.any(ids < 0)):
        raise ValueError("tree top-k proposal data is invalid")

    token_ids = [int(root_token)]
    depths = [0]
    parents = [-1]
    children: list[dict[int, int]] = [{}]
    if budget == 0 or ids.shape[0] == 0 or ids.shape[1] == 0:
        return SpeculativeTree(
            tuple(token_ids), tuple(depths), tuple(parents), tuple(children))

    width = min(budget, int(ids.shape[1]))
    depth_limit = int(ids.shape[0])
    first = float(logp[0, 0])
    # (-path log probability, stable rank path, parent, depth, rank, logp)
    heap = [(-first, (0,), 0, 1, 0, first)]
    while heap and len(token_ids) - 1 < budget:
        _negative, ranks, parent, depth, rank, path_logp = heapq.heappop(heap)
        token = int(ids[depth - 1, rank])
        if token in children[parent]:
            # Duplicate top-k IDs are malformed as distributions, but skip
            # rather than create an ambiguous target walk.
            continue
        node = len(token_ids)
        token_ids.append(token)
        depths.append(depth)
        parents.append(parent)
        children.append({})
        children[parent][token] = node

        if rank + 1 < width:
            sibling_logp = (
                path_logp
                - float(logp[depth - 1, rank])
                + float(logp[depth - 1, rank + 1])
            )
            heapq.heappush(heap, (
                -sibling_logp,
                ranks[:-1] + (rank + 1,),
                parent,
                depth,
                rank + 1,
                sibling_logp,
            ))
        if depth < depth_limit:
            child_logp = path_logp + float(logp[depth, 0])
            heapq.heappush(heap, (
                -child_logp,
                ranks + (0,),
                node,
                depth + 1,
                0,
                child_logp,
            ))

    result = SpeculativeTree(
        tuple(token_ids), tuple(depths), tuple(parents), tuple(children))
    validate_tree(result)
    return result


def greedy_tree_walk(
    tree: SpeculativeTree,
    target_tokens: list[int] | tuple[int, ...],
) -> tuple[tuple[int, ...], int]:
    """Follow target argmax tokens until no verified child matches."""
    validate_tree(tree)
    if len(target_tokens) != len(tree.token_ids):
        raise ValueError("target tree token rows do not match proposal nodes")
    accepted = [0]
    current = 0
    bonus = int(target_tokens[current])
    while bonus in tree.children[current]:
        current = tree.children[current][bonus]
        accepted.append(current)
        bonus = int(target_tokens[current])
    return tuple(accepted), bonus
