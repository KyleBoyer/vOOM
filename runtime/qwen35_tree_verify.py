"""Exact target tree verification for dense hybrid Qwen checkpoints.

Unlike a parallel WY/tree kernel, every node uses vOOM's established
one-position Qwen operators.  Nodes are merely scheduled layer-major so one
streamed weight page serves the whole tree.  Full-attention nodes see only the
prompt and their ancestors; DeltaNet nodes fork the exact parent state.
Compact per-node factors reconstruct only the accepted recurrent path.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import mlx.core as mx

from .kda_state import KDAFactorWindow, KDAStateCache
from .speculative_tree import SpeculativeTree, validate_tree


class QwenTreeKVProxy:
    """Read-only prompt K/V plus parent-indexed speculative node K/V."""

    def __init__(self, base, tree: SpeculativeTree, num_layers: int):
        self.base = base
        self.tree = tree
        self.num_layers = int(num_layers)
        self.current_node = 0
        # Preserve the established exact materialize+SDPA behavior unless the
        # caller already selected Qwen's explicit lossy paged-online path.
        # In that mode the proxy exposes the prompt pages followed by only the
        # current node's ancestor chain, avoiding one full-prefix
        # materialization for every tree node.
        self.online_attention = bool(
            getattr(base, "online_attention", False))
        self.online_attention_tile_positions = int(getattr(
            base, "online_attention_tile_positions", 2048))
        self.online_attention_page_native = bool(getattr(
            base, "online_attention_page_native", False))
        self.online_attention_pages_per_tile = int(getattr(
            base, "online_attention_pages_per_tile", 8))
        self.kda_cache = KDAStateCache(self.num_layers)
        size = len(tree.token_ids)
        self.node_keys: list[list[mx.array | None]] = [
            [None] * size for _ in range(self.num_layers)
        ]
        self.node_values: list[list[mx.array | None]] = [
            [None] * size for _ in range(self.num_layers)
        ]

    @property
    def offset(self) -> int:
        return int(self.base.offset)

    def _base_layer(self, layer: int):
        keys = getattr(self.base, "keys", None)
        values = getattr(self.base, "values", None)
        if keys is not None and values is not None:
            return keys[layer], values[layer]
        materialize = getattr(self.base, "materialize_layer", None)
        if callable(materialize):
            return materialize(layer)
        raise TypeError("Qwen tree verification requires readable layer K/V")

    def update(
        self, layer: int, key: mx.array, value: mx.array,
    ) -> tuple[mx.array, mx.array]:
        node = int(self.current_node)
        if not 0 <= node < len(self.tree.token_ids):
            raise ValueError("current Qwen tree node is invalid")
        if int(key.shape[2]) != 1 or key.shape != value.shape:
            raise ValueError("Qwen tree attention updates must be one position")
        self.node_keys[layer][node] = key
        self.node_values[layer][node] = value
        base_k, base_v = self._base_layer(layer)
        path = self.tree.path(node)
        path_k = [self.node_keys[layer][index] for index in path]
        path_v = [self.node_values[layer][index] for index in path]
        if any(item is None for item in (*path_k, *path_v)):
            raise RuntimeError("Qwen tree attention parent K/V is incomplete")
        keys = list(path_k)
        values = list(path_v)
        if base_k is not None:
            keys.insert(0, base_k)
            values.insert(0, base_v)
        return (
            keys[0] if len(keys) == 1 else mx.concatenate(keys, axis=2),
            values[0] if len(values) == 1 else mx.concatenate(values, axis=2),
        )

    def append_for_online_attention(
        self, layer: int, key: mx.array, value: mx.array,
    ) -> None:
        """Retain one speculative node without mutating the prompt cache."""
        node = int(self.current_node)
        if not self.online_attention:
            raise RuntimeError("tree proxy online attention is disabled")
        if not 0 <= node < len(self.tree.token_ids):
            raise ValueError("current Qwen tree node is invalid")
        if int(key.shape[2]) != 1 or key.shape != value.shape:
            raise ValueError("Qwen tree attention updates must be one position")
        self.node_keys[layer][node] = key
        self.node_values[layer][node] = value

    def iter_materialized_layer_chunks(
        self, layer: int, *, max_positions: int,
    ):
        """Yield bounded prompt tiles followed by the current ancestor path."""
        if not self.online_attention:
            raise RuntimeError("tree proxy online attention is disabled")
        iterator = getattr(self.base, "iter_materialized_layer_chunks", None)
        if not callable(iterator):
            raise TypeError(
                "Qwen tree online attention requires paged prompt K/V")
        yield from iterator(layer, max_positions=max_positions)

        path = self.tree.path(int(self.current_node))
        keys = [self.node_keys[layer][index] for index in path]
        values = [self.node_values[layer][index] for index in path]
        if any(item is None for item in (*keys, *values)):
            raise RuntimeError("Qwen tree attention parent K/V is incomplete")
        # Tree depths are tiny (currently <= 4), but retain the iterator's
        # contract if a larger experimental topology is supplied later.
        for start in range(0, len(path), int(max_positions)):
            part_k = keys[start:start + int(max_positions)]
            part_v = values[start:start + int(max_positions)]
            yield (
                part_k[0] if len(part_k) == 1
                else mx.concatenate(part_k, axis=2),
                part_v[0] if len(part_v) == 1
                else mx.concatenate(part_v, axis=2),
            )

    def iter_materialized_layer_pages(self, layer: int):
        """Yield exact prompt pages followed by one-token ancestor pages."""
        if not self.online_attention_page_native:
            raise RuntimeError("tree proxy page-native attention is disabled")
        iterator = getattr(self.base, "iter_materialized_layer_pages", None)
        if not callable(iterator):
            raise TypeError(
                "Qwen tree page-native attention requires paged prompt K/V")
        yield from iterator(layer)
        path = self.tree.path(int(self.current_node))
        keys = [self.node_keys[layer][index] for index in path]
        values = [self.node_values[layer][index] for index in path]
        if any(item is None for item in (*keys, *values)):
            raise RuntimeError("Qwen tree attention parent K/V is incomplete")
        yield from zip(keys, values)

    def layer_positions(self, layer: int) -> int:
        base_positions = getattr(self.base, "layer_positions", None)
        if not callable(base_positions):
            keys, _values = self._base_layer(layer)
            count = 0 if keys is None else int(keys.shape[2])
        else:
            count = int(base_positions(layer))
        return count + len(self.tree.path(int(self.current_node)))

    def commit_attention_path(self, path, destination) -> None:
        selected = tuple(int(index) for index in path)
        for layer in range(self.num_layers):
            keys = [self.node_keys[layer][index] for index in selected]
            if not keys or keys[0] is None:
                continue
            values = [self.node_values[layer][index] for index in selected]
            if any(item is None for item in (*keys, *values)):
                raise RuntimeError("Qwen tree commit has incomplete K/V")
            key = keys[0] if len(keys) == 1 else mx.concatenate(keys, axis=2)
            value = (
                values[0]
                if len(values) == 1 else mx.concatenate(values, axis=2)
            )
            destination.update(layer, key, value)


@dataclass
class QwenTreeVerification:
    tree: SpeculativeTree
    logits: mx.array
    hidden_nodes: tuple[mx.array, ...]
    tap_nodes: dict[int, tuple[mx.array, ...]]
    factors: KDAFactorWindow
    base_kda: KDAStateCache
    tree_kv: QwenTreeKVProxy
    base_layer_lengths: tuple[int, ...]

    def commit(self, path, *, target, kv) -> None:
        selected = tuple(int(index) for index in path)
        if not selected or selected[0] != 0:
            raise ValueError("Qwen tree commit path must begin at the root")
        for parent, child in zip(selected, selected[1:]):
            if self.tree.parents[child] != parent:
                raise ValueError("Qwen tree commit path is not contiguous")
        lengths = tuple(int(value) for value in kv.layer_lengths())
        if lengths != self.base_layer_lengths:
            raise RuntimeError("target K/V changed before Qwen tree commit")

        self.tree_kv.commit_attention_path(selected, kv)
        kv.kda_cache = self.factors.commit_indices(
            self.base_kda, selected)
        committed_hidden = tuple(self.hidden_nodes[index] for index in selected)
        target._h_window = mx.concatenate(committed_hidden, axis=1)
        target._h_last = committed_hidden[-1]
        target._tap_hidden = {
            layer: mx.concatenate(
                tuple(nodes[index] for index in selected), axis=1)
            for layer, nodes in self.tap_nodes.items()
        }
        mx.eval(target._h_window, target._h_last)


def verify_qwen35_tree(
    target,
    tree: SpeculativeTree,
    kv,
    *,
    tap_layers=(),
) -> QwenTreeVerification:
    """Evaluate every proposal node with released one-token target math."""
    validate_tree(tree)
    if target.cfg.model_type != "qwen3_5" or target.cfg.num_experts:
        raise ValueError("Qwen tree verifier currently supports dense qwen3_5")
    if not hasattr(kv, "kda_cache") or kv.kda_cache is None:
        raise ValueError("Qwen tree verifier requires recurrent target state")
    if not callable(getattr(kv, "layer_lengths", None)):
        raise ValueError("Qwen tree verifier requires per-layer K/V lengths")

    from .lm_head_stream import StreamedLMHead
    from .qwen35 import (
        _qwen35_attention_residual,
        _qwen35_mlp_residual,
        qwen35_rms_norm,
    )

    size = len(tree.token_ids)
    offset = int(kv.offset)
    base_lengths = tuple(int(value) for value in kv.layer_lengths())
    base_kda = kv.kda_cache.fork()
    proxy = QwenTreeKVProxy(kv, tree, target.cfg.num_hidden_layers)
    proxy.kda_cache.begin_factor_capture()
    embedded = target._embed(list(tree.token_ids))
    nodes = tuple(embedded[:, index:index + 1, :] for index in range(size))
    tapset = set(int(layer) for layer in tap_layers)
    tapped: dict[int, tuple[mx.array, ...]] = {}
    verifier_positions = size

    try:
        for layer in range(target.cfg.num_hidden_layers):
            target._select_serial_verify_layer_transient(
                verifier_positions, layer)
            target._prepare_serial_verify_layer_page(layer)
            wait_started = time.perf_counter()
            weights = target.cache.get(
                target._layer_key(layer), target._layer_names(layer))
            target.timer.add(
                "weights_wait", time.perf_counter() - wait_started)
            if target.governor is not None and target._layer_transient:
                target.governor.reserve(
                    target._layer_transient,
                    margin=target._layer_transient_margin,
                    reason="qwen-tree-verify-transient",
                )
            if target.prefetcher:
                for next_layer in range(
                    layer + 1,
                    min(
                        layer + 1 + target.rc.prefetch_depth,
                        target.cfg.num_hidden_layers,
                    ),
                ):
                    target.prefetcher.schedule(
                        target._layer_key(next_layer),
                        target._layer_names(next_layer),
                    )

            active_before = mx.get_active_memory()
            mx.reset_peak_memory()
            prefix = f"model.layers.{layer}"
            next_nodes = []
            layer_states: list[mx.array | None] = [None] * size
            layer_histories: list[tuple | None] = [None] * size
            for node, hidden in enumerate(nodes):
                proxy.current_node = node
                if target.cfg.layer_types[layer] == "linear_attention":
                    parent = tree.parents[node]
                    if parent < 0:
                        state = base_kda.state(layer)
                        history = base_kda.conv_history(layer)
                    else:
                        state = layer_states[parent]
                        history = layer_histories[parent]
                    # Always overwrite both scratch slots.  In particular, a
                    # root with an empty prompt state must not inherit the
                    # preceding sibling's endpoint from this layer-local
                    # workspace.
                    proxy.kda_cache.set_state(layer, state)
                    proxy.kda_cache.set_conv_history(layer, history)
                attended = _qwen35_attention_residual(
                    hidden,
                    weights,
                    prefix,
                    target.cfg,
                    proxy,
                    layer,
                    offset + tree.depths[node],
                )
                if target.cfg.layer_types[layer] == "linear_attention":
                    layer_states[node] = proxy.kda_cache.state(layer)
                    layer_histories[node] = proxy.kda_cache.conv_history(layer)
                output = _qwen35_mlp_residual(
                    attended,
                    weights,
                    prefix,
                    target.cfg,
                    layer,
                    target._get_experts,
                )
                next_nodes.append(output)
            mx.eval(*next_nodes)
            nodes = tuple(next_nodes)
            if layer in tapset:
                tapped[layer] = nodes
            target._record_serial_verify_layer_transient(
                verifier_positions,
                layer,
                max(
                    0,
                    int(mx.get_peak_memory())
                    - max(active_before, int(mx.get_active_memory())),
                ),
            )
            target._note_true_peak()
            del weights, layer_states, layer_histories

        factors = proxy.kda_cache.finish_factor_capture(size)
        if factors is None:
            raise RuntimeError("Qwen tree verifier omitted recurrent factors")
    except BaseException:
        proxy.kda_cache.cancel_factor_capture()
        raise

    head = target._lm_head_weight()
    if isinstance(head, StreamedLMHead):
        normalized = mx.concatenate(tuple(
            qwen35_rms_norm(
                hidden, target._norm_w, target.cfg.rms_norm_eps)
            for hidden in nodes
        ), axis=1)
        logits = head.logits_serial_rows(normalized)[0]
    else:
        rows = []
        for hidden in nodes:
            row = target._final_logits(hidden, head=head)
            mx.eval(row)
            rows.append(row)
        logits = mx.stack(rows)
    mx.eval(logits)
    return QwenTreeVerification(
        tree=tree,
        logits=logits,
        hidden_nodes=nodes,
        tap_nodes=tapped,
        factors=factors,
        base_kda=base_kda,
        tree_kv=proxy,
        base_layer_lengths=base_lengths,
    )
