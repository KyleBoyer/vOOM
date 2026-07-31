"""Correctness-free, byte-bounded DSpark expert prefetch planning."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx

from runtime.dspark_prefetch import DSparkExpertPrefetcher


class _Store:
    def __init__(self):
        self.read_bytes = 0

    def has(self, _name):
        return True

    def fetch(self, names):
        layer = int(names[0].split(".")[2])
        # Alternate the strongest two experts across layers.
        weight = mx.zeros((4, 4), dtype=mx.float32)
        weight = weight.at[layer % 4, 0].add(5.0)
        weight = weight.at[(layer + 1) % 4, 1].add(4.0)
        tensors = {
            names[0]: weight,
            names[1]: mx.zeros((4,), dtype=mx.float32),
        }
        read = sum(value.nbytes for value in tensors.values())
        self.read_bytes += read
        return tensors, 0.0, read

    def names_with_prefix(self, prefix):
        return [prefix + "w1.weight", prefix + "w2.weight"]


class _Scheduler:
    def __init__(self):
        self.calls = []

    def schedule(
        self, key, names, only_if_idle=False, page_size_hint=None
    ):
        self.calls.append(
            (key, tuple(names), only_if_idle, page_size_hint))
        return True


def test_plan_is_router_informed_confidence_and_byte_bounded():
    target = SimpleNamespace(
        cfg=SimpleNamespace(
            model_type="kimi_k3",
            hidden_size=4,
            num_hidden_layers=4,
            first_k_dense_replace=1,
            num_experts_per_tok=2,
            moe_expert_prefix="block_sparse_moe.experts",
        ),
        store=_Store(),
        _expert_storage_page_bytes=100,
        _expert_fetch_page_bytes=100,
        prefetcher=_Scheduler(),
        _dspark_expert_prefetcher=_Scheduler(),
    )
    predictor = DSparkExpertPrefetcher(
        target,
        budget_bytes=300,
        experts_per_layer=2,
        min_margin=0.0,
    )
    hidden = mx.array([[2.0, 1.0, 0.0, 0.0]])
    plan = predictor.build(hidden)

    assert plan.estimated_storage_bytes <= 300
    assert sum(map(len, plan.experts_by_layer.values())) == 3
    assert plan.router_storage_bytes == target.store.read_bytes

    scheduled = plan.schedule_before_layer(target, layer=0, depth=3)
    assert scheduled == 3
    assert all(not call[2] for call in target._dspark_expert_prefetcher.calls)
    # A second visit never duplicates the same layer's queue entries.
    assert plan.schedule_before_layer(target, layer=0, depth=3) == 0
