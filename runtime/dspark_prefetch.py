"""DSpark-hidden-state expert prefetch hints for streamed Kimi K3.

The draft's final hidden rows have the target hidden width.  Applying K3's
*actual* per-layer router to those rows gives an inexpensive, correctness-free
hint: predicted pages may be read early, but the authoritative target router
still selects and weights experts during verification.  Bad predictions can
only waste bounded I/O/cache space.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx

from . import quant


@dataclass
class DSparkExpertPrefetchPlan:
    experts_by_layer: dict[int, tuple[int, ...]] = field(default_factory=dict)
    estimated_storage_bytes: int = 0
    router_storage_bytes: int = 0
    min_margin: float = 0.0
    predicted_experts: int = 0
    authoritative_experts: int = 0
    matched_experts: int = 0
    _scheduled: set[int] = field(default_factory=set, repr=False)
    _observed: set[int] = field(default_factory=set, repr=False)

    @property
    def recall(self) -> float:
        return (
            self.matched_experts / self.authoritative_experts
            if self.authoritative_experts else 0.0
        )

    @property
    def precision(self) -> float:
        return (
            self.matched_experts / self.predicted_experts
            if self.predicted_experts else 0.0
        )

    def observe_authoritative(
        self, layer: int, expert_ids: list[int]
    ) -> None:
        if layer in self._observed:
            return
        predicted = set(self.experts_by_layer.get(layer, ()))
        actual = {int(expert) for expert in expert_ids}
        self.predicted_experts += len(predicted)
        self.authoritative_experts += len(actual)
        self.matched_experts += len(predicted & actual)
        self._observed.add(layer)

    def schedule_before_layer(self, target, layer: int, depth: int) -> int:
        """Schedule at most ``depth`` upcoming layer hints once."""
        scheduler = getattr(
            target, "_dspark_expert_prefetcher", None)
        if scheduler is None or depth <= 0:
            return 0
        scheduled = 0
        stop = min(target.cfg.num_hidden_layers, int(layer) + int(depth) + 1)
        for predicted_layer in range(int(layer), stop):
            if predicted_layer in self._scheduled:
                continue
            experts = self.experts_by_layer.get(predicted_layer, ())
            for expert in experts:
                accepted = scheduler.schedule(
                    f"layer.{predicted_layer}.expert.{expert}",
                    target.store.names_with_prefix(
                        f"model.layers.{predicted_layer}."
                        f"{target.cfg.moe_expert_prefix}.{expert}."
                    ),
                    only_if_idle=False,
                    page_size_hint=int(
                        getattr(target, "_expert_fetch_page_bytes", 0)),
                )
                scheduled += int(accepted)
            self._scheduled.add(predicted_layer)
        return scheduled


class DSparkExpertPrefetcher:
    """Build a confidence- and physical-byte-bounded K3 expert plan."""

    def __init__(
        self,
        target,
        *,
        budget_bytes: int,
        experts_per_layer: int = 8,
        min_margin: float = 0.0,
    ):
        if target.cfg.model_type != "kimi_k3":
            raise ValueError("DSpark expert prefetch currently targets Kimi K3")
        if budget_bytes < 0:
            raise ValueError("expert prefetch budget must be non-negative")
        if experts_per_layer <= 0:
            raise ValueError("experts_per_layer must be positive")
        if min_margin < 0:
            raise ValueError("minimum router margin must be non-negative")
        self.target = target
        self.budget_bytes = int(budget_bytes)
        self.experts_per_layer = int(experts_per_layer)
        self.min_margin = float(min_margin)

    def _router_names(self, layer: int) -> tuple[str, str]:
        prefix = f"model.layers.{layer}.block_sparse_moe.gate"
        return f"{prefix}.weight", f"{prefix}.e_score_correction_bias"

    def build(self, draft_hidden: mx.array) -> DSparkExpertPrefetchPlan:
        target = self.target
        plan = DSparkExpertPrefetchPlan(min_margin=self.min_margin)
        if self.budget_bytes == 0 or draft_hidden.size == 0:
            return plan
        hidden = draft_hidden.reshape(-1, target.cfg.hidden_size)
        page_bytes = max(
            1, int(getattr(target, "_expert_storage_page_bytes", 1)))
        remaining = self.budget_bytes
        top_k = int(target.cfg.num_experts_per_tok)
        first = int(target.cfg.first_k_dense_replace)

        for layer in range(first, target.cfg.num_hidden_layers):
            if remaining < page_bytes:
                break
            weight_name, bias_name = self._router_names(layer)
            if not (target.store.has(weight_name)
                    and target.store.has(bias_name)):
                continue
            tensors, _elapsed, read_bytes = target.store.fetch(
                [weight_name, bias_name])
            plan.router_storage_bytes += int(read_bytes)
            weight = tensors[weight_name]
            bias = tensors[bias_name]
            from .bf16_nf12_linear import NF12Tensor

            if isinstance(weight, (quant.QTensor, NF12Tensor)):
                logits = quant.matmul(hidden.astype(mx.float32), weight)
            else:
                logits = (
                    hidden.astype(mx.float32)
                    @ weight.astype(mx.float32).T
                )
            scores = mx.sigmoid(logits)
            biased = scores + bias
            width = min(top_k + 1, int(biased.shape[-1]))
            indices = mx.argpartition(
                -biased, kth=width - 1, axis=-1
            )[:, :width]
            selected = mx.take_along_axis(biased, indices, axis=-1)
            order = mx.argsort(-selected, axis=-1)
            indices = mx.take_along_axis(indices, order, axis=-1)
            selected = mx.take_along_axis(selected, order, axis=-1)
            mx.eval(indices, selected)

            votes: dict[int, tuple[int, float]] = {}
            for row_ids, row_scores in zip(
                indices.tolist(), selected.tolist(), strict=True
            ):
                margin = (
                    float(row_scores[top_k - 1] - row_scores[top_k])
                    if len(row_scores) > top_k else float("inf")
                )
                if margin < self.min_margin:
                    continue
                for rank, expert in enumerate(row_ids[:top_k]):
                    count, score = votes.get(int(expert), (0, 0.0))
                    votes[int(expert)] = (
                        count + 1,
                        score + float(row_scores[rank]),
                    )
            ranked = sorted(
                votes,
                key=lambda expert: (
                    -votes[expert][0], -votes[expert][1], expert),
            )
            count = min(
                self.experts_per_layer,
                len(ranked),
                remaining // page_bytes,
            )
            if count:
                plan.experts_by_layer[layer] = tuple(ranked[:count])
                consumed = count * page_bytes
                plan.estimated_storage_bytes += consumed
                remaining -= consumed
            del tensors, weight, bias, logits, scores, biased
        return plan
