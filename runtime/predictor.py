"""Phase 8: predictive expert prefetch.

MarkovExpertPredictor learns transition counts between consecutive layers' routed
expert sets: count[(layer, e, f)] += 1 whenever expert e is active at `layer` and
expert f is active at `layer+1` for the same token. Routing at layer i completes
before layer i+1's experts are needed, but only ~150 ms early on this disk — the
predictor widens that window to a full layer step by prefetching layer i+1's likely
experts as soon as layer i routes.

Counts are learned online and persisted per model (expert_transitions.json), so
held-out accuracy can improve across runs and prompts with similar workloads.
Issuing I/O from these predictions is separately gated by
``RuntimeConfig.expert_predictive_prefetch`` and remains off by default: measured
predictor recall does not prove spare storage bandwidth or a wall-clock win.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def parse_transition_tracking(raw: str | None) -> bool:
    """Strict server policy; omitted preserves historical online learning."""
    if raw is None or raw == '1':
        return True
    if raw == '0':
        return False
    raise ValueError('VMODEL_EXPERT_TRANSITION_TRACKING must be 0 or 1')


def validate_transition_tracking(tracking, *, predictive_prefetch, warm_start):
    if type(tracking) is not bool:
        raise ValueError('expert_transition_tracking must be boolean')
    if not tracking and (predictive_prefetch or warm_start != 0):
        raise ValueError(
            'disabled expert_transition_tracking conflicts with predictive prefetch or warm_start')


def make_expert_predictor(num_layers, num_experts, path, *, tracking=True,
                          predictive_prefetch=False, warm_start=0):
    """No-consumer opt-out avoids reading, learning, or writing saved history.

    Authoritative routing, usage/trace telemetry and deterministic expert-batch
    prefetch are separate. Existing history is neither deleted nor replaced.
    """
    validate_transition_tracking(tracking, predictive_prefetch=predictive_prefetch,
                                 warm_start=warm_start)
    if not tracking or not num_experts:
        return None
    return MarkovExpertPredictor(num_layers, num_experts, path=path)


class MarkovExpertPredictor:
    def __init__(self, num_layers: int, num_experts: int, path: str | Path | None = None):
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.path = Path(path) if path else None
        self.counts: dict[tuple[int, int, int], int] = defaultdict(int)
        self._prev: tuple[int, list[int]] | None = None
        self.observed = 0
        if self.path and self.path.exists():
            for key, c in json.loads(self.path.read_text()).items():
                l, e, f = (int(v) for v in key.split(","))
                self.counts[(l, e, f)] = c

    def observe(self, layer: int, experts: list[int]):
        if self._prev is not None and self._prev[0] == layer - 1:
            for e in self._prev[1]:
                for f in experts:
                    self.counts[(layer - 1, e, f)] += 1
            self.observed += 1
        self._prev = (layer, list(experts))

    def predict(self, layer: int, experts: list[int], top_m: int = 8) -> list[int]:
        """Rank layer+1's experts by transition mass from the current routed set."""
        if layer + 1 >= self.num_layers:
            return []
        scores: dict[int, int] = defaultdict(int)
        for e in experts:
            for f in range(self.num_experts):
                c = self.counts.get((layer, e, f), 0)
                if c:
                    scores[f] += c
        return [f for f, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:top_m]]

    def save(self):
        if not self.path or not self.counts:
            return
        self.path.write_text(json.dumps({f"{l},{e},{f}": c for (l, e, f), c in self.counts.items()}))

    def summary(self) -> str:
        return f"predictor: {len(self.counts)} transitions learned, {self.observed} observations this run"
