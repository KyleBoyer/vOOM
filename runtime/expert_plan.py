"""Pure-Python expert trace analysis and physical-layout planning.

This module deliberately does not read weights or import MLX.  It answers the
questions that must be settled before rewriting a large checkpoint or enabling
speculative I/O:

* do adjacent decode sweeps reuse the same experts?
* does a predictor generalize to held-out sweeps?
* would a physical order coalesce demanded expert pages?
* how many unused bytes would fixed bundles or speculative reads amplify?

The output is a *logical* plan.  Applying an order to a vpack generation is a
separate, transactional operation with its own integrity and token-identity
gates.  Keeping planning separate makes negative experiments cheap and safe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, OrderedDict, defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Mapping, Sequence


TraceEvent = tuple[int, tuple[int, ...]]
Sweep = dict[int, tuple[int, ...]]


def _canonical_experts(experts: Iterable[int]) -> tuple[int, ...]:
    """Return unique expert ids in released ascending order."""
    return tuple(sorted({int(expert) for expert in experts}))


def split_sweeps(trace: Iterable[tuple[int, Iterable[int]]]) -> list[Sweep]:
    """Split a sequential engine trace whenever its layer index resets.

    Routed layers are strictly increasing within one engine sweep.  ``<=`` is
    used rather than ``<`` so a repeated layer fails safely into a new sweep
    instead of silently replacing an earlier routing event.
    """
    sweeps: list[Sweep] = []
    current: Sweep = {}
    previous_layer: int | None = None
    for raw_layer, raw_experts in trace:
        layer = int(raw_layer)
        if layer < 0:
            raise ValueError("trace layer ids must be non-negative")
        experts = _canonical_experts(raw_experts)
        if any(expert < 0 for expert in experts):
            raise ValueError("trace expert ids must be non-negative")
        if current and previous_layer is not None and layer <= previous_layer:
            sweeps.append(current)
            current = {}
        current[layer] = experts
        previous_layer = layer
    if current:
        sweeps.append(current)
    return sweeps


def flatten_sweeps(sweeps: Sequence[Sweep]) -> list[TraceEvent]:
    return [
        (layer, _canonical_experts(experts))
        for sweep in sweeps
        for layer, experts in sorted(sweep.items())
    ]


def trace_document(
    trace: Iterable[tuple[int, Iterable[int]]],
    *,
    model: str = "",
    num_experts: int = 0,
    expert_page_bytes: int = 0,
) -> dict:
    """Build the stable JSON trace schema used by the offline planner."""
    sweeps = split_sweeps(trace)
    if num_experts < 0 or expert_page_bytes < 0:
        raise ValueError("trace metadata sizes must be non-negative")
    if num_experts and any(
            expert >= num_experts for sweep in sweeps
            for experts in sweep.values() for expert in experts):
        raise ValueError("trace expert id exceeds declared num_experts")
    return {
        "schema": "vmodel.expert-trace.v1",
        "model": str(model),
        "num_experts": int(num_experts),
        "expert_page_bytes": int(expert_page_bytes),
        "sweeps": [
            {
                "index": index,
                "routes": [
                    {"layer": layer, "experts": list(experts)}
                    for layer, experts in sorted(sweep.items())
                ],
            }
            for index, sweep in enumerate(sweeps)
        ],
    }


def write_trace(path: str | Path, trace: Iterable[tuple[int, Iterable[int]]],
                **metadata) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = trace_document(trace, **metadata)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return path


def load_trace(path: str | Path) -> tuple[dict, list[Sweep]]:
    path = Path(path)
    document = json.loads(path.read_text())
    if document.get("schema") != "vmodel.expert-trace.v1":
        raise ValueError(f"unsupported expert trace schema in {path}")
    declared_experts = int(document.get("num_experts", 0))
    page_bytes = int(document.get("expert_page_bytes", 0))
    if declared_experts < 0 or page_bytes < 0:
        raise ValueError("expert trace metadata sizes must be non-negative")
    sweeps: list[Sweep] = []
    for expected_index, raw_sweep in enumerate(document.get("sweeps", [])):
        if int(raw_sweep.get("index", -1)) != expected_index:
            raise ValueError("expert trace sweep indices must be contiguous")
        sweep: Sweep = {}
        previous_layer: int | None = None
        for route in raw_sweep.get("routes", []):
            layer = int(route["layer"])
            if previous_layer is not None and layer <= previous_layer:
                raise ValueError("expert trace layers must increase within a sweep")
            experts = _canonical_experts(route["experts"])
            if layer < 0 or any(expert < 0 for expert in experts):
                raise ValueError("expert trace ids must be non-negative")
            if declared_experts and any(
                    expert >= declared_experts for expert in experts):
                raise ValueError("trace expert id exceeds declared num_experts")
            sweep[layer] = experts
            previous_layer = layer
        if sweep:
            sweeps.append(sweep)
    if not sweeps:
        raise ValueError("expert trace contains no routed sweeps")
    return document, sweeps


def _infer_num_experts(sweeps: Sequence[Sweep]) -> int:
    ids = [expert for sweep in sweeps for experts in sweep.values()
           for expert in experts]
    return max(ids, default=-1) + 1


def expert_heat(sweeps: Sequence[Sweep]) -> dict[int, Counter[int]]:
    heat: dict[int, Counter[int]] = defaultdict(Counter)
    for sweep in sweeps:
        for layer, experts in sweep.items():
            heat[layer].update(_canonical_experts(experts))
    return dict(heat)


def coactivation_weights(
    sweeps: Sequence[Sweep],
) -> dict[int, Counter[tuple[int, int]]]:
    weights: dict[int, Counter[tuple[int, int]]] = defaultdict(Counter)
    for sweep in sweeps:
        for layer, experts in sweep.items():
            for left, right in combinations(_canonical_experts(experts), 2):
                weights[layer][(left, right)] += 1
    return dict(weights)


def build_physical_orders(
    sweeps: Sequence[Sweep], num_experts: int,
) -> dict[str, dict[int, tuple[int, ...]]]:
    """Build identity, heat, and greedy coactivation orders per layer.

    Coactivation ordering is intentionally a small deterministic heuristic, not
    a claim of optimal graph layout.  The simulator decides whether it is worth
    replacing with a stronger solver.
    """
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    layers = sorted({layer for sweep in sweeps for layer in sweep})
    heat = expert_heat(sweeps)
    edges = coactivation_weights(sweeps)
    identity = {layer: tuple(range(num_experts)) for layer in layers}
    heat_order = {
        layer: tuple(sorted(range(num_experts),
                            key=lambda expert: (-heat[layer][expert], expert)))
        for layer in layers
    }
    coactivation: dict[int, tuple[int, ...]] = {}
    for layer in layers:
        layer_heat = heat[layer]
        layer_edges = edges.get(layer, Counter())
        remaining = set(range(num_experts))
        first = min(remaining, key=lambda expert: (-layer_heat[expert], expert))
        order = [first]
        remaining.remove(first)
        while remaining:
            previous = order[-1]

            def rank(expert: int) -> tuple[int, int, int]:
                edge = ((previous, expert) if previous < expert
                        else (expert, previous))
                return (-layer_edges[edge], -layer_heat[expert], expert)

            nxt = min(remaining, key=rank)
            order.append(nxt)
            remaining.remove(nxt)
        coactivation[layer] = tuple(order)
    return {
        "identity": identity,
        "heat": heat_order,
        "coactivation": coactivation,
    }


@dataclass(frozen=True)
class LayoutResult:
    events: int
    accesses: int
    cache_hits: int
    cache_misses: int
    requests: int
    demanded_bytes: int
    physical_bytes: int
    read_amplification: float
    predicted_wall_s: float


def simulate_layout(
    sweeps: Sequence[Sweep],
    orders: Mapping[int, Sequence[int]],
    *,
    expert_page_bytes: int,
    bandwidth_mbps: float,
    request_overhead_ms: float = 0.0,
    coalesce_gap_pages: int = -1,
    bundle_pages: int = 0,
    cache_pages: int = 0,
) -> LayoutResult:
    """Charge physical bytes/requests for one immutable expert layout.

    ``coalesce_gap_pages=-1`` issues one request per demanded expert.  Zero
    merges only physically adjacent demanded pages.  Positive values permit
    reading unused gap pages and expose that amplification explicitly.

    ``bundle_pages`` models inseparable fixed-size bundles.  It is mutually
    exclusive with coalescing because a real plan should not get both benefits
    without paying both representations' storage cost. Unselected pages read
    through gaps/bundles are not admitted to the simulated cache; this is a
    conservative estimate of their future value, while still charging the
    current physical amplification.

    ``cache_pages`` replays the runtime's cumulative-frequency/recency eviction
    shape at expert-page granularity. Pinned trunk/KV capacity must already be
    subtracted by the caller. Zero disables the cache (all accesses miss).
    """
    if expert_page_bytes <= 0 or bandwidth_mbps <= 0:
        raise ValueError("expert_page_bytes and bandwidth_mbps must be positive")
    if (request_overhead_ms < 0 or coalesce_gap_pages < -1
            or bundle_pages < 0 or cache_pages < 0):
        raise ValueError("layout costs must be non-negative")
    if bundle_pages and coalesce_gap_pages >= 0:
        raise ValueError("fixed bundles and dynamic coalescing are exclusive")

    positions: dict[int, dict[int, int]] = {}
    for layer, order in orders.items():
        normalized = tuple(int(expert) for expert in order)
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"physical order for layer {layer} has duplicates")
        positions[int(layer)] = {
            expert: index for index, expert in enumerate(normalized)
        }

    demanded_pages = physical_pages = requests = events = 0
    accesses = cache_hits = cache_misses = 0
    cache: OrderedDict[tuple[int, int], None] = OrderedDict()
    frequencies: Counter[tuple[int, int]] = Counter()
    for sweep in sweeps:
        for layer, raw_experts in sweep.items():
            experts = _canonical_experts(raw_experts)
            if not experts:
                continue
            if (layer not in positions
                    or any(e not in positions[layer] for e in experts)):
                raise ValueError(f"physical order for layer {layer} is incomplete")
            events += 1
            accesses += len(experts)
            missing = []
            for expert in experts:
                key = (layer, expert)
                frequencies[key] += 1
                if cache_pages and key in cache:
                    cache_hits += 1
                    cache.move_to_end(key)
                else:
                    cache_misses += 1
                    missing.append(expert)
            if cache_pages:
                for expert in missing:
                    key = (layer, expert)
                    cache[key] = None
                    while len(cache) > cache_pages:
                        victim = min(
                            (frequencies[candidate], age, candidate)
                            for age, candidate in enumerate(cache)
                        )[2]
                        del cache[victim]
            if not missing:
                continue
            demanded_pages += len(missing)
            selected = sorted(positions[layer][expert] for expert in missing)
            if bundle_pages:
                total_pages = len(positions[layer])
                bundles = {position // bundle_pages for position in selected}
                requests += len(bundles)
                for bundle in bundles:
                    start = bundle * bundle_pages
                    physical_pages += min(bundle_pages, total_pages - start)
            elif coalesce_gap_pages < 0:
                requests += len(selected)
                physical_pages += len(selected)
            else:
                run_start = run_end = selected[0]
                for position in selected[1:]:
                    if position - run_end - 1 <= coalesce_gap_pages:
                        run_end = position
                    else:
                        requests += 1
                        physical_pages += run_end - run_start + 1
                        run_start = run_end = position
                requests += 1
                physical_pages += run_end - run_start + 1

    demanded_bytes = demanded_pages * expert_page_bytes
    physical_bytes = physical_pages * expert_page_bytes
    wall = physical_bytes / (bandwidth_mbps * 1_000_000.0)
    wall += requests * request_overhead_ms / 1000.0
    amplification = physical_bytes / demanded_bytes if demanded_bytes else 1.0
    return LayoutResult(
        events=events,
        accesses=accesses,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        requests=requests,
        demanded_bytes=demanded_bytes,
        physical_bytes=physical_bytes,
        read_amplification=amplification,
        predicted_wall_s=wall,
    )


@dataclass(frozen=True)
class PredictionResult:
    transitions: int
    predictions: int
    actual: int
    useful: int
    wasted: int
    precision: float
    recall: float
    demand_stall_bytes_saved: int
    extra_device_bytes: int


class TransitionPredictor:
    """Count-based next-layer predictor with deterministic score ordering."""

    def __init__(self):
        self.counts: Counter[tuple[int, int, int]] = Counter()

    def observe_sweep(self, sweep: Sweep) -> None:
        layers = sorted(sweep)
        for layer, nxt in zip(layers, layers[1:]):
            if nxt != layer + 1:
                continue
            for source in _canonical_experts(sweep[layer]):
                for target in _canonical_experts(sweep[nxt]):
                    self.counts[(layer, source, target)] += 1

    def ranked(self, layer: int, experts: Iterable[int]) -> list[tuple[int, int]]:
        scores: Counter[int] = Counter()
        sources = _canonical_experts(experts)
        for (seen_layer, source, target), count in self.counts.items():
            if seen_layer == layer and source in sources:
                scores[target] += count
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def evaluate_transition_predictor(
    train: Sequence[Sweep],
    held_out: Sequence[Sweep],
    *,
    top_m: int,
    expert_page_bytes: int,
    opportunistic_pages: int | None = None,
) -> PredictionResult:
    """Evaluate without training on held-out routes.

    ``opportunistic_pages`` caps predictions per layer transition to the number
    of page reads that measured idle I/O capacity can support.  ``None`` uses
    ``top_m``.  Saved stall bytes count correct early pages; extra device bytes
    count wrong pages, because actual demand bytes are otherwise unchanged.
    """
    if top_m <= 0 or expert_page_bytes <= 0:
        raise ValueError("top_m and expert_page_bytes must be positive")
    cap = top_m if opportunistic_pages is None else min(top_m, opportunistic_pages)
    if cap < 0:
        raise ValueError("opportunistic_pages must be non-negative")
    predictor = TransitionPredictor()
    for sweep in train:
        predictor.observe_sweep(sweep)

    transitions = predictions = actual_count = useful = 0
    for sweep in held_out:
        layers = sorted(sweep)
        for layer, nxt in zip(layers, layers[1:]):
            if nxt != layer + 1:
                continue
            ranked = predictor.ranked(layer, sweep[layer])
            predicted = {expert for expert, _ in ranked[:cap]}
            actual = set(_canonical_experts(sweep[nxt]))
            transitions += 1
            predictions += len(predicted)
            actual_count += len(actual)
            useful += len(predicted & actual)
    wasted = predictions - useful
    return PredictionResult(
        transitions=transitions,
        predictions=predictions,
        actual=actual_count,
        useful=useful,
        wasted=wasted,
        precision=useful / predictions if predictions else 0.0,
        recall=useful / actual_count if actual_count else 0.0,
        demand_stall_bytes_saved=useful * expert_page_bytes,
        extra_device_bytes=wasted * expert_page_bytes,
    )


@dataclass(frozen=True)
class PersistenceResult:
    comparisons: int
    predicted: int
    useful: int
    precision: float
    recall: float


def evaluate_adjacent_sweep_persistence(sweeps: Sequence[Sweep]) -> PersistenceResult:
    """Use the previous sweep's same-layer route as the next prediction."""
    comparisons = predicted = useful = actual = 0
    for previous, current in zip(sweeps, sweeps[1:]):
        for layer in sorted(set(previous) & set(current)):
            left = set(_canonical_experts(previous[layer]))
            right = set(_canonical_experts(current[layer]))
            comparisons += 1
            predicted += len(left)
            actual += len(right)
            useful += len(left & right)
    return PersistenceResult(
        comparisons=comparisons,
        predicted=predicted,
        useful=useful,
        precision=useful / predicted if predicted else 0.0,
        recall=useful / actual if actual else 0.0,
    )


def compile_plan(
    sweeps: Sequence[Sweep],
    *,
    num_experts: int = 0,
    source_document: Mapping | None = None,
) -> dict:
    num_experts = int(num_experts) or _infer_num_experts(sweeps)
    orders = build_physical_orders(sweeps, num_experts)
    canonical_trace = json.dumps(
        [{str(layer): list(experts) for layer, experts in sorted(sweep.items())}
         for sweep in sweeps],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return {
        "schema": "vmodel.expert-plan.v1",
        "source_trace_sha256": hashlib.sha256(canonical_trace).hexdigest(),
        "source_model": str((source_document or {}).get("model", "")),
        "num_experts": num_experts,
        "sweep_count": len(sweeps),
        "orders": {
            name: {str(layer): list(order) for layer, order in sorted(layout.items())}
            for name, layout in orders.items()
        },
        "application_status": "logical-only; checkpoint bytes not rewritten",
    }


def write_plan(path: str | Path, plan: Mapping) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(plan), indent=2, sort_keys=True) + "\n")
    return path


def _format_bytes(value: int) -> str:
    return f"{value / 1e9:.3f} GB"


def main() -> None:
    """CLI for held-out plan/prefetch scoring: ``python -m runtime.expert_plan``."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--num-experts", type=int, default=0)
    parser.add_argument("--expert-page-bytes", type=int, default=0)
    parser.add_argument("--bandwidth-mbps", type=float, default=315.0)
    parser.add_argument("--request-overhead-ms", type=float, default=3.0)
    parser.add_argument("--coalesce-gap-pages", type=int, default=0)
    parser.add_argument(
        "--cache-pages", type=int, default=0,
        help="expert-only page capacity after subtracting pins, trunk, and KV",
    )
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument(
        "--skip-sweeps", type=int, default=1,
        help="skip leading sweeps before fitting; engine.generate traces "
             "normally begin with one prefill sweep",
    )
    parser.add_argument("--predict-top-m", type=int, default=8)
    parser.add_argument(
        "--opportunistic-pages", type=int, default=0,
        help="measured idle-I/O page budget per transition; zero is the "
             "fail-safe default",
    )
    parser.add_argument("--plan-out", default="")
    parser.add_argument("--report-out", default="")
    args = parser.parse_args()

    document, sweeps = load_trace(args.trace)
    num_experts = args.num_experts or int(document.get("num_experts", 0))
    if num_experts <= 0:
        num_experts = _infer_num_experts(sweeps)
    page_bytes = args.expert_page_bytes or int(
        document.get("expert_page_bytes", 0))
    if page_bytes <= 0:
        parser.error("expert page size missing; pass --expert-page-bytes")
    if not 0.0 < args.train_fraction < 1.0:
        parser.error("--train-fraction must be between zero and one")
    if args.skip_sweeps < 0 or args.cache_pages < 0:
        parser.error("--skip-sweeps and --cache-pages must be non-negative")
    if args.predict_top_m <= 0 or args.opportunistic_pages < 0:
        parser.error("prediction limits must be non-negative and top-m positive")

    original_sweeps = len(sweeps)
    if args.skip_sweeps and len(sweeps) > args.skip_sweeps:
        sweeps = sweeps[args.skip_sweeps:]
    if len(sweeps) >= 2:
        split = max(1, min(
            len(sweeps) - 1, int(len(sweeps) * args.train_fraction)))
        train, held_out = sweeps[:split], sweeps[split:]
        layout_label = "held-out"
    else:
        split = 1
        train = held_out = sweeps
        layout_label = "in-sample (only one sweep available)"

    orders = build_physical_orders(train, num_experts)
    strategies: list[tuple[str, LayoutResult]] = []
    for name, order in orders.items():
        independent = simulate_layout(
            held_out, order, expert_page_bytes=page_bytes,
            bandwidth_mbps=args.bandwidth_mbps,
            request_overhead_ms=args.request_overhead_ms,
            cache_pages=args.cache_pages,
        )
        coalesced = simulate_layout(
            held_out, order, expert_page_bytes=page_bytes,
            bandwidth_mbps=args.bandwidth_mbps,
            request_overhead_ms=args.request_overhead_ms,
            coalesce_gap_pages=args.coalesce_gap_pages,
            cache_pages=args.cache_pages,
        )
        strategies.extend(((f"{name}/independent", independent),
                           (f"{name}/coalesced", coalesced)))
    for bundle_pages in (2, 4):
        bundled = simulate_layout(
            held_out, orders["coactivation"], expert_page_bytes=page_bytes,
            bandwidth_mbps=args.bandwidth_mbps,
            request_overhead_ms=args.request_overhead_ms,
            bundle_pages=bundle_pages,
            cache_pages=args.cache_pages,
        )
        strategies.append((f"coactivation/fixed-bundle-{bundle_pages}", bundled))

    print(
        f"trace: {len(sweeps)} analyzed sweeps "
        f"({original_sweeps - len(sweeps)} skipped), {num_experts} experts, "
        f"{page_bytes / 1e6:.2f} MB/expert"
    )
    print(f"\n{layout_label} layout simulation "
          f"(cold {args.cache_pages}-page expert cache):")
    print(f"{'strategy':<34} {'hit':>7} {'requests':>10} {'physical':>13} "
          f"{'amplification':>14} {'I/O floor':>12}")
    for name, result in sorted(strategies,
                               key=lambda item: item[1].predicted_wall_s):
        hit_rate = result.cache_hits / result.accesses if result.accesses else 0.0
        print(f"{name:<34} {hit_rate * 100:>6.1f}% {result.requests:>10} "
              f"{_format_bytes(result.physical_bytes):>13} "
              f"{result.read_amplification:>13.3f}x "
              f"{result.predicted_wall_s:>11.3f}s")

    prediction = evaluate_transition_predictor(
        train, held_out, top_m=args.predict_top_m,
        expert_page_bytes=page_bytes,
    )
    gated_prediction = evaluate_transition_predictor(
        train, held_out, top_m=args.predict_top_m,
        opportunistic_pages=args.opportunistic_pages,
        expert_page_bytes=page_bytes,
    )
    persistence_input = sweeps[max(0, split - 1):]
    persistence = evaluate_adjacent_sweep_persistence(persistence_input)
    print("\nheld-out prediction:")
    print(f"  transition predictor: {prediction.precision * 100:.1f}% precision, "
          f"{prediction.recall * 100:.1f}% recall, "
          f"{_format_bytes(prediction.extra_device_bytes)} extra bytes if always issued")
    print(f"  opportunistic gate ({args.opportunistic_pages} pages/transition): "
          f"{_format_bytes(gated_prediction.extra_device_bytes)} extra device bytes, "
          f"{_format_bytes(gated_prediction.demand_stall_bytes_saved)} "
          "potential stall bytes hidden")
    print(f"  previous-sweep same-layer route: "
          f"{persistence.precision * 100:.1f}% precision, "
          f"{persistence.recall * 100:.1f}% recall")
    if gated_prediction.extra_device_bytes:
        ratio = (gated_prediction.demand_stall_bytes_saved
                 / gated_prediction.extra_device_bytes)
        print(f"  useful/wasted speculative-byte ratio: {ratio:.2f}")
    elif gated_prediction.predictions:
        print("  useful/wasted speculative-byte ratio: infinite (no wasted reads)")
    else:
        print("  useful/wasted speculative-byte ratio: n/a (gate issued no reads)")

    if args.plan_out:
        plan_path = write_plan(args.plan_out, compile_plan(
            train, num_experts=num_experts, source_document=document))
        print(f"\nlogical plan written to {plan_path}")

    if args.report_out:
        report = {
            "schema": "vmodel.expert-plan-report.v1",
            "trace": str(args.trace),
            "train_sweeps": len(train),
            "held_out_sweeps": len(held_out),
            "skipped_sweeps": original_sweeps - len(sweeps),
            "layout_evaluation": layout_label,
            "num_experts": num_experts,
            "expert_page_bytes": page_bytes,
            "bandwidth_mbps": args.bandwidth_mbps,
            "request_overhead_ms": args.request_overhead_ms,
            "cache_pages": args.cache_pages,
            "strategies": {name: asdict(result) for name, result in strategies},
            "transition_predictor": asdict(prediction),
            "opportunistic_predictor": asdict(gated_prediction),
            "adjacent_sweep_persistence": asdict(persistence),
        }
        report_path = Path(args.report_out)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"machine-readable report written to {report_path}")


if __name__ == "__main__":
    main()
