#!/usr/bin/env python3
"""Standalone SuffixDecoding/prompt-lookup audit on the tool-PIC fixture.

This probe deliberately does not modify or stand in for the serving runtime.  It
uses the existing tool-catalog prompt fixture to collect a small local greedy
trace, simulates prompt lookup and a bounded *linear* suffix-tree proposer over
that trace, and optionally drives the existing exact target verifier by replacing
``runtime.speculative.ngram_propose`` only inside this process.

The reported ``sweep_speedup_upper_bound`` assumes a multi-token verification
sweep costs the same as a one-token decode sweep.  That is an upper bound, not a
wall-time claim; the live section reports the actually observed wall time.

Any invocation of this file loads an MLX model and therefore must be wrapped by
the repository-wide ``/tmp/voom-mlx-benchmark.lock`` protocol.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
import tracemalloc
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class _Node:
    __slots__ = ("count", "children")

    def __init__(self):
        self.count = 0
        self.children: dict[int, _Node] = {}


@dataclass(frozen=True)
class Proposal:
    tokens: tuple[int, ...] = ()
    score: float = 0.0
    match_length: int = 0
    source: str = "none"


class BoundedSuffixTrie:
    """Small pure-Python audit index, not a production suffix-tree candidate.

    Every suffix is expanded to at most ``max_depth`` tokens.  This has the same
    bounded-context semantics needed by the probe, but intentionally trades the
    compressed C++ implementation's construction efficiency for inspectability.
    """

    def __init__(self, max_depth: int = 64):
        if max_depth <= 0:
            raise ValueError("max_depth must be positive")
        self.max_depth = max_depth
        self.root = _Node()
        self.nodes = 1
        self.tokens_indexed = 0
        self._active_tokens: list[int] | None = None

    def _child(self, node: _Node, token: int) -> _Node:
        child = node.children.get(token)
        if child is None:
            child = _Node()
            node.children[token] = child
            self.nodes += 1
        return child

    def add_sequence(self, tokens: Sequence[int]) -> None:
        values = [int(token) for token in tokens]
        self.tokens_indexed += len(values)
        for start in range(len(values)):
            node = self.root
            for token in values[start : start + self.max_depth]:
                node = self._child(node, token)
                node.count += 1

    def start_active_sequence(self, tokens: Sequence[int]) -> None:
        if self.tokens_indexed or self._active_tokens is not None:
            raise RuntimeError("active sequence needs a fresh trie")
        self._active_tokens = [int(token) for token in tokens]
        self.add_sequence(self._active_tokens)

    def append_active(self, tokens: Iterable[int]) -> None:
        if self._active_tokens is None:
            raise RuntimeError("start_active_sequence must be called first")
        for value in tokens:
            token = int(value)
            old_length = len(self._active_tokens)
            first = max(0, old_length - self.max_depth + 1)
            for start in range(first, old_length + 1):
                node = self.root
                for previous in self._active_tokens[start:old_length]:
                    node = node.children[previous]
                node = self._child(node, token)
                node.count += 1
            self._active_tokens.append(token)
            self.tokens_indexed += 1

    def _match(self, context: Sequence[int], length: int) -> _Node | None:
        node = self.root
        for token in context[-length:]:
            node = node.children.get(int(token))
            if node is None:
                return None
        return node

    def propose(
        self,
        context: Sequence[int],
        max_tokens: int,
        *,
        max_spec_factor: float = 1.0,
        max_spec_offset: float = 0.0,
        min_token_prob: float = 0.1,
        source: str,
    ) -> Proposal:
        if max_tokens <= 0 or len(context) < 2:
            return Proposal(source=source)
        best = Proposal(source=source)
        max_match = min(self.max_depth, len(context) - 1)
        for match_length in range(1, max_match + 1):
            node = self._match(context, match_length)
            if node is None:
                # If a suffix of length p is absent, every longer suffix is absent.
                break
            budget = min(
                max_tokens,
                max(0, int(match_length * max_spec_factor + max_spec_offset + 1e-6)),
            )
            probability = 1.0
            score = 0.0
            draft: list[int] = []
            current = node
            while len(draft) < budget and current.children:
                # Deterministic token-id tie-break; frequency is the real ranking.
                token, child = min(
                    current.children.items(),
                    key=lambda item: (-item[1].count, item[0]),
                )
                probability *= child.count / current.count
                if probability < min_token_prob:
                    break
                draft.append(token)
                score += probability
                current = child
            candidate = Proposal(tuple(draft), score, match_length, source)
            if candidate.score >= best.score:
                best = candidate
        return best


def _best_suffix_proposal(
    global_tree: BoundedSuffixTrie,
    local_tree: BoundedSuffixTrie,
    context: Sequence[int],
    max_tokens: int,
    *,
    factor: float,
    offset: float,
    min_probability: float,
) -> Proposal:
    local = local_tree.propose(
        context,
        max_tokens,
        max_spec_factor=factor,
        max_spec_offset=offset,
        min_token_prob=min_probability,
        source="local",
    )
    global_ = global_tree.propose(
        context,
        max_tokens,
        max_spec_factor=factor,
        max_spec_offset=offset,
        min_token_prob=min_probability,
        source="global",
    )
    # ArcticInference's Python wrapper also gives the local tree the tie.
    return local if local.score >= global_.score else global_


class StatefulSuffixProposer:
    """Process-local adapter for voom's existing linear verifier."""

    def __init__(
        self,
        training_outputs: Sequence[Sequence[int]],
        *,
        max_depth: int,
        factor: float,
        offset: float,
        min_probability: float,
    ):
        self.global_tree = BoundedSuffixTrie(max_depth)
        for output in training_outputs:
            self.global_tree.add_sequence(output)
        self.factor = factor
        self.offset = offset
        self.min_probability = min_probability
        self.local_tree: BoundedSuffixTrie | None = None
        self.context: list[int] = []
        self.lookup_ns: list[int] = []
        self.draft_lengths: list[int] = []
        self.match_lengths: list[int] = []
        self.sources: Counter[str] = Counter()

    def start_request(self, prompt_tokens: Sequence[int]) -> None:
        self.local_tree = BoundedSuffixTrie(self.global_tree.max_depth)
        self.local_tree.start_active_sequence(prompt_tokens)
        self.context = [int(token) for token in prompt_tokens]

    def __call__(self, tokens: list[int], k: int, *unused, **unused_kwargs) -> list[int]:
        if self.local_tree is None:
            raise RuntimeError("start_request was not called")
        if tokens[: len(self.context)] != self.context:
            raise RuntimeError("verifier context diverged from committed suffix state")
        if len(tokens) > len(self.context):
            self.local_tree.append_active(tokens[len(self.context) :])
            self.context = list(tokens)
        started = time.perf_counter_ns()
        proposal = _best_suffix_proposal(
            self.global_tree,
            self.local_tree,
            tokens,
            k,
            factor=self.factor,
            offset=self.offset,
            min_probability=self.min_probability,
        )
        self.lookup_ns.append(time.perf_counter_ns() - started)
        self.draft_lengths.append(len(proposal.tokens))
        self.match_lengths.append(proposal.match_length)
        self.sources[proposal.source] += 1
        return list(proposal.tokens)

    def finish_request(self, output_tokens: Sequence[int]) -> None:
        # Only target-committed tokens are inserted into the global cache.
        self.global_tree.add_sequence(output_tokens)
        self.local_tree = None
        self.context = []


class StatefulPromptLookupProposer:
    """Timing adapter around voom's current n-gram proposer (or no-draft control)."""

    def __init__(self, function, *, disabled: bool = False):
        self.function = function
        self.disabled = disabled
        self.lookup_ns: list[int] = []
        self.draft_lengths: list[int] = []
        self.match_lengths: list[int] = []
        self.sources: Counter[str] = Counter()

    def start_request(self, prompt_tokens: Sequence[int]) -> None:
        del prompt_tokens

    def __call__(self, tokens: list[int], k: int, *unused, **unused_kwargs) -> list[int]:
        started = time.perf_counter_ns()
        draft = [] if self.disabled else self.function(tokens, k)
        self.lookup_ns.append(time.perf_counter_ns() - started)
        self.draft_lengths.append(len(draft))
        self.match_lengths.append(0)
        self.sources["none" if self.disabled else "local"] += 1
        return draft

    def finish_request(self, output_tokens: Sequence[int]) -> None:
        del output_tokens


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return float(ordered[index])


def _simulate_trace(
    prompts: Sequence[Sequence[int]],
    outputs: Sequence[Sequence[int]],
    training_outputs: Sequence[Sequence[int]],
    *,
    kind: str,
    k: int,
    max_depth: int,
    factor: float = 1.0,
    offset: float = 0.0,
    min_probability: float = 0.1,
) -> dict:
    from runtime.speculative import ngram_propose

    global_tree = BoundedSuffixTrie(max_depth)
    for output in training_outputs:
        global_tree.add_sequence(output)
    proposed = accepted = sweeps = no_draft = 0
    lookup_ns: list[int] = []
    match_lengths: list[int] = []
    sources: Counter[str] = Counter()
    per_case = []
    for prompt, truth in zip(prompts, outputs):
        if not truth:
            continue
        local = BoundedSuffixTrie(max_depth)
        local.start_active_sequence(prompt)
        response = [int(truth[0])]
        local.append_active(response)
        case_proposed = case_accepted = case_sweeps = 0
        while len(response) < len(truth):
            remaining = len(truth) - len(response)
            round_k = min(k, max(0, remaining - 1))
            context = [int(token) for token in prompt] + response
            started = time.perf_counter_ns()
            if kind == "prompt_lookup":
                draft = ngram_propose(context, round_k)
                proposal = Proposal(tuple(draft), source="local")
            elif kind == "suffix_linear":
                proposal = _best_suffix_proposal(
                    global_tree,
                    local,
                    context,
                    round_k,
                    factor=factor,
                    offset=offset,
                    min_probability=min_probability,
                )
                draft = list(proposal.tokens)
            else:
                raise ValueError(kind)
            lookup_ns.append(time.perf_counter_ns() - started)
            sources[proposal.source] += 1
            match_lengths.append(proposal.match_length)
            if not draft:
                no_draft += 1
            matched = 0
            while (matched < len(draft)
                   and draft[matched] == int(truth[len(response) + matched])):
                matched += 1
            committed = draft[:matched] + [int(truth[len(response) + matched])]
            response.extend(committed)
            local.append_active(committed)
            proposed += len(draft)
            accepted += matched
            sweeps += 1
            case_proposed += len(draft)
            case_accepted += matched
            case_sweeps += 1
        if response != [int(token) for token in truth]:
            raise AssertionError("offline verification changed target tokens")
        global_tree.add_sequence(truth)
        per_case.append({
            "output_tokens": len(truth),
            "target_sweeps": case_sweeps,
            "proposed": case_proposed,
            "accepted": case_accepted,
        })

    plain_sweeps = sum(max(0, len(output) - 1) for output in outputs)
    lookup_us = [value / 1000 for value in lookup_ns]
    return {
        "kind": kind,
        "k": k,
        "factor": factor if kind == "suffix_linear" else None,
        "min_probability": min_probability if kind == "suffix_linear" else None,
        "plain_target_sweeps": plain_sweeps,
        "target_sweeps": sweeps,
        "sweep_speedup_upper_bound": round(plain_sweeps / sweeps, 4) if sweeps else None,
        "proposed": proposed,
        "accepted": accepted,
        "proposal_acceptance": round(accepted / proposed, 6) if proposed else 0.0,
        "committed_tokens_per_target_sweep": round(plain_sweeps / sweeps, 6) if sweeps else 0.0,
        "no_draft_rounds": no_draft,
        "lookup_us_total": round(sum(lookup_us), 3),
        "lookup_us_per_round_mean": round(statistics.mean(lookup_us), 3) if lookup_us else 0.0,
        "lookup_us_per_round_p95": round(_percentile(lookup_us, 0.95), 3),
        "lookup_us_per_proposed_token": round(sum(lookup_us) / proposed, 3) if proposed else None,
        "match_length_mean": round(statistics.mean(match_lengths), 3) if match_lengths else 0.0,
        "sources": dict(sources),
        "cases": per_case,
    }


def _build_measured_tree(outputs: Sequence[Sequence[int]], max_depth: int):
    gc.collect()
    tracemalloc.start()
    before, _ = tracemalloc.get_traced_memory()
    started = time.perf_counter()
    tree = BoundedSuffixTrie(max_depth)
    for output in outputs:
        tree.add_sequence(output)
    build_s = time.perf_counter() - started
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return tree, {
        "training_requests": len(outputs),
        "training_output_tokens": sum(len(output) for output in outputs),
        "max_depth": max_depth,
        "nodes": tree.nodes,
        "build_s": round(build_s, 6),
        "tracemalloc_net_bytes": max(0, current - before),
        "tracemalloc_peak_delta_bytes": max(0, peak - before),
        "implementation": "uncompressed pure-Python bounded suffix trie",
    }


def _ratio(numerator: float, denominator: float) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _live_suffix_run(engine, prompts, plain_results, training_outputs, args):
    import runtime.speculative as speculative_module
    from runtime.speculative import SpeculativeDecoder

    original = speculative_module.ngram_propose
    if args.live_proposer == "suffix":
        proposer = StatefulSuffixProposer(
            training_outputs,
            max_depth=args.max_depth,
            factor=args.factor,
            offset=args.offset,
            min_probability=args.min_probability,
        )
    else:
        proposer = StatefulPromptLookupProposer(
            original, disabled=args.live_proposer == "none")
    decoder = SpeculativeDecoder(engine, None, k=args.k)
    suffix_results = []
    try:
        speculative_module.ngram_propose = proposer
        for prompt in prompts:
            proposer.start_request(prompt.token_ids)
            result = decoder.generate(
                prompt,
                max_tokens=args.max_tokens,
                stop=[],
                encoded_ids=list(prompt.token_ids),
            )
            proposer.finish_request(result["tokens"])
            suffix_results.append(result)
    finally:
        speculative_module.ngram_propose = original

    exact = [plain["tokens"] == suffix["tokens"]
             for plain, suffix in zip(plain_results, suffix_results)]
    mismatch_details = []
    for case_index, (plain, suffix) in enumerate(zip(plain_results, suffix_results)):
        if plain["tokens"] == suffix["tokens"]:
            continue
        shared = 0
        for expected, actual in zip(plain["tokens"], suffix["tokens"]):
            if expected != actual:
                break
            shared += 1
        mismatch_details.append({
            "case_index": case_index,
            "shared_output_prefix_tokens": shared,
            "plain_token": plain["tokens"][shared]
            if shared < len(plain["tokens"]) else None,
            "suffix_token": suffix["tokens"][shared]
            if shared < len(suffix["tokens"]) else None,
            "plain_output_tokens": len(plain["tokens"]),
            "suffix_output_tokens": len(suffix["tokens"]),
        })
    lookup_us = [value / 1000 for value in proposer.lookup_ns]
    stats = [result["stats"] for result in suffix_results]
    plain_total = sum(result["total_s"] for result in plain_results)
    suffix_total = sum(result["total_s"] for result in suffix_results)
    plain_decode = sum(result["decode_s"] for result in plain_results)
    suffix_decode = sum(result["decode_s"] for result in suffix_results)
    steady_plain = sum(result["total_s"] for result in plain_results[1:])
    steady_suffix = sum(result["total_s"] for result in suffix_results[1:])
    report = {
        "exact_cases": sum(exact),
        "cases": len(exact),
        "token_ids_compared": sum(len(result["tokens"]) for result in plain_results),
        "mismatched_case_indices": [index for index, same in enumerate(exact) if not same],
        "mismatch_details": mismatch_details,
        "plain_total_s": round(plain_total, 6),
        "suffix_total_s": round(suffix_total, 6),
        "wall_speedup": _ratio(plain_total, suffix_total),
        "steady_state_excluding_first_wall_speedup": _ratio(steady_plain, steady_suffix),
        "plain_prefill_s": round(sum(result["prefill_s"] for result in plain_results), 6),
        "suffix_prefill_s": round(sum(result["prefill_s"] for result in suffix_results), 6),
        "plain_decode_s": round(plain_decode, 6),
        "suffix_decode_s": round(suffix_decode, 6),
        "decode_wall_speedup": _ratio(plain_decode, suffix_decode),
        "proposed": sum(item.proposed for item in stats),
        "accepted": sum(item.accepted for item in stats),
        "proposal_acceptance": _ratio(
            sum(item.accepted for item in stats),
            sum(item.proposed for item in stats),
        ),
        "target_decode_sweeps": sum(max(0, item.sweeps - 1) for item in stats),
        "plain_decode_steps": sum(max(0, len(result["tokens"]) - 1)
                                  for result in plain_results),
        "lookup_us_total": round(sum(lookup_us), 3),
        "lookup_us_per_round_mean": round(statistics.mean(lookup_us), 3) if lookup_us else 0.0,
        "lookup_us_per_round_p95": round(_percentile(lookup_us, 0.95), 3),
        "draft_length_mean": round(statistics.mean(proposer.draft_lengths), 3)
        if proposer.draft_lengths else 0.0,
        "match_length_mean": round(statistics.mean(proposer.match_lengths), 3)
        if proposer.match_lengths else 0.0,
        "proposal_sources": dict(proposer.sources),
        "live_proposer": args.live_proposer,
        "plain_peak_metal_bytes": max(result["true_peak_metal_bytes"]
                                      for result in plain_results),
        "suffix_peak_metal_bytes": max(result["true_peak_metal_bytes"]
                                       for result in suffix_results),
        "kv_layout": "position_free_shared" if args.shared_pages else "concatenated_or_stepped",
        "tool_pic_reuse_exercised": False,
        "note": (
            "Prepared prompts contain tool-capsule spans, but SpeculativeDecoder "
            "re-prefills from scratch and therefore does not exercise capsule reuse."
        ),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path.home() / "models/Qwen2.5-1.5B-Instruct-mlx-mxfp4",
    )
    parser.add_argument("--tools", type=int, default=24)
    parser.add_argument("--cases", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--max-depth", type=int, default=64)
    parser.add_argument("--factor", type=float, default=1.0)
    parser.add_argument("--offset", type=float, default=0.0)
    parser.add_argument("--min-probability", type=float, default=0.1)
    parser.add_argument("--cache-mb", type=int, default=1800)
    parser.add_argument(
        "--prefill-chunk-size",
        type=int,
        default=256,
        help="production fixture default is 256; use 0 only to isolate prefill-shape effects",
    )
    parser.add_argument(
        "--shared-pages",
        action="store_true",
        help="cold-prefill/verify with PositionFreeKVCache; does not exercise PIC reuse",
    )
    parser.add_argument(
        "--streamed-target",
        action="store_true",
        help="disable the resident lazy decode loop so plain and verifier paths share streamed arithmetic",
    )
    parser.add_argument(
        "--live-proposer",
        choices=("suffix", "prompt_lookup", "none"),
        default="suffix",
        help="proposal source used only by the live exact-verifier section",
    )
    args = parser.parse_args()
    model = args.model.expanduser().resolve()
    if not (model / "config.json").exists():
        parser.error(f"model is not a local checkpoint: {model}")
    if (args.tools < 8 or args.cases <= 1 or args.max_tokens <= 1 or args.k <= 0
            or args.max_depth <= 1 or args.factor < 0
            or args.prefill_chunk_size < 0
            or not 0 <= args.min_probability <= 1):
        parser.error("invalid probe bounds")

    import mlx.core as mx
    from runtime.engine import StreamingEngine
    from tests.fixtures.qwen_tool_pic_gate import _prompt, _runtime

    rc = _runtime(
        model,
        pic=args.shared_pages,
        cache_mb=args.cache_mb,
        repair=4,
        chunk=args.prefill_chunk_size,
        shared_pages=args.shared_pages,
    )
    if not args.shared_pages:
        rc.hot_prompt_kv = False
        rc.tool_pic = False
    if args.streamed_target:
        rc.resident_fast_decode = False
    rc.prompt_kv_dir = ""
    rc.hot_prompt_kv_persist_dir = ""
    engine = StreamingEngine(model, rc)
    try:
        indices = [((index * 4) + 1) % args.tools for index in range(args.cases)]
        train_cases = [
            (f"tool_{index:03d}", f"src/training_case_{index}.py")
            for index in indices
        ]
        eval_cases = [
            (f"tool_{index:03d}", f"src/evaluation_case_{index}.py")
            for index in indices
        ]
        train_prompts = [
            _prompt(
                engine,
                model,
                args.tools,
                edited=False,
                target=target,
                path=path,
                max_tokens=args.max_tokens,
            )
            for target, path in train_cases
        ]
        eval_prompts = [
            _prompt(
                engine,
                model,
                args.tools,
                edited=False,
                target=target,
                path=path,
                max_tokens=args.max_tokens,
            )
            for target, path in eval_cases
        ]

        # Warm the ordinary engine once; the measured trace begins after this.
        engine.release_request_state()
        engine.generate(train_prompts[0], max_tokens=2, stop=[])
        training_results = []
        for prompt in train_prompts:
            engine.release_request_state()
            training_results.append(engine.generate(
                prompt, max_tokens=args.max_tokens, stop=[]))
        plain_results = []
        for prompt in eval_prompts:
            engine.release_request_state()
            plain_results.append(engine.generate(
                prompt, max_tokens=args.max_tokens, stop=[]))

        training_outputs = [result["tokens"] for result in training_results]
        eval_outputs = [result["tokens"] for result in plain_results]
        prompt_ids = [list(prompt.token_ids) for prompt in eval_prompts]
        _tree, corpus = _build_measured_tree(training_outputs, args.max_depth)

        prompt_lookup = _simulate_trace(
            prompt_ids,
            eval_outputs,
            training_outputs,
            kind="prompt_lookup",
            k=args.k,
            max_depth=args.max_depth,
        )
        suffix = _simulate_trace(
            prompt_ids,
            eval_outputs,
            training_outputs,
            kind="suffix_linear",
            k=args.k,
            max_depth=args.max_depth,
            factor=args.factor,
            offset=args.offset,
            min_probability=args.min_probability,
        )
        replay = _simulate_trace(
            [list(prompt.token_ids) for prompt in train_prompts],
            training_outputs,
            training_outputs,
            kind="suffix_linear",
            k=args.k,
            max_depth=args.max_depth,
            factor=args.factor,
            offset=args.offset,
            min_probability=args.min_probability,
        )

        grid = []
        for grid_k in sorted({2, 4, 6, 8, 12, 16, args.k}):
            for factor in (0.5, 1.0, 2.0, 4.0):
                row = _simulate_trace(
                    prompt_ids,
                    eval_outputs,
                    training_outputs,
                    kind="suffix_linear",
                    k=grid_k,
                    max_depth=args.max_depth,
                    factor=factor,
                    offset=args.offset,
                    min_probability=args.min_probability,
                )
                grid.append({
                    key: row[key]
                    for key in (
                        "k", "factor", "target_sweeps",
                        "sweep_speedup_upper_bound", "proposed", "accepted",
                        "proposal_acceptance", "lookup_us_per_round_mean",
                    )
                })
        grid.sort(key=lambda row: (
            -(row["sweep_speedup_upper_bound"] or 0),
            row["proposed"],
        ))

        live = _live_suffix_run(
            engine,
            eval_prompts,
            plain_results,
            training_outputs,
            args,
        )
        report = {
            "probe": "suffix-decoding-tool-trace-v1",
            "model": str(model),
            "fixture": "tests/fixtures/qwen_tool_pic_gate.py",
            "configuration": {
                "tools": args.tools,
                "training_cases": len(train_cases),
                "held_out_cases": len(eval_cases),
                "max_tokens": args.max_tokens,
                "k": args.k,
                "max_depth": args.max_depth,
                "factor": args.factor,
                "offset": args.offset,
                "min_probability": args.min_probability,
                "shared_pages": args.shared_pages,
                "streamed_target": args.streamed_target,
                "prefill_chunk_size": args.prefill_chunk_size,
                "live_proposer": args.live_proposer,
            },
            "prepared_prompts": {
                "token_counts": [len(prompt.token_ids) for prompt in eval_prompts],
                "tool_capsule_counts": [len(prompt.tool_capsules) for prompt in eval_prompts],
            },
            "corpus": corpus,
            "held_out_offline": {
                "prompt_lookup": prompt_lookup,
                "suffix_linear": suffix,
            },
            "exact_replay_ceiling": replay,
            "offline_parameter_grid_top8": grid[:8],
            "live_suffix_linear": live,
            "interpretation_guardrails": {
                "offline_sweep_speedup_is_wall_speedup": False,
                "tree_speculation_measured": False,
                "new_model_downloaded": False,
                "lossy_behavior_enabled_by_suffix_probe": False,
                "global_cache_contains_outputs_only": True,
            },
        }
        print(json.dumps(report, indent=2))
        return 0 if live["exact_cases"] == live["cases"] else 1
    finally:
        engine.close()
        mx.clear_cache()


if __name__ == "__main__":
    raise SystemExit(main())
