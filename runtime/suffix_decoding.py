"""Engine-local, shared-prefill linear SuffixDecoding.

This module is deliberately narrower than the draft-model adapter in
``runtime.speculative``:

* the target's ordinary ``StreamingEngine.generate`` path performs prefill,
  including exact hot-prefix or explicitly approximate tool-PIC assembly;
* this module only proposes token IDs on CPU and verifies them against that
  already-created target KV state;
* only target-verified, actually returned tokens enter request/global history.

The global history is engine-local and intentionally not serialized.  Enabling
it on a shared multi-tenant engine can expose workload membership through timing,
even though target verification prevents an unverified cached token from being
returned.  The opt-in is therefore documented and telemetered as single-tenant.

The trie is an inspectable Python implementation rather than ArcticInference's
compressed C++ suffix tree.  Token, node, request, and conservative byte bounds
are all enforced.  Half of the configured node/byte budget is reserved for the
active request tree and half for completed-output history, bounding their
combined structural footprint.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import mlx.core as mx

from .kv_cache import KVCache


# Measured by the standalone audit at about 260 B/node on its small corpus.
# Use a deliberately larger accounting unit so the configured byte cap is a
# conservative structural bound rather than a claim about exact Python RSS.
_ACCOUNTED_NODE_BYTES = 320
_ACCOUNTED_SEQUENCE_BYTES = 80
_ACCOUNTED_TOKEN_BYTES = 8


class _Node:
    __slots__ = ("count", "children")

    def __init__(self):
        self.count = 0
        self.children: dict[int, _Node] = {}


@dataclass(frozen=True)
class SuffixProposal:
    tokens: tuple[int, ...] = ()
    score: float = 0.0
    match_length: int = 0
    source: str = "none"
    cpu_s: float = 0.0


@dataclass
class SuffixDecodeStats:
    proposed: int = 0
    accepted: int = 0
    sweeps: int = 0
    cpu_s: float = 0.0
    lookup_match_tokens: int = 0
    local_rounds: int = 0
    global_rounds: int = 0


@dataclass
class SharedPrefillSuffixResult:
    logits: object
    token_times: list[float]
    stop_text: str | None
    stop_sequence: str | None
    stats: SuffixDecodeStats


def _worst_case_nodes(token_count: int, max_depth: int) -> int:
    """Maximum non-root trie nodes introduced by one all-unique sequence."""
    if token_count <= 0:
        return 0
    if token_count <= max_depth:
        return token_count * (token_count + 1) // 2
    return max_depth * (max_depth + 1) // 2 + (token_count - max_depth) * max_depth


class _BoundedSuffixTrie:
    def __init__(self, max_depth: int):
        self.max_depth = max_depth
        self.root = _Node()
        self.nodes = 1
        self._active_tokens: list[int] | None = None

    @property
    def accounted_bytes(self) -> int:
        return self.nodes * _ACCOUNTED_NODE_BYTES

    def _child(self, node: _Node, token: int) -> _Node:
        child = node.children.get(token)
        if child is None:
            child = _Node()
            node.children[token] = child
            self.nodes += 1
        return child

    def add_sequence(self, tokens: Sequence[int]) -> None:
        values = [int(token) for token in tokens]
        for start in range(len(values)):
            node = self.root
            for token in values[start : start + self.max_depth]:
                node = self._child(node, token)
                node.count += 1

    def start_active_sequence(self, tokens: Sequence[int]) -> None:
        if self._active_tokens is not None or self.nodes != 1:
            raise RuntimeError("active request needs a fresh suffix trie")
        self._active_tokens = [int(token) for token in tokens]
        self.add_sequence(self._active_tokens)

    @property
    def active_tokens(self) -> tuple[int, ...]:
        return tuple(self._active_tokens or ())

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
        factor: float,
        min_probability: float,
        source: str,
    ) -> SuffixProposal:
        if max_tokens <= 0 or len(context) < 2:
            return SuffixProposal(source=source)
        best = SuffixProposal(source=source)
        max_match = min(self.max_depth, len(context) - 1)
        for match_length in range(1, max_match + 1):
            node = self._match(context, match_length)
            if node is None:
                # Every longer suffix contains this absent suffix.
                break
            budget = min(max_tokens, max(0, int(match_length * factor + 1e-6)))
            probability = 1.0
            score = 0.0
            draft: list[int] = []
            current = node
            while len(draft) < budget and current.children:
                token, child = min(
                    current.children.items(),
                    key=lambda item: (-item[1].count, item[0]),
                )
                probability *= child.count / current.count
                if probability < min_probability:
                    break
                draft.append(token)
                score += probability
                current = child
            candidate = SuffixProposal(
                tuple(draft), score, match_length, source)
            if candidate.score >= best.score:
                best = candidate
        return best


class SuffixRequestState:
    """Bounded prompt/current-output tree for one active generation."""

    def __init__(self, prompt_tokens: Sequence[int], *, max_depth: int,
                 max_local_tokens: int, node_limit: int, byte_limit: int):
        self.max_depth = max_depth
        self.max_local_tokens = self._fit_token_limit(
            max_local_tokens, max_depth, node_limit, byte_limit)
        self._tree = _BoundedSuffixTrie(max_depth)
        tail = list(prompt_tokens[-self.max_local_tokens :])
        self._tree.start_active_sequence(tail)

    @staticmethod
    def _fit_token_limit(requested: int, max_depth: int,
                         node_limit: int, byte_limit: int) -> int:
        def fits(limit: int) -> bool:
            nodes = 1 + _worst_case_nodes(limit, max_depth)
            return (
                nodes <= node_limit
                and nodes * _ACCOUNTED_NODE_BYTES <= byte_limit
            )

        if not fits(1):
            raise ValueError(
                "suffix decoding local node/byte budget cannot hold one token")
        low, high = 1, max(1, requested)
        while low < high:
            middle = (low + high + 1) // 2
            if fits(middle):
                low = middle
            else:
                high = middle - 1
        return low

    @property
    def tokens(self) -> tuple[int, ...]:
        return self._tree.active_tokens

    @property
    def nodes(self) -> int:
        return self._tree.nodes

    @property
    def accounted_bytes(self) -> int:
        return self._tree.accounted_bytes

    def append_committed(self, tokens: Sequence[int]) -> None:
        values = [int(token) for token in tokens]
        if not values:
            return
        existing = list(self._tree.active_tokens)
        if len(existing) + len(values) <= self.max_local_tokens:
            self._tree.append_active(values)
            return
        tail = (existing + values)[-self.max_local_tokens :]
        self._tree = _BoundedSuffixTrie(self.max_depth)
        self._tree.start_active_sequence(tail)

    def propose(self, max_tokens: int, *, factor: float,
                min_probability: float) -> SuffixProposal:
        return self._tree.propose(
            self.tokens,
            max_tokens,
            factor=factor,
            min_probability=min_probability,
            source="local",
        )


class SuffixDecodingCache:
    """FIFO-bounded, engine-local cache of completed target outputs."""

    def __init__(
        self,
        *,
        identity: str,
        max_depth: int = 64,
        max_spec_tokens: int = 6,
        factor: float = 4.0,
        min_probability: float = 0.1,
        max_cached_requests: int = 256,
        max_cached_tokens: int = 32_768,
        max_nodes: int = 262_144,
        max_bytes: int = 96_000_000,
        max_local_tokens: int = 2_048,
    ):
        validate_suffix_settings(
            max_depth=max_depth,
            max_spec_tokens=max_spec_tokens,
            factor=factor,
            min_probability=min_probability,
            max_cached_requests=max_cached_requests,
            max_cached_tokens=max_cached_tokens,
            max_nodes=max_nodes,
            max_bytes=max_bytes,
            max_local_tokens=max_local_tokens,
        )
        self.identity = identity
        self.max_depth = max_depth
        self.max_spec_tokens = max_spec_tokens
        self.factor = factor
        self.min_probability = min_probability
        self.max_cached_requests = max_cached_requests
        self.max_cached_tokens = max_cached_tokens
        self.max_nodes = max_nodes
        self.max_bytes = max_bytes
        self.max_local_tokens = max_local_tokens
        self._global_node_limit = max(2, max_nodes // 2)
        self._local_node_limit = max(2, max_nodes - self._global_node_limit)
        self._global_byte_limit = max(
            _ACCOUNTED_NODE_BYTES, max_bytes // 2)
        self._local_byte_limit = max(
            _ACCOUNTED_NODE_BYTES, max_bytes - self._global_byte_limit)
        self._global = _BoundedSuffixTrie(max_depth)
        self._history: deque[tuple[int, ...]] = deque()
        self._history_tokens = 0
        self.evicted_requests = 0
        self.rejected_requests = 0

    @property
    def cached_requests(self) -> int:
        return len(self._history)

    @property
    def cached_tokens(self) -> int:
        return self._history_tokens

    @property
    def global_nodes(self) -> int:
        return self._global.nodes

    @property
    def accounted_bytes(self) -> int:
        return (
            self._global.accounted_bytes
            + self.cached_requests * _ACCOUNTED_SEQUENCE_BYTES
            + self.cached_tokens * _ACCOUNTED_TOKEN_BYTES
        )

    def begin_request(self, prompt_tokens: Sequence[int]) -> SuffixRequestState:
        return SuffixRequestState(
            prompt_tokens,
            max_depth=self.max_depth,
            max_local_tokens=self.max_local_tokens,
            node_limit=self._local_node_limit,
            byte_limit=self._local_byte_limit,
        )

    def propose(self, state: SuffixRequestState,
                max_tokens: int | None = None) -> SuffixProposal:
        budget = self.max_spec_tokens if max_tokens is None else min(
            self.max_spec_tokens, max_tokens)
        started = time.process_time()
        local = state.propose(
            budget, factor=self.factor,
            min_probability=self.min_probability)
        global_ = self._global.propose(
            state.tokens,
            budget,
            factor=self.factor,
            min_probability=self.min_probability,
            source="global",
        )
        selected = local if local.score >= global_.score else global_
        return SuffixProposal(
            selected.tokens,
            selected.score,
            selected.match_length,
            selected.source,
            time.process_time() - started,
        )

    def _rebuild(self) -> None:
        self._global = _BoundedSuffixTrie(self.max_depth)
        for sequence in self._history:
            self._global.add_sequence(sequence)

    def _evict_oldest(self) -> None:
        sequence = self._history.popleft()
        self._history_tokens -= len(sequence)
        self.evicted_requests += 1
        # Rebuild compacts Python dictionaries whose removed slots would
        # otherwise retain capacity and defeat the conservative byte budget.
        self._rebuild()

    def add_output(self, output_tokens: Sequence[int]) -> bool:
        sequence = tuple(int(token) for token in output_tokens)
        if not sequence or self.max_cached_requests == 0:
            return False
        worst_nodes = _worst_case_nodes(len(sequence), self.max_depth)
        retained_bytes = (
            _ACCOUNTED_SEQUENCE_BYTES
            + len(sequence) * _ACCOUNTED_TOKEN_BYTES
        )
        worst_bytes = worst_nodes * _ACCOUNTED_NODE_BYTES + retained_bytes
        if (len(sequence) > self.max_cached_tokens
                or worst_nodes + 1 > self._global_node_limit
                or worst_bytes > self._global_byte_limit):
            self.rejected_requests += 1
            return False
        while self._history and (
            self.cached_requests + 1 > self.max_cached_requests
            or self.cached_tokens + len(sequence) > self.max_cached_tokens
            or self.global_nodes + worst_nodes > self._global_node_limit
            or self.accounted_bytes + worst_bytes > self._global_byte_limit
        ):
            self._evict_oldest()
        if (self.cached_requests + 1 > self.max_cached_requests
                or self.cached_tokens + len(sequence) > self.max_cached_tokens
                or self.global_nodes + worst_nodes > self._global_node_limit
                or self.accounted_bytes + worst_bytes > self._global_byte_limit):
            self.rejected_requests += 1
            return False
        self._global.add_sequence(sequence)
        self._history.append(sequence)
        self._history_tokens += len(sequence)
        return True

    def telemetry(self, state: SuffixRequestState | None = None) -> dict:
        local_nodes = state.nodes if state is not None else 0
        local_bytes = state.accounted_bytes if state is not None else 0
        return {
            "suffix_decoding_cache_requests": self.cached_requests,
            "suffix_decoding_cache_tokens": self.cached_tokens,
            "suffix_decoding_cache_nodes": self.global_nodes,
            "suffix_decoding_cache_bytes": self.accounted_bytes,
            "suffix_decoding_local_nodes": local_nodes,
            "suffix_decoding_local_bytes": local_bytes,
            "suffix_decoding_cache_evictions": self.evicted_requests,
            "suffix_decoding_cache_rejections": self.rejected_requests,
            "suffix_decoding_cache_identity": self.identity[:16],
        }


def validate_suffix_settings(
    *,
    max_depth: int,
    max_spec_tokens: int,
    factor: float,
    min_probability: float,
    max_cached_requests: int,
    max_cached_tokens: int,
    max_nodes: int,
    max_bytes: int,
    max_local_tokens: int,
) -> None:
    integer_values = {
        "suffix_decoding_max_depth": (max_depth, 2),
        "suffix_decoding_k": (max_spec_tokens, 1),
        "suffix_decoding_max_cached_requests": (max_cached_requests, 0),
        "suffix_decoding_max_cached_tokens": (max_cached_tokens, 1),
        "suffix_decoding_max_nodes": (max_nodes, 4),
        # Each half-budget must hold at least a root plus one local token.
        "suffix_decoding_max_bytes": (max_bytes, 4 * _ACCOUNTED_NODE_BYTES),
        "suffix_decoding_max_local_tokens": (max_local_tokens, 1),
    }
    for name, (value, minimum) in integer_values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if (isinstance(factor, bool) or not isinstance(factor, (int, float))
            or not math.isfinite(float(factor)) or factor < 0):
        raise ValueError("suffix_decoding_factor must be finite and >= 0")
    if (isinstance(min_probability, bool)
            or not isinstance(min_probability, (int, float))
            or not math.isfinite(float(min_probability))
            or not 0 <= min_probability <= 1):
        raise ValueError(
            "suffix_decoding_min_probability must be between 0 and 1")


def model_tokenizer_fingerprint(model_dir: str | Path) -> str:
    """Cheap engine-local model/tokenizer identity; never hashes multi-GB weights."""
    directory = Path(model_dir).resolve()
    digest = hashlib.sha256(str(directory).encode("utf-8"))
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json",
                 "model.safetensors.index.json"):
        path = directory / name
        digest.update(name.encode("utf-8"))
        if path.exists():
            with path.open("rb") as handle:
                while chunk := handle.read(1 << 20):
                    digest.update(chunk)
    for path in sorted(directory.glob("*.safetensors")):
        stat = path.stat()
        digest.update(path.name.encode("utf-8"))
        digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode("ascii"))
    return digest.hexdigest()


def select_verified_tokens(proposed: Sequence[int],
                           greedy_tokens: Sequence[int]) -> tuple[int, list[int]]:
    """Return accepted proposal count and accepted-prefix-plus-target-bonus."""
    draft = [int(token) for token in proposed]
    target = [int(token) for token in greedy_tokens]
    if len(target) != len(draft) + 1:
        raise ValueError("target verifier must return one logit per input token")
    accepted = 0
    while accepted < len(draft) and target[accepted] == draft[accepted]:
        accepted += 1
    return accepted, draft[:accepted] + [target[accepted]]


def fallback_reason(engine, kv, sampling, constraint, *, terminal: bool) -> str | None:
    """Return why shared-prefill suffix decode is ineligible, else ``None``."""
    if not engine.rc.suffix_decoding:
        return "disabled"
    if terminal:
        return "terminal-after-prefill"
    if not sampling.is_greedy:
        return "stochastic-sampling"
    if constraint is not None:
        return "constrained-decoding"
    if engine.rc.resident_fast_decode or engine.rc.resident_moe_decode:
        return "resident-decode"
    if engine.cfg.num_experts or engine.cfg.model_type in ("glm_moe_dsa", "gpt_oss"):
        return "non-dense-target"
    if getattr(engine.cfg, "vision_config", None):
        return "vision-target"
    if type(kv) is not KVCache:
        return "unproven-kv-layout"
    if getattr(kv, "compressed_mla", False):
        return "compressed-kv"
    if engine._embed_rows is not None or engine._streamed_lm_head is not None:
        return "streamed-embedding-or-head"
    if engine.rc.rerank_lm_head:
        return "reranked-head"
    return None


def run_shared_prefill_suffix_decode(
    engine,
    cache: SuffixDecodingCache,
    state: SuffixRequestState,
    *,
    prompt_tokens: Sequence[int],
    generated: list[int],
    kv: KVCache,
    logits,
    max_tokens: int,
    stop: Sequence[str],
    stream_decoder,
    on_token,
    stop_match,
) -> SharedPrefillSuffixResult:
    """Verify linear suffix proposals against an already-prefilled ordinary KV."""
    stats = SuffixDecodeStats()
    token_times: list[float] = []
    stop_text = None
    matched_stop_sequence = None
    eos = set(engine.cfg.eos_token_ids)
    prompt_length = len(prompt_tokens)

    while len(generated) < max_tokens and generated[-1] not in eos:
        round_started = time.perf_counter()
        remaining = max_tokens - len(generated)
        proposal = cache.propose(state, max_tokens=max(0, remaining - 1))
        draft = list(proposal.tokens)
        stats.proposed += len(draft)
        stats.cpu_s += proposal.cpu_s
        stats.lookup_match_tokens += proposal.match_length
        if proposal.source == "global":
            stats.global_rounds += 1
        else:
            stats.local_rounds += 1

        base = kv.offset
        expected_base = prompt_length + len(generated) - 1
        if base != expected_base:
            raise RuntimeError(
                f"suffix target KV desync: {base} != {expected_base}")
        boundary = mx.get_active_memory()
        if engine.governor is not None and engine._token_transient:
            engine.governor.reserve(engine._token_transient)
        mx.reset_peak_memory()
        try:
            window_logits = engine.forward_tokens_serial_positions(
                [generated[-1]] + draft, kv)
            greedy = [int(value) for value in mx.argmax(
                window_logits, axis=-1).reshape(-1)]
            accepted, verified = select_verified_tokens(draft, greedy)
        except BaseException:
            # A verifier failure must not strand a partially appended window in
            # the request state. It is safer to surface the error than to resume
            # ordinary decoding after an unproved partial target sweep.
            if kv.offset > base:
                kv.trim(base)
            raise
        stats.sweeps += 1

        # First discard every input after the fully target-accepted prefix.
        kv.trim(base + 1 + accepted)
        emitted_this_round: list[int] = []
        for index, token in enumerate(verified):
            generated.append(token)
            emitted_this_round.append(token)
            logits = window_logits[index]
            if stop:
                decoded = engine.tokenizer.decode(generated)
                match = stop_match(decoded)
                if match is not None:
                    cut, _order, matched_stop_sequence = match
                    stop_text = decoded[:cut]
            if stop_text is None and stream_decoder is not None:
                delta = stream_decoder.push(generated)
                if delta:
                    on_token(delta)
            if (stop_text is not None or token in eos
                    or len(generated) >= max_tokens):
                break

        committed_proposals = min(accepted, len(emitted_this_round))
        stats.accepted += committed_proposals
        state.append_committed(emitted_this_round)

        # Stop/EOS may cut a fully accepted window before its end.  Preserve the
        # ordinary endpoint contract: the final returned token is not in KV.
        endpoint = prompt_length + len(generated) - 1
        if kv.offset > endpoint:
            kv.trim(endpoint)
        engine._token_transient = max(
            engine._token_transient, mx.get_peak_memory() - boundary)
        engine._note_true_peak()
        elapsed = time.perf_counter() - round_started
        if emitted_this_round:
            token_times.extend(
                [elapsed / len(emitted_this_round)] * len(emitted_this_round))
        if (stop_text is not None or generated[-1] in eos
                or len(generated) >= max_tokens):
            break

    return SharedPrefillSuffixResult(
        logits=logits,
        token_times=token_times,
        stop_text=stop_text,
        stop_sequence=matched_stop_sequence,
        stats=stats,
    )
