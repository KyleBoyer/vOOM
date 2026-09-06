"""Pure CPU history selection/ownership tests; no model or MLX imports."""

from dataclasses import FrozenInstanceError
import sys

import pytest

from runtime.completed_output_history import CompletedOutputHistory


def history(**kwargs):
    return CompletedOutputHistory(vocab_size=1000, **kwargs)


def test_only_prior_completed_outputs_supply_continuations():
    cache = history()
    assert not cache.add_output("main", [1, 2, 3, 4], completed=False)
    assert not cache.propose("main", [1, 2, 3, 4, 1, 2], max_tokens=7).tokens
    assert cache.add_output("main", [1, 2, 3, 4], completed=True)
    proposal = cache.propose("main", [9, 1, 2], max_tokens=7)
    assert proposal.tokens == (3, 4)
    assert proposal.match_length == 2
    assert cache.telemetry()["retained_requests"] == 1


def test_namespaces_and_engine_instances_do_not_share_history():
    cache = history()
    cache.add_output("one", [1, 2, 3, 4], completed=True)
    assert not cache.propose("two", [1, 2], max_tokens=7).tokens
    assert not history().propose("one", [1, 2], max_tokens=7).tokens
    cache.add_output("two", [1, 2, 8, 9], completed=True)
    assert cache.propose("one", [1, 2], max_tokens=7).tokens == (3, 4)
    assert cache.propose("two", [1, 2], max_tokens=7).tokens == (8, 9)


def test_outputs_are_not_concatenated_across_boundaries():
    cache = history()
    cache.add_output("main", [1, 2], completed=True)
    cache.add_output("main", [3, 4, 5, 6], completed=True)
    assert not cache.propose("main", [1, 2], max_tokens=7).tokens
    assert not cache.propose("main", [2, 3], max_tokens=7).tokens


def test_longest_suffix_precedes_newest_output():
    cache = history()
    cache.add_output("main", [1, 2, 3, 8, 9], completed=True)
    cache.add_output("main", [2, 3, 4, 5], completed=True)
    proposal = cache.propose("main", [1, 2, 3], max_tokens=7)
    assert proposal.tokens == (8, 9)
    assert proposal.match_length == 3


def test_newest_output_precedes_older_matching_output():
    cache = history()
    cache.add_output("main", [1, 2, 3, 4], completed=True)
    cache.add_output("main", [1, 2, 8, 9], completed=True)
    assert cache.propose("main", [1, 2], max_tokens=7).tokens == (8, 9)


def test_most_recent_eligible_occurrence_wins_within_output():
    cache = history()
    cache.add_output("main", [1, 2, 3, 4, 1, 2, 8, 9, 1, 2, 7], completed=True)
    proposal = cache.propose("main", [1, 2], max_tokens=2)
    assert proposal.tokens == (8, 9)
    assert proposal.match_length == 2


def test_miss_requires_two_continuations_and_obeys_proposal_budget():
    cache = history()
    cache.add_output("main", [1, 2, 3, 4, 5, 6], completed=True)
    assert not cache.propose("main", [4, 5], max_tokens=7).tokens
    assert not cache.propose("main", [1, 2], max_tokens=1).tokens
    assert cache.propose("main", [1, 2], max_tokens=2).tokens == (3, 4)
    assert cache.propose("main", [1, 2], max_tokens=10**100).tokens == (3, 4, 5, 6)


def test_search_bounds_clamp_to_small_sequences():
    cache = history(max_tokens=4, max_request_tokens=4)
    cache.add_output("main", [1, 2, 3, 4], completed=True)
    assert cache.propose("main", [1, 2], max_tokens=7, max_match=10**100).tokens == (3, 4)
    assert not cache.propose("main", [1, 2], max_tokens=7, min_match=3).tokens


def test_global_fifo_request_cap_includes_all_namespaces_and_lookups_do_not_touch():
    cache = history(max_requests=2)
    cache.add_output("one", [1, 2, 3, 4], completed=True)
    cache.add_output("two", [5, 6, 7, 8], completed=True)
    assert cache.propose("one", [1, 2], max_tokens=7).tokens
    cache.add_output("three", [9, 10, 11, 12], completed=True)
    assert not cache.propose("one", [1, 2], max_tokens=7).tokens
    assert cache.propose("two", [5, 6], max_tokens=7).tokens == (7, 8)
    assert cache.telemetry()["evicted_outputs"] == 1


def test_global_token_cap_can_evict_multiple_outputs():
    cache = history(max_tokens=8, max_request_tokens=6)
    cache.add_output("one", [1, 2, 3], completed=True)
    cache.add_output("two", [4, 5, 6], completed=True)
    cache.add_output("three", [7, 8, 9, 10, 11, 12], completed=True)
    assert cache.telemetry()["retained_requests"] == 1
    assert cache.telemetry()["retained_tokens"] == 6
    assert cache.telemetry()["evicted_outputs"] == 2
    assert cache.telemetry()["retained_namespace_bytes"] == len("three")


def test_incomplete_and_oversized_insertions_do_not_evict_valid_history():
    cache = history(max_requests=1, max_tokens=4, max_request_tokens=4)
    cache.add_output("main", [1, 2, 3, 4], completed=True)
    assert not cache.add_output("other", [5, 6, 7, 8], completed=False)
    assert not cache.add_output("other", [5, 6, 7, 8, 9], completed=True)
    assert cache.propose("main", [1, 2], max_tokens=7).tokens == (3, 4)
    assert cache.telemetry()["evicted_outputs"] == 0


@pytest.mark.parametrize("source", [[1, 2, 3, 4], (1, 2, 3, 4)])
def test_storage_and_returned_proposal_are_independent_and_immutable(source):
    cache = history()
    cache.add_output("main", source, completed=True)
    assert cache._outputs[0].tokens is not source
    if isinstance(source, list):
        source[:] = [9]
    proposal = cache.propose("main", [1, 2], max_tokens=7)
    assert proposal.tokens == (3, 4)
    with pytest.raises(FrozenInstanceError):
        proposal.tokens = (9, 9)
    with pytest.raises(FrozenInstanceError):
        proposal.match_length = 99
    cache.clear()
    assert proposal.tokens == (3, 4)


@pytest.mark.parametrize("namespace", [None, 2, True, "", "x" * 257,
                                        "é" * 129, "🙂" * 65, "\ud800"])
def test_invalid_namespace_is_rejected_on_insert_and_lookup(namespace):
    cache = history()
    assert not cache.add_output(namespace, [1, 2, 3, 4], completed=True)
    assert not cache.propose(namespace, [1, 2], max_tokens=7).tokens
    assert cache.telemetry()["retained_requests"] == 0


@pytest.mark.parametrize("namespace", ["x" * 256, "é" * 128, "🙂" * 64])
def test_namespace_limit_counts_utf8_bytes(namespace):
    cache = history()
    assert cache.add_output(namespace, [1, 2, 3, 4], completed=True)
    assert cache.propose(namespace, [1, 2], max_tokens=7).tokens == (3, 4)
    assert cache.telemetry()["retained_namespace_bytes"] == 256


@pytest.mark.parametrize("tokens", [None, "1234", {1, 2, 3, 4}, [], (),
                                     [True, 2, 3, 4], [1.0, 2, 3, 4],
                                     [-1, 2, 3, 4], [1000, 2, 3, 4],
                                     [1 << 10000, 2, 3, 4]])
def test_invalid_tokens_fail_closed_without_history_mutation(tokens):
    cache = history()
    assert not cache.add_output("main", tokens, completed=True)
    assert not cache.propose("main", tokens, max_tokens=7).tokens
    assert cache.telemetry()["retained_tokens"] == 0


def test_oversized_sequences_and_builtin_subclasses_are_rejected_before_iteration():
    class HostileList(list):
        def __iter__(self):
            pytest.fail("must not iterate untrusted subclass")

    class HostileString(str):
        def encode(self, *_args):
            pytest.fail("must not encode untrusted subclass")

    cache = history(max_tokens=4, max_request_tokens=4)
    assert not cache.add_output("main", HostileList([1, 2, 3, 4]), completed=True)
    assert not cache.add_output(HostileString("main"), [1, 2, 3, 4], completed=True)
    assert not cache.add_output("main", [object()] * 5, completed=True)
    assert not cache.propose("main", [object()] * 5, max_tokens=7).tokens


@pytest.mark.parametrize("completed", [None, 0, 1, "true"])
def test_completion_must_be_literal_true(completed):
    assert not history().add_output("main", [1, 2, 3, 4], completed=completed)


@pytest.mark.parametrize("changes", [
    {"max_requests": 0}, {"max_requests": True}, {"max_requests": 1.5},
    {"max_tokens": 0}, {"max_tokens": -1}, {"max_tokens": "4096"},
    {"max_request_tokens": 0}, {"max_request_tokens": False},
    {"max_request_tokens": 4097}, {"vocab_size": 0}, {"vocab_size": True},
    {"vocab_size": 1 << 31}, {"vocab_size": 1.0},
])
def test_invalid_constructor_caps(changes):
    with pytest.raises(ValueError):
        CompletedOutputHistory(**{"vocab_size": 1000, **changes})


@pytest.mark.parametrize("changes", [
    {"max_tokens": True}, {"max_tokens": 0}, {"max_tokens": -1},
    {"max_tokens": 2.0}, {"min_match": 0}, {"min_match": True},
    {"max_match": 0}, {"max_match": "6"}, {"min_match": 7},
])
def test_invalid_lookup_settings_return_miss(changes):
    cache = history()
    cache.add_output("main", [1, 2, 3, 4], completed=True)
    proposal = cache.propose("main", [1, 2], **{"max_tokens": 7, **changes})
    assert proposal.tokens == () and proposal.match_length == 0


def test_accounting_includes_namespace_and_integer_storage_with_global_bound():
    cache = CompletedOutputHistory(vocab_size=(1 << 31) - 1,
                                   max_requests=2, max_tokens=8,
                                   max_request_tokens=4)
    empty = cache.telemetry()["accounted_bytes"]
    namespace = "🙂" * 64
    sequence = [(1 << 31) - 2, 2, 3, 4]
    cache.add_output(namespace, sequence, completed=True)
    first = cache.telemetry()
    assert first["accounted_bytes"] - empty >= (
        sys.getsizeof(cache._outputs[0]) + sys.getsizeof(namespace)
        + sys.getsizeof(cache._outputs[0].tokens)
        + sum(sys.getsizeof(token) for token in sequence))
    for _ in range(10):
        cache.add_output(namespace, sequence, completed=True)
        counts = cache.telemetry()
        assert counts["accounted_bytes"] <= counts["accounted_byte_limit"]
        assert counts["retained_tokens"] <= 8
        assert counts["retained_requests"] <= 2
        assert counts["retained_namespace_bytes"] <= 512
    assert counts["operation_scratch_byte_limit"] >= first["accounted_bytes"] - empty
    assert all(type(value) is int and value >= 0 for value in counts.values())
    assert namespace not in repr(counts)
    counts["retained_tokens"] = 999
    assert cache.telemetry()["retained_tokens"] == 8


def test_clear_drops_all_owned_sequences_and_resets_counts():
    cache = history()
    empty = cache.telemetry()
    cache.add_output("private", [1, 2, 3, 4], completed=True)
    cache.propose("private", [1, 2], max_tokens=7)
    cache.clear()
    assert cache.telemetry() == empty
    assert len(cache._outputs) == 0
    assert not cache.propose("private", [1, 2], max_tokens=7).tokens
