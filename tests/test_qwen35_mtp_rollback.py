"""F94: proves KDAStateCache.fork()/reassign is an exact rollback primitive
for a rejected speculative round -- the core correctness claim
QwenMTPSpeculativeEngine's reject path depends on.

Reuses the same oracle-grade _gated_delta_net function
tests/test_qwen35_oracle.py already verifies against real (unmodified)
transformers Qwen3_5MoeGatedDeltaNet code, so this test is checking the
ROLLBACK MECHANISM specifically (fork -> mutate -> discard -> restore ->
re-feed), not re-deriving DeltaNet's own math.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import torch

from runtime.kda_state import KDAStateCache
from runtime.qwen35 import _gated_delta_net

from tests.test_qwen35_oracle import (
    HIDDEN,
    LENGTH,
    _hf_config,
    _mx_state,
    _randomize,
    _runtime_config,
)

from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeGatedDeltaNet,
)


def _weights_and_hidden(seed_weights: int, seed_hidden: int):
    config = _hf_config()
    real = Qwen3_5MoeGatedDeltaNet(config, layer_idx=0)
    _randomize(real, seed_weights)
    with torch.no_grad():
        real.A_log.copy_(torch.log(
            torch.empty_like(real.A_log).uniform_(1.0, 8.0)))
    torch.manual_seed(seed_hidden)
    hidden = torch.randn(1, LENGTH, HIDDEN)
    prefix = "model.layers.0"
    weights = _mx_state(real, f"{prefix}.linear_attn")
    return weights, hidden, prefix


def test_fork_restore_matches_never_having_taken_the_rejected_branch():
    """Simulates exactly QwenMTPSpeculativeEngine's reject path: fork() at a
    round boundary, speculatively advance (the draft), discard by
    reassigning the fork, then re-advance through only the real token.
    Must be bit-exact with a one-shot run that never saw the rejected
    branch at all."""
    weights, hidden, prefix = _weights_and_hidden(41, 42)
    cfg = _runtime_config()

    # Ground truth: process positions [0:5) in one uninterrupted call --
    # this is what SHOULD happen after a round ending in a reject at
    # position 4 (real token) with a discarded draft at position 4-would-be-5.
    ground_truth_cache = KDAStateCache(2)
    ground_truth = _gated_delta_net(
        mx.array(hidden[:, :5].numpy()), weights, prefix, cfg,
        ground_truth_cache, 0)
    mx.eval(ground_truth)

    # F94 scheme: commit [0:4), checkpoint, speculatively advance through
    # position 4 (the "draft"), then DISCARD by restoring the checkpoint and
    # re-feeding only the real position-4 token.
    live_cache = KDAStateCache(2)
    committed = _gated_delta_net(
        mx.array(hidden[:, :4].numpy()), weights, prefix, cfg, live_cache, 0)
    mx.eval(committed)
    checkpoint = live_cache.fork()

    # Speculative (draft) advance through a token that will be REJECTED --
    # deliberately different content (a different position's hidden vector)
    # to prove the checkpoint restore actually discards this, not just
    # coincidentally matches.
    rejected_draft = _gated_delta_net(
        mx.array(hidden[:, 5:6].numpy()), weights, prefix, cfg, live_cache, 0)
    mx.eval(rejected_draft)
    assert live_cache.state(0) is not None
    polluted_state = np.array(live_cache.state(0))

    # Reject: restore the checkpoint (discarding the speculative advance
    # entirely), then re-feed the real position-4 token.
    live_cache = checkpoint
    restored = _gated_delta_net(
        mx.array(hidden[:, 4:5].numpy()), weights, prefix, cfg, live_cache, 0)
    mx.eval(restored)

    committed_and_restored = mx.concatenate([committed, restored], axis=1)
    mx.eval(committed_and_restored)
    # Tolerance, not bit-exact equality: splitting one call into two (even
    # with NO discard/restore in between -- confirmed by direct A/B) moves
    # float32 reduction order enough to differ at ~1.5e-7, ordinary
    # non-associativity, the same reason every OTHER oracle comparison in
    # this suite (test_qwen35_oracle.py's _assert_close) uses a tolerance
    # rather than np.array_equal. What matters is that the restored branch
    # is close to ground truth and DEMONSTRABLY NOT equal to the polluted
    # (rejected-branch) state checked below.
    assert np.allclose(
        np.array(committed_and_restored), np.array(ground_truth),
        atol=1e-5, rtol=1e-5), (
        "fork()-restore-and-refeed must match never having taken the "
        "rejected branch, within ordinary float32 split-vs-whole tolerance")
    # The polluted (rejected-branch) state must actually differ from the
    # restored one -- otherwise this test would pass vacuously regardless
    # of whether restore worked.
    assert not np.array_equal(polluted_state, np.array(live_cache.state(0)))


def test_fork_does_not_alias_mutable_state_across_branches():
    """fork() must return an INDEPENDENT snapshot: advancing the ORIGINAL
    cache after forking must not retroactively change what the fork
    represents (kda_state.py's own docstring claims this via MLX's
    functional/replace-not-mutate array semantics)."""
    weights, hidden, prefix = _weights_and_hidden(51, 52)
    cfg = _runtime_config()

    cache = KDAStateCache(2)
    _gated_delta_net(
        mx.array(hidden[:, :3].numpy()), weights, prefix, cfg, cache, 0)
    snapshot = cache.fork()
    snapshot_state_before = np.array(snapshot.state(0))

    # Advance the ORIGINAL cache further.
    _gated_delta_net(
        mx.array(hidden[:, 3:6].numpy()), weights, prefix, cfg, cache, 0)

    snapshot_state_after = np.array(snapshot.state(0))
    assert np.array_equal(snapshot_state_before, snapshot_state_after), (
        "fork() snapshot must be unaffected by the original cache advancing")
