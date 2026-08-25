# Qwen3.8 exact-target proposal-tree verification

## Outcome

vOOM can now verify a small DFlash proposal tree while streaming each 27B
target layer once.  The implementation is correct and memory-bounded, but the
measured Q2 tree arms do not beat the existing plain or Q4-unary paths, so the
feature remains explicit and default-off.

This is exact only with respect to the target being served.  The current fast
target is the lossy MXFP4 Huihui artifact; a quantized draft cannot further
change its authoritative tokens or persistent endpoint, but it does not turn
that target into released BF16.

## Design

`runtime/speculative_tree.py` builds a best-first tree under a fixed non-root
node budget.  The scheduling structure is adapted from
[DDTree-MLX](https://github.com/humanrouter/ddtree-mlx) commit
`4b12590abc9909fb03bfdf7dd736e76cef7ebdb0` (MIT).  DFlash supplies sorted
top-16 LM-head candidates at each of its four proposal depths.  Draft scores
choose which nodes are worth verifying; they never authorize output.

`runtime/qwen35_tree_verify.py` evaluates the tree layer-major:

- Dense target weights are fetched once per layer, independent of node count.
- Full-attention nodes read the immutable prompt KV plus only their ancestor
  nodes.  Siblings cannot attend to each other.
- DeltaNet nodes run the existing one-position recurrence from their exact
  parent state and convolution history.
- The verifier retains compact decay/key/value/beta/conv factors.  After the
  target selects a path, it replays only those factors over the prompt endpoint
  and appends only that path's attention KV.
- Dead sibling KV, hidden rows, logits, and factors are released before the
  next sidecar load.

The compact-state approach follows the direction of
[SpecLA](https://arxiv.org/abs/2607.16673),
[Bole](https://arxiv.org/abs/2608.01651), and
[TreeWY](https://arxiv.org/abs/2608.20961).  vOOM does not use a reassociated
WY recurrence for authoritative target math; it keeps the existing serial
operator order and uses factors only to reconstruct the committed endpoint.

## Proof gates

The test suite checks:

1. best-first tree topology, bounds, and greedy target walk;
2. every full-attention node against an independent sequential ancestor path;
3. every DeltaNet selected path against exact scalar recurrence and conv state;
4. a complete tiny two-layer dense Qwen target, with one weight fetch per layer;
5. decoder terminal accounting: the final emitted token is never prematurely
   fed into KV/DeltaNet state; and
6. explicit refusal for grammar-constrained or stochastic tree requests.

Focused command (run only after the required memory preflight):

```bash
.venv/bin/python -m pytest -q \
  tests/test_speculative_tree.py \
  tests/test_qwen35_oracle.py::test_qwen_scalar_delta_factors_commit_tree_path_array_equal \
  tests/test_qwen35_oracle.py::test_qwen_tree_attention_nodes_and_commit_match_sequential_paths \
  tests/test_qwen35_multi_request.py::test_tree_verifier_matches_every_branch_and_commits_selected_path_exactly \
  tests/test_qwen38_dspark_sidecar.py \
  tests/test_qwen38_dflash2_adapter.py
```

Result: `83 passed`.

## Real timing

These are fresh-process runs of the fixture's synthetic six-bullet prompt with
`max_tokens=8`.  They are scheduling/correctness gates, not the unmodified
134-tool capture and not complete-answer quality evaluations.

| arm | target sweeps | accepted | factors peak | wall | peak Metal | swap-out growth |
|---|---:|---:|---:|---:|---:|---:|
| Q2 tree budget 4 | 5 | 2 | 41.073 MB | 57.8948 s | 1.464 GB | 3.064 MB |
| Q2 tree budget 8 | 4 | 3 | 73.931 MB | 49.6918 s | 1.464 GB | 4.882 MB |
| historical plain control | 7 ordinary decode steps | n/a | n/a | 47.5345 s | 2.250 GB | 2.376 MB |
| historical Q4 unary | 4 | 3 | n/a | 41.6038 s | recorded separately | within gate |

Both tree arms emitted `[271, 12, 2972, 48401, 12, 3282, 12, 12089]` and
matched the historical plain endpoint hashes for full-attention KV, recurrent
state, and convolution history.  The auxiliary final hidden projection differs
for the same known reason as serial multi-position verification; it is not
persistent target state and did not change tokens.

Tree-8 proves the intended I/O effect: doubling verified nodes from four to
eight did not double target weight reads, and it removed one target sweep.  It
still did not clear the project's 10% promotion rule.  The current decision is
therefore **STOP for automatic serving**, retain as a research primitive for a
higher-recall proposal source or future parent-conditioned tree scheduler.

## Explicit operation

The server setting is:

```bash
VMODEL_QWEN_DFLASH2_TREE_BUDGET=8
```

Allowed values are `0..8`; `0` is the unchanged default.  A real fixture run:

```bash
.venv/bin/python tests/fixtures/qwen38_dflash2_gate.py \
  --mode spec \
  --draft models/Qwen3.8-27B-DFlash2-mlx-affine2-g64 \
  --max-tokens 8 --cap 4 --tree-budget 8 --load-margin-mb 0 \
  --result logs/qwen38_dflash2_q2_tree8_spec8_20260825.json
```

The zero secondary sidecar-load margin is part of that measured command.  The
memory governor's independent critical reserve, fresh preflight, ≤8.5GB Metal
limit, and swap-growth gate remained active.
