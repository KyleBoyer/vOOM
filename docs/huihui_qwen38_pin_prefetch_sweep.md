# Huihui Qwen3.8 27B pin/prefetch cold sweep

## Scope

This is a 40-cell cold A/B for the all-MXFP4 fast artifact only. It varies:

- `VMODEL_QWEN35_PIN_LM_HEAD`: `0`, `1`
- `VMODEL_QWEN35_PIN_TRUNK_BUDGET_MB`: `0`, `500`, `1000`, `2000`
- `VMODEL_QWEN35_PREFETCH_DEPTH`: `0`, `1`, `2`, `3`, `4`

Keep `VMODEL_QWEN35_WEIGHT_CACHE_MB=2200` fixed so the comparison measures
value density rather than a changing Metal budget. The trunk value is a
requested pin cap, not guaranteed residency: the runtime subtracts the actual
persistent head/embedding pins and reserves one demand layer plus the selected
prefetch depth. Record `planned_trunk_pin_bytes` and
`weight_cache_pinned_bytes`, not merely the requested value.

Do not run the head-on arm against the released BF16 route with a 2.2 GB
cache. Its exact 2.543 GB head alone cannot fit; streaming that head remains
the lossless route. The compact all-MXFP4 head is the intended subject here.

## Matrix and cold-run procedure

Run all Cartesian-product cells below, preferably in a deterministic shuffled
order, then repeat the best three cells and the `0/0/0` baseline twice. This
limits ordering bias from filesystem and macOS cache state while keeping the
full first pass manageable.

```text
head = 0, 1
trunk_mb = 0, 500, 1000, 2000
prefetch_depth = 0, 1, 2, 3, 4
```

For each cell:

1. Stop the previous server and wait until its engine has closed. Do not run
   another model or Metal benchmark concurrently.
2. Run the mandatory 30-second preflight immediately before startup:

   ```bash
   .venv/bin/python -m runtime.memory_preflight \
     --result logs/huihui-pin-H${HEAD}-T${TRUNK}-P${PREFETCH}.preflight.json \
     --sample-seconds 30
   ```

3. Start the unchanged fast profile with only the three experiment knobs
   overridden:

   ```bash
   VMODEL_QWEN35_WEIGHT_CACHE_MB=2200 \
   VMODEL_QWEN35_PIN_LM_HEAD=${HEAD} \
   VMODEL_QWEN35_PIN_TRUNK_BUDGET_MB=${TRUNK} \
   VMODEL_QWEN35_PREFETCH_DEPTH=${PREFETCH} \
   .venv/bin/python -u -m runtime.server \
     --profile huihui-qwen38-27b-fast-agent --port 8077
   ```

4. Replay the byte-identical registered 134-tool capture once with temperature
   zero, streaming, and the same explicit 16-output-token benchmark cap used
   by the current fast baseline. Do not substitute or trim its tool schema,
   messages, or system prompt. Persist the replay telemetry under a filename
   containing `H`, `T`, and `P`. The output cap is a benchmark modification
   and must remain stated beside every timing.
5. Stop the server before the next preflight/cell. A process restart is part
   of the cold definition; a second request to the same engine is a separate
   warm-cache experiment and must not be mixed into this table.

## Evidence to retain

For every cell record:

- prefill/first-token seconds, decode seconds/token, and wall seconds;
- `weight_store_bytes_read`, `weight_cache_hits`,
  `weight_cache_pinned_hits`, `weight_cache_prefetch_hits`,
  `weight_prefetch_waits`, and `weight_prefetch_wait_s`;
- `weight_cache_resident_bytes`, `weight_cache_pinned_bytes`,
  `weight_cache_prefetched_bytes`, `planned_trunk_pin_layers`, and
  `planned_trunk_pin_bytes`;
- prefetcher scheduled and budget-skipped counts from the server summary;
- true peak Metal, minimum available memory, swap-used growth, swap-out growth,
  response token hash, and the existing task-quality score.

Derived columns should include pinned-hit bytes per pinned GB, prefetch-hit
rate, milliseconds waited per prefetch hit, GB read per output token, and wall
seconds saved versus `H0/T0/P0`.

## Acceptance and stop rules

- Reject any cell whose greedy token IDs or response hash differ from the
  `H0/T0/P0` run. Pinning and prefetching are scheduling changes only and have
  no allowance to change arithmetic.
- Reject any cell that fails the existing fast-task quality gate. In
  particular, an unchecked lossy model answer below 100/100 is not rescued by
  a latency win; retain the already accepted deterministic policy-adapter
  boundary for filtered/paginated answers.
- Reject true peak Metal above 8.5 GB, sustained macOS compression/swap growth,
  post-response available memory below the profile's 6 GB floor, or any
  `MemoryError`/capacity refusal.
- Stop increasing trunk pin when actual planned/pinned bytes plateau. The
  planner is protecting head, transient, demand, and prefetch capacity; a
  larger requested number beyond that point is not a distinct layout.
- Stop increasing prefetch depth when budget skips rise without more prefetch
  hits, or when added prefetch wait time and resident bytes do not improve
  wall time. Prefer the smallest depth within measurement noise of the best.
- Promote no automatic default from this one request. The winning cell remains
  explicit opt-in until it passes the required heterogeneous real-request
  corpus and a fresh cold replay of the unmodified capture.
