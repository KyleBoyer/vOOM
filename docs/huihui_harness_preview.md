# Huihui harness preview

September15: short text/tool requests and exact cached restarts work. **Full
Plex harness readiness is not qualified.** This explicit profile is a measured
small-catalog preview, not a claim that all optimizations or quality gates pass.

Latest correction: the opt-in initial catalog can now call supplied real tools
on explicit-action requests instead of being constrained to private search.
Two synthetic reworded inventory/calendar requests pass at106.14s/112.25s.
The full134-tool capture still fails: larger tiles reduce initial prefill to
234.15s, but its first call takes461.25s, uses the wrong exclusion field, and
fails cumulative swap-out. No final Plex score or complete workflow follows.
Neither initial-inline nor larger-tile experiments are promoted into this preview.

## Connection

- Base URL: `http://127.0.0.1:8077/v1`
- Model: `lossy-Huihui-Qwen3.8-27B-abliterated-mlx-all-mxfp4-mtpquant`
- APIs: `/v1/responses` and `/v1/chat/completions`.
- One concurrent request. Use a small tool catalog with this direct profile;
  it deliberately does not enable the experimental large-catalog gateway.
- If a client requires an API key, use a non-secret placeholder. This is a
  loopback-only, unauthenticated endpoint; do not expose it to the network.
- Set output allowance explicitly, e.g.1024. Prefill tile32/head rows8192 are
  internal dimensions, not output-token limits. Allow a long client timeout;
  many completed calls still exceed90 seconds.

## Start after host admission

Keep ChatGPT/Codex, Plex and other user apps open. Ensure no other model job or
disk-heavy benchmark is running and port8077 is unused. Never kill an unknown
process to free the port. From a terminal without pre-existing `VMODEL_*` overrides:

```bash
cd "/Volumes/Workspace NVME/git/vOOM"
.venv/bin/python -m runtime.memory_preflight \
  --result logs/huihui-preview-preflight.json \
  --sample-seconds 30 --sample-memory-window \
  --min-root-free-gb 10 --min-stable-available-gb 5.5 \
  --require-no-transcoders && \
caffeinate -is .venv/bin/python -m runtime.server \
  --profile huihui-qwen38-27b-harness-preview --port 8077
```

These are the user-approved5.5GB launch /4.5GB runtime floors. Failed admission
must leave the server stopped; no automatic threshold reduction. Keep the
terminal open; Ctrl-C stops it. `/v1/models` confirms only the registry, not
inference. No scheduler/heartbeat is needed or installed.

## Evidence and limitations

The same full-state, paged, tile32, split-MLP, CPU-grammar configuration completed
the real short weather/title captures in54.75s/31.95s, then repeated weather in
39.74s with111 cached tokens and the same selected-target output. New held-out
inventory/calendar requests (three tools, developer messages) completed in
100.86s/106.78s with exact tool names/arguments; fresh-server cached repeats took
73.27s/93.50s with identical output tokens and passing pressure gates. These
are uncached-prompt versus prefix-cached measurements, not cold-storage tests.
The fixtures override model/max1024/temp0/seed64013. Runtime changes invalidate
durable caches rather than trusting stale state.

A separate streaming JSON request naturally emitted569 tokens with all24
arithmetic results correct, but took887.48s and failed cumulative swap-out.
Do not treat that as qualified sustained-output performance.

The full134-tool HTTP capture uses a different gateway profile, which compacts
the model-side catalog/prompt. With explicit `gateway-buffered-decision` plus
`gateway-execution-auto`, original temperature1 and finite synthetic Plex results,
the latest workflow completed naturally in2287.04s (38.12min),438 final tokens.
Its matching list has exactly the four eligible titles and correctly explains
all six exclusions. The unchanged legacy rubric is75/100/FAIL (it counts excluded
titles in explanations and requires multiple pages); single-page exhaustion is
not pagination proof. Pressure also FAILS:120.95MB sampled swap-out. This was
not live Plex, not an unmodified tool-result replay, and not full49K all-schema
model execution. These gateway policies remain default-off experiments and
are not included in this direct preview. Full readiness remains unqualified.

Selected MXFP4 weights, reassociated prefill and Hermes tool convention are
approximations relative to released BF16. Exact cache/token and native-head
checks are relative to the selected representation, not released-BF16 lossless.
Vision, long contexts, full-workflow quality and sub90-second end-to-end latency
remain unqualified. No server-running claim is implied by this document.

## Latest exact memory experiment (September15)

The explicit `qwen35-factor-base-disk-audit` overlay stores immutable speculative
rollback snapshots on Workspace NVMe, checksum-verifies reloads, and deletes its
owned snapshots at generation exit. It removes approximately154MB of retained
rollback arrays during verification; logical bytes are never admission credit.
Full-geometry prefix0..5 raw-byte gates pass. It stays opt-in and is not included
in the preview default: full-workflow pressure/quality have not passed.

Measured inventory/calendar calls (synthetic three-tool held-out requests,
max1024/temp0/seed64013, full prompt state) with this overlay:

| Proposal sidecar | Uncached prompt | Cached prompt, fresh server |
|---|---|---|
| Compact MXFP4 |100.10s /106.03s |72.36s /91.65s |
| BF16 |107.21s /98.72s |87.29s /86.51s |

Both cached runs preserve exact greedy tokens and pass whole-run pressure.
Compact uncached narrowly fails the16MB cumulative swap-out gate. BF16 uncached
passes pressure and exact tokens, but the strict identical-request check fails
because the intentionally different model alias selects that sidecar. All target
tensor file objects are shared; this does not make the target lossless versus
the BF16 release. These are not cold-storage or full134-tool workflow timings.

The BF16 sidecar uses alias
`lossy-Huihui-Qwen3.8-27B-abliterated-mlx-all-mxfp4`; the compact sidecar uses
the existing `-mtpquant` alias. Neither is a universal winner. Keep the native
memory limits and ordinary admission margin; no uncached-direct-I/O or grammar
jump-forward promotion follows from these measurements.

A separate answer-only Plex diagnostic still omitted one eligible title on the
compact target. The mixed attention8bit/last4BF16 target recovered all four but
failed requested JSON formatting and pressure, taking314.05s. No output repair,
grader weakening, new full-workflow score, or harness-ready claim was applied.
