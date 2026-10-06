# Huihui harness preview

## Low-memory serving (October6 candidate)

The explicit `huihui-qwen38-27b-low-memory-online` profile is being qualified
for hosts below the former5.5GB startup buffer. It removes the proposal sidecar
and speculative rollback, and reduces prefill tiles32 ->8. Target weights and
full prepared context are unchanged; different prefill shapes still need quality
checks. Plain decoding may be slower. The live4.5GB reserve,400MB ordinary margin,
8.5GB Metal ceiling,256MB weight-cache cap and64MB exact-KV cap remain.
One-token decoding also loads attention, gate/up and down-projection weights
separately; it no longer needs the whole210MB layer page at once. This path is
explicitly selected by the low-memory profile, not enabled for other profiles.

From a clean shell without `VMODEL_*` overrides:

```bash
cd "/Volumes/Workspace NVME/git/vOOM"
caffeinate -is .venv/bin/python -m runtime.huihui_serve --port 8077
```

This launcher samples the host for30seconds and requires5.0GB available,
10GB root free, stable swap and no known transcoder. It refuses a used port
without stopping its owner. It then starts the lazy-loading loopback service;
individual allocations still fail closed if they cannot fit safely. A lower
startup buffer is not a guarantee that arbitrary large requests will fit.
Measured qualification and endpoint status are recorded in STATUS.md; the
older timings below do not qualify this new composition.

October5: use the explicit `huihui-qwen38-27b-serial-harness-candidate` below
for an experimental harness trial. It serializes tool calls (one per turn).
Full general-purpose readiness and sub90-second workflow latency are NOT met.

The captured134-tool/temp1 Plex request (declared model/max1024/seed overrides,
gateway-compacted model catalog) completed against actual provider code over
synthetic data in **1790.73s (29.8min)**:632.36s tool call +1157.80s final answer.
All four eligible titles, exact filters, protocol and provider contract PASS;
unchanged legacy score **90/100/FAIL** because it requires another page even
though this provider result is exhausted. This is not live Plex. Native memory
passes, minimum4.655GB available and zero actual swap growth. Outer runner
completion record is missing: child/response/log hashes verify, but full
wrapper source/exit provenance is incomplete. See STATUS.md for exact artifacts.

Explicit schema-constrained terminal requests in two domains pass with complete
wrapper verification: asset IDs **73.64s**, support tickets **69.50s**, exact JSON,
minimum4.893GB available and zero swap growth. The requests add JSON schema and
disable tools; schemas contain no answer constants. Plain-language JSON-only
formatting still failed the earlier filtered continuation. No output repair.
Workspace112.60s and literal-marker70.09s are earlier narrower-profile checks.

The16384-token HTTP cap includes prepared prompt+requested output; it is an
admission bound, not16K context-quality qualification. The selected MXFP4 target
is lossy versus released BF16. Vision and long contexts remain unqualified.

## Historical preview evidence (different compositions)

September16: short text/tool requests, sustained JSON and exact cached restarts work. **Full
Plex harness readiness is not qualified.** This explicit profile is a measured
small-catalog preview, not a claim that all optimizations or quality gates pass.
These paragraphs describe the older `huihui-qwen38-27b-harness-preview` profile;
the Connection and Start sections select the October candidate instead.

September16 correction: an opt-in arbitrary-order argument grammar fixes the root-path
versus library-label error. The original134-tool/temp1 request with synthetic raw
Plex pages completes in2004.74s, with correct offsets0->3 and all four eligible
matches, but legacy85/100/FAIL (excluded titles appear in explanations; unrated
wording is ambiguous). Native memory passes: minimum4.94GB and zero actual swap
growth. It is not live Plex or a latency win. A fresh569-token JSON test passes
all24 arithmetic results in913.01s with exact prior prompt/output-token/text
hashes, minimum5.37GB and zero actual swap growth. Those experiments were not
promoted into the older preview. A stray protocol marker found in a small-catalog
workspace replay motivated the September30 parser/grammar fixes described above.

Historical swap-out numbers below used psutil's Darwin Pageouts proxy, not actual
Swapouts. Keep their original verdicts; only fresh native-counter runs above
establish actual swap behavior. See STATUS.md for immutable run identities.

## Connection

- Base URL: `http://127.0.0.1:8077/v1`
- Model: `lossy-Huihui-Qwen3.8-27B-abliterated-mlx-all-mxfp4-mtpquant`
- APIs: `/v1/responses` and `/v1/chat/completions`.
- One concurrent request. The candidate uses the explicit compact gateway for
  large catalogs; this changes the model-side representation of the tools.
- Set `parallel_tool_calls:false`. Preserve the full assistant output and append
  matching tool results for each successive turn; no parallel batch is emitted.
- If a client requires an API key, use a non-secret placeholder. This is a
  loopback-only, unauthenticated endpoint; do not expose it to the network.
- Set output allowance explicitly, e.g.1024. Prefill tile32/head rows8192 are
  internal dimensions, not output-token limits. Allow a long client timeout;
  many completed calls still exceed90 seconds. Allow at least1800 seconds per
  request for the measured Plex shape; total multi-turn wall can be longer.

## Start after host admission

Keep ChatGPT/Codex, Plex and other user apps open. Ensure no other model job or
disk-heavy benchmark is running and port8077 is unused. Never kill an unknown
process to free the port. From a terminal without pre-existing `VMODEL_*` overrides:

```bash
cd "/Volumes/Workspace NVME/git/vOOM"
.venv/bin/python -m runtime.memory_preflight \
  --result logs/huihui-candidate-preflight.json \
  --sample-seconds 30 --sample-memory-window \
  --min-root-free-gb 10 --min-stable-available-gb 5.5 \
  --require-no-transcoders && \
caffeinate -is .venv/bin/python -m runtime.server \
  --profile huihui-qwen38-27b-serial-harness-candidate --port 8077
```

These are the user-approved5.5GB launch /4.5GB runtime floors. Failed admission
must leave the server stopped; no automatic threshold reduction. Keep the
terminal open; Ctrl-C stops it. `/v1/models` confirms only the registry, not
inference. No scheduler/heartbeat is needed or installed.

## Explicit JSON terminal requests

When the tool work is finished, a `/v1/responses` request may select
`tool_choice:"none"` and a schema, for example:

```json
{
  "tool_choice": "none",
  "text": {"format": {
    "type": "json_schema", "name": "asset_ids", "strict": true,
    "schema": {
      "type": "object",
      "properties": {"ids": {"type": "array", "items": {"type": "string"}}},
      "required": ["ids"], "additionalProperties": false
    }
  }}
}
```

These fields supplement the ordinary model/input/output allowance. Keep the
actual conversation and tool results as input. Enabled tools plus structured
text are rejected; use this on the final synthesis turn. Schema conformance
does not guarantee factual correctness, and this variant is not an unmodified
capture or proof that plain-language formatting instructions always work.

## Historical evidence and limitations

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
that earlier workflow completed naturally in2287.04s (38.12min),438 final tokens.
Its matching list has exactly the four eligible titles and correctly explains
all six exclusions. The unchanged legacy rubric is75/100/FAIL (it counts excluded
titles in explanations and requires multiple pages); single-page exhaustion is
not pagination proof. Pressure also FAILS:120.95MB sampled swap-out. This was
not live Plex, not an unmodified tool-result replay, and not full49K all-schema
model execution. These gateway policies remain default-off experiments and
were not included in the older direct preview. They are explicit components of
the October candidate, not general defaults. Full readiness remains unqualified.

Selected MXFP4 weights, reassociated prefill and Hermes tool convention are
approximations relative to released BF16. Exact cache/token and native-head
checks are relative to the selected representation, not released-BF16 lossless.
Vision, long contexts, full-workflow quality and sub90-second end-to-end latency
remain unqualified. No server-running claim is implied by this document.

## Historical exact memory experiment (September15)

The explicit `qwen35-factor-base-disk-audit` overlay stores immutable speculative
rollback snapshots on Workspace NVMe, checksum-verifies reloads, and deletes its
owned snapshots at generation exit. It removes approximately154MB of retained
rollback arrays during verification; logical bytes are never admission credit.
Full-geometry prefix0..5 raw-byte gates pass. It remains opt-in and was not included
in the older preview default; it is included in the October candidate.

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
