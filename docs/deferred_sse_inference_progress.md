# Deferred: inference progress SSE and Kai integration

Status: **pinned for later; do not implement yet**

## Intent

Make live inference progress a normal part of vModel's streaming responses, then
let Kai detect and display it without coupling Kai's UI directly to vModel
event names. Prefer always emitting progress events from local inference
services rather than requiring a per-request opt-in.

The eventual protocol should match or extend useful local-runtime precedents,
especially llama.cpp's streamed `prompt_progress` values and Ollama's prompt
evaluation terminology, while retaining OpenAI Responses compatibility.

## Current vModel behavior

- `/v1/responses` always emits the standard `response.created` and
  `response.in_progress` lifecycle events.
- During prefill and vision work, the default stream emits SSE comments such as
  `: prefill 4096/20000`. These keep the connection alive but conforming SSE
  parsers intentionally do not surface them as application events.
- A request with `vmodel_progress_events: true` receives typed
  `response.vmodel.prefill_progress` and
  `response.vmodel.vision_progress` events instead.
- Typed progress currently reports `phase`, `completed`, `total`, `fraction`,
  and `cache_source`.
- Chat Completions currently receives comments only.
- Decode progress is represented by ordinary output deltas; there is no useful
  final-token percentage because the actual output length is unknown.

## External precedents

- SSE comments are the standard heartbeat mechanism.
- OpenAI Responses defines `response.in_progress`, but not numeric prompt
  evaluation progress.
- llama.cpp's non-OpenAI `/completion` endpoint supports
  `return_progress: true` and streams `prompt_progress` with `total`, `cache`,
  `processed`, and `time_ms`.
- Ollama reports `prompt_eval_count`, `prompt_eval_duration`, `eval_count`, and
  `eval_duration`, primarily as terminal performance data rather than live
  prompt-evaluation percentages.
- Anthropic streams typed lifecycle events and pings and requires clients to
  tolerate unknown event types, but does not define numeric prefill progress.

Conclusion: the transport and extension pattern are conventional; the exact
progress schema is still a vendor extension.

## Proposed vModel progress protocol v1

Before declaring the event contract stable, extend the existing payload with
the fields needed for multi-phase agent requests:

```json
{
  "type": "response.vmodel.prefill_progress",
  "protocol_version": 1,
  "phase": "prefill",
  "stage": "gateway_decision",
  "completed": 4096,
  "total": 20000,
  "unit": "tokens",
  "cached": 2048,
  "fraction": 0.2048,
  "elapsed_ms": 1800,
  "cache_source": "disk"
}
```

Proposed stages include `request`, `gateway_decision`, `gateway_execution`, and
`vision`. A stage identifier is required because hidden decision and execution
prefills can otherwise reset one progress bar without explanation.

Open questions to settle before implementation:

1. Always emit typed events, or negotiate them through a capability while
   retaining comment-only compatibility for strict clients?
2. Emit both the current phase-specific event types and a normalized
   `response.vmodel.progress`, or keep only the phase-specific types?
3. Whether `cached` means the reusable prefix for this stage or all cache-backed
   tokens in the request.
4. Whether an ETA is stable enough to expose. Prefer elapsed time and measured
   throughput initially; an inaccurate ETA is worse than none.

## Capability discovery

Expose an explicit machine-readable capability through `/v1/models`, for
example:

```json
{
  "capabilities": {
    "inference_progress": "vmodel-v1"
  }
}
```

The existing `owned_by: "vmodel"` is sufficient for a temporary prototype but
should not be the permanent protocol negotiation mechanism.

## Kai integration boundary

Kai should define a provider-neutral internal event, for example
`inference-progress`, and translate provider-specific streams at the model
adapter boundary:

```text
vModel progress  ----\
llama.cpp progress ---+--> Kai inference-progress --> generic progress UI
other providers  -----/       (or indeterminate fallback)
```

For the current Mastra/AI SDK stack:

1. Detect the advertised progress capability.
2. Enable raw stream chunks for capable providers.
3. Parse the namespaced vModel progress event in the OpenAI-compatible adapter.
4. Translate it into Kai's generic `inference-progress` envelope.
5. Render labels such as `Preparing context` or `Processing image`, not the
   ambiguous `Evaluating model`.
6. Fall back to an indeterminate activity indicator when numeric progress is
   unavailable.

No component outside the provider adapter should depend on
`response.vmodel.*` event names.

## Proof gates

- Existing OpenAI SDK, Mastra, and direct SSE clients still complete streams.
- Unknown-event-tolerant clients can ignore progress without changing output.
- Kai displays progress for cold, memory-cache, and disk-cache prefills.
- Hidden gateway decision and execution stages never appear as one regressing
  progress bar.
- Disconnect/retry behavior still retains only completed exact KV chunks.
- Progress emission does not measurably reduce prefill or decode throughput.
- Golden streamed text/tool-call output is byte-identical with progress enabled
  and disabled.

## Deferred work checklist

- [ ] Finalize protocol v1 fields and compatibility behavior.
- [ ] Add `/v1/models` capability advertisement.
- [ ] Emit typed progress consistently across Responses and Chat Completions.
- [ ] Add stage-aware engine callbacks for hidden gateway phases.
- [ ] Add vModel protocol and SDK compatibility tests.
- [ ] Add Kai's provider-neutral `inference-progress` event.
- [ ] Add vModel and llama.cpp adapters.
- [ ] Add Kai UI and settings behavior.
- [ ] Run transcript, tool-call, reconnect, and performance gates.
