# Huihui harness preview

September15: short text/tool requests and exact cached restarts work. **Full
Plex harness readiness is not qualified.** This explicit profile is a measured
small-catalog preview, not a claim that all optimizations or quality gates pass.

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
the model-side catalog/prompt. Its finite synthetic Plex-response run completed
in1043.37s but returned a fixed gateway abstention instead of the requested list:
legacy rubric63/100, quality FAIL, cumulative swap-out61.62MB, pressure FAIL.
This was not live Plex, not an unmodified tool-result replay, and not full49K
all-schema model execution. The host-abstention authorship audit is now stricter.
The new `gateway-execution-auto` policy is a separate default-off experiment;
it is not included in this direct preview.

Selected MXFP4 weights, reassociated prefill and Hermes tool convention are
approximations relative to released BF16. Exact cache/token and native-head
checks are relative to the selected representation, not released-BF16 lossless.
Vision, long contexts, full-workflow quality and sub90-second end-to-end latency
remain unqualified. No server-running claim is implied by this document.
