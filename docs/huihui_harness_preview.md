# Huihui harness preview

Status, September 14: **not yet qualified for reliable harness use**. This is
one explicit launch profile for the last measured low-workspace configuration,
not another optimization. The server is not currently running.

## Connection settings

- Base URL: `http://127.0.0.1:8077/v1`
- Model: `lossy-Huihui-Qwen3.8-27B-abliterated-mlx-all-mxfp4-mtpquant`
- APIs: `/v1/responses` and `/v1/chat/completions`.
- Start with one concurrent text request and a small tool catalog. If the client
  insists on an API key, use a non-secret placeholder; this local endpoint has
  no authentication. Do not expose it to the network.
- Set the output allowance explicitly, for example 1024 tokens. The profile's
  prefill tile8 and head rows8192 do **not** cap the generated answer length.
  A larger allowance does not mean sustained long output has been qualified.

## Launch after host admission

Leave ChatGPT and Plex open. First ensure no other model/benchmark or disk-heavy
job is running and port8077 is unused; never kill an unknown process to free it.
Run from the repository in a terminal with no pre-existing `VMODEL_*` overrides:

```bash
cd "/Volumes/Workspace NVME/git/vOOM"
.venv/bin/python -m runtime.memory_preflight \
  --result logs/huihui-preview-preflight.json \
  --sample-seconds 30 --sample-memory-window \
  --min-root-free-gb 10 --min-stable-available-gb 6.70 \
  --require-no-transcoders && \
caffeinate -is .venv/bin/python -m runtime.server \
  --profile huihui-qwen38-27b-harness-preview --port 8077
```

The conservative6.70GB launch check prevents an unchanged-headroom rerun; it
does not replace the runtime governor or guarantee successful requests. A
failed preflight must leave the server stopped. Do not lower admission limits
to force startup. Keep this terminal open; Ctrl-C stops this server. Check
`http://127.0.0.1:8077/v1/models` before connecting your harness. A successful
model listing verifies the HTTP registry only, not inference readiness.

## What was actually verified

The September9 real short weather capture completed in78.8566s:111 prepared
input tokens,23 naturally completed output tokens under max1024, reference-
identical token/text/protocol checks and0.583GB true peak Metal. The fixture
overrode model, temperature0, seed64013 and output allowance1024 while preserving
the captured messages/tools/streaming setting. This was an uncached prompt,
not a proven cold-storage run or the full captured Plex harness.

The next title request failed memory admission. No completed full134-tool
harness/Plex score, broad held-out-domain pass, vision or long-output pass is
claimed. The MXFP4 weights, reassociated prefill and Hermes prompt convention
are explicit approximations relative to the released BF16 model. The native
head path preserves the selected MXFP4 head representation, not BF16 weights.

September14's30-second check was deferred: minimum available6.661GB against
the selected6.70GB launch check, no swap growth/out, no known transcoders,
and22.027GB minimum root free. No model run was started from that failed gate.
The preview profile is just the existing measured settings combined; it does
not fix the unresolved repeated-request failure.
