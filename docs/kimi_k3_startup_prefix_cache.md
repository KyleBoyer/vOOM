# Kimi K3 startup prefix and durable KV cache

`runtime.server` can seed stable Kimi K3 harness prefixes before it reports
the HTTP endpoint ready. The prefix definition is a small, versioned JSON
document; it does not copy request contents into a profile:

```json
{
  "schema": "voom.startup-prefixes.v1",
  "prefixes": [
    {
      "name": "kai-tools-system-v1",
      "model": "lossy-Kimi-K3",
      "request_file": "/absolute/path/to/a/real-responses-request.json",
      "request_sha256": "64-lowercase-hex-characters",
      "derivation": "kimi-k3-responses-static-v1",
      "cache_namespace": "default",
      "require_persistence": true
    }
  ]
}
```

Start the server with either `--prewarm-prefixes FILE` or the
`VMODEL_PREWARM_PREFIXES` path-list environment variable. The optional
`--prewarm-result FILE`/`VMODEL_PREWARM_RESULT` writes atomic, content-free
telemetry including whether startup computed the prefix cold or found a
durable endpoint loaded from the prior process.

For this Mac, the complete measured feature group is:

```sh
.venv/bin/python -m runtime.server \
  --profile kimi-k3-this-mac-fast-tier \
  --prewarm-prefixes profiles.local/kimi-k3-kai-prewarm.json \
  --prewarm-result logs/kimi_k3_startup_prewarm_result.json
```

The portable `kimi-k3-kai-fast` profile contains arithmetic and memory
settings only. The `profiles.local/kimi-k3-this-mac-fast-tier.yaml` overlay
supplies this machine's sidecars, REAP calibration, spill roots, persistent
cache root, and startup document. Machine paths remain intentionally untracked.

## Exact boundary derivation

The `kimi-k3-responses-static-v1` derivation retains the complete tool catalog
and only the leading `system`/`developer` input items. It never seeds a user or
history item. Because BPE tokenization is not prefix-stable at an arbitrary
character boundary, the runtime tokenizes two synthetic continuations (`A`
and `Z`) and uses their longest common token prefix. It then proves those IDs
are an exact prefix of the source request before any model I/O starts.

At request time, reuse still requires literal token equality and the same
cache namespace. Different tools, instructions, leading messages, renderer,
or tokenizer simply produce no match and fall back to ordinary prefill.

## What is persisted

K3 is hybrid recurrent/attention state, so ordinary token-indexed KV alone is
not sufficient. The durable hot-KV checkpoint stores:

- exact compressed-MLA latent KV for every attention layer;
- exact KDA recurrent and convolution state for every linear-attention layer;
- prompt-endpoint and continuation logits;
- exact token IDs and namespace metadata in an immutable parent-hashed segment
  DAG.

AttnRes and attention-score tiles are transient prefill scratch. They are not
needed after the endpoint's MLA/KDA state exists, so persisting them would add
disk I/O without making restart reuse more correct or faster.

Restored K3 checkpoints are materialized only long enough to validate their
exact arrays. The runtime then reattaches process-local MLA/KDA spill stores
and moves those arrays back to disk, preserving the logical endpoint while
avoiding a second resident copy during the first continuation. After a newly
computed endpoint is durably published, the same re-spill step returns the
live slot to the bounded representation.

The durable payload intentionally does not encode process-local execution
flags. Restore therefore reapplies the fingerprinted compressed/absorbed MLA
mode and key-tile size as well as spill directories. Without that step, a
restart would expand the same long-context latents into per-head K/V even when
the cold process used absorbed MLA, changing arithmetic and adding several
gigabytes of transient storage.

Single-position native-MXFP4 dense computations—both layer 0 and every MoE
layer's shared expert—also use an exact staged schedule: the gate projection,
up projection, SiLU product, and down projection are evaluated in order. These
subprojections are independent until their elementwise product and final down
projection, so the barriers alter scheduling and peak live storage, not values
or operation order within any projection.

During decode, a completed layer's updated KDA or MLA endpoint is likewise
re-spilled immediately. Layer `i` of the next token is its first possible
consumer; no layer `j > i` in the current token can read it. This exact
layer-lifetime rule prevents a restored long-context endpoint from gradually
materializing all layers in Metal during one decode sweep.

Each payload and manifest is checksummed and atomically published. The cache
fingerprint covers model/tokenizer metadata, runtime source files, arithmetic
settings, quantization, sidecar generations, expert top-k and REAP masks, RoPE,
and state-format versions. Therefore a code, checkpoint, renderer, routing, or
profile change cannot silently consume an old endpoint. A dirty source edit is
also detected; cache safety is stronger than comparing only `git rev-parse
HEAD`.

`VMODEL_K3_HOT_KV_PERSIST_MAX_CHECKPOINTS` bounds histories, while
`VMODEL_K3_HOT_KV_PERSIST_MAX_MB` bounds reachable checkpoint plus shared
segment bytes. GC is lease-aware and does not remove files being validated by
a reader.

## HTTP restart gate

`tests/fixtures/kimi_k3_http_replay_gate.py` runs two complete server
lifecycles. It requires a fresh 30-second memory preflight before each start,
replays a distinct real Kai capture through `/v1/responses`, shuts the server
down, restarts it with the same profile/code, and proves:

- startup loaded the prior durable prefix;
- both unseen HTTP requests reused exactly the declared prefix token count;
- restart startup and first-token thresholds passed;
- peak Metal stayed at or below 8.5 GB.

The gate states its two request mutations in the artifact: the captured Qwen
model ID is replaced with `lossy-Kimi-K3`, and output is bounded to two tokens.
Messages, tools, temperature, streaming mode, and all other request fields are
preserved.

`tests/fixtures/kimi_k3_decode_memory_gate.py` is the shorter prerequisite for
that expensive replay. It executes a real one-position sweep through all K3
layers and rejects a true Metal high-water above 8.5 GB.
