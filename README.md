# vOOM — virtualized LLM runtime

[![tests](https://github.com/KyleBoyer/vOOM/actions/workflows/tests.yml/badge.svg)](https://github.com/KyleBoyer/vOOM/actions/workflows/tests.yml)

*(as in **v**irtual memory, and the **OOM** error it's designed to route around)*

Proof-of-concept runtime that treats transformer weights like virtual memory:
models larger than RAM run on Apple Silicon by paging weights, layers, and
KV-cache between SSD, the OS page cache, and unified memory.

**Target machine: an M4 Mac Mini with 16 GB unified memory.** The project's
guiding question is how large a model that machine can serve, and how fast,
without quantizing or pruning the model being served (see "Goals" below).

Headline result so far: GLM-5.2's released 1.49 TB BF16 checkpoint produced a
coherent greedy stream on this 16 GB Mac, streamed from external storage. No
target weight was quantized or pruned. That proves the complete released BF16
artifact can be executed out of core at short context — it does **not** yet
prove released-model token conformance at long context, which is still under
active validation (see "Status" below).

## Goals

1. **Goal**: run a released model exactly as published (bit-for-bit BF16),
   larger than fits in RAM, on a single consumer Mac.
2. **Sub-goal**: make that faster using only *lossless* techniques
   (compression, tiering, paging, speculation, batching — never quantizing
   the target model).
3. **Side-quest**: run the same model *as fast as possible*, lossy techniques
   allowed (quantized weights, pruned experts, etc.), as a separate mode.

## Layout

```
runtime/
  config.py          HF config.json -> ModelConfig
  local_config.py     machine-local storage config (see "Extra storage" below)
  path_resolver.py     pure mountpoint health/re-resolution helpers
  model_loader.py      WeightStore: lazy per-tensor safetensors access (mx.load)
  layer_runner.py      shared block math (RMSNorm/RoPE/GQA/SwiGLU, qkv bias)
  weight_cache.py       WeightPage/WeightCache: LFU-admit, byte budget, pinning
  expert_batching.py    bounded-lifetime MoE expert-batch consumption
  prefetcher.py         background sequential prefetch into the cache
  expert_plan.py        held-out expert layout/prefetch trace simulator
  kv_cache.py           simple all-resident KV
  kv_paged.py           PagedKVCache: fixed-size pages, disk spill + reload
  quant.py              quantize-on-load (QTensor, per-module policy)
  predictor.py          Markov expert-transition predictor, persisted per model
  kv_store.py           prompt-KV persistence across requests
  memory_planner.py     placement policies -> RuntimeConfig + cost estimate
  engine.py             StreamingEngine: generate() loop / layer scheduler
  incremental_decode.py stateful byte-safe streaming detokenization
  request_state.py      single-owner text/vision request-state release
  glm.py                GLM MLA/MoE block math
  glm_dsa.py            DeepSeek Sparse Attention indexer + IndexShare
  glm_mtp.py             native multi-token-prediction draft block
  pressure.py            proactive memory governor for the Metal ceiling
  server.py              OpenAI-compatible endpoint and engine switching
  toolcalls.py           tool schemas, protocol normalization, call parsing
  structured.py          XGrammar JSON/schema/tool-constrained decoding
  qwen3vl.py             Qwen3-VL image/video tower and multimodal generation
  qwen35.py              Qwen3.5/3.6 hybrid DeltaNet/attention/MoE text runtime
  vision_positions.py    image/video grids and interleaved M-RoPE positions
  speculative.py          greedy speculative decoding (draft proposes k, target verifies)
  sampler.py              greedy/categorical temperature, top-p, top-k sampling
  telemetry.py            RSS + Metal memory + timers
experiments/
  (local research/validation scripts — not part of this repo; see below)
formats/
  packed.py              vpack: lossless byte-plane zstd weight store
  packed2.py             vpack2: coalesced archive format
  gguf_notes.md          why GGUF isn't the primary format (design rationale)
docs/
  agent_prompt_acceleration.md  compact tools, exact hot KV, capsules, YaRN
  design.md              Phase 0 design notes (format choice, MLX behavior, tiers)
  memory_model.md        tier model and placement math
```

This repo intentionally does **not** include this project's day-to-day
research journal (incident logs, benchmark numbers, the open-item queue) or
its exploratory `experiments/` scripts — those are working notes, not the
shipped runtime. What's here is the runtime code, its tests, and the design
docs that explain *how* it works.

## Setup

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install mlx safetensors huggingface_hub tokenizers psutil pyyaml \
    jinja2 numpy pillow zstandard pytest xgrammar jsonschema av
# fetch a model (everything stays on this volume)
HF_HOME=$PWD/hf_cache .venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('HuggingFaceTB/SmolLM2-135M',
    allow_patterns=['*.json','*.safetensors','tokenizer*'], local_dir='models/SmolLM2-135M')"
```

### Extra storage (optional)

By default, every model lives under this repo's own `models/` directory and
no further setup is needed. If you keep models on additional storage — a NAS
share, a second SSD, a USB drive — copy `voom.local.example.yaml` to
`voom.local.yaml` (gitignored) and fill in your own mountpoint name,
model subdirectory, and (if it's a network share that can drop) a remount
command:

```yaml
volumes_root: /Volumes
model_stores:
  - name: YourShareName
    models_subdir: models
    remount_command: >-
      osascript -e 'mount volume "smb://youruser@yournas._smb._tcp.local/YourShareName"' >/dev/null 2>&1
```

With no `voom.local.yaml` present, every path is treated as ordinary local
storage — this is not required to run the project.

## Run

```bash
# Start the tracked OpenAI/Anthropic-compatible server. Runtime defaults choose
# the bounded cache/prefetch profile for the requested local model.
.venv/bin/python -m runtime.server --port 8077

# Recommended lossy prompt profile for large agent harnesses. The model first
# sees a fixed private search/re-enable catalog; full schemas are inserted only
# if selected, and both hidden-phase KV states are checkpointed independently.
VMODEL_FAST_TOOL_GATEWAY=1 VMODEL_FAST_TOOL_GATEWAY_LIMIT=32 \
    VMODEL_FAST_TOOL_GATEWAY_SEARCH_RESULTS=4 \
    VMODEL_FAST_TOOL_GATEWAY_KV_CHUNK_SIZE=512 \
    VMODEL_TOOL_EMBEDDINGS=auto \
    VMODEL_HOT_PROMPT_KV_PERSIST_DIR=.kv_prompts/qwen3-4b-fast \
    VMODEL_HOT_PROMPT_KV_PERSIST_MAX_CHECKPOINTS=32 \
    .venv/bin/python -m runtime.server --port 8077

# Serving does not reserve an additional fixed amount of otherwise available
# RAM: the live governor enforces the Metal ceiling, evicts durable KV branches
# from RAM, and shrinks the weight cache. Proof/benchmark runs may opt into an
# extra reserve with VMODEL_HOT_PROMPT_KV_MIN_AVAILABLE_MB.

# Optional one-time semantic cache build (run without a serving model loaded).
# Raw schemas/queries are not persisted in .tool_embeddings; only hashed,
# integrity-checked vectors are stored there.
.venv/bin/python -m huggingface_hub.cli.hf download BAAI/bge-small-en-v1.5 \
    --revision 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a \
    --local-dir models/tool-embed-bge-small-en-v1.5 \
    config.json model.safetensors tokenizer.json tokenizer_config.json \
    special_tokens_map.json vocab.txt
.venv/bin/python -m runtime.tool_embeddings build \
    --capture /path/to/captured-responses-request.json

# Simpler deterministic alternative: keep the top 32 relevant schemas plus
# every tool explicitly named by the user or already used in the transcript.
VMODEL_FAST_TOOL_LIMIT=32 .venv/bin/python -m runtime.server --port 8077

# Dependency-light adapter/math regressions (no model process).
.venv/bin/python -m pytest -q tests/test_toolcalls.py tests/test_server_pure.py \
    tests/test_tool_embeddings.py tests/test_yarn_parameters.py tests/test_vision_positions.py \
    tests/test_incremental_decode.py
```

Raw HF indexes do not hash shard bodies. For proof-grade cache identity, build
and verify a full-body manifest once, then require it at server startup:

```bash
.venv/bin/python -m formats.hash_safetensors /path/to/model
VMODEL_REQUIRE_RAW_WEIGHT_HASHES=1 .venv/bin/python -m runtime.server --port 8077
```

Required mode rehashes every referenced shard before accepting weights or
persisted prompt KV, so even a same-size/same-mtime replacement fails closed.
It is opt-in because scanning a large raw checkpoint adds startup I/O; hashed
vpack2 remains the lower-overhead packed alternative.

YAML config (see `RuntimeConfig.from_yaml`):

```yaml
memory:
  max_weight_cache_mb: 6000
  pinned: {embeddings: true, lm_head: false, first_layers: 0, last_layers: 0}
  max_kv_mb: 512
  kv_page_positions: 256
prefetch: {depth: 2}
quant: {bits: 4, attention: false, mlp: true}   # mixed: attn bf16, MLP 4-bit
```

## Running the server

```bash
.venv/bin/python -m runtime.server --port 8077

# OpenAI-compatible:
curl -s http://127.0.0.1:8077/v1/models
curl -s -X POST http://127.0.0.1:8077/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<local-model-name>","messages":[{"role":"user","content":"Hi"}],"max_tokens":60}'

# Side-quest (lossy fast) mode per request:
curl ... -d '{"model":"lossy-<local-model-name>", ...}'

# Experimental Qwen2 static-YaRN factor-2 profile (capacity, not speed):
curl ... -d '{"model":"lossy-long-<local-model-name>", ...}'
```

### Qwen3.6-35B-A3B

`Qwen/Qwen3.6-35B-A3B` is supported losslessly for text generation and tool
calling through the official hybrid text architecture: three recurrent Gated
DeltaNet layers followed by one gated full-attention layer, repeating across
40 layers. The loader accepts the multimodal checkpoint wrapper but rejects
image input explicitly; its vision tower is not interchangeable with the
Qwen3-VL implementation. The checkpoint's MTP layer is preserved in the packed
store but is not yet used for speculative decoding.

The recurrent state is request-local today. Hot prompt-KV retention is disabled
for this model until the cache can preserve both DeltaNet convolution/matrix
state and attention KV exactly. This makes the first implementation correct but
means follow-up turns still prefill their full prompt. Use the bare model ID for
released BF16 weights:

```bash
curl -s -X POST http://127.0.0.1:8077/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.6-35B-A3B","input":"Reply with READY.","reasoning":{"effort":"none"},"max_output_tokens":8}'
```

For a fully resident OLMoE side-quest artifact, convert beside the source
checkpoint (one source shard is processed at a time, and interrupted runs
resume at a committed shard boundary):

```bash
# Fastest validated profile: expert-only MXFP4; trunk/router/head remain BF16.
.venv/bin/python -m formats.quantize_mlx \
  /path/to/OLMoE-1B-7B-0924-Instruct \
  /path/to/OLMoE-1B-7B-0924-Instruct-mlx-expert-mxfp4

# Higher-fidelity alternative (larger and slower).
.venv/bin/python -m formats.quantize_mlx \
  /path/to/OLMoE-1B-7B-0924-Instruct \
  /path/to/OLMoE-1B-7B-0924-Instruct-mlx-expert-mxfp8 \
  --mode mxfp8 --bits 8
```

The registry discovers complete converted siblings automatically. Requesting
`lossy-OLMoE-1B-7B-0924-Instruct` prefers the adjacent expert-MXFP4 artifact;
the MXFP8 artifact remains explicitly selectable by its advertised `lossy-...`
ID. Locally derived checkpoints are never advertised or accepted as bare
lossless IDs. On a higher-memory development machine (not the M4/16 GB
target), the fastest validated hybrid uses MXFP4 experts,
MXFP8 attention, an affine-Q2 candidate head with exact BF16 top-32 reranking, and
exact stepped KV. A real HTTP request measured 288.7 tok/s at short context;
another with a 3,415-token prompt measured 199.8 tok/s. The latter cache layout
was 1.90x the old concatenate-per-token decode path with identical tokens. Short
peak Metal was ~4.21 GB. The attention quality gate held restricted-choice
accuracy at 24/30 versus 23/30 for BF16 attention and changed held-out NLL only
3.421 -> 3.449 across 3,072 local code/prose tokens. Head reranking matched the
BF16-head path for 3,268/3,268 task and diverse held-out gate tokens. Candidate
recall remains empirical, so this is
still explicitly lossy. The underlying expert-MXFP4 path also produced the same
128/128 greedy tokens as MLX-LM under its portable prefill schedule. The
higher-fidelity expert-MXFP8/BF16-attention control measured 187.2 tok/s and
matched 319/319 BF16-head tokens at 7.69 GB. A 3,501-token MXFP8 prompt reached
8.49 GB while this machine still had ample memory available; the governor uses live
available memory
plus MLX's device-recommended working set rather than treating that number as a
universal cap. Non-streaming responses and streaming chunks expose the effective
checkpoint and weight profile. Target-M4/16-GB validation is still required.

Lossless OLMoE keeps the released BF16 expert path and arithmetic but now asks
the governor for a cache sized from the checkpoint's actual tensor payload. If
live headroom admits all 13.84 GB, exact pages stop cycling through the old 6 GB
window; otherwise the server reconstructs that established streamed profile. A
real 64-token production-path A/B matched every ID, reduced total time from
9.25s to 5.01s (1.85x; decode 1.98x), read 11.87 GB instead of 45.86 GB, and
peaked at 11.88 GB with zero candidate evictions. A gathered/fused BF16 expert
prototype was much faster but changed the target stream at token 8, so it was
rejected rather than labeled lossless. Reproduce the retained cache-only gate
with:

```bash
~/.hf-pull/bin/python tests/fixtures/olmoe_lossless_cache_gate.py \
  --model ~/models/OLMoE-1B-7B-0924-Instruct
```

Large lossless dense Qwen2 targets first receive a model-sized exact-cache
allowance fitted to the governor's sampled live headroom. If the complete target
fits, the server reconstructs untied models with their released BF16 embedding
pinned and uses the resident lazy decode loop; otherwise it reconstructs the
established 6 GB streamed target and retains exact speculative decoding. This is
an admission decision, not a fixed machine-wide memory cap. On a higher-memory
development machine (not the M4/16 GB target),
Qwen2.5-7B residency matched all 96 target-verified IDs across three prompts and
took 6.09s versus 20.88s for the speculative path (3.43x faster), exercising the
resident loop for every decode step at a 15.24 GB peak. A separate first-prompt
quick gate was 15.48s streamed versus 1.87s resident (8.28x) with identical IDs.

When residency is not admitted, the server automatically discovers an adjacent,
complete `Qwen*-1.5B-*-mlx-mxfp4` draft with the same named family, then
target-verifies every committed token with one-token arithmetic shapes. This is
used for returned or streamed requests (including string stops) with prompts up
to 2,048 tokens; longer requests fall back to ordinary target generation. The
original local Qwen2.5-7B/Qwen2.5-1.5B-MXFP4 A/B matched 32/32 target token IDs
and reduced total time from 21.45s to 6.16s (3.48x), with 7.51 GB observed peak.
Set `VMODEL_SPECULATIVE_DRAFT=off` to disable the fallback or provide an explicit
local draft path to force that experimental pairing; `VMODEL_SPECULATIVE_K`,
`VMODEL_SPECULATIVE_MAX_PROMPT_TOKENS`, and
`VMODEL_SPECULATIVE_DRAFT_CACHE_MB` tune the validated defaults (6, 2048, and
1200). `VMODEL_SPECULATIVE_TARGET_PREFETCH_WORKERS` defaults to the locally
validated two workers (8.54% faster than one on the raw Qwen-7B target; three
regressed); `VMODEL_SPECULATIVE_TARGET_PREFETCH_DEPTH` defaults to four (4.87%
over depth two with two workers). Responses disclose the exact draft checkpoint
and decode profile. A paired target/draft in-memory prompt endpoint is retained
for exact repeats from 2,048 tokens upward; a real 2,048-token repeat fell from
5.08s to 1.63s (3.12x) with identical target IDs. The threshold is
configurable with `VMODEL_SPECULATIVE_PROMPT_CACHE_MIN_TOKENS`.

Reproduce the exact resident/streamed comparison with:

```bash
~/.hf-pull/bin/python tests/fixtures/qwen_lossless_resident_gate.py \
  --target ~/models/Qwen2.5-7B-Instruct
```

Streamed lossless Qwen3 targets can use the released
[DSpark](https://arxiv.org/abs/2607.05147) semi-autoregressive drafter. The
server strictly auto-discovers a complete, shape-compatible sibling such as
`dspark_qwen3_4b_block7`; it never downloads one implicitly. Before loading it,
the server fits a model-sized exact target cache to the governor's sampled live
headroom. If the complete target fits, ordinary resident target decoding wins
and the drafter is not loaded. On this host that path matched 128/128 streamed
target IDs, took 3.28s across four prompts, and peaked at 8.06 GB; forcing
DSpark beside a fully cached target was slower (0.884x). This is why the cache
allowance is not a fixed 8.5/9 GB machine cap. The admitted resident path also
uses the existing exact in-memory prompt endpoint above the 2,048-token
threshold: an isolated repeat measured about 0.29ms versus 18.95ms from the
durable disk snapshot, with identical IDs.

When live headroom forces the target to stream, the implementation
captures the checkpoint's five post-layer target streams, drafts a full
bidirectional seven-position block, applies the sequential Markov correction,
and verifies every proposal with one-token target arithmetic before committing
it. Accept-none, partial, and accept-all rollback paths are regression-gated.

The constrained 6 GB target-cache Qwen3-4B/block-7 gate matched 128/128
ordinary-target IDs across raw
and non-thinking chat prompts, reducing aggregate wall time from 29.70s to
12.21s (2.43x) at a 9.24 GB observed Metal peak. Two target prefetch workers at
depth four reduced a representative DSpark request by another 30.8% versus the
prior default-worker/depth-two schedule; three workers regressed. A four-token verification cap
beat caps 2/3/5/6/7 by more than the project threshold on the streamed target.
Confidence scheduling remains available but defaults off: threshold 0.1 was
only about 1.1% faster over the four-prompt screen, too close to noise to become
the default. Exact paired target-KV/drafter-context reuse starts at 2,048 prompt
tokens; a real exact repeat skipped prefill (1.762s to 0.00064s) and reduced
total time from 2.79s to 1.01s (2.76x), with identical IDs and a 10.09 GB peak.

Use `VMODEL_DSPARK_DRAFT=off` to disable discovery or set it to an explicit
local checkpoint. `VMODEL_DSPARK_MAX_DRAFT_TOKENS`,
`VMODEL_DSPARK_MAX_PROMPT_TOKENS`,
`VMODEL_DSPARK_CONFIDENCE_THRESHOLD`, and
`VMODEL_DSPARK_PROMPT_CACHE_MIN_TOKENS` control the validated defaults (4,
2048, 0, and 2048). `VMODEL_DSPARK_TARGET_PREFETCH_WORKERS` and
`VMODEL_DSPARK_TARGET_PREFETCH_DEPTH` default to the validated 2 and 4. Run the
reproducible constrained-cache fallback gate with:

```bash
~/.hf-pull/bin/python tests/fixtures/qwen3_dspark_gate.py \
  --target ~/models/Qwen3-4B \
  --draft ~/models/dspark_qwen3_4b_block7
```

`VMODEL_DSPARK_TARGET_CACHE_MB` can override the model-sized configured cache
ceiling for experiments. By default the server derives it from 107% of the
checkpoint's estimated exact footprint, then the governor fits it downward to
current device/system headroom and can restore it later.

DSpark is intentionally not enabled for the lossy resident Qwen3 profile. That
target already measured 135.5 tok/s; adding exact DSpark verification fell to
50.2 tok/s (0.384x), despite matching IDs. The current lossy Qwen3 MXFP4 path
measured about 138 tok/s at a 2.94 GB peak. Head-only affine-Q2 reached about
140 tok/s but regressed held-out NLL from 3.837 to 4.176, so it was rejected;
NVFP4/affine-Q4 head alternatives improved less than 1% and were also discarded.

Small dense lossless models that pass the resident admission check also retain
one exact in-memory prompt/KV endpoint for prompts of at least 2,048 tokens,
ahead of the durable disk journal. Repeated 1.9K/4K local prompts were
6.3%/2.1% faster than disk exact hits with identical IDs. The existing
`VMODEL_HOT_PROMPT_KV_SLOTS` and `VMODEL_HOT_PROMPT_KV_MIN_TOKENS` controls apply.
The durable store now uses immutable SHA-256-verified delta generations, reader
leases, and byte/count-bounded GC; extending a known prefix no longer rewrites a
full snapshot. It defaults to a 2,048-token admission threshold
(`VMODEL_PROMPT_KV_MIN_TOKENS`): on the prior 724-entry v5 store, a
paired short one-token ABBA gate dropped median latency from 249.5ms to 77.0ms
(3.24x) with identical IDs by avoiding a 142ms metadata scan plus a 29ms
write. Obsolete v5 snapshot pairs are swept once on upgrade. Set the threshold
to zero to restore cache-every-request
behavior.

Qwen3-VL is validated end-to-end with the official
`Qwen/Qwen3-VL-2B-Instruct` checkpoint across OpenAI and Anthropic image inputs.
The exact path keeps the released BF16 tower and up to 4,096 spatial patches.
Because the complete 4.0 GB checkpoint fits the ordinary 6 GB cache allowance,
it now enables the custom M-RoPE resident decode pipeline too. A five-pair exact
BF16 ABBA gate held all 56 IDs on every run, exercised all 55 continuation
steps, and reduced median decode from 1.202s to 0.614s (1.96x) at a 4.28 GB
peak. Reproduce it with
`~/.hf-pull/bin/python tests/fixtures/qwen3vl_resident_gate.py --model
~/hf_cache/modelscope/models/Qwen3-VL-2B-Instruct`.
`lossy-Qwen3-VL-2B-Instruct` uses the quality-gated fast profile: text-MLP
MXFP4, BF16 text attention/head and vision tower, and a 1,024-patch image budget.
Uniform text MXFP4 was rejected because it reversed a two-image ordering answer;
vision-MLP MXFP4 was rejected because it was no faster and moved final vision
embeddings far outside the accepted oracle envelope. On a higher-memory
development machine (not the M4/16 GB target), the
retained profile decoded a 64-token image description at 128.7 tok/s with a
2.77 GB peak. High-resolution
layout/count/OCR gates retained `Blue`, `7`, `3`, and `CODE 731` while warm
request latency fell from roughly 0.8-1.0s exact to 0.23-0.25s fast.
The lossy resident VL decode loop carries its compressed M-RoPE position through
the same lazy pipeline used by resident text decode: an alternating five-pair
local A/B kept every token ID identical and moved median 63-token decode from
0.5020s to 0.4196s (125.5 to 150.2 tok/s, 1.20x) at the same 2.77 GB peak.

Vision weights use the governor-controlled weight cache, media in one request
share one tower fetch, and content-identical images/videos reuse exact evaluated
embeddings. An exact single-owner vision prompt-KV slot also serves identical
requests and text-only follow-up suffixes when expanded tokens and media hashes
prove prefix identity. A valid prompt-KV hit now bypasses the vision tower
entirely rather than first consulting/rebuilding embeddings: with the embedding
LRU deliberately disabled, the local exact repeat retained its ID, skipped a
measured 25.6 ms tower pass, and completed in 2.95 ms. Qwen3-VL video input
supports URL/path/base64 blocks,
bounded PyAV decoding, uniform 2-fps sampling, timestamp expansion, and
per-temporal-patch attention segmentation. The real-model oracle gate matched
Transformers token IDs and M-RoPE positions exactly (pixel max error 5.6e-8),
while correcting cross-frame attention moved final-embedding cosine from 0.551
to 0.995. A temporal color-order task passed in exact and lossy modes; lossy
decode measured 147.4 versus 88.3 tok/s (1.67x) at 64% of exact peak Metal, and
an identical exact video request was 4.76x faster through embedding plus
prompt-KV reuse.

`VMODEL_VISION_CACHE_ENTRIES` controls the embedding LRU (default 4; 0
disables it), `VMODEL_VISION_PROMPT_CACHE=0` disables vision prompt-KV reuse,
and `VMODEL_FAST_VISION_MAX_PATCHES` changes the fast per-frame quality/speed
point (validated default 1024; allowed range 256-4096). Exact attention is
bounded to 4,096 patches per image/video segment and 4,096 retained merged
tokens in aggregate. Video defaults are 16 sampled frames, 64 MiB, 60 seconds,
and 2 fps; the corresponding `VMODEL_VIDEO_*` variables are explicit caps.

Also speaks the Anthropic Messages API (`POST /v1/messages`) and the OpenAI
Responses API (`POST /v1/responses`), with tool-calling and typed text/vision
streaming; every path also works with a leading `/v1` omitted. Model IDs map to
whatever is in your local `models/`
registry, plus anything discoverable through your configured extra storage (see
above). One engine is resident at a time; switching model or mode swaps engines.

For the 131-tool agent-harness path, fast schema compaction, canonical tool
catalog ordering (so permutations hit the same exact prompt cache), exact
resident prompt-prefix reuse, the optional deterministic tool shortlist, and the
separate YaRN profile are documented in
[docs/agent_prompt_acceleration.md](docs/agent_prompt_acceleration.md).
Edited Qwen/OLMoE catalogs can additionally use the governor-admitted lossy PIC
path (`VMODEL_FAST_TOOL_PIC`, default 1): unchanged tool tails relocate their KV
while four boundary tokens per tool and every uncached/query token are recomputed.
Exact hits always win. The 60->61-tool Qwen2.5-7B gate retained all five tested
tool names, required paths, and greedy streams while reducing median edited
prefill by 6.86x versus exact-prefix reuse. The resident OLMoE gate retained its
tool call and all 32 IDs at 4.32x, while three Qwen3-VL image/tool cases retained
all 144 IDs and required paths at a 3.80x median prefill speedup. Qwen3-VL uses
explicit M-RoPE relocation and recomputes selected vision embeddings plus
DeepStack additions. Checksummed checkpoint manifests retain capsule spans, so
the first edited catalog after restart can use PIC (real Qwen gate: 1.95x).

Dense Qwen2 fast profiles can additionally opt into unrotated shared physical
pages with `VMODEL_FAST_TOOL_PIC_SHARED_PAGES=1` (default 0). A hybrid Metal/MLX
attention path keeps short page-table decode fused and materializes only a
request-local rotated view for long decode; retained LRU state drops that view and
shares immutable pages. On the Qwen2.5-7B 60->61-tool five-case gate this matched
all established-PIC streams, improved edited prefill 1.068x and total edited-request
time 1.040x, and reduced peak Metal from 4.773 GB to 4.701 GB. The server admits it
only for 7B-class-or-larger dense Qwen2: Qwen3, OLMoE, and the small Qwen2 gate did
not all clear speed plus stable-quality thresholds. Shared pages currently reject
durable KV, disk spill, vision M-RoPE, and GLM/MLA.

Sampling defaults to greedy, preserving deterministic lossless A/B gates.
Explicit positive `temperature` enables categorical sampling; `top_p`, `top_k`,
and `seed` are functional across the protocol adapters. Stochastic speculative
requests fall back to the target because speculative token-equivalence proofs
remain greedy-only. Responses report the active sampling profile.

Safety note: `INFER_LOCK` serializes validated inference and the response stream
per process; bounded body receipt/JSON decoding happens before that critical
section so a stalled uploader cannot block an unrelated generation. Keep this
endpoint development-only, not exposed to untrusted traffic. Auto-download of
unrecognized model IDs is async (returns HTTP 202
immediately and lets you poll `GET /v1/models`) and enforces a 5-GB free-space
floor, but does not yet estimate the requested repository's actual size.
Malformed bodies and invalid stream/stop types fail as 4xx responses before
model lookup; nested message/tool history and media sources are also preflighted
before model resolution, so an invalid request cannot trigger an accidental
download. Request bodies default to a 64 MiB ceiling
(`VMODEL_MAX_REQUEST_BODY_MB`) and a 30-second receive deadline
(`VMODEL_REQUEST_READ_TIMEOUT_SECONDS`). Response writes also have a 30-second
per-write deadline (`VMODEL_RESPONSE_WRITE_TIMEOUT_SECONDS`), so a client that
stops reading cannot retain the inference lock forever. Image payloads default
to 25 MiB and 64 million source pixels (`VMODEL_MAX_IMAGE_BYTES_MB`,
`VMODEL_MAX_SOURCE_IMAGE_PIXELS`).
Images and videos are decoded during preflight, so malformed data cannot fail
only after SSE headers have already been sent; remote media fetch/decode also
runs before the inference lock, so a slow media host does not stall an unrelated
request.

Tool controls are explicit rather than advisory: OpenAI/Anthropic `none` removes
tools from the rendered prompt, and `parallel_tool_calls=false` /
`disable_parallel_tool_use=true` caps output at one call. Required and
specific-tool choices use token-level XGrammar constraints over the supplied
tool names and argument schemas; JSON-object/JSON-schema structured outputs use
the same constrained decoder and generated arguments are validated again with
`jsonschema` before execution.
Generated calls are executable only when their name is in the offered catalog;
unknown markers remain assistant text. Chat Completions accepts modern
`max_completion_tokens`, opens streams with `role=assistant`, and implements
`stream_options.include_usage`. Unsupported multi-choice, logprob, penalty,
or stateful Responses controls fail explicitly rather than being ignored.

## Expert layout and prefetch experiments

MoE routing traces can be analyzed without rewriting or loading model weights.
After a representative generation, export the authoritative routed unions:

```python
engine.generate(prompt, max_tokens=64)
engine.export_expert_trace("traces/routes.json")
```

Then fit physical orders on the first part of the trace and score them on held-
out sweeps. The default skips the first sweep because `generate()` normally
starts with a multi-position prefill, and it assigns zero idle-I/O capacity to
speculative reads until that capacity has actually been measured:

```bash
.venv/bin/python -m runtime.expert_plan \
  --trace traces/routes.json \
  --bandwidth-mbps 315 \
  --request-overhead-ms 3 \
  --cache-pages 0 \
  --plan-out traces/layout-plan.json \
  --report-out traces/layout-report.json
```

Set `--cache-pages` to the expert-only capacity remaining after subtracting
pinned weights, trunk working pages, and KV; zero deliberately models an all-
miss cache. The report compares independent expert reads, demand-only
coalescing, heat and coactivation orders, and inseparable two/four-page bundles
while charging every unused byte. It also measures held-out Markov prediction
and adjacent-sweep route persistence. A generated plan is logical-only;
applying it to checkpoint bytes still requires transactional publication,
integrity verification, and a greedy token-identity gate.

Predicted expert I/O is controlled separately from deterministic layer
prefetch. It remains off by default. An explicit experiment can enable it while
refusing to queue predictions behind already-busy prefetch work:

```yaml
runtime:
  expert_predictive_prefetch: true
  expert_prefetch_idle_only: true
```

## Testing

```bash
.venv/bin/python -m pytest tests/ -q
```

On low-memory unified-memory hosts, use the serial process-sharded runner. It
starts each test module in a fresh Python process so MLX allocator state cannot
accumulate across modules, and terminates a shard (including child servers) if
system-available memory falls below 4 GB or swap occupancy grows by more than
16 MB:

```bash
.venv/bin/python tests/run_pytest_sharded.py
```

The suite is designed to run without any large model checkpoint or external
storage: it uses tiny synthetic fixtures (built on the fly) and, where
useful, cross-checks this runtime's math against the `transformers` library's
own reference implementations for the same architecture family. A handful of
tests are skipped automatically if optional dependencies (e.g. `torch`,
`transformers`) aren't installed.

## Status

- Streaming, weight caching, prefetch, KV paging, a placement planner, mixed
  precision, MoE expert paging with explicit opt-in predictive prefetch, and
  speculative decoding are all implemented, with an OpenAI/Anthropic-compatible
  server on top.
- Full-artifact short-context execution of a real 700B+-class MoE checkpoint
  is achieved on the target 16 GB machine. Released-model token conformance
  at long context (beyond the model's dense-attention window) is still being
  actively validated — some sparse-attention and speculative-decode paths are
  intentionally gated closed until they pass a strict token-for-token
  comparison against an independent reference implementation, not just this
  runtime's own internal consistency checks.
- The correctness bar throughout is target-only vs. changed-path **greedy
  A/B with byte-identical token IDs**, not a similarity heuristic: any
  mismatch is treated as a failure, and reassociated-arithmetic techniques
  (e.g. an algebraic rewrite of an attention computation) are opt-in and
  documented as such rather than assumed lossless by construction.
