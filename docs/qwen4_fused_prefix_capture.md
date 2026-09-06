# Qwen4 cold aligned-prefix capture: integration and proof contract

As of2026-09-06 the helper is wired into an **explicit, default-off**
`...hot-kv-aligned-compact-fused` profile;866 regressions pass. Real-model
state and serving gates remain next. The objective is to avoid a separate cold
prefix sweep while retaining the existing aligned1024-token hot checkpoint.
The focused1611-token workload currently reads an additional101,039,536,440
bytes for that split. This is an I/O hypothesis, not an achieved speed gain.

## Proven helper scope

`runtime.qwen4_prefix_capture.AlignedPrefixCapture` requires exact concrete
KV/KDA/Qwen4 companion types, independent complete lists and genuinely empty
position-zero state. It supports only plain, uncompressed, unwindowed RAM
caches; no DSA companion, KDA spilling/factor capture, or derived QSA pool.
The prefix is positive, strictly interior and a multiple of the original
fixed tile. Geometry is copied from config when the builder is constructed.

The helper imports no model or MLX code. Supply the concrete cache types,
BF16 activation / FP32 recurrence / int32 position dtypes, `mx.eval`, the
existing `copy_history_bits`, governor scratch reservation and a factory that
creates a new empty KVCache with fresh KDAStateCache/Qwen4ExpStateCache owners.
The caller must already charge the additional retained logical state and
backing allowance. Scratch reservation is per layer, twice the small history
payload, without credit for future release of the source's padded backings.

`begin_sweep` rechecks a cold, unchanged source and exact offset/total/tile.
`observe_tile` must see EVERY unchanged post-attention tile in original
layer-major order, including all prefix/suffix tiles. It validates local
geometry each time; global KV offset is deliberately not used while layers
cover different positions. At the prefix, it evaluates all shared arrays
(including below-indexer-threshold raw QSA state) and copies only KDA/PLE
convolution histories. Source arrays/list entries are not replaced.

Final-tile object witnesses protect non-position-indexed KDA state against
same-shape rewinds before `finish`. `finish` requires all layers and tiles,
revalidates full source state and creates/populates a new destination. It
returns `(prefix_cache, scalar_stats)` exactly once. On failure it drops private
records and clears only a previously validated, independently owned partial
destination. Never clear an invalid factory result that could alias the source.

Synthetic tests cover4/8/1024-token boundaries, BF16/F16 bit preservation,
all relevant state/metadata, source ownership, real depthwise-convolution
continuation, invalid geometry/order, interruptions and partial publication.
They do not prove full-model arithmetic/scheduling or serving pressure.

## Engine hookup contract (implemented; retain these invariants)

1. Add a default-off runtime/YAML/server identity flag and an explicit profile
   extending `qwen38-flash-next-uncensored-fp8-exact-pipeline-hot-kv-aligned-compact`.
   Do not change any existing profile or automatic behavior. Keep the existing
   retained logical/QSA-view admission charge; do not double-charge it or
   assume shared arrays prove a full physical-memory bound.
2. In `generate`, use a separate deferred Qwen4 prefix-token scalar. The private
   builder is created only inside the complete sweep's owning wrapper. Cold gating
   needs aligned+compact+hot eligibility, `pos==matched==0`, fixed layer-major
   tiles, plain RAM KV, no approximate state, persistence, adaptive paging,
   adaptive chunks, checkpoints or last-token-separate path. Unsupported
   matches/extensions keep the existing split path, not reinterpret
   their recurrence. A builder that loses eligibility must fail closed before
   any alternate sweep.
3. Defer BOTH the prefix mini-sweep and its later fork/`pos=stable_boundary`
   bookkeeping. The existing `if not fuse_qwen_boundary_scaffold` block needs
   an additional Qwen4 exclusion. Leave `pos=0`, `boundary_fork_kv=None` and
   `boundary_fork_tokens=0` until successful completion. Do not reuse the dense
   Qwen3.5 deferral variable or compact a mixed/empty cache.
4. Give `_layer_stationary_qwen4_sweep` an optional builder. Validate its sweep
   contract before fetching weights. Call `observe_tile` just AFTER the
   existing `mx.eval(post_attention)` and attention-time accounting, but BEFORE
   the routing timer reset, route/GEMM work, host overwrite and next tile.
   Do not change `range/end`, tile shapes, records, expert unions, row ordering
   or accumulator evaluation shapes. Record capture time separately and call
   the existing cap/true-peak sampler after each capture.
5. Only after the entire sweep AND final hidden restoration succeed may the
   caller invoke `finish`, set the stable boundary cache/token count and merge
   typed scalar telemetry. Do not run whole-fork compaction again: histories
   are already compact. On any exception/client interruption outside helper
   callbacks, explicitly `abort` and drop the builder. No partial cache may
   reach `last_kv`, hot slots or persistence. Matched-prefix repeats retain the
   existing proven complete-fork/compactor path.
6. Add actual call-site/flag/profile/admission/cleanup regressions and run the
   combined suite under a new fresh30-second preflight and supervised gate.

## Real-model proof order

Keep every result explicit about the modified two-real-tool / short history /
developer / streaming / greedy-seed64001 request. It is not the unmodified
134-tool capture or Plex quality proof. No synthetic replacement of state or
answer text is acceptable in a serving benchmark.

First compare a fresh max1 endpoint diagnostic (not complete-answer latency)
against BOTH independent state oracles:

- Full1611-token endpoint, hidden BF16 bits, actual prepared/generated IDs and
  raw text: `logs/qwen4_boundary_nohot_first_20260905.json` and the completed
  aligned-compact max1 control. Full state SHA
  `ef35e7f30d7a7bd28b956c9d352164a8d89659905c7df288323d89a9b6b7ac77`.
- Retained1024-token prefix: independently computed separate prefix sweep in
  `logs/qwen4_retained_allconv_memory_first_20260906.json`, diagnostic
  `fork_before`/`fork_after`. Full state SHA
  `8842fff6d80f36ce58fb03fb7d3f1a51f78bc639c4ef57471aafc71ad7fa78fe`,
  supplemental starts/windows/pool metadata SHA
  `fd4c4aefb4703727e67e08030a6c799b50d9c2e454e2de2ff2705b5b79c10c3c`.

Use a read-only observer for those retained-state hashes. Do not invoke the
older post-generation detachment diagnostic or call its pressure numbers an
uninterrupted serving pass. Check all component hashes, metadata and history
coverage, not just first-token identity or parsed tool arguments.

Then run fresh sufficient-output controls/candidates with generation witnesses,
no endpoint hashing/mutation or added allocator clearing: cold + identical
repeat, then actual extension, varied domains/tool shapes, unmodified capture
and Plex. The existing modified max512 workspace call naturally completes at
70 tokens; raw ID hash
`381c8319f7f04acfd719eb922593657e5e688b45f0273eb678ae6e9735a9ae9b`.
Record output completion, actual IDs/text, cache reuse, I/O, cold loading,
first-token/prefill/decode/HTTP wall, peak Metal and unchanged pressure gates.
Failure of any gate stays visible; no automatic promotion without broad proof.
