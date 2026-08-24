# Huihui K=64 authoritative rank capture and promotion gate

This closes the exact-head evidence gap without treating one long decode as a
corpus. The explicit live mode evaluates paired all-MXFP4 and released-BF16
head projections at real authoritative target hidden states, then persists
only:

- the exact winner's stable approximate rank;
- whether the actual top-64 shortlist contained that winner;
- whether the approximate and exact full-vocabulary winners agreed; and
- coarse request-shape buckets.

It never writes prompts, message text, tool schemas, hidden states, logits,
token IDs, generated text, or exact winner IDs. The JSONL is mode `0600`,
bounded to 1 MiB, at most 1,200 positions total, at most 128 positions per
request, and at most 200 positions per coarse shape so homogeneous traffic
cannot consume the corpus. Qwen MTP proposal projections are explicitly excluded even though the
draft and target share the physical LM head. Constrained requests exclude the
unrestricted provisional projection and record only the later grammar-aware
target rerank.

## Live capture

Run the ordinary machine preflight first and ensure no other Metal- or
same-disk-heavy job is active:

```bash
cd '/Volumes/Workspace NVME/git/vOOM'
.venv/bin/python -m runtime.memory_preflight \
  --result logs/huihui-rerank64-rank-capture.preflight.json \
  --sample-seconds 30
```

Then start the explicit rerank profile with capture enabled:

```bash
VMODEL_QWEN35_RERANK_LM_HEAD_RANK_CAPTURE="$PWD/logs/huihui-rerank64-authoritative-ranks.jsonl" \
VMODEL_QWEN35_RERANK_LM_HEAD_RANK_CAPTURE_MAX_POSITIONS=1200 \
VMODEL_QWEN35_RERANK_LM_HEAD_RANK_CAPTURE_MAX_PER_REQUEST=128 \
.venv/bin/python -u -m runtime.server \
  --profile huihui-qwen38-27b-fast-rerank64 --port 8077
```

Do not also set `VMODEL_QWEN35_RERANK_LM_HEAD_RECALL_PROBE_EVERY=1` for this
gate. Rank capture invokes the same row-paged full-scan oracle itself, only for
eligible target positions, and reuses that full scan when an ordinary recall
probe happens to coincide. Every authoritative position deliberately reads
the complete 2.543 GB BF16 source head, so this is evidence collection rather
than a latency benchmark.

Accumulate real, unmodified requests instead of generating synthetic prompts
or a single 1,000-token continuation. The fixed gate requires at least eight
requests and six distinct coarse shapes, including all of:

- streaming and non-streaming;
- greedy and stochastic sampling;
- developer-message presence and absence;
- at least two tool-count buckets; and
- at least two system-prompt-length buckets.

The capture stops at its bound and can be continued across server restarts only
when every manifest identity and bound matches exactly. A different exact
source, approximate head, K, or bound fails closed rather than mixing corpora.

## Offline promotion decision

The first JSONL line reports both bound fingerprints. The exact released BF16
fingerprint for the pinned Huihui source is currently
`02c9ed132c902d594ed8528dd4dd0b92a767980e7f6c071c49bc70d4d898a0d1`.
Have the offline gate independently content-hash the current approximate head
instead of trusting the capture's own claimed value:

```bash
.venv/bin/python tests/fixtures/huihui_qwen38_head_rank_gate.py \
  logs/huihui-rerank64-authoritative-ranks.jsonl \
  --expected-exact-fingerprint 02c9ed132c902d594ed8528dd4dd0b92a767980e7f6c071c49bc70d4d898a0d1 \
  --approximate-model-dir models/Huihui-Qwen3.8-27B-abliterated-mlx-all-mxfp4 \
  --enforce-promotion-gate \
  --output logs/huihui-rerank64-authoritative-rank-gate.json
```

There are no CLI threshold overrides. Promotion requires exactly K=64, at
least 1,000 authoritative target positions, 100% actual shortlist inclusion,
100% stable-rank recall, no boundary discrepancy, complete heterogeneous-shape
coverage, live origin, privacy-schema conformance, and both explicit artifact
bindings. Legacy paired `.npy` logits remain useful for diagnostics but are
unattested and permanently ineligible for promotion. Synthetic fixtures are
also permanently ineligible regardless of score.
