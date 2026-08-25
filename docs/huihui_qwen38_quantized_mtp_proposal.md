# Huihui Qwen3.8 quantized native-MTP proposal experiment

Date: 2026-08-24

## Scope and correctness boundary

This experiment changes only the speculative proposal model. The served target
remains the existing Huihui Qwen3.8-27B all-MXFP4 checkpoint, and every proposal
is checked by the target-authoritative serial verifier with the ordinary `p/q`
accept/reject correction. The proposal model is lossy; the target verification
path is unchanged. The artifact is explicit opt-in and is never selected by the
generic Huihui alias.

The source target already retains 23 packed `mtp.*` tensors that were superseded
by its released-BF16 MTP sidecar. `runtime.qwen_mtp_quant_clone` constructs a
zero-copy sibling model that restores only those packed mappings. It validates
the exact Qwen3.8 topology, tensor shapes, dtypes, extents, quantization
provenance, target-body mapping hash, and fast-tier alias binding. The clone and
fast-tier alias contain only metadata and symlinks; no checkpoint tensor is
copied.

Packed proposal payload: 225,659,904 bytes, versus 849,398,784 bytes for the
released-BF16 MTP sidecar. The ordinary target body and LM head are identical.

## Reproduction

```sh
.venv/bin/python -m runtime.qwen_mtp_quant_clone plan \
  --source models/Huihui-Qwen3.8-27B-abliterated-mlx-all-mxfp4 \
  --output models/Huihui-Qwen3.8-27B-abliterated-mlx-all-mxfp4-mtpquant

.venv/bin/python -m runtime.qwen_mtp_quant_clone plan-fast \
  --source /Users/kyleboyer/vmodel_fast_tier/Huihui-Qwen3.8-27B-abliterated-mlx-all-mxfp4 \
  --output /Users/kyleboyer/vmodel_fast_tier/Huihui-Qwen3.8-27B-abliterated-mlx-all-mxfp4-mtpquant \
  --target models/Huihui-Qwen3.8-27B-abliterated-mlx-all-mxfp4-mtpquant
```

Use the explicit request model
`lossy-Huihui-Qwen3.8-27B-abliterated-mlx-all-mxfp4-mtpquant`. The ordinary
`lossy-Huihui-Qwen3.8-27B-abliterated` alias continues to select released-BF16
MTP.

## Live result and decision

The live gate replayed the unmodified 178,616-byte captured request: 134 tools,
15,454 system characters, 6,339 rendered input tokens, streaming, and a 16-token
output cap. Both paths produced output SHA-256
`9b170095a170f34a248af480bab274aef28f42ebf1504196bc28cc57e5b87b81`,
accepted 8/8 proposals, used eight target sweeps, and stayed near 3.55 GB peak
Metal.

| Proposal weights | Start available | Wall | Prefill | Decode | Proposal load | Proposal bytes |
|---|---:|---:|---:|---:|---:|---:|
| released BF16 | 8.02 GB | 130.30 s | 63.48 s | 63.09 s | 3.07 s | 6.80 GB |
| MXFP4 Q4/group32 | 8.67 GB | 127.25 s | 57.42 s | 66.09 s | 0.92 s | 1.81 GB |

Decision: **STOP / do not promote**. The apparent 3.05-second wall advantage is
smaller than the run-to-run memory-pressure imbalance, and decode was three
seconds slower despite the cheaper proposal load. Reducing proposal traffic by
4.99 GB did not reduce target sweeps; the exact target verifier remains the
dominant cost. A max-64 promotion run was intentionally skipped because the
candidate did not clear the 10% max-16 wall gate.

The useful follow-up is deeper/multi-token native MTP or a higher-acceptance
sidecar that reduces full target sweeps. Further proposal-only I/O compression
is low priority unless it also preserves acceptance on a heterogeneous replay
corpus.

## Follow-up: native depth two and `q` calibration

Native released-BF16 MTP depth two was tested on the same unmodified max-16
capture with 8.45 GB initially available. It preserved the output hash and all
target/KV/KDA accounting, but accepted only 1/6 second-step proposals. The
request still needed eight target sweeps and finished in 128.47 seconds
(57.38 seconds prefill, 67.53 seconds decode). Depth two is therefore also a
STOP and depth one remains the default.

Existing sparse proposal replays were then evaluated offline across disjoint
request shapes rather than tuning on the live benchmark:

- developer-action calibration -> short-direct validation selected flat top-1;
- short-direct calibration -> 134-tool validation selected temperature/top-16
  in calibration but lost 3.89 acceptance points versus flat top-1 on the
  held-out capture;
- 134-tool calibration -> developer-action validation selected flat top-1.

No alternate `q` policy generalized across these shapes, so the runtime keeps
flat top-1. The failed held-out result is evidence against enabling a
capture-tuned sampling policy, not a reason to expand the tuning corpus until a
preferred answer appears.
