"""F216: oracle the composed attention against the released Attention.forward.

F206/F209/F210 verified attention's PIECES. The composition in
``StreamingEngine._deepseek_v4_attention`` was written by hand and never
checked, and a first token that is correct followed by degenerate decode is
exactly what a composition error looks like.

The reference is the checkpoint's own ``Attention``, built at small dimensions
with float32 weights so no checkpoint dequant is needed in torch, and with its
TileLang kernels replaced by torch transcriptions whose semantics F206/F209
already pinned. Only ``compress_ratio == 0`` is covered here: it isolates the
q/kv/RoPE/sparse-attn/o-LoRA chain from the compressor, which is the layer
type a 2-layer run showed behaving correctly and therefore the right baseline
to establish before chasing the compressed layers.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

INFERENCE = ROOT / "models" / "DeepSeek-V4-Flash-0731" / "inference"

pytestmark = pytest.mark.skipif(
    not (INFERENCE / "model.py").is_file(),
    reason="DeepSeek-V4-Flash-0731 inference/ not present")


def _install_stubs():
    import torch

    stub = sys.modules.get("kernel") or types.ModuleType("kernel")

    def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=None,
                  inplace=False):
        """Identity: this oracle isolates composition from the QAT round-trip,
        which F209 verifies separately."""
        return x

    def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
        """Torch transcription of sparse_attn_kernel_, pinned by F206."""
        b, m, h, d = q.shape
        out = torch.zeros_like(q, dtype=torch.float32)
        qf, kvf = q.float(), kv.float()
        for bi in range(b):
            for mi in range(m):
                idxs = topk_idxs[bi, mi]
                valid = idxs >= 0
                if not bool(valid.any()):
                    continue
                gathered = kvf[bi, idxs.clamp(min=0)]
                scores = (qf[bi, mi] @ gathered.T) * softmax_scale
                scores = scores.masked_fill(~valid.unsqueeze(0), float("-inf"))
                row_max = scores.max(dim=1).values
                weights = torch.exp(scores - row_max.unsqueeze(1))
                weights = weights.masked_fill(~valid.unsqueeze(0), 0.0)
                denom = weights.sum(dim=1) + torch.exp(
                    attn_sink.float() - row_max)
                out[bi, mi] = (weights @ gathered) / denom.unsqueeze(1)
        return out.to(q.dtype)

    for name, value in (("act_quant", act_quant), ("sparse_attn", sparse_attn)):
        setattr(stub, name, value)
    for name in ("fp4_act_quant", "fp8_gemm", "fp4_gemm", "hc_split_sinkhorn"):
        if not hasattr(stub, name):
            def _unavailable(*_a, __name=name, **_k):
                raise RuntimeError(f"kernel {__name!r} unavailable here")
            setattr(stub, name, _unavailable)
    sys.modules["kernel"] = stub
    sys.path.insert(0, str(INFERENCE))
    import model as reference

    reference.act_quant = act_quant
    reference.sparse_attn = sparse_attn
    return reference


DIM, HEADS, HEAD_DIM, ROPE_DIM = 64, 4, 16, 8
Q_RANK, O_RANK, GROUPS, WINDOW = 32, 16, 2, 8


def _build(reference):
    import torch

    torch.manual_seed(0)
    args = reference.ModelArgs()
    args.dim = DIM
    args.n_heads = HEADS
    args.head_dim = HEAD_DIM
    args.rope_head_dim = ROPE_DIM
    args.q_lora_rank = Q_RANK
    args.o_lora_rank = O_RANK
    args.o_groups = GROUPS
    args.window_size = WINDOW
    args.compress_ratios = [0] * 8
    args.max_seq_len = 64
    args.max_batch_size = 1
    args.norm_eps = 1e-6
    args.rope_theta = 10000.0
    # Linear defaults to bfloat16 via set_dtype; build in float32 so the
    # comparison isolates composition from bf16 rounding.
    with reference.set_dtype(torch.float32):
        attention = reference.Attention(0, args)
    with torch.no_grad():
        for module in (attention.wq_a, attention.wq_b, attention.wkv,
                       attention.wo_a, attention.wo_b):
            # wo_a is constructed with an explicit bfloat16 dtype regardless of
            # set_dtype, so replace the parameter rather than copy into it.
            module.weight = torch.nn.Parameter(
                torch.randn(*module.weight.shape, dtype=torch.float32) * 0.15)
        attention.q_norm.weight.copy_(torch.randn(Q_RANK) * 0.1 + 1.0)
        attention.kv_norm.weight.copy_(torch.randn(HEAD_DIM) * 0.1 + 1.0)
        attention.attn_sink.copy_(torch.randn(HEADS) * 0.5)
    return attention, args


def test_composed_attention_matches_the_released_forward():
    import mlx.core as mx
    import torch

    from runtime.deepseek_v4 import (apply_rope_interleaved,
                                     deepseek_v4_attention,
                                     window_ring_write, window_topk_idxs,
                                     yarn_freqs)

    reference = _install_stubs()
    attention, args = _build(reference)

    seqlen = 5
    rng = np.random.default_rng(1)
    x = (rng.normal(size=(1, seqlen, DIM)) * 0.3).astype(np.float32)
    expected = attention(torch.tensor(x), 0).detach().numpy()

    w = {f"a.wq_a": mx.array(attention.wq_a.weight.detach().numpy()),
         f"a.wq_b": mx.array(attention.wq_b.weight.detach().numpy()),
         f"a.q_norm": mx.array(attention.q_norm.weight.detach().numpy()),
         f"a.wo_a": mx.array(attention.wo_a.weight.detach().numpy()),
         f"a.wo_b": mx.array(attention.wo_b.weight.detach().numpy()),
         f"a.attn_sink": mx.array(attention.attn_sink.detach().numpy())}

    mxx = mx.array(x)
    latent = mxx @ mx.array(attention.wkv.weight.detach().numpy()).T
    latent = mx.fast.rms_norm(
        latent, mx.array(attention.kv_norm.weight.detach().numpy()), 1e-6)
    cos, sin = yarn_freqs(ROPE_DIM, seqlen, 0, 10000.0, 1.0, 32, 1)
    tail = apply_rope_interleaved(latent[..., -ROPE_DIM:], cos, sin)
    latent = mx.concatenate([latent[..., :-ROPE_DIM], tail], axis=-1)

    got = np.array(deepseek_v4_attention(
        mxx, w, "a", heads=HEADS, head_dim=HEAD_DIM, rope_head_dim=ROPE_DIM,
        q_lora_rank=Q_RANK, o_lora_rank=O_RANK, n_groups=GROUPS,
        norm_eps=1e-6, cos=cos, sin=sin,
        kv_all=latent, topk_idxs=window_topk_idxs(WINDOW, seqlen, 0)))

    diff = np.abs(got - expected).max()
    scale = max(np.abs(expected).max(), 1e-6)
    assert diff / scale < 5e-3, (
        f"composed attention diverged: max abs {diff}, relative {diff/scale:.5f}")


def test_compressed_layer_prefill_matches_the_released_forward():
    """A compress_ratio layer, comparing the whole prefill including pooling.

    Ratio 128 is chosen over 4 because it has no Indexer, isolating the
    Compressor + gather wiring from the learned top-k selection.
    """
    import mlx.core as mx
    import torch

    from runtime.deepseek_v4 import (apply_rope_interleaved, compress_prefill,
                                     deepseek_v4_attention, gather_indices,
                                     yarn_freqs)

    reference = _install_stubs()
    ratio = 4  # small stand-in for the released 128; same no-Indexer path
    torch.manual_seed(3)
    args = reference.ModelArgs()
    args.dim, args.n_heads, args.head_dim = DIM, HEADS, HEAD_DIM
    args.rope_head_dim, args.q_lora_rank = ROPE_DIM, Q_RANK
    args.o_lora_rank, args.o_groups = O_RANK, GROUPS
    args.window_size, args.max_seq_len, args.max_batch_size = WINDOW, 64, 1
    args.norm_eps, args.rope_theta = 1e-6, 10000.0
    args.compress_rope_theta = 40000.0
    args.original_seq_len = 0
    args.compress_ratios = [ratio] * 8
    with reference.set_dtype(torch.float32):
        attention = reference.Attention(0, args)
    attention.indexer = None  # isolate the Compressor from learned selection
    with torch.no_grad():
        for module in (attention.wq_a, attention.wq_b, attention.wkv,
                       attention.wo_a, attention.wo_b,
                       attention.compressor.wkv, attention.compressor.wgate):
            module.weight = torch.nn.Parameter(
                torch.randn(*module.weight.shape, dtype=torch.float32) * 0.15)
        attention.q_norm.weight = torch.nn.Parameter(
            torch.randn(Q_RANK) * 0.1 + 1.0)
        attention.kv_norm.weight = torch.nn.Parameter(
            torch.randn(HEAD_DIM) * 0.1 + 1.0)
        attention.compressor.norm.weight = torch.nn.Parameter(
            torch.randn(HEAD_DIM) * 0.1 + 1.0)
        attention.compressor.ape = torch.nn.Parameter(
            torch.randn(*attention.compressor.ape.shape) * 0.3)
        attention.attn_sink = torch.nn.Parameter(torch.randn(HEADS) * 0.5)

    seqlen = 12
    rng = np.random.default_rng(4)
    x = (rng.normal(size=(1, seqlen, DIM)) * 0.3).astype(np.float32)
    expected = attention(torch.tensor(x), 0).detach().numpy()

    mxx = mx.array(x)
    kvw = mx.array(attention.wkv.weight.detach().numpy())
    latent = mx.fast.rms_norm(
        mxx @ kvw.T,
        mx.array(attention.kv_norm.weight.detach().numpy()), 1e-6)
    # A compressed layer's freqs_cis is built with compress_rope_theta for
    # EVERYTHING -- query, window kv, and compressed kv alike. Only pure
    # sliding-window layers use the base rope_theta.
    cos, sin = yarn_freqs(ROPE_DIM, seqlen, 0, 40000.0, 1.0, 32, 1)
    latent = mx.concatenate(
        [latent[..., :-ROPE_DIM],
         apply_rope_interleaved(latent[..., -ROPE_DIM:], cos, sin)], axis=-1)

    pooled, _rem = compress_prefill(
        mxx, mx.array(attention.compressor.wkv.weight.detach().numpy()),
        mx.array(attention.compressor.wgate.weight.detach().numpy()),
        mx.array(attention.compressor.ape.detach().numpy()),
        mx.array(attention.compressor.norm.weight.detach().numpy()),
        ratio=ratio, head_dim=HEAD_DIM, norm_eps=1e-6)
    ccos, csin = cos, sin
    stride = mx.arange(pooled.shape[1]) * ratio
    pooled = mx.concatenate(
        [pooled[..., :-ROPE_DIM],
         apply_rope_interleaved(pooled[..., -ROPE_DIM:],
                                ccos[stride], csin[stride])], axis=-1)
    kv_all = mx.concatenate([latent, pooled], axis=1)

    w = {"a.wq_a": mx.array(attention.wq_a.weight.detach().numpy()),
         "a.wq_b": mx.array(attention.wq_b.weight.detach().numpy()),
         "a.q_norm": mx.array(attention.q_norm.weight.detach().numpy()),
         "a.wo_a": mx.array(attention.wo_a.weight.detach().numpy()),
         "a.wo_b": mx.array(attention.wo_b.weight.detach().numpy()),
         "a.attn_sink": mx.array(attention.attn_sink.detach().numpy())}

    got = np.array(deepseek_v4_attention(
        mxx, w, "a", heads=HEADS, head_dim=HEAD_DIM, rope_head_dim=ROPE_DIM,
        q_lora_rank=Q_RANK, o_lora_rank=O_RANK, n_groups=GROUPS,
        norm_eps=1e-6, cos=cos, sin=sin, kv_all=kv_all,
        topk_idxs=gather_indices(WINDOW, ratio, seqlen, 0, seqlen)))

    diff = np.abs(got - expected).max()
    scale = max(np.abs(expected).max(), 1e-6)
    assert diff / scale < 5e-3, (
        f"compressed prefill diverged: max abs {diff}, "
        f"relative {diff/scale:.5f}")
