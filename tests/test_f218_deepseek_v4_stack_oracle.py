"""F218: full-stack oracle -- the reference Transformer vs. our block stack.

Thirteen seams have now been verified individually and generation still
degenerates, so this stops testing seams and bisects instead: build the
released Transformer at small scale with random weights, run our stack on the
same weights, and compare AFTER EVERY LAYER. The first layer that diverges is
the fault, with no hypothesis required.

Layer ratios cover all three released regimes (0, 4, 128-stand-in) so a
compression-specific fault cannot hide.
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

DIM, HEADS, HEAD_DIM, ROPE_DIM = 64, 4, 16, 8
Q_RANK, O_RANK, GROUPS, WINDOW = 32, 16, 2, 8
EXPERTS, TOPK, INTER, HC = 8, 3, 16, 4
RATIOS = [0, 4, 0, 4]


def _install():
    import torch

    from tests.test_f216_deepseek_v4_attention_oracle import _install_stubs

    reference = _install_stubs()

    def sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20,
                 eps=1e-6):
        hc = hc_mult
        pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
        post = 2 * torch.sigmoid(
            mixes[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
        comb = mixes[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]
        comb = comb.reshape(*comb.shape[:-1], hc, hc)
        comb = comb.softmax(dim=-1) + eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
        for _ in range(sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
        return pre, post, comb

    sys.modules["kernel"].hc_split_sinkhorn = sinkhorn
    reference.hc_split_sinkhorn = sinkhorn

    def rotate_activation(x):
        """Dense Hadamard, dtype-agnostic.

        The released helper asserts bfloat16 and imports
        fast_hadamard_transform, which is not installed. F208 verified our
        butterfly against a dense Sylvester matrix, so the dense form is used
        here to keep the oracle in float32.
        """
        n = x.shape[-1]
        matrix = torch.ones((1, 1), dtype=torch.float32)
        while matrix.shape[0] < n:
            matrix = torch.cat([
                torch.cat([matrix, matrix], dim=1),
                torch.cat([matrix, -matrix], dim=1)], dim=0)
        return (x.float() @ matrix) * (n ** -0.5)

    def fp4_act_quant(x, block_size=32, inplace=False):
        return x  # QAT round-trip isolated; F209 covers it separately

    reference.rotate_activation = rotate_activation
    reference.fp4_act_quant = fp4_act_quant
    sys.modules["kernel"].fp4_act_quant = fp4_act_quant
    return reference


def _args(reference):
    args = reference.ModelArgs()
    args.dim, args.n_heads, args.head_dim = DIM, HEADS, HEAD_DIM
    args.rope_head_dim, args.q_lora_rank = ROPE_DIM, Q_RANK
    args.o_lora_rank, args.o_groups = O_RANK, GROUPS
    args.window_size, args.max_seq_len, args.max_batch_size = WINDOW, 64, 1
    args.norm_eps = args.hc_eps = 1e-6
    args.rope_theta, args.compress_rope_theta = 10000.0, 40000.0
    args.original_seq_len, args.rope_factor = 0, 40
    args.compress_ratios = RATIOS
    args.n_routed_experts, args.n_activated_experts = EXPERTS, TOPK
    args.n_shared_experts, args.moe_inter_dim = 1, INTER
    args.n_hash_layers, args.swiglu_limit = 0, 10.0
    args.score_func, args.route_scale = "sqrtsoftplus", 1.0
    args.hc_mult, args.hc_sinkhorn_iters = HC, 20
    return args


def _randomize(module):
    import torch

    for name, param in list(module.named_parameters(recurse=True)):
        holder = module
        parts = name.split(".")
        for part in parts[:-1]:
            holder = getattr(holder, part)
        shape = param.shape
        fill = (torch.randn(*shape, dtype=torch.float32) * 0.15
                if param.dim() > 1
                else torch.randn(*shape, dtype=torch.float32) * 0.2 + 0.5)
        setattr(holder, parts[-1], torch.nn.Parameter(fill))


def test_stack_matches_layer_by_layer():
    import mlx.core as mx
    import torch

    from runtime.deepseek_v4 import (apply_rope_interleaved, compress_prefill,
                                     deepseek_v4_attention, expert_swiglu,
                                     gather_indices, hc_head, moe_combine,
                                     moe_gate, run_deepseek_v4_block,
                                     yarn_freqs)

    reference = _install()
    args = _args(reference)
    torch.manual_seed(0)
    with reference.set_dtype(torch.float32):
        blocks = [reference.Block(i, args) for i in range(len(RATIOS))]
    for block in blocks:
        _randomize(block)
        block.attn.indexer = None

    seqlen = 12
    rng = np.random.default_rng(1)
    hidden = (rng.normal(size=(1, seqlen, DIM)) * 0.3).astype(np.float32)

    ref = torch.tensor(hidden).unsqueeze(2).repeat(1, 1, HC, 1)
    ours = mx.broadcast_to(mx.array(hidden)[:, :, None, :],
                           (1, seqlen, HC, DIM))

    def np_of(t):
        return t.detach().numpy() if hasattr(t, "detach") else np.array(t)

    for index, block in enumerate(blocks):
        ratio = RATIOS[index]
        ref = block(ref, 0, torch.zeros(1, seqlen, dtype=torch.long))

        theta = 40000.0 if ratio else 10000.0
        factor = 40.0 if ratio else 1.0
        cos, sin = yarn_freqs(ROPE_DIM, seqlen, 0, theta, factor, 32, 1)
        kvw = mx.array(block.attn.wkv.weight.detach().numpy())
        kvn = mx.array(block.attn.kv_norm.weight.detach().numpy())
        aw = {f"a.{k}": mx.array(getattr(block.attn, v).weight.detach().numpy()
                                 if k != "attn_sink"
                                 else block.attn.attn_sink.detach().numpy())
              for k, v in (("wq_a", "wq_a"), ("wq_b", "wq_b"),
                           ("q_norm", "q_norm"), ("wo_a", "wo_a"),
                           ("wo_b", "wo_b"), ("attn_sink", "attn_sink"))}

        def attention(t, _ratio=ratio, _cos=cos, _sin=sin, _kvw=kvw,
                      _kvn=kvn, _aw=aw, _block=block):
            lat = mx.fast.rms_norm(t @ _kvw.T, _kvn, 1e-6)
            lat = mx.concatenate(
                [lat[..., :-ROPE_DIM],
                 apply_rope_interleaved(lat[..., -ROPE_DIM:], _cos, _sin)],
                axis=-1)
            kv_all, offset = lat, lat.shape[1]
            if _ratio:
                pooled, _rem = compress_prefill(
                    t, mx.array(_block.attn.compressor.wkv.weight.detach().numpy()),
                    mx.array(_block.attn.compressor.wgate.weight.detach().numpy()),
                    mx.array(_block.attn.compressor.ape.detach().numpy()),
                    mx.array(_block.attn.compressor.norm.weight.detach().numpy()),
                    ratio=_ratio, head_dim=HEAD_DIM, norm_eps=1e-6)
                if pooled is not None:
                    stride = mx.arange(pooled.shape[1]) * _ratio
                    pooled = mx.concatenate(
                        [pooled[..., :-ROPE_DIM],
                         apply_rope_interleaved(pooled[..., -ROPE_DIM:],
                                                _cos[stride], _sin[stride])],
                        axis=-1)
                    kv_all = mx.concatenate([kv_all, pooled], axis=1)
            return deepseek_v4_attention(
                t, _aw, "a", heads=HEADS, head_dim=HEAD_DIM,
                rope_head_dim=ROPE_DIM, q_lora_rank=Q_RANK,
                o_lora_rank=O_RANK, n_groups=GROUPS, norm_eps=1e-6,
                cos=_cos, sin=_sin, kv_all=kv_all,
                topk_idxs=gather_indices(WINDOW, _ratio, seqlen, 0, offset))

        def ffn(t, _block=block):
            flat = t.reshape(-1, t.shape[-1])
            weights, indices = moe_gate(
                flat, mx.array(_block.ffn.gate.weight.detach().numpy()),
                mx.array(_block.ffn.gate.bias.detach().numpy()),
                topk=TOPK, score_func="sqrtsoftplus")

            def routed(expert, rows, scale):
                e = _block.ffn.experts[expert]
                return expert_swiglu(
                    rows, mx.array(e.w1.weight.detach().numpy()),
                    mx.array(e.w2.weight.detach().numpy()),
                    mx.array(e.w3.weight.detach().numpy()),
                    swiglu_limit=10.0, weights=scale)

            def shared(rows):
                e = _block.ffn.shared_experts
                return expert_swiglu(
                    rows, mx.array(e.w1.weight.detach().numpy()),
                    mx.array(e.w2.weight.detach().numpy()),
                    mx.array(e.w3.weight.detach().numpy()), swiglu_limit=10.0)

            return moe_combine(t, routed, weights, indices, shared,
                               n_routed_experts=EXPERTS)

        ours = run_deepseek_v4_block(
            ours,
            {k: mx.array(getattr(block, f"hc_{k}").detach().numpy())
             for k in ("attn_fn", "attn_scale", "attn_base",
                       "ffn_fn", "ffn_scale", "ffn_base")},
            {"attn": mx.array(block.attn_norm.weight.detach().numpy()),
             "ffn": mx.array(block.ffn_norm.weight.detach().numpy())},
            attention, ffn, hc_mult=HC, norm_eps=1e-6, sinkhorn_iters=20,
            hc_eps=1e-6)

        a, b = np_of(ours), np_of(ref)
        diff = np.abs(a - b).max()
        scale = max(np.abs(b).max(), 1e-6)
        assert diff / scale < 1e-2, (
            f"stack diverged at layer {index} (ratio {ratio}): "
            f"max abs {diff}, relative {diff/scale:.5f}")


def test_indexer_materially_changes_a_ratio_four_layer():
    """The stack test disables the reference's Indexer to match ours.

    That makes the one known deviation invisible to it, so this measures the
    deviation directly: same layer, same weights, same input, Indexer on versus
    off. If the two agree the unwired Indexer is harmless; if they differ it is
    the remaining candidate for degenerate generation over 21 of 43 layers.
    """
    import torch

    reference = _install()
    args = _args(reference)
    torch.manual_seed(5)
    with reference.set_dtype(torch.float32):
        # Layer 1 has compress_ratio 4, the only regime that builds an Indexer.
        attention = reference.Attention(1, args)
    _randomize(attention)

    seqlen = 16
    rng = np.random.default_rng(6)
    x = torch.tensor((rng.normal(size=(1, seqlen, DIM)) * 0.3
                      ).astype(np.float32))

    saved = attention.indexer
    assert saved is not None, "ratio 4 must construct an Indexer"

    attention.kv_cache.zero_()
    attention.compressor.kv_cache = None
    with_indexer = attention(x, 0).detach().numpy()

    attention.indexer = None
    attention.kv_cache.zero_()
    attention.compressor.kv_cache = None
    without_indexer = attention(x, 0).detach().numpy()

    diff = np.abs(with_indexer - without_indexer).max()
    scale = max(np.abs(with_indexer).max(), 1e-6)

    # MEASURED: identical. index_topk is 512 while a prompt of `seqlen` yields
    # only seqlen // ratio compressed entries, so top-k selects ALL of them and
    # the learned scores never bind. The Indexer can only change behaviour once
    # seqlen // 4 > 512, i.e. beyond roughly 2048 tokens.
    #
    # This eliminates the unwired Indexer as an explanation for degenerate
    # generation on short prompts, which is what it had been assumed to be.
    assert diff / scale < 1e-5, (
        f"Indexer now changes a short-prompt layer (relative {diff/scale:.5f}); "
        "the selection-is-inert reasoning above no longer holds")
