"""F33 harness, milestone 2c: verify runtime/glm.py's COMPACT sparse attention
output (gather selected latent rows, dense-attend over just that compact set)
numerically matches HF's real GlmMoeDsaAttention at S > index_topk, closing
the "sparse attention output, not only membership" gap named in STATUS.md's
"Current truth" section.

Builds on milestone 2a (tests/test_f33_mla_attention.py: dense MLA attention
matches HF, using index_topk >= S to keep the indexer a no-op) and milestone
2b (tests/test_f33_dsa_indexer.py: the top-k SELECTION SET matches HF at
S > index_topk, but only compared as a `set()` -- order discarded, and only
the indexer's own output, not the downstream attention computation).

The key subtlety this test actually exercises: HF's eager/SDPA backend does
NOT gather only the selected keys -- it computes attention over the FULL
causal history and additively masks non-selected positions to -inf (see
`GlmMoeDsaAttention.forward()`: `attention_mask.masked_fill(index_mask, -inf)`,
with the actual gathered/compact path reserved for the `flash-mla` kernel,
per that method's own comment "consumed by flash_mla_with_kvcache; ignored by
eager/SDPA"). This runtime's F21/F22 path instead genuinely GATHERS the
selected rows (`mx.take(lat_all[0], sel[0, 0], axis=0)`) and computes a
SMALLER dense attention over just the topk set -- mathematically equivalent
in exact arithmetic (masked positions contribute exp(-inf)=0 either way),
but floating-point summation is not associative, so the two code paths sum
in different orders and could in principle diverge more than float noise if
something else were wrong. `runtime/glm_dsa.py`'s `update_and_select` already
sorts the selection back into chronological (ascending-position) order
specifically so its reduction order matches HF's natural 0..S-1 iteration
order (see the comment there) -- this test is the first time that claim is
checked against a real numeric comparison, not just reasoned about.

Empirically confirmed (2026-07-14): max abs diff 3.58e-7 between HF's
masked-dense and this runtime's gathered-compact attention output at the
decode step where S=7 > index_topk=4 (real sparsity), using the same tiny
synthetic GlmMoeDsaConfig family as milestones 2a/2b -- float32-noise scale,
comfortably inside the 1e-4 gate this project's other F33 milestones use.

Honesty note on the chronological-sort claim specifically: an earlier version
of this file also tried to empirically demonstrate that skipping glm_dsa.py's
`mx.sort` (using raw argpartition order instead) measurably changes the
output. It doesn't, at any scale tried here (topk=4 through topk=64): the
sorted-vs-unsorted delta is ~3e-8-4e-8, indistinguishable from float32 noise,
and not even directionally consistent (unsorted was occasionally CLOSER to
HF than sorted, by noise). That ablation is not included -- asserting it
would have overstated what was actually measured. The sort may still matter
at real GLM scale (topk=2048, thousands of summed terms, where float
non-associativity compounds differently), but this synthetic harness cannot
demonstrate that scale, so the sort's necessity here rests on the source
comment's reasoning, not an independent measurement.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx
import numpy as np
import pytest
import torch

from runtime.config import ModelConfig
from runtime.glm import _glm_mlp_residual, _mla_attention
from runtime.glm_dsa import DSAState
from runtime.kv_cache import KVCache

transformers = pytest.importorskip("transformers")
from transformers import GlmMoeDsaConfig  # noqa: E402
import transformers.models.glm_moe_dsa.modeling_glm_moe_dsa as hf_mod  # noqa: E402

HIDDEN = 48
N_HEADS = 4
DN, DR, DV = 8, 8, 16  # qk_nope, qk_rope, v_head
Q_LORA = 32
KV_LORA = 16
INDEX_N_HEADS = 32   # hardcoded in runtime/glm_dsa.py -- not configurable
INDEX_HEAD_DIM = 128  # hardcoded in runtime/glm_dsa.py -- not configurable
INDEX_TOPK = 4
S = 7  # > INDEX_TOPK, so real sparsity engages at the last (decode) row
ROPE_THETA = 10000.0

ATTN_WEIGHT_NAMES = [
    "q_a_proj.weight", "q_a_layernorm.weight", "q_b_proj.weight",
    "kv_a_proj_with_mqa.weight", "kv_a_layernorm.weight", "kv_b_proj.weight",
    "o_proj.weight",
]
INDEXER_WEIGHT_NAMES = [
    "indexer.wq_b.weight", "indexer.wk.weight", "indexer.k_norm.weight",
    "indexer.k_norm.bias", "indexer.weights_proj.weight",
]


@pytest.mark.parametrize("host_spool", [False, True])
def test_glm_layer_stationary_scheduler_preselects_before_attention(
        monkeypatch, host_spool):
    """Guard the engine integration, not only DSAState's isolated helper."""
    from runtime.engine import StreamingEngine
    import runtime.glm as glm_mod

    events = []

    class FakeDSA:
        selection_query_tile_size = 4

        def preselect_full_layer(
                self, layer, hidden, weights, prefix, offset,
                *, attention_tile_width):
            events.append((
                "preselect", layer, int(hidden.shape[1]), offset,
                attention_tile_width))

        def clear_selections(self):
            events.append(("clear",))

    class FakeCache:
        def contains(self, _key):
            return True

        def get(self, _key, _names):
            return {"sentinel": mx.array([1], dtype=mx.int32)}

    def fake_attention(x, _w, _prefix, _cfg, _kv, layer, offset, **_kwargs):
        assert events and events[0][0] == "preselect"
        events.append(("attention", layer, offset, int(x.shape[1])))
        return x

    def fake_mlp(x, *_args, **_kwargs):
        events.append(("mlp", int(x.shape[1])))
        return x

    monkeypatch.setattr(glm_mod, "_glm_attention_residual", fake_attention)
    monkeypatch.setattr(glm_mod, "_glm_mlp_residual", fake_mlp)

    engine = StreamingEngine.__new__(StreamingEngine)
    engine.cfg = SimpleNamespace(
        num_hidden_layers=1,
        model_type="glm_moe_dsa",
        indexer_types=["full"],
        mlp_layer_types=["dense"],
        first_k_dense_replace=1,
    )
    engine.rc = SimpleNamespace(
        prefetch_depth=0,
        glm_dsa_mla_kv_spill_dir="",
        glm_dsa_dense_mlp_tile_size=4,
        glm_dsa_index_preallocate=False,
        glm53_layer_stationary_host_spool=host_spool,
        metal_limit_mb=8500,
    )
    engine.prefetcher = None
    engine.cache = FakeCache()
    engine.governor = None
    engine._dsv4_packed_trunk = False
    engine._prefill_layer_transient_by_positions = {}
    engine._decode_layer_transient = 0
    engine._request_profiler = None
    engine.timer = SimpleNamespace(add=lambda *_args: None)
    engine._get_experts = lambda *_args, **_kwargs: {}
    engine._iter_expert_batches = lambda *_args, **_kwargs: iter(())
    engine._layer_key = lambda layer: f"layer.{layer}"
    engine._layer_names = lambda _layer: []
    engine._select_layer_transient = lambda *_args: 0
    engine._record_layer_transient = lambda *_args: 0
    engine._restore_aggregate_layer_transient = lambda *_args: 0
    engine._note_true_peak = lambda: None

    kv = SimpleNamespace(dsa=FakeDSA(), latent_spill_enabled=False)
    x = mx.zeros(
        (1, 6, 3),
        dtype=mx.bfloat16 if host_spool else mx.float32)
    out = engine._layer_stationary_glm_sweep(
        x, kv, offset=7, tile_width=2)
    mx.eval(out)

    if host_spool:
        assert np.array_equal(
            np.asarray(out.view(mx.uint16)),
            np.zeros((1, 6, 3), dtype=np.uint16))
        stats = engine._glm53_layer_stationary_stats
        assert stats["host_spool"] == 1
        assert stats["host_spool_h2d_bytes"] > 0
        assert stats["host_spool_d2h_bytes"] > 0

    assert events == [
        ("preselect", 0, 6, 7, 2),
        ("attention", 0, 7, 2),
        ("attention", 0, 9, 2),
        ("mlp", 4),
        ("attention", 0, 11, 2),
        ("mlp", 2),
        ("clear",),
    ]


def test_glm_layer_stationary_reuses_proven_signature_without_extra_margin(
        monkeypatch):
    """A recurring GLM shape retains the governor floor, not a second pad."""
    from runtime.engine import StreamingEngine
    import runtime.glm as glm_mod

    reservations = []

    class FakeCache:
        def contains(self, _key):
            return True

        def get(self, _key, _names):
            return {"sentinel": mx.array([1], dtype=mx.int32)}

    class FakeGovernor:
        def reserve(self, incoming, margin=400_000_000, *, reason=""):
            reservations.append((int(incoming), int(margin), reason))

    monkeypatch.setattr(
        glm_mod, "_glm_attention_residual",
        lambda x, *_args, **_kwargs: x)
    monkeypatch.setattr(
        glm_mod, "_glm_mlp_residual",
        lambda x, *_args, **_kwargs: x)

    engine = StreamingEngine.__new__(StreamingEngine)
    engine.cfg = SimpleNamespace(
        num_hidden_layers=1,
        model_type="glm_moe_dsa",
        indexer_types=["shared"],
        mlp_layer_types=["dense"],
        first_k_dense_replace=1,
        num_experts=256,
        layer_types=(),
        kda_layers=(),
        full_attn_layers=(),
    )
    engine.rc = SimpleNamespace(
        prefetch_depth=0,
        glm_dsa_mla_kv_spill_dir="",
        glm_dsa_dense_mlp_tile_size=0,
        glm_dsa_index_preallocate=False,
        glm53_layer_stationary_host_spool=False,
        metal_limit_mb=8500,
    )
    engine.prefetcher = None
    engine.cache = FakeCache()
    engine.governor = FakeGovernor()
    engine._dsv4_packed_trunk = False
    engine._prefill_layer_transient_by_positions = {6: 123}
    engine._decode_layer_transient = 0
    engine._layer_transient = 0
    engine._layer_transient_margin = 0
    signature = "mla_shared+dense"
    engine._layer_transient_by_signature = {(6, signature): 123}
    engine._layer_transient_observation_counts = {(6, signature): 1}
    engine._request_profiler = None
    engine.timer = SimpleNamespace(add=lambda *_args: None)
    engine._get_experts = lambda *_args, **_kwargs: {}
    engine._iter_expert_batches = lambda *_args, **_kwargs: iter(())
    engine._layer_key = lambda layer: f"layer.{layer}"
    engine._layer_names = lambda _layer: []
    engine._record_layer_transient = lambda *_args: 0
    engine._restore_aggregate_layer_transient = lambda *_args: 0
    engine._note_true_peak = lambda: None

    out = engine._layer_stationary_glm_sweep(
        mx.zeros((1, 6, 3), dtype=mx.float32),
        SimpleNamespace(dsa=None, latent_spill_enabled=False),
        offset=0,
        tile_width=2,
    )
    mx.eval(out)

    assert reservations == [
        (123, 0, "glm53-full-attention-transient"),
        (123, 0, "glm53-full-attention-transient"),
        (123, 0, "glm53-full-attention-transient"),
    ]
    stats = engine._glm53_layer_stationary_stats
    assert stats["transient_reservation_calls"] == 3
    assert stats["transient_reservation_bytes"] == 369
    assert stats["transient_reservation_margin_bytes"] == 0
    assert stats["transient_reservation_first_margin_calls"] == 0
    assert stats["transient_reservation_recurring_calls"] == 3
    assert stats["transient_reservation_s"] >= 0.0


def _build_hf_attention(seed: int):
    torch.manual_seed(seed)
    hf_cfg = GlmMoeDsaConfig(
        vocab_size=64, hidden_size=HIDDEN, intermediate_size=64,
        moe_intermediate_size=16, num_hidden_layers=1, first_k_dense_replace=0,
        num_attention_heads=N_HEADS, num_key_value_heads=N_HEADS,
        n_shared_experts=1, n_routed_experts=8, num_experts_per_tok=2,
        kv_lora_rank=KV_LORA, q_lora_rank=Q_LORA,
        qk_rope_head_dim=DR, qk_nope_head_dim=DN, v_head_dim=DV,
        index_topk=INDEX_TOPK, index_n_heads=INDEX_N_HEADS, index_head_dim=INDEX_HEAD_DIM,
        indexer_types=["full"], attn_implementation="eager",
        rope_theta=ROPE_THETA,
    )
    attn = hf_mod.GlmMoeDsaAttention(hf_cfg, layer_idx=0)
    attn.eval()
    with torch.no_grad():
        for p in attn.parameters():
            p.normal_(mean=0.0, std=0.3)
    rope = hf_mod.GlmMoeDsaRotaryEmbedding(hf_cfg)
    return hf_cfg, attn, rope


def _runtime_config(hf_cfg) -> ModelConfig:
    return ModelConfig(
        model_type="glm_moe_dsa", hidden_size=HIDDEN, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=N_HEADS, num_key_value_heads=N_HEADS,
        vocab_size=64, rms_norm_eps=hf_cfg.rms_norm_eps, rope_theta=ROPE_THETA,
        max_position_embeddings=128, tie_word_embeddings=False, attention_bias=False,
        head_dim=DN + DR, eos_token_ids=(0,), torch_dtype="float32",
        qk_nope_head_dim=DN, qk_rope_head_dim=DR, v_head_dim=DV,
        # F92: _mla_attention now branches on q_lora_rank (0 -> single q_proj,
        # Kimi Linear's shape; nonzero -> q_a/q_b lora split, GLM's shape).
        # This fixture's weight dict is q_a/q_b-shaped (matches hf_cfg's
        # Q_LORA), so q_lora_rank must be set here too.
        q_lora_rank=Q_LORA,
        rope_interleave=True, index_topk=INDEX_TOPK, indexer_types=("full",),
    )


def _weights_from_hf(attn) -> dict:
    sd = attn.state_dict()
    w = {}
    for name in ATTN_WEIGHT_NAMES + INDEXER_WEIGHT_NAMES:
        w[f"layer0.self_attn.{name}"] = mx.array(sd[name].numpy())
    return w


def _runtime_last_row_output(w, cfg, h_mx) -> np.ndarray:
    kv = KVCache(num_layers=1)
    kv.compressed_mla = True
    kv.dsa = DSAState(cfg, key_tile_size=2)
    _ = _mla_attention(h_mx[:, : S - 1], w, "layer0", cfg, kv, layer=0, offset=0)
    out = _mla_attention(h_mx[:, S - 1 :], w, "layer0", cfg, kv, layer=0, offset=S - 1)
    mx.eval(out)
    return np.array(out)


def _runtime_last_tile_output(
    w, cfg, h_mx, tile_start: int, *, absorbed: bool = False,
) -> tuple[np.ndarray, DSAState]:
    kv = KVCache(num_layers=1)
    kv.compressed_mla = True
    kv.mla_absorbed_prefill = absorbed
    kv.dsa = DSAState(cfg, key_tile_size=2)
    # The caller chooses whether this stops exactly at K or deliberately makes
    # the next tile straddle the K boundary.
    _ = _mla_attention(
        h_mx[:, :tile_start], w, "layer0", cfg, kv,
        layer=0, offset=0)
    out = _mla_attention(
        h_mx[:, tile_start:], w, "layer0", cfg, kv,
        layer=0, offset=tile_start)
    mx.eval(out)
    return np.array(out), kv.dsa


def test_compact_sparse_attention_output_matches_hf_masked_dense():
    hf_cfg, attn, rope = _build_hf_attention(seed=3)
    torch.manual_seed(4)
    h_torch = torch.randn(1, S, HIDDEN)
    position_ids = torch.arange(S)[None, :]
    cos, sin = rope(h_torch, position_ids)

    with torch.no_grad():
        hf_out_all, _, hf_topk_all = attn(
            h_torch, (cos, sin), None, past_key_values=None, position_ids=position_ids
        )
    last_row_sel = set(int(i) for i in hf_topk_all[0, -1].tolist())
    assert last_row_sel != set(range(S)), (
        "expected real DSA sparsity at the last row (S > index_topk) -- "
        "if this fails the comparison below is not exercising the sparse path"
    )
    hf_last_out = hf_out_all[:, -1:, :].detach().numpy()

    w = _weights_from_hf(attn)
    cfg = _runtime_config(hf_cfg)
    h_mx = mx.array(h_torch.numpy())
    runtime_np = _runtime_last_row_output(w, cfg, h_mx)

    assert hf_last_out.shape == runtime_np.shape
    max_abs_diff = np.max(np.abs(hf_last_out - runtime_np))
    assert max_abs_diff < 1e-4, (
        f"compact sparse attention output mismatch at S={S} > index_topk={INDEX_TOPK}: "
        f"max abs diff {max_abs_diff}"
    )


def test_multi_query_tiled_sparse_attention_matches_hf_masked_dense():
    """F75: the first real sparse L>1 tile matches official eager output."""
    hf_cfg, attn, rope = _build_hf_attention(seed=13)
    torch.manual_seed(14)
    h_torch = torch.randn(1, S, HIDDEN)
    position_ids = torch.arange(S)[None, :]
    cos, sin = rope(h_torch, position_ids)
    with torch.no_grad():
        hf_out_all, _, hf_topk_all = attn(
            h_torch, (cos, sin), None,
            past_key_values=None, position_ids=position_ids)

    tile_start = INDEX_TOPK
    w = _weights_from_hf(attn)
    cfg = _runtime_config(hf_cfg)
    runtime_np, dsa = _runtime_last_tile_output(
        w, cfg, mx.array(h_torch.numpy()), tile_start)
    hf_np = hf_out_all[:, tile_start:, :].detach().numpy()

    assert runtime_np.shape == hf_np.shape == (1, S - tile_start, HIDDEN)
    max_abs_diff = np.max(np.abs(hf_np - runtime_np))
    assert max_abs_diff < 1e-4, (
        f"multi-query compact attention mismatch: max abs diff {max_abs_diff}")
    runtime_ids = dsa.selection_ranges[(tile_start, S - tile_start)][0].tolist()
    for row, ids in enumerate(runtime_ids, start=tile_start):
        assert ids == sorted(ids), "compact selected rows must be chronological"
        assert set(ids) == set(int(v) for v in hf_topk_all[0, row].tolist())
    assert dsa.stats["multi_query_selects"] == 1
    assert dsa.stats["score_tiles"] > 1


def test_sparse_tile_straddling_topk_masks_padded_future_rows():
    hf_cfg, attn, rope = _build_hf_attention(seed=23)
    torch.manual_seed(24)
    h_torch = torch.randn(1, S, HIDDEN)
    position_ids = torch.arange(S)[None, :]
    cos, sin = rope(h_torch, position_ids)
    with torch.no_grad():
        hf_out_all, _, _ = attn(
            h_torch, (cos, sin), None,
            past_key_values=None, position_ids=position_ids)

    tile_start = INDEX_TOPK - 2
    w = _weights_from_hf(attn)
    cfg = _runtime_config(hf_cfg)
    runtime_np, dsa = _runtime_last_tile_output(
        w, cfg, mx.array(h_torch.numpy()), tile_start)
    hf_np = hf_out_all[:, tile_start:, :].detach().numpy()
    max_abs_diff = np.max(np.abs(hf_np - runtime_np))
    assert max_abs_diff < 1e-4, (
        f"K-boundary compact attention mismatch: max abs diff {max_abs_diff}")
    ids = np.array(dsa.selection_ranges[(tile_start, S - tile_start)])
    assert np.sum(ids[0, 0] >= 0) == tile_start + 1
    assert -1 in ids[0, 0]


def test_batched_preselection_matches_hf_with_compact_attention_tiles():
    """Wide selector batches must not widen the K/V attention allocation."""
    hf_cfg, attn, rope = _build_hf_attention(seed=33)
    torch.manual_seed(34)
    h_torch = torch.randn(1, S, HIDDEN)
    position_ids = torch.arange(S)[None, :]
    cos, sin = rope(h_torch, position_ids)
    with torch.no_grad():
        hf_out_all, _, _ = attn(
            h_torch, (cos, sin), None,
            past_key_values=None, position_ids=position_ids)

    w = _weights_from_hf(attn)
    cfg = _runtime_config(hf_cfg)
    h_mx = mx.array(h_torch.numpy())
    kv = KVCache(num_layers=1)
    kv.compressed_mla = True
    kv.dsa = DSAState(
        cfg, key_tile_size=2, index_step_size=4,
        selection_query_tile_size=4)
    kv.dsa.preselect_full_layer(
        0, h_mx, w, "layer0", 0, attention_tile_width=2)
    outputs = []
    for start in range(0, S, 2):
        outputs.append(_mla_attention(
            h_mx[:, start:start + 2], w, "layer0", cfg, kv,
            layer=0, offset=start))
    runtime = mx.concatenate(outputs, axis=1)
    mx.eval(runtime)
    max_abs_diff = np.max(np.abs(
        hf_out_all.detach().numpy() - np.array(runtime)))
    assert max_abs_diff < 1e-4, (
        f"preselected compact attention mismatch: max abs diff {max_abs_diff}")
    assert kv.dsa.stats["observations"] == (S + 1) // 2
    assert kv.dsa.stats["preselection_groups"] == 1
    assert kv.dsa.stats["preselection_queries"] == S - INDEX_TOPK
    assert kv.dsa.stats["preselection_attention_ranges"] == 2
    assert kv.dsa.stats["shared_reuses"] == 0


def test_query_specific_absorbed_sparse_attention_preserves_selected_rows():
    hf_cfg, attn, _rope = _build_hf_attention(seed=43)
    torch.manual_seed(44)
    h_torch = torch.randn(1, S, HIDDEN)
    h_mx = mx.array(h_torch.numpy())
    w = _weights_from_hf(attn)
    cfg = _runtime_config(hf_cfg)
    expanded, expanded_dsa = _runtime_last_tile_output(
        w, cfg, h_mx, INDEX_TOPK)
    absorbed, absorbed_dsa = _runtime_last_tile_output(
        w, cfg, h_mx, INDEX_TOPK, absorbed=True)
    assert np.array_equal(
        np.array(expanded_dsa.selection), np.array(absorbed_dsa.selection))
    max_abs_diff = np.max(np.abs(expanded - absorbed))
    cosine = np.dot(expanded.ravel(), absorbed.ravel()) / (
        np.linalg.norm(expanded.ravel()) * np.linalg.norm(absorbed.ravel()))
    assert max_abs_diff < 1e-4
    assert cosine > 0.999999


def test_dense_glm_mlp_position_tiles_match_whole_released_activation_rows():
    """The released BF16 activation rows remain exact across batch widths."""
    rng = np.random.default_rng(53)
    hidden = 8
    intermediate = 12
    prefix = "layer0"
    cfg = SimpleNamespace(
        rms_norm_eps=1e-6,
        mlp_layer_types=("dense",),
        first_k_dense_replace=1,
    )
    weights = {
        f"{prefix}.post_attention_layernorm.weight": mx.array(
            rng.normal(size=(hidden,)).astype(np.float32)).astype(mx.bfloat16),
        f"{prefix}.mlp.gate_proj.weight": mx.array(
            rng.normal(size=(intermediate, hidden)).astype(np.float32)
        ).astype(mx.bfloat16),
        f"{prefix}.mlp.up_proj.weight": mx.array(
            rng.normal(size=(intermediate, hidden)).astype(np.float32)
        ).astype(mx.bfloat16),
        f"{prefix}.mlp.down_proj.weight": mx.array(
            rng.normal(size=(hidden, intermediate)).astype(np.float32)
        ).astype(mx.bfloat16),
    }
    x = mx.array(
        rng.normal(size=(1, 7, hidden)).astype(np.float32)
    ).astype(mx.bfloat16)
    whole = _glm_mlp_residual(
        x, weights, prefix, cfg, 0, lambda *_args, **_kwargs: {})
    tiled = mx.concatenate([
        _glm_mlp_residual(
            x[:, start:start + 3], weights, prefix, cfg, 0,
            lambda *_args, **_kwargs: {})
        for start in range(0, x.shape[1], 3)
    ], axis=1)
    mx.eval(whole, tiled)
    assert np.array_equal(
        np.asarray(whole.astype(mx.float32)),
        np.asarray(tiled.astype(mx.float32)),
    )


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        print(f"  {fn.__name__}: PASS")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
