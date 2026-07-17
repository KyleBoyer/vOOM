"""F65: architecture-faithful tiny GLM-5.2 fixture (new 2026-07-13 audit item,
flagged as the highest-leverage local enabler). Produces a ~few-MB checkpoint
with the SAME tensor names/architecture as the released model (MLA attention,
DSA indexer, noaux_tc MoE router, shared expert, MTP layer) but tiny
dimensions, so the REAL production engine/glm.py/glm_dsa.py/glm_mtp.py code
runs a full forward pass in a fraction of a second, entirely locally.

Layout: 4 trunk layers (3 dense + 1 MoE, first_k_dense_replace=3) + 1 MTP
layer at index 4 (== num_hidden_layers, matching the real 78-trunk/layer-78
convention). Builds a deterministic synthetic tokenizer (vocab semantics do
not matter for an architecture/math fixture — only valid ids, shapes, and the
production code path do). Every tensor is deterministically seeded so results
are reproducible and hashable for regression fingerprints. The fixture has no
network or pre-downloaded-model dependency.

Indexer dims (32 heads x 128) are the REAL model's hardcoded constants in
glm_dsa.py (not derived from hidden_size) — the fixture uses the same fixed
values so it exercises the exact same code, not a scaled-down approximation.

Usage: .venv/bin/python tests/fixtures/build_glm_fixture.py [out_dir]
Default out_dir: models/glm-fixture-tiny
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import mlx.core as mx

# ---- tiny architecture dimensions ----
HIDDEN = 64
INTER = 128            # dense-layer FFN width
MOE_INTER = 32          # per-expert FFN width
N_HEADS = 4
DN, DR, DV = 16, 8, 16  # qk_nope, qk_rope, v head dims (real: 192/64/256)
Q_LORA, KV_LORA = 32, 24
N_EXPERTS, TOPK, N_SHARED = 8, 2, 1
IDX_HEADS, IDX_DIM = 32, 128  # FIXED real-model constants (glm_dsa.py), not scaled
N_DENSE, N_MOE = 3, 1
N_TRUNK = N_DENSE + N_MOE
MTP_LAYER = N_TRUNK  # matches the real convention: MTP == num_hidden_layers
INDEX_TOPK = 32       # <= this bound: DSA dense-exact path stays active for tests
ROPE_THETA = 8e6
INDEX_TOPK_FREQ = 4
INDEX_SKIP_TOPK_OFFSET = 3
FIXTURE_VERSION = 7  # v7: expose MLA ranks for architecture-aware planning
VOCAB_SIZE = 49_152

SEED_COUNTER = [0]


def released_indexer_types(n_layers: int) -> list[str]:
    """Reproduce GlmMoeDsaConfig's released offset/frequency schedule exactly.

    For offset=3 and frequency=4 the prefix is F,F,F,S,S,S,F,S,S,S,...,
    not the simpler (and previously used) F,S,S,S,... cadence.
    """
    return [
        "full" if max(i - INDEX_SKIP_TOPK_OFFSET + 1, 0) % INDEX_TOPK_FREQ == 0
        else "shared"
        for i in range(n_layers)
    ]


def is_current(out_dir: Path) -> bool:
    """Return whether an existing fixture satisfies this builder's contract.

    Checks every file build() writes, not just model.safetensors -- a build
    interrupted partway through (a killed process, Ctrl-C) can leave the
    tensors written but the tokenizer copy step never reached, and a
    tensors-only check would then silently treat that half-built fixture as
    current forever."""
    try:
        config = json.loads((out_dir / "config.json").read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        (out_dir / "model.safetensors").exists()
        and (out_dir / "tokenizer.json").exists()
        and (out_dir / "tokenizer_config.json").exists()
        and config.get("fixture_version") == FIXTURE_VERSION
        and config.get("indexer_types") == released_indexer_types(N_TRUNK)
    )


def w(*shape) -> mx.array:
    """Deterministic, distinctly-seeded small random weight."""
    SEED_COUNTER[0] += 1
    mx.random.seed(1000 + SEED_COUNTER[0])
    return (mx.random.normal(shape) * 0.05).astype(mx.bfloat16)


def write_tokenizer(out_dir: Path, vocab_size: int) -> None:
    """Write a small-behaviour, full-id-range tokenizer for the math fixture.

    A byte-level alphabet preserves exact prompt differences, which makes hot-KV
    longest-common-prefix tests meaningful, while an empty BPE merge table keeps
    the implementation deterministic and simple. The full 49,152-id vocabulary
    also keeps deliberately wrong speculative-test ids in range.
    """
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel

    vocab = {"[UNK]": 0, "<s>": 1, "</s>": 2}
    for token in sorted(ByteLevel.alphabet()):
        vocab[token] = len(vocab)
    vocab.update({f"token_{i:05d}": i for i in range(len(vocab), vocab_size)})
    tokenizer = Tokenizer(BPE(vocab=vocab, merges=[], unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.add_special_tokens(["<s>", "</s>"])
    tokenizer.save(str(out_dir / "tokenizer.json"))
    (out_dir / "tokenizer_config.json").write_text(json.dumps({
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "[UNK]",
        "model_max_length": 4096,
    }, indent=2))


def build(out_dir: Path):
    # Repeated builds in one process must be byte-reproducible.  The original
    # module-global counter kept advancing and silently changed every tensor on
    # the second call.
    SEED_COUNTER[0] = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, mx.array] = {}

    def attn_block(p: str):
        tensors[f"{p}.input_layernorm.weight"] = w(HIDDEN)
        tensors[f"{p}.self_attn.q_a_proj.weight"] = w(Q_LORA, HIDDEN)
        tensors[f"{p}.self_attn.q_a_layernorm.weight"] = w(Q_LORA)
        tensors[f"{p}.self_attn.q_b_proj.weight"] = w(N_HEADS * (DN + DR), Q_LORA)
        tensors[f"{p}.self_attn.kv_a_proj_with_mqa.weight"] = w(KV_LORA + DR, HIDDEN)
        tensors[f"{p}.self_attn.kv_a_layernorm.weight"] = w(KV_LORA)
        tensors[f"{p}.self_attn.kv_b_proj.weight"] = w(N_HEADS * (DN + DV), KV_LORA)
        tensors[f"{p}.self_attn.o_proj.weight"] = w(HIDDEN, N_HEADS * DV)
        # DSA lightning indexer: FIXED 32x128 dims per the real checkpoint/code
        tensors[f"{p}.self_attn.indexer.wq_b.weight"] = w(IDX_HEADS * IDX_DIM, Q_LORA)
        tensors[f"{p}.self_attn.indexer.wk.weight"] = w(IDX_DIM, HIDDEN)
        tensors[f"{p}.self_attn.indexer.k_norm.weight"] = w(IDX_DIM)
        tensors[f"{p}.self_attn.indexer.k_norm.bias"] = mx.zeros((IDX_DIM,)).astype(mx.bfloat16)
        tensors[f"{p}.self_attn.indexer.weights_proj.weight"] = w(IDX_HEADS, HIDDEN)
        tensors[f"{p}.post_attention_layernorm.weight"] = w(HIDDEN)

    def dense_mlp(p: str):
        tensors[f"{p}.mlp.gate_proj.weight"] = w(INTER, HIDDEN)
        tensors[f"{p}.mlp.up_proj.weight"] = w(INTER, HIDDEN)
        tensors[f"{p}.mlp.down_proj.weight"] = w(HIDDEN, INTER)

    def moe_mlp(p: str):
        tensors[f"{p}.mlp.gate.weight"] = w(N_EXPERTS, HIDDEN)
        tensors[f"{p}.mlp.gate.e_score_correction_bias"] = mx.zeros((N_EXPERTS,)).astype(mx.float32)
        tensors[f"{p}.mlp.shared_experts.gate_proj.weight"] = w(MOE_INTER * N_SHARED, HIDDEN)
        tensors[f"{p}.mlp.shared_experts.up_proj.weight"] = w(MOE_INTER * N_SHARED, HIDDEN)
        tensors[f"{p}.mlp.shared_experts.down_proj.weight"] = w(HIDDEN, MOE_INTER * N_SHARED)
        for e in range(N_EXPERTS):
            tensors[f"{p}.mlp.experts.{e}.gate_proj.weight"] = w(MOE_INTER, HIDDEN)
            tensors[f"{p}.mlp.experts.{e}.up_proj.weight"] = w(MOE_INTER, HIDDEN)
            tensors[f"{p}.mlp.experts.{e}.down_proj.weight"] = w(HIDDEN, MOE_INTER)

    for i in range(N_TRUNK):
        p = f"model.layers.{i}"
        attn_block(p)
        if i < N_DENSE:
            dense_mlp(p)
        else:
            moe_mlp(p)

    # MTP layer: same full attention+MoE block (it's >= first_k_dense_replace,
    # so run_glm_block treats it as MoE) PLUS the MTP-specific glue tensors.
    p = f"model.layers.{MTP_LAYER}"
    attn_block(p)
    moe_mlp(p)
    tensors[f"{p}.enorm.weight"] = w(HIDDEN)
    tensors[f"{p}.hnorm.weight"] = w(HIDDEN)
    tensors[f"{p}.eh_proj.weight"] = w(HIDDEN, 2 * HIDDEN)
    tensors[f"{p}.shared_head.norm.weight"] = w(HIDDEN)

    # Keep the released fixture vocabulary width used by existing regression
    # tests, but generate tokenizer metadata locally instead of requiring a
    # separate model download merely to make this fixture usable.
    vocab_size = VOCAB_SIZE

    tensors["model.embed_tokens.weight"] = w(vocab_size, HIDDEN)
    tensors["model.norm.weight"] = w(HIDDEN)
    tensors["lm_head.weight"] = w(vocab_size, HIDDEN)

    mx.eval(list(tensors.values()))
    mx.save_safetensors(str(out_dir / "model.safetensors"), tensors)

    config = {
        "fixture_version": FIXTURE_VERSION,
        "model_type": "glm_moe_dsa",
        "hidden_size": HIDDEN,
        "intermediate_size": INTER,
        "num_hidden_layers": N_TRUNK,
        "num_attention_heads": N_HEADS,
        "num_key_value_heads": N_HEADS,
        "vocab_size": vocab_size,
        "rms_norm_eps": 1e-5,
        "rope_theta": ROPE_THETA,
        "max_position_embeddings": 4096,
        "tie_word_embeddings": False,
        "attention_bias": False,
        "head_dim": DN + DR,
        "eos_token_id": [2],
        "torch_dtype": "bfloat16",
        "num_experts": N_EXPERTS,
        "num_experts_per_tok": TOPK,
        "moe_intermediate_size": MOE_INTER,
        "norm_topk_prob": True,
        "first_k_dense_replace": N_DENSE,
        "qk_nope_head_dim": DN,
        "qk_rope_head_dim": DR,
        "v_head_dim": DV,
        "q_lora_rank": Q_LORA,
        "kv_lora_rank": KV_LORA,
        "n_shared_experts": N_SHARED,
        "routed_scaling_factor": 2.5,
        "rope_interleave": True,
        "index_topk": INDEX_TOPK,
        # Match the released offset=3/frequency=4 schedule exactly.  The first
        # three layers are all full indexers; only then does the F,S,S,S cadence
        # begin.  Match the release's config contract: this list covers BACKBONE
        # layers only.  MTP (layer == num_hidden_layers) always owns a full
        # indexer and dynamically computes at proposal step 0, then reuses that
        # selection at steps 1+ when index_share_for_mtp_iteration is enabled.
        # Encoding MTP as a static extra tuple entry would hide that behavior.
        "indexer_types": released_indexer_types(N_TRUNK),
        "index_topk_freq": INDEX_TOPK_FREQ,
        "index_skip_topk_offset": INDEX_SKIP_TOPK_OFFSET,
        "index_share_for_mtp_iteration": True,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    (out_dir / "generation_config.json").write_text(json.dumps({"eos_token_id": [2]}))
    write_tokenizer(out_dir, vocab_size)

    total_bytes = sum(t.nbytes for t in tensors.values())
    print(f"fixture built: {len(tensors)} tensors, {total_bytes/1e6:.1f} MB -> {out_dir}")
    print(f"trunk layers 0-{N_TRUNK-1} ({N_DENSE} dense + {N_MOE} MoE), "
          f"MTP layer {MTP_LAYER}, index_topk={INDEX_TOPK}")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("models/glm-fixture-tiny")
    build(out)
