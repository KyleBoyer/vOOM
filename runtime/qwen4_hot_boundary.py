"""Content-blind planning for the opt-in Qwen4 tile-retention experiment.

No model/tensor imports. This preserves a proposed prefill tile geometry, not
an assertion that floating-point state is equal; real endpoint gates decide.
"""


def plan_tile_retention(rc, *, requested_boundary: int, prompt_tokens: int,
                        force_paged: bool = False) -> dict:
    """Fail closed to uncached execution unless a complete stable tile exists.

    A zero boundary alone is insufficient: the existing engine would otherwise
    retain a post-generation recurrent endpoint with different reuse semantics.
    The caller must also AND ``eligible`` into request hot-cache eligibility.
    """
    tile = getattr(rc, "prefill_chunk_size", 0)
    decision = {"requested": requested_boundary, "effective": 0,
                "tile": tile, "eligible": False, "reason": "unsupported"}
    if (isinstance(tile, bool) or not isinstance(tile, int) or tile <= 0
            or getattr(rc, "hot_prompt_kv_chunk_size", 0) != tile):
        decision["reason"] = "invalid-fixed-tile"
        return decision
    if (not getattr(rc, "hot_prompt_kv", False)
            or not getattr(rc, "layer_stationary_prefill", False)):
        decision["reason"] = "cache-or-layer-stationary-disabled"
        return decision
    if (force_paged or any(getattr(rc, name, False) for name in (
            "adaptive_chunk_size", "max_kv_mb", "adaptive_kv_spill_mb",
            "paged_kv_persist", "hot_prompt_kv_persist_dir",
            "prefill_checkpoint_every", "prefill_last_token_separate",
            "qwen4_global_expert_rows", "qwen4_sparse_expert_batch_rows"))):
        decision["reason"] = "unsupported-prefill-or-persistence"
        return decision
    if (isinstance(requested_boundary, bool)
            or not isinstance(requested_boundary, int)
            or isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, int)
            or not 0 < requested_boundary < prompt_tokens):
        decision["reason"] = "no-valid-stable-prefix"
        return decision
    aligned = requested_boundary // tile * tile
    if aligned < max(1, int(getattr(rc, "hot_prompt_kv_min_tokens", 0))):
        decision["reason"] = "no-admissible-complete-tile"
        return decision
    decision.update(effective=aligned, eligible=True, reason="tile-aligned")
    return decision


def slot_matches_tile_retention(slot, *, effective_boundary: int, tile: int) -> bool:
    """Only reuse complete forks constructed under this tile policy.

    A longer prefix than the current stable boundary would bypass its fork and
    fall back to retaining a raw endpoint; reject it even if all IDs match.
    """
    covered = len(slot.tokens)
    return bool(
        tile > 0 and 0 < covered <= effective_boundary and covered % tile == 0
        and getattr(slot, "qwen4_retention_tile", 0) == tile
        and getattr(slot, "chunk_size", 0) == tile
        and not getattr(slot, "approximate", False)
        and getattr(slot, "logits", None) is None
        and getattr(slot, "prompt_logits", None) is None
        and getattr(slot, "exact_hidden", None) is None)
