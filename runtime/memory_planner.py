"""MemoryPlanner: turns (model geometry, memory budget, disk throughput, workload
profile) into a concrete RuntimeConfig — which tensors to pin, how much cache, what
precision the cache holds, prefetch depth — plus an estimate of per-token cost so
policies can be compared before spending 40 seconds on a prefill.

Placement policies:
  sequential     stream every layer, minimal cache, prefetch hides what it can.
                 Lowest RAM; token latency = full-model disk read.
  lru            budgeted LRU cache at disk precision; whatever fits stays.
                 (first/last pinning is a knob on top: pin_first/pin_last)
  latency        pick the cheapest configuration that meets the budget: prefer
                 fully-resident at disk precision, else quantize-on-load resident,
                 else mixed (attn full, MLP quantized), else partial residency.
  coding         latency policy + larger KV budget headroom and lm_head pinned:
                 code sessions run long contexts and sample many tokens.

The engine remains the LayerScheduler (sequential sweep + prefetch window); the
planner only decides placement. On Apple Silicon there is no CPU-vs-GPU split to
plan — unified memory means residency is the only placement axis (see
docs/design.md "Memory layout / tiers").
"""

from __future__ import annotations

from dataclasses import dataclass

from .engine import RuntimeConfig
from .model_loader import WeightStore

POLICIES = ("sequential", "lru", "latency", "coding")


@dataclass
class Plan:
    policy: str
    rc: RuntimeConfig
    embed_bytes: int
    lm_head_bytes: int
    layer_bytes: int  # full checkpoint weights per average layer, at disk precision
    active_layer_bytes: int  # no-cache decode traffic (only top-k routed experts)
    resident_layer_bytes: int  # per layer, at cache precision
    resident_layers: int  # layers expected to stay cached
    est_disk_bytes_per_token: int
    est_token_s: float

    def summary(self) -> str:
        n = self.rc and self.rc.quant_bits
        prec = f"{n}-bit cache" if n else "disk precision cache"
        return (
            f"policy={self.policy} ({prec})\n"
            f"  budget: {self.rc.max_weight_cache_mb}MB weights, prefetch={self.rc.prefetch_depth}, "
            f"pin lm_head={self.rc.pin_lm_head}\n"
            f"  resident: {self.resident_layers} layers "
            f"(~{self.resident_layer_bytes / 1e6:.0f}MB each at cache precision)\n"
            f"  active path: ~{self.active_layer_bytes / 1e6:.0f}MB/layer; "
            f"est. disk traffic {self.est_disk_bytes_per_token / 1e6:.0f}MB/token "
            f"-> ~{self.est_token_s:.2f}s/token steady-state"
        )


class MemoryPlanner:
    def __init__(
        self,
        store: WeightStore,
        budget_mb: int,
        disk_mb_per_s: float = 300.0,  # measured ~315 MB/s on this machine's USB SSD
        compute_s_per_layer: float = 0.003,
    ):
        self.store = store
        self.cfg = store.config
        self.budget = budget_mb * 1_000_000
        # Measured on the 16GB M4: Metal working sets beyond ~55% of physical RAM
        # get compressed/paged by macOS ("resident" becomes swap-thrash — layer
        # compute went 2ms -> 1.1s on Qwen2.5-14B q4 at 9.4GB). Clamp unless the
        # caller explicitly opts out.
        import psutil

        wired_limit = int(psutil.virtual_memory().total * 0.55)
        if self.budget > wired_limit:
            print(f"[planner] clamping budget {self.budget / 1e9:.1f}GB -> "
                  f"{wired_limit / 1e9:.1f}GB (macOS wired-memory pressure threshold)")
            self.budget = wired_limit
        self.disk_bps = disk_mb_per_s * 1e6
        self.compute_s = compute_s_per_layer

        c = self.cfg
        dtype_bytes = 2  # bf16/fp16 checkpoints
        ratio_for = getattr(store, "quantization_ratio", lambda _name: 1.0)
        uniform_ratio = getattr(
            store, "uniform_quantization_ratio", lambda _fragment: 1.0)
        embed_full_bytes = c.vocab_size * c.hidden_size * dtype_bytes
        self.embed_bytes = int(
            embed_full_bytes * ratio_for("model.embed_tokens.weight"))
        self.lm_head_bytes = (0 if c.tie_word_embeddings else int(
            embed_full_bytes * ratio_for("lm_head.weight")))
        attn_disk_ratio = uniform_ratio(".self_attn.")
        mlp_disk_ratio = uniform_ratio(".mlp.")
        if c.model_type == "glm_moe_dsa":
            # MLA is low-rank rather than ordinary Q/K/V projections, and DSA's
            # 32x128 indexer is a material part of every layer. Released GLM
            # configs provide both ranks; falling back to hidden_size is
            # deliberately conservative for incomplete third-party configs.
            q_rank = c.q_lora_rank or c.hidden_size
            kv_rank = c.kv_lora_rank or c.hidden_size
            dn, dr, dv = c.qk_nope_head_dim, c.qk_rope_head_dim, c.v_head_dim
            mla_params = (
                q_rank * c.hidden_size + q_rank
                + c.num_attention_heads * (dn + dr) * q_rank
                + (kv_rank + dr) * c.hidden_size + kv_rank
                + c.num_attention_heads * (dn + dv) * kv_rank
                + c.hidden_size * c.num_attention_heads * dv
            )
            indexer_params = (
                c.index_n_heads * c.index_head_dim * q_rank
                + c.index_head_dim * c.hidden_size
                + 2 * c.index_head_dim
                + c.index_n_heads * c.hidden_size
            ) if c.index_topk else 0
            self.attn_bytes = int(
                (mla_params + indexer_params) * dtype_bytes * attn_disk_ratio)
        else:
            self.attn_bytes = int((
                c.hidden_size * c.head_dim
                * (c.num_attention_heads + 2 * c.num_key_value_heads)
                + c.head_dim * c.num_attention_heads * c.hidden_size
            ) * dtype_bytes * attn_disk_ratio)

        dense_mlp_bytes = int(
            3 * c.hidden_size * c.intermediate_size * dtype_bytes
            * mlp_disk_ratio)
        self.norm_bytes = 2 * c.hidden_size * dtype_bytes
        n_layers = c.num_hidden_layers
        if c.num_experts:
            if len(c.mlp_layer_types) >= n_layers:
                n_dense = sum(kind == "dense" for kind in c.mlp_layer_types[:n_layers])
            else:
                n_dense = min(c.first_k_dense_replace, n_layers)
            n_moe = n_layers - n_dense
            expert_width = c.moe_intermediate_size or c.intermediate_size
            expert_bytes = int(
                3 * c.hidden_size * expert_width * dtype_bytes * mlp_disk_ratio)
            router_bytes = (
                int(c.hidden_size * c.num_experts * dtype_bytes * mlp_disk_ratio)
                + c.num_experts * 4  # correction bias is FP32
            )
            shared_bytes = c.n_shared_experts * expert_bytes
            moe_full_bytes = router_bytes + shared_bytes + c.num_experts * expert_bytes
            moe_active_bytes = (
                router_bytes + shared_bytes + c.num_experts_per_tok * expert_bytes)
            full_mlp_total = n_dense * dense_mlp_bytes + n_moe * moe_full_bytes
            active_mlp_total = n_dense * dense_mlp_bytes + n_moe * moe_active_bytes
        else:
            full_mlp_total = active_mlp_total = n_layers * dense_mlp_bytes

        self.mlp_bytes = full_mlp_total // max(n_layers, 1)
        self.active_mlp_bytes = active_mlp_total // max(n_layers, 1)
        self.layer_bytes = self.attn_bytes + self.mlp_bytes + self.norm_bytes
        self.active_layer_bytes = (
            self.attn_bytes + self.active_mlp_bytes + self.norm_bytes)

    # ---- policy entry point ------------------------------------------------

    def plan(self, policy: str = "latency") -> Plan:
        if policy == "sequential":
            return self._plan_stream()
        if policy == "lru":
            return self._plan_resident(quant_bits=0, policy="lru")
        if policy in ("latency", "coding"):
            return self._plan_latency(policy)
        raise ValueError(f"unknown policy {policy!r}, expected one of {POLICIES}")

    # ---- concrete planners ---------------------------------------------------

    def _base_rc(self) -> RuntimeConfig:
        return RuntimeConfig(
            max_weight_cache_mb=self.budget // 1_000_000,
            pin_embeddings=True,
            pin_lm_head=bool(self.lm_head_bytes) and self._fits(
                self.embed_bytes + self.lm_head_bytes + 3 * self.layer_bytes
            ),
        )

    def _fits(self, nbytes: int) -> bool:
        return nbytes <= self.budget

    def _finish(self, rc: RuntimeConfig, policy: str) -> Plan:
        if self.store.on_disk_quantized or not rc.quant_bits:
            q = 1.0
        elif rc.quant_mode == "affine":
            q = (rc.quant_bits / 8 + 4 / rc.quant_group_size) / 2
        else:
            # MX/NV modes store one byte-sized scale per group and no bias.
            q = (rc.quant_bits / 8 + 1 / rc.quant_group_size) / 2
        resident_layer = int(
            self.attn_bytes * (q if rc.quant_bits and rc.quant_attention else 1.0)
            + self.mlp_bytes * (q if rc.quant_bits and rc.quant_mlp else 1.0)
            + self.norm_bytes
        )
        pinned = self.embed_bytes + (self.lm_head_bytes if rc.pin_lm_head else 0)
        n = self.cfg.num_hidden_layers
        resident_layers = min(n, max(0, (self.budget - pinned) // max(resident_layer, 1)))
        streamed = n - resident_layers
        per_token = streamed * self.active_layer_bytes + (
            0 if rc.pin_lm_head or not self.lm_head_bytes else self.lm_head_bytes
        )
        est = per_token / self.disk_bps + n * self.compute_s
        return Plan(
            policy=policy,
            rc=rc,
            embed_bytes=self.embed_bytes,
            lm_head_bytes=self.lm_head_bytes,
            layer_bytes=self.layer_bytes,
            active_layer_bytes=self.active_layer_bytes,
            resident_layer_bytes=resident_layer,
            resident_layers=int(resident_layers),
            est_disk_bytes_per_token=int(per_token),
            est_token_s=est,
        )

    def _plan_stream(self) -> Plan:
        rc = self._base_rc()
        # cache sized to pinned set + a small working window; prefetch hides reads
        window = 3 * self.active_layer_bytes
        pinned = self.embed_bytes + (self.lm_head_bytes if rc.pin_lm_head else 0)
        rc.max_weight_cache_mb = (pinned + window) // 1_000_000 + 1
        rc.prefetch_depth = 2
        plan = self._finish(rc, "sequential")
        plan.resident_layers = 0  # streaming: nothing is expected to survive a sweep
        plan.est_disk_bytes_per_token = (
            self.cfg.num_hidden_layers * self.active_layer_bytes)
        plan.est_token_s = plan.est_disk_bytes_per_token / self.disk_bps
        return plan

    def _plan_resident(self, quant_bits: int, policy: str, quant_attention: bool = True) -> Plan:
        rc = self._base_rc()
        rc.quant_bits = quant_bits
        rc.quant_attention = quant_attention
        rc.prefetch_depth = 2
        return self._finish(rc, policy)

    def _plan_latency(self, policy: str) -> Plan:
        """Cheapest precision that reaches full residency; falls back to partial."""
        n = self.cfg.num_hidden_layers
        candidates = [
            self._plan_resident(0, policy),  # full precision resident
            self._plan_resident(8, policy),
            self._plan_resident(8, policy, quant_attention=False),
            self._plan_resident(4, policy, quant_attention=False),  # mixed: attn bf16
            self._plan_resident(4, policy),
        ]
        for plan in candidates:
            if plan.resident_layers >= n:
                break
        else:
            plan = min(candidates, key=lambda p: p.est_token_s)
        if policy == "coding":
            plan.rc.pin_lm_head = bool(self.lm_head_bytes)  # sampled every token
            plan.rc.max_kv_mb = max(512, self.budget // 1_000_000 // 8)  # long-context headroom
        return plan
