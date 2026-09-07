"""Opt-in, bounded observations at existing Qwen4 host-spool boundaries.

No MLX import, tensor traversal, synchronization, clearing, peak reset, allocation
policy or pressure verdict. Boundaries are Python progress points, not proof of
GPU retirement or causal ownership. Byte views overlap and must not be added.
"""

import json
import time

from .phase_head_witness import sample_phase_head_memory

FLAG = 'VMODEL_PREFILL_PHASE_MEMORY_WITNESS'
MAX_SAMPLES = 1024
PHASES = frozenset(('initial_hidden', 'retained_prefix', 'attention_tile',
                   'expert_batch', 'output_tile', 'layer_complete',
                   'layer_enter', 'layer_released'))


class PrefillPhaseMemoryObserver:
    def __init__(self, *, total_tokens, total_layers, tile_width, emit=None, sample=None):
        if (type(total_tokens) is not int or not 1 <= total_tokens <= 1_048_576
                or type(total_layers) is not int or not 1 <= total_layers <= 1024
                or type(tile_width) is not int or not 1 <= tile_width <= 1_048_576):
            raise ValueError('bounded integer prefill dimensions required')
        self.total_tokens, self.total_layers, self.tile_width = total_tokens, total_layers, tile_width
        self.emit = emit if emit is not None else lambda row: print(
            '[prefill-phase-memory] ' + json.dumps(row, allow_nan=False, separators=(',', ':')), flush=True)
        self.sample = sample if sample is not None else sample_phase_head_memory
        self.count = 0
        self.failed = False
        self.capped = False
        self.finished = False
        self.observation_seconds = 0.0
        self.seen = set()

    def _publish(self, row):
        try:
            self.emit(row)
        except Exception:
            # A diagnostic sink failure must not alter model execution. The
            # final trace cannot claim complete coverage after any failure.
            self.failed = True

    def record(self, target, metal, *, phase, layer_marker, completed_tokens=0,
               reported_host_spool_peak_bytes=0):
        if self.finished or self.capped:
            return
        if (phase not in PHASES or type(layer_marker) is not int
                or not 0 <= layer_marker <= self.total_layers
                or type(completed_tokens) is not int
                or not 0 <= completed_tokens <= self.total_tokens
                or type(reported_host_spool_peak_bytes) is not int
                or reported_host_spool_peak_bytes < 0):
            self.failed = True
            return
        # Two end points per tile phase; not every tile/call. Existing published
        # expert-batch boundaries remain visible, without inspecting expert IDs.
        if phase in ('attention_tile', 'output_tile') and completed_tokens not in (
                min(self.tile_width, self.total_tokens), self.total_tokens):
            return
        if self.count >= MAX_SAMPLES:
            self.capped = True
            return
        started = time.monotonic()
        try:
            values = self.sample(target, metal)
            if (not isinstance(values, dict) or values.get('available') is not True
                    or not isinstance(values.get('process_memory'), dict)
                    or values['process_memory'].get('available') is not True):
                self.failed = True
            if not isinstance(values, dict):
                values = {'available': False, 'reason': 'invalid-observation'}
        except Exception:
            self.failed = True
            values = {'available': False, 'reason': 'observation-error'}
        self.count += 1
        self.seen.add((phase, layer_marker, completed_tokens))
        elapsed = time.monotonic() - started
        self.observation_seconds += elapsed
        self._publish(dict(schema='voom.prefill-phase-memory.v1', sample_index=self.count,
            atomic=False, phase=phase, layer_marker=layer_marker,
            completed_tokens=completed_tokens, total_tokens=self.total_tokens,
            total_layers=self.total_layers, tile_width=self.tile_width,
            reported_host_spool_peak_bytes=reported_host_spool_peak_bytes,
            monotonic_s=started, observation_seconds=elapsed, values=values))

    def finish(self):
        if self.finished:
            return
        self.finished = True
        required = {('initial_hidden', 0, self.total_tokens)}
        for layer in range(self.total_layers):
            required.update((
                ('layer_enter', layer, 0),
                ('attention_tile', layer, self.total_tokens),
                ('expert_batch', layer, self.total_tokens),
                ('output_tile', layer, self.total_tokens),
                ('layer_complete', layer + 1, self.total_tokens),
                ('layer_released', layer, self.total_tokens)))
        missing = len(required - self.seen)
        self._publish(dict(schema='voom.prefill-phase-memory-end.v1', samples=self.count,
            coverage_complete=bool(self.count and not missing and not self.failed and not self.capped),
            missing_required_boundaries=missing,
            observation_failed=self.failed, capped=self.capped,
            observation_seconds=self.observation_seconds,
            synchronizes_device=False, clears_allocator_cache=False,
            layer_marker_convention='layer_complete is completed-layer count; all other per-layer markers are zero-based indices',
            scope='Python host-spool boundaries, not GPU retirement, additive physical RAM, swap attribution or causal ownership proof'))
