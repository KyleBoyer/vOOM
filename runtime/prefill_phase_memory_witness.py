"""Opt-in, bounded observations at existing Qwen4 host-spool boundaries.

No MLX import, tensor traversal, synchronization, clearing, peak reset, allocation
policy or pressure verdict. Boundaries are Python progress points, not proof of
GPU retirement or causal ownership. Byte views overlap and must not be added.
"""

import json
import time

from .phase_head_witness import sample_phase_head_memory

FLAG = 'VMODEL_PREFILL_PHASE_MEMORY_WITNESS'
MAX_SAMPLES = 1536
PHASES = frozenset(('initial_hidden', 'retained_prefix', 'attention_tile',
                   'expert_batch', 'output_tile', 'layer_complete',
                   'layer_enter', 'layer_released', 'attention_inputs_ready',
                   'attention_branch_returned', 'attention_evaluated',
                   'prefix_observed'))
BRACKET_PHASES = frozenset(('attention_inputs_ready', 'attention_branch_returned',
                          'attention_evaluated', 'prefix_observed'))
TILE_PHASES = frozenset(('attention_tile', 'output_tile')) | BRACKET_PHASES


class PrefillPhaseMemoryObserver:
    def __init__(self, *, total_tokens, total_layers, tile_width,
                 prefix_capture_enabled=False, capture_tokens=None, emit=None, sample=None):
        if (type(total_tokens) is not int or not 1 <= total_tokens <= 1_048_576
                or type(total_layers) is not int or not 1 <= total_layers <= 1024
                or type(tile_width) is not int or not 1 <= tile_width <= 1_048_576
                or type(prefix_capture_enabled) is not bool):
            raise ValueError('bounded integer prefill dimensions required')
        self.total_tokens, self.total_layers, self.tile_width = total_tokens, total_layers, tile_width
        self.prefix_capture_enabled = prefix_capture_enabled
        if prefix_capture_enabled:
            if (type(capture_tokens) is not int or not 0 < capture_tokens < total_tokens
                    or capture_tokens % tile_width):
                raise ValueError('capture boundary must be a strict interior tile end')
        elif capture_tokens is not None:
            raise ValueError('capture boundary requires enabled capture')
        self.capture_tokens = capture_tokens
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
        if (type(phase) is not str or phase not in PHASES or type(layer_marker) is not int
                or not 0 <= layer_marker <= self.total_layers
                or type(completed_tokens) is not int
                or not 0 <= completed_tokens <= self.total_tokens
                or type(reported_host_spool_peak_bytes) is not int
                or reported_host_spool_peak_bytes < 0):
            self.failed = True
            return
        # First/final tile ends, plus the exact interior capture tile for the
        # attention/capture brackets. Existing published expert-batch boundaries
        # remain visible, without inspecting expert IDs.
        if (phase in TILE_PHASES and completed_tokens not in (
                min(self.tile_width, self.total_tokens), self.total_tokens)
                and not (phase in BRACKET_PHASES and completed_tokens == self.capture_tokens)):
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
                ('attention_inputs_ready', layer, self.total_tokens),
                ('attention_branch_returned', layer, self.total_tokens),
                ('attention_evaluated', layer, self.total_tokens),
                ('attention_tile', layer, self.total_tokens),
                ('expert_batch', layer, self.total_tokens),
                ('output_tile', layer, self.total_tokens),
                ('layer_complete', layer + 1, self.total_tokens),
                ('layer_released', layer, self.total_tokens)))
            if self.prefix_capture_enabled:
                required.add(('prefix_observed', layer, self.total_tokens))
                required.update((phase, layer, self.capture_tokens) for phase in BRACKET_PHASES)
        missing = len(required - self.seen)
        self._publish(dict(schema='voom.prefill-phase-memory-end.v2', samples=self.count,
            coverage_complete=bool(self.count and not missing and not self.failed and not self.capped),
            missing_required_boundaries=missing,
            observation_failed=self.failed, capped=self.capped,
            observation_seconds=self.observation_seconds,
            synchronizes_device=False, clears_allocator_cache=False,
            prefix_capture_enabled=self.prefix_capture_enabled,
            capture_tokens=self.capture_tokens,
            layer_marker_convention='layer_complete is completed-layer count; all other per-layer markers are zero-based indices',
            scope='Python host-spool and attention/capture boundaries, not GPU retirement, additive physical RAM, swap attribution or causal ownership proof'))
