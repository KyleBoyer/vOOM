"""Private, cold-only aligned prefix capture for the opt-in Qwen4 sweep hook.

No model imports at module load. The caller must already charge the
additional retained logical state/backing allowance before beginning a sweep.
This helper reserves only copy scratch; it never credits future reclamation.

Observe EVERY original attention tile, after its existing evaluation and before
the next tile. Do not split tiles, change expert unions, or publish a cache until
the complete sweep (including the final hidden restoration) has succeeded.
"""

from dataclasses import dataclass
import time


_FIELDS = (
    ("keys", "values", "_starts", "_windows"),
    ("_state", "_conv"),
    ("qsa_keys", "qsa_positions", "qsa_pooled_keys",
     "ple_conv", "ple_context", "ple_lengths"),
)


@dataclass(frozen=True)
class _Layer:
    key: object
    value: object
    start: int
    window: object
    state: object
    conv: object
    qsa_key: object
    qsa_position: object
    ple_conv: object
    ple_context: object
    ple_length: int


class AlignedPrefixCapture:
    """Stage immutable layer records; never expose a mixed-position KV cache.

    Types, allocator/evaluation functions and the empty-cache factory are
    injected for pure lifecycle tests. Production callers must supply the
    concrete KVCache/KDAStateCache/Qwen4ExpStateCache types and the proven
    copy_history_bits implementation. Only unwindowed, uncompressed plain RAM
    state with the derived QSA pool disabled is supported in this first scope.
    """

    def __init__(self, source, cfg, *, prefix_tokens, total_tokens, tile_tokens,
                 cache_types, empty_cache, activation_dtype, state_dtype,
                 position_dtype, copy_bits, evaluate, reserve,
                 clock=time.perf_counter):
        if (any(type(n) is not int or n <= 0
                for n in (prefix_tokens, total_tokens, tile_tokens))
                or not prefix_tokens < total_tokens
                or prefix_tokens % tile_tokens):
            raise ValueError("capture needs an aligned strict interior prefix")
        count = cfg.num_hidden_layers
        kinds = tuple(cfg.layer_types)
        ple = tuple(cfg.qwen4_ple_layers)
        if (cfg.model_type != "qwen4_exp" or type(count) is not int or count <= 0
                or len(kinds) != count
                or set(kinds) != {"linear_attention", "full_attention"}
                or len(set(ple)) != len(ple)
                or any(type(i) is not int or not 0 <= i < count for i in ple)):
            raise ValueError("unsupported capture layer layout")
        dimensions = (
            cfg.linear_num_key_heads, cfg.linear_key_head_dim,
            cfg.linear_num_value_heads, cfg.linear_value_head_dim,
            cfg.linear_conv_kernel_dim, cfg.qwen4_ple_conv_kernel_size,
            cfg.qwen4_ngram_size, cfg.qwen4_hc_count, cfg.hidden_size,
            cfg.num_key_value_heads, cfg.head_dim, cfg.qwen4_indexer_head_dim,
        )
        if any(type(n) is not int or n <= 0 for n in dimensions):
            raise ValueError("capture requires positive integer dimensions")
        self.prefix_tokens = prefix_tokens
        self.total_tokens = total_tokens
        self.tile_tokens = tile_tokens
        self._count, self._kinds, self._ple = count, kinds, frozenset(ple)
        self._types = tuple(cache_types)
        if len(self._types) != 3:
            raise ValueError("capture needs three concrete cache types")
        self._activation_dtype = activation_dtype
        self._state_dtype = state_dtype
        self._position_dtype = position_dtype
        self._state_shape = (1, cfg.linear_num_value_heads,
                             cfg.linear_key_head_dim, cfg.linear_value_head_dim)
        self._conv_shape = (
            1, cfg.linear_conv_kernel_dim - 1,
            2 * cfg.linear_num_key_heads * cfg.linear_key_head_dim
            + cfg.linear_num_value_heads * cfg.linear_value_head_dim)
        self._ple_shape = (
            1, (cfg.qwen4_ple_conv_kernel_size - 1) * cfg.qwen4_ngram_size,
            cfg.qwen4_hc_count * cfg.hidden_size)
        self._context_length = cfg.qwen4_ngram_size - 1
        self._kv_heads, self._head_dim = cfg.num_key_value_heads, cfg.head_dim
        self._index_dim = cfg.qwen4_indexer_head_dim
        self._empty_cache = empty_cache
        self._copy, self._evaluate, self._reserve = copy_bits, evaluate, reserve
        self._clock = clock
        self._records = []
        self._endpoints = []
        self._next_layer = self._next_start = 0
        self._started = self._closed = False
        self._arrays = self._bytes = self._scratch_peak = 0
        self._capture_seconds = 0.0
        self._source = source
        self._owners, self._lists = self._validate_cache(source, empty=True)

    def _validate_cache(self, kv, *, empty=False):
        owners = (kv, getattr(kv, "kda_cache", None),
                  getattr(kv, "qwen4_cache", None))
        if any(type(owner) is not kind for owner, kind in zip(owners, self._types)):
            raise ValueError("capture requires concrete plain cache owners")
        _, kda, aux = owners
        if (kv.compressed_mla or getattr(kv, "dsa", None) is not None
                or kda.spill_enabled or kda._spill_meta
                or kda._factor_capture is not None or aux.qsa_pool_cache_enabled):
            raise ValueError("capture requires resident non-factor, non-pooled state")
        lists = tuple(getattr(owner, name)
                      for owner, names in zip(owners, _FIELDS) for name in names)
        if (any(type(items) is not list or len(items) != self._count for items in lists)
                or len({id(items) for items in lists}) != len(lists)):
            raise ValueError("capture requires complete independent state lists")
        if (any(type(start) is not int or start != 0 for start in kv._starts)
                or any(window is not None for window in kv._windows)
                or any(value is not None for value in aux.qsa_pooled_keys)):
            raise ValueError("capture does not support sliding, shifted or pooled state")
        if empty:
            for owner, names in zip(owners, _FIELDS):
                for name in names:
                    default = 0 if name in ("_starts", "ple_lengths") else None
                    if any(item is not None if default is None
                           else type(item) is not int or item != default
                           for item in getattr(owner, name)):
                        raise ValueError("capture must start with genuinely empty state")
            if kv.offset != 0:
                raise ValueError("capture requires a cold position-zero cache")
        return owners, lists

    def _check_source(self, source):
        if self._closed or source is not self._source:
            raise ValueError("capture is closed or belongs to another source")
        owners, lists = self._validate_cache(source)
        if (any(a is not b for a, b in zip(owners, self._owners))
                or any(a is not b for a, b in zip(lists, self._lists))):
            raise ValueError("capture source containers were replaced")

    def abort(self):
        """Drop every private reference; an aborted capture cannot be resumed."""
        self._closed = True
        self._records.clear()
        self._endpoints.clear()
        self._source = None
        self._owners = self._lists = ()

    def begin_sweep(self, source, *, offset, total_tokens, tile_tokens):
        try:
            self._check_source(source)
            if (self._started or type(offset) is not int or offset != 0
                    or type(total_tokens) is not int or total_tokens != self.total_tokens
                    or type(tile_tokens) is not int or tile_tokens != self.tile_tokens):
                raise ValueError("capture sweep contract changed")
            self._validate_cache(source, empty=True)
            self._started = True
        except BaseException:
            self.abort()
            raise

    @staticmethod
    def _array(value, shape, dtype):
        if value is None or tuple(value.shape) != shape or value.dtype != dtype:
            raise ValueError("capture layer array shape/dtype mismatch")
        return value

    def _layer(self, source, layer, positions):
        """Validate LOCAL state; global offset is mixed during layer-major work."""
        kda, aux = source.kda_cache, source.qwen4_cache
        key, value, state = source.keys[layer], source.values[layer], kda._state[layer]
        conv = kda._conv[layer]
        qkey, qpos = aux.qsa_keys[layer], aux.qsa_positions[layer]
        if self._kinds[layer] == "full_attention":
            shape = (1, self._kv_heads, positions, self._head_dim)
            self._array(key, shape, self._activation_dtype)
            self._array(value, shape, self._activation_dtype)
            self._array(qkey, (1, positions, self._index_dim), self._activation_dtype)
            self._array(qpos, (1, positions), self._position_dtype)
            if state is not None or conv is not None:
                raise ValueError("unexpected full-attention recurrent state")
        else:
            if any(a is not None for a in (key, value, qkey, qpos)):
                raise ValueError("unexpected linear-attention KV/QSA state")
            self._array(state, self._state_shape, self._state_dtype)
            if type(conv) is not tuple or len(conv) != 1:
                raise ValueError("capture requires an immutable DeltaNet history tuple")
            self._array(conv[0], self._conv_shape, self._activation_dtype)
        pconv, context, length = (
            aux.ple_conv[layer], aux.ple_context[layer], aux.ple_lengths[layer])
        if layer in self._ple:
            self._array(pconv, self._ple_shape, self._activation_dtype)
            if (type(length) is not int or length != positions
                    or type(context) is not tuple or len(context) != self._context_length
                    or any(type(token) is not int for token in context)):
                raise ValueError("capture PLE position/context mismatch")
        elif pconv is not None or context is not None or type(length) is not int or length != 0:
            raise ValueError("unexpected PLE state")
        return _Layer(key, value, source._starts[layer], source._windows[layer],
                      state, conv, qkey, qpos, pconv, context, length)

    def _capture(self, record):
        histories = ((record.conv[0],) if record.conv is not None else ())
        if record.ple_conv is not None:
            histories += (record.ple_conv,)
        payload = sum(int(value.nbytes) for value in histories)
        scratch = 2 * payload
        if scratch:
            self._reserve(scratch)
        started = self._clock()
        # Raw QSA keys may not have been consumed below the indexer threshold.
        # Materialize them too, so sharing does not retain unevaluated parents.
        self._evaluate(*(value for value in (
            record.key, record.value, record.state, record.qsa_key,
            record.qsa_position, *histories) if value is not None))
        copied = []
        for value in histories:
            replacement = self._copy(value)
            self._array(replacement, tuple(value.shape), value.dtype)
            if replacement is value:
                raise ValueError("capture history copy must have an independent owner")
            copied.append(replacement)
        result = _Layer(
            record.key, record.value, record.start, record.window, record.state,
            (copied[0],) if record.conv is not None else None,
            record.qsa_key, record.qsa_position,
            copied[-1] if record.ple_conv is not None else None,
            record.ple_context, record.ple_length)
        self._arrays += len(histories)
        self._bytes += payload
        self._scratch_peak = max(self._scratch_peak, scratch)
        self._capture_seconds += self._clock() - started
        return result

    def observe_tile(self, source, *, layer, start, end):
        """Observe one unchanged evaluated tile; return whether it was captured.

        Coverage includes suffix tiles, not merely one prefix callback/layer.
        Invalid order, skipped tiles, failures and interruption poison the
        entire private builder rather than publishing a partial checkpoint.
        """
        try:
            self._check_source(source)
            if (not self._started
                    or any(type(n) is not int for n in (layer, start, end))
                    or layer != self._next_layer or not layer < self._count
                    or start != self._next_start
                    or end != min(start + self.tile_tokens, self.total_tokens)):
                raise ValueError("capture requires the original contiguous layer/tile order")
            captured = end == self.prefix_tokens
            record = self._layer(source, layer, end)
            if captured:
                if len(self._records) != layer:
                    raise ValueError("capture prefix coverage mismatch")
                self._records.append(self._capture(record))
            if end == self.total_tokens:
                self._endpoints.append(record)
                self._next_layer += 1
                self._next_start = 0
            else:
                self._next_start = end
            return captured
        except BaseException:
            self.abort()
            raise

    def finish(self, source):
        """Publish once, AFTER the caller's complete sweep/hidden restoration."""
        owned_result = None
        try:
            self._check_source(source)
            if (not self._started or self._next_layer != self._count
                    or self._next_start != 0 or len(self._records) != self._count
                    or len(self._endpoints) != self._count
                    or source.offset != self.total_tokens):
                raise ValueError("cannot publish an incomplete prefix capture")
            for layer in range(self._count):
                current = self._layer(source, layer, self.total_tokens)
                observed = self._endpoints[layer]
                # KDA has no position scalar: a prefix matrix can have the
                # same shape as its full endpoint. Freeze the exact objects
                # observed at every final tile, not just their geometry.
                if (any(getattr(current, field) is not getattr(observed, field)
                        for field in ("key", "value", "state", "conv", "qsa_key",
                                      "qsa_position", "ple_conv"))
                        or current.ple_context != observed.ple_context
                        or current.ple_length != observed.ple_length):
                    raise ValueError("completed capture endpoint changed before publication")
            result = self._empty_cache()
            owners, lists = self._validate_cache(result, empty=True)
            if (any(a is b for a in owners for b in self._owners)
                    or any(a is b for a in lists for b in self._lists)):
                raise ValueError("capture factory must return independent empty owners")
            owned_result = result
            kda, aux = result.kda_cache, result.qwen4_cache
            for layer, record in enumerate(self._records):
                result.keys[layer], result.values[layer] = record.key, record.value
                result._starts[layer], result._windows[layer] = record.start, record.window
                kda._state[layer], kda._conv[layer] = record.state, record.conv
                aux.qsa_keys[layer], aux.qsa_positions[layer] = record.qsa_key, record.qsa_position
                aux.ple_conv[layer], aux.ple_context[layer] = record.ple_conv, record.ple_context
                aux.ple_lengths[layer] = record.ple_length
                self._layer(result, layer, self.prefix_tokens)
            if result.offset != self.prefix_tokens:
                raise ValueError("captured prefix endpoint has inconsistent positions")
            stats = {
                "qwen4_fused_prefix_capture_layers": self._count,
                "qwen4_fused_prefix_capture_tokens": self.prefix_tokens,
                "qwen4_fused_prefix_capture_arrays_copied": self._arrays,
                "qwen4_fused_prefix_capture_bytes_copied": self._bytes,
                "qwen4_fused_prefix_capture_scratch_peak_bytes": self._scratch_peak,
                "qwen4_fused_prefix_capture_seconds": self._capture_seconds,
            }
        except BaseException:
            if owned_result is not None:
                # An injected factory or traceback may still own the result.
                # Clear only an already-validated independent destination;
                # an invalid/aliasing factory result might be the source.
                for owner, names in zip(owners, _FIELDS):
                    for name in names:
                        default = 0 if name in ("_starts", "ple_lengths") else None
                        getattr(owner, name)[:] = [default] * self._count
            self.abort()
            raise
        self.abort()  # Published result owns the records' arrays, not this builder.
        return result, stats
