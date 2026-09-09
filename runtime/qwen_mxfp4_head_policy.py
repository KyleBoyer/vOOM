"""Explicit native-head placement guards and content-free I/O accounting.

No MLX import. This profile preserves the selected MXFP4 representation, not
the original BF16 release. Byte counters describe successful returned reads,
not uncached physical disk traffic, and remain separate from WeightStore.
"""

ROWS = (0, 8192, 32768, 65536)
COUNTERS = ('full_scan_calls', 'completed_scan_calls', 'failed_scan_calls',
            'full_read_extents', 'full_bytes_read', 'full_read_ns', 'full_scan_ns',
            'reservation_calls', 'reservation_ns', 'upload_ns', 'projection_ns')


def parse_rows(value):
    if isinstance(value, str) and value in tuple(map(str, ROWS)):
        return int(value)
    if type(value) is int and value in ROWS:
        return value
    raise ValueError('Qwen MXFP4 head rows must be 0, 8192, 32768, or 65536')


def identity(rows):
    rows = parse_rows(rows)
    return f'+native-mxfp4-head-rows-v1-{rows}' if rows else ''


def configure(rc, rows):
    """Set native placement on a NEW config before engine/cache construction.

    Dense fast RuntimeConfig defaults quant_lm_head to True even when the
    corresponding optional environment request is zero. Native selection
    explicitly excludes that transform; disabled mode keeps the old policy.
    """
    rc.qwen35_mxfp4_head_rows = parse_rows(rows)
    if rc.qwen35_mxfp4_head_rows:
        rc.quant_lm_head = False


def validate_cache_mb(value, rows):
    """A streamed native head no longer requires the historical whole-head cache.

    This is a retention-capacity bound, not a system-available reserve or a
    permission to allocate. Native source/pin guards and live governor still
    run before payload work. Leave every disabled-mode bound unchanged.
    """
    minimum = 256 if parse_rows(rows) else 1500
    if type(value) is not int or not minimum <= value <= 8500:
        raise ValueError(
            f'VMODEL_QWEN35_WEIGHT_CACHE_MB must be in [{minimum}, 8500]')


def validate(rc, cfg, store):
    rows = parse_rows(rc.qwen35_mxfp4_head_rows)
    if not rows:
        return
    if (cfg.model_type != 'qwen3_5' or cfg.num_experts
            or cfg.tie_word_embeddings or cfg.hidden_size != 5120
            or cfg.vocab_size != 248320 or not rc.governor):
        raise ValueError('native MXFP4 head requires untied dense Huihui and governor')
    for flag in ('pin_lm_head', 'rerank_lm_head', 'quant_lm_head',
                 'qwen35_serial_verify_suspend_lm_head', 'qwen35_phase_head_pre_admit',
                 'qwen4_phase_lm_head', 'glm53_phase_lm_head',
                 'grammar_jump_forward_lossy'):
        if getattr(rc, flag, True):
            raise ValueError('native MXFP4 head conflicts with '+flag)
    # Resolve lazy fast-tier policy before inspecting it; the row reader must
    # not silently bypass a selected representation or source overlay.
    store._ensure_raw_fast_tier_loaded()
    for flag in ('vpack2', 'packed', 'gguf',
                 'k3_scale_sidecar', 'bf16_nf12_sidecar', '_ct_int4_aux',
                 '_ct_mxfp4_aux', '_glm53_fp8_aux', '_dsv4_aux',
                 '_qwen4_fused_expert_slices'):
        if getattr(store, flag, True):
            raise ValueError('native MXFP4 head conflicts with source overlay '+flag)
    manifest = getattr(store, '_raw_fast_tier_manifest', None)
    if not isinstance(manifest, dict):
        raise ValueError('native MXFP4 head needs resolved fast-tier metadata')
    # Raw WeightStore overlays select individual tensor names, not entire
    # shards. A body-only fast tier cannot change the head's source. Preserve
    # that useful independent-device placement, but reject either head member.
    for name in ('lm_head.weight', 'lm_head.scales'):
        if name in manifest or store._real_name.get(name, name) != name:
            raise ValueError('native MXFP4 head conflicts with head source overlay')
    aux = store._quant_aux.get('lm_head.weight')
    if (aux is None or (aux.bits, aux.group_size, aux.mode, aux.scales, aux.biases)
            != (4, 32, 'mxfp4', 'lm_head.scales', None)):
        raise ValueError('native MXFP4 head requires the unchanged packed head pair')


def validate_forward(rows, positions, *, serial):
    if rows and (type(positions) is not int or not 1 <= positions <= (64 if serial else 1)):
        raise ValueError('native MXFP4 head requires singleton forward or 1..64 serial positions')


def snapshot(engine):
    if not getattr(getattr(engine, 'rc', None), 'qwen35_mxfp4_head_rows', 0):
        return None
    head = engine._streamed_lm_head
    values = head.full_scan_telemetry()
    result = {k: values[k] for k in COUNTERS}
    if any(type(v) is not int or v < 0 for v in result.values()):
        raise ValueError('invalid native MXFP4 head counters')
    return result


def delta(before, after):
    if before is None and after is None:
        return None
    if before is None or after is None:
        raise ValueError('native MXFP4 head telemetry coverage changed')
    result = {k: after[k] - before[k] for k in COUNTERS}
    if any(v < 0 for v in result.values()):
        raise ValueError('native MXFP4 head counters regressed')
    return result


def accumulate(total, before, after):
    change = delta(before, after)
    if change is not None:
        for key, value in change.items():
            total[key] = total.get(key, 0) + value


def publish(engine, stats, before, prefill_after, *, draft=None):
    after = snapshot(engine)
    total = delta(before, after)
    if total is None:
        return
    value = dict(schema='voom.native-mxfp4-head-io.v1',
        block_rows=engine.rc.qwen35_mxfp4_head_rows, total=total,
        prefill=delta(before, prefill_after), decode=delta(prefill_after, after),
        scope='successful raw head reads; separate from WeightStore; not physical disk bytes',
        weight_store_plus_head_bytes=stats['weight_store_bytes_read']+total['full_bytes_read'])
    if draft is not None:
        value['draft'] = {k: draft.get(k, 0) for k in COUNTERS}
        value['target'] = delta(value['draft'], total)
    stats['qwen35_mxfp4_head_io'] = value
