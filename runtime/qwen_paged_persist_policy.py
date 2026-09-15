"""Storage-only bounds for opt-in Qwen paged prefix journals."""

CHECKPOINTS = 'VMODEL_QWEN35_HOT_KV_PERSIST_MAX_CHECKPOINTS'
MAX_MB = 'VMODEL_QWEN35_HOT_KV_PERSIST_MAX_MB'


def request_identity(env):
    mode = env.get('VMODEL_QWEN35_PAGED_KV_PERSIST', '0')
    if mode not in ('0', '1'):
        raise ValueError('VMODEL_QWEN35_PAGED_KV_PERSIST must be 0 or 1')
    return (mode, *limits(env, default_checkpoints=64, default_max_mb=0))


def limits(env, *, default_checkpoints, default_max_mb):
    values = []
    for key, default, maximum in ((CHECKPOINTS, default_checkpoints, 64),
                                 (MAX_MB, default_max_mb, 16384)):
        raw = env.get(key)
        if raw is None:
            values.append(default)
            continue
        if not isinstance(raw, str) or not raw.isascii() or not raw.isdigit():
            raise ValueError(key + ' must be an unsigned integer')
        value = int(raw)
        if not 1 <= value <= maximum:
            raise ValueError(key + ' exceeds the qualified storage range')
        values.append(value)
    return tuple(values)
