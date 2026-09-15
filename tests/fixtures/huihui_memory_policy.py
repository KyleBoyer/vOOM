"""Explicit historical/default and September15 user-authorized memory floors."""


def available_floor(config):
    value = config.get('minimum_available_bytes', 5_300_000_000)
    if type(value) is not int or value not in (5_300_000_000, 4_500_000_000):
        raise ValueError('unqualified Huihui memory floor')
    return value


def validate(config, env, preflight):
    floor = available_floor(config)
    if floor == 4_500_000_000:
        assert 'qwen35-reserve4500-audit' in config['profiles']
        assert env.get('VMODEL_QWEN35_MIN_AVAILABLE_MB') == '4500'
        assert preflight['thresholds']['min_stable_available_bytes'] >= 5_500_000_000
        assert preflight['pressure_window']['minimum_available_bytes'] >= 5_500_000_000
    return floor
