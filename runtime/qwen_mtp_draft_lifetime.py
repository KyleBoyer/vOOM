"""Default-off native draft/head lifetime boundary; no sampling decisions."""
import os

FLAG = 'VMODEL_QWEN_MTP_RELEASE_DRAFT_BEFORE_HEAD'


def applied(stats):
    """Scalar evidence gate shared by short-request and full-workflow audits."""
    return (isinstance(stats, dict)
        and type(stats.get('enabled')) is int and stats['enabled'] == 1
        and all(type(stats.get(k)) is int and stats[k] > 0 for k in (
            'releases', 'reloads', 'logical_released_bytes',
            'observed_active_released_bytes')))


def configure(drafter, target, native_type, environ=None):
    value = (os.environ if environ is None else environ).get(FLAG, '0')
    if value not in ('0', '1'):
        raise ValueError(FLAG + ' must be 0 or 1')
    active = value == '1'
    native = type(drafter) is native_type
    if active and (not native
            or getattr(target.cfg, 'model_type', None) != 'qwen3_5'
            or target.cfg.num_experts
            or not getattr(target.rc, 'qwen35_mxfp4_head_rows', 0)
            or drafter.request_weight_representation not in (
                'released-bf16', 'mxfp4-q4-g32', 'hybrid-bf16-attn-mxfp4-mlp')):
        raise ValueError('draft/head release requires explicit dense native MTP and row-streamed head')
    if native:
        drafter._head_release_enabled = active
        drafter._head_release_stats = dict(enabled=int(active), releases=0,
            reloads=0, logical_released_bytes=0, observed_active_released_bytes=0,
            reload_s=0.0, release_s=0.0,
            scope='draft weights only; same-request counters; observed release is not admission credit')
