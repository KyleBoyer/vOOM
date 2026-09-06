"""No-MLX lifecycle/ownership gates for the not-yet-wired capture helper."""

from types import SimpleNamespace as NS

import pytest

from runtime.qwen4_prefix_capture import AlignedPrefixCapture


class Array:
    def __init__(self, shape, dtype="bf16", payload="prefix"):
        self.shape, self.dtype, self.payload = shape, dtype, payload
        self.evaluated = False
        self.nbytes = 4 if dtype in ("fp32", "int32") else 2
        for size in shape:
            self.nbytes *= size


class KDA:
    def __init__(self, count):
        self._state, self._conv = [None] * count, [None] * count
        self.spill_enabled, self._spill_meta, self._factor_capture = False, {}, None


class Aux:
    def __init__(self, count):
        for name in ("qsa_keys", "qsa_positions", "qsa_pooled_keys", "ple_conv", "ple_context"):
            setattr(self, name, [None] * count)
        self.ple_lengths = [0] * count
        self.qsa_pool_cache_enabled = False


class KV:
    compressed_mla = False

    def __init__(self, count):
        self.keys, self.values = [None] * count, [None] * count
        self._starts, self._windows = [0] * count, [None] * count
        self.kda_cache, self.qwen4_cache = KDA(count), Aux(count)

    @property
    def offset(self):
        return next((self._starts[i] + a.shape[2]
                     for i, a in enumerate(self.keys) if a is not None), 0)


def config(count=4):
    return NS(
        model_type="qwen4_exp", num_hidden_layers=count,
        layer_types=tuple("full_attention" if i % 4 == 3 else "linear_attention"
                          for i in range(count)),
        qwen4_ple_layers=(1,), linear_num_key_heads=1, linear_key_head_dim=2,
        linear_num_value_heads=2, linear_value_head_dim=3, linear_conv_kernel_dim=4,
        qwen4_ple_conv_kernel_size=2, qwen4_ngram_size=3,
        qwen4_hc_count=2, hidden_size=4, num_key_value_heads=2, head_dim=4,
        qwen4_indexer_head_dim=2)


def fill_layer(source, cfg, layer, length, array=Array):
    label = f"{layer}:{length}"
    if cfg.layer_types[layer] == "full_attention":
        source.keys[layer] = array((1, cfg.num_key_value_heads, length, cfg.head_dim), payload=label + "k")
        source.values[layer] = array((1, cfg.num_key_value_heads, length, cfg.head_dim), payload=label + "v")
        source.qwen4_cache.qsa_keys[layer] = array((1, length, cfg.qwen4_indexer_head_dim), payload=label + "q")
        source.qwen4_cache.qsa_positions[layer] = array((1, length), "int32", label + "p")
    else:
        source.kda_cache._state[layer] = array(
            (1, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim),
            "fp32", label + "s")
        source.kda_cache._conv[layer] = (array(
            (1, cfg.linear_conv_kernel_dim - 1,
             2 * cfg.linear_num_key_heads * cfg.linear_key_head_dim
             + cfg.linear_num_value_heads * cfg.linear_value_head_dim), payload=label + "c"),)
    if layer in cfg.qwen4_ple_layers:
        source.qwen4_cache.ple_conv[layer] = array(
            (1, (cfg.qwen4_ple_conv_kernel_size - 1) * cfg.qwen4_ngram_size,
             cfg.qwen4_hc_count * cfg.hidden_size), payload=label + "e")
        source.qwen4_cache.ple_context[layer] = (length - 2, length - 1)
        source.qwen4_cache.ple_lengths[layer] = length


def setup(*, count=4, prefix=4, total=7, tile=4, change_source=None,
          change_cfg=None, **overrides):
    cfg, source = config(count), KV(count)
    if change_source:
        change_source(source)
    if change_cfg:
        change_cfg(cfg)
    events, factories = [], []

    def evaluate(*arrays):
        events.append(("eval", arrays))
        for value in arrays:
            value.evaluated = True

    def copy(value):
        assert value.evaluated
        events.append(("copy", value))
        return Array(value.shape, value.dtype, value.payload)

    def factory():
        result = KV(count)
        factories.append(result)
        return result

    kwargs = dict(prefix_tokens=prefix, total_tokens=total, tile_tokens=tile,
                  cache_types=(KV, KDA, Aux), empty_cache=factory,
                  activation_dtype="bf16", state_dtype="fp32", position_dtype="int32",
                  copy_bits=copy, evaluate=evaluate,
                  reserve=lambda size: events.append(("reserve", size)))
    kwargs.update(overrides)
    capture = AlignedPrefixCapture(source, cfg, **kwargs)
    return NS(cfg=cfg, source=source, capture=capture, events=events, factories=factories)


def begin(case):
    case.capture.begin_sweep(case.source, offset=0, total_tokens=case.capture.total_tokens,
                             tile_tokens=case.capture.tile_tokens)


def sweep(case):
    for layer in range(case.cfg.num_hidden_layers):
        for start in range(0, case.capture.total_tokens, case.capture.tile_tokens):
            end = min(start + case.capture.tile_tokens, case.capture.total_tokens)
            fill_layer(case.source, case.cfg, layer, end)
            assert case.capture.observe_tile(case.source, layer=layer, start=start, end=end) == (
                end == case.capture.prefix_tokens)


def assert_dead(case):
    assert case.capture._closed
    assert case.capture._source is None
    assert not case.capture._records and not case.capture._endpoints
    assert not case.capture._owners and not case.capture._lists
    with pytest.raises(ValueError):
        case.capture.finish(case.source)


@pytest.mark.parametrize("count,prefix,total,tile", [(4, 4, 7, 4), (8, 8, 13, 4), (48, 16, 19, 8)])
def test_complete_original_tile_coverage_publishes_only_prefix_state(count, prefix, total, tile):
    case = setup(count=count, prefix=prefix, total=total, tile=tile)
    begin(case)
    sweep(case)
    assert not case.factories  # Not even an empty destination exists before finish.
    result, stats = case.capture.finish(case.source)
    assert len(case.factories) == 1 and result is case.factories[0]
    assert result.offset == prefix and case.source.offset == total
    assert stats["qwen4_fused_prefix_capture_layers"] == count
    assert stats["qwen4_fused_prefix_capture_arrays_copied"] == count * 3 // 4 + 1
    assert stats["qwen4_fused_prefix_capture_bytes_copied"] == count * 3 // 4 * 60 + 48
    assert stats["qwen4_fused_prefix_capture_scratch_peak_bytes"] == 216
    assert stats["qwen4_fused_prefix_capture_seconds"] >= 0
    for layer, kind in enumerate(case.cfg.layer_types):
        if kind == "full_attention":
            assert result.keys[layer].payload == f"{layer}:{prefix}k"
            assert result.keys[layer].evaluated
            assert result.qwen4_cache.qsa_keys[layer].evaluated
            assert result.qwen4_cache.qsa_positions[layer].evaluated
            assert case.source.keys[layer].payload == f"{layer}:{total}k"
        else:
            assert result.kda_cache._state[layer].payload == f"{layer}:{prefix}s"
            assert result.kda_cache._state[layer].evaluated
            assert result.kda_cache._conv[layer][0].payload == f"{layer}:{prefix}c"
            assert case.source.kda_cache._state[layer].payload == f"{layer}:{total}s"
    assert result.qwen4_cache.ple_lengths[1] == prefix
    assert result.qwen4_cache.ple_context[1] == (prefix - 2, prefix - 1)
    assert result.qwen4_cache.ple_conv[1].payload == f"1:{prefix}e"
    assert not result.qwen4_cache.qsa_pool_cache_enabled
    assert result.qwen4_cache.qsa_pooled_keys == [None] * count
    assert_dead(case)


def test_capture_shares_evaluated_arrays_copies_histories_and_never_changes_source():
    case = setup()
    begin(case)
    fill_layer(case.source, case.cfg, 0, 4)
    old_state, old_conv = case.source.kda_cache._state[0], case.source.kda_cache._conv[0]
    case.capture.observe_tile(case.source, layer=0, start=0, end=4)
    record = case.capture._records[0]
    assert record.state is old_state
    assert record.conv is not old_conv and record.conv[0] is not old_conv[0]
    assert record.conv[0].payload == old_conv[0].payload
    assert case.source.kda_cache._state[0] is old_state
    assert case.source.kda_cache._conv[0] is old_conv
    assert [event[0] for event in case.events] == ["reserve", "eval", "copy"]
    assert case.events[0][1] == 120  # Reserve before materialization/copy.
    assert not case.factories


@pytest.mark.parametrize("args", [
    dict(prefix=0), dict(prefix=True), dict(prefix=3), dict(prefix=7, tile=1),
    dict(prefix=8), dict(total=0), dict(tile=0), dict(tile=2.0),
])
def test_invalid_prefix_contract_fails_before_allocation(args):
    with pytest.raises(ValueError):
        setup(**args)


@pytest.mark.parametrize("field,value", [
    ("model_type", "qwen3_5"), ("num_hidden_layers", 3),
    ("layer_types", ("linear_attention",) * 4),
    ("qwen4_ple_layers", (1, 1)), ("qwen4_ple_layers", (-1,)),
    ("qwen4_ple_layers", (True,)), ("head_dim", 0), ("hidden_size", 1.5),
])
def test_invalid_geometry_fails_before_allocation(field, value):
    with pytest.raises(ValueError):
        setup(change_cfg=lambda cfg: setattr(cfg, field, value))


@pytest.mark.parametrize("mutation", [
    lambda kv: setattr(kv, "compressed_mla", True),
    lambda kv: setattr(kv, "dsa", object()),
    lambda kv: setattr(kv, "values", kv.keys),
    lambda kv: kv._starts.__setitem__(1, 1),
    lambda kv: kv._windows.__setitem__(0, 2),
    lambda kv: kv.keys.__setitem__(3, Array((1, 2, 4, 4))),
    lambda kv: kv.kda_cache._conv.__setitem__(0, (Array((1, 3, 10)),)),
    lambda kv: setattr(kv.kda_cache, "spill_enabled", True),
    lambda kv: setattr(kv.kda_cache, "_spill_meta", {0: "disk"}),
    lambda kv: setattr(kv.kda_cache, "_factor_capture", []),
    lambda kv: setattr(kv.qwen4_cache, "qsa_pool_cache_enabled", True),
    lambda kv: kv.qwen4_cache.qsa_pooled_keys.__setitem__(3, Array((1,))),
    lambda kv: kv.qwen4_cache.ple_lengths.__setitem__(1, 1),
    lambda kv: kv.qwen4_cache.ple_context.__setitem__(1, (1, 2)),
    lambda kv: setattr(kv.qwen4_cache, "qsa_keys", [None]),
])
def test_only_genuinely_empty_plain_independently_owned_caches_are_accepted(mutation):
    with pytest.raises(ValueError):
        setup(change_source=mutation)


@pytest.mark.parametrize("args", [dict(offset=1), dict(total_tokens=8), dict(tile_tokens=2)])
def test_begin_must_match_sweep_contract(args):
    case = setup()
    kwargs = dict(offset=0, total_tokens=7, tile_tokens=4) | args
    with pytest.raises(ValueError):
        case.capture.begin_sweep(case.source, **kwargs)
    assert_dead(case)


@pytest.mark.parametrize("operation", ["no_begin", "second_begin", "source_advanced", "wrong_source"])
def test_begin_lifecycle(operation):
    case = setup()
    with pytest.raises(ValueError):
        if operation == "no_begin":
            case.capture.observe_tile(case.source, layer=0, start=0, end=4)
        elif operation == "second_begin":
            begin(case)
            begin(case)
        elif operation == "source_advanced":
            fill_layer(case.source, case.cfg, 0, 4)
            begin(case)
        else:
            case.capture.begin_sweep(KV(4), offset=0, total_tokens=7, tile_tokens=4)
    assert_dead(case)


@pytest.mark.parametrize("args", [dict(layer=1), dict(start=1), dict(end=3), dict(end=7), dict(layer=False)])
def test_original_contiguous_tile_order_is_mandatory(args):
    case = setup()
    begin(case)
    fill_layer(case.source, case.cfg, 0, 4)
    with pytest.raises(ValueError):
        case.capture.observe_tile(case.source, **(dict(layer=0, start=0, end=4) | args))
    assert_dead(case)


@pytest.mark.parametrize("operation", ["duplicate", "skip_suffix", "finish_early", "replace_list", "replace_aux"])
def test_partial_or_replaced_state_never_publishes(operation):
    case = setup()
    begin(case)
    fill_layer(case.source, case.cfg, 0, 4)
    case.capture.observe_tile(case.source, layer=0, start=0, end=4)
    with pytest.raises(ValueError):
        if operation == "duplicate":
            case.capture.observe_tile(case.source, layer=0, start=0, end=4)
        elif operation == "skip_suffix":
            case.capture.observe_tile(case.source, layer=1, start=0, end=4)
        elif operation == "finish_early":
            case.capture.finish(case.source)
        else:
            fill_layer(case.source, case.cfg, 0, 7)
            if operation == "replace_list":
                case.source.kda_cache._state = list(case.source.kda_cache._state)
            else:
                case.source.qwen4_cache = Aux(4)
            case.capture.observe_tile(case.source, layer=0, start=4, end=7)
    assert not case.factories
    assert_dead(case)


@pytest.mark.parametrize("failure", ["reserve", "evaluate", "copy", "alias", "shape", "dtype", "interrupt"])
def test_capture_failure_releases_prior_layers_and_preserves_source(failure):
    case = setup()
    begin(case)
    for start, end in ((0, 4), (4, 7)):
        fill_layer(case.source, case.cfg, 0, end)
        case.capture.observe_tile(case.source, layer=0, start=start, end=end)
    fill_layer(case.source, case.cfg, 1, 4)
    original = case.source.kda_cache._conv[1], case.source.qwen4_cache.ple_conv[1]

    def fail(*args):
        if failure == "interrupt":
            raise KeyboardInterrupt()
        raise MemoryError("injected")

    if failure in ("reserve", "evaluate"):
        setattr(case.capture, "_" + failure, fail)
    elif failure in ("copy", "interrupt"):
        case.capture._copy = fail
    elif failure == "alias":
        case.capture._copy = lambda value: value
    elif failure == "shape":
        case.capture._copy = lambda value: Array((1,))
    else:
        case.capture._copy = lambda value: Array(value.shape, "fp32")
    with pytest.raises((MemoryError, ValueError, KeyboardInterrupt)):
        case.capture.observe_tile(case.source, layer=1, start=0, end=4)
    assert case.source.kda_cache._conv[1] is original[0]
    assert case.source.qwen4_cache.ple_conv[1] is original[1]
    assert not case.factories
    assert_dead(case)


@pytest.mark.parametrize("failure", ["missing_suffix", "shifted", "pool", "factory", "alias_factory"])
def test_finish_revalidates_complete_endpoint_and_factory(failure):
    case = setup()
    begin(case)
    sweep(case)
    if failure == "missing_suffix":
        fill_layer(case.source, case.cfg, 0, 4)
        # KDA has no position scalar: use the PLE length to expose incomplete state.
        fill_layer(case.source, case.cfg, 1, 4)
    elif failure == "shifted":
        case.source._starts[3] = 1
    elif failure == "pool":
        case.source.qwen4_cache.qsa_pooled_keys[3] = Array((1,))
    elif failure == "factory":
        case.capture._empty_cache = lambda: object()
    else:
        case.capture._empty_cache = lambda: case.source
    with pytest.raises(ValueError):
        case.capture.finish(case.source)
    assert_dead(case)


def test_explicit_abort_discards_partial_capture_without_factory_or_source_mutation():
    case = setup()
    begin(case)
    fill_layer(case.source, case.cfg, 0, 4)
    case.capture.observe_tile(case.source, layer=0, start=0, end=4)
    state = case.source.kda_cache._state[0]
    case.capture.abort()
    assert case.source.kda_cache._state[0] is state
    assert not case.factories
    assert_dead(case)


def test_nonboundary_tile_also_requires_complete_local_state():
    case = setup(prefix=8, total=13, tile=4)
    begin(case)
    with pytest.raises(ValueError, match="shape/dtype"):
        case.capture.observe_tile(case.source, layer=0, start=0, end=4)
    assert_dead(case)


def test_nonscalar_kda_endpoint_rewind_is_detected_even_without_ple_metadata():
    case = setup()
    begin(case)
    sweep(case)
    retained = case.capture._records[0]
    case.source.kda_cache._state[0] = retained.state
    case.source.kda_cache._conv[0] = retained.conv
    with pytest.raises(ValueError, match="endpoint changed"):
        case.capture.finish(case.source)
    assert not case.factories
    assert_dead(case)


@pytest.mark.parametrize("error", [MemoryError, KeyboardInterrupt])
def test_partial_destination_is_emptied_if_publication_fails(error):
    case = setup()
    begin(case)
    sweep(case)
    source_state = case.source.kda_cache._state[0]
    validate = case.capture._layer

    def fail_on_second_destination_layer(source, layer, positions):
        if source is not case.source and layer == 1:
            raise error("injected mid-publication")
        return validate(source, layer, positions)

    case.capture._layer = fail_on_second_destination_layer
    with pytest.raises(error):
        case.capture.finish(case.source)
    assert len(case.factories) == 1
    result = case.factories[0]
    assert result.offset == 0 and case.source.offset == 7
    assert result.kda_cache._state == [None] * 4
    assert result.kda_cache._conv == [None] * 4
    assert result.qwen4_cache.ple_conv == [None] * 4
    assert result.qwen4_cache.ple_context == [None] * 4
    assert result.qwen4_cache.ple_lengths == [0] * 4
    assert case.source.kda_cache._state[0] is source_state
    assert_dead(case)


def test_aliasing_factory_failure_does_not_clear_authoritative_source():
    case = setup()
    begin(case)
    sweep(case)
    case.capture._empty_cache = lambda: case.source
    original_state = case.source.kda_cache._state[0]
    with pytest.raises(ValueError):
        case.capture.finish(case.source)
    assert case.source.offset == 7
    assert case.source.kda_cache._state[0] is original_state
    assert_dead(case)


@pytest.mark.parametrize("mutation", [
    lambda kv: setattr(kv.kda_cache._state[0], "dtype", "bf16"),
    lambda kv: setattr(kv.kda_cache._state[0], "shape", (1, 2, 3, 2)),
    lambda kv: setattr(kv.kda_cache._conv[0][0], "shape", (1, 2, 10)),
    lambda kv: setattr(kv.kda_cache._conv[0][0], "dtype", "fp32"),
    lambda kv: kv.kda_cache._conv.__setitem__(0, list(kv.kda_cache._conv[0])),
    lambda kv: kv.keys.__setitem__(0, Array((1, 2, 4, 4))),
    lambda kv: kv.qwen4_cache.qsa_keys.__setitem__(0, Array((1, 4, 2))),
    lambda kv: kv.qwen4_cache.ple_context.__setitem__(0, (1, 2)),
])
def test_invalid_linear_layer_geometry_aborts_before_copy(mutation):
    case = setup()
    begin(case)
    fill_layer(case.source, case.cfg, 0, 4)
    mutation(case.source)
    with pytest.raises(ValueError):
        case.capture.observe_tile(case.source, layer=0, start=0, end=4)
    assert not case.events
    assert_dead(case)


@pytest.mark.parametrize("mutation", [
    lambda kv: setattr(kv.keys[3], "shape", (1, 2, 3, 4)),
    lambda kv: setattr(kv.values[3], "dtype", "fp32"),
    lambda kv: setattr(kv.qwen4_cache.qsa_keys[3], "shape", (1, 3, 2)),
    lambda kv: setattr(kv.qwen4_cache.qsa_positions[3], "dtype", "bf16"),
    lambda kv: kv.kda_cache._state.__setitem__(3, Array((1, 2, 2, 3), "fp32")),
    lambda kv: kv.qwen4_cache.ple_lengths.__setitem__(3, 4),
])
def test_invalid_full_attention_layer_geometry_releases_prior_prefixes(mutation):
    case = setup()
    begin(case)
    for layer in range(3):
        for start, end in ((0, 4), (4, 7)):
            fill_layer(case.source, case.cfg, layer, end)
            case.capture.observe_tile(case.source, layer=layer, start=start, end=end)
    fill_layer(case.source, case.cfg, 3, 4)
    mutation(case.source)
    with pytest.raises(ValueError):
        case.capture.observe_tile(case.source, layer=3, start=0, end=4)
    assert not case.factories
    assert_dead(case)
