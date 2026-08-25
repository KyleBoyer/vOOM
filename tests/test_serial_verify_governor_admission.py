"""Focused, weights-free gates for serial-verifier page admission.

These tests exercise only metadata arithmetic and fake cache/governor objects;
they never create model tensors or dispatch Metal work.
"""

from types import SimpleNamespace
import inspect

import pytest


def test_standard_mlx_quantized_layer_estimate_counts_qtensor_sidecars():
    from runtime.engine import StreamingEngine

    calls = []
    store = SimpleNamespace(
        bf16_nf12_sidecar=None,
        on_disk_quantized=True,
        mlx_quantized_resident_bytes=lambda names: (
            calls.append(tuple(names)) or 1_000
        ),
    )
    engine = SimpleNamespace(
        store=store,
        cfg=SimpleNamespace(model_type="qwen3_5"),
        _layer_names=lambda layer: [
            f"model.layers.{layer}.self_attn.q_proj.weight",
            f"model.layers.{layer}.input_layernorm.weight",
        ],
    )

    estimate = StreamingEngine._layer_fetch_bytes_estimate(engine, 7)

    assert estimate == 1_050
    assert calls == [(
        "model.layers.7.self_attn.q_proj.weight",
        "model.layers.7.input_layernorm.weight",
    )]


def test_quantized_resident_bytes_expands_physical_arrays_and_deduplicates():
    from runtime.model_loader import WeightStore

    sized = []
    store = SimpleNamespace(
        on_disk_quantized=True,
        _quant_aux={
            "layer.q.weight": SimpleNamespace(
                scales="layer.q.scales", biases="shared.biases"),
            "layer.k.weight": SimpleNamespace(
                scales="layer.k.scales", biases="shared.biases"),
        },
        storage_bytes_unknown=lambda names: [],
        storage_bytes=lambda names: sized.append(tuple(names)) or 456,
    )

    result = WeightStore.mlx_quantized_resident_bytes(
        store,
        ["layer.q.weight", "layer.k.weight", "layer.norm.weight"],
    )

    assert result == 456
    assert sized == [(
        "layer.q.weight", "layer.q.scales", "shared.biases",
        "layer.k.weight", "layer.k.scales", "layer.norm.weight",
    )]


def test_quantized_resident_bytes_refuses_incomplete_metadata():
    from runtime.model_loader import WeightStore

    store = SimpleNamespace(
        on_disk_quantized=True,
        _quant_aux={},
        storage_bytes_unknown=lambda names: [names[-1]],
        storage_bytes=lambda _names: pytest.fail(
            "partial metadata must not produce an optimistic estimate"),
    )

    assert WeightStore.mlx_quantized_resident_bytes(
        store, ["layer.weight"]
    ) == 0


class _AdmissionCache:
    def __init__(self, events, *, hit=False):
        self.events = events
        self.hit = hit

    def contains(self, key):
        self.events.append(("contains", key))
        return self.hit

    def prepare_for(self, incoming):
        self.events.append(("prepare", incoming))

    def get(self, key, names):
        self.events.append(("fetch", key, tuple(names)))
        return {}


class _AdmissionGovernor:
    def __init__(self, events, *, refuse=False):
        self.events = events
        self.refuse = refuse

    def reserve(self, incoming, *, margin, reason):
        self.events.append(("reserve", incoming, margin, reason))
        if self.refuse:
            raise MemoryError("unsafe synthetic reservation")


def _admission_engine(
    events, *, hit=False, refuse=False, exact_page_admission=True,
):
    return SimpleNamespace(
        cache=_AdmissionCache(events, hit=hit),
        governor=_AdmissionGovernor(events, refuse=refuse),
        _layer_key=lambda layer: f"layer.{layer}",
        _layer_names=lambda layer: [f"layer.{layer}.weight"],
        _layer_fetch_bytes_estimate=lambda _layer: 123,
        _layer_transient_margin=17,
        rc=SimpleNamespace(
            qwen35_serial_verify_exact_page_admission=(
                exact_page_admission
            ),
        ),
    )


def test_serial_verifier_reserves_missing_page_before_fetch():
    from runtime.engine import StreamingEngine

    events = []
    engine = _admission_engine(events)

    assert StreamingEngine._prepare_serial_verify_layer_page(engine, 3) == 123
    engine.cache.get(engine._layer_key(3), engine._layer_names(3))

    assert events == [
        ("contains", "layer.3"),
        ("prepare", 123),
        ("reserve", 123, 17, "serial-verify-layer-page"),
        ("fetch", "layer.3", ("layer.3.weight",)),
    ]


def test_serial_verifier_refuses_before_fetch_and_cache_hit_skips_admission():
    from runtime.engine import StreamingEngine

    refusal_events = []
    refusal = _admission_engine(refusal_events, refuse=True)
    with pytest.raises(MemoryError, match="unsafe synthetic reservation"):
        StreamingEngine._prepare_serial_verify_layer_page(refusal, 4)
    assert refusal_events == [
        ("contains", "layer.4"),
        ("prepare", 123),
        ("reserve", 123, 17, "serial-verify-layer-page"),
    ]

    hit_events = []
    hit = _admission_engine(hit_events, hit=True)
    assert StreamingEngine._prepare_serial_verify_layer_page(hit, 4) == 0
    assert hit_events == [("contains", "layer.4")]


def test_serial_verifier_exact_page_margin_is_explicit_opt_in():
    from runtime.engine import StreamingEngine

    events = []
    engine = _admission_engine(events, exact_page_admission=False)
    assert StreamingEngine._prepare_serial_verify_layer_page(engine, 5) == 123
    assert events[-1] == (
        "reserve", 123, 400_000_000, "serial-verify-layer-page")


def test_serial_verifier_schedules_future_pages_after_current_admission():
    """Future speculative I/O must not inflate the current-page proof."""
    from runtime.engine import StreamingEngine

    source = inspect.getsource(
        StreamingEngine.forward_tokens_serial_positions)
    loop = source.index("for layer in range(n):")
    current_admission = source.index(
        "self._prepare_serial_verify_layer_page(layer)", loop)
    current_fetch = source.index("weights = self.cache.get(", current_admission)
    future_schedule = source.index("self.prefetcher.schedule(", current_fetch)
    assert loop < current_admission < current_fetch < future_schedule
