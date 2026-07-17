"""Exact ordering gates for bulk GLM and GPT-OSS route materialization."""

import mlx.core as mx
import pytest

from runtime.glm import _group_routes as group_glm_routes
from runtime.gptoss import _group_routes as group_gptoss_routes


def _scalar_reference(indices, weights):
    groups = {}
    for position in range(indices.shape[1]):
        for lane in range(indices.shape[2]):
            groups.setdefault(int(indices[0, position, lane]), []).append(
                (position, float(weights[0, position, lane])))
    return groups


@pytest.mark.parametrize("group_routes", [group_glm_routes, group_gptoss_routes])
def test_bulk_route_transfer_matches_scalar_order_and_values(group_routes):
    indices = mx.array([[
        [7, 2, 5, 1],
        [2, 7, 3, 5],
        [5, 3, 2, 7],
    ]], dtype=mx.int32)
    weights = mx.array([[
        [0.40, 0.30, 0.20, 0.10],
        [0.35, 0.30, 0.20, 0.15],
        [0.45, 0.25, 0.20, 0.10],
    ]], dtype=mx.float32)
    mx.eval(indices, weights)

    expected = _scalar_reference(indices, weights)
    actual = group_routes(indices, weights)

    assert actual == expected
    assert list(actual) == list(expected), "first-seen expert order changed"
    assert actual[7] == expected[7], "per-expert accumulation order changed"
