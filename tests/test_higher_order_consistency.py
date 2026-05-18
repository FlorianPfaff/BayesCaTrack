"""Tests for higher-order triplet consistency penalties."""

import numpy as np
import numpy.testing as npt

from bayescatrack.association.higher_order_consistency import (
    HigherOrderConsistencyConfig,
    apply_higher_order_consistency,
    triplet_consistency_penalty,
)


def _three_session_costs() -> dict[tuple[int, int], np.ndarray]:
    return {
        (0, 1): np.array([[0.1, 3.0], [3.0, 0.1]], dtype=float),
        (1, 2): np.array([[0.1, 3.0], [3.0, 0.1]], dtype=float),
        (0, 2): np.array([[0.2, 0.25], [0.3, 0.2]], dtype=float),
    }


def test_triplet_penalty_uses_bridge_session_for_skip_edges():
    config = HigherOrderConsistencyConfig(
        triplet_weight=1.0,
        support_top_k=2,
        support_cost_cap=0.5,
        max_penalty=2.0,
    )

    penalty = triplet_consistency_penalty(
        _three_session_costs(),
        edge=(0, 2),
        session_sizes=(2, 2, 2),
        config=config,
    )

    expected = np.array([[0.0, 2.0], [2.0, 0.0]], dtype=float)
    npt.assert_allclose(penalty, expected)


def test_apply_higher_order_consistency_preserves_gated_costs():
    costs = _three_session_costs()
    costs[(0, 2)] = costs[(0, 2)].copy()
    costs[(0, 2)][0, 1] = 1.0e6
    config = HigherOrderConsistencyConfig(
        triplet_weight=1.0,
        support_top_k=2,
        support_cost_cap=0.5,
        max_penalty=2.0,
        large_cost=1.0e6,
    )

    adjusted = apply_higher_order_consistency(
        costs,
        session_sizes=(2, 2, 2),
        config=config,
    )

    assert adjusted[(0, 2)][0, 1] == 1.0e6
    assert adjusted[(0, 2)][1, 0] > costs[(0, 2)][1, 0]
    assert adjusted[(0, 2)][0, 0] == costs[(0, 2)][0, 0]


def test_zero_weight_returns_numeric_copies_without_changes():
    costs = _three_session_costs()

    adjusted = apply_higher_order_consistency(
        costs,
        session_sizes=(2, 2, 2),
        config=HigherOrderConsistencyConfig(triplet_weight=0.0),
    )

    for edge, matrix in costs.items():
        npt.assert_allclose(adjusted[edge], matrix)
        assert adjusted[edge] is not matrix
