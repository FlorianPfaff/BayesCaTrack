"""Tests for keeping calcium activity as a weak association cue."""

import numpy as np
import pytest

from bayescatrack.association.pyrecest_global_assignment import (
    _add_activity_tiebreaker_cost,
)


def test_activity_tiebreaker_is_bounded_and_availability_gated() -> None:
    base_cost = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )
    components = {
        "activity_similarity_cost": np.array(
            [
                [0.0, 1.0],
                [0.5, 0.25],
            ]
        ),
        "activity_similarity_available": np.array(
            [
                [1.0, 1.0],
                [0.0, 1.0],
            ]
        ),
    }

    cost = _add_activity_tiebreaker_cost(base_cost, components, weight=0.05)

    np.testing.assert_allclose(
        cost,
        np.array(
            [
                [1.0, 2.05],
                [3.0, 4.0125],
            ]
        ),
    )
    assert float(np.max(cost - base_cost)) <= 0.05


def test_activity_tiebreaker_clips_invalid_activity_costs() -> None:
    base_cost = np.zeros((1, 4), dtype=float)
    components = {
        "activity_similarity_cost": np.array([[np.nan, -2.0, 0.5, np.inf]]),
        "activity_similarity_available": np.ones((1, 4), dtype=float),
    }

    cost = _add_activity_tiebreaker_cost(base_cost, components, weight=0.1)

    np.testing.assert_allclose(cost, np.array([[0.0, 0.0, 0.05, 0.1]]))


def test_activity_tiebreaker_rejects_negative_weight() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _add_activity_tiebreaker_cost(np.zeros((1, 1)), {}, weight=-0.1)
