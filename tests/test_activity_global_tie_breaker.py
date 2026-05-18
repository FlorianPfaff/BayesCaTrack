from __future__ import annotations

import inspect

import numpy as np
import numpy.testing as npt
import pytest
from bayescatrack.association.activity_tie_breaker import (
    activity_tie_breaker_cost_matrix,
)
from bayescatrack.association.pyrecest_global_assignment import (
    build_registered_pairwise_costs,
    solve_global_assignment_for_sessions,
)


def test_activity_tie_breaker_cost_matrix_scales_selected_component() -> None:
    pairwise_components = {
        "activity_tiebreaker_cost": np.array(
            [[0.0, 0.5], [1.0, np.nan]], dtype=float
        )
    }

    tie_breaker_cost = activity_tie_breaker_cost_matrix(
        pairwise_components, weight=0.1
    )

    npt.assert_allclose(tie_breaker_cost, np.array([[0.0, 0.05], [0.1, 0.05]]))


def test_activity_tie_breaker_cost_matrix_validates_inputs() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        activity_tie_breaker_cost_matrix(
            {"activity_tiebreaker_cost": np.zeros((1, 1))}, weight=-1.0
        )
    with pytest.raises(KeyError, match="missing"):
        activity_tie_breaker_cost_matrix({}, component_name="missing")
    with pytest.raises(ValueError, match="two-dimensional"):
        activity_tie_breaker_cost_matrix(
            {"activity_tiebreaker_cost": np.zeros((1, 1, 1))}
        )


def test_global_assignment_exposes_activity_tie_breaker_parameters() -> None:
    for function in (build_registered_pairwise_costs, solve_global_assignment_for_sessions):
        signature = inspect.signature(function)
        assert "activity_tie_breaker_weight" in signature.parameters
        assert "activity_tie_breaker_component" in signature.parameters
        assert "activity_trace_source" in signature.parameters
        assert "activity_event_threshold" in signature.parameters
        assert signature.parameters["activity_tie_breaker_weight"].default == 0.0
        assert (
            signature.parameters["activity_tie_breaker_component"].default
            == "activity_tiebreaker_cost"
        )
