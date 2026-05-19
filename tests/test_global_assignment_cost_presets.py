"""Tests for globally advertised association-cost presets."""

# pylint: disable=protected-access

from pathlib import Path
from typing import cast

import numpy as np
from bayescatrack.association.pyrecest_global_assignment import (
    AssociationCost,
    _cost_kwargs_for_method,
    build_registered_pairwise_costs,
)
from bayescatrack.core.bridge import CalciumPlaneData, Track2pSession
from bayescatrack.experiments.track2p_benchmark import build_arg_parser


def test_every_cli_advertised_cost_has_global_assignment_preset() -> None:
    choices = _benchmark_cost_choices()

    assert "registered-soft-iou" in choices
    for cost_name in choices:
        kwargs = _cost_kwargs_for_method(cast(AssociationCost, cost_name))
        assert isinstance(kwargs, dict)


def test_registered_soft_iou_runs_through_registered_pairwise_path() -> None:
    reference_masks = np.zeros((1, 10, 10), dtype=bool)
    measurement_masks = np.zeros((2, 10, 10), dtype=bool)
    reference_masks[0, 2:4, 2:4] = True
    measurement_masks[0, 2:4, 4:6] = True
    measurement_masks[1, 7:9, 7:9] = True

    sessions = (
        Track2pSession(
            session_dir=Path("session0"),
            session_name="session0",
            session_date=None,
            plane_data=CalciumPlaneData(reference_masks),
        ),
        Track2pSession(
            session_dir=Path("session1"),
            session_name="session1",
            session_date=None,
            plane_data=CalciumPlaneData(measurement_masks),
        ),
    )

    pairwise_costs = build_registered_pairwise_costs(
        sessions,
        max_gap=1,
        cost="registered-soft-iou",
        transform_type="none",
    )
    cost_matrix = pairwise_costs[(0, 1)]

    assert cost_matrix.shape == (1, 2)
    assert np.all(np.isfinite(cost_matrix))
    assert cost_matrix[0, 0] < cost_matrix[0, 1]


def _benchmark_cost_choices() -> tuple[str, ...]:
    parser = build_arg_parser()
    for action in parser._actions:
        if "--cost" in action.option_strings:
            return tuple(str(choice) for choice in action.choices or ())
    raise AssertionError("benchmark parser does not define a --cost option")
