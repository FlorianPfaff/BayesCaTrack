from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import numpy.testing as npt
import pytest
from bayescatrack.association.bootstrap_registration import (
    BootstrapRegistrationConfig,
    fit_residual_transform_from_roi_pairs,
    refine_registered_planes_from_assignment,
)
from bayescatrack.association.pyrecest_global_assignment import GlobalAssignmentRun
from bayescatrack.core.bridge import CalciumPlaneData, Track2pSession


def _plane_from_centers(centers_yx, *, image_shape=(40, 40), source="synthetic"):
    masks = np.zeros((len(centers_yx), *image_shape), dtype=bool)
    for roi_index, (center_y, center_x) in enumerate(centers_yx):
        y = int(center_y)
        x = int(center_x)
        masks[roi_index, y - 1 : y + 2, x - 1 : x + 2] = True
    fov = np.zeros(image_shape, dtype=float)
    return CalciumPlaneData(roi_masks=masks, fov=fov, source=source)


def _session(name: str, plane: CalciumPlaneData) -> Track2pSession:
    return Track2pSession(
        session_dir=Path(name),
        session_name=name,
        session_date=date(2024, 1, 1),
        plane_data=plane,
    )


def test_fit_residual_affine_transform_from_assignment_anchors() -> None:
    reference = _plane_from_centers([(10, 10), (10, 25), (25, 10), (25, 25)])
    registered = _plane_from_centers([(12, 7), (12, 22), (27, 7), (27, 22)])
    pairs = [(0, 0), (1, 1), (2, 2), (3, 3)]

    estimate = fit_residual_transform_from_roi_pairs(
        reference,
        registered,
        pairs,
        config=BootstrapRegistrationConfig(min_matches=3, max_rmse=1.0),
    )

    assert estimate is not None
    assert estimate.transform == "affine"
    npt.assert_allclose(
        estimate.matrix_xy,
        np.array([[1.0, 0.0, 3.0], [0.0, 1.0, -2.0]]),
        atol=1.0e-9,
    )
    assert estimate.anchor_count == 4
    assert estimate.rmse == pytest.approx(0.0)


def test_refine_registered_planes_uses_only_margin_supported_edges() -> None:
    reference = _plane_from_centers([(10, 10), (10, 25), (25, 10), (25, 25)])
    registered = _plane_from_centers([(12, 7), (12, 22), (27, 7), (27, 22)])
    sessions = [_session("s0", reference), _session("s1", registered)]
    cost_matrix = np.full((4, 4), 10.0, dtype=float)
    np.fill_diagonal(cost_matrix, 0.1)
    tracks = [{0: roi_index, 1: roi_index} for roi_index in range(4)]
    run = GlobalAssignmentRun(
        result=SimpleNamespace(tracks=tracks),
        pairwise_costs={(0, 1): cost_matrix},
        session_sizes=(4, 4),
        session_edges=((0, 1),),
    )

    refined, estimates = refine_registered_planes_from_assignment(
        sessions,
        run,
        current_registered_planes={(0, 1): registered},
        config=BootstrapRegistrationConfig(
            min_matches=3,
            min_cost_margin=1.0,
            max_rmse=1.0,
        ),
    )

    assert set(estimates) == {(0, 1)}
    refined_centroids = refined[(0, 1)].centroids().T
    reference_centroids = reference.centroids().T
    npt.assert_allclose(refined_centroids, reference_centroids, atol=1.0)
