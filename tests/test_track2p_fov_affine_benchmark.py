"""Tests for the FOV-affine Track2p benchmark wrapper."""

# pylint: disable=protected-access

import numpy as np
import numpy.testing as npt

from bayescatrack import CalciumPlaneData
from bayescatrack.association.pyrecest_global_assignment import registered_iou_cost_kwargs
from bayescatrack.core import _bridge_impl
from bayescatrack.experiments.track2p_fov_affine_benchmark import (
    _dilate_mask_stack,
    _pairwise_dilated_iou_matrix,
    _soft_iou_pairwise_cost_matrix,
)


def test_pairwise_dilated_iou_matches_reference_with_overlapping_pixels():
    reference = np.zeros((2, 8, 8), dtype=bool)
    measurement = np.zeros((2, 8, 8), dtype=bool)
    reference[0, 3, 3] = True
    reference[1, 3, 4] = True
    measurement[0, 3, 5] = True
    measurement[1, 6, 6] = True

    radius = 2
    actual = _pairwise_dilated_iou_matrix(reference, measurement, radius=radius)
    expected = _bridge_impl._pairwise_iou_matrix(
        _dilate_mask_stack(reference, radius=radius),
        _dilate_mask_stack(measurement, radius=radius),
    )

    npt.assert_allclose(actual, expected)


def test_radius_zero_iou_only_cost_matches_reference_implementation():
    reference = np.zeros((2, 8, 8), dtype=bool)
    measurement = np.zeros((2, 8, 8), dtype=bool)
    reference[0, 2:4, 2:4] = True
    reference[1, 4:6, 4:6] = True
    measurement[0, 2:4, 3:5] = True
    measurement[1, 0:2, 0:2] = True
    reference_plane = CalciumPlaneData(reference)
    measurement_plane = CalciumPlaneData(measurement)
    kwargs = registered_iou_cost_kwargs()

    actual = _soft_iou_pairwise_cost_matrix(
        CalciumPlaneData.build_pairwise_cost_matrix,
        reference_plane,
        measurement_plane,
        **kwargs,
    )
    expected = CalciumPlaneData.build_pairwise_cost_matrix(
        reference_plane,
        measurement_plane,
        **kwargs,
    )

    npt.assert_allclose(actual, expected)
