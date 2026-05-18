"""Tests for local shift-search overlap costs."""

# pylint: disable=protected-access

import numpy as np
import numpy.testing as npt

from bayescatrack import CalciumPlaneData
from bayescatrack.association import pyrecest_global_assignment as assignment
from bayescatrack.association.pyrecest_global_assignment import registered_iou_cost_kwargs
from bayescatrack.shifted_overlap_costs import (
    _pairwise_shifted_overlap_matrices,
    _translate_mask_stack,
    registered_shifted_iou_cost_kwargs,
)


def test_translate_mask_stack_shifts_without_wrapping():
    masks = np.zeros((1, 5, 6), dtype=bool)
    masks[0, 1:3, 2:4] = True

    shifted = _translate_mask_stack(masks, shift_y=1, shift_x=-2)

    expected = np.zeros_like(masks)
    expected[0, 2:4, 0:2] = True
    npt.assert_array_equal(shifted, expected)


def test_shifted_iou_finds_local_translation_without_dilation():
    reference = np.zeros((1, 8, 8), dtype=bool)
    measurement = np.zeros((1, 8, 8), dtype=bool)
    reference[0, 2:4, 2:4] = True
    measurement[0, 2:4, 5:7] = True

    shifted = _pairwise_shifted_overlap_matrices(
        reference,
        measurement,
        radius=3,
        include_iou=True,
        include_mask_cosine=True,
        similarity_epsilon=1.0e-6,
    )

    npt.assert_allclose(shifted["shifted_iou"], [[1.0]])
    npt.assert_allclose(shifted["shifted_mask_cosine_similarity"], [[1.0]])
    npt.assert_allclose(shifted["best_shift_norm"], [[3.0]])


def test_shifted_iou_cost_improves_near_miss_over_exact_iou():
    reference = np.zeros((1, 8, 8), dtype=bool)
    measurement = np.zeros((1, 8, 8), dtype=bool)
    reference[0, 2:4, 2:4] = True
    measurement[0, 2:4, 5:7] = True
    reference_plane = CalciumPlaneData(reference)
    measurement_plane = CalciumPlaneData(measurement)

    exact_cost = CalciumPlaneData.build_pairwise_cost_matrix(
        reference_plane,
        measurement_plane,
        **registered_iou_cost_kwargs(),
    )
    shifted_cost, components = CalciumPlaneData.build_pairwise_cost_matrix(
        reference_plane,
        measurement_plane,
        return_components=True,
        **registered_shifted_iou_cost_kwargs(shifted_iou_radius=3),
    )

    assert exact_cost[0, 0] > 10.0
    npt.assert_allclose(shifted_cost, [[0.0]])
    npt.assert_allclose(components["iou"], [[0.0]])
    npt.assert_allclose(components["shifted_iou"], [[1.0]])
    npt.assert_allclose(components["best_shift_norm"], [[3.0]])


def test_registered_shifted_iou_preset_is_installed_for_global_assignment():
    kwargs = assignment._cost_kwargs_for_method("registered-shifted-iou")

    assert kwargs["iou_weight"] == 0.0
    assert kwargs["shifted_iou_weight"] == 1.0
    assert kwargs["shifted_iou_radius"] == 4


def test_shifted_iou_radius_zero_matches_exact_iou():
    reference = np.zeros((1, 8, 8), dtype=bool)
    measurement = np.zeros((1, 8, 8), dtype=bool)
    reference[0, 2:5, 2:5] = True
    measurement[0, 3:6, 3:6] = True

    shifted = _pairwise_shifted_overlap_matrices(
        reference,
        measurement,
        radius=0,
        include_iou=True,
        include_mask_cosine=False,
        similarity_epsilon=1.0e-6,
    )

    npt.assert_allclose(shifted["shifted_iou"], [[4.0 / 14.0]])
    npt.assert_allclose(shifted["best_shift_norm"], [[0.0]])
