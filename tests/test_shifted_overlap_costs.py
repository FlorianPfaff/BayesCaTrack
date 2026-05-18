"""Tests for local shift-search ROI overlap costs."""

import numpy as np
import numpy.testing as npt

from bayescatrack import CalciumPlaneData
from bayescatrack.association.pyrecest_global_assignment import registered_iou_cost_kwargs
from bayescatrack.association.shifted_overlap import (
    install_shifted_overlap_cost_patch,
    pairwise_shifted_overlap_matrices,
    shift_offsets,
    shifted_iou_pairwise_cost_matrix,
)


def test_shift_offsets_put_zero_shift_first():
    offsets = shift_offsets(1)

    assert offsets[0] == (0, 0)
    assert set(offsets) == {
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 0),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    }


def test_shifted_overlap_finds_coherent_translated_match():
    reference = np.zeros((1, 10, 10), dtype=bool)
    measurement = np.zeros((1, 10, 10), dtype=bool)
    reference[0, 2:4, 2:4] = True
    measurement[0, 2:4, 4:6] = True

    components = pairwise_shifted_overlap_matrices(
        reference,
        measurement,
        radius=2,
    )

    assert components["shifted_iou"][0, 0] == 1.0
    assert components["shifted_iou_shift_y"][0, 0] == 0.0
    assert components["shifted_iou_shift_x"][0, 0] == -2.0
    assert components["shifted_iou_shift_norm"][0, 0] == 2.0


def test_shifted_iou_sparse_path_matches_component_path():
    reference = np.zeros((2, 10, 10), dtype=bool)
    measurement = np.zeros((2, 10, 10), dtype=bool)
    reference[0, 2:4, 2:4] = True
    reference[1, 5:7, 5:8] = True
    measurement[0, 2:4, 4:6] = True
    measurement[1, 6:8, 5:8] = True

    with_cosine = pairwise_shifted_overlap_matrices(
        reference,
        measurement,
        radius=2,
        include_mask_cosine=True,
    )
    without_cosine = pairwise_shifted_overlap_matrices(
        reference,
        measurement,
        radius=2,
        include_mask_cosine=False,
    )

    assert "shifted_mask_cosine_similarity" not in without_cosine
    for key in (
        "shifted_iou",
        "shifted_iou_shift_y",
        "shifted_iou_shift_x",
        "shifted_iou_shift_norm",
    ):
        npt.assert_allclose(with_cosine[key], without_cosine[key])


def test_shifted_iou_patch_replaces_registered_iou_cost():
    reference = np.zeros((1, 10, 10), dtype=bool)
    measurement = np.zeros((1, 10, 10), dtype=bool)
    reference[0, 2:4, 2:4] = True
    measurement[0, 2:4, 4:6] = True

    reference_plane = CalciumPlaneData(reference)
    measurement_plane = CalciumPlaneData(measurement)
    kwargs = registered_iou_cost_kwargs()
    kwargs.update(
        {
            "shifted_iou_radius": 2,
            "use_shifted_iou_for_iou_cost": True,
            "return_components": True,
        }
    )

    original_method = install_shifted_overlap_cost_patch()
    try:
        cost, components = reference_plane.build_pairwise_cost_matrix(
            measurement_plane,
            **kwargs,
        )
    finally:
        CalciumPlaneData.build_pairwise_cost_matrix = original_method  # type: ignore[method-assign]

    assert components["iou"][0, 0] == 0.0
    assert components["shifted_iou"][0, 0] == 1.0
    assert components["iou_for_cost"][0, 0] == 1.0
    npt.assert_allclose(cost, np.zeros((1, 1)))


def test_shifted_iou_skips_shifted_cosine_when_unused(monkeypatch):
    reference = np.zeros((1, 10, 10), dtype=bool)
    measurement = np.zeros((1, 10, 10), dtype=bool)
    reference[0, 2:4, 2:4] = True
    measurement[0, 2:4, 4:6] = True

    def fail_shifted_cosine(*args, **kwargs):
        raise AssertionError("shifted mask cosine should not be computed")

    monkeypatch.setattr(
        "bayescatrack.association.shifted_overlap._bridge_impl."
        "_pairwise_mask_cosine_similarity",
        fail_shifted_cosine,
    )

    reference_plane = CalciumPlaneData(reference)
    measurement_plane = CalciumPlaneData(measurement)

    def original_method(self, other, **kwargs):
        assert self is reference_plane
        assert other is measurement_plane
        assert kwargs["return_components"] is False
        return np.zeros((1, 1), dtype=float)

    cost = shifted_iou_pairwise_cost_matrix(
        original_method,
        reference_plane,
        measurement_plane,
        shifted_iou_radius=2,
        use_shifted_iou_for_iou_cost=True,
        return_components=False,
    )

    npt.assert_allclose(cost, np.zeros((1, 1)))


def test_shifted_iou_radius_zero_preserves_registered_iou_cost():
    reference = np.zeros((2, 8, 8), dtype=bool)
    measurement = np.zeros((2, 8, 8), dtype=bool)
    reference[0, 2:4, 2:4] = True
    reference[1, 4:6, 4:6] = True
    measurement[0, 2:4, 3:5] = True
    measurement[1, 0:2, 0:2] = True

    reference_plane = CalciumPlaneData(reference)
    measurement_plane = CalciumPlaneData(measurement)
    exact_kwargs = registered_iou_cost_kwargs()
    shifted_kwargs = registered_iou_cost_kwargs()
    shifted_kwargs.update(
        {
            "shifted_iou_radius": 0,
            "use_shifted_iou_for_iou_cost": True,
        }
    )

    original_method = install_shifted_overlap_cost_patch()
    try:
        exact_cost = reference_plane.build_pairwise_cost_matrix(
            measurement_plane,
            **exact_kwargs,
        )
        shifted_cost = reference_plane.build_pairwise_cost_matrix(
            measurement_plane,
            **shifted_kwargs,
        )
    finally:
        CalciumPlaneData.build_pairwise_cost_matrix = original_method  # type: ignore[method-assign]

    npt.assert_allclose(shifted_cost, exact_cost)


def test_shifted_mask_cosine_can_be_used_as_additive_tie_breaker():
    reference = np.zeros((1, 10, 10), dtype=float)
    measurement = np.zeros((1, 10, 10), dtype=float)
    reference[0, 3:5, 3:5] = np.array([[1.0, 0.5], [0.25, 0.75]])
    measurement[0, 4:6, 3:5] = reference[0, 3:5, 3:5]

    reference_plane = CalciumPlaneData(reference)
    measurement_plane = CalciumPlaneData(measurement)
    original_method = install_shifted_overlap_cost_patch()
    try:
        # pylint: disable=unexpected-keyword-arg
        cost, components = reference_plane.build_pairwise_cost_matrix(
            measurement_plane,
            centroid_weight=0.0,
            iou_weight=0.0,
            mask_cosine_weight=0.0,
            area_weight=0.0,
            roi_feature_weight=0.0,
            shifted_iou_radius=1,
            shifted_mask_cosine_weight=1.0,
            return_components=True,
        )  # type: ignore[call-arg]
    finally:
        CalciumPlaneData.build_pairwise_cost_matrix = original_method  # type: ignore[method-assign]

    assert components["mask_cosine_similarity"][0, 0] < 1.0
    assert components["shifted_mask_cosine_similarity"][0, 0] == 1.0
    npt.assert_allclose(cost, np.zeros((1, 1)))
