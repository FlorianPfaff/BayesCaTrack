from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
from bayescatrack.association import pyrecest_global_assignment as global_assignment
from bayescatrack.association.registered_masks import (
    add_registered_roi_validity_components,
    drop_empty_registered_masks,
    expand_registered_pairwise_components,
    expand_registered_pairwise_cost_columns,
    mask_invalid_registered_roi_columns,
)


def test_registered_pairwise_costs_penalize_empty_registered_masks(
    make_track2p_session,
    monkeypatch,
):
    reference_masks = np.zeros((1, 4, 4), dtype=bool)
    reference_masks[0, 1:3, 1:3] = True
    measurement_masks = np.zeros_like(reference_masks)
    measurement_masks[0, 2:4, 2:4] = True

    reference = make_track2p_session("2024-05-01_a", reference_masks)
    measurement = make_track2p_session("2024-05-02_a", measurement_masks)
    empty_registered_plane = measurement.plane_data.with_replaced_masks(
        np.zeros_like(measurement_masks)
    )

    def _fake_register_plane_pair(*_args, **_kwargs):
        return empty_registered_plane

    monkeypatch.setattr(
        global_assignment, "register_plane_pair", _fake_register_plane_pair
    )

    pairwise_costs = global_assignment.build_registered_pairwise_costs(
        [reference, measurement],
        max_gap=1,
        cost="registered-iou",
    )

    npt.assert_array_equal(pairwise_costs[(0, 1)], np.array([[1.0e6]]))


def test_registered_shifted_iou_cost_kwargs_replace_exact_iou_term():
    kwargs = global_assignment.registered_shifted_iou_cost_kwargs(shifted_iou_radius=3)

    assert kwargs["iou_weight"] == 1.0
    assert kwargs["shifted_iou_radius"] == 3
    assert kwargs["use_shifted_iou_for_iou_cost"] is True
    assert kwargs["shifted_iou_weight"] == 0.0
    assert kwargs["shifted_mask_cosine_weight"] == 0.0
    assert kwargs["shifted_iou_shift_penalty_weight"] == 0.0


def test_roi_aware_shifted_cost_kwargs_replace_overlap_terms():
    kwargs = global_assignment.roi_aware_shifted_cost_kwargs(shifted_iou_radius=3)

    assert "iou_weight" not in kwargs
    assert "mask_cosine_weight" not in kwargs
    assert kwargs["shifted_iou_radius"] == 3
    assert kwargs["use_shifted_iou_for_iou_cost"] is True
    assert kwargs["use_shifted_mask_cosine_for_mask_cosine_cost"] is True
    assert kwargs["shifted_iou_shift_penalty_weight"] == 0.25


def test_registered_shifted_iou_cost_kwargs_reject_negative_radius():
    with pytest.raises(ValueError, match="shifted_iou_radius"):
        global_assignment.registered_shifted_iou_cost_kwargs(shifted_iou_radius=-1)
    with pytest.raises(ValueError, match="shifted_iou_shift_penalty_weight"):
        global_assignment.registered_shifted_iou_cost_kwargs(
            shifted_iou_shift_penalty_weight=-1.0
        )
    with pytest.raises(ValueError, match="shifted_iou_shift_penalty_scale"):
        global_assignment.registered_shifted_iou_cost_kwargs(
            shifted_iou_shift_penalty_scale=0.0
        )


def test_roi_aware_shifted_cost_lowers_near_miss_overlap_penalty(
    make_track2p_session,
    monkeypatch,
):
    reference_masks = np.zeros((1, 10, 10), dtype=bool)
    reference_masks[0, 2:4, 2:4] = True
    measurement_masks = np.zeros_like(reference_masks)
    measurement_masks[0, 2:4, 4:6] = True

    reference = make_track2p_session("2024-05-01_a", reference_masks)
    measurement = make_track2p_session("2024-05-02_a", measurement_masks)

    def _fake_register_plane_pair(*_args, **_kwargs):
        return measurement.plane_data

    monkeypatch.setattr(
        global_assignment, "register_plane_pair", _fake_register_plane_pair
    )

    exact_costs = global_assignment.build_registered_pairwise_costs(
        [reference, measurement],
        max_gap=1,
        cost="roi-aware",
    )
    shifted_costs = global_assignment.build_registered_pairwise_costs(
        [reference, measurement],
        max_gap=1,
        cost="roi-aware-shifted",
        pairwise_cost_kwargs={"shifted_iou_shift_penalty_weight": 0.0},
    )

    assert shifted_costs[(0, 1)][0, 0] < exact_costs[(0, 1)][0, 0]


def test_registered_shifted_iou_cost_recovers_local_residual_shift(
    make_track2p_session,
    monkeypatch,
):
    reference_masks = np.zeros((1, 10, 10), dtype=bool)
    reference_masks[0, 2:4, 2:4] = True
    measurement_masks = np.zeros_like(reference_masks)
    measurement_masks[0, 2:4, 4:6] = True

    reference = make_track2p_session("2024-05-01_a", reference_masks)
    measurement = make_track2p_session("2024-05-02_a", measurement_masks)

    def _fake_register_plane_pair(*_args, **_kwargs):
        return measurement.plane_data

    monkeypatch.setattr(
        global_assignment, "register_plane_pair", _fake_register_plane_pair
    )
    previous_pairwise_cost_method = (
        global_assignment.CalciumPlaneData.build_pairwise_cost_matrix
    )

    pairwise_costs = global_assignment.build_registered_pairwise_costs(
        [reference, measurement],
        max_gap=1,
        cost="registered-shifted-iou",
    )

    npt.assert_allclose(pairwise_costs[(0, 1)], np.zeros((1, 1)))
    assert (
        global_assignment.CalciumPlaneData.build_pairwise_cost_matrix
        is previous_pairwise_cost_method
    )


def test_empty_registered_masks_are_dropped_without_placeholders(
    make_track2p_session,
):
    roi_masks = np.zeros((3, 4, 4), dtype=bool)
    roi_masks[0, 1:3, 1:3] = True
    session = make_track2p_session("2024-05-01_a", roi_masks)

    filtered_plane, empty_rois = drop_empty_registered_masks(session.plane_data)

    assert empty_rois.tolist() == [False, True, True]
    assert filtered_plane.n_rois == 1
    npt.assert_array_equal(filtered_plane.roi_masks, roi_masks[[0]])
    assert not np.any(session.plane_data.roi_masks[1:])


def test_expand_registered_pairwise_cost_columns_marks_empty_targets():
    compact_costs = np.array([[0.25], [0.75]], dtype=float)
    empty_rois = np.array([False, True, True])

    expanded = expand_registered_pairwise_cost_columns(
        compact_costs,
        empty_rois,
        large_cost=99.0,
    )

    npt.assert_allclose(
        expanded,
        np.array([[0.25, 99.0, 99.0], [0.75, 99.0, 99.0]]),
    )


def test_expand_registered_pairwise_components_marks_invalid_columns():
    compact_components = {
        "iou": np.array([[0.75]], dtype=float),
        "gated": np.array([[False]], dtype=bool),
        "activity_similarity_available": np.array([[1.0]], dtype=float),
    }
    empty_rois = np.array([False, True, True])

    expanded = expand_registered_pairwise_components(compact_components, empty_rois)

    npt.assert_allclose(expanded["iou"][:, :1], np.array([[0.75]]))
    assert np.all(np.isnan(expanded["iou"][:, 1:]))
    npt.assert_array_equal(expanded["gated"], np.array([[False, True, True]]))
    npt.assert_allclose(
        expanded["activity_similarity_available"],
        np.array([[1.0, 0.0, 0.0]]),
    )


def test_invalid_registered_roi_columns_mask_placeholder_evidence():
    pairwise_components = {
        "iou": np.array([[0.75, 1.0]]),
        "centroid_distance": np.array([[2.0, 0.0]]),
        "area_ratio_cost": np.array([[0.1, 0.0]]),
        "session_gap": np.array([[1.0, 1.0]]),
        "gated": np.zeros((1, 2), dtype=bool),
    }

    add_registered_roi_validity_components(
        pairwise_components,
        np.array([True, False]),
        large_cost=99.0,
    )
    masked = mask_invalid_registered_roi_columns(
        pairwise_components,
        large_cost=99.0,
    )

    assert masked["registered_roi_valid"].tolist() == [[True, False]]
    assert masked["iou"].tolist() == [[0.75, 0.0]]
    assert masked["centroid_distance"].tolist() == [[2.0, 99.0]]
    assert masked["area_ratio_cost"].tolist() == [[0.1, 99.0]]
    assert masked["session_gap"].tolist() == [[1.0, 1.0]]
    assert masked["gated"].tolist() == [[False, True]]
    assert masked["registered_roi_invalid_cost"].tolist() == [[0.0, 99.0]]
