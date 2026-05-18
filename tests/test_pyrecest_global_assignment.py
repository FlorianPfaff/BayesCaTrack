from __future__ import annotations

import numpy as np
import numpy.testing as npt
from bayescatrack.association import pyrecest_global_assignment as global_assignment
from bayescatrack.association.registered_masks import replace_empty_registered_masks


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


def test_empty_registered_mask_placeholders_use_distinct_pixels(make_track2p_session):
    roi_masks = np.zeros((3, 4, 4), dtype=bool)
    roi_masks[0, 1:3, 1:3] = True
    session = make_track2p_session("2024-05-01_a", roi_masks)

    fixed_plane, empty_rois = replace_empty_registered_masks(session.plane_data)

    assert empty_rois.tolist() == [False, True, True]
    placeholder_pixels = [
        tuple(np.argwhere(fixed_plane.roi_masks[roi_index])[0])
        for roi_index in np.flatnonzero(empty_rois)
    ]
    assert len(set(placeholder_pixels)) == 2
