from __future__ import annotations

import numpy as np

from bayescatrack import CalciumPlaneData
from bayescatrack.association import pyrecest_global_assignment as global_assignment


def _single_roi_plane(mask: np.ndarray) -> CalciumPlaneData:
    return CalciumPlaneData(
        roi_masks=mask[None, :, :].astype(bool),
        roi_indices=np.asarray([0], dtype=int),
        source="synthetic",
    )


def test_soft_overlap_components_capture_near_miss_with_zero_exact_iou():
    reference_mask = np.zeros((12, 12), dtype=bool)
    measurement_mask = np.zeros((12, 12), dtype=bool)
    reference_mask[4:6, 4:6] = True
    measurement_mask[4:6, 7:9] = True
    reference = _single_roi_plane(reference_mask)
    measurement = _single_roi_plane(measurement_mask)

    _, components = reference.build_pairwise_cost_matrix(
        measurement,
        centroid_weight=0.0,
        iou_weight=0.0,
        soft_iou_weight=1.0,
        soft_iou_radius=2,
        distance_transform_overlap_weight=1.0,
        distance_transform_overlap_radius=3,
        mask_cosine_weight=0.0,
        area_weight=0.0,
        roi_feature_weight=0.0,
        return_components=True,
    )

    assert components["iou"][0, 0] == 0.0
    assert components["soft_iou"][0, 0] > 0.0
    assert components["distance_transform_overlap"][0, 0] > 0.0
    assert np.isfinite(components["pairwise_cost_matrix"][0, 0])


def test_registered_soft_iou_preset_is_available_to_global_assignment():
    kwargs = global_assignment._cost_kwargs_for_method("registered-soft-iou")

    assert kwargs["iou_weight"] == 0.0
    assert kwargs["soft_iou_weight"] > 0.0
    assert kwargs["distance_transform_overlap_weight"] > 0.0
