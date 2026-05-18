from __future__ import annotations

import numpy as np

from bayescatrack import CalciumPlaneData
from bayescatrack.soft_overlap_costs import registered_soft_iou_cost_kwargs


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

    pairwise_kwargs = registered_soft_iou_cost_kwargs(
        soft_iou_radius=2,
        distance_transform_overlap_radius=3,
        distance_transform_overlap_weight=1.0,
    )
    pairwise_kwargs["return_components"] = True
    _, components = reference.build_pairwise_cost_matrix(  # pylint: disable=unexpected-keyword-arg
        measurement, **pairwise_kwargs
    )

    assert components["iou"][0, 0] == 0.0
    assert components["soft_iou"][0, 0] > 0.0
    assert components["distance_transform_overlap"][0, 0] > 0.0
    assert np.isfinite(components["pairwise_cost_matrix"][0, 0])


def test_registered_soft_iou_preset_is_available_to_global_assignment():
    kwargs = registered_soft_iou_cost_kwargs()

    assert kwargs["iou_weight"] == 0.0
    assert kwargs["soft_iou_weight"] > 0.0
    assert kwargs["distance_transform_overlap_weight"] > 0.0
