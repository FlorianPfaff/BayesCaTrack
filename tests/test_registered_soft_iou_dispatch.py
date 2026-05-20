from __future__ import annotations

from bayescatrack.association.pyrecest_global_assignment import (
    _cost_kwargs_for_method,
    _pairwise_kwargs_use_soft_overlap,
)


def test_registered_soft_iou_has_direct_global_assignment_kwargs():
    kwargs = _cost_kwargs_for_method("registered-soft-iou")

    assert kwargs["centroid_weight"] == 0.0
    assert kwargs["iou_weight"] == 0.0
    assert kwargs["soft_iou_weight"] == 1.0
    assert kwargs["soft_iou_radius"] == 2
    assert kwargs["distance_transform_overlap_weight"] > 0.0
    assert kwargs["distance_transform_overlap_radius"] == 3
    assert kwargs["mask_cosine_weight"] == 0.0
    assert kwargs["area_weight"] == 0.0
    assert kwargs["roi_feature_weight"] == 0.0
    assert kwargs["cell_probability_weight"] == 0.0


def test_soft_overlap_detection_covers_preset_and_json_overrides():
    assert _pairwise_kwargs_use_soft_overlap(
        _cost_kwargs_for_method("registered-soft-iou")
    )
    assert _pairwise_kwargs_use_soft_overlap({"soft_iou_radius": 4})
    assert _pairwise_kwargs_use_soft_overlap(
        {"distance_transform_overlap_weight": 0.1}
    )
    assert not _pairwise_kwargs_use_soft_overlap(
        {
            "iou_weight": 1.0,
            "similarity_epsilon": 1.0e-6,
        }
    )
