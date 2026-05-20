"""Regression tests for registered soft-IoU association costs."""

from __future__ import annotations

import numpy as np
import pytest

from bayescatrack.association.pyrecest_global_assignment import (
    _cost_kwargs_for_method,
    registered_soft_iou_cost_kwargs,
)
from bayescatrack.core.bridge import CalciumPlaneData


def _single_roi_plane(mask: np.ndarray) -> CalciumPlaneData:
    roi_masks = np.asarray(mask, dtype=float).reshape(1, 1, -1)
    return CalciumPlaneData(roi_masks=roi_masks)


def test_registered_soft_iou_cost_preset_is_supported() -> None:
    kwargs = _cost_kwargs_for_method("registered-soft-iou")

    assert kwargs["soft_iou"] is True
    assert kwargs["iou_weight"] == pytest.approx(1.0)
    assert kwargs["centroid_weight"] == pytest.approx(0.0)
    assert kwargs["mask_cosine_weight"] == pytest.approx(0.0)


def test_registered_soft_iou_kwargs_do_not_mutate_registered_iou_kwargs() -> None:
    soft_kwargs = registered_soft_iou_cost_kwargs()
    registered_kwargs = _cost_kwargs_for_method("registered-iou")

    assert soft_kwargs["soft_iou"] is True
    assert "soft_iou" not in registered_kwargs


def test_soft_iou_uses_mask_weights_not_only_binary_support() -> None:
    reference = _single_roi_plane(np.array([1.0, 1.0, 0.0]))
    measurement = _single_roi_plane(np.array([1.0, 0.25, 0.0]))
    common_kwargs = {
        "centroid_weight": 0.0,
        "iou_weight": 1.0,
        "mask_cosine_weight": 0.0,
        "area_weight": 0.0,
        "roi_feature_weight": 0.0,
        "cell_probability_weight": 0.0,
        "similarity_epsilon": 1.0e-9,
    }

    binary_cost = reference.build_pairwise_cost_matrix(measurement, **common_kwargs)
    soft_cost = reference.build_pairwise_cost_matrix(
        measurement, soft_iou=True, **common_kwargs
    )

    assert binary_cost[0, 0] == pytest.approx(0.0)
    assert soft_cost[0, 0] == pytest.approx(-np.log(1.25 / 2.0))
    assert soft_cost[0, 0] > binary_cost[0, 0]


def test_soft_iou_matches_binary_iou_for_boolean_masks() -> None:
    reference = _single_roi_plane(np.array([1.0, 1.0, 0.0]))
    measurement = _single_roi_plane(np.array([1.0, 0.0, 1.0]))
    common_kwargs = {
        "centroid_weight": 0.0,
        "iou_weight": 1.0,
        "mask_cosine_weight": 0.0,
        "area_weight": 0.0,
        "roi_feature_weight": 0.0,
        "cell_probability_weight": 0.0,
    }

    binary_cost = reference.build_pairwise_cost_matrix(measurement, **common_kwargs)
    soft_cost = reference.build_pairwise_cost_matrix(
        measurement, soft_iou=True, **common_kwargs
    )

    assert soft_cost[0, 0] == pytest.approx(binary_cost[0, 0])
