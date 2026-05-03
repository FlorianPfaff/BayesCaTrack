"""Tests for covariance-normalized centroid-distance features."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias, cast

import numpy as np
from bayescatrack.association.calibrated_costs import (
    DEFAULT_ASSOCIATION_FEATURES,
    pairwise_feature_tensor,
)
from bayescatrack.core.bridge import CalciumPlaneData

CostMatrixResult: TypeAlias = np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]


def _plane_from_rectangles(
    rectangles: list[tuple[int, int, int, int]],
) -> CalciumPlaneData:
    masks = np.zeros((len(rectangles), 8, 8), dtype=bool)
    for roi_index, (row_start, col_start, row_stop, col_stop) in enumerate(rectangles):
        masks[roi_index, row_start:row_stop, col_start:col_stop] = True
    return CalciumPlaneData(roi_masks=masks)


def _mahalanobis_distances(
    reference: CalciumPlaneData,
    measurement: CalciumPlaneData,
    **kwargs: Any,
) -> np.ndarray:
    method = cast(
        Callable[..., np.ndarray],
        getattr(reference, "pairwise_mahalanobis_centroid_distances"),
    )
    return method(measurement, **kwargs)


def _build_pairwise_cost_matrix(
    reference: CalciumPlaneData,
    measurement: CalciumPlaneData,
    **kwargs: Any,
) -> CostMatrixResult:
    method = cast(
        Callable[..., CostMatrixResult],
        getattr(reference, "build_pairwise_cost_matrix"),
    )
    return method(measurement, **kwargs)


def _build_cost_matrix(
    reference: CalciumPlaneData,
    measurement: CalciumPlaneData,
    **kwargs: Any,
) -> np.ndarray:
    result = _build_pairwise_cost_matrix(reference, measurement, **kwargs)
    assert isinstance(result, np.ndarray)
    return result


def _build_cost_components(
    reference: CalciumPlaneData,
    measurement: CalciumPlaneData,
    **kwargs: Any,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    result = _build_pairwise_cost_matrix(
        reference,
        measurement,
        return_components=True,
        **kwargs,
    )
    assert isinstance(result, tuple)
    return result


def test_pairwise_mahalanobis_centroid_distances_use_roi_covariances() -> None:
    reference = _plane_from_rectangles([(1, 1, 3, 3)])
    measurement = _plane_from_rectangles([(1, 2, 3, 4)])

    distances = _mahalanobis_distances(
        reference,
        measurement,
        regularization=0.0,
    )

    assert distances.shape == (1, 1)
    np.testing.assert_allclose(distances[0, 0], np.sqrt(2.0))


def test_pairwise_cost_components_expose_mahalanobis_feature() -> None:
    reference = _plane_from_rectangles([(1, 1, 3, 3)])
    measurement = _plane_from_rectangles([(1, 2, 3, 4)])

    _, components = _build_cost_components(
        reference,
        measurement,
        mahalanobis_regularization=0.0,
    )

    np.testing.assert_allclose(
        components["mahalanobis_centroid_distance"],
        np.array([[np.sqrt(2.0)]]),
    )
    np.testing.assert_allclose(
        components["mahalanobis_centroid_cost"],
        np.array([[2.0]]),
    )

    features = pairwise_feature_tensor(
        components,
        feature_names=("mahalanobis_centroid_distance",),
    )
    np.testing.assert_allclose(features, np.array([[[np.sqrt(2.0)]]]))


def test_mahalanobis_weight_contributes_to_pairwise_cost() -> None:
    reference = _plane_from_rectangles([(1, 1, 3, 3)])
    measurement = _plane_from_rectangles([(1, 2, 3, 4)])

    base_cost = _build_cost_matrix(
        reference,
        measurement,
        mahalanobis_regularization=0.0,
    )
    weighted_cost = _build_cost_matrix(
        reference,
        measurement,
        mahalanobis_weight=0.5,
        mahalanobis_regularization=0.0,
    )

    np.testing.assert_allclose(weighted_cost - base_cost, np.array([[1.0]]))


def test_default_association_features_include_mahalanobis_centroid_distance() -> None:
    assert "mahalanobis_centroid_distance" in DEFAULT_ASSOCIATION_FEATURES
