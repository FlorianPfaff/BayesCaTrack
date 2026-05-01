"""Mahalanobis-style ROI centroid distances for calcium-imaging association."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import numpy as np


def pairwise_mahalanobis_centroid_distances(
    reference_plane: Any,
    measurement_plane: Any,
    *,
    order: str = "xy",
    weighted: bool = False,
    regularization: float = 1.0e-6,
) -> np.ndarray:
    """Return covariance-normalized centroid distances for all ROI pairs.

    The distance for pair ``(i, j)`` is

    ``sqrt((mu_i - mu_j)^T (Sigma_i + Sigma_j)^-1 (mu_i - mu_j))``.

    Summing the two ROI covariances treats centroid uncertainty from both
    sessions symmetrically and makes elongated/large ROIs less over-penalized
    along their high-variance axes. The output has shape
    ``(reference_plane.n_rois, measurement_plane.n_rois)``.
    """

    if regularization < 0.0:
        raise ValueError("regularization must be non-negative")
    if reference_plane.n_rois == 0 or measurement_plane.n_rois == 0:
        return np.zeros((reference_plane.n_rois, measurement_plane.n_rois), dtype=float)

    reference_centroids = reference_plane.centroids(order=order, weighted=weighted)
    measurement_centroids = measurement_plane.centroids(order=order, weighted=weighted)
    reference_covariances = reference_plane.position_covariances(
        order=order,
        weighted=weighted,
        regularization=regularization,
    )
    measurement_covariances = measurement_plane.position_covariances(
        order=order,
        weighted=weighted,
        regularization=regularization,
    )

    distances = np.zeros((reference_plane.n_rois, measurement_plane.n_rois), dtype=float)
    for reference_index in range(reference_plane.n_rois):
        for measurement_index in range(measurement_plane.n_rois):
            diff = reference_centroids[:, reference_index] - measurement_centroids[:, measurement_index]
            covariance = (
                reference_covariances[:, :, reference_index]
                + measurement_covariances[:, :, measurement_index]
            )
            try:
                normalized = np.linalg.solve(covariance, diff)
            except np.linalg.LinAlgError:
                normalized = np.linalg.pinv(covariance) @ diff
            squared_distance = float(diff @ normalized)
            distances[reference_index, measurement_index] = np.sqrt(max(squared_distance, 0.0))
    return distances


def add_mahalanobis_centroid_components(
    pairwise_components: MutableMapping[str, np.ndarray],
    reference_plane: Any,
    measurement_plane: Any,
    *,
    order: str = "xy",
    weighted: bool = False,
    regularization: float = 1.0e-6,
) -> MutableMapping[str, np.ndarray]:
    """Add Mahalanobis centroid-distance planes to a component dictionary."""

    distances = pairwise_mahalanobis_centroid_distances(
        reference_plane,
        measurement_plane,
        order=order,
        weighted=weighted,
        regularization=regularization,
    )
    pairwise_components["mahalanobis_centroid_distance"] = distances
    pairwise_components["mahalanobis_centroid_cost"] = distances**2
    return pairwise_components
