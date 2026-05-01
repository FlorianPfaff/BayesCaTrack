"""Compatibility patch for calibrated Mahalanobis bundle components."""

from __future__ import annotations

from typing import Any

import numpy as np

from . import calibrated_costs as _calibrated_costs

_ORIGINAL_COMPONENTS_FN_ATTR = "_bayescatrack_original_pairwise_components_from_bundle"

if not hasattr(_calibrated_costs, _ORIGINAL_COMPONENTS_FN_ATTR):
    setattr(
        _calibrated_costs,
        _ORIGINAL_COMPONENTS_FN_ATTR,
        _calibrated_costs.pairwise_components_from_bundle,
    )

_ORIGINAL_PAIRWISE_COMPONENTS_FROM_BUNDLE = getattr(
    _calibrated_costs,
    _ORIGINAL_COMPONENTS_FN_ATTR,
)


def pairwise_components_from_bundle(
    bundle: Any,
    *,
    covariance_epsilon: float = 1.0e-6,
) -> dict[str, np.ndarray]:
    """Return calibrated components with Mahalanobis features guaranteed present."""

    components = _ORIGINAL_PAIRWISE_COMPONENTS_FROM_BUNDLE(
        bundle,
        covariance_epsilon=covariance_epsilon,
    )
    if "mahalanobis_centroid_distance" in components:
        components.setdefault(
            "mahalanobis_centroid_cost",
            np.asarray(components["mahalanobis_centroid_distance"], dtype=float) ** 2,
        )
        return components

    reference_position_covariances = _reference_position_covariances(
        bundle.reference_state_covariances
    )
    measurement_position_covariances = _measurement_position_covariances(
        bundle.measurement_covariances
    )
    mahalanobis_centroid_distance = _pairwise_mahalanobis_centroid_distances(
        _reference_positions(bundle.reference_state_means),
        _measurement_positions(bundle.measurements),
        reference_position_covariances,
        measurement_position_covariances,
    )
    components["mahalanobis_centroid_distance"] = mahalanobis_centroid_distance
    components["mahalanobis_centroid_cost"] = mahalanobis_centroid_distance**2
    return components


def _reference_positions(reference_state_means: Any) -> np.ndarray:
    means = np.asarray(reference_state_means, dtype=float)
    if means.ndim != 2 or means.shape[0] != 4:
        raise ValueError("reference_state_means must have shape (4, n_roi)")
    return means[[0, 2], :]


def _measurement_positions(measurements: Any) -> np.ndarray:
    positions = np.asarray(measurements, dtype=float)
    if positions.ndim != 2 or positions.shape[0] != 2:
        raise ValueError("measurements must have shape (2, n_roi)")
    return positions


def _reference_position_covariances(reference_state_covariances: Any) -> np.ndarray:
    covariances = np.asarray(reference_state_covariances, dtype=float)
    if covariances.ndim != 3 or covariances.shape[:2] != (4, 4):
        raise ValueError("reference_state_covariances must have shape (4, 4, n_roi)")
    return covariances[[0, 2], :, :][:, [0, 2], :]


def _measurement_position_covariances(measurement_covariances: Any) -> np.ndarray:
    covariances = np.asarray(measurement_covariances, dtype=float)
    if covariances.ndim != 3 or covariances.shape[:2] != (2, 2):
        raise ValueError("measurement_covariances must have shape (2, 2, n_roi)")
    return covariances


def _pairwise_mahalanobis_centroid_distances(
    centroids_self: np.ndarray,
    centroids_other: np.ndarray,
    covariances_self: np.ndarray,
    covariances_other: np.ndarray,
) -> np.ndarray:
    """Return covariance-normalized centroid distances from bundle moments."""

    centroids_self = np.asarray(centroids_self, dtype=float)
    centroids_other = np.asarray(centroids_other, dtype=float)
    covariances_self = np.asarray(covariances_self, dtype=float)
    covariances_other = np.asarray(covariances_other, dtype=float)

    if centroids_self.ndim != 2 or centroids_self.shape[0] != 2:
        raise ValueError("centroids_self must have shape (2, n_roi)")
    if centroids_other.ndim != 2 or centroids_other.shape[0] != 2:
        raise ValueError("centroids_other must have shape (2, n_roi)")
    if covariances_self.ndim != 3 or covariances_self.shape[:2] != (2, 2):
        raise ValueError("covariances_self must have shape (2, 2, n_roi)")
    if covariances_other.ndim != 3 or covariances_other.shape[:2] != (2, 2):
        raise ValueError("covariances_other must have shape (2, 2, n_roi)")
    if centroids_self.shape[1] != covariances_self.shape[2]:
        raise ValueError("centroids_self and covariances_self must have the same n_roi")
    if centroids_other.shape[1] != covariances_other.shape[2]:
        raise ValueError("centroids_other and covariances_other must have the same n_roi")

    n_self = centroids_self.shape[1]
    n_other = centroids_other.shape[1]
    if n_self == 0 or n_other == 0:
        return np.zeros((n_self, n_other), dtype=float)

    covariances_self = 0.5 * (
        covariances_self + np.swapaxes(covariances_self, 0, 1)
    )
    covariances_other = 0.5 * (
        covariances_other + np.swapaxes(covariances_other, 0, 1)
    )

    distances = np.zeros((n_self, n_other), dtype=float)
    for reference_index in range(n_self):
        for measurement_index in range(n_other):
            diff = centroids_self[:, reference_index] - centroids_other[:, measurement_index]
            covariance = (
                covariances_self[:, :, reference_index]
                + covariances_other[:, :, measurement_index]
            )
            try:
                normalized = np.linalg.solve(covariance, diff)
            except np.linalg.LinAlgError:
                normalized = np.linalg.pinv(covariance) @ diff
            squared_distance = float(diff @ normalized)
            distances[reference_index, measurement_index] = np.sqrt(
                max(squared_distance, 0.0)
            )

    return np.nan_to_num(distances, nan=0.0, posinf=1.0e6, neginf=0.0)


_calibrated_costs.pairwise_components_from_bundle = pairwise_components_from_bundle
