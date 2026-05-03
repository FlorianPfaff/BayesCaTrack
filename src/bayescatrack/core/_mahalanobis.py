"""Mahalanobis centroid-distance extension for Track2p ROI association."""

from __future__ import annotations

from typing import Any

import numpy as np

_MAHALANOBIS_INSTALLED_ATTR = "_bayescatrack_mahalanobis_installed"


def install_mahalanobis_pairwise_features(calcium_plane_cls: type[Any]) -> None:
    """Install Mahalanobis centroid-distance helpers on ``CalciumPlaneData``.

    The core bridge implementation is intentionally kept compact. This small
    extension adds covariance-normalized centroid distances while preserving the
    existing public class and method names.
    """

    if getattr(calcium_plane_cls, _MAHALANOBIS_INSTALLED_ATTR, False):
        return

    original_build_pairwise_cost_matrix = calcium_plane_cls.build_pairwise_cost_matrix

    def pairwise_mahalanobis_centroid_distances(
        self: Any,
        other: Any,
        *,
        order: str = "xy",
        weighted: bool = False,
        regularization: float = 1.0e-6,
    ) -> np.ndarray:
        """Return covariance-normalized centroid distances for all ROI pairs.

        The distance for pair ``(i, j)`` is

        ``sqrt((mu_i - mu_j)^T (Sigma_i + Sigma_j)^-1 (mu_i - mu_j))``.
        """

        if regularization < 0.0:
            raise ValueError("regularization must be non-negative")
        if self.n_rois == 0 or other.n_rois == 0:
            return np.zeros((self.n_rois, other.n_rois), dtype=float)

        centroids_self = self.centroids(order=order, weighted=weighted)
        centroids_other = other.centroids(order=order, weighted=weighted)
        covariances_self = self.position_covariances(
            order=order,
            weighted=weighted,
            regularization=regularization,
        )
        covariances_other = other.position_covariances(
            order=order,
            weighted=weighted,
            regularization=regularization,
        )

        distances = np.zeros((self.n_rois, other.n_rois), dtype=float)
        for reference_index in range(self.n_rois):
            for measurement_index in range(other.n_rois):
                diff = (
                    centroids_self[:, reference_index]
                    - centroids_other[:, measurement_index]
                )
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
        return distances

    # pylint: disable=too-many-arguments,too-many-locals
    def build_pairwise_cost_matrix(
        self: Any,
        other: Any,
        *,
        order: str = "xy",
        weighted_centroids: bool = False,
        centroid_weight: float = 1.0,
        mahalanobis_weight: float = 0.0,
        mahalanobis_regularization: float = 1.0e-6,
        centroid_scale: float | None = None,
        max_centroid_distance: float | None = None,
        iou_weight: float = 6.0,
        mask_cosine_weight: float = 2.0,
        area_weight: float = 0.5,
        roi_feature_weight: float = 0.25,
        feature_names: Any = None,
        cell_probability_weight: float = 0.0,
        large_cost: float = 1.0e6,
        similarity_epsilon: float = 1.0e-6,
        return_components: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
        """Build a ROI-aware association cost matrix with Mahalanobis features."""

        if mahalanobis_regularization < 0.0:
            raise ValueError("mahalanobis_regularization must be non-negative")
        if mahalanobis_weight < 0.0:
            raise ValueError("mahalanobis_weight must be non-negative")

        base_cost, components = original_build_pairwise_cost_matrix(
            self,
            other,
            order=order,
            weighted_centroids=weighted_centroids,
            centroid_weight=centroid_weight,
            centroid_scale=centroid_scale,
            max_centroid_distance=max_centroid_distance,
            iou_weight=iou_weight,
            mask_cosine_weight=mask_cosine_weight,
            area_weight=area_weight,
            roi_feature_weight=roi_feature_weight,
            feature_names=feature_names,
            cell_probability_weight=cell_probability_weight,
            large_cost=large_cost,
            similarity_epsilon=similarity_epsilon,
            return_components=True,
        )
        components = dict(components)

        mahalanobis_distances = pairwise_mahalanobis_centroid_distances(
            self,
            other,
            order=order,
            weighted=weighted_centroids,
            regularization=mahalanobis_regularization,
        )
        mahalanobis_cost = mahalanobis_distances**2
        total_cost = (
            np.asarray(base_cost, dtype=float) + mahalanobis_weight * mahalanobis_cost
        )
        gated = np.asarray(
            components.get("gated", np.zeros_like(total_cost, dtype=bool)), dtype=bool
        )
        total_cost = np.where(gated, large_cost, total_cost)
        total_cost = np.nan_to_num(
            total_cost, nan=large_cost, posinf=large_cost, neginf=large_cost
        )

        components["pairwise_cost_matrix"] = total_cost
        components["mahalanobis_centroid_distance"] = mahalanobis_distances
        components["mahalanobis_centroid_cost"] = mahalanobis_cost

        if return_components:
            return total_cost, components
        return total_cost

    calcium_plane_cls.pairwise_mahalanobis_centroid_distances = (
        pairwise_mahalanobis_centroid_distances
    )
    calcium_plane_cls.build_pairwise_cost_matrix = build_pairwise_cost_matrix
    setattr(calcium_plane_cls, _MAHALANOBIS_INSTALLED_ATTR, True)
