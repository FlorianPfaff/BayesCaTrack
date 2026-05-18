"""Local image and neighborhood evidence extension for Track2p ROI association."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from ._bridge_impl import (
    _estimate_default_centroid_scale,
    _mask_support_areas,
    _pairwise_sparse_mask_dot,
)

_LOCAL_EVIDENCE_INSTALLED_ATTR = "_bayescatrack_local_evidence_installed"


def install_local_evidence_pairwise_features(calcium_plane_cls: type[Any]) -> None:
    """Install optional local-evidence terms on ``CalciumPlaneData``.

    The extension wraps the existing ROI-aware/Mahalanobis cost builder and is
    deliberately zero-default: historical benchmark configurations are unchanged
    unless a local-evidence weight, or ``local_evidence_components=True``, is
    supplied through the existing pairwise-cost kwargs path.
    """

    if getattr(calcium_plane_cls, _LOCAL_EVIDENCE_INSTALLED_ATTR, False):
        return

    original_build_pairwise_cost_matrix = calcium_plane_cls.build_pairwise_cost_matrix

    # pylint: disable=too-many-arguments,too-many-locals,too-many-branches,too-many-statements
    def build_pairwise_cost_matrix(
        self: Any,
        other: Any,
        *,
        weighted_dice_weight: float = 0.0,
        overlap_fraction_weight: float = 0.0,
        containment_weight: float = 0.0,
        distance_transform_weight: float = 0.0,
        image_patch_weight: float = 0.0,
        neighbor_constellation_weight: float = 0.0,
        centroid_rank_weight: float = 0.0,
        local_evidence_components: bool = False,
        patch_radius: int = 8,
        neighbor_k: int = 8,
        normalize_weighted_overlap: bool = True,
        return_components: bool = False,
        **base_cost_kwargs: Any,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
        """Build a cost matrix with optional local image/geometry evidence."""

        for weight_name, weight_value in {
            "weighted_dice_weight": weighted_dice_weight,
            "overlap_fraction_weight": overlap_fraction_weight,
            "containment_weight": containment_weight,
            "distance_transform_weight": distance_transform_weight,
            "image_patch_weight": image_patch_weight,
            "neighbor_constellation_weight": neighbor_constellation_weight,
            "centroid_rank_weight": centroid_rank_weight,
        }.items():
            if weight_value < 0.0:
                raise ValueError(f"{weight_name} must be non-negative")
        if patch_radius < 0:
            raise ValueError("patch_radius must be non-negative")
        if neighbor_k < 1:
            raise ValueError("neighbor_k must be at least 1")

        base_cost, components = original_build_pairwise_cost_matrix(
            self,
            other,
            return_components=True,
            **base_cost_kwargs,
        )
        total_cost = np.asarray(base_cost, dtype=float).copy()
        components = dict(components)
        similarity_epsilon = float(base_cost_kwargs.get("similarity_epsilon", 1.0e-6))
        if similarity_epsilon <= 0.0:
            raise ValueError("similarity_epsilon must be strictly positive")
        large_cost = float(base_cost_kwargs.get("large_cost", 1.0e6))
        order = str(base_cost_kwargs.get("order", "xy"))
        weighted_centroids = bool(base_cost_kwargs.get("weighted_centroids", False))
        centroid_scale = base_cost_kwargs.get("centroid_scale")

        if (
            local_evidence_components
            or weighted_dice_weight > 0.0
            or overlap_fraction_weight > 0.0
            or containment_weight > 0.0
        ):
            overlap_components = _pairwise_local_overlap_components(
                self.roi_masks,
                other.roi_masks,
                normalize_weighted_overlap=normalize_weighted_overlap,
            )
            weighted_dice_similarity = overlap_components["weighted_dice_similarity"]
            weighted_dice_cost = -np.log(
                np.clip(weighted_dice_similarity, similarity_epsilon, 1.0)
            )
            overlap_fraction_cost = -np.log(
                np.clip(
                    overlap_components["overlap_min_fraction"],
                    similarity_epsilon,
                    1.0,
                )
            )
            containment_asymmetry_cost = overlap_components["containment_asymmetry"]
            if weighted_dice_weight > 0.0:
                total_cost += weighted_dice_weight * weighted_dice_cost
            if overlap_fraction_weight > 0.0:
                total_cost += overlap_fraction_weight * overlap_fraction_cost
            if containment_weight > 0.0:
                total_cost += containment_weight * containment_asymmetry_cost
            components.update(
                {
                    "binary_intersection": overlap_components["binary_intersection"],
                    "weighted_intersection": overlap_components["weighted_intersection"],
                    "weighted_dice_similarity": weighted_dice_similarity,
                    "weighted_dice_cost": weighted_dice_cost,
                    "overlap_min_fraction": overlap_components["overlap_min_fraction"],
                    "overlap_max_fraction": overlap_components["overlap_max_fraction"],
                    "overlap_fraction_cost": overlap_fraction_cost,
                    "reference_containment": overlap_components["reference_containment"],
                    "measurement_containment": overlap_components["measurement_containment"],
                    "containment_asymmetry_cost": containment_asymmetry_cost,
                }
            )

        if local_evidence_components or distance_transform_weight > 0.0:
            spatial_scale = _estimate_default_centroid_scale(
                self,
                other,
                centroid_scale=centroid_scale,
            )
            distance_transform_cost = _pairwise_symmetric_chamfer_distance_cost(
                self.roi_masks,
                other.roi_masks,
                scale=spatial_scale,
            )
            if distance_transform_weight > 0.0:
                total_cost += distance_transform_weight * distance_transform_cost
            components["distance_transform_cost"] = distance_transform_cost

        if local_evidence_components or image_patch_weight > 0.0:
            image_patch_correlation, image_patch_valid = _pairwise_fov_patch_correlations(
                self,
                other,
                order=order,
                weighted_centroids=weighted_centroids,
                patch_radius=patch_radius,
                similarity_epsilon=similarity_epsilon,
            )
            image_patch_cost = np.where(
                image_patch_valid,
                0.5 * (1.0 - np.clip(image_patch_correlation, -1.0, 1.0)),
                0.0,
            )
            if image_patch_weight > 0.0:
                total_cost += image_patch_weight * image_patch_cost
            components["image_patch_correlation"] = image_patch_correlation
            components["image_patch_cost"] = image_patch_cost
            components["image_patch_valid"] = image_patch_valid.astype(float)

        if local_evidence_components or neighbor_constellation_weight > 0.0:
            spatial_scale = _estimate_default_centroid_scale(
                self,
                other,
                centroid_scale=centroid_scale,
            )
            neighbor_constellation_cost = _pairwise_neighbor_constellation_cost(
                self,
                other,
                order=order,
                weighted_centroids=weighted_centroids,
                neighbor_k=neighbor_k,
                scale=spatial_scale,
            )
            if neighbor_constellation_weight > 0.0:
                total_cost += neighbor_constellation_weight * neighbor_constellation_cost
            components["neighbor_constellation_cost"] = neighbor_constellation_cost

        if local_evidence_components or centroid_rank_weight > 0.0:
            centroid_distances = components.get("centroid_distance")
            if centroid_distances is None:
                centroid_distances = self.pairwise_centroid_distances(
                    other,
                    order=order,
                    weighted=weighted_centroids,
                )
            centroid_rank_cost = _pairwise_centroid_rank_cost(
                np.asarray(centroid_distances, dtype=float)
            )
            if centroid_rank_weight > 0.0:
                total_cost += centroid_rank_weight * centroid_rank_cost
            components["centroid_rank_cost"] = centroid_rank_cost

        gated = np.asarray(
            components.get("gated", np.zeros_like(total_cost, dtype=bool)), dtype=bool
        )
        total_cost = np.where(gated, large_cost, total_cost)
        total_cost = np.nan_to_num(
            total_cost,
            nan=large_cost,
            posinf=large_cost,
            neginf=large_cost,
        )
        total_cost[total_cost < 0.0] = 0.0
        components["pairwise_cost_matrix"] = total_cost

        if return_components:
            return total_cost, components
        return total_cost

    calcium_plane_cls.build_pairwise_cost_matrix = build_pairwise_cost_matrix
    setattr(calcium_plane_cls, _LOCAL_EVIDENCE_INSTALLED_ATTR, True)


def _pairwise_local_overlap_components(
    reference_masks: np.ndarray,
    measurement_masks: np.ndarray,
    *,
    normalize_weighted_overlap: bool,
) -> dict[str, np.ndarray]:
    binary_intersections = _pairwise_sparse_mask_dot(
        reference_masks, measurement_masks, binary=True
    )
    areas_reference = _mask_support_areas(reference_masks)
    areas_measurement = _mask_support_areas(measurement_masks)
    min_areas = np.minimum(areas_reference[:, None], areas_measurement[None, :])
    max_areas = np.maximum(areas_reference[:, None], areas_measurement[None, :])

    reference_containment = _safe_fraction(binary_intersections, areas_reference[:, None])
    measurement_containment = _safe_fraction(
        binary_intersections, areas_measurement[None, :]
    )

    reference_overlap_masks = _mask_stack_for_weighted_overlap(
        reference_masks, normalize_per_roi=normalize_weighted_overlap
    )
    measurement_overlap_masks = _mask_stack_for_weighted_overlap(
        measurement_masks, normalize_per_roi=normalize_weighted_overlap
    )
    weighted_intersections = _pairwise_sparse_mask_dot(
        reference_overlap_masks, measurement_overlap_masks, binary=False
    )
    weighted_areas_reference = _mask_l1_areas(reference_overlap_masks)
    weighted_areas_measurement = _mask_l1_areas(measurement_overlap_masks)
    weighted_dice_denominator = (
        weighted_areas_reference[:, None] + weighted_areas_measurement[None, :]
    )

    return {
        "binary_intersection": binary_intersections,
        "weighted_intersection": weighted_intersections,
        "weighted_dice_similarity": _safe_fraction(
            2.0 * weighted_intersections, weighted_dice_denominator
        ),
        "overlap_min_fraction": _safe_fraction(binary_intersections, min_areas),
        "overlap_max_fraction": _safe_fraction(binary_intersections, max_areas),
        "reference_containment": reference_containment,
        "measurement_containment": measurement_containment,
        "containment_asymmetry": np.abs(reference_containment - measurement_containment),
    }


def _mask_stack_for_weighted_overlap(
    masks: np.ndarray, *, normalize_per_roi: bool
) -> np.ndarray:
    mask_array = np.asarray(masks, dtype=float)
    if mask_array.ndim != 3:
        raise ValueError("masks must have shape (n_roi, height, width)")
    mask_array = np.nan_to_num(mask_array, nan=0.0, posinf=0.0, neginf=0.0)
    mask_array = np.maximum(mask_array, 0.0)
    if normalize_per_roi and mask_array.shape[0] > 0:
        flat_masks = mask_array.reshape(mask_array.shape[0], -1)
        max_values = np.max(flat_masks, axis=1)
        positive = max_values > 0.0
        if np.any(positive):
            mask_array = np.array(mask_array, copy=True)
            mask_array[positive] /= max_values[positive, None, None]
    return mask_array


def _mask_l1_areas(masks: np.ndarray) -> np.ndarray:
    mask_array = np.asarray(masks, dtype=float)
    if mask_array.ndim != 3:
        raise ValueError("masks must have shape (n_roi, height, width)")
    mask_array = np.nan_to_num(mask_array, nan=0.0, posinf=0.0, neginf=0.0)
    return np.sum(np.maximum(mask_array, 0.0), axis=(1, 2))


def _safe_fraction(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    shape = np.broadcast_shapes(numerator.shape, denominator.shape)
    numerator = np.broadcast_to(numerator, shape)
    denominator = np.broadcast_to(denominator, shape)
    result = np.zeros(shape, dtype=float)
    np.divide(numerator, denominator, out=result, where=denominator > 0.0)
    return np.clip(result, 0.0, 1.0)


def _pairwise_symmetric_chamfer_distance_cost(
    reference_masks: np.ndarray, measurement_masks: np.ndarray, *, scale: float
) -> np.ndarray:
    reference_array = np.asarray(reference_masks)
    measurement_array = np.asarray(measurement_masks)
    cost_shape = (int(reference_array.shape[0]), int(measurement_array.shape[0]))
    if cost_shape[0] == 0 or cost_shape[1] == 0:
        return np.zeros(cost_shape, dtype=float)
    if reference_array.shape[1:] != measurement_array.shape[1:]:
        raise ValueError("Mask stacks must have matching spatial shapes")

    reference_supports = _roi_support_pixel_indices(reference_array)
    measurement_supports = _roi_support_pixel_indices(measurement_array)
    reference_to_measurement = _pairwise_mean_distance_to_mask_stack(
        reference_supports, measurement_array
    )
    measurement_to_reference = _pairwise_mean_distance_to_mask_stack(
        measurement_supports, reference_array
    ).T
    normalized_scale = max(float(scale), 1.0e-6)
    cost = 0.5 * (reference_to_measurement + measurement_to_reference)
    return np.nan_to_num(cost / normalized_scale, nan=0.0, posinf=1.0e6, neginf=0.0)


def _roi_support_pixel_indices(masks: np.ndarray) -> tuple[np.ndarray, ...]:
    mask_array = np.asarray(masks) > 0
    if mask_array.ndim != 3:
        raise ValueError("masks must have shape (n_roi, height, width)")
    return tuple(np.flatnonzero(mask.reshape(-1)) for mask in mask_array)


def _pairwise_mean_distance_to_mask_stack(
    source_support_indices: Sequence[np.ndarray], target_masks: np.ndarray
) -> np.ndarray:
    target_array = np.asarray(target_masks) > 0
    if target_array.ndim != 3:
        raise ValueError("target_masks must have shape (n_roi, height, width)")
    result = np.zeros((len(source_support_indices), target_array.shape[0]), dtype=float)
    for target_index, target_mask in enumerate(target_array):
        distances = _chamfer_distance_to_mask(target_mask).reshape(-1)
        for source_index, source_pixels in enumerate(source_support_indices):
            if source_pixels.size:
                result[source_index, target_index] = float(
                    np.mean(distances[source_pixels])
                )
    return result


def _chamfer_distance_to_mask(mask: np.ndarray) -> np.ndarray:
    support = np.asarray(mask) > 0
    if support.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    height, width = support.shape
    max_distance = float(height + width)
    distances = np.full((height, width), max_distance, dtype=float)
    distances[support] = 0.0
    if not np.any(support):
        return distances

    diagonal_cost = float(np.sqrt(2.0))
    for y_index in range(height):
        for x_index in range(width):
            best = distances[y_index, x_index]
            if y_index > 0:
                best = min(best, distances[y_index - 1, x_index] + 1.0)
                if x_index > 0:
                    best = min(best, distances[y_index - 1, x_index - 1] + diagonal_cost)
                if x_index + 1 < width:
                    best = min(best, distances[y_index - 1, x_index + 1] + diagonal_cost)
            if x_index > 0:
                best = min(best, distances[y_index, x_index - 1] + 1.0)
            distances[y_index, x_index] = best

    for y_index in range(height - 1, -1, -1):
        for x_index in range(width - 1, -1, -1):
            best = distances[y_index, x_index]
            if y_index + 1 < height:
                best = min(best, distances[y_index + 1, x_index] + 1.0)
                if x_index > 0:
                    best = min(best, distances[y_index + 1, x_index - 1] + diagonal_cost)
                if x_index + 1 < width:
                    best = min(best, distances[y_index + 1, x_index + 1] + diagonal_cost)
            if x_index + 1 < width:
                best = min(best, distances[y_index, x_index + 1] + 1.0)
            distances[y_index, x_index] = best
    return distances


def _pairwise_fov_patch_correlations(
    reference_plane: Any,
    measurement_plane: Any,
    *,
    order: str,
    weighted_centroids: bool,
    patch_radius: int,
    similarity_epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    cost_shape = (reference_plane.n_rois, measurement_plane.n_rois)
    if reference_plane.fov is None or measurement_plane.fov is None:
        return np.zeros(cost_shape, dtype=float), np.zeros(cost_shape, dtype=bool)

    reference_centroids_xy = _centroids_xy(
        reference_plane, order=order, weighted=weighted_centroids
    )
    measurement_centroids_xy = _centroids_xy(
        measurement_plane, order=order, weighted=weighted_centroids
    )
    reference_patches, reference_valid = _normalized_image_patches(
        reference_plane.fov,
        reference_centroids_xy,
        patch_radius=patch_radius,
        similarity_epsilon=similarity_epsilon,
    )
    measurement_patches, measurement_valid = _normalized_image_patches(
        measurement_plane.fov,
        measurement_centroids_xy,
        patch_radius=patch_radius,
        similarity_epsilon=similarity_epsilon,
    )
    correlations = reference_patches @ measurement_patches.T
    valid_pairs = reference_valid[:, None] & measurement_valid[None, :]
    correlations = np.where(valid_pairs, np.clip(correlations, -1.0, 1.0), 0.0)
    return correlations, valid_pairs


def _centroids_xy(plane: Any, *, order: str, weighted: bool) -> np.ndarray:
    centroids = plane.centroids(order=order, weighted=weighted)
    if order == "xy":
        return np.asarray(centroids, dtype=float).T
    return np.asarray(centroids, dtype=float)[[1, 0], :].T


def _normalized_image_patches(
    image: np.ndarray,
    centroids_xy: np.ndarray,
    *,
    patch_radius: int,
    similarity_epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    image_array = np.asarray(image, dtype=float)
    centroids_xy = np.asarray(centroids_xy, dtype=float)
    patch_size = 2 * int(patch_radius) + 1
    patches = np.zeros((centroids_xy.shape[0], patch_size * patch_size), dtype=float)
    valid = np.zeros((centroids_xy.shape[0],), dtype=bool)
    if image_array.ndim != 2:
        return patches, valid

    padded = np.pad(
        image_array,
        ((patch_radius, patch_radius), (patch_radius, patch_radius)),
        mode="constant",
        constant_values=np.nan,
    )
    height, width = image_array.shape
    for roi_index, (x_coord, y_coord) in enumerate(centroids_xy):
        if not (np.isfinite(x_coord) and np.isfinite(y_coord)):
            continue
        x_index = int(round(float(x_coord)))
        y_index = int(round(float(y_coord)))
        if x_index < 0 or x_index >= width or y_index < 0 or y_index >= height:
            continue
        padded_x = x_index + patch_radius
        padded_y = y_index + patch_radius
        patch = padded[
            padded_y - patch_radius : padded_y + patch_radius + 1,
            padded_x - patch_radius : padded_x + patch_radius + 1,
        ]
        if patch.shape != (patch_size, patch_size):
            continue
        finite = np.isfinite(patch)
        if np.count_nonzero(finite) < 2:
            continue
        mean_value = float(np.mean(patch[finite]))
        centered = np.array(patch, dtype=float, copy=True)
        centered[~finite] = mean_value
        centered -= mean_value
        norm = float(np.linalg.norm(centered.ravel()))
        if norm <= similarity_epsilon:
            continue
        patches[roi_index] = centered.ravel() / norm
        valid[roi_index] = True
    return patches, valid


def _pairwise_neighbor_constellation_cost(
    reference_plane: Any,
    measurement_plane: Any,
    *,
    order: str,
    weighted_centroids: bool,
    neighbor_k: int,
    scale: float,
) -> np.ndarray:
    reference_signature = _neighbor_distance_signature(
        reference_plane,
        order=order,
        weighted=weighted_centroids,
        neighbor_k=neighbor_k,
        scale=scale,
    )
    measurement_signature = _neighbor_distance_signature(
        measurement_plane,
        order=order,
        weighted=weighted_centroids,
        neighbor_k=neighbor_k,
        scale=scale,
    )
    if reference_signature.shape[0] == 0 or measurement_signature.shape[0] == 0:
        return np.zeros(
            (reference_signature.shape[0], measurement_signature.shape[0]), dtype=float
        )
    return np.mean(
        np.abs(reference_signature[:, None, :] - measurement_signature[None, :, :]),
        axis=2,
    )


def _neighbor_distance_signature(
    plane: Any,
    *,
    order: str,
    weighted: bool,
    neighbor_k: int,
    scale: float,
) -> np.ndarray:
    centroids = plane.centroids(order=order, weighted=weighted).T
    n_rois = int(centroids.shape[0])
    signature = np.zeros((n_rois, int(neighbor_k)), dtype=float)
    if n_rois <= 1:
        return signature
    diffs = centroids[:, None, :] - centroids[None, :, :]
    distances = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(distances, np.inf)
    normalized_scale = max(float(scale), 1.0e-6)
    for roi_index in range(n_rois):
        nearest = np.sort(distances[roi_index])
        nearest = nearest[np.isfinite(nearest)]
        if nearest.size == 0:
            continue
        used = min(int(neighbor_k), int(nearest.size))
        signature[roi_index, :used] = nearest[:used] / normalized_scale
        if used < neighbor_k:
            signature[roi_index, used:] = signature[roi_index, used - 1]
    return signature


def _pairwise_centroid_rank_cost(centroid_distances: np.ndarray) -> np.ndarray:
    distances = np.asarray(centroid_distances, dtype=float)
    if distances.ndim != 2:
        raise ValueError("centroid_distances must be two-dimensional")
    n_reference, n_measurement = distances.shape
    if n_reference == 0 or n_measurement == 0:
        return np.zeros((n_reference, n_measurement), dtype=float)

    row_rank = np.zeros((n_reference, n_measurement), dtype=float)
    row_order = np.argsort(distances, axis=1, kind="stable")
    row_rank[np.arange(n_reference)[:, None], row_order] = np.arange(n_measurement)[
        None, :
    ]

    column_rank = np.zeros((n_reference, n_measurement), dtype=float)
    column_order = np.argsort(distances, axis=0, kind="stable")
    column_rank[column_order, np.arange(n_measurement)[None, :]] = np.arange(
        n_reference
    )[:, None]

    row_scale = max(n_measurement - 1, 1)
    column_scale = max(n_reference - 1, 1)
    return 0.5 * (row_rank / row_scale + column_rank / column_scale)
