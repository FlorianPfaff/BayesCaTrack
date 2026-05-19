"""Association-guided nonrigid registration refinement for Track2p-style ROIs."""

# pylint: disable=protected-access

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

import numpy as np

from .association.registered_masks import replace_empty_registered_masks
from .core.bridge import CalciumPlaneData
from .fov_affine_registration import estimate_fov_affine_transform
from .nonrigid_registration import (
    _affine_inverse_grid,
    _bspline_control_shape,
    _bspline_inverse_grid,
    _canonical_nonrigid_transform,
    _finite_image,
    _idw_inverse_grid,
    _refine_inverse_grid_by_intensity_flow,
    _tps_inverse_grid,
    _valid_sample_mask,
    _warp_image_bilinear,
    _warp_mask_stack_nearest,
)

AssociationGuidedNonrigidRegistrationTransform = Literal[
    "association-guided-bspline",
    "association-guided-b-spline",
    "association-guided-thin-plate-spline",
    "association-guided-tps",
    "association-guided-landmark-tps",
    "association-guided-local-affine-grid",
    "association-guided-optical-flow",
]

ASSOCIATION_GUIDED_NONRIGID_REGISTRATION_TRANSFORM_TYPES: tuple[str, ...] = (
    "association-guided-bspline",
    "association-guided-b-spline",
    "association-guided-thin-plate-spline",
    "association-guided-tps",
    "association-guided-landmark-tps",
    "association-guided-local-affine-grid",
    "association-guided-optical-flow",
)


@dataclass(frozen=True)
class AssociationGuidedNonrigidRegistration:
    """Result of a pseudo-link-guided dense nonrigid registration refinement."""

    reference_plane: CalciumPlaneData
    measurement_plane: CalciumPlaneData
    registered_measurement_plane: CalciumPlaneData
    transform_type: str
    selected_pseudo_links: np.ndarray
    pseudo_link_counts: tuple[int, ...]
    pseudo_link_mean_costs: tuple[float, ...]
    pseudo_link_min_margins: tuple[float, ...]
    inverse_y: np.ndarray
    inverse_x: np.ndarray


# pylint: disable=too-many-arguments,too-many-locals,too-many-branches

def register_measurement_plane_by_association_guided_nonrigid(
    reference_plane: CalciumPlaneData,
    measurement_plane: CalciumPlaneData,
    *,
    transform_type: AssociationGuidedNonrigidRegistrationTransform | str = "association-guided-tps",
    iterations: int = 2,
    min_pseudo_links: int = 4,
    max_pseudo_links: int = 256,
    pseudo_link_cost_threshold: float | None = 4.0,
    pseudo_link_min_margin: float = 0.25,
    pseudo_link_pairwise_cost_kwargs: Mapping[str, Any] | None = None,
    order: str = "xy",
    weighted_centroids: bool = False,
    grid_shape: tuple[int, int] = (5, 5),
    min_tile_size: int = 24,
    max_shift_fraction: float = 0.75,
    tps_regularization: float = 1.0e-3,
    bspline_regularization: float = 1.0e-2,
    optical_flow_iterations: int = 12,
    optical_flow_alpha: float = 25.0,
    **unused_options: object,
) -> AssociationGuidedNonrigidRegistration:
    """Refine a nonrigid FOV warp with high-confidence unsupervised ROI links.

    The refinement deliberately does not use Track2p/manual ground truth.  Each
    iteration computes ROI association costs under the current warp, keeps only
    mutual low-cost row/column best matches with a configurable margin, and uses
    those pseudo-landmarks together with the image-tile landmarks to refit the
    dense inverse warp.  If too few pseudo-links are available, the function
    falls back to the ordinary image-driven nonrigid warp instead of returning a
    brittle association-only transform.
    """

    del unused_options
    method = _canonical_association_guided_transform(transform_type)
    if reference_plane.fov is None or measurement_plane.fov is None:
        raise ValueError("Both planes must provide FOV images for association-guided registration")
    if iterations < 1:
        raise ValueError("iterations must be at least one")
    if min_pseudo_links < 0:
        raise ValueError("min_pseudo_links must be non-negative")
    if max_pseudo_links < 1:
        raise ValueError("max_pseudo_links must be at least one")
    if pseudo_link_cost_threshold is not None and pseudo_link_cost_threshold < 0.0:
        raise ValueError("pseudo_link_cost_threshold must be non-negative or None")
    if pseudo_link_min_margin < 0.0:
        raise ValueError("pseudo_link_min_margin must be non-negative")

    grid_shape = (int(grid_shape[0]), int(grid_shape[1]))
    reference = _finite_image(reference_plane.fov)
    measurement = _finite_image(measurement_plane.fov)
    estimate = estimate_fov_affine_transform(
        reference,
        measurement,
        grid_shape=grid_shape,
        min_tile_size=min_tile_size,
        max_shift_fraction=max_shift_fraction,
    )
    output_shape = reference_plane.image_shape
    fallback_y, fallback_x = _affine_inverse_grid(estimate.inverse_matrix_xy, output_shape)

    registered_plane = _registered_plane_from_inverse_grid(
        measurement_plane,
        measurement,
        fallback_y,
        fallback_x,
        output_shape=output_shape,
        transform_type=f"association-guided-{method}",
        backend="association-guided-affine-initialization",
        reason="affine FOV initialization before pseudo-link refinement",
        ops_extra={
            "association_guided_registration_iterations_requested": int(iterations),
            "association_guided_registration_iterations_completed": 0,
            "association_guided_registration_pseudo_links": 0,
        },
    )

    pairwise_cost_kwargs = _association_guided_default_pairwise_cost_kwargs()
    if pseudo_link_pairwise_cost_kwargs is not None:
        pairwise_cost_kwargs.update(dict(pseudo_link_pairwise_cost_kwargs))

    pseudo_link_counts: list[int] = []
    pseudo_link_mean_costs: list[float] = []
    pseudo_link_min_margins: list[float] = []
    selected_links = np.zeros((0, 4), dtype=float)
    applied_links = selected_links
    completed_iterations = 0
    final_inverse_y, final_inverse_x = fallback_y, fallback_x

    for iteration_index in range(int(iterations)):
        cost_matrix = _association_guided_cost_matrix(
            reference_plane,
            registered_plane,
            order=order,
            weighted_centroids=weighted_centroids,
            pairwise_cost_kwargs=pairwise_cost_kwargs,
        )
        selected_links = _select_mutual_pseudo_links(
            cost_matrix,
            cost_threshold=pseudo_link_cost_threshold,
            min_margin=pseudo_link_min_margin,
            max_links=max_pseudo_links,
        )
        pseudo_link_counts.append(int(selected_links.shape[0]))
        pseudo_link_mean_costs.append(float(np.mean(selected_links[:, 2])) if selected_links.size else float("nan"))
        pseudo_link_min_margins.append(float(np.min(selected_links[:, 3])) if selected_links.size else float("nan"))

        if selected_links.shape[0] < int(min_pseudo_links):
            if iteration_index == 0:
                final_inverse_y, final_inverse_x, backend, landmark_count = _fit_association_guided_inverse_grid(
                    reference_plane,
                    measurement_plane,
                    estimate,
                    selected_links[:0],
                    method=method,
                    output_shape=output_shape,
                    fallback_y=fallback_y,
                    fallback_x=fallback_x,
                    grid_shape=grid_shape,
                    weighted_centroids=weighted_centroids,
                    tps_regularization=tps_regularization,
                    bspline_regularization=bspline_regularization,
                    optical_flow_iterations=optical_flow_iterations,
                    optical_flow_alpha=optical_flow_alpha,
                )
                registered_plane = _registered_plane_from_inverse_grid(
                    measurement_plane,
                    measurement,
                    final_inverse_y,
                    final_inverse_x,
                    output_shape=output_shape,
                    transform_type=f"association-guided-{method}",
                    backend=backend,
                    reason="insufficient pseudo-links; image-driven nonrigid fallback",
                    ops_extra={
                        "association_guided_registration_iterations_requested": int(iterations),
                        "association_guided_registration_iterations_completed": 0,
                        "association_guided_registration_pseudo_links": int(selected_links.shape[0]),
                        "association_guided_registration_landmarks": int(landmark_count),
                    },
                )
            break

        final_inverse_y, final_inverse_x, backend, landmark_count = _fit_association_guided_inverse_grid(
            reference_plane,
            measurement_plane,
            estimate,
            selected_links,
            method=method,
            output_shape=output_shape,
            fallback_y=fallback_y,
            fallback_x=fallback_x,
            grid_shape=grid_shape,
            weighted_centroids=weighted_centroids,
            tps_regularization=tps_regularization,
            bspline_regularization=bspline_regularization,
            optical_flow_iterations=optical_flow_iterations,
            optical_flow_alpha=optical_flow_alpha,
        )
        completed_iterations = iteration_index + 1
        applied_links = selected_links
        registered_plane = _registered_plane_from_inverse_grid(
            measurement_plane,
            measurement,
            final_inverse_y,
            final_inverse_x,
            output_shape=output_shape,
            transform_type=f"association-guided-{method}",
            backend=backend,
            reason="unsupervised mutual-best pseudo-link nonrigid refinement",
            ops_extra={
                "association_guided_registration_iterations_requested": int(iterations),
                "association_guided_registration_iterations_completed": int(completed_iterations),
                "association_guided_registration_pseudo_links": int(selected_links.shape[0]),
                "association_guided_registration_landmarks": int(landmark_count),
                "association_guided_registration_pseudo_link_mean_cost": float(np.mean(selected_links[:, 2])),
                "association_guided_registration_pseudo_link_min_margin": float(np.min(selected_links[:, 3])),
            },
        )

    return AssociationGuidedNonrigidRegistration(
        reference_plane=reference_plane,
        measurement_plane=measurement_plane,
        registered_measurement_plane=registered_plane,
        transform_type=f"association-guided-{method}",
        selected_pseudo_links=np.asarray(applied_links[:, :2], dtype=int),
        pseudo_link_counts=tuple(pseudo_link_counts),
        pseudo_link_mean_costs=tuple(pseudo_link_mean_costs),
        pseudo_link_min_margins=tuple(pseudo_link_min_margins),
        inverse_y=final_inverse_y,
        inverse_x=final_inverse_x,
    )


def _canonical_association_guided_transform(transform_type: str) -> str:
    normalized = str(transform_type).lower().replace("_", "-")
    for prefix in ("association-guided-", "assoc-guided-", "pseudo-link-guided-"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break
    return _canonical_nonrigid_transform(normalized)


def _association_guided_default_pairwise_cost_kwargs() -> dict[str, Any]:
    return {
        "centroid_weight": 1.0,
        "iou_weight": 4.0,
        "mask_cosine_weight": 1.0,
        "area_weight": 0.25,
        "roi_feature_weight": 0.0,
        "cell_probability_weight": 0.0,
        "similarity_epsilon": 1.0e-6,
    }


def _association_guided_cost_matrix(
    reference_plane: CalciumPlaneData,
    registered_plane: CalciumPlaneData,
    *,
    order: str,
    weighted_centroids: bool,
    pairwise_cost_kwargs: Mapping[str, Any],
) -> np.ndarray:
    kwargs = dict(pairwise_cost_kwargs)
    kwargs.pop("return_components", None)
    registered_plane, empty_registered_rois = replace_empty_registered_masks(registered_plane)
    cost_matrix = reference_plane.build_pairwise_cost_matrix(
        registered_plane,
        order=order,
        weighted_centroids=weighted_centroids,
        return_components=False,
        **kwargs,
    )
    if isinstance(cost_matrix, tuple):
        cost_matrix = cost_matrix[0]
    cost_matrix = np.asarray(cost_matrix, dtype=float)
    cost_matrix[:, empty_registered_rois] = np.inf
    return cost_matrix


def _select_mutual_pseudo_links(
    cost_matrix: np.ndarray,
    *,
    cost_threshold: float | None,
    min_margin: float,
    max_links: int,
) -> np.ndarray:
    costs = np.asarray(cost_matrix, dtype=float)
    if costs.ndim != 2:
        raise ValueError("cost_matrix must be two-dimensional")
    if costs.size == 0:
        return np.zeros((0, 4), dtype=float)
    finite_costs = np.where(np.isfinite(costs), costs, np.inf)
    candidate_rows: list[tuple[int, int, float, float]] = []
    for reference_index in range(finite_costs.shape[0]):
        row = finite_costs[reference_index]
        if not np.any(np.isfinite(row)):
            continue
        measurement_index = int(np.argmin(row))
        best_cost = float(row[measurement_index])
        if cost_threshold is not None and best_cost > float(cost_threshold):
            continue
        column = finite_costs[:, measurement_index]
        if not np.any(np.isfinite(column)) or int(np.argmin(column)) != reference_index:
            continue
        margin = min(
            _second_best_gap(row, measurement_index, best_cost),
            _second_best_gap(column, reference_index, best_cost),
        )
        if margin < float(min_margin):
            continue
        candidate_rows.append((reference_index, measurement_index, best_cost, margin))
    if not candidate_rows:
        return np.zeros((0, 4), dtype=float)
    candidate_rows.sort(key=lambda item: (item[2], -item[3], item[0], item[1]))
    return np.asarray(candidate_rows[: int(max_links)], dtype=float)


def _second_best_gap(values: np.ndarray, best_index: int, best_cost: float) -> float:
    other = np.asarray(values, dtype=float).copy()
    other[int(best_index)] = np.inf
    second_best = float(np.min(other))
    if not np.isfinite(second_best):
        return float("inf")
    return second_best - float(best_cost)


# pylint: disable=too-many-arguments,too-many-locals

def _fit_association_guided_inverse_grid(
    reference_plane: CalciumPlaneData,
    measurement_plane: CalciumPlaneData,
    estimate: Any,
    selected_links: np.ndarray,
    *,
    method: str,
    output_shape: tuple[int, int],
    fallback_y: np.ndarray,
    fallback_x: np.ndarray,
    grid_shape: tuple[int, int],
    weighted_centroids: bool,
    tps_regularization: float,
    bspline_regularization: float,
    optical_flow_iterations: int,
    optical_flow_alpha: float,
) -> tuple[np.ndarray, np.ndarray, str, int]:
    reference_landmarks = [np.asarray(estimate.tile_reference_xy, dtype=float).reshape(-1, 2)]
    measurement_landmarks = [np.asarray(estimate.tile_measurement_xy, dtype=float).reshape(-1, 2)]
    links = np.asarray(selected_links, dtype=float)
    if links.size:
        reference_centroids_xy = reference_plane.centroids(order="xy", weighted=weighted_centroids).T
        measurement_centroids_xy = measurement_plane.centroids(order="xy", weighted=weighted_centroids).T
        reference_indices = links[:, 0].astype(int)
        measurement_indices = links[:, 1].astype(int)
        reference_landmarks.insert(0, reference_centroids_xy[reference_indices])
        measurement_landmarks.insert(0, measurement_centroids_xy[measurement_indices])

    reference_xy = np.vstack(reference_landmarks) if reference_landmarks else np.zeros((0, 2), dtype=float)
    measurement_xy = np.vstack(measurement_landmarks) if measurement_landmarks else np.zeros((0, 2), dtype=float)
    reference_xy, measurement_xy = _deduplicate_landmarks(reference_xy, measurement_xy)
    if reference_xy.shape[0] < 3:
        return fallback_y, fallback_x, "association-guided-affine-fallback-insufficient-landmarks", int(reference_xy.shape[0])

    if method == "tps":
        inverse_y, inverse_x = _tps_inverse_grid(
            reference_xy,
            measurement_xy,
            output_shape,
            tps_regularization=tps_regularization,
            fallback_y=fallback_y,
            fallback_x=fallback_x,
        )
        backend = "association-guided-thin-plate-spline-landmark-warp"
    elif method == "bspline":
        inverse_y, inverse_x = _bspline_inverse_grid(
            reference_xy,
            measurement_xy,
            output_shape,
            fallback_y=fallback_y,
            fallback_x=fallback_x,
            control_shape_yx=_bspline_control_shape(output_shape, grid_shape),
            regularization=bspline_regularization,
        )
        backend = "association-guided-tensor-product-cubic-bspline-landmark-warp"
    else:
        nearest = 4 if method == "local-affine-grid" else None
        inverse_y, inverse_x = _idw_inverse_grid(
            reference_xy,
            measurement_xy,
            output_shape,
            fallback_y=fallback_y,
            fallback_x=fallback_x,
            nearest=nearest,
            smooth_iterations=0,
        )
        backend = "association-guided-local-landmark-grid-warp" if method == "local-affine-grid" else "association-guided-smooth-landmark-displacement-warp"

    if method == "optical-flow":
        if reference_plane.fov is None or measurement_plane.fov is None:
            raise ValueError("Both planes must provide FOV images for optical-flow refinement")
        inverse_y, inverse_x = _refine_inverse_grid_by_intensity_flow(
            _finite_image(reference_plane.fov),
            _finite_image(measurement_plane.fov),
            inverse_y,
            inverse_x,
            iterations=optical_flow_iterations,
            alpha=optical_flow_alpha,
        )
        backend = "association-guided-landmark-warp-with-intensity-flow-refinement"
    return inverse_y, inverse_x, backend, int(reference_xy.shape[0])


def _deduplicate_landmarks(reference_xy: np.ndarray, measurement_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if reference_xy.shape[0] == 0:
        return reference_xy, measurement_xy
    finite = np.all(np.isfinite(reference_xy), axis=1) & np.all(np.isfinite(measurement_xy), axis=1)
    reference_xy = np.asarray(reference_xy[finite], dtype=float)
    measurement_xy = np.asarray(measurement_xy[finite], dtype=float)
    if reference_xy.shape[0] == 0:
        return reference_xy, measurement_xy
    _, unique_indices = np.unique(np.round(reference_xy, decimals=3), axis=0, return_index=True)
    unique_indices = np.sort(unique_indices)
    return reference_xy[unique_indices], measurement_xy[unique_indices]


def _registered_plane_from_inverse_grid(
    measurement_plane: CalciumPlaneData,
    measurement_image: np.ndarray,
    inverse_y: np.ndarray,
    inverse_x: np.ndarray,
    *,
    output_shape: tuple[int, int],
    transform_type: str,
    backend: str,
    reason: str,
    ops_extra: Mapping[str, Any] | None = None,
) -> CalciumPlaneData:
    registered_masks = _warp_mask_stack_nearest(
        measurement_plane.roi_masks,
        inverse_y,
        inverse_x,
        output_shape=output_shape,
    )
    registered_fov = _warp_image_bilinear(measurement_image, inverse_y, inverse_x)
    valid_fraction = float(np.mean(_valid_sample_mask(inverse_y, inverse_x, measurement_image.shape)))
    ops = {} if measurement_plane.ops is None else dict(measurement_plane.ops)
    ops.update(
        {
            "registration_backend": "bayescatrack-association-guided-nonrigid",
            "registration_transform_type": transform_type,
            "registration_backend_reason": reason,
            "association_guided_registration_backend": backend,
            "association_guided_registration_inverse_warp_valid_fraction": valid_fraction,
        }
    )
    if ops_extra is not None:
        ops.update(dict(ops_extra))
    return measurement_plane.with_replaced_masks(
        registered_masks,
        fov=registered_fov,
        source=f"{measurement_plane.source}_{transform_type}_registered".replace("--", "-"),
        ops=ops,
    )
