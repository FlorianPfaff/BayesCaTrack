"""Self-bootstrapped registration refinement for Track2p global assignment."""

# pylint: disable=too-many-arguments,too-many-instance-attributes,too-many-locals

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from bayescatrack.association import pyrecest_global_assignment as _global_assignment
from bayescatrack.association.pyrecest_global_assignment import (
    AssociationCost,
    GlobalAssignmentRun,
    SessionEdge,
    session_edge_pairs,
)
from bayescatrack.core.bridge import CalciumPlaneData, Track2pSession
from bayescatrack.fov_affine_registration import apply_affine_roi_mask_warp
from bayescatrack.track2p_registration import register_plane_pair

BootstrapTransform = Literal["translation", "affine"]


@dataclass(frozen=True)
class BootstrapRegistrationConfig:
    """Options controlling assignment-guided residual registration refinement."""

    iterations: int = 2
    transform: BootstrapTransform = "affine"
    min_matches: int = 6
    min_cost_margin: float = 0.25
    max_anchor_cost: float | None = None
    max_rmse: float | None = 8.0
    robust_refit: bool = True
    refine_skip_edges: bool = True

    def __post_init__(self) -> None:
        if self.iterations < 0:
            raise ValueError("iterations must be non-negative")
        if self.transform not in {"translation", "affine"}:
            raise ValueError("transform must be 'translation' or 'affine'")
        if self.min_matches < 1:
            raise ValueError("min_matches must be at least one")
        if self.min_cost_margin < 0.0:
            raise ValueError("min_cost_margin must be non-negative")
        if self.max_anchor_cost is not None and self.max_anchor_cost < 0.0:
            raise ValueError("max_anchor_cost must be non-negative")
        if self.max_rmse is not None and self.max_rmse <= 0.0:
            raise ValueError("max_rmse must be strictly positive")


@dataclass(frozen=True)
class ResidualRegistrationEstimate:
    """Fitted residual transform for one session edge."""

    edge: SessionEdge
    matrix_xy: np.ndarray
    transform: BootstrapTransform
    anchor_count: int
    inlier_count: int
    rmse: float


def solve_bootstrapped_global_assignment_for_sessions(
    sessions: Sequence[Track2pSession],
    *,
    bootstrap_config: BootstrapRegistrationConfig | None = None,
    max_gap: int = 2,
    cost: AssociationCost = "registered-iou",
    calibrated_model: Any | None = None,
    transform_type: str = "affine",
    start_cost: float = 5.0,
    end_cost: float = 5.0,
    gap_penalty: float = 1.0,
    cost_threshold: float | None = 6.0,
    order: str = "xy",
    weighted_centroids: bool = False,
    velocity_variance: float = 25.0,
    regularization: float = 1.0e-6,
    pairwise_cost_kwargs: Mapping[str, Any] | None = None,
    activity_tie_breaker_weight: float = 0.0,
    activity_tie_breaker_component: str = "activity_tiebreaker_cost",
    activity_trace_source: str = "auto",
    activity_event_threshold: float = 0.0,
) -> GlobalAssignmentRun:
    """Run global assignment with EM-like assignment-guided registration updates.

    The first pass uses the normal registration backend. Each bootstrap iteration
    extracts high-confidence assignment edges, fits a residual translation/affine
    warp from registered target ROI centroids to source ROI centroids, rebuilds
    the affected pairwise costs, and reruns the global path-cover solver.
    """

    sessions = list(sessions)
    config = bootstrap_config or BootstrapRegistrationConfig()
    run = _global_assignment.solve_global_assignment_for_sessions(
        sessions,
        max_gap=max_gap,
        cost=cost,
        calibrated_model=calibrated_model,
        transform_type=transform_type,
        start_cost=start_cost,
        end_cost=end_cost,
        gap_penalty=gap_penalty,
        cost_threshold=cost_threshold,
        order=order,
        weighted_centroids=weighted_centroids,
        velocity_variance=velocity_variance,
        regularization=regularization,
        pairwise_cost_kwargs=pairwise_cost_kwargs,
        activity_tie_breaker_weight=activity_tie_breaker_weight,
        activity_tie_breaker_component=activity_tie_breaker_component,
        activity_trace_source=activity_trace_source,
        activity_event_threshold=activity_event_threshold,
    )
    if config.iterations == 0 or len(sessions) < 2:
        return run

    registered_planes = _initial_registered_planes(
        sessions, max_gap=max_gap, transform_type=transform_type
    )
    session_sizes = tuple(int(session.plane_data.n_rois) for session in sessions)
    edges = session_edge_pairs(len(sessions), max_gap=max_gap)

    for _ in range(config.iterations):
        refined_planes, estimates = refine_registered_planes_from_assignment(
            sessions,
            run,
            current_registered_planes=registered_planes,
            config=config,
            order=order,
            weighted_centroids=weighted_centroids,
        )
        if not estimates:
            break
        registered_planes = refined_planes
        pairwise_costs = _build_costs_with_registered_planes(
            sessions,
            max_gap=max_gap,
            cost=cost,
            calibrated_model=calibrated_model,
            transform_type=transform_type,
            order=order,
            weighted_centroids=weighted_centroids,
            velocity_variance=velocity_variance,
            regularization=regularization,
            pairwise_cost_kwargs=pairwise_cost_kwargs,
            activity_tie_breaker_weight=activity_tie_breaker_weight,
            activity_tie_breaker_component=activity_tie_breaker_component,
            activity_trace_source=activity_trace_source,
            activity_event_threshold=activity_event_threshold,
            registered_planes_by_edge=registered_planes,
        )
        run = _global_assignment.solve_global_assignment_from_pairwise_costs(
            pairwise_costs,
            session_sizes=session_sizes,
            session_edges=edges,
            start_cost=start_cost,
            end_cost=end_cost,
            gap_penalty=gap_penalty,
            cost_threshold=cost_threshold,
        )
    return run


def _build_costs_with_registered_planes(
    sessions: Sequence[Track2pSession],
    *,
    registered_planes_by_edge: Mapping[SessionEdge, CalciumPlaneData],
    **kwargs: Any,
) -> dict[SessionEdge, np.ndarray]:
    original_register_plane_pair = _global_assignment.register_plane_pair

    def _register_plane_pair_from_cache(
        reference_plane: CalciumPlaneData,
        moving_plane: CalciumPlaneData,
        *,
        transform_type: str = "affine",
    ) -> CalciumPlaneData:
        for edge, registered_plane in registered_planes_by_edge.items():
            if (
                sessions[edge[0]].plane_data is reference_plane
                and sessions[edge[1]].plane_data is moving_plane
            ):
                return registered_plane
        return original_register_plane_pair(
            reference_plane, moving_plane, transform_type=transform_type
        )

    _global_assignment.register_plane_pair = _register_plane_pair_from_cache
    try:
        return _global_assignment.build_registered_pairwise_costs(sessions, **kwargs)
    finally:
        _global_assignment.register_plane_pair = original_register_plane_pair


def refine_registered_planes_from_assignment(
    sessions: Sequence[Track2pSession],
    assignment_run: GlobalAssignmentRun,
    *,
    current_registered_planes: Mapping[SessionEdge, CalciumPlaneData],
    config: BootstrapRegistrationConfig | None = None,
    order: str = "xy",
    weighted_centroids: bool = False,
) -> tuple[
    dict[SessionEdge, CalciumPlaneData],
    dict[SessionEdge, ResidualRegistrationEstimate],
]:
    """Return registered planes refined by residual transforms from solver anchors."""

    sessions = list(sessions)
    bootstrap_config = config or BootstrapRegistrationConfig()
    refined_planes = dict(current_registered_planes)
    estimates: dict[SessionEdge, ResidualRegistrationEstimate] = {}
    for edge in assignment_run.session_edges:
        if not bootstrap_config.refine_skip_edges and edge[1] != edge[0] + 1:
            continue
        if (
            edge not in current_registered_planes
            or edge not in assignment_run.pairwise_costs
        ):
            continue
        source_index, _ = edge
        reference_plane = sessions[source_index].plane_data
        registered_plane = current_registered_planes[edge]
        pairs = _anchor_pairs_from_assignment(
            assignment_run,
            edge,
            config=bootstrap_config,
        )
        estimate = fit_residual_transform_from_roi_pairs(
            reference_plane,
            registered_plane,
            pairs,
            edge=edge,
            config=bootstrap_config,
            order=order,
            weighted_centroids=weighted_centroids,
        )
        if estimate is None:
            continue
        refined_planes[edge] = _apply_residual_estimate(
            registered_plane,
            estimate,
            output_shape=reference_plane.image_shape,
        )
        estimates[edge] = estimate
    return refined_planes, estimates


def fit_residual_transform_from_roi_pairs(
    reference_plane: CalciumPlaneData,
    registered_measurement_plane: CalciumPlaneData,
    pairs: Sequence[tuple[int, int]],
    *,
    edge: SessionEdge = (0, 1),
    config: BootstrapRegistrationConfig | None = None,
    order: str = "xy",
    weighted_centroids: bool = False,
) -> ResidualRegistrationEstimate | None:
    """Fit a residual transform from matched registered-measurement ROI pairs."""

    bootstrap_config = config or BootstrapRegistrationConfig()
    if len(pairs) < bootstrap_config.min_matches:
        return None
    reference_xy, measurement_xy = _centroid_correspondences(
        reference_plane,
        registered_measurement_plane,
        pairs,
        order=order,
        weighted=weighted_centroids,
    )
    if reference_xy.shape[0] < bootstrap_config.min_matches:
        return None
    matrix_xy, transform = _fit_matrix(
        reference_xy,
        measurement_xy,
        preferred_transform=bootstrap_config.transform,
    )
    if matrix_xy is None:
        return None
    matrix_xy, inlier_count = _maybe_refit_inliers(
        reference_xy,
        measurement_xy,
        matrix_xy,
        transform=transform,
        enabled=bootstrap_config.robust_refit,
    )
    residual = _transform_points(measurement_xy, matrix_xy) - reference_xy
    rmse = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
    if bootstrap_config.max_rmse is not None and rmse > bootstrap_config.max_rmse:
        return None
    return ResidualRegistrationEstimate(
        edge=edge,
        matrix_xy=matrix_xy,
        transform=transform,
        anchor_count=int(reference_xy.shape[0]),
        inlier_count=int(inlier_count),
        rmse=rmse,
    )


def _initial_registered_planes(
    sessions: Sequence[Track2pSession],
    *,
    max_gap: int,
    transform_type: str,
) -> dict[SessionEdge, CalciumPlaneData]:
    return {
        edge: register_plane_pair(
            sessions[edge[0]].plane_data,
            sessions[edge[1]].plane_data,
            transform_type=transform_type,
        )
        for edge in session_edge_pairs(len(sessions), max_gap=max_gap)
    }


def _anchor_pairs_from_assignment(
    assignment_run: GlobalAssignmentRun,
    edge: SessionEdge,
    *,
    config: BootstrapRegistrationConfig,
) -> list[tuple[int, int]]:
    cost_matrix = np.asarray(assignment_run.pairwise_costs[edge], dtype=float)
    pairs: list[tuple[int, int]] = []
    for track in assignment_run.result.tracks:
        if edge[0] not in track or edge[1] not in track:
            continue
        source_roi = int(track[edge[0]])
        target_roi = int(track[edge[1]])
        if not (
            0 <= source_roi < cost_matrix.shape[0]
            and 0 <= target_roi < cost_matrix.shape[1]
        ):
            continue
        edge_cost = float(cost_matrix[source_roi, target_roi])
        if not np.isfinite(edge_cost):
            continue
        if config.max_anchor_cost is not None and edge_cost > config.max_anchor_cost:
            continue
        if _cost_margin(cost_matrix, source_roi, target_roi) < config.min_cost_margin:
            continue
        pairs.append((source_roi, target_roi))
    return pairs


def _cost_margin(cost_matrix: np.ndarray, row: int, col: int) -> float:
    selected = float(cost_matrix[row, col])
    alternatives: list[float] = []
    if cost_matrix.shape[1] > 1:
        row_costs = np.delete(cost_matrix[row, :], col)
        finite = row_costs[np.isfinite(row_costs)]
        if finite.size:
            alternatives.append(float(np.min(finite)))
    if cost_matrix.shape[0] > 1:
        col_costs = np.delete(cost_matrix[:, col], row)
        finite = col_costs[np.isfinite(col_costs)]
        if finite.size:
            alternatives.append(float(np.min(finite)))
    if not alternatives:
        return float("inf")
    return min(alternatives) - selected


def _centroid_correspondences(
    reference_plane: CalciumPlaneData,
    registered_measurement_plane: CalciumPlaneData,
    pairs: Sequence[tuple[int, int]],
    *,
    order: str,
    weighted: bool,
) -> tuple[np.ndarray, np.ndarray]:
    reference_xy: list[np.ndarray] = []
    measurement_xy: list[np.ndarray] = []
    for reference_roi, measurement_roi in pairs:
        reference_centroid = _safe_roi_centroid(
            reference_plane.roi_masks[reference_roi], order=order, weighted=weighted
        )
        measurement_centroid = _safe_roi_centroid(
            registered_measurement_plane.roi_masks[measurement_roi],
            order=order,
            weighted=weighted,
        )
        if reference_centroid is None or measurement_centroid is None:
            continue
        reference_xy.append(reference_centroid)
        measurement_xy.append(measurement_centroid)
    if not reference_xy:
        return np.zeros((0, 2), dtype=float), np.zeros((0, 2), dtype=float)
    return np.vstack(reference_xy), np.vstack(measurement_xy)


def _safe_roi_centroid(
    mask: np.ndarray,
    *,
    order: str,
    weighted: bool,
) -> np.ndarray | None:
    row_coords, col_coords = np.nonzero(mask)
    if row_coords.size == 0:
        return None
    if weighted:
        weights = np.asarray(mask[row_coords, col_coords], dtype=float)
    else:
        weights = np.ones(row_coords.shape[0], dtype=float)
    weight_sum = float(np.sum(weights))
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        return None
    centroid_y = float(np.dot(row_coords, weights) / weight_sum)
    centroid_x = float(np.dot(col_coords, weights) / weight_sum)
    if order == "xy":
        return np.asarray([centroid_x, centroid_y], dtype=float)
    if order == "yx":
        return np.asarray([centroid_y, centroid_x], dtype=float)
    raise ValueError("order must be 'xy' or 'yx'")


def _fit_matrix(
    reference_xy: np.ndarray,
    measurement_xy: np.ndarray,
    *,
    preferred_transform: BootstrapTransform,
) -> tuple[np.ndarray | None, BootstrapTransform]:
    if preferred_transform == "affine" and reference_xy.shape[0] >= 3:
        design = _design(measurement_xy)
        if np.linalg.matrix_rank(design) >= 3:
            coef, _, _, _ = np.linalg.lstsq(design, reference_xy, rcond=None)
            return np.asarray(coef.T, dtype=float), "affine"
    shift_xy = np.median(reference_xy - measurement_xy, axis=0)
    return _translation_matrix(shift_xy), "translation"


def _maybe_refit_inliers(
    reference_xy: np.ndarray,
    measurement_xy: np.ndarray,
    matrix_xy: np.ndarray,
    *,
    transform: BootstrapTransform,
    enabled: bool,
) -> tuple[np.ndarray, int]:
    if not enabled or reference_xy.shape[0] < 4:
        return matrix_xy, int(reference_xy.shape[0])
    residual_norm = np.linalg.norm(
        _transform_points(measurement_xy, matrix_xy) - reference_xy, axis=1
    )
    keep = _robust_inlier_mask(residual_norm)
    inlier_count = int(np.count_nonzero(keep))
    min_inliers = 3 if transform == "affine" else 1
    if inlier_count < min_inliers:
        return matrix_xy, int(reference_xy.shape[0])
    if (
        transform == "affine"
        and np.linalg.matrix_rank(_design(measurement_xy[keep])) < 3
    ):
        return matrix_xy, int(reference_xy.shape[0])
    refit_matrix, _ = _fit_matrix(
        reference_xy[keep],
        measurement_xy[keep],
        preferred_transform=transform,
    )
    if refit_matrix is None:
        return matrix_xy, int(reference_xy.shape[0])
    return refit_matrix, inlier_count


def _robust_inlier_mask(residual_norm: np.ndarray) -> np.ndarray:
    finite = residual_norm[np.isfinite(residual_norm)]
    if finite.size == 0:
        return np.ones_like(residual_norm, dtype=bool)
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    robust_scale = 1.4826 * mad
    cutoff = max(median + 3.0 * robust_scale, float(np.percentile(finite, 75)), 1.0)
    return np.isfinite(residual_norm) & (residual_norm <= cutoff)


def _apply_residual_estimate(
    registered_plane: CalciumPlaneData,
    estimate: ResidualRegistrationEstimate,
    *,
    output_shape: tuple[int, int],
) -> CalciumPlaneData:
    refined_masks = apply_affine_roi_mask_warp(
        registered_plane.roi_masks,
        estimate.matrix_xy,
        output_shape=output_shape,
    )
    ops = {} if registered_plane.ops is None else dict(registered_plane.ops)
    ops.update(
        {
            "bootstrap_registration_edge": tuple(int(value) for value in estimate.edge),
            "bootstrap_registration_transform": estimate.transform,
            "bootstrap_registration_matrix_xy": estimate.matrix_xy,
            "bootstrap_registration_anchor_count": int(estimate.anchor_count),
            "bootstrap_registration_inlier_count": int(estimate.inlier_count),
            "bootstrap_registration_rmse": float(estimate.rmse),
        }
    )
    return registered_plane.with_replaced_masks(
        refined_masks,
        fov=registered_plane.fov,
        source=f"{registered_plane.source}_bootstrap_registered",
        ops=ops,
    )


def _design(xy: np.ndarray) -> np.ndarray:
    return np.column_stack((xy, np.ones((xy.shape[0],), dtype=float)))


def _translation_matrix(shift_xy: np.ndarray) -> np.ndarray:
    return np.asarray(
        [[1.0, 0.0, float(shift_xy[0])], [0.0, 1.0, float(shift_xy[1])]],
        dtype=float,
    )


def _transform_points(xy: np.ndarray, matrix_xy: np.ndarray) -> np.ndarray:
    return _design(xy) @ np.asarray(matrix_xy, dtype=float).T


__all__ = [
    "BootstrapRegistrationConfig",
    "ResidualRegistrationEstimate",
    "fit_residual_transform_from_roi_pairs",
    "refine_registered_planes_from_assignment",
    "solve_bootstrapped_global_assignment_for_sessions",
]
