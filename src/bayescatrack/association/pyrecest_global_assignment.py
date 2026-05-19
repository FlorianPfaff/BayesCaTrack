"""BayesCaTrack adapter for PyRecEst global multi-session assignment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from bayescatrack.association.activity_similarity import (
    add_activity_similarity_components,
)
from bayescatrack.association.activity_tie_breaker import (
    activity_tie_breaker_cost_matrix,
)
from bayescatrack.association.calibrated_costs import (
    CalibratedAssociationModel,
    calibrated_cost_matrix_from_bundle,
)
from bayescatrack.association.higher_order_consistency import (
    HigherOrderConsistencyConfig,
    apply_higher_order_consistency,
)
from bayescatrack.association.registered_masks import (
    replace_empty_registered_masks,
)
from bayescatrack.association.shifted_overlap import (
    install_shifted_overlap_cost_patch,
    pairwise_kwargs_use_shifted_overlap,
)
from bayescatrack.core.bridge import (
    CalciumPlaneData,
    Track2pSession,
    build_session_pair_association_bundle,
)
from bayescatrack.track2p_registration import register_plane_pair

AssociationCost = Literal[
    "registered-iou",
    "registered-soft-iou",
    "registered-shifted-iou",
    "roi-aware",
    "roi-aware-shifted",
    "calibrated",
]
SessionEdge = tuple[int, int]


@dataclass(frozen=True)
class GlobalAssignmentRun:
    """Global assignment result plus the pairwise evidence used to build it."""

    result: Any
    pairwise_costs: dict[SessionEdge, np.ndarray]
    session_sizes: tuple[int, ...]
    session_edges: tuple[SessionEdge, ...]


def registered_iou_cost_kwargs(
    *, similarity_epsilon: float = 1.0e-6
) -> dict[str, float]:
    """Return cost kwargs for a Track2p-style registered IoU ablation."""

    return {
        "centroid_weight": 0.0,
        "iou_weight": 1.0,
        "mask_cosine_weight": 0.0,
        "area_weight": 0.0,
        "roi_feature_weight": 0.0,
        "cell_probability_weight": 0.0,
        "similarity_epsilon": float(similarity_epsilon),
    }


def registered_shifted_iou_cost_kwargs(
    *,
    similarity_epsilon: float = 1.0e-6,
    shifted_iou_radius: int = 2,
    shifted_iou_shift_penalty_weight: float = 0.0,
    shifted_iou_shift_penalty_scale: float | None = None,
) -> dict[str, float | int | bool]:
    """Return registered-IoU kwargs with local shifted-overlap matching enabled."""

    radius = int(shifted_iou_radius)
    if radius < 0:
        raise ValueError("shifted_iou_radius must be non-negative")
    shift_penalty_weight = float(shifted_iou_shift_penalty_weight)
    if shift_penalty_weight < 0.0:
        raise ValueError("shifted_iou_shift_penalty_weight must be non-negative")
    shift_penalty_scale = (
        None
        if shifted_iou_shift_penalty_scale is None
        else float(shifted_iou_shift_penalty_scale)
    )
    if shift_penalty_scale is not None and shift_penalty_scale <= 0.0:
        raise ValueError("shifted_iou_shift_penalty_scale must be strictly positive")
    kwargs: dict[str, float | int | bool] = dict(
        registered_iou_cost_kwargs(similarity_epsilon=similarity_epsilon)
    )
    kwargs.update(
        {
            "shifted_iou_radius": radius,
            "use_shifted_iou_for_iou_cost": radius > 0,
            "shifted_iou_weight": 0.0,
            "shifted_mask_cosine_weight": 0.0,
            "shifted_iou_shift_penalty_weight": shift_penalty_weight,
        }
    )
    if shift_penalty_scale is not None:
        kwargs["shifted_iou_shift_penalty_scale"] = shift_penalty_scale
    return kwargs


def roi_aware_cost_kwargs() -> dict[str, float]:
    """Return the default BayesCaTrack ROI-aware cost configuration."""

    return {}


def roi_aware_shifted_cost_kwargs(
    *,
    shifted_iou_radius: int = 2,
    shifted_iou_shift_penalty_weight: float = 0.25,
    shifted_iou_shift_penalty_scale: float | None = None,
) -> dict[str, float | int | bool]:
    """Return ROI-aware kwargs that use shifted overlap for residual registration error."""

    kwargs = registered_shifted_iou_cost_kwargs(
        shifted_iou_radius=shifted_iou_radius,
        shifted_iou_shift_penalty_weight=shifted_iou_shift_penalty_weight,
        shifted_iou_shift_penalty_scale=shifted_iou_shift_penalty_scale,
    )
    kwargs.pop("iou_weight", None)
    kwargs.pop("mask_cosine_weight", None)
    kwargs.pop("centroid_weight", None)
    kwargs.pop("area_weight", None)
    kwargs.pop("roi_feature_weight", None)
    kwargs.pop("cell_probability_weight", None)
    kwargs["use_shifted_mask_cosine_for_mask_cosine_cost"] = (
        int(kwargs["shifted_iou_radius"]) > 0
    )
    return kwargs


def session_edge_pairs(
    num_sessions: int, *, max_gap: int = 2
) -> tuple[SessionEdge, ...]:
    """Return forward session edges admitted by the max-gap skip-session policy."""

    if num_sessions < 0:
        raise ValueError("num_sessions must be non-negative")
    if max_gap < 1:
        raise ValueError("max_gap must be at least 1")
    return tuple(
        (source, target)
        for source in range(max(0, num_sessions - 1))
        for target in range(source + 1, min(num_sessions, source + max_gap + 1))
    )


# pylint: disable=too-many-arguments,too-many-locals
def build_registered_pairwise_costs(
    sessions: Sequence[Track2pSession],
    *,
    max_gap: int = 2,
    cost: AssociationCost = "registered-iou",
    calibrated_model: CalibratedAssociationModel | None = None,
    transform_type: str = "affine",
    registration_kwargs: Mapping[str, Any] | None = None,
    order: str = "xy",
    weighted_centroids: bool = False,
    velocity_variance: float = 25.0,
    regularization: float = 1.0e-6,
    pairwise_cost_kwargs: Mapping[str, Any] | None = None,
    return_pairwise_components: bool = False,
    activity_tie_breaker_weight: float = 0.0,
    activity_tie_breaker_component: str = "activity_tiebreaker_cost",
    activity_trace_source: str = "auto",
    activity_event_threshold: float = 0.0,
) -> dict[SessionEdge, np.ndarray]:
    """Build registered pairwise cost matrices for consecutive and skip-session edges."""

    sessions = list(sessions)
    if cost == "calibrated" and calibrated_model is None:
        raise ValueError("calibrated_model is required when cost='calibrated'")
    if activity_tie_breaker_weight < 0.0:
        raise ValueError("activity_tie_breaker_weight must be non-negative")

    needs_activity_components = (
        return_pairwise_components
        or cost == "calibrated"
        or activity_tie_breaker_weight > 0.0
    )

    base_cost_kwargs = _cost_kwargs_for_method(cost)
    if pairwise_cost_kwargs is not None:
        base_cost_kwargs.update(dict(pairwise_cost_kwargs))

    previous_pairwise_cost_method = None
    if pairwise_kwargs_use_shifted_overlap(base_cost_kwargs):
        previous_pairwise_cost_method = install_shifted_overlap_cost_patch()
    try:
        pairwise_costs: dict[SessionEdge, np.ndarray] = {}
        for source_session, target_session in session_edge_pairs(
            len(sessions), max_gap=max_gap
        ):
            registered_measurement_plane = register_plane_pair(
                sessions[source_session].plane_data,
                sessions[target_session].plane_data,
                transform_type=transform_type,
                registration_kwargs=registration_kwargs,
            )
            registered_measurement_plane, empty_registered_rois = (
                replace_empty_registered_masks(registered_measurement_plane)
            )
            bundle = build_session_pair_association_bundle(
                sessions[source_session],
                sessions[target_session],
                measurement_plane_in_reference_frame=registered_measurement_plane,
                order=order,
                weighted_centroids=weighted_centroids,
                velocity_variance=velocity_variance,
                regularization=regularization,
                pairwise_cost_kwargs=base_cost_kwargs,
                return_pairwise_components=needs_activity_components,
            )
            if needs_activity_components:
                add_activity_similarity_components(
                    bundle.pairwise_components,
                    sessions[source_session].plane_data,
                    registered_measurement_plane,
                    trace_source=activity_trace_source,
                    event_threshold=activity_event_threshold,
                )
            if cost == "calibrated":
                assert calibrated_model is not None
                cost_matrix = calibrated_cost_matrix_from_bundle(
                    bundle,
                    calibrated_model,
                    session_gap=target_session - source_session,
                )
            else:
                cost_matrix = np.asarray(bundle.pairwise_cost_matrix, dtype=float)
            if activity_tie_breaker_weight > 0.0:
                cost_matrix = np.asarray(
                    cost_matrix, dtype=float
                ) + activity_tie_breaker_cost_matrix(
                    bundle.pairwise_components,
                    component_name=activity_tie_breaker_component,
                    weight=activity_tie_breaker_weight,
                )
            pairwise_costs[(source_session, target_session)] = (
                _penalize_empty_registered_roi_columns(
                    cost_matrix,
                    empty_registered_rois,
                    large_cost=float(base_cost_kwargs.get("large_cost", 1.0e6)),
                )
            )
        return pairwise_costs
    finally:
        if previous_pairwise_cost_method is not None:
            CalciumPlaneData.build_pairwise_cost_matrix = (  # type: ignore[method-assign]
                previous_pairwise_cost_method
            )


# pylint: disable=too-many-arguments,too-many-locals
def solve_global_assignment_for_sessions(
    sessions: Sequence[Track2pSession],
    *,
    max_gap: int = 2,
    cost: AssociationCost = "registered-iou",
    calibrated_model: CalibratedAssociationModel | None = None,
    transform_type: str = "affine",
    registration_kwargs: Mapping[str, Any] | None = None,
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
    higher_order_consistency_config: (
        HigherOrderConsistencyConfig | Mapping[str, Any] | None
    ) = None,
) -> GlobalAssignmentRun:
    """Run PyRecEst's global path-cover assignment on registered BayesCaTrack costs."""

    sessions = list(sessions)
    pairwise_costs = build_registered_pairwise_costs(
        sessions,
        max_gap=max_gap,
        cost=cost,
        calibrated_model=calibrated_model,
        transform_type=transform_type,
        registration_kwargs=registration_kwargs,
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
    session_sizes = tuple(int(session.plane_data.n_rois) for session in sessions)
    if higher_order_consistency_config is not None:
        pairwise_costs = apply_higher_order_consistency(
            pairwise_costs,
            session_sizes=session_sizes,
            config=higher_order_consistency_config,
        )
    session_edges = session_edge_pairs(len(sessions), max_gap=max_gap)
    return solve_global_assignment_from_pairwise_costs(
        pairwise_costs,
        session_sizes=session_sizes,
        session_edges=session_edges,
        start_cost=start_cost,
        end_cost=end_cost,
        gap_penalty=gap_penalty,
        cost_threshold=cost_threshold,
    )


def solve_global_assignment_from_pairwise_costs(
    pairwise_costs: Mapping[SessionEdge, np.ndarray],
    *,
    session_sizes: Sequence[int],
    session_edges: Sequence[SessionEdge] | None = None,
    start_cost: float = 5.0,
    end_cost: float = 5.0,
    gap_penalty: float = 1.0,
    cost_threshold: float | None = 6.0,
) -> GlobalAssignmentRun:
    """Run PyRecEst's global assignment using already-built pairwise costs.

    This is useful for fold-internal solver-prior tuning: registration and
    pairwise cost construction are expensive, while start/end/gap/threshold
    sweeps only require re-running the path-cover solver on the same cost
    matrices.
    """

    costs = dict(pairwise_costs)
    sizes = tuple(int(size) for size in session_sizes)
    edges = tuple(session_edges) if session_edges is not None else tuple(sorted(costs))
    result = _load_pyrecest_multisession_solver()(
        costs,
        session_sizes=sizes,
        start_cost=float(start_cost),
        end_cost=float(end_cost),
        gap_penalty=float(gap_penalty),
        cost_threshold=None if cost_threshold is None else float(cost_threshold),
    )
    return GlobalAssignmentRun(
        result=result,
        pairwise_costs=costs,
        session_sizes=sizes,
        session_edges=edges,
    )


def tracks_to_suite2p_index_matrix(
    tracks: Sequence[Mapping[int, int]], sessions: Sequence[Track2pSession]
) -> np.ndarray:
    """Convert solver tracks in loaded-ROI coordinates to original Suite2p indices."""
    sessions = list(sessions)
    detection_matrix = np.asarray(
        _load_pyrecest_tracks_to_index_matrix()(
            list(tracks),
            session_sizes=tuple(int(session.plane_data.n_rois) for session in sessions),
            fill_value=-1,
        ),
        dtype=int,
    )

    matrix = np.empty(detection_matrix.shape, dtype=object)
    matrix[:] = None
    roi_indices_by_session = [_roi_indices_for_session(session) for session in sessions]

    for session_index, roi_indices in enumerate(roi_indices_by_session):
        detection_indices = detection_matrix[:, session_index]
        present = detection_indices >= 0
        if not np.any(present):
            continue

        present_detection_indices = detection_indices[present]
        invalid = present_detection_indices >= roi_indices.shape[0]
        if np.any(invalid):
            raise IndexError(
                f"detection index {int(present_detection_indices[invalid][0])} "
                f"out of bounds for session {session_index}"
            )
        matrix[present, session_index] = [
            int(value) for value in roi_indices[present_detection_indices]
        ]
    return matrix


def _cost_kwargs_for_method(cost: AssociationCost) -> dict[str, Any]:
    if cost == "registered-iou":
        return registered_iou_cost_kwargs()
    if cost == "registered-shifted-iou":
        return registered_shifted_iou_cost_kwargs()
    if cost == "roi-aware-shifted":
        return roi_aware_shifted_cost_kwargs()
    if cost in {"roi-aware", "calibrated"}:
        return roi_aware_cost_kwargs()
    raise ValueError(f"Unsupported association cost: {cost}")


def _penalize_empty_registered_roi_columns(
    cost_matrix: np.ndarray, empty_registered_rois: np.ndarray, *, large_cost: float
) -> np.ndarray:
    cost_matrix = np.asarray(cost_matrix, dtype=float).copy()
    if empty_registered_rois.shape != (cost_matrix.shape[1],):
        raise ValueError(
            "empty_registered_rois must have one entry for each measurement ROI"
        )
    cost_matrix[:, empty_registered_rois] = large_cost
    return cost_matrix


def _roi_indices_for_session(session: Track2pSession) -> np.ndarray:
    plane = session.plane_data
    if plane.roi_indices is not None:
        return np.asarray(plane.roi_indices, dtype=int)
    return np.arange(plane.n_rois, dtype=int)


def _load_pyrecest_tracks_to_index_matrix() -> Any:
    try:
        from pyrecest.utils import tracks_to_index_matrix
    except ImportError:
        try:
            from pyrecest.utils.multisession_assignment_score import (
                tracks_to_index_matrix,
            )
        except ImportError:
            return _local_tracks_to_index_matrix
    return tracks_to_index_matrix


def _local_tracks_to_index_matrix(
    tracks: Sequence[Mapping[int, int]],
    session_sizes: Sequence[int] | None = None,
    *,
    fill_value: int = -1,
) -> np.ndarray:
    """Small fallback matching PyRecEst's dense track matrix convention."""

    max_session = max(
        (int(session_index) for track in tracks for session_index in track),
        default=-1,
    )
    n_sessions = max(max_session + 1, len(session_sizes or ()))
    matrix: np.ndarray = np.full((len(tracks), n_sessions), fill_value, dtype=int)
    for track_index, track in enumerate(tracks):
        for session_index, detection_index in track.items():
            matrix[track_index, int(session_index)] = int(detection_index)
    return matrix


def _load_pyrecest_multisession_solver() -> Any:
    try:
        from pyrecest.utils import solve_multisession_assignment
    except ImportError:
        try:
            from pyrecest.utils.multisession_assignment import (
                solve_multisession_assignment,
            )
        except (
            ImportError
        ) as exc:  # pragma: no cover - exercised in runtime environments without PyRecEst
            raise ImportError(
                "PyRecEst with pyrecest.utils.solve_multisession_assignment is required "
                "for global-assignment benchmarks."
            ) from exc
    return solve_multisession_assignment
