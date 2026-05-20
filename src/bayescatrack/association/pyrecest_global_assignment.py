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
    add_registered_roi_validity_components,
    drop_empty_registered_masks,
    expand_registered_pairwise_cost_columns,
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
from bayescatrack.soft_overlap_costs import (
    install_soft_overlap_costs,
    registered_soft_iou_cost_kwargs as _registered_soft_overlap_cost_kwargs,
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
SOFT_OVERLAP_COST_KWARG_NAMES = frozenset(
    {
        "soft_iou_weight",
        "soft_iou_radius",
        "distance_transform_overlap_weight",
        "distance_transform_overlap_radius",
        "distance_transform_overlap_scale",
    }
)


@dataclass(frozen=True)
class TripletSupportConsistencyConfig:
    """Penalty for skip edges that lack support through intermediate sessions."""

    triplet_weight: float = 0.0
    support_top_k: int = 3
    support_cost_cap: float | None = None
    max_penalty: float | None = None


@dataclass(frozen=True)
class GlobalAssignmentRun:
    """Global assignment result plus the pairwise evidence used to build it."""

    result: Any
    pairwise_costs: dict[SessionEdge, np.ndarray]
    session_sizes: tuple[int, ...]
    session_edges: tuple[SessionEdge, ...]


def registered_iou_cost_kwargs(
    *, similarity_epsilon: float = 1.0e-6
) -> dict[str, Any]:
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


def registered_soft_iou_cost_kwargs(**kwargs: Any) -> dict[str, Any]:
    """Return registered near-miss soft-overlap cost kwargs."""

    return dict(_registered_soft_overlap_cost_kwargs(**kwargs))


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


def apply_triplet_support_consistency(
    pairwise_costs: Mapping[SessionEdge, np.ndarray],
    *,
    config: TripletSupportConsistencyConfig | None,
) -> dict[SessionEdge, np.ndarray]:
    """Penalize skip-session edges without a compatible two-hop support path.

    A direct edge ``source -> target`` is considered supported when at least one
    intermediate session ``middle`` contains a low-cost path
    ``source -> middle -> target``.  The penalty is applied only to skip edges,
    so consecutive-session costs remain unchanged.
    """

    adjusted = {
        edge: np.asarray(cost_matrix, dtype=float).copy()
        for edge, cost_matrix in pairwise_costs.items()
    }
    if config is None or config.triplet_weight <= 0.0:
        return adjusted
    if config.support_top_k < 1:
        raise ValueError("support_top_k must be at least 1")
    if config.support_cost_cap is not None and config.support_cost_cap < 0.0:
        raise ValueError("support_cost_cap must be non-negative when provided")
    if config.max_penalty is not None and config.max_penalty < 0.0:
        raise ValueError("max_penalty must be non-negative when provided")

    penalty_value = float(config.triplet_weight)
    if config.max_penalty is not None:
        penalty_value = min(penalty_value, float(config.max_penalty))
    if penalty_value <= 0.0:
        return adjusted

    support_cost_cap = (
        None if config.support_cost_cap is None else float(config.support_cost_cap)
    )
    for (source_session, target_session), direct_costs in tuple(adjusted.items()):
        if target_session - source_session < 2:
            continue
        support_costs = _best_triplet_support_costs(
            pairwise_costs,
            source_session=source_session,
            target_session=target_session,
            support_top_k=config.support_top_k,
        )
        if support_costs is None:
            continue
        if support_costs.shape != direct_costs.shape:
            raise ValueError(
                "Triplet support matrix shape mismatch for edge "
                f"{(source_session, target_session)!r}: "
                f"expected {direct_costs.shape}, got {support_costs.shape}"
            )
        unsupported = ~np.isfinite(support_costs)
        if support_cost_cap is not None:
            unsupported |= support_costs > support_cost_cap
        direct_costs[unsupported] += penalty_value

    return adjusted


def _best_triplet_support_costs(
    pairwise_costs: Mapping[SessionEdge, np.ndarray],
    *,
    source_session: int,
    target_session: int,
    support_top_k: int,
) -> np.ndarray | None:
    best_support: np.ndarray | None = None
    for middle_session in range(source_session + 1, target_session):
        left = pairwise_costs.get((source_session, middle_session))
        right = pairwise_costs.get((middle_session, target_session))
        if left is None or right is None:
            continue
        candidate_support = _best_two_step_path_costs(
            left,
            right,
            support_top_k=support_top_k,
        )
        if best_support is None:
            best_support = candidate_support
        else:
            best_support = np.minimum(best_support, candidate_support)
    return best_support


def _best_two_step_path_costs(
    source_to_middle: np.ndarray,
    middle_to_target: np.ndarray,
    *,
    support_top_k: int,
) -> np.ndarray:
    left = np.asarray(source_to_middle, dtype=float)
    right = np.asarray(middle_to_target, dtype=float)
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("Pairwise support costs must be two-dimensional matrices")
    if left.shape[1] != right.shape[0]:
        raise ValueError(
            "Incompatible two-hop support shapes: "
            f"{left.shape} cannot compose with {right.shape}"
        )
    best = np.full((left.shape[0], right.shape[1]), np.inf, dtype=float)
    for source_index, row in enumerate(left):
        middle_candidates = _top_finite_indices(row, support_top_k)
        if middle_candidates.size == 0:
            continue
        best[source_index, :] = np.min(
            row[middle_candidates, None] + right[middle_candidates, :], axis=0
        )
    return best


def _top_finite_indices(costs: np.ndarray, top_k: int) -> np.ndarray:
    finite_indices = np.flatnonzero(np.isfinite(costs))
    if finite_indices.size <= top_k:
        return finite_indices
    finite_costs = np.asarray(costs, dtype=float)[finite_indices]
    selected = np.argpartition(finite_costs, top_k - 1)[:top_k]
    return finite_indices[selected]


# pylint: disable=too-many-arguments,too-many-locals
def build_registered_pairwise_costs(
    sessions: Sequence[Track2pSession],
    *,
    max_gap: int = 2,
    cost: AssociationCost = "registered-iou",
    calibrated_model: CalibratedAssociationModel | None = None,
    transform_type: str = "affine",
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
    if _pairwise_kwargs_use_soft_overlap(base_cost_kwargs):
        install_soft_overlap_costs()

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
            )
            registered_measurement_plane, empty_registered_rois = (
                drop_empty_registered_masks(registered_measurement_plane)
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
                add_registered_roi_validity_components(
                    bundle.pairwise_components,
                    ~empty_registered_rois,
                    large_cost=float(base_cost_kwargs.get("large_cost", 1.0e6)),
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
                expand_registered_pairwise_cost_columns(
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
    if cost == "registered-soft-iou":
        return registered_soft_iou_cost_kwargs()
    if cost == "registered-shifted-iou":
        return registered_shifted_iou_cost_kwargs()
    if cost == "roi-aware-shifted":
        return roi_aware_shifted_cost_kwargs()
    if cost in {"roi-aware", "calibrated"}:
        return roi_aware_cost_kwargs()
    raise ValueError(f"Unsupported association cost: {cost}")


def _pairwise_kwargs_use_soft_overlap(pairwise_cost_kwargs: Mapping[str, Any]) -> bool:
    """Return whether pairwise kwargs need the soft-overlap cost extension."""

    return bool(SOFT_OVERLAP_COST_KWARG_NAMES.intersection(pairwise_cost_kwargs))


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
