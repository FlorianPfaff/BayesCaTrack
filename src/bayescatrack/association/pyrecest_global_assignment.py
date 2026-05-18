"""BayesCaTrack adapter for PyRecEst global multi-session assignment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from bayescatrack.association.activity_similarity import (
    add_activity_similarity_components,
)
from bayescatrack.association.calibrated_costs import (
    CalibratedAssociationModel,
    calibrated_cost_matrix_from_bundle,
)
from bayescatrack.association.registered_masks import (
    replace_empty_registered_masks,
)
from bayescatrack.core.bridge import (
    Track2pSession,
    build_session_pair_association_bundle,
)
from bayescatrack.track2p_registration import register_plane_pair

AssociationCost = Literal["registered-iou", "roi-aware", "calibrated"]
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


def roi_aware_cost_kwargs() -> dict[str, float]:
    """Return the default BayesCaTrack ROI-aware cost configuration."""

    return {}


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
    order: str = "xy",
    weighted_centroids: bool = False,
    velocity_variance: float = 25.0,
    regularization: float = 1.0e-6,
    pairwise_cost_kwargs: Mapping[str, Any] | None = None,
    return_pairwise_components: bool = False,
) -> dict[SessionEdge, np.ndarray]:
    """Build registered pairwise cost matrices for consecutive and skip-session edges."""

    sessions = list(sessions)
    if cost == "calibrated" and calibrated_model is None:
        raise ValueError("calibrated_model is required when cost='calibrated'")

    base_cost_kwargs = _cost_kwargs_for_method(cost)
    if pairwise_cost_kwargs is not None:
        base_cost_kwargs.update(dict(pairwise_cost_kwargs))

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
            return_pairwise_components=return_pairwise_components
            or cost == "calibrated",
        )
        if return_pairwise_components or cost == "calibrated":
            add_activity_similarity_components(
                bundle.pairwise_components,
                sessions[source_session].plane_data,
                registered_measurement_plane,
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
        pairwise_costs[(source_session, target_session)] = (
            _penalize_empty_registered_roi_columns(
                cost_matrix,
                empty_registered_rois,
                large_cost=float(base_cost_kwargs.get("large_cost", 1.0e6)),
            )
        )
    return pairwise_costs


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
    )
    session_sizes = tuple(int(session.plane_data.n_rois) for session in sessions)
    session_edges = session_edge_pairs(len(sessions), max_gap=max_gap)
    result = _load_pyrecest_multisession_solver()(
        pairwise_costs,
        session_sizes=session_sizes,
        start_cost=float(start_cost),
        end_cost=float(end_cost),
        gap_penalty=float(gap_penalty),
        cost_threshold=cost_threshold,
    )
    return GlobalAssignmentRun(
        result=result,
        pairwise_costs=pairwise_costs,
        session_sizes=session_sizes,
        session_edges=session_edges,
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
    matrix = np.full((len(tracks), n_sessions), fill_value, dtype=int)
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
