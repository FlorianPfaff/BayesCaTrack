"""BayesCaTrack adapter for PyRecEst global multi-session assignment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from bayescatrack.core.bridge import Track2pSession, build_session_pair_association_bundle
from bayescatrack.track2p_registration import register_plane_pair

AssociationCost = Literal["registered-iou", "roi-aware"]


@dataclass(frozen=True)
class GlobalAssignmentRun:
    """Global assignment result plus the pairwise evidence used to build it."""

    result: Any
    pairwise_costs: dict[tuple[int, int], np.ndarray]
    session_sizes: tuple[int, ...]


def registered_iou_cost_kwargs(*, similarity_epsilon: float = 1.0e-6) -> dict[str, float]:
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


# pylint: disable=too-many-arguments
def build_registered_pairwise_costs(
    sessions: Sequence[Track2pSession],
    *,
    max_gap: int = 2,
    cost: AssociationCost = "registered-iou",
    transform_type: str = "affine",
    order: str = "xy",
    weighted_centroids: bool = False,
    velocity_variance: float = 25.0,
    regularization: float = 1.0e-6,
    pairwise_cost_kwargs: Mapping[str, Any] | None = None,
    return_pairwise_components: bool = False,
) -> dict[tuple[int, int], np.ndarray]:
    """Build registered pairwise cost matrices for consecutive and skip-session edges."""

    sessions = list(sessions)
    if max_gap < 1:
        raise ValueError("max_gap must be at least 1")

    base_cost_kwargs = _cost_kwargs_for_method(cost)
    if pairwise_cost_kwargs is not None:
        base_cost_kwargs.update(dict(pairwise_cost_kwargs))

    pairwise_costs: dict[tuple[int, int], np.ndarray] = {}
    for source_session in range(len(sessions) - 1):
        last_target = min(len(sessions), source_session + max_gap + 1)
        for target_session in range(source_session + 1, last_target):
            registered_measurement_plane = register_plane_pair(
                sessions[source_session].plane_data,
                sessions[target_session].plane_data,
                transform_type=transform_type,
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
                return_pairwise_components=return_pairwise_components,
            )
            pairwise_costs[(source_session, target_session)] = np.asarray(bundle.pairwise_cost_matrix, dtype=float)
    return pairwise_costs


# pylint: disable=too-many-arguments
def solve_global_assignment_for_sessions(
    sessions: Sequence[Track2pSession],
    *,
    max_gap: int = 2,
    cost: AssociationCost = "registered-iou",
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
        transform_type=transform_type,
        order=order,
        weighted_centroids=weighted_centroids,
        velocity_variance=velocity_variance,
        regularization=regularization,
        pairwise_cost_kwargs=pairwise_cost_kwargs,
    )
    session_sizes = tuple(int(session.plane_data.n_rois) for session in sessions)
    solver = _load_pyrecest_multisession_solver()
    result = solver(
        pairwise_costs,
        session_sizes=session_sizes,
        start_cost=float(start_cost),
        end_cost=float(end_cost),
        gap_penalty=float(gap_penalty),
        cost_threshold=cost_threshold,
    )
    return GlobalAssignmentRun(result=result, pairwise_costs=pairwise_costs, session_sizes=session_sizes)


def tracks_to_suite2p_index_matrix(tracks: Sequence[Mapping[int, int]], sessions: Sequence[Track2pSession]) -> np.ndarray:
    """Convert solver tracks in loaded-ROI coordinates to original Suite2p indices."""

    sessions = list(sessions)
    matrix = np.empty((len(tracks), len(sessions)), dtype=object)
    matrix[:] = None
    roi_indices_by_session = [_roi_indices_for_session(session) for session in sessions]

    for track_index, track in enumerate(tracks):
        for session_index, detection_index in track.items():
            session_index = int(session_index)
            detection_index = int(detection_index)
            if session_index < 0 or session_index >= len(sessions):
                raise IndexError(f"session index {session_index} out of bounds")
            roi_indices = roi_indices_by_session[session_index]
            if detection_index < 0 or detection_index >= roi_indices.shape[0]:
                raise IndexError(f"detection index {detection_index} out of bounds for session {session_index}")
            matrix[track_index, session_index] = int(roi_indices[detection_index])
    return matrix


def _cost_kwargs_for_method(cost: AssociationCost) -> dict[str, Any]:
    if cost == "registered-iou":
        return registered_iou_cost_kwargs()
    if cost == "roi-aware":
        return roi_aware_cost_kwargs()
    raise ValueError(f"Unsupported association cost: {cost}")


def _roi_indices_for_session(session: Track2pSession) -> np.ndarray:
    plane = session.plane_data
    if plane.roi_indices is not None:
        return np.asarray(plane.roi_indices, dtype=int)
    return np.arange(plane.n_rois, dtype=int)


def _load_pyrecest_multisession_solver() -> Any:
    try:
        from pyrecest.utils.multisession_assignment import solve_multisession_assignment
    except ImportError as exc:  # pragma: no cover - exercised in runtime environments without PyRecEst
        raise ImportError(
            "PyRecEst with pyrecest.utils.multisession_assignment is required for global-assignment ablations."
        ) from exc
    return solve_multisession_assignment
