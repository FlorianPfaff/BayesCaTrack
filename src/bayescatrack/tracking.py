"""End-to-end registered subject tracking runner.

This module owns the workflow that is larger than a single association helper:

* load a Track2p/Suite2p subject,
* build registered pairwise association costs, including skip-session edges,
* solve the multi-session path-cover assignment globally by default,
* keep the old consecutive Hungarian stitching path as an explicit ablation, and
* report internal cost/coverage summaries.

It intentionally does not compare against a reference. Ground-truth scoring can
be done by passing ``track_rows`` to reference/benchmark code outside this runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from . import Track2pSession, load_track2p_subject
from .association.pyrecest_global_assignment import (
    AssociationCost,
    GlobalAssignmentRun,
    solve_global_assignment_for_sessions,
    tracks_to_suite2p_index_matrix,
)
from .matching import (
    DEFAULT_ASSIGNMENT_MAX_COST,
    SessionMatchResult,
    build_track_rows_from_bundles,
)
from .registration import (
    RegisteredConsecutiveBundles,
    RegistrationModel,
    build_registered_consecutive_session_association_bundles,
)


TrackingMethod = Literal["global", "pairwise"]


@dataclass(frozen=True)
class SubjectTrackingResult:
    """Predicted longitudinal tracks plus internal tracking summaries."""

    sessions: tuple[Track2pSession, ...]
    registered_bundles: RegisteredConsecutiveBundles
    match_results: tuple[SessionMatchResult, ...]
    session_names: tuple[str, ...]
    track_rows: np.ndarray
    link_costs: np.ndarray
    fill_value: int = -1
    tracking_method: TrackingMethod = "global"
    global_assignment: GlobalAssignmentRun | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "sessions", tuple(self.sessions))
        object.__setattr__(self, "match_results", tuple(self.match_results))
        object.__setattr__(
            self, "tracking_method", _validate_tracking_method(self.tracking_method)
        )
        object.__setattr__(
            self, "session_names", tuple(str(name) for name in self.session_names)
        )

        track_rows = np.asarray(self.track_rows, dtype=int)
        if track_rows.ndim != 2:
            raise ValueError("track_rows must be two-dimensional")
        if track_rows.shape[1] != len(self.session_names):
            raise ValueError("track_rows must have one column per session name")
        object.__setattr__(self, "track_rows", track_rows)

        link_costs = np.asarray(self.link_costs, dtype=float)
        expected_link_shape = (track_rows.shape[0], max(track_rows.shape[1] - 1, 0))
        if link_costs.shape != expected_link_shape:
            raise ValueError(f"link_costs must have shape {expected_link_shape}")
        object.__setattr__(self, "link_costs", link_costs)

    @property
    def n_tracks(self) -> int:
        return int(self.track_rows.shape[0])

    @property
    def solver(self) -> str:
        return "global-assignment" if self.tracking_method == "global" else "pairwise"

    @property
    def n_sessions(self) -> int:
        return int(self.track_rows.shape[1])

    def track_lengths(self) -> np.ndarray:
        """Return the number of non-missing session entries per track."""

        return np.sum(self.track_rows != self.fill_value, axis=1).astype(int)

    def complete_track_mask(self) -> np.ndarray:
        """Return a boolean mask for tracks present in every session."""

        return self.track_lengths() == self.n_sessions

    def score_summary(self) -> dict[str, Any]:
        """Return internal tracking coverage and cost summaries.

        These scores do not require ground truth. Costs are assignment costs from
        the ROI-aware pairwise cost matrices, so lower link-cost values indicate
        stronger associations under the configured cost model.
        """

        track_lengths = self.track_lengths()
        complete_mask = self.complete_track_mask()
        finite_link_costs = self.link_costs[np.isfinite(self.link_costs)]
        pair_summaries = [
            _match_result_summary(
                match_result,
                n_reference_rois=self.sessions[pair_index].plane_data.n_rois,
                n_measurement_rois=self.sessions[pair_index + 1].plane_data.n_rois,
            )
            for pair_index, match_result in enumerate(self.match_results)
        ]
        n_pairwise_matches = int(
            sum(summary["n_matches"] for summary in pair_summaries)
        )
        global_session_edges: tuple[tuple[int, int], ...] = ()
        if self.global_assignment is not None:
            global_session_edges = tuple(self.global_assignment.session_edges)

        return {
            "tracking_method": self.tracking_method,
            "solver": self.solver,
            "n_sessions": self.n_sessions,
            "session_names": self.session_names,
            "global_session_edges": global_session_edges,
            "n_tracks_started": self.n_tracks,
            "n_complete_tracks": int(np.sum(complete_mask)),
            "complete_track_fraction": _safe_ratio(
                float(np.sum(complete_mask)), float(self.n_tracks)
            ),
            "mean_track_length": _mean_or_nan(track_lengths),
            "median_track_length": _median_or_nan(track_lengths),
            "max_track_length": int(np.max(track_lengths)) if track_lengths.size else 0,
            "n_pairwise_matches": int(n_pairwise_matches),
            "mean_link_cost": _mean_or_nan(finite_link_costs),
            "median_link_cost": _median_or_nan(finite_link_costs),
            "max_link_cost": _max_or_nan(finite_link_costs),
            "total_link_cost": (
                float(np.sum(finite_link_costs)) if finite_link_costs.size else 0.0
            ),
            "pairs": pair_summaries,
        }

    def to_export_dict(self) -> dict[str, Any]:
        """Return arrays and summary metadata suitable for serialization."""

        return {
            "session_names": np.asarray(self.session_names, dtype=object),
            "track_rows": self.track_rows,
            "link_costs": self.link_costs,
            "track_lengths": self.track_lengths(),
            "complete_track_mask": self.complete_track_mask(),
            "fill_value": np.asarray(self.fill_value, dtype=int),
            "tracking_method": np.asarray(self.tracking_method, dtype=object),
            "solver": np.asarray(self.solver, dtype=object),
            "scores": self.score_summary(),
        }


# pylint: disable=too-many-arguments,too-many-locals
def run_registered_subject_tracking(
    subject_dir: str | Path,
    *,
    plane_name: str = "plane0",
    input_format: str = "auto",
    include_behavior: bool = True,
    order: str = "xy",
    weighted_centroids: bool = False,
    # jscpd:ignore-start
    velocity_variance: float = 25.0,
    regularization: float = 1e-6,
    registration_model: RegistrationModel = "affine",
    registration_max_cost: float | None = None,
    registration_max_iterations: int = 25,
    registration_tolerance: float = 1e-8,
    min_matches: int | None = None,
    allow_reflection: bool = False,
    pairwise_cost_kwargs: Mapping[str, Any] | None = None,
    return_pairwise_components: bool = True,
    binarize_registered_masks: bool = False,
    registered_mask_threshold: float = 0.5,
    # jscpd:ignore-end
    assignment_max_cost: float | None = DEFAULT_ASSIGNMENT_MAX_COST,
    tracking_method: TrackingMethod = "global",
    association_cost: AssociationCost = "roi-aware",
    max_gap: int = 2,
    start_cost: float = 5.0,
    end_cost: float = 5.0,
    gap_penalty: float = 1.0,
    start_roi_indices: Sequence[int] | None = None,
    start_session_index: int = 0,
    fill_value: int = -1,
    **suite2p_kwargs: Any,
) -> SubjectTrackingResult:
    """Run registered ROI-aware tracking for one Track2p-style subject.

    The returned ``track_rows`` matrix has one row per seed ROI in
    ``start_session_index`` and one column per session. Entries are Suite2p ROI
    indices. Missing links are filled with ``fill_value``. By default, the runner
    uses PyRecEst's global multi-session path-cover solver over registered
    consecutive and skip-session edges. Set ``tracking_method="pairwise"`` to
    recover the old consecutive Hungarian-assignment stitching behavior.

    ``assignment_max_cost`` is used as the pairwise Hungarian gate in pairwise
    mode and as the global solver's edge-cost threshold in global mode. Pass
    ``None`` explicitly to disable that threshold/gate.
    """

    tracking_method = _validate_tracking_method(tracking_method)
    sessions = _load_subject_sessions(
        subject_dir,
        plane_name,
        input_format,
        include_behavior,
        suite2p_kwargs,
    )
    if not sessions:
        raise ValueError("No sessions were found")

    start_session_index = int(start_session_index)
    if start_session_index < 0 or start_session_index >= len(sessions):
        raise IndexError(
            f"start_session_index {start_session_index} out of bounds "
            f"for {len(sessions)} sessions"
        )

    session_names = tuple(session.session_name for session in sessions)
    if len(sessions) == 1:
        first_plane = sessions[0].plane_data
        if start_roi_indices is None:
            first_indices = (
                np.asarray(first_plane.roi_indices, dtype=int)
                if first_plane.roi_indices is not None
                else np.arange(first_plane.n_rois, dtype=int)
            )
        else:
            first_indices = np.asarray(start_roi_indices, dtype=int)
        track_rows = np.asarray(first_indices, dtype=int).reshape(-1, 1)
        link_costs = np.zeros((track_rows.shape[0], 0), dtype=float)
        return SubjectTrackingResult(
            sessions=sessions,
            registered_bundles=RegisteredConsecutiveBundles(bundles=[]),
            match_results=(),
            session_names=session_names,
            track_rows=track_rows,
            link_costs=link_costs,
            fill_value=fill_value,
            tracking_method=tracking_method,
        )

    if tracking_method == "global":
        global_assignment = solve_global_assignment_for_sessions(
            sessions,
            max_gap=max_gap,
            cost=association_cost,
            transform_type=_global_transform_type_from_registration_model(
                registration_model
            ),
            start_cost=start_cost,
            end_cost=end_cost,
            gap_penalty=gap_penalty,
            cost_threshold=assignment_max_cost,
            order=order,
            weighted_centroids=weighted_centroids,
            velocity_variance=velocity_variance,
            regularization=regularization,
            pairwise_cost_kwargs=pairwise_cost_kwargs,
        )
        track_rows = _coerce_global_track_rows(
            tracks_to_suite2p_index_matrix(global_assignment.result.tracks, sessions),
            fill_value=fill_value,
        )
        if start_roi_indices is None:
            start_roi_indices = _roi_indices_for_session(sessions[start_session_index])
        track_rows = _restrict_track_rows_to_start_rois(
            track_rows,
            start_roi_indices=start_roi_indices,
            start_session_index=start_session_index,
            fill_value=fill_value,
        )
        match_results = _consecutive_match_results_from_global_assignment(
            global_assignment,
            sessions,
            track_rows,
            fill_value=fill_value,
        )
        link_costs = _build_link_cost_matrix(
            track_rows,
            match_results,
            fill_value=fill_value,
        )
        return SubjectTrackingResult(
            sessions=sessions,
            registered_bundles=RegisteredConsecutiveBundles(bundles=[]),
            match_results=match_results,
            session_names=session_names,
            track_rows=track_rows,
            link_costs=link_costs,
            fill_value=fill_value,
            tracking_method="global",
            global_assignment=global_assignment,
        )

    registered_bundles = build_registered_consecutive_session_association_bundles(
        sessions,
        order=order,
        weighted_centroids=weighted_centroids,
        velocity_variance=velocity_variance,
        regularization=regularization,
        registration_model=registration_model,
        registration_max_cost=registration_max_cost,
        registration_max_iterations=registration_max_iterations,
        registration_tolerance=registration_tolerance,
        min_matches=min_matches,
        allow_reflection=allow_reflection,
        pairwise_cost_kwargs=pairwise_cost_kwargs,
        return_pairwise_components=return_pairwise_components,
        binarize_registered_masks=binarize_registered_masks,
        registered_mask_threshold=registered_mask_threshold,
    )
    association_bundles = [
        bundle.association_bundle for bundle in registered_bundles.bundles
    ]
    solved_session_names, track_rows, match_results = build_track_rows_from_bundles(
        association_bundles,
        max_cost=assignment_max_cost,
        start_roi_indices=start_roi_indices,
        start_session_index=start_session_index,
        fill_value=fill_value,
    )
    link_costs = _build_link_cost_matrix(
        track_rows,
        match_results,
        fill_value=fill_value,
    )

    return SubjectTrackingResult(
        sessions=sessions,
        registered_bundles=registered_bundles,
        match_results=tuple(match_results),
        session_names=solved_session_names,
        track_rows=track_rows,
        link_costs=link_costs,
        fill_value=fill_value,
        tracking_method="pairwise",
    )


def _validate_tracking_method(value: str) -> TrackingMethod:
    if value not in {"global", "pairwise"}:
        raise ValueError("tracking_method must be either 'global' or 'pairwise'")
    return value  # type: ignore[return-value]


def _global_transform_type_from_registration_model(
    registration_model: RegistrationModel,
) -> str:
    if registration_model == "translation":
        return "fov-translation"
    return str(registration_model)


def _coerce_global_track_rows(track_rows: np.ndarray, *, fill_value: int) -> np.ndarray:
    track_rows = np.asarray(track_rows, dtype=object)
    if track_rows.ndim != 2:
        raise ValueError("global assignment track matrix must be two-dimensional")

    coerced = np.full(track_rows.shape, int(fill_value), dtype=int)
    for index, value in np.ndenumerate(track_rows):
        if value is None:
            continue
        int_value = int(value)
        if int_value < 0:
            continue
        coerced[index] = int_value
    return coerced


def _restrict_track_rows_to_start_rois(
    track_rows: np.ndarray,
    *,
    start_roi_indices: Sequence[int],
    start_session_index: int,
    fill_value: int,
) -> np.ndarray:
    track_rows = np.asarray(track_rows, dtype=int)
    start_roi_indices = np.asarray(start_roi_indices, dtype=int).reshape(-1)
    restricted = np.full(
        (start_roi_indices.shape[0], track_rows.shape[1]),
        int(fill_value),
        dtype=int,
    )

    row_by_start_roi: dict[int, np.ndarray] = {}
    for row in track_rows:
        start_roi = int(row[start_session_index])
        if start_roi == fill_value:
            continue
        row_by_start_roi.setdefault(start_roi, row.copy())

    for row_index, start_roi in enumerate(start_roi_indices):
        start_roi = int(start_roi)
        matched_row = row_by_start_roi.get(start_roi)
        if matched_row is None:
            restricted[row_index, start_session_index] = start_roi
        else:
            restricted[row_index] = matched_row
    return restricted


def _consecutive_match_results_from_global_assignment(
    global_assignment: GlobalAssignmentRun,
    sessions: Sequence[Track2pSession],
    track_rows: np.ndarray,
    *,
    fill_value: int,
) -> tuple[SessionMatchResult, ...]:
    sessions = tuple(sessions)
    roi_position_maps = [_roi_position_map_for_session(session) for session in sessions]
    track_rows = np.asarray(track_rows, dtype=int)
    match_results: list[SessionMatchResult] = []

    for pair_index in range(max(len(sessions) - 1, 0)):
        cost_matrix = global_assignment.pairwise_costs.get(
            (pair_index, pair_index + 1)
        )
        if cost_matrix is None:
            match_results.append(_empty_consecutive_match_result(sessions, pair_index))
            continue

        cost_matrix = np.asarray(cost_matrix, dtype=float)
        reference_positions: list[int] = []
        measurement_positions: list[int] = []
        reference_roi_indices: list[int] = []
        measurement_roi_indices: list[int] = []
        costs: list[float] = []
        seen_pairs: set[tuple[int, int]] = set()
        reference_position_by_roi = roi_position_maps[pair_index]
        measurement_position_by_roi = roi_position_maps[pair_index + 1]

        for row in track_rows:
            reference_roi = int(row[pair_index])
            measurement_roi = int(row[pair_index + 1])
            if fill_value in (reference_roi, measurement_roi):
                continue
            pair = (reference_roi, measurement_roi)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            if (
                reference_roi not in reference_position_by_roi
                or measurement_roi not in measurement_position_by_roi
            ):
                continue
            reference_position = int(reference_position_by_roi[reference_roi])
            measurement_position = int(measurement_position_by_roi[measurement_roi])
            reference_positions.append(reference_position)
            measurement_positions.append(measurement_position)
            reference_roi_indices.append(reference_roi)
            measurement_roi_indices.append(measurement_roi)
            costs.append(float(cost_matrix[reference_position, measurement_position]))

        match_results.append(
            SessionMatchResult(
                reference_session_name=str(sessions[pair_index].session_name),
                measurement_session_name=str(sessions[pair_index + 1].session_name),
                reference_positions=np.asarray(reference_positions, dtype=int),
                measurement_positions=np.asarray(measurement_positions, dtype=int),
                reference_roi_indices=np.asarray(reference_roi_indices, dtype=int),
                measurement_roi_indices=np.asarray(measurement_roi_indices, dtype=int),
                costs=np.asarray(costs, dtype=float),
            )
        )
    return tuple(match_results)


def _empty_consecutive_match_result(
    sessions: Sequence[Track2pSession], pair_index: int
) -> SessionMatchResult:
    return SessionMatchResult(
        reference_session_name=str(sessions[pair_index].session_name),
        measurement_session_name=str(sessions[pair_index + 1].session_name),
        reference_positions=np.asarray([], dtype=int),
        measurement_positions=np.asarray([], dtype=int),
        reference_roi_indices=np.asarray([], dtype=int),
        measurement_roi_indices=np.asarray([], dtype=int),
        costs=np.asarray([], dtype=float),
    )


def _roi_indices_for_session(session: Track2pSession) -> np.ndarray:
    plane = session.plane_data
    if plane.roi_indices is not None:
        return np.asarray(plane.roi_indices, dtype=int)
    return np.arange(plane.n_rois, dtype=int)


def _roi_position_map_for_session(session: Track2pSession) -> dict[int, int]:
    return {
        int(roi_index): int(position)
        for position, roi_index in enumerate(_roi_indices_for_session(session))
    }


def _load_subject_sessions(
    subject_dir: str | Path,
    plane_name: str,
    input_format: str,
    include_behavior: bool,
    suite2p_kwargs: Mapping[str, Any],
) -> tuple[Track2pSession, ...]:
    load_kwargs = dict(suite2p_kwargs)
    load_kwargs.update(
        plane_name=plane_name,
        input_format=input_format,
        include_behavior=include_behavior,
    )
    return tuple(load_track2p_subject(subject_dir, **load_kwargs))


def _build_link_cost_matrix(
    track_rows: np.ndarray,
    match_results: Sequence[SessionMatchResult],
    *,
    fill_value: int,
) -> np.ndarray:
    track_rows = np.asarray(track_rows, dtype=int)
    link_costs = np.full((track_rows.shape[0], max(track_rows.shape[1] - 1, 0)), np.nan)
    for pair_index, match_result in enumerate(match_results):
        cost_lookup = {
            (int(reference_roi), int(measurement_roi)): float(cost)
            for reference_roi, measurement_roi, cost in zip(
                match_result.reference_roi_indices,
                match_result.measurement_roi_indices,
                match_result.costs,
                strict=True,
            )
        }
        for track_index, row in enumerate(track_rows):
            reference_roi = int(row[pair_index])
            measurement_roi = int(row[pair_index + 1])
            if fill_value in (reference_roi, measurement_roi):
                continue
            link_costs[track_index, pair_index] = cost_lookup.get(
                (reference_roi, measurement_roi),
                np.nan,
            )
    return link_costs


def _match_result_summary(
    match_result: SessionMatchResult,
    *,
    n_reference_rois: int,
    n_measurement_rois: int,
) -> dict[str, Any]:
    costs = np.asarray(match_result.costs, dtype=float)
    n_reference_rois = int(n_reference_rois)
    n_measurement_rois = int(n_measurement_rois)
    return {
        "reference_session_name": match_result.reference_session_name,
        "measurement_session_name": match_result.measurement_session_name,
        "n_reference_rois": n_reference_rois,
        "n_measurement_rois": n_measurement_rois,
        "n_matches": match_result.n_matches,
        "reference_match_fraction": _safe_ratio(
            match_result.n_matches, n_reference_rois
        ),
        "measurement_match_fraction": _safe_ratio(
            match_result.n_matches, n_measurement_rois
        ),
        "mean_cost": _mean_or_nan(costs),
        "median_cost": _median_or_nan(costs),
        "max_cost": _max_or_nan(costs),
        "total_cost": float(np.sum(costs)) if costs.size else 0.0,
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 1.0
    return float(numerator) / float(denominator)


def _mean_or_nan(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.mean(values)) if values.size else float("nan")


def _median_or_nan(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.median(values)) if values.size else float("nan")


def _max_or_nan(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.max(values)) if values.size else float("nan")
