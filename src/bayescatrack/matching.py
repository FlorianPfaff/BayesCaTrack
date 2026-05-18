"""Utilities for solving Track2p/PyRecEst association bundles.

This module complements the ROI-aware bundle construction in
``track2p_pyrecest_bridge`` by providing the missing glue needed for benchmark
workflows:

* solve one bundle's pairwise cost matrix into ROI matches,
* solve a sequence of consecutive bundles,
* stitch those matches into wide track rows, and
* export the result as a simple CSV file.

The CSV format is intentionally minimal and compatible with downstream
benchmark/evaluation code that expects one row per reconstructed track and one
column per session.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:  # pragma: no cover - exercised in real runtime/CI only
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - defensive fallback only
    linear_sum_assignment = None


DEFAULT_ASSIGNMENT_MAX_COST = 6.0


@dataclass(frozen=True)
class SessionMatchResult:
    """Linear-assignment solution for one consecutive session pair."""

    reference_session_name: str
    measurement_session_name: str
    reference_positions: np.ndarray
    measurement_positions: np.ndarray
    reference_roi_indices: np.ndarray
    measurement_roi_indices: np.ndarray
    costs: np.ndarray

    def __post_init__(self) -> None:
        for field_name in (
            "reference_positions",
            "measurement_positions",
            "reference_roi_indices",
            "measurement_roi_indices",
            "costs",
        ):
            value = np.asarray(getattr(self, field_name))
            if value.ndim != 1:
                raise ValueError(f"{field_name} must be one-dimensional")
            object.__setattr__(self, field_name, value)

        n_matches = self.reference_positions.shape[0]
        if any(
            np.asarray(getattr(self, field_name)).shape[0] != n_matches
            for field_name in (
                "measurement_positions",
                "reference_roi_indices",
                "measurement_roi_indices",
                "costs",
            )
        ):
            raise ValueError("all SessionMatchResult arrays must have equal length")

    @property
    def n_matches(self) -> int:
        return int(self.reference_positions.shape[0])

    def as_roi_index_mapping(self) -> dict[int, int]:
        """Return matches as ``reference_roi_index -> measurement_roi_index``."""

        return {
            int(reference_roi): int(measurement_roi)
            for reference_roi, measurement_roi in zip(
                self.reference_roi_indices,
                self.measurement_roi_indices,
                strict=True,
            )
        }

    def as_pair_array(self) -> np.ndarray:
        """Return matches as a ``(n_matches, 2)`` integer array."""

        if self.n_matches == 0:
            return np.zeros((0, 2), dtype=int)
        return np.column_stack(
            (self.reference_roi_indices, self.measurement_roi_indices)
        ).astype(int)


# pylint: disable=too-many-locals


def solve_bundle_linear_assignment(
    bundle: Any,
    *,
    max_cost: float | None = DEFAULT_ASSIGNMENT_MAX_COST,
) -> SessionMatchResult:
    """Solve a :class:`SessionAssociationBundle` via linear assignment.

    Parameters
    ----------
    bundle
        Any object exposing the attributes used by
        :class:`track2p_pyrecest_bridge.SessionAssociationBundle`.
    max_cost
        Assignment gate. Candidate pairs with assignment cost larger than this
        threshold are excluded from the linear-assignment objective and discarded
        from the returned matches. Pass ``None`` to keep every finite assignment.
    """

    if linear_sum_assignment is None:
        raise ImportError(
            "solve_bundle_linear_assignment requires scipy.optimize.linear_sum_assignment"
        )

    cost_matrix = np.asarray(bundle.pairwise_cost_matrix, dtype=float)
    if cost_matrix.ndim != 2:
        raise ValueError("bundle.pairwise_cost_matrix must be two-dimensional")
    if max_cost is not None:
        max_cost = float(max_cost)
        if not np.isfinite(max_cost) or max_cost < 0.0:
            raise ValueError("max_cost must be a finite non-negative value")

    if cost_matrix.shape[0] == 0 or cost_matrix.shape[1] == 0:
        return _empty_match_result(bundle)

    assignment_cost_matrix = cost_matrix
    valid_assignment_mask: np.ndarray | None = None
    if max_cost is not None:
        valid_assignment_mask = np.isfinite(cost_matrix) & (cost_matrix <= max_cost)
        if not np.any(valid_assignment_mask):
            return _empty_match_result(bundle)
        assignment_cost_matrix = _gate_cost_matrix_for_linear_assignment(
            cost_matrix,
            valid_assignment_mask,
            max_cost=max_cost,
        )

    reference_positions, measurement_positions = linear_sum_assignment(
        assignment_cost_matrix
    )
    assignment_costs = cost_matrix[reference_positions, measurement_positions]
    if valid_assignment_mask is not None:
        keep = valid_assignment_mask[reference_positions, measurement_positions]
        reference_positions = reference_positions[keep]
        measurement_positions = measurement_positions[keep]
        assignment_costs = assignment_costs[keep]

    reference_positions = np.asarray(reference_positions, dtype=int)
    measurement_positions = np.asarray(measurement_positions, dtype=int)
    assignment_costs = np.asarray(assignment_costs, dtype=float)
    reference_roi_indices = np.asarray(bundle.reference_roi_indices, dtype=int)[
        reference_positions
    ]
    measurement_roi_indices = np.asarray(bundle.measurement_roi_indices, dtype=int)[
        measurement_positions
    ]

    return SessionMatchResult(
        reference_session_name=str(bundle.reference_session_name),
        measurement_session_name=str(bundle.measurement_session_name),
        reference_positions=reference_positions,
        measurement_positions=measurement_positions,
        reference_roi_indices=reference_roi_indices,
        measurement_roi_indices=measurement_roi_indices,
        costs=assignment_costs,
    )


def solve_consecutive_bundle_linear_assignments(
    bundles: Sequence[Any],
    *,
    max_cost: float | None = DEFAULT_ASSIGNMENT_MAX_COST,
) -> list[SessionMatchResult]:
    """Solve a sequence of consecutive bundles into ROI-index matches."""

    return [
        solve_bundle_linear_assignment(bundle, max_cost=max_cost) for bundle in bundles
    ]


def build_track_rows_from_matches(
    session_names: Sequence[str],
    matches: Sequence[
        SessionMatchResult
        | Mapping[int, int]
        | np.ndarray
        | tuple[Sequence[int], Sequence[int]]
    ],
    *,
    start_roi_indices: Sequence[int] | None = None,
    start_session_index: int = 0,
    fill_value: int = -1,
) -> np.ndarray:
    """Stitch consecutive matches into wide track rows.

    Parameters
    ----------
    session_names
        Ordered session names, one per session.
    matches
        One match representation per consecutive session pair.
    start_roi_indices
        ROI indices from ``start_session_index`` from which tracks should be grown.
        If omitted, indices are inferred from the adjacent match mapping where
        possible.
    start_session_index
        Session column that contains ``start_roi_indices``.
    fill_value
        Integer used for missing/unmatched entries.
    """

    session_names = tuple(str(name) for name in session_names)
    if len(session_names) == 0:
        raise ValueError("session_names must not be empty")
    if len(matches) != max(len(session_names) - 1, 0):
        raise ValueError("matches must have length len(session_names) - 1")

    start_session_index = int(start_session_index)
    if start_session_index < 0 or start_session_index >= len(session_names):
        raise IndexError(
            f"start_session_index {start_session_index} out of bounds "
            f"for {len(session_names)} sessions"
        )

    normalized_matches = [_normalize_match_mapping(match) for match in matches]

    if start_roi_indices is None:
        start_roi_indices = _default_start_roi_indices(
            normalized_matches,
            start_session_index=start_session_index,
            n_sessions=len(session_names),
        )
    else:
        start_roi_indices = [int(index) for index in start_roi_indices]

    reverse_matches = [
        _invert_match_mapping(normalized_matches[match_index])
        for match_index in range(start_session_index)
    ]

    track_rows = np.full(
        (len(start_roi_indices), len(session_names)), int(fill_value), dtype=int
    )
    if len(start_roi_indices) == 0:
        return track_rows

    track_rows[:, start_session_index] = np.asarray(start_roi_indices, dtype=int)
    for row_index, start_roi in enumerate(start_roi_indices):
        current_roi = int(start_roi)
        for match_index in range(start_session_index, len(normalized_matches)):
            mapping = normalized_matches[match_index]
            next_roi = int(mapping.get(current_roi, fill_value))
            track_rows[row_index, match_index + 1] = next_roi
            if next_roi == fill_value:
                break
            current_roi = next_roi

        current_roi = int(start_roi)
        for match_index in range(start_session_index - 1, -1, -1):
            previous_roi = int(reverse_matches[match_index].get(current_roi, fill_value))
            track_rows[row_index, match_index] = previous_roi
            if previous_roi == fill_value:
                break
            current_roi = previous_roi
    return track_rows


def build_track_rows_from_bundles(
    bundles: Sequence[Any],
    *,
    max_cost: float | None = DEFAULT_ASSIGNMENT_MAX_COST,
    start_roi_indices: Sequence[int] | None = None,
    start_session_index: int = 0,
    fill_value: int = -1,
) -> tuple[tuple[str, ...], np.ndarray, list[SessionMatchResult]]:
    """Solve consecutive bundles and stitch them into wide track rows.

    ``max_cost`` is forwarded to :func:`solve_bundle_linear_assignment`.
    Omitting it keeps the default assignment gate; passing ``None`` explicitly
    disables cost gating and keeps every finite Hungarian assignment.
    """

    bundles = list(bundles)
    if not bundles:
        raise ValueError("bundles must not be empty")

    match_results = solve_consecutive_bundle_linear_assignments(
        bundles,
        max_cost=max_cost,
    )
    session_names = _session_names_from_bundles(bundles)
    if start_roi_indices is None:
        start_roi_indices = _bundle_roi_indices_for_session(bundles, start_session_index)
    track_rows = build_track_rows_from_matches(
        session_names,
        match_results,
        start_roi_indices=start_roi_indices,
        start_session_index=start_session_index,
        fill_value=fill_value,
    )
    return session_names, track_rows, match_results


def export_track_rows_csv(
    output_path: str | Path,
    session_names: Sequence[str],
    track_rows: np.ndarray,
    *,
    include_track_id: bool = True,
) -> Path:
    """Export wide track rows as a CSV file."""

    output_path = Path(output_path)
    session_names = [str(name) for name in session_names]
    track_rows = np.asarray(track_rows, dtype=int)
    if track_rows.ndim != 2:
        raise ValueError("track_rows must be two-dimensional")
    if track_rows.shape[1] != len(session_names):
        raise ValueError(
            "track_rows second dimension must equal the number of session names"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        header = (["track_id"] if include_track_id else []) + session_names
        writer.writerow(header)
        for track_index, row in enumerate(track_rows):
            values = [int(value) for value in row]
            if include_track_id:
                writer.writerow([track_index, *values])
            else:
                writer.writerow(values)
    return output_path


def _session_names_from_bundles(bundles: Sequence[Any]) -> tuple[str, ...]:
    if not bundles:
        raise ValueError("bundles must not be empty")

    session_names = [str(bundles[0].reference_session_name)]
    for bundle in bundles:
        reference_session_name = str(bundle.reference_session_name)
        measurement_session_name = str(bundle.measurement_session_name)
        if session_names[-1] != reference_session_name:
            raise ValueError("bundles must refer to consecutive sessions in order")
        session_names.append(measurement_session_name)
    return tuple(session_names)


def _bundle_roi_indices_for_session(
    bundles: Sequence[Any], session_index: int
) -> np.ndarray:
    """Return ROI indices for one session from consecutive association bundles."""

    n_sessions = len(bundles) + 1
    session_index = int(session_index)
    if session_index < 0 or session_index >= n_sessions:
        raise IndexError(
            f"start_session_index {session_index} out of bounds for {n_sessions} sessions"
        )
    if session_index == 0:
        return np.asarray(bundles[0].reference_roi_indices, dtype=int)
    if session_index == n_sessions - 1:
        return np.asarray(bundles[-1].measurement_roi_indices, dtype=int)
    return np.asarray(bundles[session_index].reference_roi_indices, dtype=int)


def _default_start_roi_indices(
    normalized_matches: Sequence[Mapping[int, int]],
    *,
    start_session_index: int,
    n_sessions: int,
) -> list[int]:
    if n_sessions == 1:
        raise ValueError(
            "start_roi_indices must be provided when there are no consecutive matches"
        )
    if start_session_index < len(normalized_matches):
        return sorted(normalized_matches[start_session_index])
    return sorted(_invert_match_mapping(normalized_matches[start_session_index - 1]))


def _invert_match_mapping(mapping: Mapping[int, int]) -> dict[int, int]:
    inverted: dict[int, int] = {}
    for reference_roi, measurement_roi in mapping.items():
        measurement_roi = int(measurement_roi)
        if measurement_roi in inverted:
            raise ValueError(
                "match mappings must be one-to-one to stitch tracks backward from a later start session"
            )
        inverted[measurement_roi] = int(reference_roi)
    return inverted


def _normalize_match_mapping(
    match: (
        SessionMatchResult
        | Mapping[int, int]
        | np.ndarray
        | tuple[Sequence[int], Sequence[int]]
    ),
) -> dict[int, int]:
    if isinstance(match, SessionMatchResult):
        return match.as_roi_index_mapping()
    if isinstance(match, Mapping):
        return {
            int(reference_roi): int(measurement_roi)
            for reference_roi, measurement_roi in match.items()
        }
    if isinstance(match, tuple) and len(match) == 2:
        reference_roi_indices = [int(value) for value in match[0]]
        measurement_roi_indices = [int(value) for value in match[1]]
        if len(reference_roi_indices) != len(measurement_roi_indices):
            raise ValueError("tuple-based matches must have equal lengths")
        return dict(zip(reference_roi_indices, measurement_roi_indices, strict=True))

    match_array = np.asarray(match)
    if match_array.ndim == 2 and match_array.shape[1] == 2:
        return {
            int(reference_roi): int(measurement_roi)
            for reference_roi, measurement_roi in match_array.tolist()
        }
    raise TypeError("unsupported match representation")


def _empty_match_result(bundle: Any) -> SessionMatchResult:
    empty = np.zeros((0,), dtype=int)
    empty_costs = np.zeros((0,), dtype=float)
    return SessionMatchResult(
        reference_session_name=str(bundle.reference_session_name),
        measurement_session_name=str(bundle.measurement_session_name),
        reference_positions=empty,
        measurement_positions=empty,
        reference_roi_indices=empty,
        measurement_roi_indices=empty,
        costs=empty_costs,
    )


def _gate_cost_matrix_for_linear_assignment(
    cost_matrix: np.ndarray,
    valid_assignment_mask: np.ndarray,
    *,
    max_cost: float,
) -> np.ndarray:
    """Return a cost matrix where invalid candidate links cannot win cheaply.

    Simply solving the original matrix and discarding over-threshold assignments
    afterward can lower match cardinality. For example, an assignment containing
    one over-threshold link plus one very cheap link may have a lower total cost
    than two valid but moderate-cost links. Replacing invalid entries with a
    dominating penalty makes the Hungarian objective maximize the number of valid
    links first, then minimize their total cost.
    """

    valid_costs = np.asarray(cost_matrix[valid_assignment_mask], dtype=float)
    if valid_costs.size == 0:
        raise ValueError("valid_assignment_mask must contain at least one True entry")

    valid_min = float(np.min(valid_costs))
    valid_max = float(np.max(valid_costs))
    cost_span = max(valid_max - valid_min, 1.0)
    cost_scale = max(
        abs(valid_min),
        abs(valid_max),
        abs(float(max_cost)),
        cost_span,
        1.0,
    )
    max_assignments = min(cost_matrix.shape)
    invalid_penalty = (max_assignments + 1) * (cost_scale + cost_span + 1.0)

    return np.where(valid_assignment_mask, cost_matrix, invalid_penalty)
