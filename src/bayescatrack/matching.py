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
    max_cost: float | None = None,
) -> SessionMatchResult:
    """Solve a :class:`SessionAssociationBundle` via linear assignment.

    Parameters
    ----------
    bundle
        Any object exposing the attributes used by
        :class:`track2p_pyrecest_bridge.SessionAssociationBundle`.
    max_cost
        Optional post-assignment gate. Matched pairs with assignment cost larger
        than this threshold are discarded.
    """

    if linear_sum_assignment is None:
        raise ImportError(
            "solve_bundle_linear_assignment requires scipy.optimize.linear_sum_assignment"
        )

    cost_matrix = np.asarray(bundle.pairwise_cost_matrix, dtype=float)
    if cost_matrix.ndim != 2:
        raise ValueError("bundle.pairwise_cost_matrix must be two-dimensional")
    if max_cost is not None and max_cost < 0.0:
        raise ValueError("max_cost must be non-negative")

    if cost_matrix.shape[0] == 0 or cost_matrix.shape[1] == 0:
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

    reference_positions, measurement_positions = linear_sum_assignment(cost_matrix)
    assignment_costs = cost_matrix[reference_positions, measurement_positions]
    if max_cost is not None:
        keep = assignment_costs <= max_cost
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
    max_cost: float | None = None,
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
        ROI indices from the first session from which tracks should be grown.
        If omitted, the sorted keys of the first match mapping are used.
    fill_value
        Integer used for missing/unmatched entries.
    """

    session_names = tuple(str(name) for name in session_names)
    if len(session_names) == 0:
        raise ValueError("session_names must not be empty")
    if len(matches) != max(len(session_names) - 1, 0):
        raise ValueError("matches must have length len(session_names) - 1")

    normalized_matches = [_normalize_match_mapping(match) for match in matches]

    if start_roi_indices is None:
        if not normalized_matches:
            raise ValueError(
                "start_roi_indices must be provided when there are no consecutive matches"
            )
        start_roi_indices = sorted(normalized_matches[0])
    else:
        start_roi_indices = [int(index) for index in start_roi_indices]

    track_rows = np.full(
        (len(start_roi_indices), len(session_names)), int(fill_value), dtype=int
    )
    if len(start_roi_indices) == 0:
        return track_rows

    track_rows[:, 0] = np.asarray(start_roi_indices, dtype=int)
    for row_index, start_roi in enumerate(start_roi_indices):
        current_roi = int(start_roi)
        for match_index, mapping in enumerate(normalized_matches):
            next_roi = int(mapping.get(current_roi, fill_value))
            track_rows[row_index, match_index + 1] = next_roi
            if next_roi == fill_value:
                break
            current_roi = next_roi
    return track_rows


def build_track_rows_from_bundles(
    bundles: Sequence[Any],
    *,
    max_cost: float | None = None,
    start_roi_indices: Sequence[int] | None = None,
    fill_value: int = -1,
) -> tuple[tuple[str, ...], np.ndarray, list[SessionMatchResult]]:
    """Solve consecutive bundles and stitch them into wide track rows."""

    bundles = list(bundles)
    if not bundles:
        raise ValueError("bundles must not be empty")

    match_results = solve_consecutive_bundle_linear_assignments(
        bundles,
        max_cost=max_cost,
    )
    session_names = _session_names_from_bundles(bundles)
    if start_roi_indices is None:
        start_roi_indices = np.asarray(bundles[0].reference_roi_indices, dtype=int)
    track_rows = build_track_rows_from_matches(
        session_names,
        match_results,
        start_roi_indices=start_roi_indices,
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
