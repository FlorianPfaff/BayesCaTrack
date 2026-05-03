"""Ground-truth evaluation helpers for Track2p-style track tables.

This module uses BayesCaTrack matching helpers to load Track2p-style ground-truth
and prediction CSV files and score predicted track tables with exact full-track
and prefix-horizon metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .matching import build_track_rows_from_matches

_MISSING_VALUE_STRINGS = {"", "na", "nan", "none", "null", "-"}
_TRACK_ID_HEADERS = {
    "track_id",
    "track",
    "id",
    "gt_id",
    "gt_track",
    "trackid",
}
_LONG_TRACK_HEADERS = _TRACK_ID_HEADERS | {"trajectory", "traj", "trajectory_id"}
_LONG_SESSION_HEADERS = {"session", "session_name", "day", "dataset", "recording"}
_LONG_ROI_HEADERS = {
    "roi",
    "roi_index",
    "roi_idx",
    "cell",
    "cell_id",
    "cell_index",
    "s2p_index",
    "s2p_idx",
    "index",
}
_SESSION_DIR_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}(?:_.+)?$")


@dataclass(frozen=True)
class TrackTable:
    """Track table with one row per track and one column per session."""

    session_names: tuple[str, ...]
    tracks: np.ndarray

    def __post_init__(self) -> None:
        session_names = tuple(str(name) for name in self.session_names)
        tracks = np.asarray(self.tracks, dtype=int)
        if tracks.ndim != 2:
            raise ValueError("tracks must have shape (n_tracks, n_sessions)")
        if tracks.shape[1] != len(session_names):
            raise ValueError(
                "tracks second dimension must equal the number of session names"
            )
        if len(session_names) == 0:
            raise ValueError("session_names must not be empty")
        object.__setattr__(self, "session_names", session_names)
        object.__setattr__(self, "tracks", tracks)

    @property
    def n_tracks(self) -> int:
        return int(self.tracks.shape[0])

    @property
    def n_sessions(self) -> int:
        return int(self.tracks.shape[1])

    def aligned_to(self, session_names: Sequence[str]) -> "TrackTable":
        """Return a copy with columns reordered to ``session_names``."""
        session_names = tuple(str(name) for name in session_names)
        if set(session_names) != set(self.session_names):
            raise ValueError("session names must match exactly for alignment")
        if session_names == self.session_names:
            return self
        indices = [self.session_names.index(name) for name in session_names]
        return TrackTable(session_names=session_names, tracks=self.tracks[:, indices])

    def row_tuples(
        self,
        *,
        horizon: int | None = None,
        require_complete: bool = False,
    ) -> list[tuple[int, ...]]:
        """Return rows as integer tuples, optionally truncated to ``horizon``."""
        if horizon is None:
            rows = self.tracks
        else:
            if not 1 <= horizon <= self.n_sessions:
                raise ValueError("horizon must be between 1 and the number of sessions")
            rows = self.tracks[:, :horizon]

        tuples: list[tuple[int, ...]] = []
        for row in rows.tolist():
            row_tuple = tuple(int(value) for value in row)
            if require_complete and any(value < 0 for value in row_tuple):
                continue
            tuples.append(row_tuple)
        return tuples


@dataclass(frozen=True)
class TrackEvaluation:
    """Summary of Track2p-style benchmark metrics."""

    complete_tracks: float
    proportion_correct_by_horizon: dict[int, float]
    n_ground_truth_tracks: int
    n_predicted_tracks: int
    n_exact_full_track_matches: int

    def to_json_dict(self) -> dict[str, object]:
        return {
            "complete_tracks": float(self.complete_tracks),
            "proportion_correct_by_horizon": {
                str(horizon): float(value)
                for horizon, value in self.proportion_correct_by_horizon.items()
            },
            "n_ground_truth_tracks": int(self.n_ground_truth_tracks),
            "n_predicted_tracks": int(self.n_predicted_tracks),
            "n_exact_full_track_matches": int(self.n_exact_full_track_matches),
        }


def _normalize_header(header: str) -> str:
    return header.strip().lower().replace(" ", "_")


def _parse_roi_value(  # pylint: disable=too-many-return-statements
    value: str | int | float | None,
) -> int:
    if value is None:
        return -1
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return -1
        if float(value).is_integer():
            return int(value)
        raise ValueError(f"ROI index must be integer-like, got {value!r}")

    text = str(value).strip()
    if _normalize_header(text) in _MISSING_VALUE_STRINGS:
        return -1
    number = float(text)
    if np.isnan(number):
        return -1
    if not float(number).is_integer():
        raise ValueError(f"ROI index must be integer-like, got {value!r}")
    return int(number)


def _parse_semicolon_roi_values(value: str, *, n_sessions: int) -> list[int]:
    """Parse Track2p ground-truth rows stored as ``roi;roi;...`` strings."""

    parts = str(value).strip().split(";")
    if len(parts) < n_sessions:
        parts.extend([""] * (n_sessions - len(parts)))
    extra_parts = parts[n_sessions:]
    nonempty_parts = [
        part for part in parts if _normalize_header(part) not in _MISSING_VALUE_STRINGS
    ]
    nonempty_extra_parts = [
        part
        for part in extra_parts
        if _normalize_header(part) not in _MISSING_VALUE_STRINGS
    ]
    if nonempty_extra_parts:
        raise ValueError(
            "semicolon-encoded track has more non-empty entries than sessions "
            f"({len(nonempty_parts)} non-empty values, {n_sessions} sessions): {value!r}"
        )
    return [_parse_roi_value(part) for part in parts[:n_sessions]]


def _infer_semicolon_session_count(
    headers: Sequence[str],
    rows: Sequence[Mapping[str, str]],
) -> int | None:
    """Infer the number of sessions for ``track_id,track`` semicolon CSVs."""

    widths: list[int] = []
    for row in rows:
        data_headers: list[str] = []
        for header in headers:
            value = str(row.get(header, "")).strip()
            if _normalize_header(header) in _TRACK_ID_HEADERS and ";" not in value:
                continue
            data_headers.append(header)
        values = [str(row.get(header, "")).strip() for header in data_headers]
        semicolon_values = [value for value in values if ";" in value]
        if not semicolon_values:
            continue
        nonempty_plain_values = [
            value
            for value in values
            if ";" not in value
            and _normalize_header(value) not in _MISSING_VALUE_STRINGS
        ]
        if len(semicolon_values) == 1 and not nonempty_plain_values:
            widths.append(len(semicolon_values[0].split(";")))
            continue
        raise ValueError(
            "CSV row mixes semicolon-encoded tracks with per-session ROI values"
        )

    if not widths:
        return None
    return max(widths)


def _semicolon_encoded_row(
    headers: Sequence[str],
    row: Mapping[str, str],
    *,
    n_sessions: int,
) -> list[int] | None:
    """Return a semicolon-encoded track row, if the CSV row uses that representation."""

    data_headers: list[str] = []
    for header in headers:
        value = str(row.get(header, "")).strip()
        if _normalize_header(header) in _TRACK_ID_HEADERS and ";" not in value:
            continue
        data_headers.append(header)
    values = [str(row.get(header, "")).strip() for header in data_headers]
    semicolon_values = [value for value in values if ";" in value]
    if not semicolon_values:
        return None
    nonempty_plain_values = [
        value
        for value in values
        if ";" not in value and _normalize_header(value) not in _MISSING_VALUE_STRINGS
    ]
    if len(semicolon_values) == 1 and not nonempty_plain_values:
        return _parse_semicolon_roi_values(semicolon_values[0], n_sessions=n_sessions)
    raise ValueError(
        "CSV row mixes semicolon-encoded tracks with per-session ROI values"
    )


def _rows_from_csv(csv_path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    csv_path = Path(csv_path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file {csv_path} has no header row")
        fieldnames = [str(name) for name in reader.fieldnames]
        rows = [{str(key): value for key, value in row.items()} for row in reader]
    if not rows:
        raise ValueError(f"CSV file {csv_path} contains no data rows")
    return fieldnames, rows


def _looks_like_long_format(headers: Sequence[str]) -> bool:
    normalized = {_normalize_header(header) for header in headers}
    return (
        bool(normalized & _LONG_TRACK_HEADERS)
        and bool(normalized & _LONG_SESSION_HEADERS)
        and bool(normalized & _LONG_ROI_HEADERS)
    )


def _find_matching_header(headers: Sequence[str], candidates: set[str]) -> str | None:
    for header in headers:
        if _normalize_header(header) in candidates:
            return header
    return None


def _load_long_format(  # pylint: disable=too-many-locals
    headers: Sequence[str],
    rows: Sequence[Mapping[str, str]],
    session_names: Sequence[str] | None,
) -> TrackTable:
    track_header = _find_matching_header(headers, _LONG_TRACK_HEADERS)
    session_header = _find_matching_header(headers, _LONG_SESSION_HEADERS)
    roi_header = _find_matching_header(headers, _LONG_ROI_HEADERS)
    if track_header is None or session_header is None or roi_header is None:
        raise ValueError("could not infer track/session/roi columns from long CSV")

    if session_names is None:
        ordered_sessions: list[str] = []
        seen_sessions: set[str] = set()
        for row in rows:
            session_name = str(row[session_header]).strip()
            if session_name not in seen_sessions:
                seen_sessions.add(session_name)
                ordered_sessions.append(session_name)
        session_names = ordered_sessions
    else:
        session_names = [str(name) for name in session_names]

    session_to_index = {name: index for index, name in enumerate(session_names)}
    grouped: dict[str, np.ndarray] = {}
    for row in rows:
        track_id = str(row[track_header]).strip()
        session_name = str(row[session_header]).strip()
        if session_name not in session_to_index:
            continue
        roi_index = _parse_roi_value(row[roi_header])
        if track_id not in grouped:
            grouped[track_id] = np.full((len(session_names),), -1, dtype=int)
        grouped[track_id][session_to_index[session_name]] = roi_index

    ordered_track_ids = sorted(grouped)
    if ordered_track_ids:
        tracks = np.vstack([grouped[track_id] for track_id in ordered_track_ids])
    else:
        tracks = np.zeros((0, len(session_names)), dtype=int)
    return TrackTable(session_names=tuple(session_names), tracks=tracks)


def _load_wide_format(
    headers: Sequence[str],
    rows: Sequence[Mapping[str, str]],
    session_names: Sequence[str] | None,
) -> TrackTable:
    if session_names is None:
        semicolon_session_count = _infer_semicolon_session_count(headers, rows)
        if semicolon_session_count is not None:
            session_names = [
                f"session_{index}" for index in range(semicolon_session_count)
            ]
        else:
            candidate_headers: list[str] = []
            for header in headers:
                if _normalize_header(header) in _TRACK_ID_HEADERS:
                    continue
                candidate_headers.append(header)
            if not candidate_headers:
                raise ValueError("could not infer any session columns from wide CSV")
            session_names = [str(name) for name in candidate_headers]
    else:
        session_names = [str(name) for name in session_names]

    tracks = np.full((len(rows), len(session_names)), -1, dtype=int)
    for row_index, row in enumerate(rows):
        semicolon_values = _semicolon_encoded_row(
            headers, row, n_sessions=len(session_names)
        )
        if semicolon_values is not None:
            tracks[row_index, :] = semicolon_values
            continue
        for session_index, session_name in enumerate(session_names):
            if session_name not in row:
                raise ValueError(
                    f"session column {session_name!r} not present in CSV row"
                )
            tracks[row_index, session_index] = _parse_roi_value(row[session_name])
    return TrackTable(session_names=tuple(session_names), tracks=tracks)


def _looks_like_track2p_session_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / "suite2p").exists() or (path / "data_npy").exists():
        return True
    return _SESSION_DIR_PATTERN.match(path.name) is not None


def _infer_subject_session_names_from_ground_truth_path(
    csv_path: str | Path,
) -> tuple[str, ...]:
    """Infer sibling Track2p session folders for a subject-level ground_truth.csv."""

    parent = Path(csv_path).parent
    if not parent.exists() or not parent.is_dir():
        return ()
    session_dirs = [
        child
        for child in parent.iterdir()
        if child.name != "track2p" and _looks_like_track2p_session_dir(child)
    ]
    session_dirs.sort(key=lambda path: path.name)
    return tuple(path.name for path in session_dirs)


def load_track_table_csv(
    csv_path: str | Path,
    *,
    session_names: Sequence[str] | None = None,
) -> TrackTable:
    """Load a track table from a wide or long CSV file."""
    headers, rows = _rows_from_csv(csv_path)
    if _looks_like_long_format(headers):
        return _load_long_format(headers, rows, session_names)
    return _load_wide_format(headers, rows, session_names)


def load_track2p_ground_truth_csv(
    csv_path: str | Path,
    *,
    session_names: Sequence[str] | None = None,
) -> TrackTable:
    """Load a Track2p ``ground_truth.csv`` file.

    If ``session_names`` is omitted and the CSV sits in a Track2p subject folder,
    sibling session directory names are used. This prevents semicolon-encoded
    ``track_id,track`` files from being mistaken for one-session wide CSVs.
    """

    if session_names is None:
        inferred_session_names = _infer_subject_session_names_from_ground_truth_path(
            csv_path
        )
        if inferred_session_names:
            session_names = inferred_session_names
    return load_track_table_csv(csv_path, session_names=session_names)


def tracks_from_consecutive_matches(
    session_names: Sequence[str],
    matches: Sequence[
        Mapping[int, int]
        | Sequence[tuple[int, int]]
        | np.ndarray
        | tuple[Sequence[int], Sequence[int]]
    ],
    *,
    start_roi_indices: Sequence[int] | None = None,
) -> TrackTable:
    """Reconstruct wide tracks from consecutive pairwise assignments."""
    track_rows = build_track_rows_from_matches(
        session_names,
        matches,
        start_roi_indices=start_roi_indices,
        fill_value=-1,
    )
    return TrackTable(
        session_names=tuple(str(name) for name in session_names), tracks=track_rows
    )


def _align_prediction_to_ground_truth(
    ground_truth: TrackTable,
    prediction: TrackTable,
) -> TrackTable:
    if set(ground_truth.session_names) != set(prediction.session_names):
        raise ValueError(
            "ground truth and prediction must refer to the same set of sessions"
        )
    return prediction.aligned_to(ground_truth.session_names)


def _row_counter(
    track_table: TrackTable,
    *,
    horizon: int | None = None,
    require_complete: bool = False,
) -> Counter[tuple[int, ...]]:
    return Counter(
        track_table.row_tuples(
            horizon=horizon,
            require_complete=require_complete,
        )
    )


def _multiset_intersection_size(
    left: Counter[tuple[int, ...]],
    right: Counter[tuple[int, ...]],
) -> int:
    return int(sum((left & right).values()))


def complete_tracks_score(ground_truth: TrackTable, prediction: TrackTable) -> float:
    """Return the Track2p 'complete tracks' (CT) score.

    This is the F1 score where positives are exact full-track reconstructions.
    Duplicate predicted tracks therefore still count as false positives.
    """
    prediction = _align_prediction_to_ground_truth(ground_truth, prediction)
    ground_truth_rows = _row_counter(ground_truth)
    prediction_rows = _row_counter(prediction)
    true_positives = _multiset_intersection_size(ground_truth_rows, prediction_rows)
    false_positives = prediction.n_tracks - true_positives
    false_negatives = ground_truth.n_tracks - true_positives
    denominator = 2 * true_positives + false_positives + false_negatives
    if denominator == 0:
        return 0.0
    return (2.0 * true_positives) / denominator


def proportion_correct_by_horizon(
    ground_truth: TrackTable,
    prediction: TrackTable,
) -> dict[int, float]:
    """Return the fraction of GT tracks reconstructed correctly up to each horizon."""
    prediction = _align_prediction_to_ground_truth(ground_truth, prediction)
    result: dict[int, float] = {}
    denominator = ground_truth.n_tracks
    if denominator == 0:
        return {horizon: 0.0 for horizon in range(2, ground_truth.n_sessions + 1)}

    for horizon in range(2, ground_truth.n_sessions + 1):
        ground_truth_rows = _row_counter(ground_truth, horizon=horizon)
        prediction_rows = _row_counter(
            prediction,
            horizon=horizon,
            require_complete=True,
        )
        correctly_reconstructed = _multiset_intersection_size(
            ground_truth_rows,
            prediction_rows,
        )
        result[horizon] = correctly_reconstructed / denominator
    return result


def evaluate_track_table_prediction(
    ground_truth: TrackTable,
    prediction: TrackTable,
) -> TrackEvaluation:
    """Compute the Track2p benchmark metrics for one prediction."""
    prediction = _align_prediction_to_ground_truth(ground_truth, prediction)
    exact_matches = _multiset_intersection_size(
        _row_counter(ground_truth),
        _row_counter(prediction),
    )
    return TrackEvaluation(
        complete_tracks=complete_tracks_score(ground_truth, prediction),
        proportion_correct_by_horizon=proportion_correct_by_horizon(
            ground_truth,
            prediction,
        ),
        n_ground_truth_tracks=ground_truth.n_tracks,
        n_predicted_tracks=prediction.n_tracks,
        n_exact_full_track_matches=exact_matches,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate predicted Track2p-style tracks against ground truth."
    )
    parser.add_argument("ground_truth_csv", help="Path to Track2p ground_truth.csv")
    parser.add_argument("prediction_csv", help="Path to predicted tracks CSV")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    ground_truth = load_track2p_ground_truth_csv(args.ground_truth_csv)
    prediction = load_track_table_csv(
        args.prediction_csv,
        session_names=ground_truth.session_names,
    )
    evaluation = evaluate_track_table_prediction(ground_truth, prediction)
    print(json.dumps(evaluation.to_json_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
