"""Reference and evaluation utilities for Track2p-style benchmarks.

This module keeps benchmark-native supervision outside the core bridge loader.
It turns Track2p reference artifacts into explicit longitudinal identity labels
and provides small helpers for evaluating predicted associations.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import re
from typing import Any, Sequence

import numpy as np

from . import load_track2p_subject

_SESSION_NAME_PATTERN = re.compile(r"^(?P<session_date>\d{4}-\d{2}-\d{2})(?:_.+)?$")
_PLANE_NAME_PATTERN = re.compile(r"^plane(?P<index>\d+)$")
_MISSING_STRINGS = {"", "none", "nan", "null"}


@dataclass(frozen=True)
class Track2pReference:
    """Ground-truth longitudinal identities in Suite2p ROI indexing."""

    session_names: tuple[str, ...]
    suite2p_indices: np.ndarray
    session_dates: tuple[date | None, ...] = ()
    curated_mask: np.ndarray | None = None
    source: str = "unknown"

    def __post_init__(self) -> None:
        session_names = tuple(str(name) for name in self.session_names)
        if not session_names:
            raise ValueError("session_names must not be empty")
        object.__setattr__(self, "session_names", session_names)

        indices = _as_nullable_int_matrix(self.suite2p_indices)
        if indices.ndim != 2 or indices.shape[1] != len(session_names):
            raise ValueError(
                "suite2p_indices must have shape (n_tracks, n_sessions) "
                "with one column per session"
            )
        object.__setattr__(self, "suite2p_indices", indices)

        if self.session_dates:
            session_dates = tuple(self.session_dates)
            if len(session_dates) != len(session_names):
                raise ValueError("session_dates must match the number of session_names")
        else:
            session_dates = tuple(None for _ in session_names)
        object.__setattr__(self, "session_dates", session_dates)

        if self.curated_mask is not None:
            curated_mask = np.asarray(self.curated_mask, dtype=bool).reshape(-1)
            if curated_mask.shape != (indices.shape[0],):
                raise ValueError("curated_mask must have shape (n_tracks,)")
            object.__setattr__(self, "curated_mask", curated_mask)

    @property
    def n_tracks(self) -> int:
        return int(self.suite2p_indices.shape[0])

    @property
    def n_sessions(self) -> int:
        return int(self.suite2p_indices.shape[1])

    def present_mask(self) -> np.ndarray:
        """Return a boolean matrix marking track presence per session."""

        return np.vectorize(lambda value: value is not None, otypes=[bool])(self.suite2p_indices)

    def all_day_mask(self) -> np.ndarray:
        """Return a boolean mask for tracks present in every session."""

        return np.all(self.present_mask(), axis=1)

    def _filtered_indices(self, *, curated_only: bool = False) -> np.ndarray:
        keep = np.ones((self.n_tracks,), dtype=bool)
        if curated_only:
            if self.curated_mask is None:
                raise ValueError("No curated_mask is available for this reference")
            keep &= self.curated_mask
        return self.suite2p_indices[keep]

    def pairwise_matches(
        self,
        session_a: int,
        session_b: int,
        *,
        curated_only: bool = False,
    ) -> np.ndarray:
        """Return ground-truth pairs ``(roi_in_a, roi_in_b)`` for two sessions."""

        _validate_session_index(session_a, self.n_sessions)
        _validate_session_index(session_b, self.n_sessions)
        indices = self._filtered_indices(curated_only=curated_only)

        pairs: list[tuple[int, int]] = []
        for track_idx in range(indices.shape[0]):
            roi_a = indices[track_idx, session_a]
            roi_b = indices[track_idx, session_b]
            if roi_a is None or roi_b is None:
                continue
            pairs.append((int(roi_a), int(roi_b)))

        if not pairs:
            return np.zeros((0, 2), dtype=int)

        pair_array = np.asarray(pairs, dtype=int)
        order = np.lexsort((pair_array[:, 1], pair_array[:, 0]))
        return pair_array[order]

    def to_session_track_labels(
        self,
        n_rois_per_session: Sequence[int] | None = None,
        *,
        fill_value: int = -1,
        curated_only: bool = False,
    ) -> list[np.ndarray]:
        """Return one label vector per session in Suite2p ROI indexing."""

        indices = self._filtered_indices(curated_only=curated_only)

        if n_rois_per_session is None:
            n_rois = []
            for session_idx in range(self.n_sessions):
                present = [int(value) for value in indices[:, session_idx] if value is not None]
                n_rois.append(max(present) + 1 if present else 0)
        else:
            if len(n_rois_per_session) != self.n_sessions:
                raise ValueError("n_rois_per_session must have one entry per session")
            n_rois = [int(count) for count in n_rois_per_session]
            if any(count < 0 for count in n_rois):
                raise ValueError("n_rois_per_session must contain non-negative integers")

        labels = [np.full((count,), int(fill_value), dtype=int) for count in n_rois]
        for track_idx in range(indices.shape[0]):
            for session_idx in range(self.n_sessions):
                roi_idx = indices[track_idx, session_idx]
                if roi_idx is None:
                    continue
                roi_idx = int(roi_idx)
                if roi_idx < 0 or roi_idx >= labels[session_idx].shape[0]:
                    raise ValueError(
                        f"ROI index {roi_idx} is out of bounds for session {session_idx}"
                    )
                if labels[session_idx][roi_idx] != fill_value:
                    raise ValueError(
                        "Multiple tracks map to the same ROI index in session "
                        f"{session_idx}: ROI {roi_idx}"
                    )
                labels[session_idx][roi_idx] = track_idx
        return labels


def load_track2p_reference(
    track2p_dir: str | Path,
    *,
    plane_name: str = "plane0",
    prefer_suite2p_indices: bool = True,
) -> Track2pReference:
    """Load Track2p identities from a processed Track2p output folder."""

    track2p_dir = Path(track2p_dir)
    if track2p_dir.name != "track2p" and (track2p_dir / "track2p").exists():
        track2p_dir = track2p_dir / "track2p"

    track_ops_path = track2p_dir / "track_ops.npy"
    if not track_ops_path.exists():
        raise FileNotFoundError(f"Could not find {track_ops_path}")

    track_ops = np.load(track_ops_path, allow_pickle=True).item()
    plane_index = _plane_index_from_name(plane_name)

    suite2p_indices_path = track2p_dir / f"{plane_name}_suite2p_indices.npy"
    match_mat_path = track2p_dir / f"{plane_name}_match_mat.npy"
    if prefer_suite2p_indices and suite2p_indices_path.exists():
        suite2p_indices = np.load(suite2p_indices_path, allow_pickle=True)
        source = "track2p_output_suite2p_indices"
    elif match_mat_path.exists():
        suite2p_indices = np.load(match_mat_path, allow_pickle=True)
        source = "track2p_output_match_mat"
    elif suite2p_indices_path.exists():
        suite2p_indices = np.load(suite2p_indices_path, allow_pickle=True)
        source = "track2p_output_suite2p_indices"
    else:
        raise FileNotFoundError(
            "Could not find either "
            f"{suite2p_indices_path.name} or {match_mat_path.name}"
        )

    suite2p_indices = _as_nullable_int_matrix(suite2p_indices)
    session_names = _session_names_from_track_ops(track_ops, suite2p_indices.shape[1])
    session_dates = _session_dates_from_names(session_names)
    curated_mask = _extract_curated_mask(track_ops, plane_index, suite2p_indices.shape[0])

    return Track2pReference(
        session_names=session_names,
        suite2p_indices=suite2p_indices,
        session_dates=session_dates,
        curated_mask=curated_mask,
        source=source,
    )


def load_aligned_subject_reference(
    subject_dir: str | Path,
    *,
    plane_name: str = "plane0",
    input_format: str = "auto",
) -> Track2pReference:
    """Build a reference from already row-aligned matched Suite2p sessions."""

    sessions = load_track2p_subject(subject_dir, plane_name=plane_name, input_format=input_format)
    if not sessions:
        raise ValueError("No sessions were found")

    n_tracks = max(session.plane_data.n_rois for session in sessions)
    suite2p_indices = np.empty((n_tracks, len(sessions)), dtype=object)
    suite2p_indices[:] = None

    for session_idx, session in enumerate(sessions):
        n_rois = int(session.plane_data.n_rois)
        for roi_idx in range(n_rois):
            suite2p_indices[roi_idx, session_idx] = roi_idx

    curated_mask = np.ones((n_tracks,), dtype=bool)
    return Track2pReference(
        session_names=tuple(session.session_name for session in sessions),
        suite2p_indices=suite2p_indices,
        session_dates=tuple(session.session_date for session in sessions),
        curated_mask=curated_mask,
        source="aligned_subject_rows",
    )


def pairs_from_label_vectors(labels_a: Sequence[Any], labels_b: Sequence[Any]) -> np.ndarray:
    """Convert two per-session label vectors into explicit ROI pairs."""

    mapping_a = _label_vector_to_mapping(labels_a)
    mapping_b = _label_vector_to_mapping(labels_b)
    shared_labels = sorted(set(mapping_a).intersection(mapping_b))
    if not shared_labels:
        return np.zeros((0, 2), dtype=int)

    return np.asarray(
        [(mapping_a[label], mapping_b[label]) for label in shared_labels],
        dtype=int,
    )


def score_pairwise_matches(
    predicted_pairs: Sequence[Sequence[Any]] | np.ndarray,
    reference_pairs: Sequence[Sequence[Any]] | np.ndarray,
) -> dict[str, float | int]:
    """Score explicit pairwise matches with precision/recall/F1."""

    predicted = _pair_set(predicted_pairs)
    reference = _pair_set(reference_pairs)

    true_positives = len(predicted & reference)
    false_positives = len(predicted - reference)
    false_negatives = len(reference - predicted)

    precision = _safe_ratio(true_positives, true_positives + false_positives)
    recall = _safe_ratio(true_positives, true_positives + false_negatives)
    f1 = _safe_ratio(2.0 * precision * recall, precision + recall)

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def score_label_vectors_against_reference(
    *,
    labels_a: Sequence[Any],
    labels_b: Sequence[Any],
    reference: Track2pReference,
    session_a: int,
    session_b: int,
    curated_only: bool = False,
) -> dict[str, float | int]:
    """Score predicted session label vectors against a Track2p reference."""

    predicted_pairs = pairs_from_label_vectors(labels_a, labels_b)
    reference_pairs = reference.pairwise_matches(
        session_a,
        session_b,
        curated_only=curated_only,
    )
    return score_pairwise_matches(predicted_pairs, reference_pairs)


def _as_nullable_int_matrix(array_like: Any) -> np.ndarray:
    array = np.asarray(array_like, dtype=object)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    matrix = np.empty(array.shape, dtype=object)
    for index, value in np.ndenumerate(array):
        matrix[index] = _parse_optional_int(value)
    return matrix


def _parse_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        if value.strip().lower() in _MISSING_STRINGS:
            return None
        value = value.strip()
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None
    try:
        integer_value = int(value)
    except (TypeError, ValueError):
        return None
    if integer_value < 0:
        return None
    return integer_value


def _plane_index_from_name(plane_name: str) -> int:
    match = _PLANE_NAME_PATTERN.match(plane_name)
    if match is None:
        raise ValueError(f"plane_name must follow the 'plane#' convention, got {plane_name!r}")
    return int(match.group("index"))


def _session_names_from_track_ops(track_ops: dict[str, Any], n_sessions: int) -> tuple[str, ...]:
    if "all_ds_path" not in track_ops:
        raise KeyError("track_ops.npy does not contain all_ds_path")
    path_list = [Path(str(path)) for path in np.asarray(track_ops["all_ds_path"], dtype=object).reshape(-1).tolist()]
    if len(path_list) != n_sessions:
        raise ValueError(
            "The number of paths in track_ops.all_ds_path does not match the number of sessions"
        )
    return tuple(path.name if path.name else str(path) for path in path_list)


def _session_dates_from_names(session_names: Sequence[str]) -> tuple[date | None, ...]:
    dates: list[date | None] = []
    for session_name in session_names:
        match = _SESSION_NAME_PATTERN.match(session_name)
        dates.append(date.fromisoformat(match.group("session_date")) if match else None)
    return tuple(dates)


def _extract_curated_mask(
    track_ops: dict[str, Any],
    plane_index: int,
    n_tracks: int,
) -> np.ndarray | None:
    for key in (f"vector_curation_plane_{plane_index}", "vector_curation"):
        if key not in track_ops:
            continue
        curated = np.asarray(track_ops[key], dtype=float).reshape(-1)
        if curated.shape != (n_tracks,):
            raise ValueError(f"Curation vector {key!r} has incompatible shape {curated.shape}")
        return curated > 0.5
    return None


def _validate_session_index(session_index: int, n_sessions: int) -> None:
    if session_index < 0 or session_index >= n_sessions:
        raise IndexError(f"session index {session_index} out of bounds for {n_sessions} sessions")


def _label_vector_to_mapping(labels: Sequence[Any]) -> dict[int, int]:
    label_array = np.asarray(labels, dtype=object).reshape(-1)
    mapping: dict[int, int] = {}
    for roi_idx, label in enumerate(label_array):
        track_id = _parse_optional_int(label)
        if track_id is None:
            continue
        if track_id in mapping:
            raise ValueError(f"Label vector contains the same track id ({track_id}) more than once")
        mapping[track_id] = int(roi_idx)
    return mapping


def _pair_set(pairs: Sequence[Sequence[Any]] | np.ndarray) -> set[tuple[int, int]]:
    pair_array = np.asarray(pairs, dtype=object)
    if pair_array.size == 0:
        return set()
    if pair_array.ndim != 2 or pair_array.shape[1] != 2:
        raise ValueError("Pair arrays must have shape (n_pairs, 2)")

    normalized: set[tuple[int, int]] = set()
    for first, second in pair_array.tolist():
        first_int = _parse_optional_int(first)
        second_int = _parse_optional_int(second)
        if first_int is None or second_int is None:
            raise ValueError("Pair arrays must not contain missing values")
        normalized.add((first_int, second_int))
    return normalized


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 1.0
    return float(numerator) / float(denominator)
