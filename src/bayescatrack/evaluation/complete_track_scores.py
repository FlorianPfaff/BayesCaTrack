"""Track-level scoring helpers for longitudinal ROI identity matrices."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

_MISSING_STRINGS = {"", "none", "nan", "null"}

__all__ = (
    "complete_track_set",
    "normalize_track_matrix",
    "pairwise_track_set",
    "score_complete_tracks",
    "score_pairwise_tracks",
    "score_track_matrices",
    "summarize_tracks",
    "track_lengths",
)


def normalize_track_matrix(track_matrix: Any) -> np.ndarray:
    """Return an object matrix containing integer ROI indices or ``None``."""

    matrix = np.asarray(track_matrix, dtype=object)
    if matrix.ndim != 2:
        raise ValueError("track_matrix must have shape (n_tracks, n_sessions)")

    normalized = np.empty(matrix.shape, dtype=object)
    for index, value in np.ndenumerate(matrix):
        normalized[index] = _parse_optional_int(value)
    return normalized


def track_lengths(track_matrix: Any) -> np.ndarray:
    """Return the number of present sessions for each track row."""

    matrix = normalize_track_matrix(track_matrix)
    if matrix.shape[0] == 0:
        return np.zeros((0,), dtype=int)
    present = np.vectorize(lambda value: value is not None, otypes=[bool])(matrix)
    return np.sum(present, axis=1, dtype=int)


def complete_track_set(
    track_matrix: Any,
    *,
    session_indices: Sequence[int] | None = None,
) -> set[tuple[int, ...]]:
    """Return exact full-track tuples for tracks present in every selected session."""

    matrix = normalize_track_matrix(track_matrix)
    selected_sessions = _selected_sessions(matrix, session_indices)
    complete_tracks: set[tuple[int, ...]] = set()
    for row in matrix:
        values = [row[session_idx] for session_idx in selected_sessions]
        if all(value is not None for value in values):
            complete_tracks.add(tuple(int(value) for value in values))
    return complete_tracks


def pairwise_track_set(
    track_matrix: Any,
    *,
    session_pairs: Iterable[tuple[int, int]] | None = None,
) -> set[tuple[int, int, int, int]]:
    """Return pairwise identity links encoded as ``(session_a, session_b, roi_a, roi_b)``."""

    matrix = normalize_track_matrix(track_matrix)
    pairs = list(_adjacent_session_pairs(matrix) if session_pairs is None else session_pairs)
    normalized_pairs: set[tuple[int, int, int, int]] = set()
    for session_a, session_b in pairs:
        _validate_session_index(matrix, session_a)
        _validate_session_index(matrix, session_b)
        if session_a >= session_b:
            raise ValueError("session_pairs must point forward in time")
        for row in matrix:
            roi_a = row[session_a]
            roi_b = row[session_b]
            if roi_a is None or roi_b is None:
                continue
            normalized_pairs.add((int(session_a), int(session_b), int(roi_a), int(roi_b)))
    return normalized_pairs


def score_complete_tracks(
    predicted_track_matrix: Any,
    reference_track_matrix: Any,
    *,
    session_indices: Sequence[int] | None = None,
) -> dict[str, float | int]:
    """Score exact complete-track recovery with precision, recall, and F1."""

    predicted = complete_track_set(predicted_track_matrix, session_indices=session_indices)
    reference = complete_track_set(reference_track_matrix, session_indices=session_indices)
    counts = _set_classification_counts(predicted, reference)
    metrics = _precision_recall_f1(*counts)
    return {
        "complete_track_true_positives": counts[0],
        "complete_track_false_positives": counts[1],
        "complete_track_false_negatives": counts[2],
        "complete_track_precision": metrics["precision"],
        "complete_track_recall": metrics["recall"],
        "complete_track_f1": metrics["f1"],
        "complete_tracks": len(predicted),
        "reference_complete_tracks": len(reference),
    }


def score_pairwise_tracks(
    predicted_track_matrix: Any,
    reference_track_matrix: Any,
    *,
    session_pairs: Iterable[tuple[int, int]] | None = None,
) -> dict[str, float | int]:
    """Score pairwise links induced by track matrices."""

    predicted = pairwise_track_set(predicted_track_matrix, session_pairs=session_pairs)
    reference = pairwise_track_set(reference_track_matrix, session_pairs=session_pairs)
    counts = _set_classification_counts(predicted, reference)
    metrics = _precision_recall_f1(*counts)
    return {
        "pairwise_true_positives": counts[0],
        "pairwise_false_positives": counts[1],
        "pairwise_false_negatives": counts[2],
        "pairwise_precision": metrics["precision"],
        "pairwise_recall": metrics["recall"],
        "pairwise_f1": metrics["f1"],
        "pairwise_links": len(predicted),
        "reference_pairwise_links": len(reference),
    }


def summarize_tracks(track_matrix: Any) -> dict[str, float | int]:
    """Summarize the number and length of predicted tracks."""

    matrix = normalize_track_matrix(track_matrix)
    lengths = track_lengths(matrix)
    n_tracks = int(matrix.shape[0])
    mean_track_length = float(np.mean(lengths)) if lengths.size else 0.0
    return {
        "tracks": n_tracks,
        "mean_track_length": mean_track_length,
        "max_track_length": int(np.max(lengths)) if lengths.size else 0,
    }


def score_track_matrices(
    predicted_track_matrix: Any,
    reference_track_matrix: Any,
    *,
    session_pairs: Iterable[tuple[int, int]] | None = None,
    complete_session_indices: Sequence[int] | None = None,
) -> dict[str, float | int]:
    """Return pairwise, complete-track, and length metrics for two track matrices."""

    predicted = normalize_track_matrix(predicted_track_matrix)
    reference = normalize_track_matrix(reference_track_matrix)
    if predicted.shape[1] != reference.shape[1]:
        raise ValueError("Predicted and reference matrices must have the same number of sessions")

    scores: dict[str, float | int] = {}
    scores.update(score_pairwise_tracks(predicted, reference, session_pairs=session_pairs))
    scores.update(score_complete_tracks(predicted, reference, session_indices=complete_session_indices))
    scores.update(summarize_tracks(predicted))
    return scores


def _parse_optional_int(value: Any) -> int | None:
    candidate = _coerce_optional_scalar(value)
    if candidate is None:
        return None
    try:
        integer_value = int(candidate)
    except (TypeError, ValueError):
        return None
    return integer_value if integer_value >= 0 else None


def _coerce_optional_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        stripped = value.strip()
        return None if stripped.lower() in _MISSING_STRINGS else stripped
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None
    return value


def _set_classification_counts(
    predicted: set[tuple[int, ...]],
    reference: set[tuple[int, ...]],
) -> tuple[int, int, int]:
    return (
        len(predicted & reference),
        len(predicted - reference),
        len(reference - predicted),
    )


def _precision_recall_f1(
    true_positives: int,
    false_positives: int,
    false_negatives: int,
) -> dict[str, float]:
    precision = _safe_ratio(true_positives, true_positives + false_positives)
    recall = _safe_ratio(true_positives, true_positives + false_negatives)
    return {
        "precision": precision,
        "recall": recall,
        "f1": _safe_ratio(2.0 * precision * recall, precision + recall),
    }


def _selected_sessions(matrix: np.ndarray, session_indices: Sequence[int] | None) -> list[int]:
    if session_indices is None:
        return list(range(matrix.shape[1]))
    selected = [int(session_idx) for session_idx in session_indices]
    for session_idx in selected:
        _validate_session_index(matrix, session_idx)
    return selected


def _adjacent_session_pairs(matrix: np.ndarray) -> list[tuple[int, int]]:
    return [(session_idx, session_idx + 1) for session_idx in range(max(0, matrix.shape[1] - 1))]


def _validate_session_index(matrix: np.ndarray, session_idx: int) -> None:
    if session_idx < 0 or session_idx >= matrix.shape[1]:
        raise IndexError(f"session index {session_idx} out of bounds for {matrix.shape[1]} sessions")


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 1.0
    return float(numerator) / float(denominator)
