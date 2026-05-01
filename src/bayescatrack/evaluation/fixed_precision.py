"""Fixed-precision diagnostics for longitudinal complete-track recovery."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .complete_track_scores import complete_track_set, normalize_track_matrix

DEFAULT_COMPLETE_TRACK_FIXED_PRECISIONS = (0.9, 0.95, 0.99)

__all__ = (
    "DEFAULT_COMPLETE_TRACK_FIXED_PRECISIONS",
    "score_complete_tracks_at_fixed_precision",
)


# pylint: disable=too-many-locals
def score_complete_tracks_at_fixed_precision(
    predicted_track_matrix: Any,
    reference_track_matrix: Any,
    *,
    target_precisions: Sequence[float] = DEFAULT_COMPLETE_TRACK_FIXED_PRECISIONS,
    track_scores: Sequence[float] | None = None,
    session_indices: Sequence[int] | None = None,
) -> dict[str, float | int]:
    """Report complete-track recovery at fixed precision operating points.

    The diagnostic sweeps score thresholds over predicted complete tracks and
    reports the maximum number of true complete tracks retained while satisfying
    each target precision. Higher ``track_scores`` values are treated as more
    reliable. If no scores are supplied, all predicted tracks receive the same
    score, yielding a conservative all-or-nothing operating point.
    """

    predicted_matrix = normalize_track_matrix(predicted_track_matrix)
    reference_matrix = normalize_track_matrix(reference_track_matrix)
    if predicted_matrix.shape[1] != reference_matrix.shape[1]:
        raise ValueError("Predicted and reference matrices must have the same number of sessions")

    selected_sessions = _resolve_session_indices(predicted_matrix.shape[1], session_indices)
    _resolve_session_indices(reference_matrix.shape[1], selected_sessions)

    targets = _validate_target_precisions(target_precisions)
    if not targets:
        return {}

    scores = _score_array_for_track_matrix(predicted_matrix, track_scores)
    predicted = _scored_complete_track_dict(predicted_matrix, scores, selected_sessions)
    reference = complete_track_set(reference_matrix, session_indices=selected_sessions)

    diagnostics: dict[str, float | int] = {}
    for target_precision in targets:
        suffix = _fixed_precision_metric_suffix(target_precision)
        true_positives, predictions, precision, recall, threshold = _best_operating_point(
            predicted,
            reference,
            target_precision=target_precision,
        )
        diagnostics.update(
            {
                f"complete_tracks_at_fixed_precision_{suffix}": true_positives,
                f"complete_track_predictions_at_fixed_precision_{suffix}": predictions,
                f"complete_track_precision_at_fixed_precision_{suffix}": precision,
                f"complete_track_recall_at_fixed_precision_{suffix}": recall,
                f"complete_track_score_threshold_at_fixed_precision_{suffix}": threshold,
            }
        )
    return diagnostics


def _best_operating_point(
    predicted: dict[tuple[int, ...], float],
    reference: set[tuple[int, ...]],
    *,
    target_precision: float,
) -> tuple[int, int, float, float, float]:
    best_rank = (0, 1.0, 0, 0)
    best_result = (0, 0, 1.0, 0.0, float("inf"))
    for threshold in sorted(set(predicted.values()), reverse=True):
        retained = {track for track, score in predicted.items() if score >= threshold}
        true_positives = len(retained.intersection(reference))
        false_positives = len(retained.difference(reference))
        false_negatives = len(reference.difference(retained))
        precision = _safe_ratio(true_positives, true_positives + false_positives)
        recall = _safe_ratio(true_positives, true_positives + false_negatives)
        rank = (true_positives, precision, -false_positives, len(retained))
        if precision >= target_precision and rank > best_rank:
            best_rank = rank
            best_result = (true_positives, len(retained), precision, recall, float(threshold))
    return best_result


def _scored_complete_track_dict(
    matrix: np.ndarray,
    track_scores: np.ndarray,
    selected_sessions: Sequence[int],
) -> dict[tuple[int, ...], float]:
    scored_tracks: dict[tuple[int, ...], float] = {}
    for row, score in zip(matrix, track_scores, strict=True):
        values = [row[session_idx] for session_idx in selected_sessions]
        if not all(value is not None for value in values):
            continue
        track = tuple(int(value) for value in values)
        previous_score = scored_tracks.get(track)
        if previous_score is None or score > previous_score:
            scored_tracks[track] = float(score)
    return scored_tracks


def _score_array_for_track_matrix(matrix: np.ndarray, track_scores: Sequence[float] | None) -> np.ndarray:
    if track_scores is None:
        return np.ones((matrix.shape[0],), dtype=float)
    scores = np.asarray(track_scores, dtype=float)
    if scores.ndim != 1 or scores.shape[0] != matrix.shape[0]:
        raise ValueError("track_scores must contain exactly one score per predicted track")
    if not np.all(np.isfinite(scores)):
        raise ValueError("track_scores must contain only finite values")
    return scores


def _resolve_session_indices(num_sessions: int, session_indices: Sequence[int] | None) -> list[int]:
    if session_indices is None:
        return list(range(num_sessions))
    selected: list[int] = []
    for candidate in session_indices:
        session_idx = int(candidate)
        if session_idx < 0 or session_idx >= num_sessions:
            raise IndexError(f"session index {session_idx} out of bounds for {num_sessions} sessions")
        selected.append(session_idx)
    return selected


def _validate_target_precisions(target_precisions: Sequence[float]) -> tuple[float, ...]:
    targets = tuple(float(target_precision) for target_precision in target_precisions)
    for target_precision in targets:
        if not 0.0 <= target_precision <= 1.0:
            raise ValueError("target precisions must be between 0 and 1")
    return targets


def _fixed_precision_metric_suffix(target_precision: float) -> str:
    text = f"{float(target_precision):.6f}".rstrip("0").rstrip(".")
    return text.replace("-", "minus_").replace(".", "_")


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 1.0
    return float(numerator) / float(denominator)
