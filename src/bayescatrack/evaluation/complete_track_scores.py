"""Track-level scoring helpers for longitudinal ROI identity matrices."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

_MISSING = object()
_MISSING_STRINGS = {"", "none", "nan", "null"}

__all__ = (
    "complete_track_set",
    "normalize_track_matrix",
    "pairwise_track_set",
    "reference_fragment_counts",
    "score_complete_tracks",
    "score_false_continuations",
    "score_fragmentation",
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
    """Return the number of present sessions for each predicted track."""
    matrix = normalize_track_matrix(track_matrix)
    if matrix.shape[0] == 0:
        return np.zeros((0,), dtype=int)
    return np.asarray(
        [sum(value is not None for value in row) for row in matrix], dtype=int
    )


def complete_track_set(
    track_matrix: Any, *, session_indices: Sequence[int] | None = None
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
    links: set[tuple[int, int, int, int]] = set()
    for session_a, session_b in _session_pairs(matrix, session_pairs):
        for row in matrix:
            roi_a = row[session_a]
            roi_b = row[session_b]
            if roi_a is not None and roi_b is not None:
                links.add((int(session_a), int(session_b), int(roi_a), int(roi_b)))
    return links


def reference_fragment_counts(
    predicted_track_matrix: Any,
    reference_track_matrix: Any,
) -> np.ndarray:
    """Return the number of predicted fragments covering each reference track.

    A reference track is covered by a predicted fragment when any ROI in the
    predicted row matches the reference ROI in the same session. Splitting one
    reference identity across multiple predicted rows therefore produces a
    fragment count greater than one. Missing sessions do not introduce extra
    fragments; they are handled by recall/complete-track metrics.
    """
    predicted = normalize_track_matrix(predicted_track_matrix)
    reference = normalize_track_matrix(reference_track_matrix)
    if predicted.shape[1] != reference.shape[1]:
        raise ValueError(
            "Predicted and reference matrices must have the same number of sessions"
        )

    predicted_lookup = _predicted_track_lookup(predicted)
    fragment_counts = np.zeros(reference.shape[0], dtype=int)
    for reference_idx, reference_row in enumerate(reference):
        covering_predicted_tracks: set[int] = set()
        for session_idx, roi in enumerate(reference_row):
            if roi is None:
                continue
            covering_predicted_tracks.update(
                predicted_lookup.get((int(session_idx), int(roi)), ())
            )
        fragment_counts[reference_idx] = len(covering_predicted_tracks)
    return fragment_counts


def score_complete_tracks(
    predicted_track_matrix: Any,
    reference_track_matrix: Any,
    *,
    session_indices: Sequence[int] | None = None,
) -> dict[str, float | int]:
    """Score exact complete-track recovery with precision, recall, and F1."""
    predicted = complete_track_set(
        predicted_track_matrix, session_indices=session_indices
    )
    reference = complete_track_set(
        reference_track_matrix, session_indices=session_indices
    )
    return _score_identity_sets(
        predicted,
        reference,
        prefix="complete_track",
        predicted_total_name="complete_tracks",
        reference_total_name="reference_complete_tracks",
    )


def score_pairwise_tracks(
    predicted_track_matrix: Any,
    reference_track_matrix: Any,
    *,
    session_pairs: Iterable[tuple[int, int]] | None = None,
) -> dict[str, float | int]:
    """Score pairwise links induced by track matrices."""
    predicted = pairwise_track_set(predicted_track_matrix, session_pairs=session_pairs)
    reference = pairwise_track_set(reference_track_matrix, session_pairs=session_pairs)
    return _score_identity_sets(
        predicted,
        reference,
        prefix="pairwise",
        predicted_total_name="pairwise_links",
        reference_total_name="reference_pairwise_links",
    )


def score_false_continuations(
    predicted_track_matrix: Any,
    reference_track_matrix: Any,
    *,
    session_pairs: Iterable[tuple[int, int]] | None = None,
) -> dict[str, float | int]:
    """Score predicted forward links that contradict the reference identity map."""
    predicted = normalize_track_matrix(predicted_track_matrix)
    reference = normalize_track_matrix(reference_track_matrix)
    if predicted.shape[1] != reference.shape[1]:
        raise ValueError(
            "Predicted and reference matrices must have the same number of sessions"
        )

    reference_lookup = _reference_roi_lookup(reference)
    valid: set[tuple[int, int, int, int]] = set()
    false: set[tuple[int, int, int, int]] = set()
    unknown_source: set[tuple[int, int, int, int]] = set()
    for session_a, session_b in _session_pairs(predicted, session_pairs):
        for row in predicted:
            roi_a = row[session_a]
            roi_b = row[session_b]
            if roi_a is None or roi_b is None:
                continue
            link = (int(session_a), int(session_b), int(roi_a), int(roi_b))
            reference_track_idx = reference_lookup.get((int(session_a), int(roi_a)))
            if reference_track_idx is None:
                unknown_source.add(link)
                continue
            expected_roi_b = reference[reference_track_idx, int(session_b)]
            if expected_roi_b is None or int(roi_b) != int(expected_roi_b):
                false.add(link)
            else:
                valid.add(link)

    labeled = len(valid) + len(false)
    return {
        "false_continuations": len(false),
        "valid_continuations": len(valid),
        "labeled_predicted_continuations": labeled,
        "unknown_source_continuations": len(unknown_source),
        "false_continuation_rate": _zero_ratio(len(false), labeled),
    }


def score_fragmentation(
    predicted_track_matrix: Any,
    reference_track_matrix: Any,
) -> dict[str, float | int]:
    """Score fragmentation of reference identities across predicted tracks.

    Fragmentation is measured per reference identity. A reference identity with
    ROIs distributed across two predicted rows has two fragments and one
    fragmentation event. Uncovered reference identities have zero fragments;
    they affect recall metrics, but they are not counted as fragmented.
    """
    reference = normalize_track_matrix(reference_track_matrix)
    fragment_counts = reference_fragment_counts(predicted_track_matrix, reference)
    reference_lengths = track_lengths(reference)
    valid_reference_mask = reference_lengths > 0
    valid_fragment_counts = fragment_counts[valid_reference_mask]

    reference_tracks = int(valid_fragment_counts.size)
    covered_mask = valid_fragment_counts > 0
    covered_reference_tracks = int(np.count_nonzero(covered_mask))
    fragmented_reference_tracks = int(np.count_nonzero(valid_fragment_counts > 1))
    fragmentation_events = int(
        np.sum(np.maximum(valid_fragment_counts - 1, 0), dtype=int)
    )
    fragments = int(np.sum(valid_fragment_counts, dtype=int))

    return {
        "fragmentation_reference_tracks": reference_tracks,
        "fragmentation_covered_reference_tracks": covered_reference_tracks,
        "fragmentation_fragmented_reference_tracks": fragmented_reference_tracks,
        "fragmentation_fragments": fragments,
        "fragmentation_events": fragmentation_events,
        "fragmentation_rate": _zero_ratio(
            fragmented_reference_tracks, reference_tracks
        ),
        "fragmentation_covered_rate": _zero_ratio(
            fragmented_reference_tracks, covered_reference_tracks
        ),
        "fragmentation_mean_fragments_per_reference_track": _mean_or_zero(
            valid_fragment_counts
        ),
        "fragmentation_mean_fragments_per_covered_reference_track": _mean_or_zero(
            valid_fragment_counts[covered_mask]
        ),
        "fragmentation_max_fragments_per_reference_track": (
            int(np.max(valid_fragment_counts)) if reference_tracks else 0
        ),
    }


def summarize_tracks(track_matrix: Any) -> dict[str, float | int]:
    """Summarize the number and length of predicted tracks."""
    matrix = normalize_track_matrix(track_matrix)
    lengths = track_lengths(matrix)
    return {
        "tracks": int(matrix.shape[0]),
        "mean_track_length": float(np.mean(lengths)) if lengths.size else 0.0,
        "max_track_length": int(np.max(lengths)) if lengths.size else 0,
    }


def score_track_matrices(
    predicted_track_matrix: Any,
    reference_track_matrix: Any,
    *,
    session_pairs: Iterable[tuple[int, int]] | None = None,
    complete_session_indices: Sequence[int] | None = None,
) -> dict[str, float | int]:
    """Return pairwise, complete-track, false-continuation, fragmentation, length, and track-error metrics."""
    predicted = normalize_track_matrix(predicted_track_matrix)
    reference = normalize_track_matrix(reference_track_matrix)
    if predicted.shape[1] != reference.shape[1]:
        raise ValueError(
            "Predicted and reference matrices must have the same number of sessions"
        )
    scores: dict[str, float | int] = {}
    scores.update(
        score_pairwise_tracks(predicted, reference, session_pairs=session_pairs)
    )
    scores.update(
        score_complete_tracks(
            predicted, reference, session_indices=complete_session_indices
        )
    )
    scores.update(
        score_false_continuations(predicted, reference, session_pairs=session_pairs)
    )
    scores.update(score_fragmentation(predicted, reference))
    scores.update(summarize_tracks(predicted))

    from .track_error_ledger import summarize_track_errors

    track_error_scores = dict(
        summarize_track_errors(predicted, reference, session_pairs=session_pairs)
    )
    false_continuation_link_rate = track_error_scores.pop(
        "false_continuation_rate", None
    )
    scores.update(track_error_scores)
    if false_continuation_link_rate is not None:
        scores["false_continuation_link_rate"] = false_continuation_link_rate
    return scores


def _score_identity_sets(
    predicted: set[Any],
    reference: set[Any],
    *,
    prefix: str,
    predicted_total_name: str,
    reference_total_name: str,
) -> dict[str, float | int]:
    true_positives, false_positives, false_negatives = _confusion_counts(
        predicted, reference
    )
    precision, recall, f1 = _precision_recall_f1(
        true_positives, false_positives, false_negatives
    )
    return {
        f"{prefix}_true_positives": true_positives,
        f"{prefix}_false_positives": false_positives,
        f"{prefix}_false_negatives": false_negatives,
        f"{prefix}_precision": precision,
        f"{prefix}_recall": recall,
        f"{prefix}_f1": f1,
        predicted_total_name: len(predicted),
        reference_total_name: len(reference),
    }


def _confusion_counts(predicted: set[Any], reference: set[Any]) -> tuple[int, int, int]:
    return (
        len(predicted & reference),
        len(predicted - reference),
        len(reference - predicted),
    )


def _precision_recall_f1(
    true_positives: int, false_positives: int, false_negatives: int
) -> tuple[float, float, float]:
    precision = _safe_ratio(true_positives, true_positives + false_positives)
    recall = _safe_ratio(true_positives, true_positives + false_negatives)
    return precision, recall, _safe_ratio(2.0 * precision * recall, precision + recall)


def _reference_roi_lookup(reference: np.ndarray) -> dict[tuple[int, int], int]:
    lookup: dict[tuple[int, int], int] = {}
    ambiguous_keys: set[tuple[int, int]] = set()
    for track_idx, row in enumerate(reference):
        for session_idx, roi in enumerate(row):
            if roi is None:
                continue
            key = (int(session_idx), int(roi))
            previous = lookup.get(key)
            if previous is None:
                lookup[key] = int(track_idx)
            elif previous != int(track_idx):
                ambiguous_keys.add(key)
    for key in ambiguous_keys:
        lookup.pop(key, None)
    return lookup


def _predicted_track_lookup(matrix: np.ndarray) -> dict[tuple[int, int], set[int]]:
    lookup: dict[tuple[int, int], set[int]] = {}
    for track_idx, row in enumerate(matrix):
        for session_idx, roi in enumerate(row):
            if roi is None:
                continue
            lookup.setdefault((int(session_idx), int(roi)), set()).add(int(track_idx))
    return lookup


def _parse_optional_int(value: Any) -> int | None:
    candidate = _optional_int_candidate(value)
    if candidate is _MISSING:
        return None
    try:
        parsed = int(candidate)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _optional_int_candidate(value: Any) -> Any:
    if value is None:
        return _MISSING
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        stripped = value.strip()
        return _MISSING if stripped.casefold() in _MISSING_STRINGS else stripped
    if isinstance(value, (float, np.floating)) and bool(np.isnan(value)):
        return _MISSING
    return value


def _selected_sessions(
    matrix: np.ndarray, session_indices: Sequence[int] | None
) -> list[int]:
    selected = (
        list(range(matrix.shape[1]))
        if session_indices is None
        else [int(idx) for idx in session_indices]
    )
    for session_idx in selected:
        _validate_session_index(matrix, session_idx)
    return selected


def _session_pairs(
    matrix: np.ndarray, session_pairs: Iterable[tuple[int, int]] | None
) -> list[tuple[int, int]]:
    pairs = (
        [(idx, idx + 1) for idx in range(max(0, matrix.shape[1] - 1))]
        if session_pairs is None
        else list(session_pairs)
    )
    for session_a, session_b in pairs:
        _validate_session_index(matrix, session_a)
        _validate_session_index(matrix, session_b)
        if session_a >= session_b:
            raise ValueError("session_pairs must point forward in time")
    return [(int(session_a), int(session_b)) for session_a, session_b in pairs]


def _validate_session_index(matrix: np.ndarray, session_idx: int) -> None:
    if session_idx < 0 or session_idx >= matrix.shape[1]:
        raise IndexError(
            f"session index {session_idx} out of bounds for {matrix.shape[1]} sessions"
        )


def _safe_ratio(numerator: float, denominator: float) -> float:
    return 1.0 if denominator == 0 else float(numerator) / float(denominator)


def _zero_ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else float(numerator) / float(denominator)


def _mean_or_zero(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else 0.0
