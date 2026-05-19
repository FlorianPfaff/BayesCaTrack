"""Track-level scoring helpers for longitudinal ROI identity matrices.

The implementation is provided by :mod:`pyrecest.utils.track_evaluation`.
BayesCaTrack keeps this module as a compatibility import path for benchmark
code and downstream users.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from pyrecest.utils.track_evaluation import (
    complete_track_set,
    normalize_track_matrix,
    pairwise_track_set,
    reference_fragment_counts,
    score_complete_tracks,
    score_false_continuations,
    score_fragmentation,
    score_pairwise_tracks,
    score_track_matrices as _pyrecest_score_track_matrices,
    summarize_tracks,
    track_lengths,
)

TrackLink = tuple[int, int, int, int]


def score_track_matrices(
    predicted_track_matrix: Any,
    reference_track_matrix: Any,
    *,
    session_pairs: Iterable[tuple[int, int]] | None = None,
    complete_session_indices: Sequence[int] | None = None,
) -> dict[str, float | int]:
    """Return aggregate track metrics with duplicate-aware benchmark F1 counts.

    PyRecEst's generic track metrics score pairwise links and complete tracks as
    sets. That is useful for identity-set comparison, but Track2p-style
    benchmark rows should count duplicate predicted rows/links as false
    positives. This wrapper preserves the PyRecEst diagnostic ledger and summary
    fields, then replaces pairwise, track-link, and complete-track precision,
    recall, F1, and count fields with multiset counts.
    """

    normalized_session_pairs = _normalize_session_pairs(session_pairs)
    normalized_complete_session_indices = _normalize_complete_session_indices(
        complete_session_indices
    )
    scores = dict(
        _pyrecest_score_track_matrices(
            predicted_track_matrix,
            reference_track_matrix,
            session_pairs=normalized_session_pairs,
            complete_session_indices=normalized_complete_session_indices,
        )
    )
    predicted = normalize_track_matrix(predicted_track_matrix)
    reference = normalize_track_matrix(reference_track_matrix)
    _validate_compatible_shapes(predicted, reference)

    scores.update(
        _score_multiset_track_links(
            predicted,
            reference,
            session_pairs=normalized_session_pairs,
            prefix="track_link",
            predicted_total_name="track_links",
            reference_total_name="reference_track_links",
        )
    )
    scores.update(
        _score_multiset_track_links(
            predicted,
            reference,
            session_pairs=normalized_session_pairs,
            prefix="pairwise",
            predicted_total_name="pairwise_links",
            reference_total_name="reference_pairwise_links",
        )
    )
    scores.update(
        _score_multiset_complete_tracks(
            predicted,
            reference,
            session_indices=normalized_complete_session_indices,
        )
    )
    return scores


def _normalize_session_pairs(
    session_pairs: Iterable[tuple[int, int]] | None,
) -> tuple[tuple[int, int], ...] | None:
    if session_pairs is None:
        return None
    return tuple((int(session_a), int(session_b)) for session_a, session_b in session_pairs)


def _normalize_complete_session_indices(
    session_indices: Sequence[int] | None,
) -> tuple[int, ...] | None:
    if session_indices is None:
        return None
    return tuple(int(session_index) for session_index in session_indices)


def _score_multiset_track_links(
    predicted: np.ndarray,
    reference: np.ndarray,
    *,
    session_pairs: Iterable[tuple[int, int]] | None,
    prefix: str,
    predicted_total_name: str,
    reference_total_name: str,
) -> dict[str, float | int]:
    return _score_identity_multisets(
        _track_link_counter(predicted, session_pairs=session_pairs),
        _track_link_counter(reference, session_pairs=session_pairs),
        prefix=prefix,
        predicted_total_name=predicted_total_name,
        reference_total_name=reference_total_name,
    )


def _score_multiset_complete_tracks(
    predicted: np.ndarray,
    reference: np.ndarray,
    *,
    session_indices: Sequence[int] | None,
) -> dict[str, float | int]:
    return _score_identity_multisets(
        _complete_track_counter(predicted, session_indices=session_indices),
        _complete_track_counter(reference, session_indices=session_indices),
        prefix="complete_track",
        predicted_total_name="complete_tracks",
        reference_total_name="reference_complete_tracks",
    )


def _score_identity_multisets(
    predicted: Counter[Any],
    reference: Counter[Any],
    *,
    prefix: str,
    predicted_total_name: str,
    reference_total_name: str,
) -> dict[str, float | int]:
    true_positives = int(sum((predicted & reference).values()))
    predicted_total = int(sum(predicted.values()))
    reference_total = int(sum(reference.values()))
    false_positives = predicted_total - true_positives
    false_negatives = reference_total - true_positives
    precision = _safe_ratio(true_positives, true_positives + false_positives)
    recall = _safe_ratio(true_positives, true_positives + false_negatives)
    f1 = _safe_ratio(2.0 * precision * recall, precision + recall)
    return {
        f"{prefix}_true_positives": true_positives,
        f"{prefix}_false_positives": false_positives,
        f"{prefix}_false_negatives": false_negatives,
        f"{prefix}_precision": precision,
        f"{prefix}_recall": recall,
        f"{prefix}_f1": f1,
        predicted_total_name: predicted_total,
        reference_total_name: reference_total,
    }


def _complete_track_counter(
    track_matrix: np.ndarray,
    *,
    session_indices: Sequence[int] | None,
) -> Counter[tuple[int, ...]]:
    selected_sessions = _selected_sessions(track_matrix, session_indices)
    counter: Counter[tuple[int, ...]] = Counter()
    for row in track_matrix:
        values = [row[session_index] for session_index in selected_sessions]
        if all(value is not None for value in values):
            counter[tuple(int(value) for value in values)] += 1
    return counter


def _track_link_counter(
    track_matrix: np.ndarray,
    *,
    session_pairs: Iterable[tuple[int, int]] | None,
) -> Counter[TrackLink]:
    counter: Counter[TrackLink] = Counter()
    for session_a, session_b in _session_pairs(track_matrix, session_pairs):
        for row in track_matrix:
            observation_a = row[session_a]
            observation_b = row[session_b]
            if observation_a is not None and observation_b is not None:
                counter[
                    (
                        int(session_a),
                        int(session_b),
                        int(observation_a),
                        int(observation_b),
                    )
                ] += 1
    return counter


def _selected_sessions(
    matrix: np.ndarray,
    session_indices: Sequence[int] | None,
) -> tuple[int, ...]:
    selected = (
        tuple(range(matrix.shape[1]))
        if session_indices is None
        else tuple(int(index) for index in session_indices)
    )
    if not selected:
        raise ValueError("At least one session must be selected")
    for session_index in selected:
        _validate_session_index(matrix, session_index)
    return selected


def _session_pairs(
    matrix: np.ndarray,
    session_pairs: Iterable[tuple[int, int]] | None,
) -> tuple[tuple[int, int], ...]:
    pairs = (
        tuple((index, index + 1) for index in range(max(0, matrix.shape[1] - 1)))
        if session_pairs is None
        else tuple((int(session_a), int(session_b)) for session_a, session_b in session_pairs)
    )
    for session_a, session_b in pairs:
        _validate_session_index(matrix, session_a)
        _validate_session_index(matrix, session_b)
        if session_a >= session_b:
            raise ValueError("session_pairs must point forward in time")
    return pairs


def _validate_session_index(matrix: np.ndarray, session_index: int) -> None:
    if session_index < 0 or session_index >= matrix.shape[1]:
        raise IndexError(
            f"session index {session_index} out of bounds for {matrix.shape[1]} sessions"
        )


def _validate_compatible_shapes(predicted: np.ndarray, reference: np.ndarray) -> None:
    if predicted.shape[1] != reference.shape[1]:
        raise ValueError(
            "Predicted and reference matrices must have the same number of sessions"
        )


def _safe_ratio(numerator: float, denominator: float) -> float:
    return 1.0 if denominator == 0 else float(numerator) / float(denominator)


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
