"""Track-level scoring helpers for longitudinal ROI identity matrices.

BayesCaTrack exposes a compatibility import path for PyRecEst's generic track
matrix helpers.  The public ``*_set`` helpers intentionally keep their historic
set-valued behavior, but the benchmark-facing scores below use multisets so that
duplicate predicted tracks are counted as extra predictions instead of being
silently collapsed before precision/recall are computed.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any, TypeAlias

import numpy as np
from pyrecest.utils.track_evaluation import (
    complete_track_set,
    normalize_track_matrix,
    pairwise_track_set,
    reference_fragment_counts,
    score_false_continuations,
    score_fragmentation,
    summarize_track_errors,
    summarize_tracks,
    track_lengths,
)

TrackLink: TypeAlias = tuple[int, int, int, int]

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


def score_complete_tracks(
    predicted_track_matrix: Any,
    reference_track_matrix: Any,
    *,
    session_indices: Sequence[int] | None = None,
) -> dict[str, float | int]:
    """Score exact complete-track recovery while preserving duplicates.

    The compatibility helper :func:`complete_track_set` still returns a set, but
    using that set for scoring loses multiplicity.  Duplicate predicted complete
    rows must therefore count as additional false positives.
    """

    predicted = normalize_track_matrix(predicted_track_matrix)
    reference = normalize_track_matrix(reference_track_matrix)
    _validate_compatible_shapes(predicted, reference)
    return _score_identity_counters(
        _complete_track_counter(predicted, session_indices=session_indices),
        _complete_track_counter(reference, session_indices=session_indices),
        prefix="complete_track",
        predicted_total_name="complete_tracks",
        reference_total_name="reference_complete_tracks",
    )


def score_track_links(
    predicted_track_matrix: Any,
    reference_track_matrix: Any,
    *,
    session_pairs: Iterable[tuple[int, int]] | None = None,
) -> dict[str, float | int]:
    """Score pairwise links with multiplicity-preserving counters."""

    predicted = normalize_track_matrix(predicted_track_matrix)
    reference = normalize_track_matrix(reference_track_matrix)
    _validate_compatible_shapes(predicted, reference)
    return _score_identity_counters(
        _track_pair_counter(predicted, session_pairs=session_pairs),
        _track_pair_counter(reference, session_pairs=session_pairs),
        prefix="track_link",
        predicted_total_name="track_links",
        reference_total_name="reference_track_links",
    )


def score_pairwise_tracks(
    predicted_track_matrix: Any,
    reference_track_matrix: Any,
    *,
    session_pairs: Iterable[tuple[int, int]] | None = None,
) -> dict[str, float | int]:
    """Score BayesCaTrack pairwise links while preserving duplicates."""

    predicted = normalize_track_matrix(predicted_track_matrix)
    reference = normalize_track_matrix(reference_track_matrix)
    _validate_compatible_shapes(predicted, reference)
    return _score_identity_counters(
        _track_pair_counter(predicted, session_pairs=session_pairs),
        _track_pair_counter(reference, session_pairs=session_pairs),
        prefix="pairwise",
        predicted_total_name="pairwise_links",
        reference_total_name="reference_pairwise_links",
    )


def score_track_matrices(
    predicted_track_matrix: Any,
    reference_track_matrix: Any,
    *,
    session_pairs: Iterable[tuple[int, int]] | None = None,
    complete_session_indices: Sequence[int] | None = None,
) -> dict[str, float | int]:
    """Return aggregate track metrics with duplicate-aware precision/recall."""

    predicted = normalize_track_matrix(predicted_track_matrix)
    reference = normalize_track_matrix(reference_track_matrix)
    _validate_compatible_shapes(predicted, reference)

    scores: dict[str, float | int] = {}
    scores.update(score_track_links(predicted, reference, session_pairs=session_pairs))
    scores.update(
        score_pairwise_tracks(predicted, reference, session_pairs=session_pairs)
    )
    scores.update(
        score_complete_tracks(
            predicted, reference, session_indices=complete_session_indices
        )
    )
    scores.update(score_false_continuations(predicted, reference, session_pairs=session_pairs))
    scores.update(score_fragmentation(predicted, reference))
    scores.update(summarize_tracks(predicted))

    error_scores = dict(
        summarize_track_errors(predicted, reference, session_pairs=session_pairs)
    )
    false_continuation_link_rate = error_scores.pop("false_continuation_rate", None)
    scores.update(error_scores)
    if false_continuation_link_rate is not None:
        scores["false_continuation_link_rate"] = false_continuation_link_rate
    return scores


def _complete_track_counter(
    track_matrix: Any, *, session_indices: Sequence[int] | None = None
) -> Counter[tuple[int, ...]]:
    matrix = normalize_track_matrix(track_matrix)
    selected_sessions = _selected_sessions(matrix, session_indices)
    tracks: Counter[tuple[int, ...]] = Counter()
    for row in matrix:
        values = [row[session_idx] for session_idx in selected_sessions]
        if all(value is not None for value in values):
            tracks[tuple(int(value) for value in values)] += 1
    return tracks


def _track_pair_counter(
    track_matrix: Any,
    *,
    session_pairs: Iterable[tuple[int, int]] | None = None,
) -> Counter[TrackLink]:
    matrix = normalize_track_matrix(track_matrix)
    links: Counter[TrackLink] = Counter()
    for session_a, session_b in _session_pairs(matrix, session_pairs):
        for row in matrix:
            obs_a = row[session_a]
            obs_b = row[session_b]
            if obs_a is not None and obs_b is not None:
                links[(int(session_a), int(session_b), int(obs_a), int(obs_b))] += 1
    return links


def _score_identity_counters(
    predicted: Counter[Any],
    reference: Counter[Any],
    *,
    prefix: str,
    predicted_total_name: str,
    reference_total_name: str,
) -> dict[str, float | int]:
    true_positives = sum((predicted & reference).values())
    false_positives = sum((predicted - reference).values())
    false_negatives = sum((reference - predicted).values())
    precision = _safe_ratio(true_positives, true_positives + false_positives)
    recall = _safe_ratio(true_positives, true_positives + false_negatives)
    f1 = _safe_ratio(2.0 * precision * recall, precision + recall)
    return {
        f"{prefix}_true_positives": int(true_positives),
        f"{prefix}_false_positives": int(false_positives),
        f"{prefix}_false_negatives": int(false_negatives),
        f"{prefix}_precision": precision,
        f"{prefix}_recall": recall,
        f"{prefix}_f1": f1,
        predicted_total_name: int(sum(predicted.values())),
        reference_total_name: int(sum(reference.values())),
    }


def _selected_sessions(
    matrix: np.ndarray, session_indices: Sequence[int] | None
) -> list[int]:
    selected = (
        list(range(matrix.shape[1]))
        if session_indices is None
        else [int(index) for index in session_indices]
    )
    for session_idx in selected:
        _validate_session_index(matrix, session_idx)
    return selected


def _session_pairs(
    matrix: np.ndarray, session_pairs: Iterable[tuple[int, int]] | None
) -> tuple[tuple[int, int], ...]:
    pairs = (
        tuple((idx, idx + 1) for idx in range(max(0, matrix.shape[1] - 1)))
        if session_pairs is None
        else tuple((int(a), int(b)) for a, b in session_pairs)
    )
    for session_a, session_b in pairs:
        _validate_session_index(matrix, session_a)
        _validate_session_index(matrix, session_b)
        if session_a >= session_b:
            raise ValueError("session_pairs must point forward in time")
    return pairs


def _validate_session_index(matrix: np.ndarray, session_idx: int) -> None:
    if session_idx < 0 or session_idx >= matrix.shape[1]:
        raise IndexError(
            f"session index {session_idx} out of bounds for {matrix.shape[1]} sessions"
        )


def _validate_compatible_shapes(predicted: np.ndarray, reference: np.ndarray) -> None:
    if predicted.shape[1] != reference.shape[1]:
        raise ValueError(
            "Predicted and reference matrices must have the same number of sessions"
        )


def _safe_ratio(numerator: float, denominator: float) -> float:
    return 1.0 if denominator == 0 else float(numerator) / float(denominator)
