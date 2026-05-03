"""Track-level error ledgers for longitudinal ROI identity matrices."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

import numpy as np

from .complete_track_scores import normalize_track_matrix, pairwise_track_set

Observation = tuple[int, int]
Link = tuple[int, int, int, int]

__all__ = ("summarize_track_errors", "track_error_ledger")


def track_error_ledger(
    predicted_track_matrix: Any,
    reference_track_matrix: Any,
    *,
    session_pairs: Iterable[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Return detailed track-level errors between predicted and reference identities.

    The returned ledger separates reference-track errors, predicted-track errors, and
    link-level false-continuation/missed-link errors. All identifiers are matrix row
    indices and Suite2p ROI indices after :func:`normalize_track_matrix` parsing.
    """

    predicted = normalize_track_matrix(predicted_track_matrix)
    reference = normalize_track_matrix(reference_track_matrix)
    _validate_compatible_shapes(predicted, reference)
    pairs = _session_pairs(predicted, session_pairs)

    predicted_observations = _observations_by_track(predicted)
    reference_observations = _observations_by_track(reference)
    predicted_lookup, predicted_duplicates = _observation_lookup(predicted_observations)
    reference_lookup, reference_duplicates = _observation_lookup(reference_observations)

    predicted_links = pairwise_track_set(predicted, session_pairs=pairs)
    reference_links = pairwise_track_set(reference, session_pairs=pairs)
    predicted_rows = _predicted_track_rows(
        predicted_observations, reference_lookup, pairs
    )
    reference_rows = _reference_track_rows(
        reference_observations, predicted_lookup, pairs
    )
    false_links = sorted(predicted_links.difference(reference_links))
    missed_links = sorted(reference_links.difference(predicted_links))
    link_rows = [
        _false_continuation_row(link, predicted_lookup, reference_lookup)
        for link in false_links
    ] + [
        _missed_reference_link_row(link, predicted_lookup, reference_lookup)
        for link in missed_links
    ]
    duplicate_rows = _duplicate_rows(
        "predicted", predicted_duplicates
    ) + _duplicate_rows("reference", reference_duplicates)
    summary = _summary_rows(
        predicted_rows,
        reference_rows,
        false_links=false_links,
        missed_links=missed_links,
        predicted_links=predicted_links,
        reference_links=reference_links,
        predicted_duplicates=predicted_duplicates,
        reference_duplicates=reference_duplicates,
    )
    return {
        "summary": summary,
        "predicted_tracks": predicted_rows,
        "reference_tracks": reference_rows,
        "link_errors": link_rows,
        "duplicate_observations": duplicate_rows,
    }


def summarize_track_errors(
    predicted_track_matrix: Any,
    reference_track_matrix: Any,
    *,
    session_pairs: Iterable[tuple[int, int]] | None = None,
) -> dict[str, int | float]:
    """Return aggregate track-level error metrics for benchmark CSV tables."""

    return track_error_ledger(
        predicted_track_matrix, reference_track_matrix, session_pairs=session_pairs
    )["summary"]


def _predicted_track_rows(
    predicted_observations: list[list[Observation]],
    reference_lookup: dict[Observation, int],
    session_pairs: tuple[tuple[int, int], ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for predicted_track_id, observations in enumerate(predicted_observations):
        reference_ids = [
            reference_lookup[observation]
            for observation in observations
            if observation in reference_lookup
        ]
        reference_counts = Counter(reference_ids)
        unknown_observations = [
            observation
            for observation in observations
            if observation not in reference_lookup
        ]
        false_links = _predicted_track_false_links(
            observations, reference_lookup, session_pairs
        )
        identity_switches = _identity_switches(observations, reference_lookup)
        rows.append(
            {
                "track_id": int(predicted_track_id),
                "category": _predicted_track_category(
                    reference_counts, unknown_observations, identity_switches
                ),
                "length": len(observations),
                "reference_track_ids": sorted(
                    int(track_id) for track_id in reference_counts
                ),
                "dominant_reference_track_id": _dominant_counter_key(reference_counts),
                "matched_reference_observations": int(sum(reference_counts.values())),
                "unreferenced_observations": len(unknown_observations),
                "identity_switches": identity_switches,
                "false_continuation_links": false_links,
            }
        )
    return rows


def _reference_track_rows(
    reference_observations: list[list[Observation]],
    predicted_lookup: dict[Observation, int],
    session_pairs: tuple[tuple[int, int], ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for reference_track_id, observations in enumerate(reference_observations):
        predicted_ids = [
            predicted_lookup[observation]
            for observation in observations
            if observation in predicted_lookup
        ]
        predicted_counts = Counter(predicted_ids)
        missed_observations = [
            observation
            for observation in observations
            if observation not in predicted_lookup
        ]
        missed_links = _reference_track_missed_links(
            observations, predicted_lookup, session_pairs
        )
        rows.append(
            {
                "track_id": int(reference_track_id),
                "category": _reference_track_category(
                    predicted_counts, missed_observations
                ),
                "length": len(observations),
                "predicted_track_ids": sorted(
                    int(track_id) for track_id in predicted_counts
                ),
                "dominant_predicted_track_id": _dominant_counter_key(predicted_counts),
                "matched_predicted_observations": int(sum(predicted_counts.values())),
                "missed_observations": len(missed_observations),
                "fragment_count": max(0, len(predicted_counts) - 1),
                "missed_reference_links": missed_links,
            }
        )
    return rows


def _summary_rows(
    predicted_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
    *,
    false_links: list[Link],
    missed_links: list[Link],
    predicted_links: set[Link],
    reference_links: set[Link],
    predicted_duplicates: list[tuple[Observation, int, int]],
    reference_duplicates: list[tuple[Observation, int, int]],
) -> dict[str, int | float]:
    reference_tracks = len(reference_rows)
    predicted_tracks = len(predicted_rows)
    fragmented_reference_tracks = sum(
        1 for row in reference_rows if row["fragment_count"] > 0
    )
    missed_reference_tracks = sum(
        1 for row in reference_rows if row["category"] == "missed"
    )
    partial_reference_tracks = sum(
        1
        for row in reference_rows
        if row["category"] in {"partial", "fragmented_partial"}
    )
    mixed_identity_tracks = sum(
        1 for row in predicted_rows if row["category"] == "mixed_identity"
    )
    spurious_tracks = sum(1 for row in predicted_rows if row["category"] == "spurious")
    false_continuation_links = len(false_links)
    missed_reference_links = len(missed_links)
    return {
        "identity_switches": int(
            sum(row["identity_switches"] for row in predicted_rows)
        ),
        "mixed_identity_tracks": int(mixed_identity_tracks),
        "spurious_tracks": int(spurious_tracks),
        "spurious_predicted_observations": int(
            sum(row["unreferenced_observations"] for row in predicted_rows)
        ),
        "false_continuation_links": int(false_continuation_links),
        "false_continuation_rate": _safe_error_rate(
            false_continuation_links, len(predicted_links)
        ),
        "missed_reference_links": int(missed_reference_links),
        "missed_reference_link_rate": _safe_error_rate(
            missed_reference_links, len(reference_links)
        ),
        "fragmented_reference_tracks": int(fragmented_reference_tracks),
        "track_fragmentations": int(
            sum(row["fragment_count"] for row in reference_rows)
        ),
        "track_fragmentation_rate": _safe_error_rate(
            fragmented_reference_tracks, reference_tracks
        ),
        "missed_reference_tracks": int(missed_reference_tracks),
        "missed_reference_track_rate": _safe_error_rate(
            missed_reference_tracks, reference_tracks
        ),
        "partial_reference_tracks": int(partial_reference_tracks),
        "missed_reference_observations": int(
            sum(row["missed_observations"] for row in reference_rows)
        ),
        "predicted_duplicate_observations": len(predicted_duplicates),
        "reference_duplicate_observations": len(reference_duplicates),
        "predicted_tracks_with_errors": int(
            sum(1 for row in predicted_rows if row["category"] != "single_identity")
        ),
        "reference_tracks_with_errors": int(
            sum(1 for row in reference_rows if row["category"] != "recovered")
        ),
        "predicted_track_error_rate": _safe_error_rate(
            sum(1 for row in predicted_rows if row["category"] != "single_identity"),
            predicted_tracks,
        ),
        "reference_track_error_rate": _safe_error_rate(
            sum(1 for row in reference_rows if row["category"] != "recovered"),
            reference_tracks,
        ),
    }


def _false_continuation_row(
    link: Link,
    predicted_lookup: dict[Observation, int],
    reference_lookup: dict[Observation, int],
) -> dict[str, Any]:
    session_a, session_b, roi_a, roi_b = link
    obs_a = (session_a, roi_a)
    obs_b = (session_b, roi_b)
    reference_track_a = reference_lookup.get(obs_a)
    reference_track_b = reference_lookup.get(obs_b)
    if reference_track_a is None or reference_track_b is None:
        reason = "unreferenced_observation"
    elif reference_track_a != reference_track_b:
        reason = "different_reference_tracks"
    else:
        reason = "missing_reference_link"
    return {
        "error_type": "false_continuation",
        "reason": reason,
        "session_a": int(session_a),
        "session_b": int(session_b),
        "roi_a": int(roi_a),
        "roi_b": int(roi_b),
        "predicted_track_id": _same_or_none(
            predicted_lookup.get(obs_a), predicted_lookup.get(obs_b)
        ),
        "reference_track_a": _optional_int(reference_track_a),
        "reference_track_b": _optional_int(reference_track_b),
    }


def _missed_reference_link_row(
    link: Link,
    predicted_lookup: dict[Observation, int],
    reference_lookup: dict[Observation, int],
) -> dict[str, Any]:
    session_a, session_b, roi_a, roi_b = link
    obs_a = (session_a, roi_a)
    obs_b = (session_b, roi_b)
    predicted_track_a = predicted_lookup.get(obs_a)
    predicted_track_b = predicted_lookup.get(obs_b)
    if predicted_track_a is None or predicted_track_b is None:
        reason = "missing_prediction_observation"
    elif predicted_track_a != predicted_track_b:
        reason = "split_across_predicted_tracks"
    else:
        reason = "missing_predicted_link"
    return {
        "error_type": "missed_reference_link",
        "reason": reason,
        "session_a": int(session_a),
        "session_b": int(session_b),
        "roi_a": int(roi_a),
        "roi_b": int(roi_b),
        "reference_track_id": _same_or_none(
            reference_lookup.get(obs_a), reference_lookup.get(obs_b)
        ),
        "predicted_track_a": _optional_int(predicted_track_a),
        "predicted_track_b": _optional_int(predicted_track_b),
    }


def _predicted_track_false_links(
    observations: list[Observation],
    reference_lookup: dict[Observation, int],
    session_pairs: tuple[tuple[int, int], ...],
) -> int:
    by_session = dict(observations)
    false_links = 0
    for session_a, session_b in session_pairs:
        if session_a not in by_session or session_b not in by_session:
            continue
        reference_a = reference_lookup.get((session_a, by_session[session_a]))
        reference_b = reference_lookup.get((session_b, by_session[session_b]))
        if reference_a is None or reference_b is None or reference_a != reference_b:
            false_links += 1
    return false_links


def _reference_track_missed_links(
    observations: list[Observation],
    predicted_lookup: dict[Observation, int],
    session_pairs: tuple[tuple[int, int], ...],
) -> int:
    by_session = dict(observations)
    missed_links = 0
    for session_a, session_b in session_pairs:
        if session_a not in by_session or session_b not in by_session:
            continue
        predicted_a = predicted_lookup.get((session_a, by_session[session_a]))
        predicted_b = predicted_lookup.get((session_b, by_session[session_b]))
        if predicted_a is None or predicted_b is None or predicted_a != predicted_b:
            missed_links += 1
    return missed_links


def _identity_switches(
    observations: list[Observation], reference_lookup: dict[Observation, int]
) -> int:
    last_reference_id: int | None = None
    switches = 0
    for observation in sorted(observations):
        reference_id = reference_lookup.get(observation)
        if reference_id is None:
            continue
        if last_reference_id is not None and reference_id != last_reference_id:
            switches += 1
        last_reference_id = reference_id
    return switches


def _predicted_track_category(
    reference_counts: Counter[int],
    unknown_observations: list[Observation],
    identity_switches: int,
) -> str:
    if not reference_counts:
        return "spurious"
    if identity_switches > 0 or len(reference_counts) > 1:
        return "mixed_identity"
    if unknown_observations:
        return "single_identity_with_unreferenced_observations"
    return "single_identity"


def _reference_track_category(
    predicted_counts: Counter[int],
    missed_observations: list[Observation],
) -> str:
    if not predicted_counts:
        return "missed"
    fragmented = len(predicted_counts) > 1
    partial = bool(missed_observations)
    if fragmented and partial:
        return "fragmented_partial"
    if fragmented:
        return "fragmented"
    if partial:
        return "partial"
    return "recovered"


def _observations_by_track(matrix: np.ndarray) -> list[list[Observation]]:
    observations: list[list[Observation]] = []
    for row in matrix:
        track_observations = [
            (int(session_idx), int(value))
            for session_idx, value in enumerate(row)
            if value is not None
        ]
        observations.append(track_observations)
    return observations


def _observation_lookup(
    observations_by_track: list[list[Observation]],
) -> tuple[dict[Observation, int], list[tuple[Observation, int, int]]]:
    lookup: dict[Observation, int] = {}
    duplicates: list[tuple[Observation, int, int]] = []
    for track_id, observations in enumerate(observations_by_track):
        for observation in observations:
            first_track_id = lookup.get(observation)
            if first_track_id is None:
                lookup[observation] = int(track_id)
            else:
                duplicates.append((observation, int(first_track_id), int(track_id)))
    return lookup, duplicates


def _duplicate_rows(
    kind: str, duplicates: list[tuple[Observation, int, int]]
) -> list[dict[str, int | str]]:
    return [
        {
            "matrix": kind,
            "session": int(observation[0]),
            "roi": int(observation[1]),
            "first_track_id": int(first_track_id),
            "duplicate_track_id": int(duplicate_track_id),
        }
        for observation, first_track_id, duplicate_track_id in duplicates
    ]


def _session_pairs(
    matrix: np.ndarray, session_pairs: Iterable[tuple[int, int]] | None
) -> tuple[tuple[int, int], ...]:
    if session_pairs is None:
        return tuple(
            (session_idx, session_idx + 1)
            for session_idx in range(max(0, matrix.shape[1] - 1))
        )
    pairs = tuple(
        (int(session_a), int(session_b)) for session_a, session_b in session_pairs
    )
    for session_a, session_b in pairs:
        if (
            session_a < 0
            or session_b < 0
            or session_a >= matrix.shape[1]
            or session_b >= matrix.shape[1]
        ):
            raise IndexError(
                f"session pair {(session_a, session_b)!r} is out of bounds for {matrix.shape[1]} sessions"
            )
        if session_a >= session_b:
            raise ValueError("session_pairs must point forward in time")
    return pairs


def _validate_compatible_shapes(predicted: np.ndarray, reference: np.ndarray) -> None:
    if predicted.shape[1] != reference.shape[1]:
        raise ValueError(
            "Predicted and reference matrices must have the same number of sessions"
        )


def _dominant_counter_key(counter: Counter[int]) -> int | None:
    if not counter:
        return None
    return int(max(counter, key=lambda key: (counter[key], -key)))


def _same_or_none(left: int | None, right: int | None) -> int | None:
    if left is None or right is None or left != right:
        return None
    return int(left)


def _optional_int(value: int | None) -> int | None:
    return None if value is None else int(value)


def _safe_error_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)
