"""Pairwise edge-ranking diagnostics for manual-GT ROI links."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

import numpy as np

ScoreDirection = Literal["cost", "similarity"]

DEFAULT_HIT_KS = (1, 3, 5, 10)
DEFAULT_GROUP_KEYS = ("subject", "session_a", "session_b", "session_gap", "score_name")

SIMILARITY_SCORE_NAMES = frozenset(
    {
        "iou",
        "registered_iou",
        "mask_iou",
        "mask_cosine_similarity",
        "covariance_shape_similarity",
        "activity_correlation",
        "activity_similarity",
        "match_probability",
        "p_match",
        "p_same",
    }
)


def rank_labeled_edges(
    labels: Any,
    score_matrices: Mapping[str, Any],
    *,
    reference_roi_indices: Any,
    measurement_roi_indices: Any,
    score_directions: Mapping[str, ScoreDirection] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> list[dict[str, float | int | str]]:
    """Rank each positive label against row and column alternatives.

    Parameters
    ----------
    labels
        Binary ``(n_reference, n_measurement)`` matrix in loaded-ROI coordinates.
    score_matrices
        Named score or cost matrices with the same shape as ``labels``.
    reference_roi_indices, measurement_roi_indices
        Original Suite2p ROI indices corresponding to label rows and columns.
    score_directions
        Optional map declaring whether lower values (``"cost"``) or higher values
        (``"similarity"``) are better for each score. Unspecified names are
        inferred conservatively from their names.
    metadata
        Scalar fields copied into every output row, for example subject and
        session identifiers.

    Returns
    -------
    list of dict
        One row per positive GT edge per score matrix. Positive margins mean the
        GT edge beats the best same-row or same-column false competitor.
    """

    label_matrix = _as_label_matrix(labels)
    reference_indices = np.asarray(reference_roi_indices, dtype=int).reshape(-1)
    measurement_indices = np.asarray(measurement_roi_indices, dtype=int).reshape(-1)
    if label_matrix.shape != (reference_indices.size, measurement_indices.size):
        raise ValueError(
            "labels must have shape (len(reference_roi_indices), len(measurement_roi_indices))"
        )

    matrices = _validated_score_matrices(score_matrices, label_matrix.shape)
    directions = _score_directions_for(matrices, score_directions)
    base_metadata = dict(metadata or {})
    rows: list[dict[str, float | int | str]] = []

    positive_positions = np.argwhere(label_matrix)
    for row_index, column_index in positive_positions:
        reference_roi_index = int(reference_indices[row_index])
        measurement_roi_index = int(measurement_indices[column_index])
        for score_name, matrix in matrices.items():
            direction = directions[score_name]
            row_details = _rank_details(matrix[row_index, :], int(column_index), direction)
            column_details = _rank_details(
                matrix[:, column_index], int(row_index), direction
            )
            output_row: dict[str, float | int | str] = {
                **base_metadata,
                "reference_roi_index": reference_roi_index,
                "measurement_roi_index": measurement_roi_index,
                "score_name": score_name,
                "score_direction": direction,
                "edge_present": 1,
                "missing_reason": "",
                "true_score": _float_or_nan(matrix[row_index, column_index]),
                "true_is_finite": int(row_details["true_is_finite"]),
                "row_rank": int(row_details["rank"]),
                "column_rank": int(column_details["rank"]),
                "row_better_count": int(row_details["better_count"]),
                "column_better_count": int(column_details["better_count"]),
                "row_tie_count": int(row_details["tie_count"]),
                "column_tie_count": int(column_details["tie_count"]),
                "row_candidate_count": int(row_details["candidate_count"]),
                "column_candidate_count": int(column_details["candidate_count"]),
                "row_finite_candidate_count": int(row_details["finite_candidate_count"]),
                "column_finite_candidate_count": int(
                    column_details["finite_candidate_count"]
                ),
                "best_false_row_score": _float_or_nan(
                    row_details["best_false_score"]
                ),
                "best_false_column_score": _float_or_nan(
                    column_details["best_false_score"]
                ),
                "best_false_row_roi_index": _roi_index_or_minus_one(
                    measurement_indices, row_details["best_false_index"]
                ),
                "best_false_column_roi_index": _roi_index_or_minus_one(
                    reference_indices, column_details["best_false_index"]
                ),
                "row_margin": _float_or_nan(row_details["margin"]),
                "column_margin": _float_or_nan(column_details["margin"]),
            }
            rows.append(output_row)
    return rows


def missing_reference_edge_rows(
    reference_matches: Iterable[tuple[int, int]],
    *,
    reference_roi_indices: Any,
    measurement_roi_indices: Any,
    score_names: Sequence[str],
    score_directions: Mapping[str, ScoreDirection] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> list[dict[str, float | int | str]]:
    """Return diagnostic rows for manual-GT edges absent from the candidate matrix."""

    reference_indices = {int(value) for value in np.asarray(reference_roi_indices, dtype=int).reshape(-1)}
    measurement_indices = {int(value) for value in np.asarray(measurement_roi_indices, dtype=int).reshape(-1)}
    base_metadata = dict(metadata or {})
    directions = {
        score_name: _score_direction(score_name, score_directions)
        for score_name in score_names
    }
    rows: list[dict[str, float | int | str]] = []
    seen: set[tuple[int, int, str]] = set()
    for reference_roi_index, measurement_roi_index in reference_matches:
        reference_roi_index = int(reference_roi_index)
        measurement_roi_index = int(measurement_roi_index)
        reference_present = reference_roi_index in reference_indices
        measurement_present = measurement_roi_index in measurement_indices
        if reference_present and measurement_present:
            continue
        if not reference_present and not measurement_present:
            reason = "both_rois_missing"
        elif not reference_present:
            reason = "reference_roi_missing"
        else:
            reason = "measurement_roi_missing"
        for score_name in score_names:
            key = (reference_roi_index, measurement_roi_index, score_name)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    **base_metadata,
                    "reference_roi_index": reference_roi_index,
                    "measurement_roi_index": measurement_roi_index,
                    "score_name": score_name,
                    "score_direction": directions[score_name],
                    "edge_present": 0,
                    "missing_reason": reason,
                    "true_score": np.nan,
                    "true_is_finite": 0,
                    "row_rank": -1,
                    "column_rank": -1,
                    "row_better_count": -1,
                    "column_better_count": -1,
                    "row_tie_count": -1,
                    "column_tie_count": -1,
                    "row_candidate_count": -1,
                    "column_candidate_count": -1,
                    "row_finite_candidate_count": -1,
                    "column_finite_candidate_count": -1,
                    "best_false_row_score": np.nan,
                    "best_false_column_score": np.nan,
                    "best_false_row_roi_index": -1,
                    "best_false_column_roi_index": -1,
                    "row_margin": np.nan,
                    "column_margin": np.nan,
                }
            )
    return rows


def summarize_edge_ranking_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_keys: Sequence[str] = DEFAULT_GROUP_KEYS,
    hit_ks: Sequence[int] = DEFAULT_HIT_KS,
) -> list[dict[str, float | int | str]]:
    """Aggregate edge-ranking rows into hit-rate and margin summaries.

    Hit rates use all GT rows in the group as the denominator, including missing
    candidate edges. ``*_present`` variants use only present, finite true edges.
    """

    groups: "OrderedDict[tuple[Any, ...], list[Mapping[str, Any]]]" = OrderedDict()
    for row in rows:
        key = tuple(row.get(group_key, "") for group_key in group_keys)
        groups.setdefault(key, []).append(row)

    hit_ks = tuple(int(value) for value in hit_ks)
    summaries: list[dict[str, float | int | str]] = []
    for key, group_rows in groups.items():
        summary: dict[str, float | int | str] = dict(zip(group_keys, key))
        present_rows = [_row for _row in group_rows if _truthy_int(_row.get("edge_present", 0))]
        finite_rows = [
            _row for _row in present_rows if _truthy_int(_row.get("true_is_finite", 0))
        ]
        n_gt_edges = len(group_rows)
        n_present = len(present_rows)
        n_finite = len(finite_rows)
        summary.update(
            {
                "gt_edges": int(n_gt_edges),
                "present_edges": int(n_present),
                "missing_edges": int(n_gt_edges - n_present),
                "finite_true_edges": int(n_finite),
                "median_row_rank": _median_of(finite_rows, "row_rank"),
                "median_column_rank": _median_of(finite_rows, "column_rank"),
                "median_row_margin": _median_of(finite_rows, "row_margin"),
                "median_column_margin": _median_of(finite_rows, "column_margin"),
                "mean_row_margin": _mean_of(finite_rows, "row_margin"),
                "mean_column_margin": _mean_of(finite_rows, "column_margin"),
                "row_positive_margin_rate": _rate(
                    finite_rows, lambda row: _finite_float(row.get("row_margin")) > 0.0
                ),
                "column_positive_margin_rate": _rate(
                    finite_rows,
                    lambda row: _finite_float(row.get("column_margin")) > 0.0,
                ),
                "mutual_top1_rate": _rate(
                    group_rows,
                    lambda row: _truthy_int(row.get("edge_present", 0))
                    and _truthy_int(row.get("true_is_finite", 0))
                    and _safe_int(row.get("row_rank"), default=0) <= 1
                    and _safe_int(row.get("column_rank"), default=0) <= 1,
                ),
                "mutual_top1_rate_present": _rate(
                    finite_rows,
                    lambda row: _safe_int(row.get("row_rank"), default=0) <= 1
                    and _safe_int(row.get("column_rank"), default=0) <= 1,
                ),
            }
        )
        for k_value in hit_ks:
            summary[f"row_hit_at_{k_value}"] = _rate(
                group_rows,
                lambda row, k_value=k_value: _truthy_int(row.get("edge_present", 0))
                and _truthy_int(row.get("true_is_finite", 0))
                and _safe_int(row.get("row_rank"), default=0) <= k_value,
            )
            summary[f"column_hit_at_{k_value}"] = _rate(
                group_rows,
                lambda row, k_value=k_value: _truthy_int(row.get("edge_present", 0))
                and _truthy_int(row.get("true_is_finite", 0))
                and _safe_int(row.get("column_rank"), default=0) <= k_value,
            )
            summary[f"row_hit_at_{k_value}_present"] = _rate(
                finite_rows,
                lambda row, k_value=k_value: _safe_int(row.get("row_rank"), default=0)
                <= k_value,
            )
            summary[f"column_hit_at_{k_value}_present"] = _rate(
                finite_rows,
                lambda row, k_value=k_value: _safe_int(
                    row.get("column_rank"), default=0
                )
                <= k_value,
            )
        summaries.append(summary)
    return summaries


def score_matrices_from_feature_tensor(
    features: Any,
    feature_names: Sequence[str],
) -> dict[str, np.ndarray]:
    """Return ``{feature_name: feature_plane}`` from an ``(..., n_features)`` tensor."""

    feature_array = np.asarray(features, dtype=float)
    names = tuple(feature_names)
    if feature_array.ndim != 3:
        raise ValueError("features must have shape (n_reference, n_measurement, n_features)")
    if feature_array.shape[-1] != len(names):
        raise ValueError("feature_names length must match the last feature dimension")
    return {
        feature_name: np.asarray(feature_array[:, :, feature_index], dtype=float)
        for feature_index, feature_name in enumerate(names)
    }


def infer_score_direction(score_name: str) -> ScoreDirection:
    """Infer whether a score name is cost-like or similarity-like."""

    normalized = str(score_name).strip().lower()
    if normalized in SIMILARITY_SCORE_NAMES:
        return "similarity"
    if normalized.endswith("_similarity") or normalized.endswith("_correlation"):
        return "similarity"
    if normalized.startswith("p_") or normalized.endswith("_probability"):
        return "similarity"
    return "cost"


def _as_label_matrix(labels: Any) -> np.ndarray:
    label_matrix = np.asarray(labels)
    if label_matrix.ndim != 2:
        raise ValueError("labels must be a two-dimensional matrix")
    return label_matrix.astype(bool)


def _validated_score_matrices(
    score_matrices: Mapping[str, Any], shape: tuple[int, int]
) -> dict[str, np.ndarray]:
    matrices: dict[str, np.ndarray] = {}
    for score_name, score_values in score_matrices.items():
        matrix = np.asarray(score_values, dtype=float)
        if matrix.shape != shape:
            raise ValueError(
                f"Score matrix {score_name!r} has shape {matrix.shape}, expected {shape}"
            )
        matrices[str(score_name)] = matrix
    if not matrices:
        raise ValueError("At least one score matrix is required")
    return matrices


def _score_directions_for(
    score_matrices: Mapping[str, Any],
    score_directions: Mapping[str, ScoreDirection] | None,
) -> dict[str, ScoreDirection]:
    return {
        score_name: _score_direction(score_name, score_directions)
        for score_name in score_matrices
    }


def _score_direction(
    score_name: str,
    score_directions: Mapping[str, ScoreDirection] | None,
) -> ScoreDirection:
    if score_directions and score_name in score_directions:
        direction = score_directions[score_name]
        if direction not in {"cost", "similarity"}:
            raise ValueError(
                f"score_directions[{score_name!r}] must be 'cost' or 'similarity'"
            )
        return direction
    return infer_score_direction(score_name)


def _rank_details(
    values: np.ndarray,
    true_index: int,
    direction: ScoreDirection,
) -> dict[str, float | int | bool]:
    values = np.asarray(values, dtype=float).reshape(-1)
    if true_index < 0 or true_index >= values.size:
        raise IndexError("true_index is out of bounds for candidate vector")

    finite = np.isfinite(values)
    true_value = float(values[true_index])
    true_is_finite = bool(np.isfinite(true_value))
    finite_candidate_count = int(np.count_nonzero(finite))

    false_mask = finite.copy()
    false_mask[true_index] = False
    false_indices = np.flatnonzero(false_mask)

    if true_is_finite:
        if direction == "cost":
            better_count = int(np.count_nonzero(finite & (values < true_value)))
            tie_count = int(np.count_nonzero(finite & (values == true_value))) - 1
        else:
            better_count = int(np.count_nonzero(finite & (values > true_value)))
            tie_count = int(np.count_nonzero(finite & (values == true_value))) - 1
        rank = 1 + better_count
    else:
        better_count = finite_candidate_count
        tie_count = 0
        rank = finite_candidate_count + 1

    if false_indices.size:
        if direction == "cost":
            best_false_local = int(np.argmin(values[false_indices]))
        else:
            best_false_local = int(np.argmax(values[false_indices]))
        best_false_index = int(false_indices[best_false_local])
        best_false_score = float(values[best_false_index])
        if true_is_finite:
            margin = (
                best_false_score - true_value
                if direction == "cost"
                else true_value - best_false_score
            )
        else:
            margin = np.nan
    else:
        best_false_index = -1
        best_false_score = np.nan
        margin = np.nan

    return {
        "rank": int(rank),
        "better_count": int(better_count),
        "tie_count": int(max(tie_count, 0)),
        "candidate_count": int(values.size),
        "finite_candidate_count": int(finite_candidate_count),
        "best_false_index": int(best_false_index),
        "best_false_score": float(best_false_score),
        "margin": float(margin),
        "true_is_finite": bool(true_is_finite),
    }


def _roi_index_or_minus_one(indices: np.ndarray, position: Any) -> int:
    position_int = _safe_int(position, default=-1)
    if position_int < 0 or position_int >= len(indices):
        return -1
    return int(indices[position_int])


def _truthy_int(value: Any) -> bool:
    return _safe_int(value, default=0) != 0


def _safe_int(value: Any, *, default: int) -> int:
    try:
        if value is None:
            return int(default)
        if isinstance(value, float) and not np.isfinite(value):
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _finite_float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return np.nan
    return numeric if np.isfinite(numeric) else np.nan


def _float_or_nan(value: Any) -> float:
    return _finite_float(value)


def _finite_values(rows: Sequence[Mapping[str, Any]], field_name: str) -> np.ndarray:
    values = np.asarray([_finite_float(row.get(field_name)) for row in rows], dtype=float)
    return values[np.isfinite(values)]


def _median_of(rows: Sequence[Mapping[str, Any]], field_name: str) -> float:
    values = _finite_values(rows, field_name)
    if values.size == 0:
        return float("nan")
    return float(np.median(values))


def _mean_of(rows: Sequence[Mapping[str, Any]], field_name: str) -> float:
    values = _finite_values(rows, field_name)
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def _rate(rows: Sequence[Mapping[str, Any]], predicate: Any) -> float:
    if not rows:
        return float("nan")
    return float(sum(1 for row in rows if predicate(row)) / len(rows))


__all__ = [
    "DEFAULT_GROUP_KEYS",
    "DEFAULT_HIT_KS",
    "SIMILARITY_SCORE_NAMES",
    "ScoreDirection",
    "infer_score_direction",
    "missing_reference_edge_rows",
    "rank_labeled_edges",
    "score_matrices_from_feature_tensor",
    "summarize_edge_ranking_rows",
]
