"""Calibration diagnostics for pairwise association probabilities."""

# pylint: disable=undefined-all-variable

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias

import numpy as np

CalibrationBinRow: TypeAlias = dict[str, float | int | None]

__all__ = (
    "CalibrationBinRow",
    "brier_score",
    "calibration_summary",
    "expected_calibration_error",
    "format_reliability_bin_table",
    "maximum_calibration_error",
    "reliability_bin_table",
)


def reliability_bin_table(
    probabilities: Any,
    labels: Any,
    *,
    n_bins: int = 10,
    include_empty_bins: bool = True,
) -> list[CalibrationBinRow]:
    """Return equal-width reliability bins for predicted match probabilities."""

    probabilities, labels = _validate_probability_label_inputs(probabilities, labels)
    n_bins = _validate_n_bins(n_bins)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    assignments = np.searchsorted(edges, probabilities, side="right") - 1
    assignments = np.clip(assignments, 0, n_bins - 1)
    n_examples = int(probabilities.shape[0])

    rows: list[CalibrationBinRow] = []
    for bin_index in range(n_bins):
        in_bin = assignments == bin_index
        count = int(np.count_nonzero(in_bin))
        if count == 0:
            if include_empty_bins:
                rows.append(_empty_bin_row(bin_index, edges))
            continue

        bin_probabilities = probabilities[in_bin]
        bin_labels = labels[in_bin]
        positive_count = int(np.sum(bin_labels))
        mean_probability = float(np.mean(bin_probabilities))
        empirical_probability = float(np.mean(bin_labels))
        signed_error = empirical_probability - mean_probability
        rows.append(
            {
                "bin_index": int(bin_index),
                "probability_lower": float(edges[bin_index]),
                "probability_upper": float(edges[bin_index + 1]),
                "count": count,
                "positive_count": positive_count,
                "negative_count": int(count - positive_count),
                "mean_predicted_probability": mean_probability,
                "empirical_positive_rate": empirical_probability,
                "signed_calibration_error": float(signed_error),
                "absolute_calibration_error": float(abs(signed_error)),
                "bin_brier_score": float(np.mean((bin_probabilities - bin_labels) ** 2)),
                "weight": float(count / n_examples),
            }
        )
    return rows


def expected_calibration_error(probabilities: Any, labels: Any, *, n_bins: int = 10) -> float:
    """Return equal-width expected calibration error for binary probabilities."""

    rows = reliability_bin_table(probabilities, labels, n_bins=n_bins, include_empty_bins=False)
    return float(
        sum(
            _required_float(row, "weight")
            * _required_float(row, "absolute_calibration_error")
            for row in rows
        )
    )


def maximum_calibration_error(probabilities: Any, labels: Any, *, n_bins: int = 10) -> float:
    """Return maximum equal-width calibration error over non-empty bins."""

    rows = reliability_bin_table(probabilities, labels, n_bins=n_bins, include_empty_bins=False)
    return float(
        max(
            (_required_float(row, "absolute_calibration_error") for row in rows),
            default=0.0,
        )
    )


def calibration_summary(
    probabilities: Any,
    labels: Any,
    *,
    n_bins: int = 10,
) -> dict[str, float | int]:
    """Return Brier, ECE, and MCE-style scalar calibration diagnostics."""

    probabilities, labels = _validate_probability_label_inputs(probabilities, labels)
    n_bins = _validate_n_bins(n_bins)
    rows = reliability_bin_table(probabilities, labels, n_bins=n_bins, include_empty_bins=False)
    positive_examples = int(np.sum(labels))
    expected_error = float(
        sum(
            _required_float(row, "weight")
            * _required_float(row, "absolute_calibration_error")
            for row in rows
        )
    )
    maximum_error = float(
        max(
            (_required_float(row, "absolute_calibration_error") for row in rows),
            default=0.0,
        )
    )
    return {
        "calibration_examples": int(probabilities.shape[0]),
        "calibration_positive_examples": positive_examples,
        "calibration_negative_examples": int(probabilities.shape[0] - positive_examples),
        "calibration_bins": int(n_bins),
        "calibration_n_bins": int(n_bins),
        "calibration_occupied_bins": int(len(rows)),
        "calibration_brier_score": brier_score(probabilities, labels),
        "calibration_expected_error": expected_error,
        "calibration_ece": expected_error,
        "calibration_maximum_error": maximum_error,
        "calibration_mce": maximum_error,
        "calibration_mean_predicted_probability": float(np.mean(probabilities)),
        "calibration_empirical_positive_rate": float(np.mean(labels)),
    }


def brier_score(probabilities: Any, labels: Any) -> float:
    """Return the mean squared error between probabilities and binary labels."""

    probabilities, labels = _validate_probability_label_inputs(probabilities, labels)
    return float(np.mean((probabilities - labels) ** 2))


def format_reliability_bin_table(rows: Sequence[Mapping[str, object]]) -> str:
    """Format reliability-bin rows as a Markdown table."""

    metadata_columns = ("subject", "held_out_subject", "training_subjects")
    standard_columns = (
        "bin_index",
        "probability_lower",
        "probability_upper",
        "count",
        "positive_count",
        "mean_predicted_probability",
        "empirical_positive_rate",
        "absolute_calibration_error",
        "bin_brier_score",
        "weight",
    )
    columns = tuple(column for column in metadata_columns if any(column in row for row in rows)) + standard_columns
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [header, separator]
    for row in rows:
        body.append("| " + " | ".join(_format_table_value(row.get(column)) for column in columns) + " |")
    return "\n".join(body)


def _validate_probability_label_inputs(probabilities: Any, labels: Any) -> tuple[np.ndarray, np.ndarray]:
    probability_array = np.asarray(probabilities, dtype=float).reshape(-1)
    label_array = np.asarray(labels).reshape(-1)
    if probability_array.shape[0] == 0:
        raise ValueError("At least one probability is required")
    if probability_array.shape[0] != label_array.shape[0]:
        raise ValueError("probabilities and labels must contain the same number of entries")
    if not np.all(np.isfinite(probability_array)):
        raise ValueError("probabilities must be finite")
    if np.any((probability_array < 0.0) | (probability_array > 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")

    unique_labels = np.unique(label_array)
    if not np.all(np.isin(unique_labels, [0, 1, False, True])):
        raise ValueError("labels must be binary values 0/1 or False/True")
    return probability_array, label_array.astype(float)


def _validate_n_bins(n_bins: int) -> int:
    try:
        parsed = int(n_bins)
    except (TypeError, ValueError) as exc:
        raise ValueError("n_bins must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError("n_bins must be a positive integer")
    return parsed


def _empty_bin_row(bin_index: int, edges: np.ndarray) -> CalibrationBinRow:
    return {
        "bin_index": int(bin_index),
        "probability_lower": float(edges[bin_index]),
        "probability_upper": float(edges[bin_index + 1]),
        "count": 0,
        "positive_count": 0,
        "negative_count": 0,
        "mean_predicted_probability": None,
        "empirical_positive_rate": None,
        "signed_calibration_error": None,
        "absolute_calibration_error": None,
        "bin_brier_score": None,
        "weight": 0.0,
    }


def _required_float(row: Mapping[str, object], key: str) -> float:
    value = row[key]
    if value is None:
        raise ValueError(f"Reliability bin row {key!r} must not be empty")
    return float(value)


def _format_table_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    return str(value)
