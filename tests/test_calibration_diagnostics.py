from __future__ import annotations

import pytest
from bayescatrack.evaluation.calibration_diagnostics import (
    calibration_summary,
    expected_calibration_error,
    maximum_calibration_error,
    reliability_bin_table,
)


def test_expected_calibration_error_is_weighted_over_nonempty_bins():
    probabilities = [0.2, 0.8]
    labels = [0, 1]

    assert expected_calibration_error(
        probabilities, labels, n_bins=10
    ) == pytest.approx(0.2)
    assert maximum_calibration_error(probabilities, labels, n_bins=10) == pytest.approx(
        0.2
    )


def test_perfect_empirical_bin_calibration_has_zero_ece():
    probabilities = [0.5, 0.5]
    labels = [0, 1]

    assert expected_calibration_error(probabilities, labels, n_bins=2) == pytest.approx(
        0.0
    )
    assert maximum_calibration_error(probabilities, labels, n_bins=2) == pytest.approx(
        0.0
    )


def test_probability_one_is_included_in_last_bin():
    rows = reliability_bin_table([0.0, 1.0], [0, 1], n_bins=2)

    assert rows[0]["count"] == 1
    assert rows[1]["count"] == 1
    assert rows[0]["mean_predicted_probability"] == pytest.approx(0.0)
    assert rows[1]["mean_predicted_probability"] == pytest.approx(1.0)


def test_calibration_summary_reports_ece_and_mce_aliases():
    summary = calibration_summary([0.25, 0.75], [0, 1], n_bins=2)

    assert summary["calibration_examples"] == 2
    assert summary["calibration_positive_examples"] == 1
    assert summary["calibration_negative_examples"] == 1
    assert summary["calibration_brier_score"] == pytest.approx(0.0625)
    assert summary["calibration_expected_error"] == pytest.approx(0.25)
    assert summary["calibration_ece"] == pytest.approx(0.25)
    assert summary["calibration_maximum_error"] == pytest.approx(0.25)
    assert summary["calibration_mce"] == pytest.approx(0.25)
    assert summary["calibration_n_bins"] == 2


def test_calibration_inputs_are_validated():
    with pytest.raises(ValueError, match="same number"):
        expected_calibration_error([0.5], [0, 1])
    with pytest.raises(ValueError, match="At least one"):
        expected_calibration_error([], [])
    with pytest.raises(ValueError, match="finite"):
        expected_calibration_error([float("nan")], [1])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        expected_calibration_error([1.1], [1])
    with pytest.raises(ValueError, match="binary"):
        expected_calibration_error([0.5], [2])
    with pytest.raises(ValueError, match="positive integer"):
        expected_calibration_error([0.5], [1], n_bins=0)
