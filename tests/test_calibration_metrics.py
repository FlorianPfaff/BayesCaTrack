from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.evaluation.calibration_metrics import brier_score


def test_brier_score_matches_mean_squared_probability_error():
    probabilities = np.array([0.0, 0.25, 0.75, 1.0])
    labels = np.array([0, 0, 1, 1])

    assert brier_score(probabilities, labels) == pytest.approx(0.03125)


def test_brier_score_supports_sample_weights():
    probabilities = np.array([0.0, 1.0, 0.5])
    labels = np.array([0, 0, 1])
    weights = np.array([1.0, 3.0, 1.0])

    assert brier_score(probabilities, labels, sample_weight=weights) == pytest.approx(
        0.65
    )


@pytest.mark.parametrize(
    ("probabilities", "labels", "message"),
    [
        ([0.2], [0, 1], "same flattened shape"),
        ([], [], "At least one"),
        ([1.2], [1], "lie in"),
        ([0.5], [2], "binary"),
    ],
)
def test_brier_score_rejects_invalid_inputs(probabilities, labels, message):
    with pytest.raises(ValueError, match=message):
        brier_score(probabilities, labels)
