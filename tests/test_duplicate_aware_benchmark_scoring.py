from __future__ import annotations

import numpy as np
import pytest

from bayescatrack.evaluation.complete_track_scores import score_track_matrices


def test_score_track_matrices_penalizes_duplicate_complete_tracks():
    reference = np.array([[0, 10, 20]], dtype=object)
    predicted = np.array([[0, 10, 20], [0, 10, 20]], dtype=object)

    scores = score_track_matrices(predicted, reference)

    assert scores["complete_track_true_positives"] == 1
    assert scores["complete_track_false_positives"] == 1
    assert scores["complete_track_false_negatives"] == 0
    assert scores["complete_tracks"] == 2
    assert scores["reference_complete_tracks"] == 1
    assert scores["complete_track_precision"] == pytest.approx(0.5)
    assert scores["complete_track_recall"] == pytest.approx(1.0)
    assert scores["complete_track_f1"] == pytest.approx(2.0 / 3.0)


def test_score_track_matrices_penalizes_duplicate_pairwise_links():
    reference = np.array([[0, 10, 20]], dtype=object)
    predicted = np.array([[0, 10, 20], [0, 10, 20]], dtype=object)

    scores = score_track_matrices(predicted, reference)

    assert scores["pairwise_true_positives"] == 2
    assert scores["pairwise_false_positives"] == 2
    assert scores["pairwise_false_negatives"] == 0
    assert scores["pairwise_links"] == 4
    assert scores["reference_pairwise_links"] == 2
    assert scores["pairwise_precision"] == pytest.approx(0.5)
    assert scores["pairwise_recall"] == pytest.approx(1.0)
    assert scores["pairwise_f1"] == pytest.approx(2.0 / 3.0)


def test_score_track_matrices_respects_duplicate_scoring_with_session_subsets():
    reference = np.array([[0, 10, 20]], dtype=object)
    predicted = np.array([[0, 10, 20], [0, 10, 99]], dtype=object)

    scores = score_track_matrices(
        predicted,
        reference,
        session_pairs=((0, 1),),
        complete_session_indices=(0, 1),
    )

    assert scores["pairwise_true_positives"] == 1
    assert scores["pairwise_false_positives"] == 1
    assert scores["pairwise_false_negatives"] == 0
    assert scores["pairwise_f1"] == pytest.approx(2.0 / 3.0)
    assert scores["complete_track_true_positives"] == 1
    assert scores["complete_track_false_positives"] == 1
    assert scores["complete_track_false_negatives"] == 0
    assert scores["complete_track_f1"] == pytest.approx(2.0 / 3.0)
