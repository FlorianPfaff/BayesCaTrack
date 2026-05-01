import numpy as np
import pytest

from bayescatrack.evaluation.complete_track_scores import (
    complete_track_set,
    identity_switch_events,
    pairwise_track_set,
    score_complete_tracks,
    score_identity_switches,
    score_pairwise_tracks,
    score_track_matrices,
    track_lengths,
)
from bayescatrack.evaluation.fixed_precision import score_complete_tracks_at_fixed_precision
from bayescatrack.evaluation.track2p_metrics import score_track_matrix_against_reference
from bayescatrack.reference import Track2pReference


def test_complete_track_and_pairwise_scoring():
    reference = np.array(
        [
            [0, 10, 20],
            [1, 11, 21],
            [2, None, 22],
        ],
        dtype=object,
    )
    predicted = np.array(
        [
            [0, 10, 20],
            [1, None, 21],
            [3, 13, 23],
        ],
        dtype=object,
    )

    assert complete_track_set(reference) == {(0, 10, 20), (1, 11, 21)}
    assert pairwise_track_set(predicted) == {(0, 1, 0, 10), (1, 2, 10, 20), (0, 1, 3, 13), (1, 2, 13, 23)}

    complete_scores = score_complete_tracks(predicted, reference)
    assert complete_scores["complete_track_true_positives"] == 1
    assert complete_scores["complete_track_false_positives"] == 1
    assert complete_scores["complete_track_false_negatives"] == 1
    assert complete_scores["complete_track_f1"] == pytest.approx(0.5)

    pairwise_scores = score_pairwise_tracks(predicted, reference)
    assert pairwise_scores["pairwise_true_positives"] == 2
    assert pairwise_scores["pairwise_false_positives"] == 2
    assert pairwise_scores["pairwise_false_negatives"] == 2
    assert pairwise_scores["pairwise_f1"] == pytest.approx(0.5)

    scores = score_track_matrices(predicted, reference)
    assert scores["complete_tracks"] == 2
    assert scores["mean_track_length"] == pytest.approx(8 / 3)
    np.testing.assert_array_equal(track_lengths(predicted), np.array([3, 2, 3]))


def test_identity_switch_diagnostics_count_reference_track_changes():
    reference = np.array(
        [
            [0, 10, 20, 30],
            [1, 11, 21, 31],
        ],
        dtype=object,
    )
    predicted = np.array(
        [
            [0, 10, None, None],
            [None, None, 20, 30],
            [1, 11, 21, 31],
        ],
        dtype=object,
    )

    assert identity_switch_events(predicted, reference) == [
        {
            "reference_track": 0,
            "previous_session": 1,
            "session": 2,
            "previous_predicted_track": 0,
            "predicted_track": 1,
            "previous_roi": 10,
            "roi": 20,
        }
    ]

    scores = score_identity_switches(predicted, reference)
    assert scores["identity_switches"] == 1
    assert scores["reference_tracks"] == 2
    assert scores["reference_tracks_with_predictions"] == 2
    assert scores["reference_tracks_with_identity_switches"] == 1
    assert scores["identity_switches_per_reference_track"] == pytest.approx(0.5)
    assert scores["identity_switches_per_matched_reference_track"] == pytest.approx(0.5)

    matrix_scores = score_track_matrices(predicted, reference)
    assert matrix_scores["identity_switches"] == 1


def test_identity_switch_diagnostics_ignore_misses_without_track_change():
    reference = np.array([[0, 10, 20]], dtype=object)
    predicted = np.array([[0, None, 20]], dtype=object)

    assert identity_switch_events(predicted, reference) == []
    assert score_identity_switches(predicted, reference)["identity_switches"] == 0


def test_identity_switch_diagnostics_reject_duplicate_predicted_roi_in_session():
    reference = np.array([[0, 10]], dtype=object)
    predicted = np.array([[0, 10], [0, None]], dtype=object)

    with pytest.raises(ValueError, match="ROI 0 more than once in session 0"):
        score_identity_switches(predicted, reference)


def test_complete_tracks_at_fixed_precision_sweeps_scored_thresholds():
    reference = np.array(
        [
            [0, 10, 20],
            [1, 11, 21],
            [2, 12, 22],
        ],
        dtype=object,
    )
    predicted = np.array(
        [
            [0, 10, 20],
            [7, 17, 27],
            [1, 11, 21],
            [2, None, 22],
        ],
        dtype=object,
    )

    scores = score_complete_tracks_at_fixed_precision(
        predicted,
        reference,
        target_precisions=(0.75, 0.60),
        track_scores=(0.9, 0.8, 0.7, 0.99),
    )

    assert scores["complete_tracks_at_fixed_precision_0_75"] == 1
    assert scores["complete_track_predictions_at_fixed_precision_0_75"] == 1
    assert scores["complete_track_precision_at_fixed_precision_0_75"] == pytest.approx(1.0)
    assert scores["complete_track_recall_at_fixed_precision_0_75"] == pytest.approx(1 / 3)
    assert scores["complete_track_score_threshold_at_fixed_precision_0_75"] == pytest.approx(0.9)
    assert scores["complete_tracks_at_fixed_precision_0_6"] == 2
    assert scores["complete_track_predictions_at_fixed_precision_0_6"] == 3
    assert scores["complete_track_precision_at_fixed_precision_0_6"] == pytest.approx(2 / 3)


def test_complete_tracks_at_fixed_precision_uses_all_or_nothing_without_scores():
    reference = np.array([[0, 10], [1, 11]], dtype=object)
    predicted = np.array([[0, 10], [7, 17]], dtype=object)

    scores = score_complete_tracks_at_fixed_precision(predicted, reference, target_precisions=(0.9, 0.5))

    assert scores["complete_tracks_at_fixed_precision_0_9"] == 0
    assert scores["complete_track_predictions_at_fixed_precision_0_9"] == 0
    assert scores["complete_track_score_threshold_at_fixed_precision_0_9"] == pytest.approx(float("inf"))
    assert scores["complete_tracks_at_fixed_precision_0_5"] == 1
    assert scores["complete_track_predictions_at_fixed_precision_0_5"] == 2


def test_fixed_precision_rejects_invalid_track_scores():
    with pytest.raises(ValueError, match="one score per predicted track"):
        score_complete_tracks_at_fixed_precision(np.zeros((2, 2)), np.zeros((1, 2)), track_scores=(1.0,))

    with pytest.raises(ValueError, match="finite"):
        score_complete_tracks_at_fixed_precision(np.zeros((1, 2)), np.zeros((1, 2)), track_scores=(float("nan"),))


def test_track2p_reference_scoring_can_filter_curated_rows():
    reference = Track2pReference(
        session_names=("day0", "day1", "day2"),
        suite2p_indices=np.array([[0, 10, 20], [1, 11, 21]], dtype=object),
        curated_mask=np.array([True, False]),
    )
    predicted = np.array([[0, 10, 20], [1, 11, 21]], dtype=object)

    scores = score_track_matrix_against_reference(predicted, reference, curated_only=True)

    assert scores["complete_track_precision"] == pytest.approx(0.5)
    assert scores["complete_track_recall"] == pytest.approx(1.0)
    assert scores["reference_complete_tracks"] == 1


def test_score_track_matrices_requires_same_number_of_sessions():
    with pytest.raises(ValueError, match="same number of sessions"):
        score_track_matrices(np.zeros((1, 2)), np.zeros((1, 3)))
