import numpy as np
import pytest
from bayescatrack.evaluation.complete_track_scores import (
    complete_track_set,
    pairwise_track_set,
    reference_fragment_counts,
    score_complete_tracks,
    score_false_continuations,
    score_fragmentation,
    score_pairwise_tracks,
    score_track_matrices,
    track_lengths,
)
from bayescatrack.evaluation.fixed_precision import (
    score_complete_tracks_at_fixed_precision,
)
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
    assert pairwise_track_set(predicted) == {
        (0, 1, 0, 10),
        (1, 2, 10, 20),
        (0, 1, 3, 13),
        (1, 2, 13, 23),
    }

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
    assert scores["fragmentation_events"] == 0
    np.testing.assert_array_equal(track_lengths(predicted), np.array([3, 2, 3]))


def test_fragmentation_scores_reference_identities_split_across_predicted_tracks():
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
            [0, 10, None],
            [None, None, 20],
            [1, 11, 21],
            [3, 13, 23],
        ],
        dtype=object,
    )

    np.testing.assert_array_equal(
        reference_fragment_counts(predicted, reference), np.array([2, 1, 0])
    )

    scores = score_fragmentation(predicted, reference)
    assert scores["fragmentation_reference_tracks"] == 3
    assert scores["fragmentation_covered_reference_tracks"] == 2
    assert scores["fragmentation_fragmented_reference_tracks"] == 1
    assert scores["fragmentation_fragments"] == 3
    assert scores["fragmentation_events"] == 1
    assert scores["fragmentation_rate"] == pytest.approx(1 / 3)
    assert scores["fragmentation_covered_rate"] == pytest.approx(1 / 2)
    assert scores["fragmentation_mean_fragments_per_reference_track"] == pytest.approx(
        1.0
    )
    assert scores[
        "fragmentation_mean_fragments_per_covered_reference_track"
    ] == pytest.approx(1.5)
    assert scores["fragmentation_max_fragments_per_reference_track"] == 2

    matrix_scores = score_track_matrices(predicted, reference)
    assert matrix_scores["fragmentation_events"] == 1
    assert matrix_scores["fragmentation_rate"] == pytest.approx(1 / 3)


def test_fragmentation_scores_ignore_empty_reference_rows():
    reference = np.array([[None, None], [0, 1]], dtype=object)
    predicted = np.array([[0, 1]], dtype=object)

    scores = score_fragmentation(predicted, reference)

    assert scores["fragmentation_reference_tracks"] == 1
    assert scores["fragmentation_covered_reference_tracks"] == 1
    assert scores["fragmentation_events"] == 0
    assert scores["fragmentation_rate"] == pytest.approx(0.0)


def test_false_continuation_rate_counts_labeled_wrong_continuations():
    reference = np.array(
        [
            [0, 10, None],
            [1, None, 21],
        ],
        dtype=object,
    )
    predicted = np.array(
        [
            [0, 10, 30],
            [1, 11, 21],
            [99, 100, 101],
        ],
        dtype=object,
    )

    scores = score_false_continuations(predicted, reference)

    assert scores["valid_continuations"] == 1
    assert scores["false_continuations"] == 2
    assert scores["labeled_predicted_continuations"] == 3
    assert scores["unknown_source_continuations"] == 3
    assert scores["false_continuation_rate"] == pytest.approx(2 / 3)

    combined_scores = score_track_matrices(predicted, reference)
    assert combined_scores["false_continuation_rate"] == pytest.approx(2 / 3)


def test_false_continuation_rate_uses_requested_session_pairs():
    reference = np.array([[0, None, 20]], dtype=object)
    predicted = np.array([[0, 10, 20]], dtype=object)

    adjacent_scores = score_false_continuations(predicted, reference)
    assert adjacent_scores["false_continuations"] == 1
    assert adjacent_scores["unknown_source_continuations"] == 1

    skip_scores = score_false_continuations(
        predicted, reference, session_pairs=[(0, 2)]
    )
    assert skip_scores["valid_continuations"] == 1
    assert skip_scores["false_continuation_rate"] == pytest.approx(0.0)


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
    assert scores["complete_track_precision_at_fixed_precision_0_75"] == pytest.approx(
        1.0
    )
    assert scores["complete_track_recall_at_fixed_precision_0_75"] == pytest.approx(
        1 / 3
    )
    assert scores[
        "complete_track_score_threshold_at_fixed_precision_0_75"
    ] == pytest.approx(0.9)
    assert scores["complete_tracks_at_fixed_precision_0_6"] == 2
    assert scores["complete_track_predictions_at_fixed_precision_0_6"] == 3
    assert scores["complete_track_precision_at_fixed_precision_0_6"] == pytest.approx(
        2 / 3
    )


def test_complete_tracks_at_fixed_precision_uses_all_or_nothing_without_scores():
    reference = np.array([[0, 10], [1, 11]], dtype=object)
    predicted = np.array([[0, 10], [7, 17]], dtype=object)

    scores = score_complete_tracks_at_fixed_precision(
        predicted, reference, target_precisions=(0.9, 0.5)
    )

    assert scores["complete_tracks_at_fixed_precision_0_9"] == 0
    assert scores["complete_track_predictions_at_fixed_precision_0_9"] == 0
    assert scores[
        "complete_track_score_threshold_at_fixed_precision_0_9"
    ] == pytest.approx(float("inf"))
    assert scores["complete_tracks_at_fixed_precision_0_5"] == 1
    assert scores["complete_track_predictions_at_fixed_precision_0_5"] == 2


def test_fixed_precision_rejects_invalid_track_scores():
    with pytest.raises(ValueError, match="one score per predicted track"):
        score_complete_tracks_at_fixed_precision(
            np.zeros((2, 2)), np.zeros((1, 2)), track_scores=(1.0,)
        )

    with pytest.raises(ValueError, match="finite"):
        score_complete_tracks_at_fixed_precision(
            np.zeros((1, 2)), np.zeros((1, 2)), track_scores=(float("nan"),)
        )


def test_track2p_reference_scoring_can_filter_curated_rows():
    reference = Track2pReference(
        session_names=("day0", "day1", "day2"),
        suite2p_indices=np.array([[0, 10, 20], [1, 11, 21]], dtype=object),
        curated_mask=np.array([True, False]),
    )
    predicted = np.array([[0, 10, 20], [1, 11, 21]], dtype=object)

    scores = score_track_matrix_against_reference(
        predicted, reference, curated_only=True
    )

    assert scores["complete_track_precision"] == pytest.approx(0.5)
    assert scores["complete_track_recall"] == pytest.approx(1.0)
    assert scores["reference_complete_tracks"] == 1


def test_score_track_matrices_requires_same_number_of_sessions():
    with pytest.raises(ValueError, match="same number of sessions"):
        score_track_matrices(np.zeros((1, 2)), np.zeros((1, 3)))
