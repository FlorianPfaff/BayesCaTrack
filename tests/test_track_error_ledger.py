import numpy as np
import pytest
from bayescatrack.evaluation.complete_track_scores import score_track_matrices
from bayescatrack.evaluation.track_error_ledger import (
    summarize_track_errors,
    track_error_ledger,
)


def test_track_error_ledger_reports_switches_fragmentation_and_misses():
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
            [0, 10, 21],
            [1, 11, None],
            [99, 98, 97],
        ],
        dtype=object,
    )

    ledger = track_error_ledger(predicted, reference)
    summary = ledger["summary"]

    assert summary["identity_switches"] == 1
    assert summary["mixed_identity_tracks"] == 1
    assert summary["spurious_tracks"] == 1
    assert summary["spurious_predicted_observations"] == 3
    assert summary["false_continuation_links"] == 3
    assert summary["missed_reference_links"] == 4
    assert summary["fragmented_reference_tracks"] == 1
    assert summary["track_fragmentations"] == 1
    assert summary["missed_reference_tracks"] == 1
    assert summary["partial_reference_tracks"] == 1
    assert summary["missed_reference_observations"] == 4
    assert summary["false_continuation_rate"] == pytest.approx(3 / 5)
    assert summary["missed_reference_link_rate"] == pytest.approx(4 / 6)

    assert ledger["predicted_tracks"][0]["category"] == "mixed_identity"
    assert ledger["predicted_tracks"][2]["category"] == "spurious"
    assert ledger["reference_tracks"][0]["category"] == "partial"
    assert ledger["reference_tracks"][1]["category"] == "fragmented"
    assert ledger["reference_tracks"][2]["category"] == "missed"

    false_continuations = [
        row
        for row in ledger["link_errors"]
        if row["error_type"] == "false_continuation"
    ]
    missed_links = [
        row
        for row in ledger["link_errors"]
        if row["error_type"] == "missed_reference_link"
    ]
    assert len(false_continuations) == 3
    assert len(missed_links) == 4
    assert any(
        row["reason"] == "different_reference_tracks" for row in false_continuations
    )
    assert any(row["reason"] == "split_across_predicted_tracks" for row in missed_links)


def test_track_error_summary_is_included_in_score_track_matrices():
    reference = np.array([[0, 10, 20]], dtype=object)
    predicted = np.array([[0, 10, 99]], dtype=object)

    scores = score_track_matrices(predicted, reference)

    assert scores["pairwise_false_positives"] == 1
    assert scores["false_continuation_links"] == 1
    assert scores["missed_reference_links"] == 1
    assert scores["identity_switches"] == 0
    assert scores["spurious_predicted_observations"] == 1
    assert scores["false_continuation_link_rate"] == pytest.approx(1 / 2)


def test_track_error_ledger_honors_custom_session_pairs():
    reference = np.array([[0, 10, 20]], dtype=object)
    predicted = np.array([[0, None, 20]], dtype=object)

    summary = summarize_track_errors(predicted, reference, session_pairs=[(0, 2)])

    assert summary["false_continuation_links"] == 0
    assert summary["missed_reference_links"] == 0
    assert summary["partial_reference_tracks"] == 1
    assert summary["missed_reference_observations"] == 1
