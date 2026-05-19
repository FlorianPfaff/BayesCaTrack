import numpy as np
import pytest
from bayescatrack.ground_truth_eval import (
    TrackTable,
    evaluate_track_table_prediction,
    load_track2p_ground_truth_csv,
)


def test_evaluate_track_table_prediction_scores_exact_tracks():
    ground_truth = TrackTable(
        session_names=("s1", "s2"),
        tracks=np.array([[1, 2], [3, 4]], dtype=int),
    )
    prediction = TrackTable(
        session_names=("s1", "s2"),
        tracks=np.array([[1, 2], [3, 5]], dtype=int),
    )

    evaluation = evaluate_track_table_prediction(ground_truth, prediction)

    assert evaluation.n_exact_full_track_matches == 1
    assert evaluation.complete_tracks == 0.5
    assert evaluation.proportion_correct_by_horizon[2] == 0.5


def test_complete_tracks_score_ignores_incomplete_rows_in_denominator():
    ground_truth = TrackTable(
        session_names=("s1", "s2", "s3"),
        tracks=np.array([[1, 2, 3], [4, -1, 6]], dtype=int),
    )
    prediction = TrackTable(
        session_names=("s1", "s2", "s3"),
        tracks=np.array([[1, 2, 3], [7, -1, 9]], dtype=int),
    )

    evaluation = evaluate_track_table_prediction(ground_truth, prediction)

    assert evaluation.complete_tracks == 1.0
    assert evaluation.n_exact_full_track_matches == 1


def test_exact_full_track_matches_do_not_count_matching_incomplete_rows():
    ground_truth = TrackTable(
        session_names=("s1", "s2", "s3"),
        tracks=np.array([[1, 2, 3], [4, -1, 6]], dtype=int),
    )
    prediction = TrackTable(
        session_names=("s1", "s2", "s3"),
        tracks=np.array([[1, 2, 3], [4, -1, 6]], dtype=int),
    )

    evaluation = evaluate_track_table_prediction(ground_truth, prediction)

    assert evaluation.n_exact_full_track_matches == 1
    assert evaluation.complete_tracks == 1.0


def test_load_track2p_ground_truth_csv_supports_semicolon_encoded_rows(tmp_path):
    ground_truth_path = tmp_path / "ground_truth.csv"
    ground_truth_path.write_text(
        "track_id,track\n0,67;38;15;169;;;\n1,11;;13;14\n",
        encoding="utf-8",
    )

    table = load_track2p_ground_truth_csv(
        ground_truth_path,
        session_names=("s1", "s2", "s3", "s4"),
    )

    assert table.session_names == ("s1", "s2", "s3", "s4")
    np.testing.assert_array_equal(
        table.tracks,
        np.array([[67, 38, 15, 169], [11, -1, 13, 14]], dtype=int),
    )


def test_load_track2p_ground_truth_csv_infers_semicolon_width_without_session_names(
    tmp_path,
):
    ground_truth_path = tmp_path / "ground_truth.csv"
    ground_truth_path.write_text(
        "track_id,track\n0,67;38;15;169;;;\n1,11;;13;14\n",
        encoding="utf-8",
    )

    table = load_track2p_ground_truth_csv(ground_truth_path)

    assert table.session_names == (
        "session_0",
        "session_1",
        "session_2",
        "session_3",
        "session_4",
        "session_5",
        "session_6",
    )
    np.testing.assert_array_equal(
        table.tracks,
        np.array(
            [[67, 38, 15, 169, -1, -1, -1], [11, -1, 13, 14, -1, -1, -1]],
            dtype=int,
        ),
    )


def test_load_track2p_ground_truth_csv_infers_subject_session_dirs(tmp_path):
    subject_dir = tmp_path / "jm038"
    subject_dir.mkdir()
    for session_name in (
        "2024-05-01_a",
        "2024-05-02_a",
        "2024-05-03_a",
        "2024-05-04_a",
    ):
        (subject_dir / session_name / "suite2p" / "plane0").mkdir(parents=True)
    ground_truth_path = subject_dir / "ground_truth.csv"
    ground_truth_path.write_text(
        "track_id,track\n0,67;38;15;169;;;\n1,11;;13;14\n",
        encoding="utf-8",
    )

    table = load_track2p_ground_truth_csv(ground_truth_path)

    assert table.session_names == (
        "2024-05-01_a",
        "2024-05-02_a",
        "2024-05-03_a",
        "2024-05-04_a",
    )
    np.testing.assert_array_equal(
        table.tracks,
        np.array([[67, 38, 15, 169], [11, -1, 13, 14]], dtype=int),
    )


def test_load_track2p_ground_truth_csv_rejects_duplicate_long_entries(tmp_path):
    ground_truth_path = tmp_path / "ground_truth.csv"
    ground_truth_path.write_text(
        "track_id,session,roi\n"
        "cell_0,s1,7\n"
        "cell_0,s1,8\n"
        "cell_0,s2,9\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate entries"):
        load_track2p_ground_truth_csv(
            ground_truth_path,
            session_names=("s1", "s2"),
        )


def test_load_track2p_ground_truth_csv_rejects_malformed_roi_tokens(tmp_path):
    ground_truth_path = tmp_path / "ground_truth.csv"
    ground_truth_path.write_text(
        "track_id,s1,s2\n0,67,38\n1,11,typo\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_track2p_ground_truth_csv(
            ground_truth_path,
            session_names=("s1", "s2"),
        )
