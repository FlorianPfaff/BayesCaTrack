import numpy as np

from bayescatrack.ground_truth_eval import TrackTable, evaluate_track_table_prediction


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
