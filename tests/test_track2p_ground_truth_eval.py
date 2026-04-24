from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import numpy.testing as npt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_PATH))

from track2p_pyrecest_bridge.ground_truth_eval import (  # noqa: E402
    TrackTable,
    complete_tracks_score,
    evaluate_track_table_prediction,
    load_track2p_ground_truth_csv,
    load_track_table_csv,
    proportion_correct_by_horizon,
    tracks_from_consecutive_matches,
)


def _write_csv(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def test_load_track2p_ground_truth_csv_wide_format(tmp_path: Path) -> None:
    csv_path = tmp_path / "ground_truth.csv"
    _write_csv(
        csv_path,
        ["track_id", "2024-05-01_a", "2024-05-02_a", "2024-05-03_a"],
        [
            [0, 10, 11, 12],
            [1, 20, 21, ""],
        ],
    )

    table = load_track2p_ground_truth_csv(csv_path)

    assert table.session_names == (
        "2024-05-01_a",
        "2024-05-02_a",
        "2024-05-03_a",
    )
    npt.assert_array_equal(table.tracks, np.array([[10, 11, 12], [20, 21, -1]]))


def test_load_track_table_csv_long_format(tmp_path: Path) -> None:
    csv_path = tmp_path / "prediction.csv"
    _write_csv(
        csv_path,
        ["track_id", "session", "roi_index"],
        [
            ["a", "2024-05-01_a", 10],
            ["a", "2024-05-02_a", 11],
            ["a", "2024-05-03_a", 12],
            ["b", "2024-05-01_a", 20],
            ["b", "2024-05-02_a", 21],
        ],
    )

    table = load_track_table_csv(csv_path)

    assert table.session_names == (
        "2024-05-01_a",
        "2024-05-02_a",
        "2024-05-03_a",
    )
    npt.assert_array_equal(table.tracks, np.array([[10, 11, 12], [20, 21, -1]]))


def test_tracks_from_consecutive_matches() -> None:
    prediction = tracks_from_consecutive_matches(
        ["s0", "s1", "s2", "s3"],
        [
            {10: 110, 20: 120},
            np.array([[110, 210], [120, 220]], dtype=int),
            ([210, 220], [310, 320]),
        ],
    )

    assert prediction.session_names == ("s0", "s1", "s2", "s3")
    npt.assert_array_equal(
        prediction.tracks,
        np.array(
            [
                [10, 110, 210, 310],
                [20, 120, 220, 320],
            ],
            dtype=int,
        ),
    )


def test_complete_tracks_score_and_proportion_correct() -> None:
    ground_truth = TrackTable(
        session_names=("s0", "s1", "s2"),
        tracks=np.array([[10, 11, 12], [20, 21, 22]], dtype=int),
    )
    prediction = TrackTable(
        session_names=("s0", "s1", "s2"),
        tracks=np.array(
            [
                [10, 11, 12],
                [20, 21, -1],
                [30, 31, 32],
            ],
            dtype=int,
        ),
    )

    ct = complete_tracks_score(ground_truth, prediction)
    prop = proportion_correct_by_horizon(ground_truth, prediction)

    assert np.isclose(ct, 0.4)
    assert prop == {2: 1.0, 3: 0.5}


def test_complete_tracks_score_counts_duplicate_predictions() -> None:
    ground_truth = TrackTable(
        session_names=("s0", "s1"),
        tracks=np.array([[10, 11]], dtype=int),
    )
    prediction = TrackTable(
        session_names=("s0", "s1"),
        tracks=np.array([[10, 11], [10, 11]], dtype=int),
    )

    assert np.isclose(complete_tracks_score(ground_truth, prediction), 2.0 / 3.0)


def test_evaluate_track_table_prediction_aligns_by_session_name() -> None:
    ground_truth = TrackTable(
        session_names=("s0", "s1", "s2"),
        tracks=np.array([[10, 11, 12]], dtype=int),
    )
    prediction = TrackTable(
        session_names=("s2", "s0", "s1"),
        tracks=np.array([[12, 10, 11]], dtype=int),
    )

    evaluation = evaluate_track_table_prediction(ground_truth, prediction)

    assert np.isclose(evaluation.complete_tracks, 1.0)
    assert evaluation.proportion_correct_by_horizon == {2: 1.0, 3: 1.0}
    assert evaluation.n_exact_full_track_matches == 1
