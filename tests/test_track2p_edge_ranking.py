from __future__ import annotations

import csv

from bayescatrack.experiments.track2p_benchmark import Track2pBenchmarkConfig
from bayescatrack.experiments.track2p_edge_ranking import run_track2p_edge_ranking
from tests.test_track2p_benchmark import _write_ground_truth_csv, _write_subject


def test_track2p_edge_ranking_writes_edge_and_summary_csvs(
    tmp_path, monkeypatch, write_raw_npy_session
):
    subject_dir = tmp_path / "jm001"
    _write_subject(subject_dir, write_raw_npy_session, write_reference=False)
    _write_ground_truth_csv(
        subject_dir,
        ("2024-05-01_a", "2024-05-02_a", "2024-05-03_a"),
        ((0, 0, 0), (1, 1, 1)),
    )

    from bayescatrack.association import calibrated_costs

    monkeypatch.setattr(
        calibrated_costs,
        "register_plane_pair",
        lambda _reference, moving, **_kwargs: moving,
    )

    output_path = tmp_path / "edge_ranking.csv"
    summary_path = tmp_path / "edge_ranking_summary.csv"
    edge_rows, summary_rows = run_track2p_edge_ranking(
        Track2pBenchmarkConfig(
            data=tmp_path,
            method="global-assignment",
            cost="registered-iou",
            max_gap=2,
        ),
        output_path,
        summary_output_path=summary_path,
        feature_names=("pairwise_cost_matrix", "iou"),
        similarity_features=("iou",),
    )

    assert edge_rows == 12
    assert summary_rows == 6

    with output_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == edge_rows
    assert {row["subject"] for row in rows} == {"jm001"}
    assert {row["session_gap"] for row in rows} == {"1", "2"}
    assert {row["score_name"] for row in rows} == {"pairwise_cost_matrix", "iou"}
    assert {row["score_direction"] for row in rows if row["score_name"] == "iou"} == {
        "similarity"
    }
    assert all(row["edge_present"] == "1" for row in rows)
    assert all(int(row["row_rank"]) == 1 for row in rows)

    with summary_path.open(newline="", encoding="utf-8") as handle:
        summaries = list(csv.DictReader(handle))
    assert len(summaries) == summary_rows
    assert all(float(row["row_hit_at_1"]) == 1.0 for row in summaries)
    assert all(float(row["column_hit_at_1"]) == 1.0 for row in summaries)
