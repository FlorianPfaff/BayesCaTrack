from __future__ import annotations

import numpy as np
import pytest

from bayescatrack.evaluation.edge_ranking import (
    missing_reference_edge_rows,
    rank_labeled_edges,
    summarize_edge_ranking_rows,
)


def test_rank_labeled_edges_reports_row_column_ranks_and_margins():
    labels = np.array([[0, 1, 0], [0, 0, 1]])
    costs = np.array([[0.0, 2.0, 1.0], [5.0, 4.0, 3.0]])

    rows = rank_labeled_edges(
        labels,
        {"cost": costs},
        reference_roi_indices=np.array([10, 11]),
        measurement_roi_indices=np.array([20, 21, 22]),
        metadata={"subject": "jm_test", "session_a": 0, "session_b": 1, "session_gap": 1},
    )

    first = next(row for row in rows if row["reference_roi_index"] == 10)
    assert first["measurement_roi_index"] == 21
    assert first["score_direction"] == "cost"
    assert first["row_rank"] == 3
    assert first["column_rank"] == 1
    assert first["best_false_row_roi_index"] == 20
    assert first["row_margin"] == pytest.approx(-2.0)

    second = next(row for row in rows if row["reference_roi_index"] == 11)
    assert second["row_rank"] == 1
    assert second["column_rank"] == 2
    assert second["row_margin"] == pytest.approx(1.0)
    assert second["column_margin"] == pytest.approx(-2.0)


def test_rank_labeled_edges_supports_similarity_scores():
    labels = np.array([[1, 0, 0]])
    iou = np.array([[0.8, 0.6, 0.1]])

    rows = rank_labeled_edges(
        labels,
        {"iou": iou},
        reference_roi_indices=np.array([7]),
        measurement_roi_indices=np.array([1, 2, 3]),
    )

    assert rows[0]["score_direction"] == "similarity"
    assert rows[0]["row_rank"] == 1
    assert rows[0]["row_margin"] == pytest.approx(0.2)


def test_missing_reference_edges_are_counted_in_summary_denominator():
    present_rows = rank_labeled_edges(
        np.array([[1]]),
        {"cost": np.array([[0.0]])},
        reference_roi_indices=np.array([1]),
        measurement_roi_indices=np.array([2]),
        metadata={"subject": "jm_test", "session_a": 0, "session_b": 1, "session_gap": 1},
    )
    missing_rows = missing_reference_edge_rows(
        [(1, 2), (3, 4)],
        reference_roi_indices=np.array([1]),
        measurement_roi_indices=np.array([2]),
        score_names=("cost",),
        metadata={"subject": "jm_test", "session_a": 0, "session_b": 1, "session_gap": 1},
    )

    summary = summarize_edge_ranking_rows(present_rows + missing_rows)

    assert len(summary) == 1
    assert summary[0]["gt_edges"] == 2
    assert summary[0]["present_edges"] == 1
    assert summary[0]["missing_edges"] == 1
    assert summary[0]["row_hit_at_1"] == pytest.approx(0.5)
    assert summary[0]["row_hit_at_1_present"] == pytest.approx(1.0)
